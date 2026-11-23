"""Screen a longer future control period using an unchanged command reserve."""

import argparse
import json
from pathlib import Path

from .critical_update_screen import load, rows


def first_idle_start(ready, cost, occupied):
    start = ready
    for begin, end in occupied:
        if start + cost <= begin:
            break
        if end > start:
            start = end
    return start


def replay(trial, vehicle, late_cycle, blocked_cycle):
    identity = load(trial / "execution_identity.json")
    config = identity["live_config"]
    origin_ms = identity["origin_us"] / 1000
    selected = {}
    occupied = []
    for row in rows(trial):
        if row["vehicle"] == vehicle and row["cycle"] in (late_cycle, blocked_cycle):
            selected[row["cycle"]] = row
        if row.get("inference_dispatched"):
            occupied.append((row["calendar_dispatch"]["board_start_ms"] - origin_ms,
                             row["calendar_finished"]["board_done_ms"] - origin_ms))
    occupied.sort()
    late, blocked = selected[late_cycle], selected[blocked_cycle]
    evidence = blocked["calendar_dispatch"]["dispatch_rejection_evidence"]
    if (late["status"] != "command_return_late" or not late.get("paired_observe_accepted") or
            not blocked.get("observe_accepted") or blocked["held_version"] != late["new_held_version"] or
            blocked["calendar_dispatch"]["dispatch_rejection"] != "no_model_fits_remaining_delivery_budget"):
        raise ValueError("Expected a verified late command and a later budget rejection for the same held version")
    original_period = blocked["period_ms"]
    reserve = evidence["command_reserve_ms"]
    if reserve <= original_period:
        raise ValueError("Observed command reserve does not exceed the current control period")
    service = evidence["required_service_ms"]
    post = max(reserve, config["post_ms"], (late["command_return_us"] - late["output_ready_us"]) / 1000)
    original_release = blocked["planned_release_us"] / 1000 - origin_ms
    hold = blocked["hold_deadline_us"] / 1000 - origin_ms
    finish = blocked["slot_return_us"] / 1000 - origin_ms
    commit = finish + config["decision_ms"]
    # Decide before the next old release. No published slot is edited.
    original_next = original_release + original_period
    candidates = []
    for period in sorted(p for p in config["periods_ms"] if p >= original_period):
        release = original_release + period
        start = first_idle_start(release + config["prep_ms"], service, occupied)
        completion = start + service + post
        due = min(release + period, release + config["fresh_ms"], hold)
        candidates.append(dict(period_ms=period, release_ms=release, dpu_start_ms=start,
                               predicted_return_ms=completion, effective_deadline_ms=due,
                               slack_ms=due-completion, fits=commit < original_next and
                               release > commit and completion < due))
    return dict(vehicle=vehicle, original_period_ms=original_period, late_cycle=late_cycle,
                after_blocked_cycle=blocked_cycle, unchanged_reserve_ms=reserve, charged_post_ms=post,
                dpu_service_ms=service, old_next_release_ms=original_next, projected_commit_ms=commit,
                observed_hold_deadline_ms=hold, candidates=candidates,
                scope="Fixed-timestamp screen with peers' observed DPU intervals unchanged; no simulated command delivery.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--vehicle", required=True)
    parser.add_argument("--late-cycle", type=int, required=True)
    parser.add_argument("--blocked-cycle", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = replay(args.trial, args.vehicle, args.late_cycle, args.blocked_cycle)
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
