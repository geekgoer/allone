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


def fit_with_peer_slack(ready, cost, latest_finish, peers, edf_deadline=None):
    """Insert one nonpreemptive job, preserving peer order and their logged guards."""
    previous_end = ready
    for index, peer in enumerate(peers):
        begin, end, _, _, _, _ = peer
        if end <= ready:
            continue
        if begin < ready:
            previous_end = end
            continue
        start = max(ready, previous_end)
        if start + cost >= latest_finish:
            return None
        if edf_deadline is not None and any(p[4] <= start and p[5] <= edf_deadline for p in peers[index:]):
            previous_end = end
            continue
        finish = start + cost
        delayed = []
        for p_start, p_end, slack, key, _, _ in peers[index:]:
            slip = finish - p_start
            if slip <= 0:
                return dict(start_ms=start, delayed_peers=delayed)
            if slip >= slack:
                break
            delayed.append(dict(job=key, delay_ms=slip, remaining_slack_ms=slack-slip))
            finish = p_end + slip
        else:
            return dict(start_ms=start, delayed_peers=delayed)
        if edf_deadline is not None:
            # With plain EDF, the ready job starts at this first eligible point.
            return None
        previous_end = end
    start = max(ready, previous_end)
    if start + cost < latest_finish:
        return dict(start_ms=start, delayed_peers=[])
    return None


def peer_displacement_slack(row):
    # Preserve the command guard, the observed ACK deadline, and this worker's
    # ability to return before its next release. Never move a failed peer.
    admission = row.get("command_admission") or {}
    if row["status"] != "delivered" or not admission.get("accepted"):
        return 0.
    due = admission["deadline_us"]
    return min(due - admission["checked_us"] - admission["reserve_us"],
               due - row["command_return_us"],
               row["planned_release_us"] + row["period_ms"] * 1000 - row["slot_return_us"]) / 1000


def replay(trial, vehicle, late_cycle, blocked_cycle, allow_peer_slack=False):
    identity = load(trial / "execution_identity.json")
    config = identity["live_config"]
    origin_ms = identity["origin_us"] / 1000
    selected = {}
    occupied = []
    for row in rows(trial):
        if row["vehicle"] == vehicle and row["cycle"] in (late_cycle, blocked_cycle):
            selected[row["cycle"]] = row
        if row.get("inference_dispatched") and not (row["vehicle"] == vehicle and row["cycle"] > blocked_cycle):
            occupied.append((row["calendar_dispatch"]["board_start_ms"] - origin_ms,
                             row["calendar_finished"]["board_done_ms"] - origin_ms,
                             peer_displacement_slack(row), f'{row["vehicle"]}:{row["cycle"]}',
                             row["dpu_enqueue_us"] / 1000 - origin_ms,
                             row["actuation_deadline_us"] / 1000 - origin_ms))
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
    # Reject slack replay on a window containing non-dispatched drops by peers.
    # Moving the device may then change their workers' finish time, for which
    # this insertion model has no bound.
    if allow_peer_slack:
        unmodelled = [f'{row["vehicle"]}:{row["cycle"]}' for row in rows(trial)
                      if row["vehicle"] != vehicle and row["status"] != "delivered" and
                      finish <= row.get("slot_return_us", row["actual_publish_us"]) / 1000 - origin_ms and
                      row["planned_release_us"] / 1000 - origin_ms < hold]
    intervals = [(item[0], item[1]) for item in occupied]
    for period in sorted(p for p in config["periods_ms"] if p >= original_period):
        release = original_release + period
        ready = release + config["prep_ms"]
        start = first_idle_start(ready, service, intervals)
        due = min(release + period, release + config["fresh_ms"], hold)
        completion = start + service + post
        timely_commit = commit < original_next and release > commit
        candidates.append(dict(period_ms=period, release_ms=release, dpu_start_ms=start,
                               predicted_return_ms=completion, effective_deadline_ms=due,
                               slack_ms=due-completion, fits=timely_commit and completion < due))
        if allow_peer_slack:
            window = [peer for peer in occupied if peer[1] > ready]
            insertion = fit_with_peer_slack(ready, service, due-post, window) if not unmodelled and timely_commit else None
            edf = fit_with_peer_slack(ready, service, due-post, window, edf_deadline=due) if insertion else None
            record = dict(optimistic_slack_without_queue_ms=due-ready-service-post, insertion=insertion)
            if insertion:
                record["predicted_return_ms"] = insertion["start_ms"] + service + post
                record["slack_ms"] = due - record["predicted_return_ms"]
            record["edf_insertion"] = edf
            if edf:
                edf["slack_ms"] = due - (edf["start_ms"] + service + post)
            candidates[-1]["peer_slack_screen"] = record
    return dict(vehicle=vehicle, original_period_ms=original_period, late_cycle=late_cycle,
                after_blocked_cycle=blocked_cycle, unchanged_reserve_ms=reserve, charged_post_ms=post,
                dpu_service_ms=service, old_next_release_ms=original_next, projected_commit_ms=commit,
                observed_hold_deadline_ms=hold, candidates=candidates,
                peer_slack_unmodelled=unmodelled if allow_peer_slack else None,
                scope="Fixed-timestamp screen with peers' observed DPU intervals unchanged; no simulated command delivery.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--vehicle", required=True)
    parser.add_argument("--late-cycle", type=int, required=True)
    parser.add_argument("--blocked-cycle", type=int, required=True)
    parser.add_argument("--allow-peer-slack", action="store_true",
                        help="also screen one insertion that leaves peer order unchanged and respects logged guards")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = replay(args.trial, args.vehicle, args.late_cycle, args.blocked_cycle, args.allow_peer_slack)
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
