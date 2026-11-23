"""Check one dropped dispatch with its observed order and a shorter lock wait."""

import argparse
import json
import math
from pathlib import Path

from .critical_update_screen import load, rows


def project(start, other_cost, reserve, deadline, services, wait):
    return sum(start + wait + other_cost + service + reserve < deadline for service in services)


def replay(trial, vehicle, cycle):
    origin = load(trial / "execution_identity.json")["origin_us"]
    requests = [r for r in rows(trial) if r["vehicle"] == vehicle and r["cycle"] in (cycle - 1, cycle)]
    by_cycle = {r["cycle"]: r for r in requests}
    target, previous = by_cycle[cycle], by_cycle[cycle - 1]
    if target["status"] != "dropped_inference" or target.get("inference_dispatched"):
        raise ValueError("Target must have been dropped before execution")
    begin, end = (target[k] for k in ("calendar_take_start_us", "calendar_take_return_us"))
    wait = target["calendar_take_lock_wait_us"]
    overhead = end - begin - wait
    if overhead < 0:
        raise ValueError("Inconsistent dispatch clock values")
    latest = target["actuation_deadline_us"]
    reserve = previous["calendar_dispatch"]["dispatch_rejection_evidence"]["command_reserve_ms"] * 1000
    if previous.get("command_sent") or not end >= latest > begin:
        raise ValueError("Expected a missed previous update and a deadline that elapsed in take()")
    lookback_us = 2_000_000
    durations, ran_first = [], []
    for r in rows(trial):
        if not r.get("inference_dispatched"):
            continue
        start_us = r["calendar_dispatch"]["board_start_ms"] * 1000
        done_us = r["calendar_finished"]["board_done_ms"] * 1000
        if r["model_id"] == target["model_id"] and begin - lookback_us <= done_us <= begin:
            durations.append(done_us - start_us)
        if target["dpu_enqueue_us"] <= start_us < begin:
            ran_first.append(dict(vehicle=r["vehicle"], cycle=r["cycle"], start_ms=(start_us-origin)/1000))
    if not durations:
        raise ValueError("No completed observations of this model before the target")
    durations.sort()
    p99 = durations[math.ceil(0.99 * len(durations)) - 1]
    snapshots = []
    with (trial / "plans.jsonl").open(encoding="utf-8") as f:
        for r in map(json.loads, f):
            if begin <= r["start_us"] < end and (timing := r.get("management_timing", {}).get("snapshot")):
                jobs = (r.get("snapshot") or {}).get("calendar", {}).get("jobs", [])
                snapshots.append(dict(start_ms=(r["start_us"]-origin)/1000,
                                      build_wall_ms=timing["build"]["wall_ms"], job_count=len(jobs),
                                      target_present=any(j["name"] == f"{vehicle}:{cycle}" for j in jobs),
                                      install_status=r.get("install", {}).get("status")))
    return dict(vehicle=vehicle, cycle=cycle, model=target["model_id"],
                dpu_order_unchanged=True, peer_dispatches_before_target=sorted(ran_first, key=lambda r: r["start_ms"]),
                observed_lock_wait_wall_ms=wait/1000, take_nonwait_ms=overhead/1000,
                budget_at_take_entry_ms=(latest-begin)/1000, reserve_from_previous_ms=reserve/1000,
                same_model_prior_count=len(durations), prior_p99_service_ms=p99/1000,
                prior_max_service_ms=durations[-1]/1000,
                p99_wait_headroom_ms=(latest-begin-overhead-reserve-p99)/1000,
                prior_service_samples_fitting={str(w/1000): project(begin, overhead, reserve, latest, durations, w)
                                               for w in (wait, 0, 1000, 2000)},
                snapshot_starts_during_take=snapshots,
                scope="Local counterfactual only. Prior durations are empirical; downstream host timings may change.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--vehicle", required=True)
    parser.add_argument("--cycle", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = replay(args.trial, args.vehicle, args.cycle)
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
