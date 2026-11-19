"""Offline audit of saved scripted-switch logs; no inference, RPC or training.

Legacy runs accumulated absolute host timestamps. Validate that exact schedule,
including its endpoint rounding, rather than silently using the repaired one.
Original reports and logs are never overwritten.
"""

import argparse
import json
from pathlib import Path

from .decision_dataset import atomic_json, sha256
from .dynamic_probe_common import read_json, summarize_timing
from .joint_switch_flight import SwitchPlan
from .replay_online_joint import output_age


VEHICLES = ("Drone1", "Drone2", "Drone3")


def legacy_slots(plan, start, duration, phase):
    scheduled, cycle = start+phase, 0
    while scheduled < start+duration-1e-9:
        choice = plan.at(scheduled-start)
        yield cycle, scheduled, choice
        scheduled += choice["period_ms"]/1000
        cycle += 1


def recompute(source, rows):
    if source.get("experiment") != "scripted_joint_switch_mechanism":
        raise ValueError("requires a scripted-switch flight report")
    if source["phase"] not in ("sync", "staggered"):
        raise ValueError("unknown flight phase")
    first = [r for r in rows if r["vehicle"] == VEHICLES[0] and r["cycle"] == 0]
    if len(first) != 1:
        raise ValueError("cannot reconstruct the flight clock: missing/duplicate first request")
    start, duration = first[0]["scheduled_s"], source["duration_seconds"]
    plan = SwitchPlan(source["switch_plan"], source["switch_plan"])
    schedules = {}
    current = {}
    for i, vehicle in enumerate(VEHICLES):
        phase = i*plan.steps[0]["period_ms"]/3000 if source["phase"] == "staggered" else 0.
        schedules[vehicle] = list(legacy_slots(plan, start, duration, phase))
        current[vehicle] = list(plan.slots(start, duration, phase))
    checks = {}
    for vehicle, slots in schedules.items():
        actual = sorted((r for r in rows if r["vehicle"] == vehicle), key=lambda r: r["cycle"])
        checks[vehicle] = len(actual) == len(slots) and all(
            r["cycle"] == cycle and abs(r["scheduled_s"]-scheduled) <= 1e-8
            and r["model"] == choice["model"]
            and r["period_ms"] == r["deadline_ms"] == choice["period_ms"]
            for r, (cycle, scheduled, choice) in zip(actual, slots))
    planned = sum(map(len, schedules.values()))
    schedule_ok = all(checks.values()) and len(rows) == planned
    timing = summarize_timing(rows, None)
    dispatches = [{**r, "ready_s": r["dnn_dispatch_s"]}
                  for r in rows if "dnn_dispatch_s" in r]
    realtime_ok = bool(schedule_ok and source["status"] == "COMPLETE" and not source["errors"]
        and timing["output_deadline_misses"] == timing["dnn_dispatch_deadline_misses"] == 0
        and timing["inference_errors"] == timing["control_deadline_aborts"] == 0
        and timing["dnn_commands_dispatched"] == len(rows)
        and all(r.get("command_duration_ms") == r["period_ms"] for r in rows))
    acceptance = bool(realtime_ok and source["flight_passed"]
                      and source["clock_ratio_passed"] and source["tracking_window_covered"])
    return {
        "status": "PASS" if acceptance else "FAIL",
        "scope": "historical legacy-schedule audit, not a flight test of the repaired scheduler",
        "original_flags": {k: source[k] for k in ("flight_passed", "realtime_passed", "acceptance_passed")},
        "original_planned_releases": source["planned_releases"],
        "legacy_host_schedule_releases": planned,
        "repaired_relative_schedule_releases": sum(map(len, current.values())),
        "per_vehicle_schedule_matched": checks,
        "corrected_realtime_passed": realtime_ok,
        "corrected_acceptance_passed": acceptance,
        "command_input_age": output_age(dispatches, start, [3], duration, duration, vehicle_names=VEHICLES),
        "timing": timing,
        "tracking_rmse_m": source["tracking_rmse_m"],
        "model_switches": sum(bool(r.get("model_changed")) for r in rows),
        "period_switches": sum(bool(r.get("period_changed")) for r in rows),
        "limitations": source["limitations"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report, output = args.report.resolve(), args.output.resolve()
    requests = report.parent / "requests.jsonl"
    if output in (report, requests, report.parent / "flight_trace.json") or output.exists():
        parser.error("output must be a new file, not an original artifact")
    hashes = {p.name: sha256(p) for p in (report, requests)}
    source = read_json(report)
    with requests.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    result = recompute(source, rows)
    if hashes != {p.name: sha256(p) for p in (report, requests)}:
        raise ValueError("source artifacts changed while auditing")
    result.update(source_report=str(report), source_sha256=hashes,
                  audit_source_sha256=sha256(Path(__file__)), original_artifacts_modified=False)
    atomic_json(output, result)
    print(json.dumps({k: result[k] for k in ("status", "original_planned_releases",
        "legacy_host_schedule_releases", "repaired_relative_schedule_releases",
        "corrected_realtime_passed", "corrected_acceptance_passed", "command_input_age")}, indent=2))
    print(f"Audit: {output}")
    if not result["corrected_acceptance_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
