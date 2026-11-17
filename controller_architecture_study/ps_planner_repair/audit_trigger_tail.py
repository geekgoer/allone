"""Recompute the paired-trigger and live-tail evidence from preserved runs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def mean(values):
    return sum(values) / len(values)


def audit_run(folder):
    requests = lines(folder / "requests.jsonl")
    plans = [r for r in lines(folder / "plans.jsonl") if r.get("event") == "obligation_plan"]
    analysis = read(folder.parent / ("analysis_" + folder.name) / "analysis.json")
    assert analysis["audit_passed"] and analysis["calendar_timing_acceptances"] == 1
    assert analysis["complete_flights_observed"] == 1
    assert len(requests) == 1334 and all(r["status"] == "delivered" for r in requests)
    assert all(r["model_id"] == 11 for r in requests)
    assert max(r.get("decision", {}).get("checks", 0) for r in plans) <= 8
    paired_gaps = [(r["start_us"] - plans[i - 1]["end_us"]) / 1000
                   for i, r in enumerate(plans) if i and r.get("trigger") == "paired_observation"]
    active = [r for r in plans if r.get("management_timing", {}).get("planning_stages")]
    worst = max(active, key=lambda r: r["management_timing"]["planning_stages"]["diagnosis"]["cpu_ms"])
    diagnosis_over_1ms = [r for r in active if r["management_timing"]["planning_stages"]["diagnosis"]["cpu_ms"] > 1]
    faults = Counter()
    for r in active:
        for key in ("minor_faults", "major_faults"):
            faults[key] += r["management_timing"].get("planning_rusage", {}).get(key, 0)
    return {
        "run": folder.name,
        "requests_delivered": len(requests),
        "plans": len(plans),
        "triggers": dict(Counter(r.get("trigger") for r in plans)),
        "paired_gap_from_previous_plan_ms": paired_gaps,
        "installed_actions": dict(Counter(r.get("decision", {}).get("action") for r in plans
                                          if r.get("install", {}).get("installed"))),
        "diagnosis_over_1ms": len(diagnosis_over_1ms),
        "worst_diagnosis": {
            "cpu_ms": worst["management_timing"]["planning_stages"]["diagnosis"]["cpu_ms"],
            "wall_ms": worst["management_timing"]["planning_stages"]["diagnosis"]["wall_ms"],
            "planning_cpu_ms": worst["management_timing"]["planning"]["cpu_ms"],
            "checks": worst.get("decision", {}).get("checks"),
            "rusage": worst["management_timing"].get("planning_rusage"),
        },
        "planning_faults_total": dict(faults),
        "audit_passed": True,
        "requests_sha256": hashlib.sha256((folder / "requests.jsonl").read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="ps_management_trigger_20261117_004")
    parser.add_argument("--calibration", type=Path, required=True, help="ps_management_trigger_20261117_002")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Output must be new")
    manifest = read(args.results / "control_source" / "manifest.json")
    assert all(hashlib.sha256((args.results / "control_source" / name).read_bytes()).hexdigest() == digest
               for name, digest in manifest.items())
    runs = [audit_run(args.results / "live_raw" / name) for name in ("final_off_001", "final_on_001")]
    config_off = read(args.results / "control_source" / "config_pressure30_001_paired_off.json")
    config_on = read(args.results / "control_source" / "config_pressure30_001_repair.json")
    assert config_off.pop("paired_trigger") is False
    assert config_on.pop("paired_trigger") is True
    assert config_off == config_on
    tests = read(args.results / "live_raw" / "arm_tests.json")
    assert all(t["exit_code"] == 0 for t in tests)
    calibration = []
    for i, mode in enumerate(("plain", "profiled", "profiled", "plain")):
        report = read(args.calibration / "offline_profile" / f"instrument_{i}_{mode}.json")
        assert report["decision_parity"] and report["cases"] == 91
        calibration.append({"mode": mode, "mean_cpu_ms": mean([r["cpu_ms"] for r in report["records"]])})
    plain = mean([r["mean_cpu_ms"] for r in calibration if r["mode"] == "plain"])
    profiled = mean([r["mean_cpu_ms"] for r in calibration if r["mode"] == "profiled"])
    replay = read(args.results / "live_diagnosis_replay_001.json")
    assert replay["decision_parity"] and len(replay["records"]) == 40
    result = {
        "runs": runs,
        "source_manifest_files_verified": len(manifest),
        "paired_configs_identical_except_trigger": True,
        "arm_tests": [t["stdout"].strip() for t in tests],
        "instrumentation_calibration": {"records": 4 * 182, "plain_mean_cpu_ms": plain,
                                        "profiled_mean_cpu_ms": profiled,
                                        "difference_ms": profiled - plain},
        "outlier_replay": {"repetitions": 40,
                           "planner_cpu_mean_ms": mean([r["cpu_ms"] for r in replay["records"]]),
                           "planner_cpu_max_ms": max(r["cpu_ms"] for r in replay["records"]),
                           "diagnosis_cpu_mean_ms": mean([r["phases"]["diagnosis"]["cpu_ms"] for r in replay["records"]]),
                           "diagnosis_cpu_max_ms": max(r["phases"]["diagnosis"]["cpu_ms"] for r in replay["records"]),
                           "decision_parity": True},
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"runs": runs, "calibration": result["instrumentation_calibration"],
                      "outlier_replay": result["outlier_replay"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
