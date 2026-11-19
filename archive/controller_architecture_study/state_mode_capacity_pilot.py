"""24 paired flights spanning the previously excluded cheap shared subnets."""

import json
from pathlib import Path
from types import SimpleNamespace

from .decision_dataset import ROOT, atomic_json, sha256
from .fly_convex_policy import load_airsim_client
from .probe_airsim_three import load_layout
from .state_mode_calibration import RESULTS as CALIBRATION, moderate_case, tail_metrics
from .state_mode_flight import flight
from .state_mode_probe import OUTPUT, SupernetPool, make_trials, read_json


RESULTS = OUTPUT.with_name("state_mode_cheap_capacity_v1")
CONFIGURATIONS = ((1, 128), (3, 256), (11, 256), (33, 1024))


def build_plan(original, calibration):
    required = {"teacher_moderate_nominal", "teacher_moderate_perturbed", "supernet_max_moderate_perturbed"}
    passed = {r["id"] for r in calibration["trials"] if r["supplementary"]["sustained_joint_passed"]}
    if not required <= passed:
        raise ValueError("teacher/maximum-subnet calibration is incomplete or failed; do not expand the pilot")
    modes = []
    for depth, width in CONFIGURATIONS:
        item = next(r for r in original["all_offline_modes"] if r["depth"] == depth and r["width"] == width)
        modes.append({**item, "name": f"d{depth:02d}_w{width:04d}"})
    moderate, layout = moderate_case()
    frozen = next(r["case"] for r in read_json(CALIBRATION/"setup.json")["jobs"]
                  if r["id"] == "teacher_moderate_perturbed")
    if moderate != frozen:
        raise ValueError("moderate case changed since teacher calibration")
    cases = [next(c for c in original["cases"] if c["id"] == name)
             for name in ("straight_h1_v0.4", "straight_h0.3_v0.4")]+[moderate]
    straight, digest = load_layout(48, 0)
    if any(c["route_sha256"] != digest for c in cases[:2]):
        raise ValueError("straight route changed")
    plan = {"modes": modes, "cases": cases, "repeats": 2, "duration_s": 3.,
        "period_ms": original["period_ms"], "delivery_ms": original["delivery_ms"],
        "delivery_tolerance_ms": original["delivery_tolerance_ms"], "device": "cpu",
        "trials": make_trials(cases, modes, 2, 20261023),
        "selection_rule": "predeclared legal (1,128),(3,256),(11,256),(33,1024); no flight-outcome selection",
        "scope": "two matched repeats per state/mode; exploratory, not a reliability estimate or certified map",
        "tail_requirement": "original control/safety/progress plus last 0.5s observed p<=0.3m and v<=0.6m/s",
        "test_read": False, "training_performed": False}
    return plan, {"straight": straight, "moderate_turn": layout}


def summarize(plan, results):
    rows = []
    for case in plan["cases"]:
        for mode in plan["modes"]:
            found = [r for r in results if r["report"]["trial"]["case"] == case["id"]
                     and r["report"]["trial"]["mode"] == mode["name"]]
            rows.append({"case": case["id"], "mode": mode["name"], "effective_macs": mode["effective_macs"],
                "completed": len(found), "expected": plan["repeats"],
                "original_control_successes": sum(r["report"]["control_success"] for r in found),
                "sustained_control_successes": sum(r["supplementary"]["sustained_control_passed"] for r in found),
                "sustained_joint_successes": sum(r["supplementary"]["sustained_joint_passed"] for r in found),
                "timing_invalid": sum(not r["report"]["timing_comparable"] for r in found),
                "errors": [r["report"]["error"] for r in found if r["report"]["error"]]})
    return {"scope": plan["scope"], "completed": len(results), "expected": len(plan["trials"]),
        "rows": rows, "trials": results, "test_read": False}


def main():
    from airsim_interface import require_airsim

    original = read_json(OUTPUT/"setup.json")["plan"]
    for name, digest in original["dependency_sha256"].items():
        if sha256(ROOT/name) != digest:
            raise ValueError(f"frozen E1 source changed: {name}")
    for name, digest in original["artifacts"].items():
        if sha256(Path(original["export_dir"])/name) != digest:
            raise ValueError(f"frozen E1 model changed: {name}")
    if sha256(ROOT/"controller_architecture_study/state_mode_calibration.py") != read_json(CALIBRATION/"setup.json")["driver_sha256"]:
        raise ValueError("calibration source changed since the live trials")
    plan, layouts = build_plan(original, read_json(CALIBRATION/"summary.json"))
    metadata = {"driver_sha256": sha256(Path(__file__)), "E1_setup_sha256": sha256(OUTPUT/"setup.json"),
        "calibration_setup_sha256": sha256(CALIBRATION/"setup.json"),
        "calibration_summary_sha256": sha256(CALIBRATION/"summary.json"), "plan": plan}
    metadata = json.loads(json.dumps(metadata))
    RESULTS.mkdir(parents=True, exist_ok=True)
    if (RESULTS/"setup.json").exists() and read_json(RESULTS/"setup.json") != metadata:
        raise ValueError("capacity pilot plan changed; retain previous results")
    atomic_json(RESULTS/"setup.json", metadata)
    pool = SupernetPool(Path(original["export_dir"]), plan["modes"], "cpu")
    load_airsim_client()
    client = require_airsim().MultirotorClient(ip=original["host"], port=original["port"], timeout_value=3.)
    client.confirmConnection()
    args = SimpleNamespace(vehicle=original["vehicle"], duration=plan["duration_s"],
        period_ms=plan["period_ms"], delivery_ms=plan["delivery_ms"],
        delivery_tolerance_ms=plan["delivery_tolerance_ms"])
    cases = {c["id"]: c for c in plan["cases"]}
    results = []
    for index, trial in enumerate(plan["trials"], 1):
        case = cases[trial["case"]]
        directory = RESULTS/trial["id"]
        directory.mkdir(parents=True, exist_ok=True)
        if (directory/"report.json").exists():
            report = read_json(directory/"report.json")
        else:
            print(f"[{index}/{len(plan['trials'])}] {trial['id']}", flush=True)
            report = flight(client, pool, layouts[case["geometry"]], case, trial, args, directory)
        tail = tail_metrics(read_json(directory/"trace.json"), report)
        atomic_json(directory/"tail_metrics.json", tail)
        results.append({"report": report, "supplementary": tail})
        atomic_json(RESULTS/"summary.json", summarize(plan, results))
        print(f"  recovery={report['control_success']} tail={tail['tail_tracking_passed']} "
            f"timing={report['timing_comparable']} error={report['error']}", flush=True)
        if report["status"] != "finished":
            raise SystemExit(1)
    print(f"Capacity pilot report: {RESULTS/'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
