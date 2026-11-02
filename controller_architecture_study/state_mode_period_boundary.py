"""Predeclared period/control interaction pilot: 3 states x 3 periods x 3 controllers x 2 repeats."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from . import fast_convex_preprocess as preprocess
from .decision_dataset import ROOT, atomic_json, sha256
from .fly_convex_policy import load_airsim_client
from .state_mode_boundary import RESULTS as BOUNDARY, BoundaryPool, CONTROLLERS, build_cases, summary
from .state_mode_calibration import tail_metrics
from .state_mode_flight import flight
from .state_mode_probe import OUTPUT, read_json, make_trials


RESULTS = OUTPUT.with_name("state_mode_period_boundary_v1")
PERIODS_MS = (100., 150., 200.)
STATES = ("baseline", "outward_v2", "roll_minus30")


def build_plan():
    original, layout = build_cases()
    cases = []
    for source in original:
        if source["id"] not in STATES:
            continue
        for period in PERIODS_MS:
            case = deepcopy(source)
            case.update(id=f"{source['id']}_T{period:g}", source_case=source["id"], period_ms=period)
            cases.append(case)
    trials = make_trials(cases, [{"name": m} for m in CONTROLLERS], 2, 20261025)
    return {"cases": cases, "trials": trials, "repeats": 2, "duration_s": 3.,
        "periods_ms": list(PERIODS_MS), "controllers": list(CONTROLLERS),
        "scope": "54 fixed period/state/controller trials; not a certified delay envelope or runtime T/c optimizer",
        "selection": "baseline, outward_v2, roll_minus30 fixed before period trials; all 3 periods and controllers retained",
        "test_read": False, "training_performed": False}, layout


def main():
    from airsim_interface import require_airsim

    original = read_json(OUTPUT/"setup.json")["plan"]
    for name, digest in original["dependency_sha256"].items():
        if sha256(ROOT/name) != digest:
            raise ValueError(f"frozen E1 source changed: {name}")
    for name, digest in original["artifacts"].items():
        if sha256(Path(original["export_dir"])/name) != digest:
            raise ValueError(f"frozen network changed: {name}")
    if sha256(ROOT/"controller_architecture_study/state_mode_boundary.py") != read_json(BOUNDARY/"setup.json")["driver_sha256"]:
        raise ValueError("boundary implementation changed")
    for phase in ("discovery", "confirmation"):
        count = read_json(BOUNDARY/phase/"summary.json")["completed"]
        if count != len(read_json(BOUNDARY/phase/"setup.json")["trials"]):
            raise ValueError("finish boundary discovery and confirmation before period extension")
    plan, layout = build_plan()
    pool = BoundaryPool(original)
    metadata = {"driver_sha256": sha256(Path(__file__)), "E1_setup_sha256": sha256(OUTPUT/"setup.json"),
        "boundary_setup_sha256": sha256(BOUNDARY/"setup.json"),
        "boundary_confirmation_sha256": sha256(BOUNDARY/"confirmation/summary.json"),
        "plan": plan, "runtime": pool.network.runtime, "teacher": pool.teacher.teacher.signature(),
        "delivery_ms": original["delivery_ms"], "delivery_tolerance_ms": original["delivery_tolerance_ms"],
        "protocol": original["protocol"],
        "timing": "first send target stays20ms; only subsequent release/hold period varies; response deadline is each job's next release"}
    metadata = json.loads(json.dumps(metadata))
    RESULTS.mkdir(parents=True, exist_ok=True)
    if (RESULTS/"setup.json").exists() and read_json(RESULTS/"setup.json") != metadata:
        raise ValueError("period plan changed; retain earlier results")
    atomic_json(RESULTS/"setup.json", metadata)
    done = [read_json(RESULTS/t["id"]/'result.json') for t in plan["trials"] if (RESULTS/t["id"]/'result.json').exists()]
    if any(r["report"]["status"] != "finished" for r in done):
        raise ValueError("interrupted prior trial retained; inspect before retrying")
    atomic_json(RESULTS/"summary.json", {**summary(done), "expected": len(plan["trials"]), "periods_ms": list(PERIODS_MS)})
    pending = [t for t in plan["trials"] if t["id"] not in {r["report"]["trial"]["id"] for r in done}]
    if not pending:
        print(f"All {len(done)} period trials already saved: {RESULTS/'summary.json'}")
        return
    load_airsim_client()
    client = require_airsim().MultirotorClient(ip=original["host"], port=original["port"], timeout_value=3.)
    client.confirmConnection()
    if client.simIsPause() or original["vehicle"] not in client.listVehicles():
        raise RuntimeError("manual Play must be active and unpaused")
    foreign = [n for n in client.simListSceneObjects("Falcon.*Gate.*") if not n.startswith("FalconModeGate_")]
    if foreign:
        raise RuntimeError("unrelated generated gates present; do not mix experiments")
    cases = {c["id"]: c for c in plan["cases"]}
    for trial in pending:
        case = cases[trial["case"]]
        directory = RESULTS/trial["id"]
        directory.mkdir(parents=True, exist_ok=True)
        print(f"period [{len(done)+1}/{len(plan['trials'])}] {trial['id']}", flush=True)
        args = SimpleNamespace(vehicle=original["vehicle"], duration=3., period_ms=case["period_ms"],
            delivery_ms=original["delivery_ms"], delivery_tolerance_ms=original["delivery_tolerance_ms"])
        if (directory/"report.json").exists():
            report = read_json(directory/"report.json")
        else:
            with patch.object(preprocess, "encode", pool.teacher.encode):
                report = flight(client, pool, layout, case, trial, args, directory)
        row = {"report": report, "supplementary": tail_metrics(read_json(directory/"trace.json"), report)}
        atomic_json(directory/"result.json", row)
        done.append(row)
        atomic_json(RESULTS/"summary.json", {**summary(done), "expected": len(plan["trials"]), "periods_ms": list(PERIODS_MS)})
        print(f"  sustained={row['supplementary']['sustained_control_passed']} "
            f"timing={report['timing_comparable']} error={report['error']}", flush=True)
        if report["status"] != "finished":
            raise SystemExit(1)
    print(f"Period report: {RESULTS/'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
