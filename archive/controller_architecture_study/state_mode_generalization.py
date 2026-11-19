"""Frozen validation-context extension; no route selection using neural outcomes."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from state_adapter import normalize_quaternion, quaternion_multiply
from . import fast_convex_preprocess as preprocess
from .decision_dataset import ROOT, atomic_json, sha256
from .dynamic_probe_common import BASE, DATA
from .fly_convex_policy import load_airsim_client
from .fly_decision_dnn import StoredRoutePlanner
from .probe_airsim_three import load_layout
from .state_mode_boundary import BoundaryPool, CONTROLLERS, summary
from .state_mode_calibration import nominal_case, reference_audit, tail_metrics
from .state_mode_flight import flight
from .state_mode_probe import OUTPUT, PROTOCOL, make_trials, normal_frame, read_json


RESULTS = OUTPUT.with_name("state_mode_generalization_v1")
# Fractions of [1.5 s, saved route duration - 3.3 s], NOT curvature-selected.
# Two phases on an unseen validation route and a second 3D route/environment.
CONTEXTS = ((49, 1, .30), (49, 1, .55), (50, 2, .40))
PERIODS_MS = (50., 100., 125.)
PROFILES = (("baseline", 0.), ("roll_minus30", -30.))


def build_plan():
    splits = read_json(DATA/"split_manifest.json")["environment_ids"]
    cases, layouts, contexts = [], {}, []
    radius = PROTOCOL["gate_width_m"]/2-PROTOCOL["vehicle_radius_m"]-PROTOCOL["clearance_m"]
    for index, (environment, route, fraction) in enumerate(CONTEXTS):
        if environment not in splits["validation"] or environment in splits["test"]:
            raise ValueError("generalization contexts must use validation environments only")
        context = f"ctx{index}_e{environment}_r{route}"
        layout, digest = load_layout(environment, route)
        layouts[context] = layout
        planner = StoredRoutePlanner(layout)
        phase = 1.5+fraction*(planner.trajectory.duration-4.8)
        if not 1.5 <= phase <= planner.trajectory.duration-3.3:
            raise ValueError("reference window too short")
        ref = preprocess.FastReferenceWindow(planner.trajectory, planner.config).window(phase)[0]
        normal, derivative = normal_frame(ref)
        position = ref.position+normal*(radius-1.)
        velocity = ref.velocity+normal*(.4-np.dot(position-ref.position, derivative))
        base = {"id": context, "context": context, "geometry": context, "route_index": route,
            "environment_id": environment, "route_sha256": digest, "reference_phase_s": float(phase),
            "position": position.tolist(), "velocity": velocity.tolist(),
            "quaternion": ref.orientation.tolist(), "requested_h_m": 1., "requested_h_dot_mps": -.4,
            "held_control": [float(np.linalg.norm(ref.acceleration-[0., 0., 9.80665])), 0., 0., 0.]}
        contexts.append({"id": context, "environment": environment, "route": route,
            "phase_fraction": fraction, "phase_s": float(phase), "route_sha256": digest})
        for profile, roll in PROFILES:
            angle = np.radians(roll)
            delta = np.array([np.cos(angle/2), np.sin(angle/2), 0., 0.])
            for period in PERIODS_MS:
                case = deepcopy(base)
                case.update(id=f"{context}_{profile}_T{period:g}", source_case=f"{context}_{profile}",
                    profile=profile, period_ms=period, kind="recovery",
                    quaternion=normalize_quaternion(quaternion_multiply(ref.orientation, delta)).tolist(),
                    body_rotation_error_deg=[roll, 0., 0.])
                cases.append(case)
        nominal = nominal_case(base, layout)
        nominal.update(id=f"{context}_nominal_T50", source_case=f"{context}_nominal", profile="nominal",
            period_ms=50., kind="nominal", body_rotation_error_deg=[0., 0., 0.])
        cases.append(nominal)
    recovery = [c for c in cases if c["kind"] == "recovery"]
    trials = make_trials(recovery, [{"name": m} for m in CONTROLLERS], 2, 20261027)
    nominal_trials = [{"id": f"nominal_{c['context']}", "case": c["id"], "mode": "QP_teacher", "repeat": 0}
                      for c in cases if c["kind"] == "nominal"]
    return {"cases": cases, "contexts": contexts, "trials": nominal_trials+trials,
        "controllers": list(CONTROLLERS), "periods_ms": list(PERIODS_MS), "duration_s": 3.,
        "repeats": 2, "selection": "fixed route IDs and phase fractions; no filtering, retiming, neural-outcome selection or threshold tuning",
        "scope": "108 recovery trials plus3 teacher nominal controls; validation-context pilot, NOT held-out test or safety certification",
        "test_read": False, "training_performed": False}, layouts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="execute pending trials in the existing manual AirSim Play")
    args = parser.parse_args()
    original = read_json(OUTPUT/"setup.json")["plan"]
    for name, digest in original["dependency_sha256"].items():
        if sha256(ROOT/name) != digest:
            raise ValueError(f"frozen E1 source changed: {name}")
    for name, digest in original["artifacts"].items():
        if sha256(Path(original["export_dir"])/name) != digest:
            raise ValueError(f"frozen model artifact changed: {name}")
    plan, layouts = build_plan()
    pool = BoundaryPool(original)
    dependencies = ("state_mode_boundary.py", "state_mode_calibration.py", "state_mode_baseline_check.py")
    metadata = {"driver_sha256": sha256(Path(__file__)), "E1_setup_sha256": sha256(OUTPUT/"setup.json"),
        "dependency_sha256": {n: sha256(Path(__file__).parent/n) for n in dependencies},
        "route_manifest_sha256": sha256(BASE/"manifest.json"),
        "split_manifest_sha256": sha256(DATA/"split_manifest.json"),
        "plan": plan, "runtime": pool.network.runtime, "teacher": pool.teacher.teacher.signature(),
        "delivery_ms": original["delivery_ms"], "delivery_tolerance_ms": original["delivery_tolerance_ms"],
        "protocol": original["protocol"],
        "timing": "same20ms first delivery target; subsequent actuation gap approximately T, not T+20; host/RPC proxies only"}
    metadata = json.loads(json.dumps(metadata))
    RESULTS.mkdir(parents=True, exist_ok=True)
    if (RESULTS/"setup.json").exists() and read_json(RESULTS/"setup.json") != metadata:
        raise ValueError("frozen generalization setup changed; preserve prior results")
    atomic_json(RESULTS/"setup.json", metadata)
    cases = {c["id"]: c for c in plan["cases"]}
    if not (RESULTS/"reference_audit.json").exists():
        audit = {c["context"]: reference_audit(c, layouts[c["context"]], pool.teacher.teacher)
                 for c in plan["cases"] if c["kind"] == "nominal"}
        atomic_json(RESULTS/"reference_audit.json", audit)
    done = [read_json(RESULTS/t["id"]/'result.json') for t in plan["trials"]
            if (RESULTS/t["id"]/'result.json').exists()]
    if any(r["report"]["status"] != "finished" for r in done):
        raise ValueError("interrupted prior trial retained; inspect before retrying")
    def save_summary():
        atomic_json(RESULTS/"summary.json", {**summary(done), "expected": len(plan["trials"])})
    save_summary()
    completed = {r["report"]["trial"]["id"] for r in done}
    pending = [t for t in plan["trials"] if t["id"] not in completed]
    print(f"Frozen plan: {len(plan['contexts'])} contexts, {len(plan['trials'])} trials; {len(pending)} pending", flush=True)
    if not args.run or not pending:
        print(f"Plan/report: {RESULTS/'setup.json'}", flush=True)
        return
    from airsim_interface import require_airsim
    load_airsim_client()
    client = require_airsim().MultirotorClient(ip=original["host"], port=original["port"], timeout_value=3.)
    client.confirmConnection()
    if client.simIsPause() or original["vehicle"] not in client.listVehicles():
        raise RuntimeError("manual Play must be active and unpaused")
    foreign = [n for n in client.simListSceneObjects("Falcon.*Gate.*") if not n.startswith("FalconModeGate_")]
    if foreign:
        raise RuntimeError("unrelated generated gates present; do not mix experiments")
    for trial in pending:
        case = cases[trial["case"]]
        directory = RESULTS/trial["id"]
        directory.mkdir(parents=True, exist_ok=True)
        print(f"generalization [{len(done)+1}/{len(plan['trials'])}] {trial['id']}", flush=True)
        flight_args = SimpleNamespace(vehicle=original["vehicle"], duration=3., period_ms=case["period_ms"],
            delivery_ms=original["delivery_ms"], delivery_tolerance_ms=original["delivery_tolerance_ms"])
        if (directory/"report.json").exists():
            report = read_json(directory/"report.json")
        else:
            with patch.object(preprocess, "encode", pool.teacher.encode):
                report = flight(client, pool, layouts[case["context"]], case, trial, flight_args, directory)
        row = {"report": report, "supplementary": tail_metrics(read_json(directory/"trace.json"), report)}
        atomic_json(directory/"result.json", row)
        done.append(row)
        save_summary()
        print(f"  sustained={row['supplementary']['sustained_control_passed']} "
              f"timing={report['timing_comparable']} error={report['error']}", flush=True)
        if report["status"] != "finished":
            raise SystemExit(1)
    print(f"Generalization report: {RESULTS/'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
