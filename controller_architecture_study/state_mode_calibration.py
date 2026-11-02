"""Small teacher/reference calibration; never relabel or overwrite E1 trials."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from . import fast_convex_preprocess as preprocess
from .decision_dataset import ROOT, atomic_json, sha256
from .fly_decision_dnn import StoredRoutePlanner
from .probe_airsim_three import load_layout
from .state_mode_baseline_check import TeacherProbe
from .state_mode_flight import flight
from .state_mode_probe import OUTPUT, PROTOCOL, SupernetPool, normal_frame, read_json


RESULTS = OUTPUT.with_name("state_mode_calibration_v1")
TAIL_SECONDS = .5


def tail_metrics(trace, report, duration=3.):
    """Stricter supplementary observation: both error bounds throughout the tail."""
    times = np.asarray([r["t"] for r in trace])
    first = max(0, int(np.searchsorted(times, duration-TAIL_SECONDS, side="right"))-1)
    tail = trace[first:]
    covered = bool(len(tail) >= 2 and tail[0]["t"] <= duration-TAIL_SECONDS
                   and tail[-1]["t"] >= duration and report["window_covered"])
    good = bool(covered and all(r["position_error_m"] <= PROTOCOL["position_tolerance_m"]
                and r["velocity_error_mps"] <= PROTOCOL["velocity_tolerance_mps"] for r in tail))
    return {"tail_seconds": TAIL_SECONDS, "tail_covered": covered,
        "tail_tracking_passed": good,
        "tail_max_position_error_m": max((r["position_error_m"] for r in tail), default=None),
        "tail_max_velocity_error_mps": max((r["velocity_error_mps"] for r in tail), default=None),
        "sustained_control_passed": bool(report["control_success"] and good),
        "sustained_joint_passed": bool(report["joint_success"] and good),
        "note": "Finite sampled observation, not continuous-time certification; original E1 result unchanged"}


def moderate_case():
    """Use an existing curved route at its peak planar curvature; no retiming."""
    layout, digest = load_layout(48, 1)
    planner = StoredRoutePlanner(layout)
    window = preprocess.FastReferenceWindow(planner.trajectory, planner.config)
    times = np.linspace(1.5, planner.trajectory.duration-3.3, 150)
    refs = [window.window(float(t), horizon=1)[0] for t in times]
    index = max(range(len(times)), key=lambda i: abs(float(np.cross(refs[i].velocity[:2], refs[i].acceleration[:2]))))
    ref = refs[index]
    normal, derivative = normal_frame(ref)
    radius = PROTOCOL["gate_width_m"]/2-PROTOCOL["vehicle_radius_m"]-PROTOCOL["clearance_m"]
    position = ref.position+normal*(radius-1.)
    velocity = ref.velocity+normal*(.4-np.dot(position-ref.position, derivative))
    return {"id": "moderate_turn_h1_v0.4", "geometry": "moderate_turn", "route_index": 1,
        "environment_id": 48, "route_sha256": digest, "reference_phase_s": float(times[index]),
        "position": position.tolist(), "velocity": velocity.tolist(), "quaternion": ref.orientation.tolist(),
        "requested_h_m": 1., "requested_h_dot_mps": -.4,
        "held_control": [float(np.linalg.norm(ref.acceleration-[0., 0., 9.80665])), 0., 0., 0.]}, layout


def nominal_case(case, layout):
    case = deepcopy(case)
    planner = StoredRoutePlanner(layout)
    ref = preprocess.FastReferenceWindow(planner.trajectory, planner.config).window(case["reference_phase_s"])[0]
    case.update(id=case["geometry"]+"_nominal", position=ref.position.tolist(),
        velocity=ref.velocity.tolist(), quaternion=ref.orientation.tolist(),
        requested_h_m=PROTOCOL["gate_width_m"]/2-PROTOCOL["vehicle_radius_m"]-PROTOCOL["clearance_m"],
        requested_h_dot_mps=0.)
    return case


def reference_audit(case, layout, teacher):
    planner = StoredRoutePlanner(layout)
    window = preprocess.FastReferenceWindow(planner.trajectory, planner.config)
    acceleration, velocity, body_rates, active = [], [], [], 0
    from airsim_interface import RawDroneState

    for t in np.arange(0., 3.01, .05):
        refs = window.window(case["reference_phase_s"]+float(t))
        ref = refs[0]
        state = RawDroneState(ref.position, ref.orientation, ref.velocity, np.zeros(3), np.zeros(3))
        _, info = teacher.query(state, refs)
        active += int(info["acceleration_bound_active"] or info["tilt_bound_active"])
        acceleration.append(ref.acceleration)
        velocity.append(ref.velocity)
        body_rates.append(ref.body_rates)
    a, v = np.asarray(acceleration), np.asarray(velocity)
    tilt = np.degrees(np.arctan2(np.linalg.norm(a[:, :2], axis=1), teacher.settings.gravity-a[:, 2]))
    return {"route_index": case["route_index"], "route_sha256": case["route_sha256"],
        "phase_s": case["reference_phase_s"], "time_scaled": False,
        "reference_speed_min_max_mps": [float(np.linalg.norm(v, axis=1).min()), float(np.linalg.norm(v, axis=1).max())],
        "max_abs_reference_acceleration_mps2": abs(a).max(axis=0).tolist(),
        "max_reference_tilt_deg": float(tilt.max()),
        "max_abs_reference_body_rates_rps": abs(np.asarray(body_rates)).max(axis=0).tolist(),
        "nominal_state_queries": len(a), "queries_with_active_prediction_constraints": active,
        "note": "Active constraints can occur anywhere in the QP prediction, not only its first action"}


def main():
    import torch
    from airsim_interface import require_airsim
    from .fly_convex_policy import load_airsim_client

    torch.set_num_threads(1)
    plan = read_json(OUTPUT/"setup.json")["plan"]
    for name, digest in plan["dependency_sha256"].items():
        if sha256(ROOT/name) != digest:
            raise ValueError(f"frozen E1 source changed: {name}")
    for name, digest in plan["artifacts"].items():
        if sha256(Path(plan["export_dir"])/name) != digest:
            raise ValueError(f"frozen E1 model changed: {name}")
    moderate, moderate_layout = moderate_case()
    hard = next(c for c in plan["cases"] if c["id"] == "turn_h0.6_v0.4")
    hard_layout, hard_hash = load_layout(48, 3)
    if hard_hash != hard["route_sha256"]:
        raise ValueError("hard reference route changed")
    jobs = [("teacher_hard_nominal", "QP_teacher", nominal_case(hard, hard_layout), hard_layout),
            ("teacher_moderate_nominal", "QP_teacher", nominal_case(moderate, moderate_layout), moderate_layout),
            ("teacher_moderate_perturbed", "QP_teacher", moderate, moderate_layout)]
    largest = max(plan["modes"], key=lambda r: r["effective_macs"])
    jobs.append(("supernet_max_moderate_perturbed", largest["name"], moderate, moderate_layout))
    teacher = TeacherProbe()
    pool = SupernetPool(Path(plan["export_dir"]), [largest], "cpu")
    metadata = {"driver_sha256": sha256(Path(__file__)), "E1_setup_sha256": sha256(OUTPUT/"setup.json"),
        "teacher_adapter_sha256": sha256(ROOT/"controller_architecture_study/state_mode_baseline_check.py"),
        "teacher": teacher.teacher.signature(), "duration_s": 3., "tail_seconds": TAIL_SECONDS,
        "protocol": plan["protocol"], "period_ms": plan["period_ms"], "delivery_ms": plan["delivery_ms"],
        "delivery_tolerance_ms": plan["delivery_tolerance_ms"],
        "jobs": [{"id": i, "mode": m, "case": c} for i,m,c,_ in jobs],
        "purpose": "four single-trial diagnostics, no reliability estimate or state map",
        "test_read": False, "training_performed": False}
    metadata = json.loads(json.dumps(metadata))
    RESULTS.mkdir(parents=True, exist_ok=True)
    if (RESULTS/"setup.json").exists() and read_json(RESULTS/"setup.json") != metadata:
        raise ValueError("calibration plan changed; preserve the earlier run")
    atomic_json(RESULTS/"setup.json", metadata)
    atomic_json(RESULTS/"reference_audit.json", {
        "hard": reference_audit(hard, hard_layout, teacher.teacher),
        "moderate": reference_audit(moderate, moderate_layout, teacher.teacher)})
    load_airsim_client()
    client = require_airsim().MultirotorClient(ip=plan["host"], port=plan["port"], timeout_value=3.)
    client.confirmConnection()
    args = SimpleNamespace(vehicle=plan["vehicle"], duration=3., period_ms=plan["period_ms"],
        delivery_ms=plan["delivery_ms"], delivery_tolerance_ms=plan["delivery_tolerance_ms"])
    results = []
    for name, mode, case, layout in jobs:
        directory = RESULTS/name
        directory.mkdir(parents=True, exist_ok=True)
        if (directory/"report.json").exists():
            report = read_json(directory/"report.json")
        else:
            print(f"Calibration: {name}", flush=True)
            trial = {"id": name, "case": case["id"], "mode": mode, "repeat": 0}
            if mode == "QP_teacher":
                with patch.object(preprocess, "encode", teacher.encode):
                    report = flight(client, teacher, layout, case, trial, args, directory)
            else:
                report = flight(client, pool, layout, case, trial, args, directory)
        tail = tail_metrics(read_json(directory/"trace.json"), report)
        atomic_json(directory/"tail_metrics.json", tail)
        results.append({"id": name, "report": report, "supplementary": tail})
        atomic_json(RESULTS/"summary.json", {"scope": metadata["purpose"], "trials": results})
        print(f"  original_recovery={report['control_success']} tail={tail['tail_tracking_passed']} "
              f"timing={report['timing_comparable']} sustained_joint={tail['sustained_joint_passed']}", flush=True)
        if report["status"] != "finished":
            raise SystemExit(1)
    print(f"Calibration report: {RESULTS/'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
