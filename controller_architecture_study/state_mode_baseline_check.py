"""Three diagnostic controls after E1: matched QP teacher and nominal supernet."""

from copy import deepcopy
import json
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import patch

from . import fast_convex_preprocess as preprocess
from .convex_teacher import ConvexTeacher
from .decision_dataset import ROOT, atomic_json, sha256
from .fly_convex_policy import load_airsim_client
from .fly_decision_dnn import StoredRoutePlanner
from .probe_airsim_three import load_layout
from .state_mode_probe import OUTPUT, SupernetPool, read_json
from .state_mode_flight import flight


class TeacherProbe:
    """Observe unchanged encoder arguments to query the actual live QP teacher."""
    def __init__(self):
        self.teacher = ConvexTeacher()
        self.original_encode = preprocess.encode
        self.problem = None

    def encode(self, state, references):
        feature = self.original_encode(state, references)
        self.problem = state, references
        return feature

    def predict(self, name, feature):
        if self.problem is None:
            raise RuntimeError("teacher needs the exact state/reference from this request")
        started = time.perf_counter()
        action, _ = self.teacher.query(*self.problem)
        return action.astype(float), {"inference_ms": (time.perf_counter()-started)*1000}


def main():
    import torch
    from airsim_interface import require_airsim

    torch.set_num_threads(1)
    original = read_json(OUTPUT/"setup.json")
    plan = original["plan"]
    for name, digest in plan["dependency_sha256"].items():
        if sha256(ROOT/name) != digest:
            raise ValueError(f"E1 source changed: {name}")
    for name, digest in plan["artifacts"].items():
        if sha256(Path(plan["export_dir"])/name) != digest:
            raise ValueError(f"E1 model artifact changed: {name}")
    teacher = TeacherProbe()
    largest = max(plan["modes"], key=lambda r: r["effective_macs"])
    pool = SupernetPool(Path(plan["export_dir"]), [largest], "cpu")
    cases = {r["id"]: r for r in plan["cases"]}
    layouts = {}
    for geometry, route in (("straight", 0), ("turn", 3)):
        layouts[geometry], digest = load_layout(48, route)
        expected = next(c["route_sha256"] for c in cases.values() if c["geometry"] == geometry)
        if digest != expected:
            raise ValueError("baseline route differs from frozen E1")
    nominal = deepcopy(cases["turn_h0.6_v0.4"])
    planner = StoredRoutePlanner(layouts["turn"])
    ref = preprocess.FastReferenceWindow(planner.trajectory, planner.config).window(nominal["reference_phase_s"])[0]
    nominal.update(id="turn_nominal", position=ref.position.tolist(), velocity=ref.velocity.tolist(),
        quaternion=ref.orientation.tolist(), requested_h_m=plan["protocol"]["gate_width_m"]/2
            -plan["protocol"]["vehicle_radius_m"]-plan["protocol"]["clearance_m"], requested_h_dot_mps=0.)
    jobs = [("teacher_turn_h0.6_v0.4", "QP_teacher", cases["turn_h0.6_v0.4"]),
            ("teacher_straight_h0.3_v1", "QP_teacher", cases["straight_h0.3_v1"]),
            ("supernet_max_turn_nominal", largest["name"], nominal)]
    output = OUTPUT.with_name("state_mode_baseline_check_v1")
    output.mkdir(parents=True, exist_ok=True)
    metadata = {"source_sha256": sha256(Path(__file__)), "E1_setup_sha256": sha256(OUTPUT/"setup.json"),
        "teacher": teacher.teacher.signature(), "jobs": [{"id": i, "mode": m, "case": c} for i,m,c in jobs],
        "purpose": "diagnostic single trials, not reliability estimates; unchanged E1 thresholds and timing",
        "encoder_hook": "records original state/references and returns the original unmodified 237D feature",
        "test_read": False}
    metadata = json.loads(json.dumps(metadata))
    setup_path = output/"setup.json"
    if setup_path.exists() and read_json(setup_path) != metadata:
        raise ValueError("baseline setup changed; retain previous diagnostics")
    atomic_json(setup_path, metadata)
    load_airsim_client()
    client = require_airsim().MultirotorClient(ip=plan["host"], port=plan["port"], timeout_value=3.)
    client.confirmConnection()
    args = SimpleNamespace(vehicle=plan["vehicle"], duration=plan["duration_s"],
        period_ms=plan["period_ms"], delivery_ms=plan["delivery_ms"],
        delivery_tolerance_ms=plan["delivery_tolerance_ms"])
    reports = []
    for name, mode, case in jobs:
        directory = output/name
        directory.mkdir(parents=True, exist_ok=True)
        path = directory/"report.json"
        if path.exists():
            report = read_json(path)
        else:
            print(f"Diagnostic: {name}", flush=True)
            trial = {"id": name, "case": case["id"], "mode": mode, "repeat": 0}
            if mode == "QP_teacher":
                with patch.object(preprocess, "encode", teacher.encode):
                    report = flight(client, teacher, layouts[case["geometry"]], case, trial, args, directory)
            else:
                report = flight(client, pool, layouts[case["geometry"]], case, trial, args, directory)
        reports.append(report)
        atomic_json(output/"summary.json", {"scope": metadata["purpose"], "trials": reports})
        print(f"  recovery={report['control_success']} timing={report['timing_comparable']} "
            f"criterion={report['recovery_criterion_passed']} error={report['error']}", flush=True)
        if report["status"] != "finished":
            raise SystemExit(1)
    print(f"Diagnostic report: {output/'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
