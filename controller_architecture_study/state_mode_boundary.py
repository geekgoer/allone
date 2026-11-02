"""Bounded discovery/confirmation of teacher-recoverable neural-control limits."""

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
from .fly_convex_policy import load_airsim_client
from .fly_decision_dnn import StoredRoutePlanner
from .state_mode_baseline_check import TeacherProbe
from .state_mode_calibration import RESULTS as CALIBRATION, moderate_case, tail_metrics
from .state_mode_flight import flight
from .state_mode_probe import OUTPUT, PROTOCOL, SupernetPool, normal_frame, read_json, make_trials


RESULTS = OUTPUT.with_name("state_mode_boundary_v1")
CONTROLLERS = ("QP_teacher", "d01_w0128", "d33_w1024")
# All conditions are declared before flight. Signed rotations are body-relative,
# not AirSim API rate commands. Geometry/reference and update timing are fixed.
PROFILES = (
    ("baseline", .4, 0., (0., 0., 0.)),
    ("outward_v1", 1., 0., (0., 0., 0.)),
    ("outward_v2", 2., 0., (0., 0., 0.)),
    ("forward_minus2", .4, -2., (0., 0., 0.)),
    ("forward_plus2", .4, 2., (0., 0., 0.)),
    ("roll_plus15", .4, 0., (15., 0., 0.)),
    ("roll_minus15", .4, 0., (-15., 0., 0.)),
    ("roll_plus30", .4, 0., (30., 0., 0.)),
    ("roll_minus30", .4, 0., (-30., 0., 0.)),
    ("pitch_plus20", .4, 0., (0., 20., 0.)),
    ("pitch_minus20", .4, 0., (0., -20., 0.)),
    ("outward_v1_roll_plus20", 1., 0., (20., 0., 0.)),
    ("outward_v1_roll_minus20", 1., 0., (-20., 0., 0.)),
)


def build_cases():
    base, layout = moderate_case()
    planner = StoredRoutePlanner(layout)
    ref = preprocess.FastReferenceWindow(planner.trajectory, planner.config).window(base["reference_phase_s"])[0]
    normal, _ = normal_frame(ref)
    tangent = np.r_[ref.velocity[:2]/np.linalg.norm(ref.velocity[:2]), 0.]
    cases = []
    for name, closing, forward, degrees in PROFILES:
        case = deepcopy(base)
        vector = np.radians(degrees)
        angle = float(np.linalg.norm(vector))
        delta = np.r_[np.cos(angle/2), vector*np.sinc(angle/(2*np.pi))/2]
        q = normalize_quaternion(quaternion_multiply(ref.orientation, delta))
        case.update(id=name, velocity=(np.asarray(base["velocity"])+normal*(closing-.4)+tangent*forward).tolist(),
            quaternion=q.tolist(), requested_h_dot_mps=-closing,
            body_rotation_error_deg=list(degrees), forward_velocity_delta_mps=forward)
        cases.append(case)
    return cases, layout


def summary(trials):
    by_case = {}
    for row in trials:
        trial = row["report"]["trial"]
        counts = by_case.setdefault(trial["case"], {}).setdefault(trial["mode"],
            {"completed": 0, "control": 0, "sustained": 0, "joint": 0, "timing_invalid": 0, "errors": []})
        counts["completed"] += 1
        counts["control"] += int(row["report"]["control_success"])
        counts["sustained"] += int(row["supplementary"]["sustained_control_passed"])
        counts["joint"] += int(row["supplementary"]["sustained_joint_passed"])
        counts["timing_invalid"] += int(not row["report"]["timing_comparable"])
        if row["report"]["error"]:
            counts["errors"].append(row["report"]["error"])
    return {"completed": len(trials), "by_case": by_case, "trials": trials,
        "scope": "bounded pilot; discovery and confirmation remain separate, no reliability/safety certificate",
        "test_read": False, "training_performed": False}


def confirmation_cases(discovery):
    """At most four cases; both directions count. No significance claim after selection."""
    groups = discovery["by_case"]
    if len(discovery["trials"]) != len(PROFILES)*len(CONTROLLERS):
        raise ValueError("complete the entire declared discovery set before selecting confirmation cases")
    selected = ["baseline"]
    positive, reverse, comparable = [], [], []
    for case, models in groups.items():
        if set(models) != set(CONTROLLERS) or any(r["completed"] != 1 for r in models.values()):
            raise ValueError("discovery groups are incomplete")
        if any(r["timing_invalid"] for r in models.values()) or not models["QP_teacher"]["joint"]:
            continue
        comparable.append(case)
        small, large = models[CONTROLLERS[1]]["sustained"], models[CONTROLLERS[2]]["sustained"]
        if large and not small:
            positive.append(case)
        if small and not large:
            reverse.append(case)
    for direction in (positive, reverse):
        remaining = sorted(set(direction)-set(selected))
        if remaining:
            selected.append(remaining[0])
    # Always include one near-threshold teacher-recoverable case. If no binary
    # split exists, confirmation stays at baseline + this one case (18 trials).
    margins = {}
    for row in discovery["trials"]:
        trial = row["report"]["trial"]
        if trial["mode"] == "QP_teacher" and trial["case"] in comparable and trial["case"] not in selected:
            tail = row["supplementary"]
            ratio = max(tail["tail_max_position_error_m"]/PROTOCOL["position_tolerance_m"],
                        tail["tail_max_velocity_error_mps"]/PROTOCOL["velocity_tolerance_mps"])
            margins[trial["case"]] = abs(1.-ratio)
    if margins:
        selected.append(min(margins, key=lambda k: (margins[k], k)))
    return selected[:4]


class BoundaryPool:
    def __init__(self, original):
        modes = [r for r in original["all_offline_modes"] if (r["depth"], r["width"]) in ((1, 128), (33, 1024))]
        modes = [{**r, "name": f"d{r['depth']:02d}_w{r['width']:04d}"} for r in modes]
        self.network = SupernetPool(Path(original["export_dir"]), modes, "cpu")
        self.teacher = TeacherProbe()

    def predict(self, name, feature):
        controller = self.teacher if name == "QP_teacher" else self.network
        return controller.predict(name, feature)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("discovery", "confirmation"), default="discovery")
    args = parser.parse_args()
    original = read_json(OUTPUT/"setup.json")["plan"]
    for name, digest in original["dependency_sha256"].items():
        if sha256(ROOT/name) != digest:
            raise ValueError(f"frozen E1 source changed: {name}")
    for name, digest in original["artifacts"].items():
        if sha256(Path(original["export_dir"])/name) != digest:
            raise ValueError(f"frozen model artifact changed: {name}")
    calibration = read_json(CALIBRATION/"setup.json")
    if sha256(ROOT/"controller_architecture_study/state_mode_calibration.py") != calibration["driver_sha256"]:
        raise ValueError("calibration implementation changed")
    controls = read_json(CALIBRATION/"summary.json")["trials"]
    required = {"teacher_moderate_nominal", "teacher_moderate_perturbed", "supernet_max_moderate_perturbed"}
    if not required <= {r["id"] for r in controls if r["supplementary"]["sustained_joint_passed"]}:
        raise ValueError("teacher/maximum baseline not validated for this reference")
    cases, layout = build_cases()
    pool = BoundaryPool(original)
    metadata = {"driver_sha256": sha256(Path(__file__)), "E1_setup_sha256": sha256(OUTPUT/"setup.json"),
        "calibration_setup_sha256": sha256(CALIBRATION/"setup.json"),
        "teacher_adapter_sha256": sha256(ROOT/"controller_architecture_study/state_mode_baseline_check.py"),
        "cases": cases, "controllers": list(CONTROLLERS), "teacher": pool.teacher.teacher.signature(),
        "runtime": pool.network.runtime, "duration_s": 3., "protocol": original["protocol"],
        "period_ms": original["period_ms"], "delivery_ms": original["delivery_ms"],
        "delivery_tolerance_ms": original["delivery_tolerance_ms"],
        "confirmation_rule": "baseline + first lexical split in each direction with teacher/all timings passing + one nearest-threshold teacher passing case; max4 cases, 3 NEW repeats each",
        "scope": "39 discovery trials; at most36 confirmation; all discovery failures retained",
        "test_read": False, "training_performed": False}
    metadata = json.loads(json.dumps(metadata))
    RESULTS.mkdir(parents=True, exist_ok=True)
    if (RESULTS/"setup.json").exists() and read_json(RESULTS/"setup.json") != metadata:
        raise ValueError("boundary setup changed; preserve prior data in its original directory")
    atomic_json(RESULTS/"setup.json", metadata)
    phase = RESULTS/args.phase
    phase.mkdir(parents=True, exist_ok=True)
    selected = [c["id"] for c in cases]
    repeats = 1
    if args.phase == "confirmation":
        selected = confirmation_cases(read_json(RESULTS/"discovery/summary.json"))
        repeats = 3
    used = [c for c in cases if c["id"] in selected]
    trials = make_trials(used, [{"name": m} for m in CONTROLLERS], repeats,
                         20261023+(args.phase == "confirmation"))
    plan = {"cases": selected, "repeats": repeats, "trials": trials,
        "discovery_summary_sha256": sha256(RESULTS/"discovery/summary.json") if args.phase == "confirmation" else None}
    if (phase/"setup.json").exists() and read_json(phase/"setup.json") != plan:
        raise ValueError("phase setup changed")
    atomic_json(phase/"setup.json", plan)
    done = [read_json(phase/t["id"]/'result.json') for t in trials if (phase/t["id"]/'result.json').exists()]
    if any(r["report"]["status"] != "finished" for r in done):
        raise ValueError("a previous trial was interrupted or invalid; inspect it rather than silently retrying")
    atomic_json(phase/"summary.json", summary(done))
    pending = [t for t in trials if t["id"] not in {r["report"]["trial"]["id"] for r in done}]
    if not pending:
        print(f"All {len(done)} trials already saved: {phase/'summary.json'}")
        return
    # Warm real case inputs and save aligned initial outputs without changing
    # any controller. This is an OFFLINE comparison, not a runtime error signal.
    from airsim_interface import RawDroneState, require_airsim
    planner = StoredRoutePlanner(layout)
    window = preprocess.FastReferenceWindow(planner.trajectory, planner.config)
    initial = []
    for case in used:
        state = RawDroneState(case["position"], case["quaternion"], case["velocity"], np.zeros(3), np.zeros(3))
        refs = window.window(case["reference_phase_s"])
        feature = pool.teacher.encode(state, refs)
        initial.append({"case": case["id"], "outputs": {m: pool.predict(m, feature)[0].tolist() for m in CONTROLLERS}})
    if not (phase/"initial_outputs.json").exists():
        atomic_json(phase/"initial_outputs.json", initial)
    load_airsim_client()
    client = require_airsim().MultirotorClient(ip=original["host"], port=original["port"], timeout_value=3.)
    client.confirmConnection()
    if original["vehicle"] not in client.listVehicles() or client.simIsPause():
        raise RuntimeError("manual Play must be active and unpaused")
    foreign = [n for n in client.simListSceneObjects("Falcon.*Gate.*") if not n.startswith("FalconModeGate_")]
    if foreign:
        raise RuntimeError("unrelated generated gates present; do not mix experiments")
    flight_args = SimpleNamespace(vehicle=original["vehicle"], duration=3., period_ms=original["period_ms"],
        delivery_ms=original["delivery_ms"], delivery_tolerance_ms=original["delivery_tolerance_ms"])
    cases = {c["id"]: c for c in used}
    for trial in pending:
        directory = phase/trial["id"]
        directory.mkdir(parents=True, exist_ok=True)
        print(f"{args.phase} [{len(done)+1}/{len(trials)}] {trial['id']}", flush=True)
        if (directory/"report.json").exists():
            report = read_json(directory/"report.json")
        else:
            with patch.object(preprocess, "encode", pool.teacher.encode):
                report = flight(client, pool, layout, cases[trial["case"]], trial, flight_args, directory)
        row = {"report": report, "supplementary": tail_metrics(read_json(directory/"trace.json"), report)}
        atomic_json(directory/"result.json", row)
        done.append(row)
        atomic_json(phase/"summary.json", summary(done))
        print(f"  sustained={row['supplementary']['sustained_control_passed']} "
            f"timing={report['timing_comparable']} error={report['error']}", flush=True)
        if report["status"] != "finished":
            raise SystemExit(1)
    print(f"Boundary report: {phase/'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
