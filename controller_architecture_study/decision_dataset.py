"""Independent NMPC queries on committed C2 routes; never connects to AirSim."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from airsim_interface import RawDroneState
from environment import Gate
from falcon_paper_fidelity.acado_mpc import AcadoRtiMPC
from falcon_paper_fidelity.motion_primitive import MotionPrimitiveConfig, QuinticMotionPrimitivePlanner
from falcon_paper_fidelity.mpc_complete_300d import build_mpc_complete_300d, canonicalize_problem
from state_adapter import normalize_quaternion, quaternion_multiply


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "controller_architecture_study/results/decision_nmpc_v1"
CONTRACT = "decision_nmpc_local_relative_pv_300d_v1"
FAMILIES = ("startup", "tracking", "transition", "braking", "position_recovery",
            "velocity_recovery", "attitude_recovery", "combined_recovery")
PLANNER_CONFIG = MotionPrimitiveConfig(nominal_speed=5., terminal_speed=5., maximum_acceleration=8.)
TEACHER_CONFIG = {"minimum_iterations": 4, "maximum_iterations": 32,
                  "action_delta_tolerance": 1e-4, "kkt_tolerance": 1e-3}


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_npz(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **values)
    temporary.replace(path)


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def local_problem(state, references):
    # q_perception=0: common translation leaves the MPC problem unchanged.
    origin = state.position
    state, references = canonicalize_problem(state, references)
    return replace(state, position=np.zeros(3)), [replace(ref, position=ref.position - origin) for ref in references]


def build_features(state, references):
    """Online encoder: local position; reference velocities relative to current v.

    Z-up/FLU: [current p=0,q,v] + 21*[relative p,q,relative v] + 20*planner u.
    Planner u is acceleration/body-rate feed-forward, NOT optimized MPC output.
    """
    local_state, local_refs = local_problem(state, references)
    feature = build_mpc_complete_300d(local_state, local_refs)
    feature[10:220].reshape(21, 10)[:, 7:10] -= feature[7:10]
    return feature.astype(np.float32)


class RefinedTeacher:
    """Offline, fully reset default ACADO solver; sequential queries only.

    initialize() clears primal AND dual workspace and restores the native default
    weights/limits. This is deliberately separate from legacy reset() semantics.
    """

    def __init__(self):
        self.controller = AcadoRtiMPC()

    def query(self, state, references):
        state, references = local_problem(state, references)
        status = int(self.controller._dll.falcon_acado_initialize())
        self.controller._raise_status(status, "independent query initialize")
        self.controller.reset()
        previous = None
        delta = float("inf")
        started = time.perf_counter()
        for iteration in range(1, TEACHER_CONFIG["maximum_iterations"] + 1):
            control = self.controller.solve(state, references)
            if previous is not None:
                delta = float(np.max(np.abs(control - previous)))
            previous = control.copy()
            kkt = abs(self.controller.last_diagnostics["kkt"])
            if (iteration >= TEACHER_CONFIG["minimum_iterations"]
                    and delta <= TEACHER_CONFIG["action_delta_tolerance"]
                    and np.isfinite(kkt) and kkt <= TEACHER_CONFIG["kkt_tolerance"]):
                lower, upper = np.array([2., -3., -3., -2.]), np.array([20., 3., 3., 2.])
                if np.any(control < lower - 1e-4) or np.any(control > upper + 1e-4):
                    raise RuntimeError("teacher control outside compiled limits")
                return control.astype(np.float32), {"iterations": iteration, "kkt": kkt,
                    "action_delta": delta, "solve_ms": (time.perf_counter() - started) * 1000}
        raise RuntimeError(f"unconverged teacher: kkt={kkt:.6g}, action_delta={delta:.6g}")


def route_for_environment(environment_id, planner):
    rng = np.random.default_rng(20261002 + environment_id)
    # Every split contains all four geometric families. Realizations are disjoint.
    kind = environment_id % 4
    x = np.arange(1., 6.) * rng.uniform(9., 11.)
    amplitude = rng.uniform(.8, 2.5)
    y = np.zeros(5) if kind == 0 else amplitude * np.array([0., 1., -1., 1., 0.])
    z = np.full(5, -4.)
    if kind >= 2:
        z += rng.uniform(.3, .9) * np.array([0., -1., 0., -1., 0.])
    y += rng.uniform(-.3, .3, 5)
    # One heading per route: do not introduce discontinuous yaw at gate boundaries.
    yaw = np.full(5, rng.uniform(-.15, .15)) if kind == 3 else np.zeros(5)
    gates = [Gate(i + 1, np.array([x[i], y[i], z[i]]),
                  np.array([np.cos(yaw[i] / 2), 0., 0., np.sin(yaw[i] / 2)])) for i in range(5)]
    trajectory = planner.plan_continuous_route(np.array([0., 0., -4.]), np.zeros(3),
                                              np.zeros(3), gates, final_exit_distance=5.)
    return trajectory, {"environment_id": environment_id, "geometry_family": kind,
        "gates": [{"index": gate.index, "center": gate.center.tolist(),
                   "orientation": gate.orientation.tolist()} for gate in gates],
        "duration": trajectory.duration,
        "segments": [{"start_time": segment.start_time, "duration": segment.primitive.duration,
                      "coefficients": segment.primitive.coefficients.tolist(),
                      "heading_orientation": segment.heading_orientation.tolist()} for segment in trajectory.segments]}


def sample_problem(trajectory, planner, rng, family):
    if family == 0:
        phase = rng.uniform(0., min(2., trajectory.segments[0].end_time))
        if rng.random() < .25:
            phase = 0.
    elif family == 2:
        boundary = rng.choice(trajectory.boundary_times)
        phase = np.clip(boundary + rng.uniform(-.8, .8), 0., trajectory.duration)
    elif family == 3:
        phase = rng.uniform(trajectory.segments[-1].start_time, trajectory.duration)
    else:
        phase = rng.uniform(0., trajectory.duration)
    references = planner.reference_window_for_trajectory(trajectory, elapsed=float(phase), dt=.1, horizon=21)
    reference = references[0]
    position, velocity, quaternion = reference.position.copy(), reference.velocity.copy(), reference.orientation.copy()
    if family in (0, 1, 2, 3):
        position += rng.uniform(-.08, .08, 3)
        velocity += rng.uniform(-.15, .15, 3)
    if family in (4, 7):
        position += rng.uniform(-1., 1., 3) * np.array([.6, 1., .6])
    if family in (5, 7):
        velocity += rng.uniform(-1.5, 1.5, 3)
    if family in (6, 7):
        angle = rng.uniform(-.25, .25, 3)
        magnitude = np.linalg.norm(angle)
        perturbation = np.r_[np.cos(magnitude / 2), angle * np.sin(magnitude / 2) / max(magnitude, 1e-12)]
        quaternion = normalize_quaternion(quaternion_multiply(quaternion, perturbation))
    if family == 0:
        velocity *= min(1., 3.5 / max(np.linalg.norm(velocity), 1e-12))
        if phase == 0.:
            velocity[:] = 0.
    if np.linalg.norm(velocity) > 8.:
        velocity *= 8. / np.linalg.norm(velocity)
    # Current acceleration is not an ACADO feedback state; no nuisance noise.
    state = RawDroneState(position, quaternion, velocity, np.zeros(3), np.zeros(3))
    return state, references, float(phase)


def generate(output, *, smoke=False, resume=False):
    teacher = RefinedTeacher()
    planner = QuinticMotionPrimitivePlanner(PLANNER_CONFIG)
    per_environment = 8 if smoke else 1000
    config = {"contract": CONTRACT, "seed": 20261002, "environments": 40,
              "samples_per_environment": per_environment, "chunk_size": 100,
              "families": list(FAMILIES), "teacher": TEACHER_CONFIG, "max_attempts": 8,
              "dll_sha256": sha256(teacher.controller.library_path),
              "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in
                  (Path(__file__), ROOT / "falcon_paper_fidelity/acado_mpc.py",
                   ROOT / "falcon_paper_fidelity/mpc_complete_300d.py",
                   ROOT / "falcon_paper_fidelity/motion_primitive.py")}}
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not resume or manifest["config"] != config:
            raise ValueError("output exists: use --resume with unchanged code/config/DLL")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("refusing to overwrite a nonempty unrecognized output")
        manifest = {"config": config, "chunks": {}, "complete": False}
        atomic_json(manifest_path, manifest)
    layouts = []
    started = time.perf_counter()
    for environment_id in range(40):
        trajectory, layout = route_for_environment(environment_id, planner)
        layouts.append(layout)
        for start in range(0, per_environment, 100):
            name = f"env{environment_id:02d}_{start:04d}.npz"
            path = output / "chunks" / name
            if name in manifest["chunks"]:
                if not path.exists() or sha256(path) != manifest["chunks"][name]["sha256"]:
                    raise ValueError(f"missing or changed completed chunk: {path}")
                continue
            records, rejected = [], Counter()
            for index in range(start, min(start + 100, per_environment)):
                row_id = environment_id * per_environment + index
                rng = np.random.default_rng(np.random.SeedSequence([20261002, environment_id, index]))
                family = index % len(FAMILIES)
                for attempt in range(8):
                    state, references, phase = sample_problem(trajectory, planner, rng, family)
                    try:
                        control, diagnostics = teacher.query(state, references)
                    except RuntimeError as exc:
                        rejected[FAMILIES[family] + ": " + str(exc).split(":")[0]] += 1
                        if attempt == 7:
                            atomic_json(output / "failure.json", {"row_id": row_id, "family": FAMILIES[family],
                                "error": str(exc), "rejected_in_chunk": dict(rejected)})
                            raise
                        continue
                    repeat_delta = 0.
                    if index == 0:
                        repeat, _ = teacher.query(state, references)
                        repeat_delta = float(np.max(np.abs(repeat - control)))
                        if repeat_delta > 2e-5:
                            raise RuntimeError(f"independent label repeatability failed: {repeat_delta}")
                    records.append({"X": build_features(state, references), "U": control,
                        "row_id": row_id, "environment_id": environment_id, "family_id": family,
                        "phase_seconds": phase, "actual_speed": float(np.linalg.norm(state.velocity)),
                        "reference_speed": float(np.linalg.norm(references[0].velocity)),
                        "position_error": float(np.linalg.norm(state.position - references[0].position)),
                        "repeat_delta": repeat_delta, "repeat_checked": index == 0, "rejected_before_acceptance": attempt,
                        **diagnostics})
                    break
            values = {key: np.asarray([record[key] for record in records]) for key in records[0]}
            atomic_npz(path, values)
            manifest["chunks"][name] = {"sha256": sha256(path), "samples": len(records), "rejected": dict(rejected)}
            atomic_json(manifest_path, manifest)
            completed = sum(chunk["samples"] for chunk in manifest["chunks"].values())
            print(f"accepted={completed}/{40 * per_environment} env={environment_id} chunk={start} "
                  f"rejected={sum(rejected.values())} elapsed_this_run={time.perf_counter() - started:.1f}s", flush=True)
    atomic_json(output / "environments.json", layouts)
    manifest["complete"] = True
    atomic_json(manifest_path, manifest)
    print(f"Collection complete: {output}; run prepare next.", flush=True)


def prepare(output):
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if not manifest["complete"] or manifest["config"]["contract"] != CONTRACT:
        raise ValueError("collection is incomplete or has the wrong contract")
    blocks = []
    for name, metadata in sorted(manifest["chunks"].items()):
        path = output / "chunks" / name
        if sha256(path) != metadata["sha256"]:
            raise ValueError(f"changed chunk: {path}")
        with np.load(path, allow_pickle=False) as archive:
            blocks.append({key: archive[key] for key in archive.files})
    data = {key: np.concatenate([block[key] for block in blocks]) for key in blocks[0]}
    expected = 40 * manifest["config"]["samples_per_environment"]
    if (data["X"].shape != (expected, 300) or data["U"].shape != (expected, 4)
            or len(np.unique(data["row_id"])) != expected
            or any(not np.isfinite(value).all() for value in data.values())):
        raise ValueError("invalid shapes, duplicate rows or nonfinite values")
    if np.max(data["kkt"]) > TEACHER_CONFIG["kkt_tolerance"] or np.max(data["action_delta"]) > TEACHER_CONFIG["action_delta_tolerance"]:
        raise ValueError("unconverged labels in dataset")
    lower, upper = np.array([2., -3., -3., -2.]), np.array([20., 3., 3., 2.])
    if np.any(data["U"] < lower - 1e-4) or np.any(data["U"] > upper + 1e-4):
        raise ValueError("labels exceed compiled control limits")
    if np.any(data["X"][:, :3] != 0.) or np.max(data["repeat_delta"]) > 2e-5:
        raise ValueError("invalid local origin or failed label repeatability")
    dataset = output / "dnn_dataset"
    # Fixed environment split, never reshuffle anchors across partitions.
    environment_splits = {"train": list(range(32)), "validation": list(range(32, 36)), "test": list(range(36, 40))}
    split_manifest = {"contract": CONTRACT, "source_manifest_sha256": sha256(output / "manifest.json"),
                      "environment_ids": environment_splits, "counts": {}}
    saved_path = dataset / "split_manifest.json"
    if saved_path.exists():
        saved = json.loads(saved_path.read_text(encoding="utf-8"))
        if saved["source_manifest_sha256"] != split_manifest["source_manifest_sha256"]:
            raise ValueError("prepared data source changed; use a new output")
    for split, ids in environment_splits.items():
        selected = np.isin(data["environment_id"], ids)
        atomic_npz(dataset / f"{split}.npz", {key: value[selected] for key, value in data.items()})
        split_manifest["counts"][split] = int(selected.sum())
    train = data["environment_id"] < 32
    normalization = {"input_contract": CONTRACT, "fit_split": "train", "fit_samples": int(train.sum())}
    for name, field in (("x", "X"), ("u", "U")):
        values = data[field][train].astype(np.float64)
        normalization[name + "_mean"] = values.mean(axis=0).tolist()
        normalization[name + "_std"] = np.maximum(values.std(axis=0), 1e-6).tolist()
    atomic_json(dataset / "normalization.json", normalization)
    atomic_json(saved_path, split_manifest)
    saturated = np.any((data["U"] <= lower + .01) | (data["U"] >= upper - .01), axis=1)
    rejected = Counter()
    for chunk in manifest["chunks"].values():
        rejected.update(chunk["rejected"])
    audit = {"status": "PASS", "samples": expected, "input_contract": CONTRACT,
             "split_counts": split_manifest["counts"], "max_kkt": float(data["kkt"].max()),
             "max_label_repeat_delta": float(data["repeat_delta"].max()),
             "repeat_checked_samples": int(data["repeat_checked"].sum()),
             "actual_speed_min_max": [float(data["actual_speed"].min()), float(data["actual_speed"].max())],
             "speed_le_3_5_fraction": float(np.mean(data["actual_speed"] <= 3.5)),
             "rest_start_samples": int(np.sum((data["family_id"] == 0) & (data["phase_seconds"] == 0.) & (data["actual_speed"] == 0.))),
             "rejected_queries": dict(rejected), "saturated_fraction": float(saturated.mean()),
             "families": {family: {"samples": int(np.sum(data["family_id"] == index)),
                          "saturated_fraction": float(saturated[data["family_id"] == index].mean())}
                          for index, family in enumerate(FAMILIES)},
             "test_used_for_model_selection": False,
             "note": "Dataset quality check, not a DNN closed-loop guarantee. Independent recovery states are not verified rollouts."}
    atomic_json(output / "data_audit.json", audit)
    print(json.dumps(audit, indent=2), flush=True)
    print(f"Prepared dataset: {dataset}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("generate", "prepare"))
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = EXPERIMENT.with_name(EXPERIMENT.name + "_smoke") if args.smoke_test else EXPERIMENT
    if args.stage == "generate":
        generate(output, smoke=args.smoke_test, resume=args.resume)
    else:
        prepare(output)


if __name__ == "__main__":
    main()
