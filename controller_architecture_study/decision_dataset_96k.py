"""96k independent ACADO queries; multiple committed routes per held-out group."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np

from airsim_interface import RawDroneState
from state_adapter import normalize_quaternion, quaternion_multiply
from . import decision_dataset as original
from .decision_dataset import (CONTRACT, FAMILIES, PLANNER_CONFIG, ROOT, TEACHER_CONFIG,
                               RefinedTeacher, atomic_json, atomic_npz, build_features, sha256)
from falcon_paper_fidelity.motion_primitive import QuinticMotionPrimitivePlanner


EXPERIMENT = ROOT / "controller_architecture_study/results/decision_nmpc_96k_v2"
VERSION = "independent_state_multiroute_96k_v2"
ENVIRONMENTS = 60
SPLITS = {"train": list(range(48)), "validation": list(range(48, 54)), "test": list(range(54, 60))}


def edge_uniform(rng, radius):
    # Same distribution as the old 12D sampler: 30% focus on an outer 15% band.
    radius = np.broadcast_to(np.asarray(radius, dtype=float), (3,))
    value = rng.uniform(-radius, radius)
    focus = rng.random(3) < .30
    sign = np.where(rng.random(3) < .5, -1., 1.)
    edge = sign * (radius - rng.uniform(0., .30 * radius))
    return np.where(focus, edge, value)


def sample_problem(trajectory, planner, rng, family):
    # Reuse the committed phase/window policy, not its uniform state perturbation.
    _, references, phase = original.sample_problem(trajectory, planner, rng, family)
    reference = references[0]
    position, velocity = reference.position.copy(), reference.velocity.copy()
    quaternion = reference.orientation.copy()
    if family in (0, 1, 2, 3):
        position += edge_uniform(rng, .08)
        velocity += edge_uniform(rng, .15)
    if family in (4, 7):
        position += edge_uniform(rng, [.6, 1., .6])
    if family in (5, 7):
        velocity += edge_uniform(rng, 1.5)
    if family in (6, 7):
        angle = edge_uniform(rng, .25)
        magnitude = np.linalg.norm(angle)
        delta = np.r_[np.cos(magnitude / 2), angle * np.sin(magnitude / 2) / max(magnitude, 1e-12)]
        quaternion = normalize_quaternion(quaternion_multiply(quaternion, delta))
    if family == 0:
        velocity *= min(1., 3.5 / max(np.linalg.norm(velocity), 1e-12))
        if phase == 0.:
            velocity[:] = 0.
    velocity *= min(1., 8. / max(np.linalg.norm(velocity), 1e-12))
    return RawDroneState(position, quaternion, velocity, np.zeros(3), np.zeros(3)), references, phase


def route_for_group(environment_id, route_index, routes_per_environment, planner):
    route_id = environment_id * routes_per_environment + route_index
    # Disjoint from v1's 0..39 routes. Twenty realizations contain five of each geometry.
    trajectory, layout = original.route_for_environment(200000 + route_id, planner)
    layout.update(environment_id=environment_id, route_id=route_id,
                  parent_route_generator_id=200000 + route_id)
    orientation = layout["gates"][0]["orientation"]
    layout["heading_yaw"] = float(2 * np.arctan2(orientation[3], orientation[0]))
    return trajectory, layout


def generate(output, *, smoke=False, resume=False):
    teacher = RefinedTeacher()
    planner = QuinticMotionPrimitivePlanner(PLANNER_CONFIG)
    routes_per_environment, samples_per_route = (1, 8) if smoke else (20, 80)
    config = {"version": VERSION, "contract": CONTRACT, "seed": 20261002,
              "environments": ENVIRONMENTS, "routes_per_environment": routes_per_environment,
              "samples_per_route": samples_per_route, "families": list(FAMILIES),
              "edge_probability": .30, "edge_band_fraction": .15,
              "teacher": TEACHER_CONFIG, "max_attempts": 8, "environment_splits": SPLITS,
              "dll_sha256": sha256(teacher.controller.library_path),
              "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in
                  (Path(__file__), Path(original.__file__), ROOT / "falcon_paper_fidelity/acado_mpc.py",
                   ROOT / "falcon_paper_fidelity/mpc_complete_300d.py",
                   ROOT / "falcon_paper_fidelity/motion_primitive.py", ROOT / "state_adapter.py",
                   ROOT / "airsim_interface.py", ROOT / "environment.py")}}
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
    started = time.perf_counter()
    total = ENVIRONMENTS * routes_per_environment * samples_per_route
    for environment_id in range(ENVIRONMENTS):
        for route_index in range(routes_per_environment):
            name = f"env{environment_id:02d}_route{route_index:02d}"
            chunk_path = output / "chunks" / f"{name}.npz"
            route_path = output / "routes" / f"{name}.json"
            if name in manifest["chunks"]:
                metadata = manifest["chunks"][name]
                if (not chunk_path.exists() or not route_path.exists()
                        or sha256(chunk_path) != metadata["sha256"]
                        or sha256(route_path) != metadata["route_sha256"]):
                    raise ValueError(f"missing or changed completed chunk/route: {name}")
                continue
            trajectory, layout = route_for_group(environment_id, route_index, routes_per_environment, planner)
            records, rejected = [], Counter()
            for index in range(samples_per_route):
                row_id = layout["route_id"] * samples_per_route + index
                rng = np.random.default_rng(np.random.SeedSequence([20261002, environment_id, route_index, index]))
                family = index % len(FAMILIES)
                for attempt in range(config["max_attempts"]):
                    state, references, phase = sample_problem(trajectory, planner, rng, family)
                    try:
                        control, diagnostics = teacher.query(state, references)
                    except RuntimeError as exc:
                        rejected[FAMILIES[family] + ": " + str(exc).split(":")[0]] += 1
                        if attempt == config["max_attempts"] - 1:
                            atomic_json(output / "failure.json", {"row_id": row_id, "route_id": layout["route_id"],
                                "family": FAMILIES[family], "error": str(exc), "rejected_in_chunk": dict(rejected)})
                            raise
                        continue
                    repeat_delta = 0.
                    if index == 0:
                        repeated, _ = teacher.query(state, references)
                        repeat_delta = float(np.max(np.abs(repeated - control)))
                        if repeat_delta > 2e-5:
                            raise RuntimeError(f"independent label repeatability failed: {repeat_delta}")
                    records.append({"X": build_features(state, references), "U": control,
                        "row_id": row_id, "environment_id": environment_id, "route_id": layout["route_id"],
                        "geometry_family": layout["geometry_family"], "heading_yaw": layout["heading_yaw"],
                        "family_id": family, "phase_seconds": phase,
                        "actual_speed": float(np.linalg.norm(state.velocity)),
                        "reference_speed": float(np.linalg.norm(references[0].velocity)),
                        "position_error": float(np.linalg.norm(state.position - references[0].position)),
                        "repeat_delta": repeat_delta, "repeat_checked": index == 0,
                        "rejected_before_acceptance": attempt, **diagnostics})
                    break
            atomic_json(route_path, layout)
            atomic_npz(chunk_path, {key: np.asarray([row[key] for row in records]) for key in records[0]})
            manifest["chunks"][name] = {"sha256": sha256(chunk_path), "route_sha256": sha256(route_path),
                                        "samples": len(records), "rejected": dict(rejected)}
            atomic_json(manifest_path, manifest)
            completed = sum(chunk["samples"] for chunk in manifest["chunks"].values())
            print(f"accepted={completed}/{total} env={environment_id} route={route_index} "
                  f"rejected={sum(rejected.values())} elapsed_this_run={time.perf_counter() - started:.1f}s", flush=True)
    manifest["complete"] = True
    atomic_json(manifest_path, manifest)
    print(f"Collection complete: {output}; run prepare next.", flush=True)


def prepare(output):
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    config = manifest["config"]
    if not manifest["complete"] or config["version"] != VERSION or config["contract"] != CONTRACT:
        raise ValueError("collection is incomplete or has the wrong version/contract")
    blocks = []
    for name, metadata in sorted(manifest["chunks"].items()):
        path, route_path = output / "chunks" / f"{name}.npz", output / "routes" / f"{name}.json"
        if sha256(path) != metadata["sha256"] or sha256(route_path) != metadata["route_sha256"]:
            raise ValueError(f"changed chunk/route: {name}")
        with np.load(path, allow_pickle=False) as archive:
            blocks.append({key: archive[key] for key in archive.files})
    data = {key: np.concatenate([block[key] for block in blocks]) for key in blocks[0]}
    per_route = config["samples_per_route"]
    per_environment = config["routes_per_environment"] * per_route
    expected = ENVIRONMENTS * per_environment
    if (data["X"].shape != (expected, 300) or data["U"].shape != (expected, 4)
            or not np.array_equal(np.sort(data["row_id"]), np.arange(expected))
            or any(not np.isfinite(value).all() for value in data.values())
            or np.any(data["route_id"] != data["row_id"] // per_route)
            or np.any(data["environment_id"] != data["row_id"] // per_environment)
            or np.any(data["family_id"] != data["row_id"] % len(FAMILIES))):
        raise ValueError("invalid shapes, row/route/environment/family mapping or nonfinite values")
    if (np.max(data["kkt"]) > TEACHER_CONFIG["kkt_tolerance"]
            or np.max(data["action_delta"]) > TEACHER_CONFIG["action_delta_tolerance"]):
        raise ValueError("unconverged labels in dataset")
    lower, upper = np.array([2., -3., -3., -2.]), np.array([20., 3., 3., 2.])
    if np.any(data["U"] < lower - 1e-4) or np.any(data["U"] > upper + 1e-4):
        raise ValueError("labels exceed compiled control limits")
    if np.any(data["X"][:, :3] != 0.) or np.max(data["repeat_delta"]) > 2e-5:
        raise ValueError("invalid local origin or failed label repeatability")
    dataset = output / "dnn_dataset"
    split_manifest = {"version": VERSION, "contract": CONTRACT,
                      "source_manifest_sha256": sha256(output / "manifest.json"),
                      "environment_ids": SPLITS, "route_ids": {}, "counts": {}}
    saved_path = dataset / "split_manifest.json"
    if saved_path.exists():
        saved = json.loads(saved_path.read_text(encoding="utf-8"))
        if saved["source_manifest_sha256"] != split_manifest["source_manifest_sha256"]:
            raise ValueError("prepared data source changed; use a new output")
    for split, ids in SPLITS.items():
        selected = np.isin(data["environment_id"], ids)
        atomic_npz(dataset / f"{split}.npz", {key: value[selected] for key, value in data.items()})
        split_manifest["counts"][split] = int(selected.sum())
        split_manifest["route_ids"][split] = np.unique(data["route_id"][selected]).tolist()
    train = np.isin(data["environment_id"], SPLITS["train"])
    validation = np.isin(data["environment_id"], SPLITS["validation"])
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
    minimum, maximum = data["X"][train].min(axis=0), data["X"][train].max(axis=0)
    tolerance = 1e-6 * np.maximum(1., maximum - minimum)
    outside = (data["X"][validation] < minimum - tolerance) | (data["X"][validation] > maximum + tolerance)
    outside_per_dimension = outside.mean(axis=0)
    audit = {"status": "PASS", "version": VERSION, "samples": expected, "input_contract": CONTRACT,
             "environments": ENVIRONMENTS, "routes": int(len(np.unique(data["route_id"]))),
             "split_counts": split_manifest["counts"],
             "split_route_counts": {key: len(ids) for key, ids in split_manifest["route_ids"].items()},
             "max_kkt": float(data["kkt"].max()), "max_label_repeat_delta": float(data["repeat_delta"].max()),
             "repeat_checked_samples": int(data["repeat_checked"].sum()),
             "actual_speed_min_max": [float(data["actual_speed"].min()), float(data["actual_speed"].max())],
             "speed_le_3_5_fraction": float(np.mean(data["actual_speed"] <= 3.5)),
             "rest_start_samples": int(np.sum((data["family_id"] == 0) & (data["phase_seconds"] == 0.) & (data["actual_speed"] == 0.))),
             "rejected_queries": dict(rejected), "saturated_fraction": float(saturated.mean()),
             "families": {family: {"samples": int(np.sum(data["family_id"] == index)),
                          "saturated_fraction": float(saturated[data["family_id"] == index].mean())}
                          for index, family in enumerate(FAMILIES)},
             "validation_coverage": {"rows_outside_train_min_max_fraction": float(np.mean(outside.any(axis=1))),
                 "outside_fraction_per_input_dimension": outside_per_dimension.tolist(),
                 "route_heading_yaw_range_rad": {key: [float(data["heading_yaw"][mask].min()),
                     float(data["heading_yaw"][mask].max())] for key, mask in (("train", train), ("validation", validation))},
                 "note": "Marginal range diagnostic only, not joint-distribution coverage or an acceptance criterion."},
             "test_used_for_model_selection": False,
             "note": "Numerical data audit, not a DNN closed-loop guarantee or a promise of no train/validation gap."}
    atomic_json(output / "data_audit.json", audit)
    # Keep the console compact; full 300-dimensional coverage remains in data_audit.json.
    print(json.dumps({key: value for key, value in audit.items() if key != "validation_coverage"}, indent=2), flush=True)
    print(f"Prepared dataset: {dataset}\nFull audit: {output / 'data_audit.json'}", flush=True)


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
