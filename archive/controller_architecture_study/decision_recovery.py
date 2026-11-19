"""Train-only recovery augmentation of the frozen 96k dataset; no AirSim or Test reads."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import time

import numpy as np

from airsim_interface import RawDroneState
from state_adapter import (normalize_quaternion, quaternion_inverse,
                           quaternion_multiply, quaternion_to_rotation_vector)
from .decision_dataset import (CONTRACT, ROOT, TEACHER_CONFIG, RefinedTeacher,
                               atomic_json, atomic_npz, build_features, sha256)
from .decision_dataset_96k import EXPERIMENT, VERSION, edge_uniform
from .fly_decision_dnn import StoredRoutePlanner


OUTPUT = EXPERIMENT.with_name("decision_recovery_v1")
KINDS = ("position_velocity", "position_attitude", "velocity_attitude",
         "scaled_combined", "saturated_train_neighbor")
LOWER = np.array([2., -3., -3., -2.])
UPPER = np.array([20., 3., 3., 2.])
POSITION_RADIUS = np.array([.6, 1., .6])


def read_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def source_info(base):
    source = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    split = json.loads((base / "dnn_dataset/split_manifest.json").read_text(encoding="utf-8"))
    if (not source["complete"] or source["config"]["version"] != VERSION
            or split["contract"] != CONTRACT
            or split["source_manifest_sha256"] != sha256(base / "manifest.json")):
        raise ValueError("requires unchanged, complete 96k source and prepared split")
    ids = split["environment_ids"]
    if set(ids["train"]) & (set(ids["validation"]) | set(ids["test"])):
        raise ValueError("source Train and held-out groups overlap")
    return source, split


def sample_state(reference, rng, kind, feature=None):
    """Change state only: the caller holds phase and all 21 references fixed."""
    p = rng.uniform(-.05, .05, 3)
    v = rng.uniform(-.10, .10, 3)
    angle = rng.uniform(-.015, .015, 3)
    scale = rng.uniform(.10, 1.)
    if kind in (0, 1, 3):
        p = rng.uniform(-POSITION_RADIUS, POSITION_RADIUS) * scale
    if kind in (0, 2, 3):
        v = rng.uniform(-1.5, 1.5, 3) * scale
    if kind in (1, 2, 3):
        angle = rng.uniform(-.25, .25, 3) * scale
    if kind == 3 and rng.random() < .20:
        # Keep hard edges too; most added rows fill the interior rather than corners.
        p, v, angle = edge_uniform(rng, POSITION_RADIUS), edge_uniform(rng, 1.5), edge_uniform(rng, .25)
    if kind == 4:
        # Stored features are Z-up/FLU; planner/state sampling uses AirSim NED/FRD.
        signs = np.array([1., -1., -1.])
        p = np.clip(-feature[10:13] * signs + rng.uniform(-.08, .08, 3), -POSITION_RADIUS, POSITION_RADIUS)
        v = np.clip(feature[7:10] * signs - reference.velocity + rng.uniform(-.20, .20, 3), -1.5, 1.5)
        relative = quaternion_multiply(quaternion_inverse(reference.orientation), feature[3:7] * [1., 1., -1., -1.])
        angle = np.clip(quaternion_to_rotation_vector(relative) + rng.uniform(-.03, .03, 3), -.25, .25)
    magnitude = np.linalg.norm(angle)
    delta = np.r_[np.cos(magnitude / 2), angle * np.sin(magnitude / 2) / max(magnitude, 1e-12)]
    quaternion = normalize_quaternion(quaternion_multiply(reference.orientation, delta))
    velocity = reference.velocity + v
    velocity *= min(1., 8. / max(np.linalg.norm(velocity), 1e-12))
    return RawDroneState(reference.position + p, quaternion, velocity, np.zeros(3), np.zeros(3))


def source_route(base, source, name):
    metadata = source["chunks"][name]
    route, chunk = base / "routes" / f"{name}.json", base / "chunks" / f"{name}.npz"
    if sha256(route) != metadata["route_sha256"] or sha256(chunk) != metadata["sha256"]:
        raise ValueError(f"source route/chunk changed: {name}")
    return json.loads(route.read_text(encoding="utf-8")), read_npz(chunk)


def generate(base, output, *, resume=False):
    base, output = base.resolve(), output.resolve()
    if output == base or base in output.parents or output in base.parents:
        raise ValueError("augmentation output must be separate from the source")
    source, split = source_info(base)
    teacher = RefinedTeacher()
    if sha256(teacher.controller.library_path) != source["config"]["dll_sha256"]:
        raise ValueError("teacher DLL differs from the original labels")
    config = {"version": "train_recovery_32_per_route_v1", "contract": CONTRACT,
        "base": str(base), "base_manifest_sha256": sha256(base / "manifest.json"),
        "base_file_sha256": {name: sha256(base / "dnn_dataset" / name) for name in
                            ("train.npz", "validation.npz", "normalization.json", "split_manifest.json")},
        "teacher": TEACHER_CONFIG, "dll_sha256": sha256(teacher.controller.library_path),
        "seed": 20261019, "samples_per_route": 32, "max_attempts": 8, "kinds": list(KINDS),
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in
            (Path(__file__), ROOT / "controller_architecture_study/decision_dataset.py",
             ROOT / "controller_architecture_study/decision_dataset_96k.py",
             ROOT / "controller_architecture_study/fly_decision_dnn.py",
             ROOT / "falcon_paper_fidelity/acado_mpc.py", ROOT / "state_adapter.py",
             ROOT / "falcon_paper_fidelity/motion_primitive.py", ROOT / "falcon_paper_fidelity/mpc_complete_300d.py")}}
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not resume or manifest["config"] != config:
            raise ValueError("use --resume with unchanged source/code/DLL, or a new output")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("refusing nonempty unrecognized output")
        manifest = {"config": config, "chunks": {}, "complete": False}
        atomic_json(manifest_path, manifest)
    train_routes = split["route_ids"]["train"]
    routes_per_environment = source["config"]["routes_per_environment"]
    row_offset = sum(split["counts"].values())
    started = time.perf_counter()
    for route_number, route_id in enumerate(train_routes):
        env, route_index = divmod(route_id, routes_per_environment)
        if env not in split["environment_ids"]["train"]:
            raise ValueError("non-Train route requested")
        name = f"env{env:02d}_route{route_index:02d}"
        layout, original = source_route(base, source, name)
        path = output / "chunks" / f"{name}.npz"
        if name in manifest["chunks"]:
            if sha256(path) != manifest["chunks"][name]["sha256"]:
                raise ValueError(f"changed completed augmentation chunk: {name}")
            continue
        planner = StoredRoutePlanner(layout)
        rng_anchor = np.random.default_rng([config["seed"], route_id])
        candidates = np.flatnonzero(original["family_id"] == 7)
        anchors = rng_anchor.permutation(candidates)
        saturated_count = np.sum((original["U"][candidates] <= LOWER + .01)
                                 | (original["U"][candidates] >= UPPER - .01), axis=1)
        hard = candidates[np.argsort(-saturated_count, kind="stable")[:2]]
        records, rejected = [], Counter()
        for index in range(32):
            kind = index // 4 if index < 12 else (3 if index < 24 else 4)
            anchor = int(anchors[0] if index < 12 else anchors[min(1, len(anchors)-1)] if index < 24
                         else hard[min((index - 24) // 4, len(hard)-1)])
            phase = float(original["phase_seconds"][anchor])
            refs = planner.reference_window_for_trajectory(planner.trajectory, elapsed=phase, dt=.1, horizon=21)
            rng = np.random.default_rng([config["seed"], route_id, index])
            for attempt in range(config["max_attempts"]):
                state = sample_state(refs[0], rng, kind, original["X"][anchor])
                try:
                    control, diagnostics = teacher.query(state, refs)
                except RuntimeError as exc:
                    rejected[KINDS[kind] + ": " + str(exc).split(":")[0]] += 1
                    if attempt == config["max_attempts"] - 1:
                        atomic_json(output / "failure.json", {"route": name, "index": index,
                            "phase": phase, "kind": KINDS[kind], "error": str(exc)})
                        raise
                    continue
                repeat_delta = 0.
                if index == 0:
                    repeated, _ = teacher.query(state, refs)
                    repeat_delta = float(np.max(np.abs(repeated - control)))
                    if repeat_delta > 2e-5:
                        raise RuntimeError("independent teacher repeatability failed")
                record = {key: values[anchor] for key, values in original.items()}
                record.update(X=build_features(state, refs), U=control,
                    row_id=row_offset + route_number * 32 + index, family_id=7, phase_seconds=phase,
                    actual_speed=float(np.linalg.norm(state.velocity)),
                    reference_speed=float(np.linalg.norm(refs[0].velocity)),
                    position_error=float(np.linalg.norm(state.position - refs[0].position)),
                    repeat_delta=repeat_delta, repeat_checked=index == 0,
                    rejected_before_acceptance=attempt, augmentation_kind=kind,
                    anchor_row_id=int(original["row_id"][anchor]), **diagnostics)
                records.append(record)
                break
        atomic_npz(path, {key: np.asarray([row[key] for row in records]) for key in records[0]})
        manifest["chunks"][name] = {"sha256": sha256(path), "samples": len(records), "rejected": dict(rejected)}
        atomic_json(manifest_path, manifest)
        print(f"added={len(manifest['chunks']) * 32}/{len(train_routes) * 32} route={name} "
              f"rejected={sum(rejected.values())} elapsed_this_run={time.perf_counter()-started:.1f}s", flush=True)
    manifest["complete"] = True
    atomic_json(manifest_path, manifest)
    print("Augmentation complete. Run prepare next.", flush=True)


def prepare(base, output):
    source, split = source_info(base)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = manifest["config"]
    if not manifest["complete"] or config["base_manifest_sha256"] != sha256(base / "manifest.json"):
        raise ValueError("augmentation is incomplete or source changed")
    for name, expected in config["base_file_sha256"].items():
        if sha256(base / "dnn_dataset" / name) != expected:
            raise ValueError(f"source data changed: {name}")
    blocks = []
    for name, metadata in sorted(manifest["chunks"].items()):
        source_route(base, source, name)
        path = output / "chunks" / f"{name}.npz"
        if sha256(path) != metadata["sha256"]:
            raise ValueError(f"changed augmentation chunk: {name}")
        blocks.append(read_npz(path))
    added = {key: np.concatenate([block[key] for block in blocks]) for key in blocks[0]}
    expected = len(split["route_ids"]["train"]) * 32
    if (added["X"].shape != (expected, 300) or added["U"].shape != (expected, 4)
            or not np.array_equal(np.sort(added["row_id"]), np.arange(sum(split["counts"].values()), sum(split["counts"].values()) + expected))
            or not np.isin(added["route_id"], split["route_ids"]["train"]).all()
            or not np.isin(added["environment_id"], split["environment_ids"]["train"]).all()
            or any(not np.isfinite(value).all() for value in added.values())
            or np.max(added["kkt"]) > TEACHER_CONFIG["kkt_tolerance"]
            or np.max(added["action_delta"]) > TEACHER_CONFIG["action_delta_tolerance"]
            or np.max(added["repeat_delta"]) > 2e-5
            or np.any(added["U"] < LOWER - 1e-4) or np.any(added["U"] > UPPER + 1e-4)
            or np.any(added["X"][:, :3] != 0)):
        raise ValueError("invalid augmentation rows, labels or provenance")
    old = read_npz(base / "dnn_dataset/train.npz")
    merged = {key: np.concatenate([values, added[key]]) for key, values in old.items()}
    merged["augmentation_kind"] = np.r_[np.full(len(old["X"]), -1, int), added["augmentation_kind"]]
    merged["anchor_row_id"] = np.r_[old["row_id"], added["anchor_row_id"]]
    dataset = output / "dnn_dataset"
    saved_path = dataset / "split_manifest.json"
    if saved_path.exists() and json.loads(saved_path.read_text())["augmentation_manifest_sha256"] != sha256(manifest_path):
        raise ValueError("prepared augmentation changed; choose a new output")
    atomic_npz(dataset / "train.npz", merged)
    for name in ("validation.npz", "normalization.json"):
        temporary = dataset / (name + ".tmp")
        shutil.copyfile(base / "dnn_dataset" / name, temporary)
        temporary.replace(dataset / name)
    # This is a derived split: lineage remains the old source; augmentation is explicit.
    prepared = {**split, "augmentation_manifest_sha256": sha256(manifest_path),
        "base_data_dir": str((base / "dnn_dataset").resolve()),
        "counts": {**split["counts"], "train": len(merged["X"])},
        "file_sha256": {name: sha256(dataset / name) for name in
                       ("train.npz", "validation.npz", "normalization.json")},
        "test_external_path": str((base / "dnn_dataset/test.npz").resolve()),
        "test_materialized_or_read": False}
    atomic_json(saved_path, prepared)
    saturated = np.any((added["U"] <= LOWER + .01) | (added["U"] >= UPPER - .01), axis=1)
    audit = {"status": "PASS", "old_train_samples": len(old["X"]), "added_train_samples": expected,
        "train_samples": len(merged["X"]), "validation_byte_identical":
            sha256(dataset / "validation.npz") == config["base_file_sha256"]["validation.npz"],
        "normalization_byte_identical": sha256(dataset / "normalization.json") == config["base_file_sha256"]["normalization.json"],
        "test_read": False, "new_route_ids": sorted(np.unique(added["route_id"]).tolist()),
        "max_kkt": float(added["kkt"].max()), "max_repeat_delta": float(added["repeat_delta"].max()),
        "kinds": {name: {"samples": int(np.sum(added["augmentation_kind"] == i)),
                    "saturated_fraction": float(saturated[added["augmentation_kind"] == i].mean())}
                  for i, name in enumerate(KINDS)},
        "note": "Numerical audit only; no promise of smaller-model ordering or closed-loop success."}
    atomic_json(output / "data_audit.json", audit)
    print(json.dumps({key: value for key, value in audit.items() if key != "new_route_ids"}, indent=2), flush=True)
    print(f"Prepared: {dataset}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("generate", "prepare"))
    parser.add_argument("--base", type=Path, default=EXPERIMENT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.stage == "generate":
        generate(args.base, args.output, resume=args.resume)
    else:
        prepare(args.base.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
