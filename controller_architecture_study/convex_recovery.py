"""Frozen-feature ablation and resumable Train-only recovery augmentation."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .convex_dataset import EXPERIMENT, SEED, sample_problem
from .convex_features import CONTRACT, KEPT, REMOVED, project, encode
from .convex_teacher import ConvexTeacher
from .decision_dataset import ROOT, atomic_json, atomic_npz, sha256
from .fly_convex_policy import verify_dataset
from .fly_decision_dnn import StoredRoutePlanner


OUTPUT = ROOT / "controller_architecture_study/results/decision_convex_recovery_v1"
ARMS = ("features_only", "features_recovery")
# 80 fresh queries per Train route, with an independent 8-query audit stream.
FAMILY_SCHEDULE = (4,)*16 + (5,)*16 + (6,)*8 + (7,)*40
VERSION = "convex_train_recovery_237d_v1"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def source_info(base, smoke=False):
    split = verify_dataset(base)
    expected = {"train": 76800, "validation": 9600, "test": 9600}
    if not smoke and split["counts"] != expected:
        raise ValueError("requires the complete frozen 96k source dataset")
    collection = read_json(base/"manifest.json")
    if not collection["complete"]:
        raise ValueError("incomplete source collection")
    return split, collection


def specification(base, split, collection, smoke):
    paths = (Path(__file__), Path(__file__).with_name("convex_features.py"))
    return {"version": VERSION, "contract": CONTRACT, "smoke": smoke,
        "base": str(base), "source_manifest_sha256": split["source_manifest_sha256"],
        "base_artifact_sha256": split["artifact_sha256"], "environment_ids": split["environment_ids"],
        "routes_per_environment": collection["config"]["routes_per_environment"],
        "added_family_schedule": list(range(4, 8)) if smoke else list(FAMILY_SCHEDULE),
        "probe_family_schedule": list(range(8)), "seed": SEED + 101,
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in paths},
        "teacher": collection["config"]["teacher"]}


def validate_rows(data):
    n = len(data["U"])
    if (data["X"].shape != (n, 237) or data["U"].shape != (n, 4) or n == 0
            or any(not np.isfinite(value).all() for value in data.values())
            or data["constraint_violation"].max() > 2e-4
            or data["stationarity_scaled"].max() > 2e-4 or data["repeat_delta"].max() > 2e-5
            or np.any(data["U"] < [2.-1e-4, -3.-1e-4, -3.-1e-4, -2.-1e-4])
            or np.any(data["U"] > [20.+1e-4, 3.+1e-4, 3.+1e-4, 2.+1e-4])):
        raise ValueError("invalid recovery rows, labels or QP residuals")


def generate(base, output, *, resume=False, smoke=False):
    base, output = base.resolve(), output.resolve()
    if base == output or base in output.parents or output in base.parents:
        raise ValueError("augmentation output must be separate from the original experiment")
    split, collection = source_info(base, smoke)
    config = specification(base, split, collection, smoke)
    teacher = ConvexTeacher()
    if json.loads(json.dumps(teacher.signature())) != config["teacher"]:
        raise ValueError("teacher/runtime differs from frozen source")
    manifest_path = output/"manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if not resume or manifest["config"] != config:
            raise ValueError("use --resume with unchanged settings/code, or a new directory")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("refusing to overwrite an unrecognized output directory")
        manifest = {"config": config, "chunks": {}, "complete": False}
        atomic_json(manifest_path, manifest)
    train_envs = split["environment_ids"]["train"]
    available = collection["config"]["environment_ids"]
    train_envs = [env for env in train_envs if env in available]
    routes = config["routes_per_environment"]
    started = time.perf_counter()
    total = len(train_envs)*routes*2
    for env in train_envs:
        for route in range(routes):
            route_name = f"env{env:02d}_route{route:02d}"
            route_path = base/"routes"/f"{route_name}.json"
            if sha256(route_path) != collection["chunks"][route_name]["route_sha256"]:
                raise ValueError(f"source route changed: {route_name}")
            layout = read_json(route_path)
            planner = StoredRoutePlanner(layout)
            for stream, kind in enumerate(("added", "probe")):
                name = f"{route_name}_{kind}"
                path = output/"chunks"/f"{name}.npz"
                if name in manifest["chunks"]:
                    if sha256(path) != manifest["chunks"][name]["sha256"]:
                        raise ValueError(f"completed chunk changed: {name}")
                    continue
                schedule = config["added_family_schedule" if kind == "added" else "probe_family_schedule"]
                rows = []
                for index, family in enumerate(schedule):
                    rng = np.random.default_rng([config["seed"], env, route, stream, index])
                    state, refs, phase = sample_problem(planner.trajectory, planner, rng, family)
                    try:
                        control, diagnostic = teacher.query(state, refs)
                        repeat = teacher.query(state, refs)[0] if index == 0 else control
                        rows.append({"X": encode(state, refs), "U": control,
                            "row_id": 96000 + (env*20+route)*256 + stream*128 + index,
                            "environment_id": env, "route_id": layout["route_id"], "family_id": family,
                            "phase_seconds": phase, "actual_speed": float(np.linalg.norm(state.velocity)),
                            "reference_speed": float(np.linalg.norm(refs[0].velocity)),
                            "position_error": float(np.linalg.norm(state.position-refs[0].position)),
                            "repeat_delta": float(abs(control-repeat).max()), "repeat_checked": index == 0,
                            **diagnostic})
                    except Exception as exc:
                        atomic_json(output/"failure.json", {"chunk": name, "index": index, "error": str(exc)})
                        raise
                values = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
                validate_rows(values)
                atomic_npz(path, values)
                manifest["chunks"][name] = {"sha256": sha256(path), "kind": kind, "samples": len(rows),
                    "environment_id": env, "route_id": layout["route_id"]}
                atomic_json(manifest_path, manifest)
            completed = len(manifest["chunks"])
            print(f"chunks={completed}/{total} train_env={env} route={route} "
                  f"elapsed_this_run={time.perf_counter()-started:.1f}s", flush=True)
    manifest["complete"] = True
    atomic_json(manifest_path, manifest)
    print(f"Recovery collection complete: {output}", flush=True)


def prepare(base, output):
    manifest = read_json(output/"manifest.json")
    c = manifest["config"]
    split, collection = source_info(base, c["smoke"])
    if not manifest["complete"] or c != specification(base, split, collection, c["smoke"]):
        raise ValueError("incomplete or changed collection")
    blocks = {"added": [], "probe": []}
    envs = set(collection["config"]["environment_ids"]) & set(split["environment_ids"]["train"])
    expected = {f"env{env:02d}_route{route:02d}_{kind}" for env in envs
                for route in range(c["routes_per_environment"]) for kind in blocks}
    if set(manifest["chunks"]) != expected:
        raise ValueError("missing or extra recovery chunks")
    for name in sorted(expected):
        meta, path = manifest["chunks"][name], output/"chunks"/f"{name}.npz"
        if sha256(path) != meta["sha256"]:
            raise ValueError(f"recovery chunk changed: {name}")
        rows = load_npz(path)
        validate_rows(rows)
        if (not np.all(rows["environment_id"] == meta["environment_id"])
                or not np.all(rows["route_id"] == meta["route_id"])
                or not set(rows["environment_id"]) <= envs):
            raise ValueError("non-Train or incorrect recovery route")
        blocks[meta["kind"]].append(rows)
    extra = {kind: {key: np.concatenate([block[key] for block in values]) for key in values[0]}
             for kind, values in blocks.items()}
    all_ids = np.r_[extra["added"]["row_id"], extra["probe"]["row_id"]]
    if len(np.unique(all_ids)) != len(all_ids):
        raise ValueError("overlapping recovery/probe identifiers")
    source = base/"dnn_dataset"
    # Test is transformed losslessly, not evaluated or used to select anything.
    original = {name: load_npz(source/f"{name}.npz") for name in ("train", "validation", "test")}
    if sha256(source/"test.npz") != split["artifact_sha256"]["test.npz"]:
        raise ValueError("source Test file changed")
    norm = read_json(source/"normalization.json")
    norm.update(input_contract=CONTRACT, kept_original_indices=list(KEPT), fit_split="original_train",
                fit_samples=len(original["train"]["X"]),
                normalization_policy="freeze original Train scales for both ablation arms")
    for key in ("x_mean", "x_std"):
        norm[key] = np.asarray(norm[key])[list(KEPT)].tolist()
    reference_hash = None
    for arm in ARMS:
        data_dir = output/"datasets"/arm
        counts = {}
        for name, values in original.items():
            transformed = {key: (project(value) if key == "X" else value) for key, value in values.items()}
            if name == "train":
                # Preserve every original field; new geometry metadata comes from its route.
                augmented = extra["added"]
                if arm == "features_recovery":
                    for key in transformed:
                        if key not in augmented:
                            if key not in ("geometry_family", "heading_yaw"):
                                raise ValueError(f"missing fresh query field: {key}")
                            mapping = {int(route): value for route, value in zip(values["route_id"], values[key])}
                            augmented[key] = np.asarray([mapping[int(route)] for route in augmented["route_id"]])
                    transformed = {key: np.concatenate([value, augmented[key]]) for key, value in transformed.items()}
            counts[name] = len(transformed["X"])
            atomic_npz(data_dir/f"{name}.npz", transformed)
        for kind, values in (("audit", original["train"]), ("probe", extra["probe"])):
            audit_dir = data_dir/kind
            atomic_npz(audit_dir/"train.npz", {key: project(value) if kind == "audit" and key == "X" else value
                                             for key, value in values.items()})
            atomic_json(audit_dir/"normalization.json", norm)
        atomic_json(data_dir/"normalization.json", norm)
        hashes = {name: sha256(data_dir/name) for name in
                  ("train.npz", "validation.npz", "test.npz", "normalization.json", "audit/train.npz", "probe/train.npz",
                   "audit/normalization.json", "probe/normalization.json")}
        fixed = {key: value for key, value in hashes.items() if key != "train.npz"}
        if reference_hash is not None and reference_hash != fixed:
            raise ValueError("fixed evaluation rows/normalization differ between arms")
        reference_hash = fixed
        atomic_json(data_dir/"split_manifest.json", {"version": VERSION, "contract": CONTRACT,
            "arm": arm, "base": str(base), "source_manifest_sha256": split["source_manifest_sha256"],
            "augmentation_manifest_sha256": sha256(output/"manifest.json"),
            "environment_ids": split["environment_ids"], "kept_original_indices": list(KEPT),
            "counts": counts, "artifact_sha256": hashes})
    report = {"status": "PASS", "input_dim": 237, "removed_original_indices": list(REMOVED),
        "original_train_samples": len(original["train"]["X"]), "added_train_samples": len(extra["added"]["X"]),
        "probe_samples": len(extra["probe"]["X"]), "probe_used_for_training": False,
        "added_family_counts": {str(i): int(np.sum(extra["added"]["family_id"] == i)) for i in range(8)},
        "heldout_rows_labels_and_environment_splits_preserved": True,
        "same_fixed_evaluation_and_normalization_across_arms": True,
        "test_transformed_only": True, "test_evaluated": False,
        "note": "A controlled ablation, not a guarantee of lower Validation error or monotonic capacity."}
    atomic_json(output/"data_audit.json", report)
    print(json.dumps(report, indent=2), flush=True)


def verify_arm(data_dir, base, *, full=True):
    source = verify_dataset(base, full=False)
    split = read_json(data_dir/"split_manifest.json")
    output = data_dir.parent.parent
    if (split["version"] != VERSION or split["contract"] != CONTRACT or split["base"] != str(base.resolve())
            or split["source_manifest_sha256"] != source["source_manifest_sha256"]
            or split["environment_ids"] != source["environment_ids"] or split["kept_original_indices"] != list(KEPT)
            or split["augmentation_manifest_sha256"] != sha256(output/"manifest.json")
            or read_json(output/"data_audit.json")["status"] != "PASS"):
        raise ValueError("ablation source/contract/audit mismatch")
    config = read_json(output/"manifest.json")["config"]
    for name, expected in config["source_sha256"].items():
        if sha256(ROOT/name) != expected:
            raise ValueError(f"ablation source code changed: {name}")
    names = ("train.npz", "validation.npz", "normalization.json", "audit/train.npz", "probe/train.npz",
             "audit/normalization.json", "probe/normalization.json") if full else ("normalization.json",)
    for name in names:
        if sha256(data_dir/name) != split["artifact_sha256"][name]:
            raise ValueError(f"ablation artifact changed: {name}")
    return split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("generate", "prepare"))
    parser.add_argument("--base", type=Path, default=EXPERIMENT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    base, output = args.base.resolve(), args.output.resolve()
    if args.smoke_test:
        output = output.with_name(output.name+"_smoke")
    if args.stage == "generate":
        generate(base, output, resume=args.resume, smoke=args.smoke_test)
    else:
        prepare(base, output)


if __name__ == "__main__":
    main()
