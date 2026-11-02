"""Create leakage-safe train/validation/test splits for MPC imitation learning."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Mapping

import numpy as np

from dnn_model_utils import input_feature_names


DEFAULT_DATASET = Path("teacher_dataset_full_route_continuous_v2/final/teacher_dataset_full_route_continuous_v2.npz")
DEFAULT_OUTPUT = Path("dnn_dataset_full_route_v2")


def environment_split_masks(environment_id: np.ndarray) -> dict[str, np.ndarray]:
    """Use deterministic, disjoint environment blocks for 80/10/10 splits.

    The original 40-environment experiment therefore remains 32/4/4.  The
    robust collection has 60 environments and becomes 48/6/6 without any
    random re-split or sample-level leakage.
    """
    ids = np.asarray(environment_id, dtype=np.int64)
    unique_ids = np.unique(ids)
    if unique_ids.size < 3:
        raise ValueError("at least three environments are required for a split")
    train_count = min(
        unique_ids.size - 2,
        max(1, int(np.floor(unique_ids.size * 0.8))),
    )
    validation_count = max(1, int(np.floor(unique_ids.size * 0.1)))
    if train_count + validation_count >= unique_ids.size:
        validation_count = unique_ids.size - train_count - 1
    train_ids = unique_ids[:train_count]
    validation_ids = unique_ids[train_count:train_count + validation_count]
    test_ids = unique_ids[train_count + validation_count:]
    return {
        "train": np.isin(ids, train_ids),
        "validation": np.isin(ids, validation_ids),
        "test": np.isin(ids, test_ids),
    }


def normalization_from_train(x: np.ndarray, u: np.ndarray) -> dict[str, np.ndarray]:
    x_mean = np.mean(x, axis=0, dtype=np.float64)
    x_std = np.std(x, axis=0, dtype=np.float64)
    u_mean = np.mean(u, axis=0, dtype=np.float64)
    u_std = np.std(u, axis=0, dtype=np.float64)
    return {
        "x_mean": x_mean,
        "x_std": np.maximum(x_std, 1e-8),
        "u_mean": u_mean,
        "u_std": np.maximum(u_std, 1e-8),
    }


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def prepare(
    dataset_path: Path,
    output_dir: Path,
    *,
    environment_split_ids: Mapping[str, list[int]] | None = None,
) -> dict[str, object]:
    with np.load(dataset_path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    required = {"X", "U", "environment_id", "episode_id", "initial_state_id"}
    missing = sorted(required.difference(arrays))
    if missing:
        raise ValueError(f"dataset is missing arrays: {missing}")
    if arrays["X"].ndim != 2 or arrays["X"].shape[1] < 1 or arrays["U"].shape[1:] != (4,):
        raise ValueError("expected X.shape=(N, D) and U.shape=(N, 4)")
    input_dim = int(arrays["X"].shape[1])
    source_manifest_path = dataset_path.resolve().parent.parent / "manifest.json"
    source_manifest: dict[str, object] = {}
    if source_manifest_path.exists():
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        declared_dimensions = source_manifest.get("input_dimensions")
        if declared_dimensions is not None and int(declared_dimensions) != input_dim:
            raise ValueError("source manifest input dimension does not match dataset X")

    environment_values = np.unique(arrays["environment_id"])
    expected = np.arange(environment_values.size, dtype=np.int64)
    if not np.array_equal(environment_values, expected):
        raise ValueError(
            "formal split requires contiguous environment IDs starting at 0; "
            f"found {environment_values.tolist()}"
        )

    if environment_split_ids is None:
        masks = environment_split_masks(arrays["environment_id"])
    else:
        required_splits = {"train", "validation", "test"}
        if set(environment_split_ids) != required_splits:
            raise ValueError("explicit environment split must contain exactly train, validation and test")
        listed = [int(value) for split in ("train", "validation", "test") for value in environment_split_ids[split]]
        if sorted(listed) != environment_values.astype(int).tolist() or len(set(listed)) != len(listed):
            raise ValueError("explicit environment split must partition every environment ID exactly once")
        masks = {
            split: np.isin(arrays["environment_id"], np.asarray(values, dtype=np.int64))
            for split, values in environment_split_ids.items()
        }
    if np.any(sum(mask.astype(np.int8) for mask in masks.values()) != 1):
        raise ValueError("every sample must belong to exactly one split")

    normalization = normalization_from_train(arrays["X"][masks["train"]], arrays["U"][masks["train"]])
    output_dir.mkdir(parents=True, exist_ok=True)
    split_summary: dict[str, object] = {}
    row_count = int(arrays["X"].shape[0])
    keep_keys = tuple(
        key
        for key, value in arrays.items()
        if np.asarray(value).ndim >= 1 and np.asarray(value).shape[0] == row_count
    )
    for name, mask in masks.items():
        selected = {key: arrays[key][mask] for key in keep_keys if key in arrays}
        _atomic_npz(output_dir / f"{name}.npz", selected)
        split_summary[name] = {
            "samples": int(np.count_nonzero(mask)),
            "episodes": int(np.unique(arrays["episode_id"][mask]).size),
            "initial_states": sorted(np.unique(arrays["initial_state_id"][mask]).astype(int).tolist()),
            "environments": sorted(np.unique(arrays["environment_id"][mask]).astype(int).tolist()),
        }

    normalization_json = {key: value.tolist() for key, value in normalization.items()}
    (output_dir / "normalization.json").write_text(
        json.dumps(normalization_json, indent=2) + "\n", encoding="utf-8"
    )
    declared_features = source_manifest.get("input_features")
    feature_names = (
        list(declared_features)
        if isinstance(declared_features, list)
        else list(input_feature_names(input_dim))
    )
    if len(feature_names) != input_dim:
        raise ValueError("source manifest input feature count does not match dataset X")
    manifest = {
        "source": str(dataset_path.resolve()),
        "source_manifest": str(source_manifest_path) if source_manifest else None,
        "teacher_policy": source_manifest.get("strategy"),
        "input_contract": source_manifest.get("input_contract"),
        "split_unit": "environment_id",
        "split_ratio": {"train": 0.8, "validation": 0.1, "test": 0.1},
        "explicit_environment_split": (
            {split: [int(value) for value in values] for split, values in environment_split_ids.items()}
            if environment_split_ids is not None else None
        ),
        "input_dimensions": input_dim,
        "input_features": feature_names,
        "output_dimensions": 4,
        "reference_speed_values": (
            sorted(float(value) for value in np.unique(arrays["reference_speed"]))
            if "reference_speed" in arrays
            else []
        ),
        "stored_arrays": list(keep_keys),
        "optional_arrays": [
            key for key in (
                "path_context", "reference_speed", "data_source", "perturbation",
                "student_control", "shield_active",
            )
            if key in arrays
        ],
        "splits": split_summary,
    }
    (output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = prepare(args.dataset, args.output_dir)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
