"""Merge the existing 160k set with real zero-speed launch rollouts.

The split is performed inside each environment, at episode granularity.
Every environment therefore appears in train, validation, and test; the
split is not an environment holdout.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Mapping

import numpy as np

from falcon_paper_fidelity.contracts import (
    TRAJECTORY_INPUT_CONTRACT,
    TRAJECTORY_INPUT_FEATURES,
)
from prepare_dnn_dataset import normalization_from_train


REQUIRED = (
    "X",
    "U",
    "episode_id",
    "initial_state_id",
    "environment_id",
    "step_id",
)
OPTIONAL = (
    "position",
    "velocity",
    "acceleration",
    "reference_position",
    "reference_speed",
    "source_kind",
    "sample_family",
    "data_source",
)


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


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    missing = sorted(set(REQUIRED).difference(arrays))
    if missing:
        raise ValueError(f"{path} is missing {missing}")
    if arrays["X"].ndim != 2 or arrays["X"].shape[1] != 12:
        raise ValueError(f"{path}: X must have shape (N, 12)")
    if arrays["U"].shape != (len(arrays["X"]), 4):
        raise ValueError(f"{path}: U must have shape (N, 4)")
    for key in REQUIRED:
        if len(arrays[key]) != len(arrays["X"]):
            raise ValueError(f"{path}: {key} has an inconsistent row count")
    return arrays


def _load_existing_dataset(data_dir: Path) -> dict[str, np.ndarray]:
    parts = [_load(data_dir / f"{name}.npz") for name in ("train", "validation", "test")]
    keys = set(parts[0])
    if any(set(part) != keys for part in parts[1:]):
        raise ValueError("existing train/validation/test files do not have identical keys")
    return {key: np.concatenate([part[key] for part in parts], axis=0) for key in sorted(keys)}


def _with_defaults(
    arrays: dict[str, np.ndarray],
    *,
    source_kind: int | None = None,
    sample_family: int | None = None,
) -> dict[str, np.ndarray]:
    rows = len(arrays["X"])
    result = {key: np.asarray(arrays[key]) for key in REQUIRED if key in arrays}
    for key in OPTIONAL:
        if key in arrays:
            result[key] = np.asarray(arrays[key])
    if source_kind is not None:
        result["source_kind"] = np.full(rows, source_kind, dtype=np.int8)
        result["data_source"] = np.full(rows, source_kind, dtype=np.int8)
    if sample_family is not None:
        result["sample_family"] = np.full(rows, sample_family, dtype=np.int8)
    if "reference_speed" not in result:
        result["reference_speed"] = np.full(rows, 10.0, dtype=np.float32)
    return result


def _align_and_concat(
    base: dict[str, np.ndarray],
    launch: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    keys = sorted(set(base).intersection(launch))
    required = set(REQUIRED)
    missing = sorted(required.difference(keys))
    if missing:
        raise ValueError(f"cannot merge; common data are missing {missing}")
    result: dict[str, np.ndarray] = {}
    for key in keys:
        left = np.asarray(base[key])
        right = np.asarray(launch[key])
        if left.ndim != right.ndim or left.shape[1:] != right.shape[1:]:
            raise ValueError(f"shape mismatch for {key}: {left.shape} vs {right.shape}")
        result[key] = np.concatenate((left, right), axis=0)
    return result


def _episode_split(
    arrays: dict[str, np.ndarray],
    *,
    seed: int,
) -> dict[str, np.ndarray]:
    env = np.asarray(arrays["environment_id"], dtype=np.int64)
    episode = np.asarray(arrays["episode_id"], dtype=np.int64)
    groups = np.stack((env, episode), axis=1)
    unique_groups = np.unique(groups, axis=0)
    if unique_groups.shape[0] < 3:
        raise ValueError("not enough episode groups for splitting")

    assignments: dict[str, set[tuple[int, int]]] = {
        "train": set(),
        "validation": set(),
        "test": set(),
    }
    rng = np.random.default_rng(seed)
    for environment_id in np.unique(env):
        local = [
            (int(item[0]), int(item[1]))
            for item in unique_groups
            if int(item[0]) == int(environment_id)
        ]
        if len(local) < 3:
            raise ValueError(
                f"environment {environment_id} has only {len(local)} episode groups"
            )
        local = [local[index] for index in rng.permutation(len(local))]
        train_count = max(1, int(np.floor(len(local) * 0.8)))
        validation_count = max(1, int(np.floor(len(local) * 0.1)))
        if train_count + validation_count >= len(local):
            validation_count = 1
            train_count = len(local) - 2
        assignments["train"].update(local[:train_count])
        assignments["validation"].update(
            local[train_count : train_count + validation_count]
        )
        assignments["test"].update(local[train_count + validation_count :])

    row_groups = [tuple(map(int, item)) for item in groups]
    masks: dict[str, np.ndarray] = {}
    for split, selected in assignments.items():
        masks[split] = np.asarray([item in selected for item in row_groups], dtype=bool)
    if np.any(sum(mask.astype(np.int8) for mask in masks.values()) != 1):
        raise RuntimeError("episode split is not a disjoint partition")
    return masks


def _summary(arrays: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, object]:
    return {
        "samples": int(np.count_nonzero(mask)),
        "episodes": int(np.unique(arrays["episode_id"][mask]).size),
        "initial_states": sorted(
            np.unique(arrays["initial_state_id"][mask]).astype(int).tolist()
        ),
        "environments": sorted(
            np.unique(arrays["environment_id"][mask]).astype(int).tolist()
        ),
        "source_kind_fraction": {
            str(int(value)): float(np.mean(arrays["source_kind"][mask] == value))
            for value in np.unique(arrays["source_kind"][mask])
        },
        "low_speed_error_fraction": float(
            np.mean(np.abs(arrays["X"][mask, 6]) > 3.0)
        ),
    }


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=here / "dnn_dataset_v4_cold_rti_labels_v1",
    )
    parser.add_argument(
        "--launch",
        type=Path,
        default=here
        / "launch_40k_airsim_acado_10ms_v2_measured"
        / "final"
        / "falcon_launch_40k_10ms_v2_measured.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=here / "dnn_dataset_v5_launch_augmented_200k_v1",
    )
    parser.add_argument("--seed", type=int, default=20260919)
    args = parser.parse_args()

    base = _with_defaults(_load_existing_dataset(args.base_dir.resolve()))
    launch = _with_defaults(
        _load(args.launch.resolve()),
        source_kind=4,
        sample_family=-2,
    )

    if not np.allclose(
        np.unique(base["reference_speed"]),
        np.asarray([10.0]),
        atol=1e-3,
    ):
        raise ValueError("base dataset is not the expected 10 m/s dataset")
    if not np.allclose(
        np.unique(launch["reference_speed"]),
        np.asarray([10.0]),
        atol=1e-3,
    ):
        raise ValueError("launch dataset is not a fixed 10 m/s dataset")
    if "velocity" not in launch:
        raise ValueError("launch dataset must include measured velocity for startup validation")
    launch_speed = np.linalg.norm(launch["velocity"], axis=1)
    launch_low_speed_fraction = float(np.mean(launch_speed <= 3.5))
    if launch_low_speed_fraction < 0.05:
        raise ValueError(
            "launch dataset does not contain enough real startup coverage: "
            f"fraction_speed_le_3p5={launch_low_speed_fraction:.3f}"
        )

    base_episode_max = int(np.max(base["episode_id"]))
    launch["episode_id"] = launch["episode_id"].astype(np.int64) + base_episode_max + 1
    combined = _align_and_concat(base, launch)
    if combined["X"].shape != (200000, 12):
        raise ValueError(f"expected exactly 200000 rows, got {combined['X'].shape}")

    masks = _episode_split(combined, seed=args.seed)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    normalization = normalization_from_train(
        combined["X"][masks["train"]],
        combined["U"][masks["train"]],
    )
    for split, mask in masks.items():
        selected = {key: value[mask] for key, value in combined.items()}
        _atomic_npz(output_dir / f"{split}.npz", selected)
    (output_dir / "normalization.json").write_text(
        json.dumps({key: value.tolist() for key, value in normalization.items()}, indent=2)
        + "\n",
        encoding="utf-8",
    )

    summaries = {split: _summary(combined, mask) for split, mask in masks.items()}
    manifest = {
        "status": "valid",
        "strategy": "base_160k_plus_real_zero_speed_launch_rollouts_measured_start_v2",
        "input_contract": TRAJECTORY_INPUT_CONTRACT,
        "input_features": list(TRAJECTORY_INPUT_FEATURES),
        "input_dimensions": 12,
        "output_dimensions": 4,
        "total_samples": int(len(combined["X"])),
        "base_dataset": str(args.base_dir.resolve()),
        "launch_dataset": str(args.launch.resolve()),
        "split_unit": "episode_within_environment",
        "environment_leakage": False,
        "every_environment_in_all_splits": all(
            len(set(summaries[split]["environments"])) == 40
            for split in ("train", "validation", "test")
        ),
        "splits": summaries,
        "normalization_fit": "train_only",
        "launch_source_kind": 4,
        "launch_sample_family": -2,
        "launch_low_speed_fraction_speed_le_3p5": launch_low_speed_fraction,
    }
    (output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
