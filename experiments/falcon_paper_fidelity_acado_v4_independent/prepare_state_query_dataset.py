"""Create deterministic group-disjoint train/validation/test state-query splits."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from falcon_paper_fidelity.contracts import INPUT_CONTRACT, INPUT_FEATURES, OUTPUT_FEATURES
from falcon_paper_fidelity.recovery_dataset import FAMILY_NAMES
from falcon_paper_fidelity.recovery_dataset import CONTROL_BUCKET_NAMES
from prepare_dnn_dataset import normalization_from_train


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


def _split_groups(group_ids: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    unique = np.unique(group_ids)
    if unique.size < 10:
        raise ValueError("at least 10 query groups are required for an 80/10/10 split")
    shuffled = np.random.default_rng(seed).permutation(unique)
    train_end = int(np.floor(0.8 * unique.size))
    validation_end = train_end + int(np.floor(0.1 * unique.size))
    selected = {
        "train": shuffled[:train_end],
        "validation": shuffled[train_end:validation_end],
        "test": shuffled[validation_end:],
    }
    return {name: np.isin(group_ids, ids) for name, ids in selected.items()}


def prepare(
    teacher_root: Path,
    output_dir: Path,
    seed: int,
    max_samples: int | None = None,
) -> dict[str, object]:
    teacher_root = teacher_root.resolve()
    manifest_path = teacher_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    final_path = teacher_root / str(manifest["final_dataset"])
    with np.load(final_path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    required = {"X", "U", "sample_id", "query_group_id", "sample_family"}
    missing = sorted(required.difference(arrays))
    if missing:
        raise ValueError(f"teacher dataset is missing arrays: {missing}")
    if arrays["X"].shape[1:] != (12,) or arrays["U"].shape[1:] != (4,):
        raise ValueError("expected X=(N,12) and U=(N,4)")
    source_row_count = int(arrays["X"].shape[0])
    if max_samples is not None:
        if max_samples < 1:
            raise ValueError("max_samples must be positive")
        selected_count = min(int(max_samples), source_row_count)
        order = np.argsort(arrays["sample_id"], kind="stable")[:selected_count]
        arrays = {
            key: value[order]
            if np.asarray(value).ndim >= 1 and np.asarray(value).shape[0] == source_row_count
            else value
            for key, value in arrays.items()
        }
    row_count = int(arrays["X"].shape[0])
    if arrays["U"].shape[0] != row_count:
        raise ValueError("X/U row count mismatch")
    masks = _split_groups(arrays["query_group_id"], seed)
    membership = sum(mask.astype(np.int8) for mask in masks.values())
    if not np.all(membership == 1):
        raise RuntimeError("each row must belong to exactly one split")
    normalization = normalization_from_train(
        arrays["X"][masks["train"]], arrays["U"][masks["train"]]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    keep_keys = [
        key for key, value in arrays.items()
        if np.asarray(value).ndim >= 1 and np.asarray(value).shape[0] == row_count
    ]
    summaries: dict[str, object] = {}
    group_sets: dict[str, set[int]] = {}
    for name, mask in masks.items():
        _atomic_npz(output_dir / f"{name}.npz", {key: arrays[key][mask] for key in keep_keys})
        groups = set(np.unique(arrays["query_group_id"][mask]).astype(int).tolist())
        group_sets[name] = groups
        family_counts = {
            FAMILY_NAMES[index]: int(np.count_nonzero(arrays["sample_family"][mask] == index))
            for index in range(len(FAMILY_NAMES))
        }
        control_bucket_counts = (
            {
                CONTROL_BUCKET_NAMES[index]: int(
                    np.count_nonzero(arrays["control_bucket"][mask] == index)
                )
                for index in range(len(CONTROL_BUCKET_NAMES))
            }
            if "control_bucket" in arrays
            else {}
        )
        summaries[name] = {
            "samples": int(np.count_nonzero(mask)),
            "query_groups": len(groups),
            "family_counts": family_counts,
            "control_bucket_counts": control_bucket_counts,
        }
    if any(
        group_sets[left].intersection(group_sets[right])
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise RuntimeError("query group leakage detected")
    normalization_payload = {key: value.tolist() for key, value in normalization.items()}
    (output_dir / "normalization.json").write_text(
        json.dumps(normalization_payload, indent=2) + "\n", encoding="utf-8"
    )
    result = {
        "source": str(final_path),
        "source_samples": source_row_count,
        "selected_samples": row_count,
        "selection": (
            "lowest deterministic sample_id values"
            if max_samples is not None and row_count < source_row_count
            else "all source samples"
        ),
        "teacher_manifest": str(manifest_path),
        "teacher_policy": manifest.get("strategy"),
        "mpc_backend": manifest.get("mpc_backend"),
        "input_contract": INPUT_CONTRACT,
        "input_dimensions": 12,
        "input_features": list(INPUT_FEATURES),
        "output_dimensions": 4,
        "output_features": list(OUTPUT_FEATURES),
        "reference_speed_values": sorted(
            float(value) for value in np.unique(arrays["reference_speed"])
        ),
        "split_unit": "query_group_id",
        "split_seed": seed,
        "split_ratio": {"train": 0.8, "validation": 0.1, "test": 0.1},
        "group_leakage": False,
        "stored_arrays": keep_keys,
        "splits": summaries,
    }
    (output_dir / "split_manifest.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--teacher-root", type=Path, default=here / "teacher_state_queries_480k")
    parser.add_argument("--output-dir", type=Path, default=here / "dnn_dataset")
    parser.add_argument("--seed", type=int, default=32025)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="deterministically keep at most this many lowest sample IDs before splitting",
    )
    args = parser.parse_args()
    prepare(args.teacher_root, args.output_dir, args.seed, args.max_samples)


if __name__ == "__main__":
    main()
