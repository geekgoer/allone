"""Create deterministic rollout-group-disjoint splits and train-only normalization."""

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
from falcon_paper_fidelity.recovery_dataset import CONTROL_BUCKET_NAMES
from falcon_paper_fidelity.short_rollout_dataset import FAMILY_NAMES
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


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-root", type=Path, default=here / "teacher_short_rollout_10ms_120k")
    parser.add_argument("--output-dir", type=Path, default=here / "dnn_dataset_short_rollout_10ms_120k")
    parser.add_argument("--seed", type=int, default=71025)
    args = parser.parse_args()

    teacher_root = args.teacher_root.resolve()
    manifest = json.loads((teacher_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise RuntimeError("teacher dataset must be complete before splitting")
    with np.load(teacher_root / manifest["final_dataset"], allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    x, u = arrays["X"], arrays["U"]
    if x.ndim != 2 or x.shape[1] != 12 or u.shape != (len(x), 4):
        raise RuntimeError("expected X=(N,12), U=(N,4)")
    groups = np.unique(arrays["query_group_id"])
    if len(groups) < 10:
        raise RuntimeError("at least ten full rollouts are required for an 80/10/10 split")
    shuffled = np.random.default_rng(args.seed).permutation(groups)
    train_end = int(np.floor(0.8 * len(groups)))
    validation_end = train_end + int(np.floor(0.1 * len(groups)))
    group_splits = {
        "train": shuffled[:train_end],
        "validation": shuffled[train_end:validation_end],
        "test": shuffled[validation_end:],
    }
    masks = {name: np.isin(arrays["query_group_id"], values) for name, values in group_splits.items()}
    membership = sum(mask.astype(np.int8) for mask in masks.values())
    if not np.all(membership == 1):
        raise RuntimeError("split membership is not disjoint and complete")
    normalization = normalization_from_train(x[masks["train"]], u[masks["train"]])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stored = [key for key, value in arrays.items() if np.asarray(value).ndim and len(value) == len(x)]
    summaries: dict[str, object] = {}
    for name, mask in masks.items():
        _atomic_npz(args.output_dir / f"{name}.npz", {key: arrays[key][mask] for key in stored})
        summaries[name] = {
            "samples": int(np.count_nonzero(mask)),
            "rollouts": int(len(group_splits[name])),
            "family_counts": {FAMILY_NAMES[index]: int(np.count_nonzero(arrays["sample_family"][mask] == index)) for index in range(len(FAMILY_NAMES))},
            "control_bucket_counts": {CONTROL_BUCKET_NAMES[index]: int(np.count_nonzero(arrays["control_bucket"][mask] == index)) for index in range(len(CONTROL_BUCKET_NAMES))},
        }
    (args.output_dir / "normalization.json").write_text(
        json.dumps({key: value.tolist() for key, value in normalization.items()}, indent=2) + "\n",
        encoding="utf-8",
    )
    result = {
        "status": "valid",
        "source": str((teacher_root / manifest["final_dataset"]).resolve()),
        "teacher_policy": manifest.get("strategy"),
        "input_contract": INPUT_CONTRACT,
        "input_dimensions": 12,
        "input_features": list(INPUT_FEATURES),
        "output_dimensions": 4,
        "output_features": list(OUTPUT_FEATURES),
        "reference_speed_values": [float(manifest["reference_speed_mps"])],
        "split_unit": "query_group_id (one complete five-step rollout)",
        "split_seed": args.seed,
        "split_ratio": {"train": 0.8, "validation": 0.1, "test": 0.1},
        "group_leakage": False,
        "stored_arrays": stored,
        "splits": summaries,
    }
    (args.output_dir / "split_manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
