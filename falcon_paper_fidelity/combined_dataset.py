"""Build a robust-training split without altering the immutable paper core."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from prepare_dnn_dataset import normalization_from_train

from .contracts import INPUT_CONTRACT, INPUT_FEATURES, OUTPUT_FEATURES


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


def combine(
    paper_splits: str | Path,
    recovery_dataset: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 12025,
) -> dict[str, Any]:
    paper_root = Path(paper_splits).expanduser().resolve()
    recovery_path = Path(recovery_dataset).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    with np.load(recovery_path, allow_pickle=False) as archive:
        recovery = {key: np.asarray(archive[key]) for key in archive.files}
    if recovery["X"].shape[1:] != (12,) or recovery["U"].shape[1:] != (4,):
        raise ValueError("recovery dataset must use the FALCON 12-D/4-D contract")
    count = len(recovery["X"])
    indices = np.arange(count)
    np.random.default_rng(seed).shuffle(indices)
    train_stop = int(np.floor(0.8 * count))
    validation_stop = train_stop + int(np.floor(0.1 * count))
    recovery_indices = {
        "train": indices[:train_stop],
        "validation": indices[train_stop:validation_stop],
        "test": indices[validation_stop:],
    }
    split_summary: dict[str, Any] = {}
    combined: dict[str, dict[str, np.ndarray]] = {}
    for split, selected in recovery_indices.items():
        with np.load(paper_root / f"{split}.npz", allow_pickle=False) as archive:
            paper_x = np.asarray(archive["X"], dtype=np.float32)
            paper_u = np.asarray(archive["U"], dtype=np.float32)
        x = np.concatenate((paper_x, recovery["X"][selected].astype(np.float32)), axis=0)
        u = np.concatenate((paper_u, recovery["U"][selected].astype(np.float32)), axis=0)
        data_source = np.concatenate(
            (np.zeros(len(paper_x), dtype=np.int8), np.ones(len(selected), dtype=np.int8))
        )
        family = np.concatenate(
            (np.full(len(paper_x), -1, dtype=np.int8), recovery["sample_family"][selected].astype(np.int8))
        )
        order = np.arange(len(x))
        np.random.default_rng(seed + {"train": 1, "validation": 2, "test": 3}[split]).shuffle(order)
        combined[split] = {
            "X": x[order],
            "U": u[order],
            "data_source": data_source[order],
            "sample_family": family[order],
        }
        split_summary[split] = {
            "samples": int(len(x)),
            "paper_core_samples": int(len(paper_x)),
            "recovery_samples": int(len(selected)),
        }
    normalization = normalization_from_train(combined["train"]["X"], combined["train"]["U"])
    output.mkdir(parents=True, exist_ok=True)
    for split, arrays in combined.items():
        _atomic_npz(output / f"{split}.npz", arrays)
    (output / "normalization.json").write_text(
        json.dumps({key: value.tolist() for key, value in normalization.items()}, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "dataset_role": "robust_extension_training_not_table_ii_reproduction",
        "paper_splits": str(paper_root),
        "recovery_dataset": str(recovery_path),
        "teacher_policy": "falcon_paper_core_teacher_plus_independent_recovery_queries_v1",
        "input_contract": INPUT_CONTRACT,
        "input_dimensions": 12,
        "input_features": list(INPUT_FEATURES),
        "output_dimensions": 4,
        "output_features": list(OUTPUT_FEATURES),
        "split_seed": int(seed),
        "reference_speed_values": [5.0],
        "splits": split_summary,
    }
    (output / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest
