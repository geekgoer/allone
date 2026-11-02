"""Deterministic 80/10/10 split for the 40,000-sample paper core."""

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


def prepare_episode_split(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 2025,
    require_paper_core: bool = True,
) -> dict[str, Any]:
    """Split whole episodes, never adjacent rows from one trajectory.

    FALCON publishes an 80/10/10 random split but does not state whether rows
    or rollouts were shuffled.  Whole-episode randomization is the conservative
    interpretation: it preserves the published ratios while preventing direct
    temporal-neighbour leakage.
    """

    source = Path(dataset_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    with np.load(source, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    required = {"X", "U", "episode_id", "initial_state_id", "environment_id"}
    missing = sorted(required.difference(arrays))
    if missing:
        raise ValueError(f"dataset is missing arrays: {missing}")
    if arrays["X"].ndim != 2 or arrays["X"].shape[1] != 12:
        raise ValueError("paper model requires X.shape=(N, 12)")
    if arrays["U"].shape != (len(arrays["X"]), 4):
        raise ValueError("paper model requires U.shape=(N, 4)")
    episode_ids = np.unique(arrays["episode_id"].astype(np.int64))
    if require_paper_core and (len(arrays["X"]) != 40000 or episode_ids.size != 400):
        raise ValueError(
            "paper core must contain exactly 40,000 rows in 400 episodes; "
            "use --allow-noncore only for a separately named robustness extension"
        )
    if episode_ids.size < 10:
        raise ValueError("at least ten episodes are required")
    rng = np.random.default_rng(seed)
    shuffled = episode_ids.copy()
    rng.shuffle(shuffled)
    train_count = int(np.floor(0.8 * len(shuffled)))
    validation_count = int(np.floor(0.1 * len(shuffled)))
    split_ids = {
        "train": np.sort(shuffled[:train_count]),
        "validation": np.sort(shuffled[train_count : train_count + validation_count]),
        "test": np.sort(shuffled[train_count + validation_count :]),
    }
    masks = {name: np.isin(arrays["episode_id"], ids) for name, ids in split_ids.items()}
    membership = sum(mask.astype(np.int8) for mask in masks.values())
    if not np.all(membership == 1):
        raise RuntimeError("split masks are not disjoint and exhaustive")
    normalization = normalization_from_train(arrays["X"][masks["train"]], arrays["U"][masks["train"]])
    output.mkdir(parents=True, exist_ok=True)
    rows = len(arrays["X"])
    aligned_keys = [key for key, value in arrays.items() if np.asarray(value).ndim and len(value) == rows]
    summaries: dict[str, Any] = {}
    for name, mask in masks.items():
        _atomic_npz(output / f"{name}.npz", {key: arrays[key][mask] for key in aligned_keys})
        summaries[name] = {
            "samples": int(mask.sum()),
            "episodes": int(len(split_ids[name])),
            "episode_ids": split_ids[name].astype(int).tolist(),
            "initial_state_ids": sorted(np.unique(arrays["initial_state_id"][mask]).astype(int).tolist()),
            "environment_ids": sorted(np.unique(arrays["environment_id"][mask]).astype(int).tolist()),
        }
    (output / "normalization.json").write_text(
        json.dumps({key: value.tolist() for key, value in normalization.items()}, indent=2) + "\n",
        encoding="utf-8",
    )
    source_manifest_path = source.parent.parent / "manifest.json"
    source_manifest = (
        json.loads(source_manifest_path.read_text(encoding="utf-8"))
        if source_manifest_path.exists()
        else {}
    )
    manifest = {
        "source": str(source),
        "source_manifest": str(source_manifest_path) if source_manifest else None,
        "teacher_policy": source_manifest.get("strategy"),
        "input_contract": INPUT_CONTRACT,
        "input_dimensions": 12,
        "input_features": list(INPUT_FEATURES),
        "output_dimensions": 4,
        "output_features": list(OUTPUT_FEATURES),
        "split_ratio": {"train": 0.8, "validation": 0.1, "test": 0.1},
        "split_unit": "whole_episode",
        "split_seed": int(seed),
        "paper_ambiguity": "paper states random 80/10/10 but does not publish the split unit or seed",
        "temporal_neighbor_leakage": False,
        "reference_speed_values": (
            sorted(float(item) for item in np.unique(arrays["reference_speed"]))
            if "reference_speed" in arrays
            else []
        ),
        "stored_arrays": aligned_keys,
        "splits": summaries,
    }
    (output / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest
