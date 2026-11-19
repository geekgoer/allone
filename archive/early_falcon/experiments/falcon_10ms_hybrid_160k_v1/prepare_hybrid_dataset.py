"""Merge the immutable 40k core with 120k extensions and create safe splits."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_store import atomic_write_json
from falcon_paper_fidelity.contracts import (
    TRAJECTORY_INPUT_CONTRACT,
    TRAJECTORY_INPUT_FEATURES,
)
from prepare_dnn_dataset import prepare


CORE_SOURCE = 0
REQUIRED = (
    "X", "U", "episode_id", "initial_state_id", "environment_id", "step_id",
    "target_gate_id", "position", "quaternion", "velocity", "acceleration",
    "reference_position", "reference_orientation", "reference_speed", "mpc_solve_time",
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
    if arrays["X"].ndim != 2 or arrays["X"].shape[1] != 12 or arrays["U"].shape[1:] != (4,):
        raise ValueError(f"{path} does not contain X=(N,12), U=(N,4)")
    return arrays


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, default=here / "core_40k_airsim_acado_10ms_v2_safe_start" / "final" / "falcon_hybrid_core_40k_10ms_v2_safe_start.npz")
    parser.add_argument("--extensions", type=Path, default=here / "extensions_120k_phase_consistent_10ms_v4_environment_balanced" / "final" / "falcon_hybrid_extensions_120k_10ms_v4_environment_balanced.npz")
    parser.add_argument("--combined-root", type=Path, default=here / "hybrid_teacher_160k_10ms_v4_environment_balanced")
    parser.add_argument("--output-dir", type=Path, default=here / "dnn_dataset_hybrid_160k_10ms_v4_environment_balanced")
    parser.add_argument("--split-plan", type=Path, default=here / "environment_split_v4.json")
    args = parser.parse_args()

    core = _load(args.core.resolve())
    extension = _load(args.extensions.resolve())
    if len(core["X"]) != 40000 or len(extension["X"]) != 120000:
        raise ValueError(f"expected core/extensions sizes 40000/120000, got {len(core['X'])}/{len(extension['X'])}")
    core_kind = np.full(len(core["X"]), CORE_SOURCE, dtype=np.int8)
    extension_kind = extension.get("source_kind")
    if extension_kind is None:
        raise ValueError("extension source_kind is required")
    if not np.array_equal(np.unique(core["environment_id"]), np.arange(40, dtype=np.int64)):
        raise ValueError("core must contain environment IDs 0..39")
    if not np.array_equal(np.unique(extension["environment_id"]), np.arange(40, dtype=np.int64)):
        raise ValueError("extensions must inherit environment IDs 0..39")

    combined: dict[str, np.ndarray] = {
        key: np.concatenate((core[key], extension[key]), axis=0)
        for key in REQUIRED
    }
    combined["source_kind"] = np.concatenate((core_kind, extension_kind.astype(np.int8)), axis=0)
    combined["control_bucket"] = np.concatenate((
        np.full(len(core["X"]), -1, dtype=np.int8),
        np.asarray(extension["control_bucket"], dtype=np.int8),
    ))
    combined["sample_family"] = np.concatenate((
        np.full(len(core["X"]), -1, dtype=np.int8),
        np.asarray(extension["sample_family"], dtype=np.int8),
    ))
    combined["query_group_id"] = np.concatenate((
        np.asarray(core["episode_id"], dtype=np.int64) + 90_000_000,
        np.asarray(extension["query_group_id"], dtype=np.int64),
    ))
    combined["rollout_step"] = np.concatenate((
        np.asarray(core["step_id"], dtype=np.int8),
        np.asarray(extension.get("rollout_step", np.zeros(len(extension["X"]), dtype=np.int8)), dtype=np.int8),
    ))
    combined["data_source"] = combined["source_kind"].copy()
    if combined["X"].shape != (160000, 12) or combined["U"].shape != (160000, 4):
        raise RuntimeError("combined dataset has an unexpected shape")

    root = args.combined_root.resolve()
    final_path = root / "final" / "falcon_hybrid_teacher_160k_10ms_v4_environment_balanced.npz"
    _atomic_npz(final_path, combined)
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "strategy": "hybrid_10ms_core_plus_phase_consistent_acado_extensions_v1",
        "paper_relationship": "contains immutable paper-style 40k continuous core plus 120k disclosed robustness extension",
        "input_contract": TRAJECTORY_INPUT_CONTRACT,
        "input_dimensions": 12,
        "input_features": list(TRAJECTORY_INPUT_FEATURES),
        "output_dimensions": 4,
        "reference_speed_values": [10.0],
        "core_source": str(args.core.resolve()),
        "extension_source": str(args.extensions.resolve()),
        "final_dataset": str(Path("final") / final_path.name),
        "X_shape": list(combined["X"].shape),
        "U_shape": list(combined["U"].shape),
        "source_kind_counts": {str(kind): int(np.count_nonzero(combined["source_kind"] == kind)) for kind in range(4)},
        "split_policy": "environment_id; environments 0..31 train, 32..35 validation, 36..39 test; no episode/rollout environment leakage",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(root / "manifest.json", manifest)
    split_plan = json.loads(args.split_plan.resolve().read_text(encoding="utf-8"))
    result = prepare(
        final_path, args.output_dir.resolve(),
        environment_split_ids=split_plan["splits"],
    )
    result["hybrid_source_kind_counts"] = manifest["source_kind_counts"]
    (args.output_dir.resolve() / "split_manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"combined": str(final_path), "samples": len(combined["X"]), "dataset": str(args.output_dir.resolve()), "splits": result["splits"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
