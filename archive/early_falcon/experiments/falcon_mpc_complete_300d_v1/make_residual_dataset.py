"""Create a residual-control version of the 300-D dataset.

The first four values of the 80-D reference-control tail are the planner's
stage-0 feed-forward control.  This script changes only the target:

    residual = MPC_first_control - planner_reference_control_stage_0

The input and split membership remain unchanged.  Physical residual error is
exactly the same as reconstructed control error because the same baseline is
added to prediction and target at every row.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


SPLITS = ("train", "validation", "test")
INPUT_DIM = 300
BASELINE_START = 220
BASELINE_DIM = 4


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _normalization(train_x: np.ndarray, train_u: np.ndarray) -> dict[str, Any]:
    x = train_x.astype(np.float64)
    u = train_u.astype(np.float64)
    return {
        "input_contract": "falcon_mpc_complete_solver_window_300d_relative_pv_v1",
        "output_contract": "falcon_mpc_residual_control_4d_v1",
        "x_mean": x.mean(axis=0).tolist(),
        "x_std": np.maximum(x.std(axis=0), 1e-6).tolist(),
        "u_mean": u.mean(axis=0).tolist(),
        "u_std": np.maximum(u.std(axis=0), 1e-6).tolist(),
        "fit_split": "train",
        "fit_samples": int(len(train_x)),
        "target_definition": "MPC_first_control - reference_control_stage_0",
        "reference_control_slice": [BASELINE_START, BASELINE_START + BASELINE_DIM],
    }


def make_dataset(source: Path, output: Path) -> None:
    source = source.resolve()
    output = output.resolve()
    if source == output:
        raise ValueError("source and output must be different directories")

    loaded: dict[str, dict[str, np.ndarray]] = {}
    for split in SPLITS:
        with np.load(source / f"{split}.npz", allow_pickle=False) as archive:
            arrays = {key: np.asarray(archive[key]) for key in archive.files}
        if arrays["X"].shape[1:] != (INPUT_DIM,):
            raise ValueError(f"{split}.X is not 300-D: {arrays['X'].shape}")
        if arrays["U"].shape[1:] != (BASELINE_DIM,):
            raise ValueError(f"{split}.U is not 4-D: {arrays['U'].shape}")
        loaded[split] = arrays

    transformed: dict[str, dict[str, np.ndarray]] = {}
    for split in SPLITS:
        arrays = dict(loaded[split])
        baseline = arrays["X"][:, BASELINE_START:BASELINE_START + BASELINE_DIM].astype(np.float32)
        original = arrays["U"].astype(np.float32)
        arrays["U_original"] = original.copy()
        arrays["U_reference_baseline"] = baseline.copy()
        arrays["U"] = original - baseline
        transformed[split] = arrays
        _atomic_npz(output / f"{split}.npz", arrays)

    _atomic_json(output / "normalization.json", _normalization(
        transformed["train"]["X"], transformed["train"]["U"]))

    manifest_path = source / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest["source_dataset"] = str(source)
    manifest["derived_dataset"] = True
    manifest["output_contract"] = "falcon_mpc_residual_control_4d_v1"
    manifest["target_definition"] = "MPC_first_control - reference_control_stage_0"
    manifest["reference_control_slice"] = [BASELINE_START, BASELINE_START + BASELINE_DIM]
    manifest["normalization_fit"] = "train only"
    _atomic_json(output / "split_manifest.json", manifest)

    train = transformed["train"]
    print(json.dumps({
        "source": str(source),
        "output": str(output),
        "samples": {name: int(len(transformed[name]["X"])) for name in SPLITS},
        "input_shape": list(train["X"].shape),
        "residual_mean": train["U"].mean(axis=0).tolist(),
        "residual_std": train["U"].std(axis=0).tolist(),
        "original_control_std": train["U_original"].std(axis=0).tolist(),
        "finite": bool(all(np.isfinite(transformed[name]["U"]).all() for name in SPLITS)),
    }, indent=2))


def main() -> None:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=base / "dnn_dataset_300d_140k_v1_relative_pv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=base / "dnn_dataset_300d_140k_v1_relative_pv_residual",
    )
    args = parser.parse_args()
    make_dataset(args.source, args.output)


if __name__ == "__main__":
    main()
