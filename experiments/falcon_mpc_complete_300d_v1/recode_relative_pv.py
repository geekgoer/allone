"""Create a lossless, better-conditioned 300-D dataset.

The original 300-D contract stores absolute current/reference position and
velocity.  This recoding subtracts the current position and velocity from
every reference node.  It does not change the physical MPC problem or labels;
the transformation is exactly invertible at deployment because the current
state is already available.
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
REFERENCE_OFFSET = 10
REFERENCE_NODES = 21
REFERENCE_STATE_DIM = 10


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


def _recode(x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != INPUT_DIM:
        raise ValueError(f"expected X=(N,{INPUT_DIM}), got {values.shape}")

    result = values.copy()
    state_position = result[:, 0:3].copy()
    state_velocity = result[:, 7:10].copy()
    references = result[:, REFERENCE_OFFSET : REFERENCE_OFFSET + REFERENCE_NODES * REFERENCE_STATE_DIM]
    references = references.reshape(-1, REFERENCE_NODES, REFERENCE_STATE_DIM)
    references[:, :, 0:3] -= state_position[:, None, :]
    references[:, :, 7:10] -= state_velocity[:, None, :]
    result[:, REFERENCE_OFFSET : REFERENCE_OFFSET + REFERENCE_NODES * REFERENCE_STATE_DIM] = (
        references.reshape(-1, REFERENCE_NODES * REFERENCE_STATE_DIM)
    )
    return result.astype(np.float32)


def _normalization(train_x: np.ndarray, train_u: np.ndarray) -> dict[str, Any]:
    x = train_x.astype(np.float64)
    u = train_u.astype(np.float64)
    x_mean = x.mean(axis=0)
    x_std = np.maximum(x.std(axis=0), 1e-6)
    u_mean = u.mean(axis=0)
    u_std = np.maximum(u.std(axis=0), 1e-6)
    return {
        "input_contract": "falcon_mpc_complete_solver_window_300d_relative_pv_v1",
        "output_contract": "falcon_acado_first_control_4d_v1",
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "u_mean": u_mean.tolist(),
        "u_std": u_std.tolist(),
        "fit_split": "train",
        "fit_samples": int(len(train_x)),
        "recoding": {
            "type": "lossless_reference_relative_to_current_state",
            "reference_position": "reference_position - current_position",
            "reference_velocity": "reference_velocity - current_velocity",
            "reference_orientation": "unchanged",
            "reference_controls": "unchanged",
        },
    }


def recode(source: Path, output: Path) -> None:
    source = source.resolve()
    output = output.resolve()
    if source == output:
        raise ValueError("source and output must be different directories")
    if not (source / "train.npz").exists():
        raise FileNotFoundError(source / "train.npz")

    loaded: dict[str, dict[str, np.ndarray]] = {}
    for split in SPLITS:
        with np.load(source / f"{split}.npz", allow_pickle=False) as archive:
            loaded[split] = {key: np.asarray(archive[key]) for key in archive.files}
        if loaded[split]["X"].shape[1:] != (INPUT_DIM,):
            raise ValueError(f"{split}.X is not 300-D: {loaded[split]['X'].shape}")
        if loaded[split]["U"].shape[1:] != (4,):
            raise ValueError(f"{split}.U is not 4-D: {loaded[split]['U'].shape}")

    recoded: dict[str, dict[str, np.ndarray]] = {}
    for split in SPLITS:
        arrays = dict(loaded[split])
        arrays["X"] = _recode(arrays["X"])
        recoded[split] = arrays
        _atomic_npz(output / f"{split}.npz", arrays)

    normalization = _normalization(recoded["train"]["X"], recoded["train"]["U"])
    _atomic_json(output / "normalization.json", normalization)

    manifest_path = source / "split_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {}
    manifest["source_dataset"] = str(source)
    manifest["derived_dataset"] = True
    manifest["input_contract"] = normalization["input_contract"]
    manifest["recoding"] = normalization["recoding"]
    manifest["normalization_fit"] = "train only"
    manifest["source_test_read"] = True
    _atomic_json(output / "split_manifest.json", manifest)

    print(json.dumps({
        "source": str(source),
        "output": str(output),
        "splits": {name: int(len(recoded[name]["X"])) for name in SPLITS},
        "input_shape": list(recoded["train"]["X"].shape),
        "max_abs_finite": bool(all(np.isfinite(recoded[name]["X"]).all() for name in SPLITS)),
        "position_std_before": loaded["train"]["X"][:, 10:13].std(axis=0).tolist(),
        "position_std_after": recoded["train"]["X"][:, 10:13].std(axis=0).tolist(),
        "velocity_std_before": loaded["train"]["X"][:, 17:20].std(axis=0).tolist(),
        "velocity_std_after": recoded["train"]["X"][:, 17:20].std(axis=0).tolist(),
    }, indent=2))


def main() -> None:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=base / "dnn_dataset_300d_140k_v1_within_environment",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=base / "dnn_dataset_300d_140k_v1_relative_pv",
    )
    args = parser.parse_args()
    recode(args.source, args.output)


if __name__ == "__main__":
    main()
