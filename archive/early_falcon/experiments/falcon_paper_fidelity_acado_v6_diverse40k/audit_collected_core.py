"""Validate diversity and control saturation in a completed v6 40k core."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_CORE = (
    HERE
    / "paper_core_40k_falcon12_v6_diverse"
    / "final"
    / "falcon_paper_core_40k_falcon12_v6_diverse.npz"
)


def _stats(values: np.ndarray) -> dict[str, object]:
    return {
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, default=DEFAULT_CORE)
    parser.add_argument("--report", type=Path, default=HERE / "core_diversity_audit.json")
    args = parser.parse_args()
    with np.load(args.core.resolve(), allow_pickle=False) as archive:
        data = {key: np.asarray(archive[key]) for key in archive.files}
    if data["X"].shape != (40000, 12) or data["U"].shape != (40000, 4):
        raise ValueError(f"unexpected shapes X={data['X'].shape}, U={data['U'].shape}")
    if not np.isfinite(data["X"]).all() or not np.isfinite(data["U"]).all():
        raise ValueError("dataset contains NaN or Inf")

    u = data["U"].astype(np.float64)
    lower = np.array([2.0, -3.0, -3.0, -2.0])
    upper = np.array([20.0, 3.0, 3.0, 2.0])
    saturated_by_output = np.mean(
        (u <= lower[None, :] + 1e-3) | (u >= upper[None, :] - 1e-3), axis=0
    )
    saturated_any = float(
        np.mean(np.any((u <= lower[None, :] + 1e-3) | (u >= upper[None, :] - 1e-3), axis=1))
    )
    payload = {
        "status": "PASS",
        "core": str(args.core.resolve()),
        "samples": int(len(u)),
        "episodes": int(np.unique(data["episode_id"]).size),
        "initial_states": int(np.unique(data["initial_state_id"]).size),
        "environments": int(np.unique(data["environment_id"]).size),
        "X_internal_trajectory_relative": _stats(data["X"].astype(np.float64)),
        "U_physical": _stats(u),
        "control_saturation_fraction_by_output": saturated_by_output.tolist(),
        "control_saturation_fraction_any": saturated_any,
        "note": (
            "The collector stores trajectory-relative internal X. The paper-facing "
            "gate-relative Euler 12-D X is constructed by prepare_paper40k_dataset.py."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
