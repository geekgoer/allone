"""Audit the hybrid teacher data before committing GPU time to training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _summary(values: np.ndarray) -> dict[str, float]:
    return {name: float(np.percentile(values, q)) for name, q in (("min", 0), ("p01", 1), ("p50", 50), ("p99", 99), ("max", 100))}


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=here / "hybrid_teacher_160k_10ms_v4_environment_balanced" / "final" / "falcon_hybrid_teacher_160k_10ms_v4_environment_balanced.npz")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    with np.load(args.dataset.resolve(), allow_pickle=False) as archive:
        data = {key: np.asarray(archive[key]) for key in archive.files}
    x, u = data["X"], data["U"]
    if x.shape != (160000, 12) or u.shape != (160000, 4):
        raise RuntimeError(f"expected X=(160000,12), U=(160000,4); got {x.shape}, {u.shape}")
    if not np.isfinite(x).all() or not np.isfinite(u).all():
        raise RuntimeError("dataset contains NaN or Inf")
    speed = np.linalg.norm(data["velocity"], axis=1)
    reference_distance = np.linalg.norm(x[:, 0:3], axis=1)
    source_kind = data["source_kind"]
    result = {
        "status": "valid",
        "dataset": str(args.dataset.resolve()),
        "samples": int(len(x)),
        "X_shape": list(x.shape),
        "U_shape": list(u.shape),
        "source_kind_counts": {str(kind): int(np.count_nonzero(source_kind == kind)) for kind in range(4)},
        "speed_mps": _summary(speed),
        "reference_position_error_m": _summary(reference_distance),
        "low_speed_fraction_v_le_3p5": float(np.mean(speed <= 3.5)),
        "near_reference_fraction_error_le_5": float(np.mean(reference_distance <= 5.0)),
        "saturated_fraction": float(np.mean(data.get("control_bucket", np.zeros(len(x), dtype=np.int8)) == 2)),
        "environment_ids": np.unique(data["environment_id"]).astype(int).tolist(),
        "reference_speed_values": np.unique(data["reference_speed"]).astype(float).tolist(),
    }
    text = json.dumps(result, indent=2)
    print(text)
    if not args.json and (result["low_speed_fraction_v_le_3p5"] <= 0.0 or result["near_reference_fraction_error_le_5"] <= 0.0):
        raise RuntimeError("hybrid data failed essential launch or near-reference coverage")


if __name__ == "__main__":
    main()
