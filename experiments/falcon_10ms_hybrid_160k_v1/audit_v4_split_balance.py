"""Reject a v4 split if source/difficulty coverage differs materially by split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SPLITS = ("train", "validation", "test")


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=HERE / "dnn_dataset_hybrid_160k_10ms_v4_environment_balanced")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    datasets = {name: _load(data_dir / f"{name}.npz") for name in SPLITS}
    result: dict[str, object] = {"data_dir": str(data_dir), "splits": {}}
    extension_profiles: list[np.ndarray] = []
    low_speed_profiles: list[float] = []
    for name, data in datasets.items():
        source = data["source_kind"].astype(np.int64)
        bucket = data["control_bucket"].astype(np.int64)
        velocity = np.linalg.norm(data["velocity"], axis=1)
        source_fraction = np.asarray([np.mean(source == kind) for kind in range(4)])
        extension = bucket >= 0
        bucket_fraction = np.asarray([np.mean(bucket[extension] == kind) for kind in range(3)])
        extension_profiles.append(bucket_fraction)
        low_speed_profiles.append(float(np.mean(velocity <= 3.5)))
        result["splits"][name] = {
            "samples": int(len(source)),
            "source_fraction": source_fraction.round(6).tolist(),
            "extension_bucket_fraction": bucket_fraction.round(6).tolist(),
            "low_speed_fraction_v_le_3p5": low_speed_profiles[-1],
            "environment_ids": sorted(np.unique(data["environment_id"]).astype(int).tolist()),
        }
    source_ok = all(np.allclose(info["source_fraction"], [0.25] * 4, atol=1e-8) for info in result["splits"].values())
    bucket_spread = float(np.max(np.ptp(np.stack(extension_profiles, axis=0), axis=0)))
    low_speed_spread = float(max(low_speed_profiles) - min(low_speed_profiles))
    result["checks"] = {
        "source_kind_exactly_balanced": source_ok,
        "maximum_extension_bucket_fraction_spread": bucket_spread,
        "maximum_low_speed_fraction_spread": low_speed_spread,
        "bucket_balance_threshold": 0.02,
        "low_speed_balance_threshold": 0.002,
    }
    result["status"] = "valid" if source_ok and bucket_spread <= 0.02 and low_speed_spread <= 0.002 else "invalid"
    print(json.dumps(result, indent=2))
    if result["status"] != "valid":
        raise SystemExit("v4 split balance audit failed; do not train this data")


if __name__ == "__main__":
    main()
