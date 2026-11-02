"""Diagnose whether each stored 12-D input contract identifies MPC labels.

This is a read-only dataset diagnostic.  It compares local-neighbour label
consistency for the published gate-relative input, an acceleration ablation,
and the trajectory-relative input actually used when the MPC labels were
generated.  A lower neighbour MSE means that nearby inputs more consistently
map to nearby controls.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors


HERE = Path(__file__).resolve().parent
DEFAULT_DATASET = HERE / "dnn_dataset_paper_gate_12d_euler"


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _standardize(train: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, dtype=np.float64)
    std = np.maximum(train.std(axis=0, dtype=np.float64), 1e-8)
    return (train - mean) / std, (query - mean) / std


def _neighbour_report(
    train_x: np.ndarray,
    train_u: np.ndarray,
    query_x: np.ndarray,
    query_u: np.ndarray,
    *,
    neighbours: int,
) -> dict[str, float | int]:
    train_z, query_z = _standardize(train_x, query_x)
    model = NearestNeighbors(n_neighbors=neighbours, algorithm="auto", n_jobs=-1)
    model.fit(train_z)
    distances, indices = model.kneighbors(query_z, return_distance=True)
    neighbour_u = train_u[indices]
    prediction = neighbour_u.mean(axis=1)
    error = prediction - query_u
    local_mean = neighbour_u.mean(axis=1, keepdims=True)
    local_variance = np.mean((neighbour_u - local_mean) ** 2)
    return {
        "dimensions": int(train_x.shape[1]),
        "neighbours": int(neighbours),
        "physical_mse": float(np.mean(error**2)),
        "physical_rmse": float(np.sqrt(np.mean(error**2))),
        "mean_standardized_neighbour_distance": float(np.mean(distances)),
        "mean_local_label_variance": float(local_variance),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--neighbours", type=int, default=5)
    args = parser.parse_args()

    train = _load(args.dataset / "train.npz")
    query = _load(args.dataset / f"{args.split}.npz")
    if "X_source_trajectory_relative" not in train:
        raise SystemExit("Dataset does not retain X_source_trajectory_relative")

    variants = {
        "paper_gate_12d": (train["X"], query["X"]),
        "paper_gate_without_acceleration_9d": (train["X"][:, :9], query["X"][:, :9]),
        "source_trajectory_12d": (
            train["X_source_trajectory_relative"],
            query["X_source_trajectory_relative"],
        ),
    }
    report = {
        "dataset": str(args.dataset.resolve()),
        "query_split": args.split,
        "train_samples": int(len(train["U"])),
        "query_samples": int(len(query["U"])),
        "interpretation": (
            "This is a non-parametric information-contract diagnostic, not a DNN score. "
            "Lower neighbour MSE and local label variance indicate a more identifiable X-to-U mapping."
        ),
        "variants": {
            name: _neighbour_report(
                train_x,
                train["U"],
                query_x,
                query["U"],
                neighbours=args.neighbours,
            )
            for name, (train_x, query_x) in variants.items()
        },
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
