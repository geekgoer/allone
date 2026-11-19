"""Evaluate a trained paper-style SVR on a dataset and a small candidate grid."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np


def _csv_numbers(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("numeric list must not be empty")
    return values


def _predict(payload: dict, x: np.ndarray) -> np.ndarray:
    scale = np.asarray(payload["feature_maximum_for_c_T_R"], dtype=np.float64)
    transformed = x.astype(np.float64, copy=True)
    transformed[:, :3] /= scale
    return np.asarray(payload["model"].predict(transformed), dtype=np.float64)


def _dataset_rows(dataset: Path) -> tuple[list[dict[str, str]], np.ndarray, np.ndarray]:
    with dataset.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [row for row in reader if row.get("valid_for_svr", "").lower() == "true"]
    if not rows:
        raise ValueError("dataset has no valid_for_svr=true rows")
    x = np.asarray(
        [
            [
                1.0 / float(row["beta_test_normalized_mse"]),
                float(row["period_ms"]),
                float(row["R_max_ms"]),
                float(row["epsilon"]),
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    y = np.asarray([float(row["J_paper_style"]) for row in rows], dtype=np.float64)
    return rows, x, y


def _grid(args: argparse.Namespace) -> np.ndarray:
    values = []
    for c in _csv_numbers(args.c):
        for period in _csv_numbers(args.periods_ms):
            for response in _csv_numbers(args.r_ms):
                for epsilon in _csv_numbers(args.epsilon):
                    values.append([c, period, response, epsilon])
    return np.asarray(values, dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--c", default="1000,3000,5000,10000")
    parser.add_argument("--periods-ms", default="30,60,100")
    parser.add_argument("--r-ms", default="5,10,20,30,40")
    parser.add_argument("--epsilon", default="0,0.8,1.6")
    args = parser.parse_args()

    payload = joblib.load(args.model.resolve())
    rows, x, y = _dataset_rows(args.dataset.resolve())
    prediction = _predict(payload, x)
    candidates = _grid(args)
    candidate_prediction = _predict(payload, candidates)
    output = {
        "model": str(args.model.resolve()),
        "dataset": str(args.dataset.resolve()),
        "records": len(rows),
        "observed": {
            "target_min": float(y.min()),
            "target_max": float(y.max()),
            "prediction_min": float(prediction.min()),
            "prediction_max": float(prediction.max()),
            "negative_prediction_count": int(np.sum(prediction <= 0.0)),
        },
        "candidate_grid": [
            {
                "c": float(item[0]),
                "period_ms": float(item[1]),
                "R_ms": float(item[2]),
                "epsilon": float(item[3]),
                "predicted_J": float(value),
            }
            for item, value in zip(candidates, candidate_prediction)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["observed"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
