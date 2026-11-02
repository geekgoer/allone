"""Inventory six existing FALCON checkpoints without modifying them."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .compute_complexity import checkpoint_size, complexity


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "experiments/paper_track_p_12d_5ms_v1/dnn_dataset"
CONFIG = Path(__file__).resolve().parent / "configs/baseline_models.json"
OUTPUT = Path(__file__).resolve().parent / "results/falcon_baseline.csv"


def freeze(output: Path = OUTPUT) -> list[dict]:
    expected_norm = json.loads((DATA / "normalization.json").read_text(encoding="utf-8"))
    entries = json.loads(CONFIG.read_text(encoding="utf-8"))["models"]
    rows = []
    for entry in entries:
        folder = ROOT / entry["run_dir"]
        metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
        model_norm = json.loads((folder / "normalization.json").read_text(encoding="utf-8"))
        for key in ("x_mean", "x_std", "u_mean", "u_std"):
            if not np.allclose(model_norm[key], expected_norm[key], rtol=0, atol=1e-6):
                raise ValueError(f"{entry['model_id']} normalization mismatch: {key}")
        depth, width = int(metrics["depth"]), int(metrics["width"])
        if entry["model_id"] != f"depth{depth}_width{width}":
            raise ValueError(f"model ID mismatch: {entry['model_id']}")
        cost = complexity(12, 4, depth - 1, width)
        if int(metrics["parameters"]) != cost["parameter_count"]:
            raise ValueError(f"parameter count mismatch: {entry['model_id']}")
        checkpoint = folder / "best_model.pt"
        row = {
            "model_id": entry["model_id"], "hidden_layers": depth - 1, "hidden_width": width,
            "normalized_mse": metrics["test_normalized"]["mse"],
            "physical_mse": metrics["test"]["mse"],
            "physical_rmse": metrics["test"]["rmse"],
            **cost, "checkpoint_size_bytes": checkpoint_size(checkpoint),
            "checkpoint_path": str(checkpoint), "source_metrics": str(folder / "metrics.json"),
            "seed": metrics["seed"], "epochs_completed": metrics["epochs_completed"],
        }
        rows.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    rows = freeze(args.output)
    print(f"Frozen {len(rows)} existing checkpoints: {args.output}")


if __name__ == "__main__":
    main()
