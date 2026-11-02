"""Summarize normalized and physical test MSE across architecture seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=here / "runs_pilot")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    rows: list[dict[str, object]] = []
    for metrics_path in sorted(args.runs_root.glob("seed_*/depth*_width*/metrics.json")):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append({
            "seed": int(metrics.get("seed", metrics_path.parents[1].name.split("_")[-1])),
            "depth": int(metrics["depth"]),
            "width": int(metrics["width"]),
            "parameters": int(metrics["parameters"]),
            "best_epoch": int(metrics["best_epoch"]),
            "test_normalized_mse": float(metrics["test_normalized"]["mse"]),
            "test_physical_mse": float(metrics["test"]["mse"]),
            "path": str(metrics_path.parent.resolve()),
        })
    if not rows:
        raise FileNotFoundError(f"no completed metrics found below {args.runs_root}")
    groups: dict[tuple[int, int], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((int(row["depth"]), int(row["width"])), []).append(row)
    architecture_summary = []
    for (depth, width), values in groups.items():
        normalized = [float(item["test_normalized_mse"]) for item in values]
        physical = [float(item["test_physical_mse"]) for item in values]
        architecture_summary.append({
            "depth": depth,
            "width": width,
            "parameters": int(values[0]["parameters"]),
            "seeds": len(values),
            "test_normalized_mse_mean": mean(normalized),
            "test_normalized_mse_std": pstdev(normalized),
            "test_physical_mse_mean": mean(physical),
            "test_physical_mse_std": pstdev(physical),
        })
    architecture_summary.sort(key=lambda item: int(item["parameters"]))
    smallest = architecture_summary[0]
    largest = architecture_summary[-1]
    relative_gain = (
        float(smallest["test_normalized_mse_mean"])
        - float(largest["test_normalized_mse_mean"])
    ) / float(smallest["test_normalized_mse_mean"])
    report = {
        "runs_root": str(args.runs_root.resolve()),
        "metric_primary": "test_normalized_mse",
        "runs": rows,
        "architectures": architecture_summary,
        "smallest_to_largest_normalized_mse_relative_gain": relative_gain,
        "capacity_pilot_pass_10_percent": relative_gain >= 0.10,
        "note": "A passing pilot supports, but cannot guarantee, monotonic behavior for all six architectures.",
    }
    output = args.output or args.runs_root / "capacity_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
