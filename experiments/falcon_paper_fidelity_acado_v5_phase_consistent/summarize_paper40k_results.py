"""Summarize paper40k capacity trends without recovery-data metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=here / "runs_paper40k_pilot")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for path in sorted(args.runs_root.glob("seed_*/depth*_width*/metrics.json")):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "seed": int(metrics["seed"]),
            "depth": int(metrics["depth"]),
            "width": int(metrics["width"]),
            "parameters": int(metrics["parameters"]),
            "best_epoch": int(metrics["best_epoch"]),
            "epochs_completed": int(metrics["epochs_completed"]),
            "test_normalized_mse": float(metrics["test_normalized"]["mse"]),
            "test_physical_mse": float(metrics["test"]["mse"]),
            "path": str(path.parent.resolve()),
        })
    if not rows:
        raise FileNotFoundError(f"no completed runs found below {args.runs_root}")
    rows.sort(key=lambda item: int(item["parameters"]))
    mse = np.asarray([float(item["test_normalized_mse"]) for item in rows])
    parameters = np.asarray([float(item["parameters"]) for item in rows])
    comparisons = []
    lookup = {(int(item["depth"]), int(item["width"])): item for item in rows}
    for width in (512, 1024):
        for low, high in ((10, 22), (22, 34)):
            if (low, width) in lookup and (high, width) in lookup:
                left = float(lookup[(low, width)]["test_normalized_mse"])
                right = float(lookup[(high, width)]["test_normalized_mse"])
                comparisons.append({
                    "comparison": f"depth{high} <= depth{low} at width{width}",
                    "passed": right <= left,
                    "relative_improvement": (left - right) / left,
                })
    for depth in (10, 22, 34):
        if (depth, 512) in lookup and (depth, 1024) in lookup:
            left = float(lookup[(depth, 512)]["test_normalized_mse"])
            right = float(lookup[(depth, 1024)]["test_normalized_mse"])
            comparisons.append({
                "comparison": f"width1024 <= width512 at depth{depth}",
                "passed": right <= left,
                "relative_improvement": (left - right) / left,
            })
    report = {
        "runs_root": str(args.runs_root.resolve()),
        "metric_primary": "test_normalized_mse",
        "runs": rows,
        "smallest_to_largest_relative_gain": float((mse[0] - mse[-1]) / mse[0]) if len(rows) > 1 else None,
        "log_parameters_vs_log_mse_correlation": float(np.corrcoef(np.log(parameters), np.log(mse))[0, 1]) if len(rows) > 1 else None,
        "capacity_comparisons": comparisons,
        "strict_all_comparisons_pass": bool(comparisons) and all(bool(item["passed"]) for item in comparisons),
        "disclosure": "Results use only the 40k paper core with a deterministic row-wise 80/10/10 split.",
    }
    output = args.output or args.runs_root / "capacity_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
