"""Summarize capacity trends and core/recovery test errors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev

import numpy as np


def _subset_mse(predictions: Path) -> dict[str, float | int]:
    with np.load(predictions, allow_pickle=False) as archive:
        error = np.asarray(archive["error"], dtype=np.float64)
        source = np.asarray(archive["data_source"], dtype=np.int8)
    result: dict[str, float | int] = {}
    for value, name in ((0, "paper_core"), (1, "phase_recovery")):
        mask = source == value
        result[f"{name}_samples"] = int(np.count_nonzero(mask))
        result[f"{name}_physical_mse"] = float(np.mean(error[mask] ** 2)) if np.any(mask) else float("nan")
    return result


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=here / "runs_pilot")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    rows: list[dict[str, object]] = []
    for metrics_path in sorted(args.runs_root.glob("seed_*/depth*_width*/metrics.json")):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        row: dict[str, object] = {
            "seed": int(metrics["seed"]),
            "depth": int(metrics["depth"]),
            "width": int(metrics["width"]),
            "parameters": int(metrics["parameters"]),
            "best_epoch": int(metrics["best_epoch"]),
            "test_normalized_mse": float(metrics["test_normalized"]["mse"]),
            "test_physical_mse": float(metrics["test"]["mse"]),
            "path": str(metrics_path.parent.resolve()),
        }
        row.update(_subset_mse(metrics_path.parent / "test_predictions.npz"))
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"no completed runs found below {args.runs_root}")
    grouped: dict[tuple[int, int], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((int(row["depth"]), int(row["width"])), []).append(row)
    architectures: list[dict[str, object]] = []
    for (depth, width), values in grouped.items():
        normalized = [float(item["test_normalized_mse"]) for item in values]
        physical = [float(item["test_physical_mse"]) for item in values]
        core = [float(item["paper_core_physical_mse"]) for item in values]
        recovery = [float(item["phase_recovery_physical_mse"]) for item in values]
        architectures.append(
            {
                "depth": depth,
                "width": width,
                "parameters": int(values[0]["parameters"]),
                "seeds": len(values),
                "test_normalized_mse_mean": mean(normalized),
                "test_normalized_mse_std": pstdev(normalized),
                "test_physical_mse_mean": mean(physical),
                "paper_core_physical_mse_mean": mean(core),
                "phase_recovery_physical_mse_mean": mean(recovery),
            }
        )
    architectures.sort(key=lambda item: int(item["parameters"]))
    lookup = {(int(item["depth"]), int(item["width"])): item for item in architectures}
    comparisons = []
    for width in (512, 1024):
        for smaller, larger in ((10, 22), (22, 34)):
            if (smaller, width) in lookup and (larger, width) in lookup:
                left = float(lookup[(smaller, width)]["test_normalized_mse_mean"])
                right = float(lookup[(larger, width)]["test_normalized_mse_mean"])
                comparisons.append({
                    "comparison": f"depth{larger} <= depth{smaller} at width{width}",
                    "passed": right <= left,
                    "relative_improvement": (left - right) / left,
                })
    for depth in (10, 22, 34):
        if (depth, 512) in lookup and (depth, 1024) in lookup:
            left = float(lookup[(depth, 512)]["test_normalized_mse_mean"])
            right = float(lookup[(depth, 1024)]["test_normalized_mse_mean"])
            comparisons.append({
                "comparison": f"width1024 <= width512 at depth{depth}",
                "passed": right <= left,
                "relative_improvement": (left - right) / left,
            })
    smallest = architectures[0]
    largest = architectures[-1]
    smallest_mse = float(smallest["test_normalized_mse_mean"])
    largest_mse = float(largest["test_normalized_mse_mean"])
    mse_values = np.asarray(
        [float(item["test_normalized_mse_mean"]) for item in architectures]
    )
    parameter_values = np.asarray(
        [float(item["parameters"]) for item in architectures]
    )
    log_capacity_correlation = (
        float(np.corrcoef(np.log(parameter_values), np.log(mse_values))[0, 1])
        if len(architectures) > 1
        else float("nan")
    )
    report = {
        "runs_root": str(args.runs_root.resolve()),
        "metric_primary": "test_normalized_mse",
        "runs": rows,
        "architectures": architectures,
        "capacity_comparisons": comparisons,
        "capacity_comparisons_passed": int(sum(bool(item["passed"]) for item in comparisons)),
        "capacity_comparisons_total": len(comparisons),
        "smallest_to_largest_relative_gain": (smallest_mse - largest_mse) / smallest_mse,
        "max_to_min_test_normalized_mse_ratio": float(np.max(mse_values) / np.min(mse_values)),
        "log_parameters_vs_log_mse_correlation": log_capacity_correlation,
        "strict_all_comparisons_pass": bool(comparisons) and all(bool(item["passed"]) for item in comparisons),
        "interpretation": (
            "Trends are measured on untouched test data. No checkpoint is selected from test results; "
            "non-monotonic outcomes remain visible rather than being edited away."
        ),
    }
    output = args.output or args.runs_root / "capacity_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
