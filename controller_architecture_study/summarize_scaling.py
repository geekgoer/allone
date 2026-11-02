"""Summarize measured models only; never infer a SuperNet backbone in advance."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def _scatter_svg(rows: list[dict], x_key: str, title: str, path: Path) -> None:
    xs = [math.log10(float(row[x_key])) for row in rows]
    ys = [math.log10(float(row["physical_mse"])) for row in rows]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    xspan, yspan = max(xmax - xmin, 0.1), max(ymax - ymin, 0.1)
    labels = []
    for row, x, y in zip(rows, xs, ys):
        px = 85 + 850 * (x - xmin) / xspan
        py = 480 - 380 * (y - ymin) / yspan
        color = "#cc5a26" if row["source"] == "FALCON baseline" else "#1267ad"
        labels.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="{color}"/><text x="{px+8:.1f}" y="{py-7:.1f}" font-size="11">{row["model_id"]}</text>')
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="550"><rect width="1120" height="550" fill="white"/><text x="85" y="35" font-size="18">{title}</text><path d="M85 100 V480 H935" fill="none" stroke="black"/><text x="85" y="520">{x_key} (log10), range {10**xmin:.3g}–{10**xmax:.3g}</text><text x="85" y="75">physical MSE (log10), range {10**ymin:.3g}–{10**ymax:.3g}</text>{"".join(labels)}</svg>\n'
    path.write_text(svg, encoding="utf-8")


def _depth_width_svg(rows: list[dict], path: Path) -> None:
    widths = sorted({int(row["hidden_width"]) for row in rows})
    layers = sorted({int(row["hidden_layers"]) for row in rows})
    ys = [math.log10(float(row["physical_mse"])) for row in rows]
    lo, hi = min(ys), max(ys)
    span = max(hi - lo, 0.1)
    marks = []
    for row in rows:
        x = 80 + 850 * (int(row["hidden_layers"]) - layers[0]) / max(layers[-1] - layers[0], 1)
        y = 470 - 350 * (math.log10(float(row["physical_mse"])) - lo) / span
        color = ["#3177b0", "#2b9b76", "#a672c3", "#cf7447", "#777777"][widths.index(int(row["hidden_width"])) % 5]
        shape = f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}"/>' if row["source"] != "FALCON baseline" else f'<rect x="{x-5:.1f}" y="{y-5:.1f}" width="10" height="10" fill="{color}"/>'
        marks.append(shape + f'<text x="{x+9:.1f}" y="{y-7:.1f}" font-size="10">{row["hidden_layers"]}×{row["hidden_width"]}</text>')
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="540"><rect width="1120" height="540" fill="white"/><text x="80" y="35" font-size="18">Depth / width vs physical MSE (circles=new, squares=baseline)</text><path d="M80 120 V470 H930" fill="none" stroke="black"/><text x="80" y="515">hidden layers: {layers[0]}–{layers[-1]}; width encoded by color</text><text x="80" y="85">physical MSE log10: {10**lo:.3g}–{10**hi:.3g}; widths={widths}</text>{"".join(marks)}</svg>\n', encoding="utf-8")


def summarize(results_dir: Path = HERE / "results") -> list[dict]:
    baseline = results_dir / "falcon_baseline.csv"
    if not baseline.exists():
        raise FileNotFoundError("freeze the six baselines first")
    rows = []
    for item in csv.DictReader(baseline.open(encoding="utf-8", newline="")):
        checkpoint = Path(item["checkpoint_path"])
        predictions = checkpoint.parent / "test_predictions.npz"
        with np.load(predictions, allow_pickle=False) as archive:
            absolute = np.abs(archive["error"])
            mse = float(np.mean(archive["error"] ** 2))
        if not math.isclose(mse, float(item["physical_mse"]), rel_tol=1e-3):
            raise ValueError(f"baseline prediction mismatch: {item['model_id']}")
        rows.append({"source": "FALCON baseline", "model_id": item["model_id"], "hidden_layers": item["hidden_layers"],
                     "hidden_width": item["hidden_width"], "parameter_count": item["parameter_count"], "mac_count": item["mac_count"],
                     "normalized_mse": item["normalized_mse"], "physical_mse": item["physical_mse"], "physical_rmse": item["physical_rmse"],
                     "p95_error": float(np.percentile(absolute, 95)), "p99_error": float(np.percentile(absolute, 99)),
                     "checkpoint_size_bytes": item["checkpoint_size_bytes"], "checkpoint_path": item["checkpoint_path"]})
    complete = []
    for metrics_path in sorted(results_dir.glob("plain_mlp_h*_w*_seed0/offline_metrics.json")):
        folder = metrics_path.parent
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))["test"]
        cost = json.loads((folder / "complexity.json").read_text(encoding="utf-8"))
        rows.append({"source": "M1 seed0", "model_id": folder.name, "hidden_layers": config["hidden_layers"],
                     "hidden_width": config["hidden_width"], "parameter_count": cost["parameter_count"], "mac_count": cost["mac_count"],
                     "normalized_mse": metrics["normalized_mse"], "physical_mse": metrics["physical_mse"], "physical_rmse": metrics["physical_rmse"],
                     "p95_error": metrics["p95_error"], "p99_error": metrics["p99_error"],
                     "checkpoint_size_bytes": cost["checkpoint_size_bytes"], "checkpoint_path": str(folder / "best_model.pt")})
        complete.append(folder.name)
    summary = results_dir / "architecture_scaling_summary.csv"
    with summary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _scatter_svg(rows, "mac_count", "MAC vs physical MSE", results_dir / "mac_vs_physical_mse.svg")
    _scatter_svg(rows, "parameter_count", "Parameters vs physical MSE", results_dir / "parameters_vs_physical_mse.svg")
    _depth_width_svg(rows, results_dir / "depth_width_scaling.svg")
    report = HERE / "docs/m1_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    expected = {f"plain_mlp_h{h}_w{w}_seed0" for h, w in ((2,64),(2,128),(4,64),(4,128),(4,256),(6,128),(6,256),(8,128),(8,256),(10,256),(10,512))}
    pending = sorted(expected - set(complete))
    status = "11/11 completed" if not pending else f"{len(complete)}/11 completed; pending: {', '.join(pending)}"
    report.write_text("# M1 — Controller Complexity Characterization\n\n"
                      "## 1. Motivation\n\nMeasure the accuracy–complexity tradeoff before designing a compact elastic controller.\n\n"
                      "## 2. Immutable FALCON baseline\n\nSix existing checkpoints and reported held-out test metrics are inventoried in `../results/falcon_baseline.csv`; none were retrained.\n\n"
                      "## 3. Experiment protocol\n\nSee `experiment_protocol.md`. New models share the same dataset, split, normalization and normalized-MSE objective.\n\n"
                      f"## 4. Tested architectures\n\n{status}. New models use seed 0.\n\n"
                      "## 5. Offline metrics\n\nSee `../results/architecture_scaling_summary.csv` and each completed run's `offline_metrics.json`.\n\n"
                      "## 6. Complexity metrics\n\nMACs and parameters count FC operations at batch size one, excluding bias additions; see the summary CSV.\n\n"
                      "## 7. MAC–quality Pareto\n\nSee `../results/mac_vs_physical_mse.svg` and `../results/parameters_vs_physical_mse.svg`. Baselines used different seeds and training budgets, so the plot is descriptive, not a controlled architecture-only comparison.\n\n"
                      "## 8. Candidate saturation region\n\nPending full 11-model sweep and examination of measured marginal gains; no absolute threshold or predetermined winner.\n\n"
                      "## 9. Candidate models for closed-loop evaluation\n\nPending measured sweep. Select a small, knee-point, slightly-overprovisioned and FALCON reference (3–5 models) only after reviewing the complete curves.\n\n"
                      "## 10. Limitations\n\nOne seed per new architecture, offline imitation errors do not establish closed-loop safety, and FALCON baseline training recipes vary. No hardware latency is inferred from MACs.\n\n"
                      "## 11. Next step\n\nComplete the coarse sweep, review the Pareto curves, then choose candidates for AirSim closed-loop validation. Do not select the SuperNet backbone yet.\n", encoding="utf-8")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()
    rows = summarize(args.results_dir)
    print(f"Summarized {len(rows)} measured models in {args.results_dir}")


if __name__ == "__main__":
    main()
