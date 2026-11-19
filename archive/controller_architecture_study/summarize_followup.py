"""Aggregate completed follow-up runs without treating missing seeds as failures."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from .run_followup import DEFAULT_CONFIG, DEFAULT_RESULTS, build_jobs


HERE = Path(__file__).resolve().parent
COARSE = HERE / "results"


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(config_path: Path = DEFAULT_CONFIG, results_dir: Path = DEFAULT_RESULTS) -> tuple[list[dict], list[dict]]:
    followup = json.loads(config_path.read_text(encoding="utf-8"))
    protocol = json.loads((config_path.parent / followup["protocol"]).read_text(encoding="utf-8"))
    jobs = build_jobs(followup, protocol, "all")
    reference = {("plain_mlp", layers, width) for layers, width in ((2, 128), (4, 128), (8, 128), (10, 512))}
    existing = [(f"plain_mlp_h{h}_w{w}_seed0", {"model_type": "plain_mlp", "hidden_layers": h, "hidden_width": w, "seed": 0}) for _, h, w in sorted(reference)]
    rows = []
    pending = []
    runs = [(name, config, COARSE / name) for name, config in existing]
    runs += [(name, config, results_dir / name) for name, config in jobs]
    for name, config, folder in runs:
        metrics_path = folder / "offline_metrics.json"
        if not metrics_path.exists():
            pending.append(name)
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        cost = json.loads((folder / "complexity.json").read_text(encoding="utf-8"))
        with (folder / "train_log.csv").open(encoding="utf-8", newline="") as handle:
            history = list(csv.DictReader(handle))
        best = min(history, key=lambda item: float(item["validation_normalized_mse"]))
        rows.append({"model_id": name, "model_type": config["model_type"], "hidden_layers": config["hidden_layers"],
                     "hidden_width": config["hidden_width"], "seed": config["seed"], "best_epoch": int(best["epoch"]),
                     "epochs_completed": int(history[-1]["epoch"]), "train_normalized_mse_at_best": float(best["train_normalized_mse"]),
                     "validation_normalized_mse": metrics["validation"]["normalized_mse"],
                     "test_normalized_mse": metrics["test"]["normalized_mse"], "test_physical_mse": metrics["test"]["physical_mse"],
                     "test_p95_abs_error": metrics["test"]["p95_error"], "test_p99_abs_error": metrics["test"]["p99_error"],
                     "parameter_count": cost["parameter_count"], "mac_count": cost["mac_count"],
                     "mean_grad_norm_before_clip_at_best": best.get("mean_grad_norm_before_clip", ""),
                     "clip_fraction_at_best": best.get("clip_fraction", ""), "checkpoint_path": str(folder / "best_model.pt")})
    fields = list(rows[0]) if rows else ["model_id", "model_type", "hidden_layers", "hidden_width", "seed", "best_epoch",
                                          "epochs_completed", "train_normalized_mse_at_best", "validation_normalized_mse",
                                          "test_normalized_mse", "test_physical_mse", "test_p95_abs_error", "test_p99_abs_error",
                                          "parameter_count", "mac_count", "mean_grad_norm_before_clip_at_best", "clip_fraction_at_best", "checkpoint_path"]
    _write_csv(results_dir / "per_run.csv", rows, fields)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["model_type"], row["hidden_layers"], row["hidden_width"])].append(row)
    aggregates = []
    for (model_type, layers, width), group in sorted(groups.items()):
        vals = [float(row["test_physical_mse"]) for row in group]
        norm = [float(row["test_normalized_mse"]) for row in group]
        aggregates.append({"model_type": model_type, "hidden_layers": layers, "hidden_width": width,
                           "seeds_completed": ",".join(str(row["seed"]) for row in sorted(group, key=lambda r: r["seed"])),
                           "n": len(group), "test_physical_mse_mean": statistics.mean(vals),
                           "test_physical_mse_std": statistics.stdev(vals) if len(vals) > 1 else "",
                           "test_normalized_mse_mean": statistics.mean(norm),
                           "test_normalized_mse_std": statistics.stdev(norm) if len(norm) > 1 else "",
                           "parameter_count": group[0]["parameter_count"], "mac_count": group[0]["mac_count"]})
    aggregate_fields = ["model_type", "hidden_layers", "hidden_width", "seeds_completed", "n", "test_physical_mse_mean",
                        "test_physical_mse_std", "test_normalized_mse_mean", "test_normalized_mse_std", "parameter_count", "mac_count"]
    _write_csv(results_dir / "aggregate.csv", aggregates, aggregate_fields)
    report = ["# M1 Follow-up: seed stability and residual ablation", "", f"Completed {len(rows)} of {len(existing) + len(jobs)} expected measured runs (including four original seed-0 models).", ""]
    if pending:
        report += ["Pending: " + ", ".join(pending), ""]
    report += ["Plain and residual models use identical FC dimensions and MAC counts. Residual models additionally use elementwise adds and fixed branch scaling; see protocol. One-seed results are not a robustness estimate.", ""]
    shapes = sorted({(row["hidden_layers"], row["hidden_width"]) for row in rows})
    for layers, width in shapes:
        matched = {(row["model_type"], row["seed"]): row for row in rows
                   if (row["hidden_layers"], row["hidden_width"]) == (layers, width)}
        seeds = sorted({seed for _, seed in matched})
        pairs = [(seed, matched[("plain_mlp", seed)], matched[("residual_mlp", seed)]) for seed in seeds
                 if ("plain_mlp", seed) in matched and ("residual_mlp", seed) in matched]
        if pairs:
            report += [f"## Matched {layers}×{width} seed comparison", "", "| Seed | Plain physical MSE | Residual physical MSE | Residual / plain |", "|---:|---:|---:|---:|"]
            for seed, plain, residual in pairs:
                ratio = residual["test_physical_mse"] / plain["test_physical_mse"]
                report.append(f"| {seed} | {plain['test_physical_mse']:.6g} | {residual['test_physical_mse']:.6g} | {ratio:.3f} |")
            report += ["", "This is an offline result, not evidence of closed-loop safety.", ""]
    diagnostic = results_dir / "plain_mlp_h10_w512_seed0_lr3e-5_finetune" / "offline_metrics.json"
    original = next((row for row in rows if row["model_id"] == "plain_mlp_h10_w512_seed0"), None)
    if diagnostic.exists() and original:
        tuned = json.loads(diagnostic.read_text(encoding="utf-8"))
        report += ["## 10×512 lower-LR continuation diagnostic", "",
                   f"Original test physical MSE: {original['test_physical_mse']:.6g}; continued: {tuned['test']['physical_mse']:.6g}.",
                   f"Original validation normalized MSE: {original['validation_normalized_mse']:.6g}; continued: {tuned['validation']['normalized_mse']:.6g}.",
                   "This run restarts Adam from the seed-0 best checkpoint at a lower learning rate. It is not an independent seed or a same-budget architecture comparison.", ""]
    report += ["## Interpretation boundary", "", "Check train error alongside validation/test error: high train error indicates optimization difficulty, while low train error with a large validation gap suggests generalization problems. Do not infer a universal optimal architecture from these seeds alone.", ""]
    (results_dir / "analysis.md").write_text("\n".join(report), encoding="utf-8")
    return rows, aggregates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    rows, groups = summarize(args.config, args.results_dir)
    print(f"Recorded {len(rows)} runs in {len(groups)} architecture groups: {args.results_dir}")


if __name__ == "__main__":
    main()
