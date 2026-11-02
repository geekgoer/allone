"""Summarize the completed 237D sweep without loading models or Test samples."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .decision_dataset import atomic_json, sha256
from .run_convex_scenic_falcon18 import OUTPUT


def main():
    source = OUTPUT/"comparison.json"
    report = json.loads(source.read_text(encoding="utf-8"))
    rows = report["models"]
    if len(rows) != 18 or any(row["status"] != "complete" or row["epochs"] != 600 for row in rows):
        raise ValueError("requires eighteen completed 600-epoch models")
    output = OUTPUT/"analysis_20261003"
    output.mkdir(exist_ok=True)
    by_shape = {(row["hidden_layers"], row["width"]): row for row in rows}
    histories = {}
    for row in rows:
        run = Path(row["run"])
        done = json.loads((run/"completion.json").read_text(encoding="utf-8"))
        metrics = json.loads((run/"offline_metrics.json").read_text(encoding="utf-8"))
        if (done["epochs_completed"] != 600 or done["best_epoch"] != row["best_epoch"]
                or metrics["validation"]["normalized_mse"] != row["validation_mse"]
                or not (run/"best_model.pt").is_file() or not (run/"last_model.pt").is_file()):
            raise ValueError(f"completion artifacts disagree: {run}")
        with (run/"train_log.csv").open(newline="", encoding="utf-8") as handle:
            histories[row["model"]] = list(csv.DictReader(handle))
        family = row["validation"]["by_family_id"]["7"]
        row["combined_recovery_error_fraction"] = (family["samples"]*family["normalized_mse"] /
                                                     (row["validation"]["samples"]*row["validation_mse"]))
    frontier = [row["model"] for row in rows if not any(
        other["parameter_count"] <= row["parameter_count"] and other["validation_mse"] <= row["validation_mse"]
        and (other["parameter_count"] < row["parameter_count"] or other["validation_mse"] < row["validation_mse"])
        for other in rows)]
    best = min(rows, key=lambda row: row["validation_mse"])
    total = sum(row["parameter_count"] for row in rows)
    ranked = sorted(rows, key=lambda row: row["validation_mse"])
    fields = ("model", "architecture_group", "hidden_layers", "width", "parameter_count", "mac_count",
              "epochs", "best_epoch", "train_mse", "fixed_train_mse", "fresh_train_mse", "validation_mse",
              "validation_mixture_ratio", "validation_fixed_ratio", "validation_fresh_ratio",
              "combined_recovery_error_fraction")
    with (output/"all18_ranked.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ranked)
    summary = {"source": str(source), "source_sha256": sha256(source), "complete": len(rows),
        "test_evaluated": False, "seed_count": 1, "best_model": best["model"],
        "best_validation_mse": best["validation_mse"], "parameter_validation_frontier": frontier,
        "frontier_is_not_measured_latency": True,
        "h4_width256_to1024_validation_reduction_percent": 100*(1-by_shape[4, 1024]["validation_mse"]/by_shape[4, 256]["validation_mse"]),
        "h3_width128_to1024_validation_reduction_percent": 100*(1-by_shape[3, 1024]["validation_mse"]/by_shape[3, 128]["validation_mse"]),
        "h6_to_h33_w1024_validation_increase_percent": 100*(by_shape[33, 1024]["validation_mse"]/by_shape[6, 1024]["validation_mse"]-1),
        "h4_to_h33_w1024_train_increase_percent": 100*(by_shape[33, 1024]["train_mse"]/by_shape[4, 1024]["train_mse"]-1),
        "total_parameters": total, "raw_weights_fp32_mib": total*4/2**20, "raw_weights_int8_mib": total/2**20,
        "best_combined_recovery_error_fraction": best["combined_recovery_error_fraction"],
        "validation_fresh_ratio_range": [min(row["validation_fresh_ratio"] for row in rows), max(row["validation_fresh_ratio"] for row in rows)],
        "last100_improvement_percent_range": [min(row["convergence"]["best_validation_improvement_percent"] for row in rows),
                                               max(row["convergence"]["best_validation_improvement_percent"] for row in rows)],
        "logs_sha256": {row["model"]: sha256(Path(row["run"])/"train_log.csv") for row in rows}}
    atomic_json(output/"summary.json", summary)
    colors = ("#0072B2", "#D55E00", "#009E73", "#6B4C9A")
    with plt.rc_context({"font.size": 10, "svg.fonttype": "none", "pdf.fonttype": 42}):
        fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
        ax = axes[0, 0]
        for hidden, marker, color in zip((3, 4), ("o", "s"), colors):
            selected = sorted([r for r in rows if r["hidden_layers"] == hidden], key=lambda r: r["width"])
            ax.plot([r["width"] for r in selected], [r["validation_mse"] for r in selected], marker=marker,
                    color=color, label=f"{hidden} hidden layers")
        ax.set(xlabel="Width", ylabel="Validation normalized MSE (log10 scale)", yscale="log",
               title="A. Wider models at fixed depth")
        ax.set_xticks([128, 256, 512, 1024]); ax.legend()
        ax = axes[0, 1]
        for width, marker, color in zip((512, 1024), ("o", "s"), colors):
            selected = sorted([r for r in rows if r["width"] == width], key=lambda r: r["hidden_layers"])
            ax.plot([r["hidden_layers"] for r in selected], [r["validation_mse"] for r in selected],
                    marker=marker, color=color, label=f"width {width}")
        ax.set(xlabel="Hidden FC layers", ylabel="Validation normalized MSE (log10 scale)", yscale="log",
               title="B. Depth has an interior optimum")
        ax.legend()
        ax = axes[1, 0]
        for group, marker, color in zip(("SCENIC", "FALCON"), ("o", "^"), colors):
            selected = [r for r in rows if r["architecture_group"] == group]
            ax.scatter([r["parameter_count"] for r in selected], [r["validation_mse"] for r in selected],
                       marker=marker, color=color, label=group, s=45)
        pareto = sorted([r for r in rows if r["model"] in frontier], key=lambda r: r["parameter_count"])
        ax.plot([r["parameter_count"] for r in pareto], [r["validation_mse"] for r in pareto],
                "--", color="#444444", label="Observed Pareto frontier", linewidth=1)
        ax.set(xlabel="Parameter count (log10 scale; not latency)", ylabel="Validation normalized MSE (log10 scale)",
               xscale="log", yscale="log", title="C. All 18 models, seed 0")
        ax.legend(fontsize=8)
        ax = axes[1, 1]
        selected = [by_shape[h, w] for h, w in ((4, 256), (4, 512), (4, 1024), (6, 1024), (21, 1024), (33, 1024))]
        for key, label, marker, color in zip(("train_mse", "fixed_train_mse", "fresh_train_mse", "validation_mse"),
                ("Recovery-weighted Train", "Original Train subset", "Fresh Train-route states", "Held-out Validation"),
                ("o", "s", "D", "^"), colors):
            ax.plot(range(len(selected)), [r[key] for r in selected], marker=marker, color=color, label=label)
        ax.set_xticks(range(len(selected)), [f"{r['hidden_layers']}x{r['width']}" for r in selected])
        ax.set(xlabel="Hidden layers x width", ylabel="Normalized MSE (log10 scale)", yscale="log",
               title="D. Memorization gap versus fresh-state error")
        ax.legend(fontsize=8)
        for ax in axes.flat:
            ax.grid(True, alpha=.2)
        fig.suptitle("Frozen 237D recovery dataset | 600 epochs each | one seed per architecture", fontsize=13)
        fig.savefig(output/"capacity_and_gap.png", dpi=180, facecolor="white")
        fig.savefig(output/"capacity_and_gap.svg", facecolor="white")
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
        for shape, color in zip(((4, 512), (6, 1024), (21, 1024), (33, 1024)), colors):
            row = by_shape[shape]
            log = histories[row["model"]]
            epoch = np.array([int(r["epoch"]) for r in log])
            best_values = np.array([float(r["best_validation_normalized_mse"]) for r in log])
            for ax, mask in ((axes[0], epoch >= 1), (axes[1], epoch >= 400)):
                ax.plot(epoch[mask], best_values[mask], color=color, label=f"{shape[0]}x{shape[1]}")
                ax.set(xlabel="Epoch", ylabel="Best-so-far Validation normalized MSE", yscale="log")
                ax.grid(True, alpha=.2)
        axes[0].set_title("Full training: cumulative minimum, not raw epoch loss")
        axes[1].set_title("Late training: improvement continues")
        axes[1].legend()
        fig.savefig(output/"convergence.png", dpi=180, facecolor="white")
        plt.close(fig)
    atomic_json(output/"figure_manifest.json", {"source_report_sha256": summary["source_sha256"],
        "matplotlib_version": matplotlib.__version__, "numpy_version": np.__version__, "seed_count": 1,
        "uncertainty": "No confidence intervals: one trained seed per architecture.",
        "transformations": "Positive MSE plotted on log10 axes. No smoothing. Convergence uses logged cumulative minima.",
        "exclusions": "None in panels A-C except explicit matched-depth/width subsets. Panel D lists six representative shapes.",
        "frontier": "Observed parameter-count/Validation frontier; not measured latency or proof of dominance in closed loop.",
        "destination": "General research discussion; no journal-specific formatting claim.",
        "alt_text": "At fixed depth widths improve Validation. Width1024 reaches its lowest error at six hidden layers. "
                    "Deeper plain models deteriorate. Fresh-state and Validation errors remain closer than training errors."})
    lines = ["# 18模型结果表（237D、seed=0、600 epoch）", "", "所有指标为冻结输出尺度上的normalized MSE。按Validation排名；Test未评估。", "",
             "|排名|隐藏层×宽度|组别|参数量|混合Train|原Train|新状态探针|Validation|Val/新状态|", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for rank, row in enumerate(ranked, 1):
        lines.append(f"|{rank}|{row['hidden_layers']}×{row['width']}|{row['architecture_group']}|{row['parameter_count']:,}|"
                     f"{row['train_mse']:.6e}|{row['fixed_train_mse']:.6e}|{row['fresh_train_mse']:.6e}|"
                     f"{row['validation_mse']:.6e}|{row['validation_fresh_ratio']:.3f}|")
    lines += ["", "![复杂度与gap](capacity_and_gap.png)", "", "![收敛](convergence.png)", "",
              "图中连线仅帮助观察所列离散结构，不表示中间未训练结构的测量。所有模型为单seed，未画虚构误差条。"]
    (output/"RESULTS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "logs_sha256"}, indent=2))
    print(f"Analysis: {output}")


if __name__ == "__main__":
    main()
