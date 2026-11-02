"""Summarize SCENIC Table III models against the existing controller results."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


HERE = Path(__file__).resolve().parent
CONFIG = HERE / "configs/scenic_table3_v1.json"
RESULTS = HERE / "results/scenic_table3_v1"
OUT = RESULTS / "comparison"


def metrics(folder: Path) -> dict:
    data = json.loads((folder / "offline_metrics.json").read_text(encoding="utf-8"))
    cost = json.loads((folder / "complexity.json").read_text(encoding="utf-8"))
    return {
        "test_normalized_mse": float(data["test"]["normalized_mse"]),
        "test_physical_mse": float(data["test"]["physical_mse"]),
        "test_rmse": float(data["test"]["physical_rmse"]),
        "p95_error": float(data["test"]["p95_error"]),
        "parameters": int(cost["parameter_count"]),
        "macs": int(cost["mac_count"]),
        "samples": int(data["test"]["samples"]),
    }


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    existing = {tuple(item) for item in config["already_trained_in_previous_coarse_sweep"]}
    rows = []
    for item in config["architectures"]:
        layers, width = int(item["hidden_layers"]), int(item["hidden_width"])
        name = f"plain_mlp_h{layers}_w{width}_seed{config['seed']}"
        if (layers, width) in existing:
            folder = HERE / "results" / name
            source = "reused_previous_coarse_sweep"
        else:
            folder = RESULTS / name
            source = "new_scenic_table3_training"
        if not (folder / "offline_metrics.json").exists():
            raise FileNotFoundError(f"Incomplete model: {folder}")
        row = {"table_index": int(item["index"]), "paper_L": int(item["paper_L"]),
               "hidden_layers": layers, "hidden_width": width, "model_id": name,
               "paper_test_loss": float(item["paper_test_loss"]), "source": source}
        row.update(metrics(folder))
        rows.append(row)
    rows.sort(key=lambda row: row["table_index"])
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "scenic_table3_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    ranked = sorted(rows, key=lambda row: row["test_normalized_mse"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout="constrained")
    labels = [f"{r['table_index']}: {r['hidden_layers']}x{r['hidden_width']}" for r in ranked]
    axes[0].barh(range(len(ranked)), [r["test_normalized_mse"] for r in ranked],
                 color=["#0072B2" if r["source"].startswith("reused") else "#D55E00" for r in ranked])
    axes[0].set_xscale("log")
    axes[0].set_yticks(range(len(ranked)), labels, fontsize=8)
    axes[0].set_xlabel("Our test normalized MSE")
    axes[0].set_title("SCENIC Table III models ranked by our test MSE")
    axes[0].legend(handles=[
        Patch(facecolor="#0072B2", label="Reused previous coarse sweep"),
        Patch(facecolor="#D55E00", label="New SCENIC Table III training"),
    ], loc="lower right", fontsize=8, frameon=True)
    axes[1].scatter([r["paper_test_loss"] for r in rows], [r["test_normalized_mse"] for r in rows],
                    c=["#0072B2" if r["source"].startswith("reused") else "#D55E00" for r in rows], s=60)
    for r in rows:
        axes[1].annotate(str(r["table_index"]), (r["paper_test_loss"], r["test_normalized_mse"]),
                         xytext=(4, 4), textcoords="offset points", fontsize=8)
    axes[1].set(xscale="log", yscale="log", xlabel="SCENIC paper Table III test loss",
                ylabel="Our test normalized MSE", title="Paper value versus reproduction value")
    for ax in axes:
        ax.grid(alpha=0.2, which="both")
    fig.suptitle("SCENIC Table III versus the current MPC imitation setup")
    fig.savefig(OUT / "scenic_table3_comparison.png", dpi=180, facecolor="white")
    fig.savefig(OUT / "scenic_table3_comparison.svg", facecolor="white")
    plt.close(fig)

    best = ranked[0]
    report = [
        "# SCENIC Table III comparison",
        "",
        "The table contains 12 paper architectures. Four reuse an existing coarse-sweep run and eight are newly trained in this directory.",
        "",
        f"Best under the current setup: `{best['model_id']}`, test normalized MSE={best['test_normalized_mse']:.6g}, physical MSE={best['test_physical_mse']:.6g}.",
        "",
        "The paper test-loss values are reported for the SCENIC authors' simulator, dataset, controller and preprocessing. They are included for architecture/ranking context, but are not numerically comparable to the current MSE unless those experimental definitions are reproduced exactly.",
        "",
        "Files: `scenic_table3_comparison.csv` and `scenic_table3_comparison.png`.",
    ]
    (OUT / "analysis.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "models": len(rows), "new": sum(r["source"].startswith("new") for r in rows), "reused": sum(r["source"].startswith("reused") for r in rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
