"""Plot the 29 previous and 12 deep-narrow controller experiments together."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "controller_architecture_study"
OLD = STUDY / "results" / "mse_all_29_20260926"
NEW = STUDY / "results" / "deep_narrow_v1" / "analysis_report_20260929"
OUT = STUDY / "results" / "mse_all_41_20260929"

GROUP_LABELS = {
    "FALCON original": "FALCON original",
    "Coarse sweep": "Coarse sweep plain",
    "Seed repeats": "Seed-repeat plain",
    "Residual": "Earlier residual",
    "Deep-narrow plain": "Deep-narrow plain",
    "Deep-narrow residual": "Deep-narrow residual",
}
COLORS = {
    "FALCON original": "#0072B2",
    "Coarse sweep": "#E69F00",
    "Seed repeats": "#009E73",
    "Residual": "#CC79A7",
    "Deep-narrow plain": "#D55E00",
    "Deep-narrow residual": "#56B4E9",
}
MARKERS = {"plain": "o", "residual": "s"}


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def mlp_params(hidden_layers: int, width: int, input_dim: int = 12, output_dim: int = 4) -> int:
    return input_dim * width + width + max(hidden_layers - 1, 0) * (width * width + width) + output_dim * width + output_dim


def load_rows() -> list[dict]:
    rows = []
    for item in read_csv(OLD / "all_29_metrics.csv"):
        architecture = item["architecture"]
        if architecture == "FALCON":
            model_type = "plain"
            params = next((int(x["parameter_count"]) for x in read_csv(STUDY / "results" / "architecture_scaling_summary.csv") if x["model_id"] == item["model_id"]), None)
        else:
            model_type = "residual" if item["group"] == "Residual" else "plain"
            dims = architecture.rsplit(" ", 1)[-1]
            layers, width = (int(value) for value in dims.split("x"))
            params = mlp_params(layers, width)
        rows.append({
            "model_id": item["model_id"],
            "label": item["label"],
            "group": item["group"],
            "group_label": GROUP_LABELS[item["group"]],
            "model_type": model_type,
            "parameters": params if params is not None else "",
            "test_normalized_mse": float(item["test_normalized_mse"]),
            "test_physical_mse": float(item["test_physical_mse"]),
            "source": "previous_29",
        })
    for item in read_csv(NEW / "deep_narrow_metrics.csv"):
        group = "Deep-narrow residual" if item["model_type"] == "residual_mlp" else "Deep-narrow plain"
        kind = "residual" if item["model_type"] == "residual_mlp" else "plain"
        rows.append({
            "model_id": item["model_id"],
            "label": f'{"R" if kind == "residual" else "P"} {item["layers"]}x{item["width"]}',
            "group": group,
            "group_label": GROUP_LABELS[group],
            "model_type": kind,
            "parameters": int(item["parameters"]),
            "test_normalized_mse": float(item["test_normalized_mse"]),
            "test_physical_mse": float(item["test_physical_mse"]),
            "source": "deep_narrow_12",
        })
    if len(rows) != 41 or len({row["model_id"] + row["source"] for row in rows}) != 41:
        raise RuntimeError(f"Expected 41 rows, found {len(rows)}")
    if any(row["parameters"] in (None, "") for row in rows):
        raise RuntimeError("Missing parameter count")
    return rows


def write_csv(rows: list[dict]) -> None:
    with (OUT / "all_41_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def legends(ax):
    group_handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS[key],
                             markeredgecolor=COLORS[key], markersize=7, label=GROUP_LABELS[key])
                     for key in GROUP_LABELS]
    type_handles = [Line2D([0], [0], marker=MARKERS[key], color="#444444", linestyle="none",
                           markersize=7, label=f'{key} network') for key in ("plain", "residual")]
    ax.legend(handles=group_handles + type_handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              frameon=False, fontsize=9, title="Color: experiment source\nShape: network type")


def plot(rows: list[dict]) -> None:
    ordered = sorted(rows, key=lambda row: row["test_physical_mse"])
    fig, axes = plt.subplots(1, 3, figsize=(22, 17), gridspec_kw={"width_ratios": [1.2, 1.2, 1.0]}, layout="constrained")

    for ax, metric, title, xlabel in (
        (axes[0], "test_normalized_mse", "All 41 models: normalized test MSE", "Normalized MSE"),
        (axes[1], "test_physical_mse", "All 41 models: physical-output test MSE", "Physical MSE"),
    ):
        y = list(range(len(ordered)))
        for index, row in enumerate(ordered):
            ax.scatter(row[metric], index, color=COLORS[row["group"]], marker=MARKERS[row["model_type"]], s=45)
        ax.set_xscale("log")
        ax.set_yticks(y, [row["label"] for row in ordered], fontsize=7.5)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid(axis="x", which="both", alpha=0.2)
    legends(axes[1])

    ax = axes[2]
    for row in rows:
        ax.scatter(row["parameters"], row["test_physical_mse"], color=COLORS[row["group"]],
                   marker=MARKERS[row["model_type"]], s=48)
    ax.set(xscale="log", yscale="log", xlabel="Parameters", ylabel="Physical test MSE",
           title="Accuracy versus parameter count")
    ax.grid(which="both", alpha=0.2)
    for row in rows:
        if row["group"] in ("Deep-narrow plain", "Deep-narrow residual"):
            ax.annotate(row["label"], (row["parameters"], row["test_physical_mse"]),
                        xytext=(4, 4), textcoords="offset points", fontsize=7)
    fig.suptitle("Complete controller architecture comparison: 29 previous + 12 deep-narrow models", fontsize=16)
    fig.text(0.01, 0.01, "All points are retained. Same 48,000-sample test split; log axes are used only to display the full range.", fontsize=9)
    fig.savefig(OUT / "all_41_model_comparison.png", dpi=180, facecolor="white")
    plt.close(fig)


def write_manifest(rows: list[dict]) -> None:
    counts = {key: sum(row["group"] == key for row in rows) for key in GROUP_LABELS}
    manifest = {
        "models": len(rows),
        "counts": counts,
        "metric": "DNN-minus-MPC error on the shared 48,000-sample test split",
        "color_encoding": GROUP_LABELS,
        "shape_encoding": {"o": "plain network", "s": "residual network"},
        "sources": [str(OLD / "all_29_metrics.csv"), str(NEW / "deep_narrow_metrics.csv")],
        "transformations": ["none for MSE values", "log axis for display only"],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    write_csv(rows)
    plot(rows)
    write_manifest(rows)
    best = min(rows, key=lambda row: row["test_physical_mse"])
    print(json.dumps({"output": str(OUT), "models": len(rows), "best": best["model_id"], "counts": {key: sum(row["group"] == key for row in rows) for key in GROUP_LABELS}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
