"""Plot reproducible offline imitation losses and the five seed-80 gate runs."""

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "controller_architecture_study"
OUT = STUDY / "results" / "mse_comparison_20260926"
BASELINE = STUDY / "results" / "falcon_baseline.csv"
PILOT = STUDY / "results" / "closed_loop_pilot"
NEW_MODELS = [
    ("Plain 2x128", "plain_mlp_h2_w128_seed0", STUDY / "results" / "plain_mlp_h2_w128_seed0"),
    ("Plain 4x128", "plain_mlp_h4_w128_seed0", STUDY / "results" / "plain_mlp_h4_w128_seed0"),
    ("Plain 8x128", "plain_mlp_h8_w128_seed0", STUDY / "results" / "plain_mlp_h8_w128_seed0"),
    (
        "Residual 10x512",
        "residual_mlp_h10_w512_seed0",
        STUDY / "results" / "followup_v1" / "residual_mlp_h10_w512_seed0",
    ),
]
PILOT_REPORTS = [
    ("Plain 2x128", "plain_h2_w128_seed80.json"),
    ("Plain 4x128", "plain_h4_w128_seed80.json"),
    ("Plain 8x128", "plain_h8_w128_seed80.json"),
    ("Residual 10x512", "residual_h10_w512_seed80.json"),
    ("FALCON 22x512", "falcon_depth22_width512_seed80.json"),
]
COLORS = {"FALCON": "#0072B2", "Study": "#D55E00"}


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_offline():
    rows = []
    for item in read_csv(BASELINE):
        metrics_path = Path(item["source_metrics"])
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append({
            "model": "FALCON " + item["model_id"].replace("depth", "").replace("_width", "x"),
            "group": "FALCON",
            "test_normalized_mse": float(metrics["test_normalized"]["mse"]),
            "test_physical_mse": float(metrics["test"]["mse"]),
            "test_samples": int(metrics["test"]["samples"]),
            "best_epoch": int(metrics["best_epoch"]),
            "source_metrics": str(metrics_path),
            "history_csv": str(metrics_path.parent / "history.csv"),
        })
    for label, _, folder in NEW_MODELS:
        metrics_path = folder / "offline_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))["test"]
        history_path = folder / "train_log.csv"
        history = read_csv(history_path)
        best = min(history, key=lambda row: float(row["validation_normalized_mse"]))
        rows.append({
            "model": label,
            "group": "Study",
            "test_normalized_mse": float(metrics["normalized_mse"]),
            "test_physical_mse": float(metrics["physical_mse"]),
            "test_samples": int(metrics["samples"]),
            "best_epoch": int(best["epoch"]),
            "source_metrics": str(metrics_path),
            "history_csv": str(history_path),
        })
    if len(rows) != 10 or {row["test_samples"] for row in rows} != {48000}:
        raise ValueError("Expected six FALCON and four study models on the 48,000-sample test split")
    return rows


def plot_test_mse(rows):
    fig, axes = plt.subplots(1, 2, figsize=(12.7, 6.2), sharey=True)
    y_positions = list(range(len(rows)))[::-1]
    for ax, key, title in zip(
        axes,
        ("test_normalized_mse", "test_physical_mse"),
        ("Normalized control-output MSE", "Physical control-output MSE"),
    ):
        values = [row[key] for row in rows]
        for y, row in zip(y_positions, rows):
            value = row[key]
            ax.scatter(value, y, s=65, marker="o" if row["group"] == "FALCON" else "s",
                       color=COLORS[row["group"]], zorder=3)
            ax.annotate(f"{value:.2e}", (value, y), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=8)
        ax.set_xscale("log")
        ax.set_xlim(min(values) / 1.6, max(values) * 2.6)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Mean squared error across 4 controls (log scale)")
        ax.grid(axis="x", which="both", alpha=0.22)
        ax.axhline(3.5, color="#999999", lw=0.8)
        ax.set_ylim(-0.7, 9.7)
        ax.set_yticks(y_positions, [row["model"] for row in rows], fontsize=9)
    fig.suptitle("DNN imitation of MPC: held-out test set (48,000 states)", fontsize=14)
    fig.text(0.29, 0.025, "Same 12D dataset; blue circles = original six, orange squares = study candidates. Lower is better.", fontsize=9)
    fig.subplots_adjust(left=0.27, right=0.98, top=0.88, bottom=0.15, wspace=0.12)
    fig.savefig(OUT / "offline_test_mse.png", dpi=180)
    plt.close(fig)


def plot_validation_curves(rows):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    for ax, group, palette in zip(axes, ("FALCON", "Study"), (plt.cm.tab10.colors, plt.cm.Dark2.colors)):
        group_rows = [row for row in rows if row["group"] == group]
        for idx, row in enumerate(group_rows):
            history = read_csv(Path(row["history_csv"]))
            epoch = [int(item["epoch"]) for item in history]
            loss = [float(item["validation_normalized_mse"]) for item in history]
            color = palette[idx]
            ax.plot(epoch, loss, color=color, lw=1.15, label=row["model"])
            best_i = min(range(len(loss)), key=loss.__getitem__)
            ax.scatter(epoch[best_i], loss[best_i], color=color, s=24, zorder=3)
        ax.set_yscale("log")
        ax.set_title("Original FALCON (6)" if group == "FALCON" else "Architecture study (4)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation normalized MSE" if group == "FALCON" else "")
        ax.grid(alpha=0.22)
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("Validation loss histories: raw per-epoch values, best checkpoint marked", fontsize=13)
    fig.text(0.08, 0.025, "Runs used different seeds and training budgets; compare held-out test MSE separately. Logarithmic y-axis.", fontsize=9)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.16, wspace=0.16)
    fig.savefig(OUT / "validation_mse_curves.png", dpi=180)
    plt.close(fig)


def load_pilot():
    rows = []
    for label, filename in PILOT_REPORTS:
        path = PILOT / filename
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["seed"] != 80 or report["gate_count"] != 5 or not report["completed"] or len(report["crossings"]) != 5:
            raise ValueError(f"Incomplete or unmatched pilot run: {path}")
        offsets = [math.hypot(item["lateral_offset"], item["vertical_offset"])
                   for item in report["crossings"]]
        rows.append({
            "model": label,
            "seed": report["seed"],
            "crossings": len(offsets),
            "samples": report["samples"],
            "gate_plane_mse_m2": sum(value * value for value in offsets) / len(offsets),
            "gate_plane_rmse_m": math.sqrt(sum(value * value for value in offsets) / len(offsets)),
            **{f"gate_{idx}_radial_offset_m": value for idx, value in enumerate(offsets, 1)},
            "source_report": str(path),
        })
    return rows


def plot_gate_crossings(rows):
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2))
    colors = plt.cm.tab10.colors
    for idx, row in enumerate(rows):
        offsets = [row[f"gate_{gate}_radial_offset_m"] for gate in range(1, 6)]
        axes[0].plot(range(1, 6), offsets, marker="o", lw=1.5, color=colors[idx], label=row["model"])
    axes[0].set_xticks(range(1, 6))
    axes[0].set_xlabel("Gate number")
    axes[0].set_ylabel("Radial crossing offset (m)")
    axes[0].set_title("Error at each gate plane")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.22)
    y_positions = list(range(len(rows)))[::-1]
    for y, row, color in zip(y_positions, rows, colors):
        value = row["gate_plane_mse_m2"]
        axes[1].scatter(value, y, s=65, color=color)
        axes[1].annotate(f"{value:.5f}", (value, y), xytext=(7, 0), textcoords="offset points", va="center", fontsize=9)
    axes[1].set_yticks(y_positions, [row["model"] for row in rows], fontsize=9)
    axes[1].set_xlim(0, max(row["gate_plane_mse_m2"] for row in rows) * 1.45)
    axes[1].set_xlabel("Mean(lateral² + vertical²) across 5 gates (m²)")
    axes[1].set_title("Gate-crossing MSE (one flight each)")
    axes[1].grid(axis="x", alpha=0.22)
    fig.suptitle("AirSim closed loop: seed 80, Drone1, all 5/5 gates crossed", fontsize=13)
    fig.text(0.07, 0.025, "Single-run geometry metric, not MPC imitation loss; no uncertainty estimate from one seed.", fontsize=9)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.85, bottom=0.18, wspace=0.4)
    fig.savefig(OUT / "closed_loop_gate_crossing_mse.png", dpi=180)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
    offline = load_offline()
    pilot = load_pilot()
    write_csv(OUT / "offline_test_mse.csv", offline)
    write_csv(OUT / "closed_loop_gate_crossing_mse.csv", pilot)
    plot_test_mse(offline)
    plot_validation_curves(offline)
    plot_gate_crossings(pilot)
    for row in offline:
        print(f"{row['model']:<22} test_normalized_mse={row['test_normalized_mse']:.8g} test_physical_mse={row['test_physical_mse']:.8g}")
    print(f"Figures and source tables: {OUT}")


if __name__ == "__main__":
    main()
