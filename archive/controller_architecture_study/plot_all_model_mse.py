"""Plot every completed, non-smoke M1 controller on the shared 12D dataset."""

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "controller_architecture_study"
RESULTS = STUDY / "results"
OUT = RESULTS / "mse_all_29_20260926"
DATASET = "experiments/paper_track_p_12d_5ms_v1/dnn_dataset"
COLORS = {
    "FALCON original": "#0072B2",
    "Coarse sweep": "#D55E00",
    "Seed repeats": "#009E73",
    "Residual": "#CC79A7",
}
MARKERS = {"FALCON original": "o", "Coarse sweep": "s", "Seed repeats": "^", "Residual": "D"}
GROUPS = list(COLORS)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def add_falcon(rows):
    for item in read_csv(RESULTS / "falcon_baseline.csv"):
        source = Path(item["source_metrics"])
        metrics = json.loads(source.read_text(encoding="utf-8"))
        rows.append({
            "model_id": item["model_id"],
            "label": "F " + item["model_id"].replace("depth", "").replace("_width", "x"),
            "group": "FALCON original",
            "architecture": "FALCON",
            "seed": int(item["seed"]),
            "epochs_completed": int(item["epochs_completed"]),
            "best_epoch": int(metrics["best_epoch"]),
            "test_normalized_mse": float(metrics["test_normalized"]["mse"]),
            "test_physical_mse": float(metrics["test"]["mse"]),
            "test_samples": int(metrics["test"]["samples"]),
            "metrics_path": str(source),
            "history_path": str(source.parent / "history.csv"),
        })


def add_study_run(rows, group, model_type, layers, width, seed, folder):
    model_id = f"{model_type}_h{layers}_w{width}_seed{seed}"
    metrics_path = folder / "offline_metrics.json"
    history_path = folder / "train_log.csv"
    config_path = folder / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))["test"]
    history = read_csv(history_path)
    if config["data_dir"].replace("\\", "/").rstrip("/") != DATASET:
        raise ValueError(f"Different dataset: {config_path}")
    if config["seed"] != seed or config["hidden_layers"] != layers or config["hidden_width"] != width:
        raise ValueError(f"Mismatched configuration: {config_path}")
    if int(metrics["samples"]) != 48000 or not history:
        raise ValueError(f"Missing full test evaluation or history: {folder}")
    best = min(history, key=lambda item: float(item["validation_normalized_mse"]))
    prefix = "R" if model_type == "residual_mlp" else "P"
    rows.append({
        "model_id": model_id,
        "label": f"{prefix} {layers}x{width} s{seed}",
        "group": group,
        "architecture": f"{model_type} {layers}x{width}",
        "seed": seed,
        "epochs_completed": int(history[-1]["epoch"]),
        "best_epoch": int(best["epoch"]),
        "test_normalized_mse": float(metrics["normalized_mse"]),
        "test_physical_mse": float(metrics["physical_mse"]),
        "test_samples": int(metrics["samples"]),
        "metrics_path": str(metrics_path),
        "history_path": str(history_path),
    })


def load_models():
    rows = []
    add_falcon(rows)
    sweep = json.loads((STUDY / "configs" / "coarse_sweep.json").read_text(encoding="utf-8"))
    for layers, width in sweep["architectures"]:
        folder = RESULTS / f"plain_mlp_h{layers}_w{width}_seed0"
        add_study_run(rows, "Coarse sweep", "plain_mlp", layers, width, 0, folder)
    for layers, width in ((2, 128), (4, 128), (8, 128), (10, 512)):
        for seed in (1, 2):
            folder = RESULTS / "followup_v1" / f"plain_mlp_h{layers}_w{width}_seed{seed}"
            add_study_run(rows, "Seed repeats", "plain_mlp", layers, width, seed, folder)
    for layers, width, seeds in ((4, 128, (0,)), (10, 512, (0, 1, 2))):
        for seed in seeds:
            folder = RESULTS / "followup_v1" / f"residual_mlp_h{layers}_w{width}_seed{seed}"
            add_study_run(rows, "Residual", "residual_mlp", layers, width, seed, folder)
    expected = {"FALCON original": 6, "Coarse sweep": 11, "Seed repeats": 8, "Residual": 4}
    counts = {group: sum(row["group"] == group for row in rows) for group in GROUPS}
    if counts != expected or len({row["model_id"] for row in rows}) != 29:
        raise ValueError(f"Missing or duplicated formal runs: {counts}")
    if {row["test_samples"] for row in rows} != {48000}:
        raise ValueError("The test sample counts are not comparable")
    for row in rows:
        for key in ("test_normalized_mse", "test_physical_mse"):
            if not math.isfinite(row[key]) or row[key] <= 0:
                raise ValueError(f"Invalid log-scale value in {row['model_id']}")
    return rows


def plot_test_mse(rows):
    fig, axes = plt.subplots(1, 2, figsize=(15, 13.5), sharey=True)
    positions = list(range(len(rows)))[::-1]
    boundaries = []
    count = 0
    for group in GROUPS[:-1]:
        count += sum(row["group"] == group for row in rows)
        boundaries.append(len(rows) - count - 0.5)
    for ax, key, title in zip(
        axes,
        ("test_normalized_mse", "test_physical_mse"),
        ("Normalized test MSE", "Physical-output test MSE"),
    ):
        values = [row[key] for row in rows]
        for y, row in zip(positions, rows):
            value = row[key]
            ax.scatter(value, y, color=COLORS[row["group"]], marker=MARKERS[row["group"]], s=45, zorder=3)
            ax.annotate(f"{value:.2e}", (value, y), xytext=(6, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=7.4)
        for boundary in boundaries:
            ax.axhline(boundary, color="#888888", lw=0.8)
        ax.set(xscale="log", xlim=(min(values) / 1.7, max(values) * 2.7),
               ylim=(-0.8, len(rows) - 0.2), title=title, xlabel="MSE across four control outputs (log10 axis)")
        ax.set_yticks(positions, [row["label"] for row in rows], fontsize=8)
        ax.grid(axis="x", which="both", alpha=0.2)
    fig.suptitle("All 29 completed controllers: MPC imitation on the same 48,000-state test split", fontsize=14)
    fig.text(0.26, 0.035, "F=original FALCON; P=plain MLP; R=residual MLP; s=training seed. Each point is one checkpoint.", fontsize=9)
    fig.subplots_adjust(left=0.24, right=0.98, top=0.92, bottom=0.095, wspace=0.13)
    fig.savefig(OUT / "all_29_test_mse.png", dpi=180, facecolor="white")
    plt.close(fig)


def plot_histories(rows, field, filename, title):
    fig, axes = plt.subplots(4, 1, figsize=(13.4, 14.2))
    for ax, group in zip(axes, GROUPS):
        subset = [row for row in rows if row["group"] == group]
        palette = plt.cm.tab20.colors
        for index, row in enumerate(subset):
            history = read_csv(Path(row["history_path"]))
            epoch = [int(item["epoch"]) for item in history]
            loss = [float(item[field]) for item in history]
            if any(not math.isfinite(value) or value <= 0 for value in loss):
                raise ValueError(f"Nonpositive loss in {row['history_path']}")
            color = palette[(2 * index) % len(palette)]
            style = ("-", "--", ":")[row["seed"] % 3] if group != "FALCON original" else "-"
            ax.plot(epoch, loss, color=color, linestyle=style, lw=1.0, label=row["label"])
            if field == "validation_normalized_mse":
                best_index = min(range(len(loss)), key=loss.__getitem__)
                ax.scatter(epoch[best_index], loss[best_index], color=color, s=17, zorder=3)
        ax.set_yscale("log")
        ax.set_ylabel("Normalized MSE")
        ax.set_xlabel("Epoch")
        ax.set_title(f"{group} ({len(subset)} runs)", loc="left", fontsize=11)
        ax.grid(alpha=0.2)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=14)
    fig.text(0.08, 0.02, "Raw per-epoch data; no smoothing. Different seeds/budgets for original FALCON; test MSE is shown separately.", fontsize=9)
    fig.subplots_adjust(left=0.09, right=0.71, top=0.94, bottom=0.07, hspace=0.48)
    fig.savefig(OUT / filename, dpi=180, facecolor="white")
    plt.close(fig)


def main():
    if OUT.exists():
        raise FileExistsError(f"Output already exists; refusing to overwrite: {OUT}")
    rows = load_models()
    OUT.mkdir(parents=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
    write_csv(OUT / "all_29_metrics.csv", rows)
    plot_test_mse(rows)
    plot_histories(rows, "train_normalized_mse", "all_29_training_curves.png", "Training normalized MSE: all 29 completed runs")
    plot_histories(rows, "validation_normalized_mse", "all_29_validation_curves.png", "Validation normalized MSE: all 29 completed runs")
    manifest = {
        "dataset": DATASET,
        "test_samples_per_model": 48000,
        "counts": {group: sum(row["group"] == group for row in rows) for group in GROUPS},
        "metric": "mean squared DNN-minus-MPC error over samples and four control outputs",
        "normalization": "(U - training_u_mean) / training_u_std; original FALCON JSON differs only by float32 rounding",
        "included": "six frozen FALCON models, eleven coarse-sweep seed-0 models, eight independent seed repeats, four residual runs",
        "excluded": "smoke/verification runs, exported copies of checkpoints, unrun optional low-LR fine-tune",
        "curves": "raw per-epoch training and validation normalized MSE; no smoothing or interpolation",
        "uncertainty": "all individual seeds plotted; no confidence intervals inferred from n<=3",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUT), "counts": manifest["counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
