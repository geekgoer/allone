"""Convex-teacher architecture sweeps, serial processes and epoch resume."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from .decision_dataset import ROOT, FAMILIES, atomic_json
from .convex_dataset import EXPERIMENT, CONTRACT
from .fly_convex_policy import verify_dataset
from .compute_complexity import complexity


WIDTHS = (64, 128, 256, 1024)
ARCHITECTURE_SETS = ("widths4", "scenic_falcon15")


def architectures(architecture_set):
    if architecture_set == "widths4":
        return [(4, width) for width in WIDTHS]
    if architecture_set != "scenic_falcon15":
        raise ValueError(f"unknown architecture set: {architecture_set}")
    path = Path(__file__).with_name("configs")/"scenic_table3_v2_layercount_corrected.json"
    scenic = json.loads(path.read_text(encoding="utf-8"))["architectures"]
    pairs = [(item["hidden_layers"], item["hidden_width"]) for item in scenic]
    if len(pairs) != 12 or any(2*item["hidden_layers"]+1 != item["paper_L"] for item in scenic):
        raise ValueError("requires the corrected twelve SCENIC architectures")
    # Existing FALCON depth counts all FC layers, including the output FC.
    pairs += [(depth-1, 512) for depth in (10, 22, 34)]
    if len(set(pairs)) != 15:
        raise ValueError("requires fifteen distinct architectures")
    return sorted(pairs, key=lambda pair: complexity(300, 4, *pair)["parameter_count"])


def run_state(config, run):
    saved = run/"config.json"
    if saved.exists() and json.loads(saved.read_text(encoding="utf-8")) != config:
        raise ValueError(f"configuration changed; use a different run directory: {run}")
    if (run/"completion.json").exists():
        if not saved.exists() or not all((run/name).is_file() for name in
                                        ("best_model.pt", "last_model.pt", "offline_metrics.json")):
            raise ValueError(f"incomplete artifacts despite completion marker: {run}")
        return "complete"
    return "resume" if saved.exists() else "new"


def build_jobs(output, seeds, epochs, architecture_set="widths4"):
    if epochs < 150 or not seeds or min(seeds) < 0 or len(set(seeds)) != len(seeds):
        raise ValueError("use unique nonnegative seeds and at least 150 epochs")
    manifest = verify_dataset(output)
    audit = json.loads((output/"data_audit.json").read_text(encoding="utf-8"))
    if (audit["status"] != "PASS" or audit["samples"] != 96000
            or manifest["counts"] != {"train": 76800, "validation": 9600, "test": 9600}):
        raise ValueError("prepare the complete 96k dataset, not the smoke dataset")
    jobs = []
    for seed in seeds:
        for depth, width in architectures(architecture_set):
            name = f"plain_mlp_h{depth}_w{width}_seed{seed}"
            config = {"input_dim": 300, "output_dim": 4, "input_contract": CONTRACT,
                "model_type": "plain_mlp", "activation": "relu", "residual": False,
                "hidden_layers": depth, "hidden_width": width, "seed": seed, "batch_size": 512,
                "learning_rate": 1e-4, "min_learning_rate": 1e-5, "lr_schedule": "cosine", "weight_decay": 1e-5,
                "max_epochs": epochs, "min_epochs": 150, "patience": 100, "gradient_clip_norm": 1.,
                "data_dir": str((output/"dnn_dataset").resolve()), "evaluation_splits": ["train", "validation"],
                "dataset_source_sha256": manifest["source_manifest_sha256"],
                "dataset_file_sha256": {key: manifest["artifact_sha256"][key] for key in
                                        ("train.npz", "validation.npz", "normalization.json")}}
            jobs.append((name, config, output/"runs"/name))
    return jobs


def summarize(output, architecture_set="widths4"):
    rows = []
    selected = architectures(architecture_set)
    for run in sorted((output/"runs").glob("plain_mlp_h*_w*_seed*")):
        if not (run/"config.json").exists():
            continue
        config = json.loads((run/"config.json").read_text(encoding="utf-8"))
        depth, width = config["hidden_layers"], config["hidden_width"]
        if (depth, width) not in selected:
            continue
        row = {"model": run.name, "hidden_layers": depth, "width": width, "seed": config["seed"],
               "architecture_group": ("FALCON" if (depth, width) in ((9, 512), (21, 512), (33, 512))
                                      else "SCENIC" if architecture_set == "scenic_falcon15" else "widths4"),
               "run": str(run), "status": run_state(config, run), **complexity(300, 4, depth, width)}
        if row["status"] == "complete":
            metrics = json.loads((run/"offline_metrics.json").read_text(encoding="utf-8"))
            done = json.loads((run/"completion.json").read_text(encoding="utf-8"))
            t, v = metrics["train"]["normalized_mse"], metrics["validation"]["normalized_mse"]
            row.update(train_mse=t, validation_mse=v, gap_absolute=v-t,
                validation_train_ratio=None if t == 0 else v/t, epochs=done["epochs_completed"], best_epoch=done["best_epoch"],
                validation_physical_rmse=metrics["validation"]["physical_rmse"],
                validation_per_output_rmse=metrics["validation"]["per_output_rmse"],
                validation_p99_abs_error=metrics["validation"]["p99_abs_error"],
                by_family={family: {"train_mse": metrics["train"]["by_family_id"][str(i)]["normalized_mse"],
                                    "validation_mse": metrics["validation"]["by_family_id"][str(i)]["normalized_mse"]}
                           for i, family in enumerate(FAMILIES)})
        rows.append(row)
    aggregate = []
    for depth, width in selected:
        completed = [row for row in rows if row["hidden_layers"] == depth and row["width"] == width
                     and row["status"] == "complete"]
        if completed:
            values = np.array([row["validation_mse"] for row in completed])
            aggregate.append({"hidden_layers": depth, "width": width, "seeds_completed": len(completed), "seeds": [r["seed"] for r in completed],
                "validation_mse_mean": float(values.mean()),
                "validation_mse_std": None if len(values) < 2 else float(values.std(ddof=1)),
                "provisional_single_seed": len(values) == 1})
    rows.sort(key=lambda row: (row["parameter_count"], row["seed"]))
    path = output/"runs"/("comparison.json" if architecture_set == "widths4" else "comparison_scenic_falcon15.json")
    atomic_json(path, {"contract": CONTRACT, "architecture_set": architecture_set,
        "architectures_requested": len(selected), "test_evaluated": False, "models": rows,
        "by_architecture": aggregate, "by_width": aggregate if architecture_set == "widths4" else [],
        "note": "Same best checkpoint in eval mode for Train/Validation. Report all seeds; do not select seeds by test or enforce monotonicity."})
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=EXPERIMENT)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--architecture-set", choices=ARCHITECTURE_SETS, default="widths4")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.summarize_only:
        print(f"Comparison: {summarize(output, args.architecture_set)}", flush=True)
        return
    jobs = build_jobs(output, args.seeds, args.epochs, args.architecture_set)
    states = [run_state(config, run) for _, config, run in jobs]
    print(f"Architecture set: {args.architecture_set}; {len(jobs)} model runs", flush=True)
    for (name, config, run), state in zip(jobs, states):
        print(f"{name}: {state}, max={config['max_epochs']} epochs -> {run}", flush=True)
    if args.dry_run:
        return
    for (name, config, run), state in zip(jobs, states):
        if state == "complete":
            continue
        config_path = output/"job_configs"/f"{name}.json"
        atomic_json(config_path, config)
        command = [sys.executable, "-m", "controller_architecture_study.train_controller", "--config", str(config_path),
                   "--output-dir", str(run), "--device", args.device]
        if state == "resume":
            command.append("--resume")
        try:
            # Separate processes free GPU allocations; failure stops the queue.
            subprocess.run(command, cwd=ROOT, check=True)
        finally:
            summarize(output, args.architecture_set)
    print(f"Comparison: {summarize(output, args.architecture_set)}", flush=True)


if __name__ == "__main__":
    main()
