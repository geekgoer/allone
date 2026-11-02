"""Four fixed-depth architectures, three serial Train/Validation-only ablation arms."""

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

from .decision_dataset import ROOT, atomic_json, sha256
from .decision_dataset_96k import EXPERIMENT
from .decision_recovery import OUTPUT, source_info
from .run_capacity_finetune import job_state


WIDTHS = (64, 128, 256, 1024)
ARMS = ("old_control", "added_only", "added_regularized")


def build_jobs(base, output, arms, epochs):
    _, split = source_info(base)
    original = base / "dnn_dataset"
    if set(arms) - set(ARMS) or epochs < 100:
        raise ValueError("requires known arms and at least 100 augmented-data epochs")
    new_count = split["counts"]["train"] + 32 * len(split["route_ids"]["train"])
    augmentation = None
    if any(arm != "old_control" for arm in arms):
        augmentation = json.loads((output / "dnn_dataset/split_manifest.json").read_text(encoding="utf-8"))
        collection = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        audit = json.loads((output / "data_audit.json").read_text(encoding="utf-8"))
        if (audit["status"] != "PASS" or augmentation["source_manifest_sha256"] != split["source_manifest_sha256"]
                or not audit["validation_byte_identical"] or not audit["normalization_byte_identical"]
                or augmentation["augmentation_manifest_sha256"] != sha256(output / "manifest.json")
                or augmentation["base_data_dir"] != str(original.resolve())
                or augmentation["route_ids"] != split["route_ids"]
                or augmentation["counts"]["train"] != new_count or audit["train_samples"] != new_count):
            raise ValueError("augmentation audit/source mismatch; prepare the frozen augmentation first")
        for name, expected in collection["config"]["base_file_sha256"].items():
            if sha256(original / name) != expected:
                raise ValueError(f"original dataset changed after augmentation: {name}")
        for name, expected in augmentation["file_sha256"].items():
            if sha256(output / "dnn_dataset" / name) != expected:
                raise ValueError(f"changed augmented dataset: {name}")
        for name in ("normalization.json", "validation.npz"):
            if sha256(original / name) != augmentation["file_sha256"][name]:
                raise ValueError(f"held-out or normalization file differs: {name}")
    original_hashes = {name: sha256(original / name) for name in
                       ("train.npz", "validation.npz", "normalization.json")}
    old_batches = (split["counts"]["train"] + 511) // 512
    # Fixed production recipe: 32 additional queries for every Train route.
    new_batches = (new_count + 511) // 512
    jobs = []
    for arm in arms:
        for width in WIDTHS:
            name = f"plain_mlp_h4_w{width}_seed0"
            source = base / "runs" / name / "best_model.pt"
            config = json.loads((source.parent / "config.json").read_text(encoding="utf-8"))
            if (config["input_dim"] != 300 or config["hidden_layers"] != 4 or config["hidden_width"] != width
                    or config["model_type"] != "plain_mlp" or config["activation"] != "relu"
                    or config["residual"] or config["dataset_source_sha256"] != split["source_manifest_sha256"]):
                raise ValueError(f"source model architecture or provenance mismatch: {name}")
            old = arm == "old_control"
            budget = round(epochs * new_batches / old_batches) if old else epochs
            data_dir = original if old else output / "dnn_dataset"
            config.update(init_checkpoint=str(source.resolve()), source_checkpoint_sha256=sha256(source),
                data_dir=str(data_dir.resolve()), preserve_initial_best=True, seed=0,
                evaluation_splits=["train", "validation"], learning_rate=1e-4, batch_size=512,
                max_epochs=budget, min_epochs=round(100 * new_batches / old_batches) if old else 100,
                patience=round(80 * new_batches / old_batches) if old else 80, gradient_clip_norm=1.,
                weight_decay=1e-5 if arm == "added_regularized" else 0.,
                lr_schedule="cosine" if arm == "added_regularized" else "constant",
                min_learning_rate=1e-5, ablation_arm=arm,
                dataset_file_sha256=original_hashes if old else augmentation["file_sha256"],
                augmentation_manifest_sha256=None if old else augmentation["augmentation_manifest_sha256"],
                training_samples=split["counts"]["train"] if old else new_count,
                maximum_optimizer_updates=budget * (old_batches if old else new_batches))
            config.pop("train_limit", None)
            config.pop("validation_limit", None)
            jobs.append((name, config, output / "runs" / arm / name))
    return jobs


def summarize(output):
    rows = []
    for arm in ARMS:
        for width in WIDTHS:
            name = f"plain_mlp_h4_w{width}_seed0"
            run = output / "runs" / arm / name
            if not (run / "config.json").exists():
                rows.append({"arm": arm, "model": name, "status": "not_started"})
                continue
            config = json.loads((run / "config.json").read_text(encoding="utf-8"))
            row = {"arm": arm, "model": name, "status": job_state(config, run), "run": str(run),
                   "training_samples": config["training_samples"]}
            if row["status"] == "complete":
                before = json.loads((run / "initial_metrics.json").read_text(encoding="utf-8"))["validation"]
                after = json.loads((run / "offline_metrics.json").read_text(encoding="utf-8"))
                completion = json.loads((run / "completion.json").read_text(encoding="utf-8"))
                row.update(best_epoch=completion["best_epoch"], epochs=completion["epochs_completed"],
                    baseline_validation_mse=before["normalized_mse"], validation_mse=after["validation"]["normalized_mse"],
                    train_mse=after["train"]["normalized_mse"],
                    combined_validation_mse=after["validation"].get("by_family_id", {}).get("7", {}).get("normalized_mse"),
                    validation_per_output_rmse=after["validation"].get("per_output_rmse"),
                    validation_by_family=after["validation"].get("by_family_id"))
            rows.append(row)
    path = output / "runs/comparison.json"
    atomic_json(path, {"test_evaluated": False, "models": rows,
        "note": "Compare fixed Validation. Train MSE has different populations across arms. "
                "old_control vs added_only isolates augmentation at matched maximum update budgets; "
                "added_only vs added_regularized bundles L2 and LR decay. Early stops can change actual budgets."})
    fields = ("arm", "model", "status", "training_samples", "epochs", "best_epoch", "baseline_validation_mse",
              "train_mse", "validation_mse", "combined_validation_mse", "run")
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path.with_suffix(".csv"))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=(*ARMS, "all"), default="all")
    parser.add_argument("--base", type=Path, default=EXPERIMENT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--epochs", type=int, default=300, help="additional epochs for each augmented-data arm")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    base, output = args.base.resolve(), args.output.resolve()
    if output == base or base in output.parents or output in base.parents:
        raise ValueError("results must be separate from the original experiment")
    if args.summarize_only:
        summarize(output)
        print(f"Comparison: {output / 'runs/comparison.json'}")
        return
    jobs = build_jobs(base, output, ARMS if args.arm == "all" else (args.arm,), args.epochs)
    states = [job_state(config, run) for _, config, run in jobs]
    for (name, config, run), state in zip(jobs, states):
        print(f"{config['ablation_arm']}/{name}: {state}, max={config['max_epochs']} epochs, "
              f"updates<={config['maximum_optimizer_updates']}, destination={run}", flush=True)
    if args.dry_run:
        print("Dry run only; no training or writes.")
        return
    for (name, config, run), state in zip(jobs, states):
        if state == "complete":
            continue
        config_path = output / "job_configs" / config["ablation_arm"] / f"{name}.json"
        atomic_json(config_path, config)
        command = [sys.executable, "-m", "controller_architecture_study.train_controller", "--config", str(config_path),
                   "--output-dir", str(run), "--device", args.device]
        if state == "resume":
            command.append("--resume")
        try:
            subprocess.run(command, cwd=ROOT, check=True)
        finally:
            summarize(output)
    print(f"Comparison: {output / 'runs/comparison.json'}", flush=True)


if __name__ == "__main__":
    main()
