"""Four widths, two 237D data arms, serial resumable and update-matched training."""

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

from .compute_complexity import complexity
from .convex_dataset import EXPERIMENT
from .convex_features import CONTRACT
from .convex_recovery import OUTPUT, ARMS, read_json, verify_arm
from .decision_dataset import ROOT, atomic_json, sha256
from .eval_offline import evaluate, load_checkpoint
from .train_convex_capacity import run_state


WIDTHS = (128, 256, 512, 1024)


def selection(arms, seeds):
    if not arms or len(set(arms)) != len(arms) or any(arm not in ARMS for arm in arms):
        raise ValueError("select unique known data arms")
    if not seeds or min(seeds) < 0 or len(set(seeds)) != len(seeds):
        raise ValueError("select unique nonnegative seeds")


def build_jobs(base, output, epochs, *, arms=ARMS, seeds=(0,)):
    selection(arms, seeds)
    if epochs < 2 or epochs % 2:
        raise ValueError("use an even original-data epoch budget >= 2")
    jobs = []
    for arm in arms:
        data_dir = output/"datasets"/arm
        manifest = verify_arm(data_dir, base)
        expected = 76800 if arm == "features_only" else 153600
        if manifest["counts"] != {"train": expected, "validation": 9600, "test": 9600}:
            raise ValueError("requires the complete prepared production ablation, not smoke data")
        budget = epochs if arm == "features_only" else epochs//2
        for seed, width in ((seed, width) for seed in seeds for width in WIDTHS):
            name = f"plain_mlp_h4_w{width}_seed{seed}"
            config = {"input_dim": 237, "output_dim": 4, "input_contract": CONTRACT,
                "model_type": "plain_mlp", "activation": "relu", "residual": False,
                "hidden_layers": 4, "hidden_width": width, "seed": seed, "batch_size": 512,
                "learning_rate": 1e-4, "min_learning_rate": 1e-5, "lr_schedule": "cosine", "weight_decay": 1e-5,
                # Fixed maximum updates isolate data effects; best selection still uses Validation only.
                "max_epochs": budget, "min_epochs": budget, "patience": budget, "gradient_clip_norm": 1.,
                "data_dir": str(data_dir.resolve()), "evaluation_splits": ["train", "validation"],
                "ablation_arm": arm, "training_samples": expected,
                "maximum_optimizer_updates": budget*(expected//512),
                "dataset_source_sha256": manifest["source_manifest_sha256"],
                "augmentation_manifest_sha256": manifest["augmentation_manifest_sha256"],
                "dataset_file_sha256": {key: value for key, value in manifest["artifact_sha256"].items()
                                        if key != "test.npz"}}
            jobs.append((name, config, output/"runs"/arm/name))
    return jobs


def audit_run(run, base, device):
    config = read_json(run/"config.json")
    data_dir = Path(config["data_dir"])
    manifest = verify_arm(data_dir, base)
    checkpoint = run/"best_model.pt"
    signature = {"checkpoint_sha256": sha256(checkpoint),
                 "artifact_sha256": {key: manifest["artifact_sha256"][key] for key in
                                     ("normalization.json", "audit/train.npz", "probe/train.npz",
                                      "audit/normalization.json", "probe/normalization.json")}}
    path = run/"fixed_state_audit.json"
    if path.exists() and read_json(path)["signature"] == signature:
        return
    model = load_checkpoint(checkpoint, device=device)
    metrics = {"signature": signature, "test_evaluated": False,
               "original_train": evaluate(model, data_dir/"audit", "train", device=device),
               "fresh_train_states": evaluate(model, data_dir/"probe", "train", device=device)}
    atomic_json(path, metrics)


def seed_statistics(rows, arms, seeds):
    aggregate, paired = [], []
    for arm in arms:
        complete = { (row["width"], row["seed"]): row for row in rows
                     if row["arm"] == arm and row["status"] == "complete" }
        for width in WIDTHS:
            found = [complete[width, seed] for seed in seeds if (width, seed) in complete]
            entry = {"arm": arm, "width": width, "seeds": [row["seed"] for row in found],
                     "seeds_completed": len(found), "panel_complete": len(found) == len(seeds)}
            for metric in ("fixed_train_mse", "fresh_train_mse", "validation_mse", "gap_absolute"):
                values = np.asarray([row[metric] for row in found])
                entry[metric+"_mean"] = None if not len(values) else float(values.mean())
                entry[metric+"_std"] = None if len(values) < 2 else float(values.std(ddof=1))
            aggregate.append(entry)
        for narrow, wide in zip(WIDTHS[:-1], WIDTHS[1:]):
            common = [seed for seed in seeds if (narrow, seed) in complete and (wide, seed) in complete]
            deltas = [complete[wide, seed]["validation_mse"]-complete[narrow, seed]["validation_mse"] for seed in common]
            paired.append({"arm": arm, "narrow_width": narrow, "wide_width": wide,
                "seeds": common, "panel_complete": len(common) == len(seeds),
                "validation_wide_minus_narrow": deltas, "wider_better_count": sum(delta < 0 for delta in deltas),
                "mean_delta": None if not deltas else float(np.mean(deltas))})
    return aggregate, paired


def summarize(base, output, *, arms=ARMS, seeds=(0,)):
    selection(arms, seeds)
    rows = []
    for arm in arms:
        for seed, width in ((seed, width) for seed in seeds for width in WIDTHS):
            name = f"plain_mlp_h4_w{width}_seed{seed}"
            run = output/"runs"/arm/name
            row = {"arm": arm, "model": name, "width": width, "seed": seed, "hidden_layers": 4,
                   "run": str(run), "status": "not_started", **complexity(237, 4, 4, width)}
            if (run/"config.json").exists():
                config = read_json(run/"config.json")
                row.update(status=run_state(config, run), training_samples=config["training_samples"],
                           maximum_optimizer_updates=config["maximum_optimizer_updates"])
                if row["status"] == "complete":
                    metrics, done = read_json(run/"offline_metrics.json"), read_json(run/"completion.json")
                    row.update(train_mse=metrics["train"]["normalized_mse"],
                        validation_mse=metrics["validation"]["normalized_mse"],
                        epochs=done["epochs_completed"], best_epoch=done["best_epoch"],
                        validation=metrics["validation"])
                    if (run/"fixed_state_audit.json").exists():
                        audit = read_json(run/"fixed_state_audit.json")
                        t, v = audit["original_train"]["normalized_mse"], row["validation_mse"]
                        row.update(fixed_train_mse=t, fresh_train_mse=audit["fresh_train_states"]["normalized_mse"],
                                   gap_absolute=v-t, validation_train_ratio=None if t == 0 else v/t,
                                   fixed_state_audit=audit)
                    else:
                        row["status"] = "trained_pending_audit"
            rows.append(row)
    baseline = []
    for width in WIDTHS:
        run = base/"runs"/f"plain_mlp_h4_w{width}_seed0"
        entry = {"width": width, "run": str(run), "status": "not_available"}
        if (run/"completion.json").exists():
            old, done = read_json(run/"offline_metrics.json"), read_json(run/"completion.json")
            entry.update(status="complete", train_mse=old["train"]["normalized_mse"],
                         validation_mse=old["validation"]["normalized_mse"], epochs=done["epochs_completed"])
        baseline.append(entry)
    legacy = tuple(arms) == ARMS and tuple(seeds) == (0,)
    suffix = "_".join(arms)+"_seeds"+"_".join(map(str, seeds))
    path = output/"runs"/("comparison.json" if legacy else f"comparison_{suffix}.json")
    aggregate, paired = seed_statistics(rows, arms, seeds)
    atomic_json(path, {"contract": CONTRACT, "architectures": [[4, width] for width in WIDTHS],
        "test_evaluated": False, "models": rows, "old_300d_reference": baseline,
        "seeds_requested": list(seeds), "arms_requested": list(arms),
        "by_architecture": aggregate, "paired_width_comparisons": paired,
        "note": "Compare identical Validation and frozen output scales. gap_absolute uses original Train, "
                "not the recovery-weighted training mixture. Same maximum optimizer updates, not equal wall time. "
                "Report every seed, not the best seed. Sample standard deviations are not confidence intervals. "
                "Monotonic loss and closed-loop robustness are not guaranteed."})
    fields = ("arm", "model", "seed", "status", "parameter_count", "epochs", "best_epoch", "training_samples",
              "maximum_optimizer_updates", "train_mse", "fixed_train_mse", "fresh_train_mse", "validation_mse",
              "gap_absolute", "validation_train_ratio", "run")
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path.with_suffix(".csv"))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=EXPERIMENT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--epochs", type=int, default=400, help="original-data epochs; augmentation uses half")
    parser.add_argument("--arm", choices=("all", *ARMS), default="all")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    base, output = args.base.resolve(), args.output.resolve()
    selected = {"arms": ARMS if args.arm == "all" else (args.arm,), "seeds": tuple(args.seeds)}
    if args.summarize_only:
        print(f"Comparison: {summarize(base, output, **selected)}", flush=True)
        return
    jobs = build_jobs(base, output, args.epochs, **selected)
    states = [run_state(config, run) for _, config, run in jobs]
    for (name, config, run), state in zip(jobs, states):
        print(f"{config['ablation_arm']}/{name}: {state}, epochs={config['max_epochs']}, "
              f"updates={config['maximum_optimizer_updates']} -> {run}", flush=True)
    if args.dry_run:
        return
    import torch

    torch.set_num_threads(1)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else (
        "cpu" if args.device == "auto" else args.device)
    for (name, config, run), state in zip(jobs, states):
        try:
            if state != "complete":
                config_path = output/"job_configs"/config["ablation_arm"]/f"{name}.json"
                atomic_json(config_path, config)
                command = [sys.executable, "-m", "controller_architecture_study.train_controller", "--config", str(config_path),
                           "--output-dir", str(run), "--device", device]
                if state == "resume":
                    command.append("--resume")
                subprocess.run(command, cwd=ROOT, check=True)
            # CPU audits keep the parent process from reserving GPU memory between children.
            audit_run(run, base, "cpu")
        finally:
            summarize(base, output, **selected)
    print(f"Comparison: {summarize(base, output, **selected)}", flush=True)


if __name__ == "__main__":
    main()
