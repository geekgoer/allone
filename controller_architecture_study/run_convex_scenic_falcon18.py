"""Eighteen paper-inspired plain MLPs on frozen 237D recovery data, serial and resumable."""

import argparse
import csv
from pathlib import Path
import subprocess
import sys

from .compute_complexity import complexity
from .convex_dataset import EXPERIMENT
from .convex_features import CONTRACT
from .convex_recovery import OUTPUT as RECOVERY, read_json, verify_arm
from .decision_dataset import ROOT, atomic_json, sha256
from .run_convex_recovery import audit_run
from .train_convex_capacity import run_state


OUTPUT = ROOT/"controller_architecture_study/results/decision_convex_recovery_scenic_falcon18_v1"
CATALOG = Path(__file__).with_name("configs")/"scenic_table3_v2_layercount_corrected.json"
FALCON_CATALOG = CATALOG.with_name("baseline_models.json")


def architectures():
    scenic = read_json(CATALOG)["architectures"]
    if len(scenic) != 12 or any(2*item["hidden_layers"]+1 != item["paper_L"] for item in scenic):
        raise ValueError("requires the corrected twelve SCENIC architectures")
    rows = [{"group": "SCENIC", "hidden_layers": item["hidden_layers"], "width": item["hidden_width"],
             "paper_layers": item["paper_L"]} for item in scenic]
    falcon_ids = [item["model_id"] for item in read_json(FALCON_CATALOG)["models"]]
    expected = [f"depth{depth}_width{width}" for depth in (10, 22, 34) for width in (512, 1024)]
    if sorted(falcon_ids) != sorted(expected):
        raise ValueError("requires all six FALCON depth/width pairs")
    # FALCON depth counts FC layers including output, not hidden FC+ReLU operations.
    rows += [{"group": "FALCON", "hidden_layers": depth-1, "width": width, "paper_layers": depth}
             for depth in (10, 22, 34) for width in (512, 1024)]
    if len({(row["hidden_layers"], row["width"]) for row in rows}) != 18:
        raise ValueError("requires eighteen distinct architectures")
    return sorted(rows, key=lambda row: complexity(237, 4, row["hidden_layers"], row["width"])["parameter_count"])


def build_jobs(base, data_dir, output, epochs=600, seeds=(0,)):
    if epochs < 2 or not seeds or min(seeds) < 0 or len(set(seeds)) != len(seeds):
        raise ValueError("use an epoch budget >= 2 and unique nonnegative seeds")
    base, data_dir, output = base.resolve(), data_dir.resolve(), output.resolve()
    for protected in (base, data_dir.parent.parent):
        if output == protected or protected in output.parents:
            raise ValueError("new sweep output must be separate from the frozen data and previous runs")
    manifest = verify_arm(data_dir, base)
    if (manifest["arm"] != "features_recovery"
            or manifest["counts"] != {"train": 153600, "validation": 9600, "test": 9600}):
        raise ValueError("requires the complete frozen features_recovery dataset, not smoke data")
    catalog_hashes = {path.name: sha256(path) for path in (CATALOG, FALCON_CATALOG)}
    trainer_hashes = {name: sha256(ROOT/"controller_architecture_study"/name) for name in
                      ("train_controller.py", "models/factory.py", "models/plain_mlp.py", "eval_offline.py")}
    jobs = []
    for seed in seeds:
        for architecture in architectures():
            hidden, width = architecture["hidden_layers"], architecture["width"]
            name = f"plain_mlp_h{hidden}_w{width}_seed{seed}"
            config = {"input_dim": 237, "output_dim": 4, "input_contract": CONTRACT,
                "model_type": "plain_mlp", "activation": "relu", "residual": False,
                "hidden_layers": hidden, "hidden_width": width, "seed": seed, "batch_size": 512,
                "learning_rate": 1e-4, "min_learning_rate": 1e-5, "lr_schedule": "cosine", "weight_decay": 1e-5,
                # Unlike the old two-arm ablation, epochs is the actual budget: no halving.
                "max_epochs": epochs, "min_epochs": epochs, "patience": epochs, "gradient_clip_norm": 1.,
                "data_dir": str(data_dir), "evaluation_splits": ["train", "validation"],
                "ablation_arm": "features_recovery", "training_samples": 153600,
                "maximum_optimizer_updates": epochs*300,
                "architecture_group": architecture["group"], "paper_layers": architecture["paper_layers"],
                "architecture_catalog_sha256": catalog_hashes, "trainer_source_sha256": trainer_hashes,
                "dataset_source_sha256": manifest["source_manifest_sha256"],
                "augmentation_manifest_sha256": manifest["augmentation_manifest_sha256"],
                "dataset_file_sha256": {key: value for key, value in manifest["artifact_sha256"].items()
                                        if key != "test.npz"}}
            jobs.append((name, config, output/"runs"/name))
    return jobs


def convergence(run):
    with (run/"train_log.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    tail = rows[-100:]
    previous = float(rows[-101]["best_validation_normalized_mse"]) if len(rows) > 100 else None
    final = float(rows[-1]["best_validation_normalized_mse"])
    return {"last_window_epochs": len(tail),
            "best_validation_improvement_percent": None if previous is None or previous == 0 else 100*(previous-final)/previous,
            "mean_grad_norm_before_clip": sum(float(row["mean_grad_norm_before_clip"]) for row in tail)/len(tail),
            "mean_clip_fraction": sum(float(row["clip_fraction"]) for row in tail)/len(tail)}


def summarize(jobs, output):
    rows = []
    for name, config, run in jobs:
        state = run_state(config, run)
        row = {"model": name, "model_type": config["model_type"], "architecture_group": config["architecture_group"],
               "hidden_layers": config["hidden_layers"], "width": config["hidden_width"],
               "seed": config["seed"], "run": str(run), "epoch_budget": config["max_epochs"],
               "maximum_optimizer_updates": config["maximum_optimizer_updates"],
               "status": {"new": "not_started", "resume": "in_progress", "complete": "trained_pending_audit"}[state],
               **complexity(237, 4, config["hidden_layers"], config["hidden_width"])}
        if state == "complete":
            metrics, done = read_json(run/"offline_metrics.json"), read_json(run/"completion.json")
            if done["epochs_completed"] != config["max_epochs"] or "test" in metrics:
                raise ValueError(f"completion violates the fixed-budget/no-Test protocol: {run}")
            train, val = metrics["train"]["normalized_mse"], metrics["validation"]["normalized_mse"]
            row.update(train_mse=train, validation_mse=val, gap_mixture_absolute=val-train,
                       validation_mixture_ratio=None if train == 0 else val/train,
                       epochs=done["epochs_completed"], best_epoch=done["best_epoch"],
                       train=metrics["train"], validation=metrics["validation"], convergence=convergence(run))
            if (run/"fixed_state_audit.json").exists():
                audit = read_json(run/"fixed_state_audit.json")
                fixed, fresh = audit["original_train"]["normalized_mse"], audit["fresh_train_states"]["normalized_mse"]
                row.update(status="complete", fixed_train_mse=fixed, fresh_train_mse=fresh,
                           gap_absolute=val-fixed, validation_fixed_ratio=None if fixed == 0 else val/fixed,
                           validation_fresh_ratio=None if fresh == 0 else val/fresh, fixed_state_audit=audit)
        rows.append(row)
    rows.sort(key=lambda row: (row["parameter_count"], row["seed"]))
    path = output/"comparison.json"
    atomic_json(path, {"contract": CONTRACT, "architecture_set": jobs[0][1].get("architecture_set", "scenic12_falcon6"),
        "architectures_requested": len({(c["hidden_layers"], c["hidden_width"]) for _, c, _ in jobs}), "runs_requested": len(jobs),
        "runs_completed": sum(row["status"] == "complete" for row in rows), "test_evaluated": False,
        "epoch_budget_per_model": jobs[0][1]["max_epochs"], "models": rows,
        "note": "All models start from scratch on the same frozen 237D Train/Validation and output scales. "
                "Actual epochs are not halved; compare the same best-Validation checkpoint on every audit. "
                "fixed_train is the original Train subset; fresh_train is independent states on Train routes, "
                "not held-out environments. Paper-inspired architectures, not exact paper reproduction. "
                "Do not select a best seed or assume monotonic loss across depth/width."})
    fields = ("model", "architecture_group", "hidden_layers", "width", "seed", "status", "parameter_count", "mac_count",
              "epoch_budget", "epochs", "best_epoch", "train_mse", "fixed_train_mse", "fresh_train_mse",
              "validation_mse", "gap_mixture_absolute", "gap_absolute", "validation_mixture_ratio",
              "validation_fixed_ratio", "validation_fresh_ratio", "run")
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path.with_suffix(".csv"))
    return path


def execute(jobs, base, output, device):
    # Check the whole queue before starting; changed configs must not overwrite saved runs.
    states = [run_state(config, run) for _, config, run in jobs]
    summarize(jobs, output)
    for index, ((name, config, run), state) in enumerate(zip(jobs, states), 1):
        print(f"Model {index}/{len(jobs)}: {name}, {state}", flush=True)
        try:
            if state != "complete":
                config_path = output/"job_configs"/f"{name}.json"
                atomic_json(config_path, config)
                command = [sys.executable, "-m", "controller_architecture_study.train_controller",
                           "--config", str(config_path), "--output-dir", str(run), "--device", device]
                if state == "resume":
                    command.append("--resume")
                subprocess.run(command, cwd=ROOT, check=True)
            # Reuse the independent CPU audit; no parent-side CUDA model remains between children.
            print(f"CPU fixed/fresh-state audit: {name} (training is finished for this model)", flush=True)
            audit_run(run, base, "cpu")
        finally:
            summarize(jobs, output)
    return output/"comparison.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=EXPERIMENT)
    parser.add_argument("--data-dir", type=Path, default=RECOVERY/"datasets/features_recovery")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--epochs", type=int, default=600, help="actual epochs for EVERY model; not halved")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.dry_run and args.summarize_only:
        parser.error("--dry-run and --summarize-only cannot be combined")
    base, output = args.base.resolve(), args.output.resolve()
    jobs = build_jobs(base, args.data_dir, output, args.epochs, tuple(args.seeds))
    for name, config, run in jobs:
        print(f"{config['architecture_group']} {name}: {run_state(config, run)}, "
              f"actual_epochs={config['max_epochs']}, updates={config['maximum_optimizer_updates']} -> {run}", flush=True)
    if args.dry_run:
        return
    if args.summarize_only:
        print(f"Comparison: {summarize(jobs, output)}", flush=True)
        return
    import torch

    torch.set_num_threads(1)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else (
        "cpu" if args.device == "auto" else args.device)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; training has not started")
    print(f"Comparison: {execute(jobs, base, output, device)}", flush=True)


if __name__ == "__main__":
    main()
