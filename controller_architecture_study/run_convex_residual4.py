"""Four matched residual/plain comparisons; frozen data, serial 600-epoch training."""

import argparse
from copy import deepcopy
from pathlib import Path

from . import run_convex_scenic_falcon18 as sweep
from .decision_dataset import ROOT, atomic_json, sha256
from .convex_recovery import read_json


OUTPUT = ROOT/"controller_architecture_study/results/decision_convex_residual4_v1"
SHAPES = ((4, 512), (6, 1024), (21, 1024), (33, 1024))


def build_jobs():
    plain_jobs = sweep.build_jobs(sweep.EXPERIMENT, sweep.RECOVERY/"datasets/features_recovery", sweep.OUTPUT)
    jobs = []
    for name, plain, run in plain_jobs:
        if (plain["hidden_layers"], plain["hidden_width"]) not in SHAPES:
            continue
        if sweep.run_state(plain, run) != "complete":
            raise ValueError(f"matched 600-epoch plain baseline is not complete: {run}")
        done = read_json(run/"completion.json")
        if done["epochs_completed"] != 600 or "test" in read_json(run/"offline_metrics.json"):
            raise ValueError(f"baseline violates fixed-budget/no-Test protocol: {run}")
        config = deepcopy(plain)
        config.update(model_type="residual_mlp", residual=True, architecture_set="residual4_matched_plain",
                      plain_baseline=str(run), plain_metrics_sha256=sha256(run/"offline_metrics.json"))
        config["trainer_source_sha256"]["models/residual_mlp.py"] = sha256(
            ROOT/"controller_architecture_study/models/residual_mlp.py")
        residual_name = name.replace("plain_mlp_", "residual_mlp_", 1)
        jobs.append((residual_name, config, OUTPUT/"runs"/residual_name))
    if len(jobs) != 4:
        raise ValueError("requires four matched residual architectures")
    return jobs


def paired_report(jobs):
    current = read_json(sweep.summarize(jobs, OUTPUT))
    paired = []
    for _, config, run in jobs:
        row = next(item for item in current["models"] if item["run"] == str(run))
        plain_run = Path(config["plain_baseline"])
        if sha256(plain_run/"offline_metrics.json") != config["plain_metrics_sha256"]:
            raise ValueError(f"plain baseline metrics changed: {plain_run}")
        plain = read_json(plain_run/"offline_metrics.json")
        entry = {"model": row["model"], "status": row["status"], "parameters": row["parameter_count"],
                 "plain_run": str(plain_run), "residual_run": str(run),
                 "plain_train_mse": plain["train"]["normalized_mse"],
                 "plain_validation_mse": plain["validation"]["normalized_mse"]}
        if "validation_mse" in row:
            entry.update(residual_train_mse=row["train_mse"], residual_validation_mse=row["validation_mse"],
                         epochs=row["epochs"], best_epoch=row["best_epoch"],
                         validation_reduction_percent=100*(1-row["validation_mse"]/entry["plain_validation_mse"]))
        if row["status"] == "complete":
            entry.update(residual_fixed_train_mse=row["fixed_train_mse"],
                         residual_fresh_train_mse=row["fresh_train_mse"],
                         validation_fresh_ratio=row["validation_fresh_ratio"])
        paired.append(entry)
    path = OUTPUT/"residual_vs_plain.json"
    atomic_json(path, {"test_evaluated": False, "seed": 0, "epochs_per_model": 600, "models": paired,
        "note": "Same frozen data, normalization, seed, FC parameter count and training budget. "
                "Residual implementation includes scaled skips and its own initialization; "
                "this is a residual-architecture package comparison, not a skip-only causal isolation. "
                "FC MACs exclude skip additions/scales; measure actual latency separately. "
                "Single-seed results are exploratory, not statistical significance."})
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.dry_run and args.summarize_only:
        parser.error("choose --dry-run or --summarize-only")
    jobs = build_jobs()
    for name, config, run in jobs:
        print(f"{name}: {sweep.run_state(config, run)}, actual_epochs=600 -> {run}", flush=True)
    if args.dry_run:
        return
    if args.summarize_only:
        print(f"Paired comparison: {paired_report(jobs)}", flush=True)
        return
    import torch

    torch.set_num_threads(1)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; training has not started")
    try:
        sweep.execute(jobs, sweep.EXPERIMENT, OUTPUT, args.device)
    finally:
        print(f"Paired comparison: {paired_report(jobs)}", flush=True)


if __name__ == "__main__":
    main()
