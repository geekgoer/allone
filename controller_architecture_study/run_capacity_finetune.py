"""Run the three low-LR continuation jobs serially, with automatic resume.

Each model runs in a separate process so its GPU allocations are released on
exit. Source datasets/checkpoints are read-only. Test data are not evaluated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
HERE = ROOT / "controller_architecture_study"
DEFAULT_CONFIG = HERE / "configs/capacity_finetune_v1.json"
DEFAULT_RESULTS = HERE / "results/capacity_finetune_v1"


def write_json(path: Path, payload) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_jobs(config_path: Path, results: Path) -> list[tuple[str, dict, Path]]:
    suite = json.loads(config_path.read_text(encoding="utf-8"))
    jobs = []
    for entry in suite["jobs"]:
        source = (ROOT / entry["checkpoint"]).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        config = json.loads((source.parent / "config.json").read_text(encoding="utf-8"))
        config.update(suite["overrides"])
        config["init_checkpoint"] = str(source)
        with source.open("rb") as handle:
            config["source_checkpoint_sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
        if not config.get("preserve_initial_best") or config.get("evaluation_splits") != ["train", "validation"]:
            raise ValueError("this diagnostic requires initial-best preservation and train/validation only")
        target = (results / entry["name"]).resolve()
        if target == source.parent or (ROOT / config["data_dir"]).resolve() in (target, *target.parents):
            raise ValueError("results must be separate from source runs and the dataset")
        jobs.append((entry["name"], config, target))
    return jobs


def job_state(config: dict, target: Path) -> str:
    saved_config = target / "config.json"
    if saved_config.exists() and json.loads(saved_config.read_text(encoding="utf-8")) != config:
        raise ValueError(f"config or source checkpoint changed; choose a new results directory: {target}")
    if (target / "completion.json").exists():
        if not saved_config.exists() or not all((target / name).is_file() for name in
                ("offline_metrics.json", "initial_metrics.json", "best_model.pt", "last_model.pt")):
            raise ValueError(f"incomplete artifacts despite completion marker: {target}")
        return "complete"
    return "resume" if saved_config.exists() else "new"


def summarize(jobs, results: Path) -> list[dict]:
    rows = []
    for name, config, target in jobs:
        row = {"model": name, "status": job_state(config, target), "run": str(target)}
        if row["status"] == "complete":
            initial = json.loads((target / "initial_metrics.json").read_text(encoding="utf-8"))
            final = json.loads((target / "offline_metrics.json").read_text(encoding="utf-8"))
            completed = json.loads((target / "completion.json").read_text(encoding="utf-8"))
            before, after = initial["validation"]["normalized_mse"], final["validation"]["normalized_mse"]
            row.update({"epochs_completed": completed["epochs_completed"], "best_epoch": completed["best_epoch"],
                        "initial_train_mse": initial["train"]["normalized_mse"],
                        "train_mse": final["train"]["normalized_mse"],
                        "initial_validation_mse": before, "validation_mse": after,
                        "validation_improvement_percent": 100 * (before - after) / before if before else None})
        rows.append(row)
    write_json(results / "comparison.json", {"test_split_evaluated": False, "models": rows})
    fields = ["model", "status", "epochs_completed", "best_epoch", "initial_train_mse", "train_mse",
              "initial_validation_mse", "validation_mse", "validation_improvement_percent", "run"]
    temporary = results / "comparison.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(results / "comparison.csv")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    jobs = build_jobs(args.config, args.results_dir)
    states = [job_state(config, target) for _, config, target in jobs]
    for (name, config, target), state in zip(jobs, states):
        print(f"{name}: {state}; lr={config['learning_rate']}; up to {config['max_epochs']} additional epochs -> {target}", flush=True)
    if args.dry_run:
        print("Dry run only: no training or result files written.", flush=True)
        return
    args.results_dir.mkdir(parents=True, exist_ok=True)
    config_dir = args.results_dir / "job_configs"
    if not args.summarize_only:
        config_dir.mkdir(exist_ok=True)
        summarize(jobs, args.results_dir)
        for (name, config, target), state in zip(jobs, states):
            if state == "complete":
                print(f"Skipping completed model: {name}", flush=True)
                continue
            config_path = (config_dir / f"{name}.json").resolve()
            write_json(config_path, config)
            command = [sys.executable, "-m", "controller_architecture_study.train_controller",
                       "--config", str(config_path), "--output-dir", str(target), "--device", args.device]
            if state == "resume":
                command.append("--resume")
            print(f"Starting {name} in its own process", flush=True)
            try:
                subprocess.run(command, cwd=ROOT, check=True)
            except (KeyboardInterrupt, subprocess.CalledProcessError):
                print("Stopped. Saved models/epochs are retained; rerun the same command to continue.", flush=True)
                raise
            finally:
                summarize(jobs, args.results_dir)
    rows = summarize(jobs, args.results_dir)
    print(json.dumps(rows, indent=2), flush=True)
    print(f"Comparison: {(args.results_dir / 'comparison.json').resolve()}", flush=True)


if __name__ == "__main__":
    main()
