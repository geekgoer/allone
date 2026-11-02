"""Serial decision-policy pilot or capacity grid; completed models are reused."""

import argparse
import json
import subprocess
import sys

from .decision_dataset import CONTRACT, EXPERIMENT, ROOT, atomic_json
from .decision_dataset_96k import EXPERIMENT as EXPERIMENT_96K, VERSION as VERSION_96K
from .compute_complexity import complexity


PILOT_GRID = ((4, 128), (8, 256), (8, 512))
# Separate width and depth effects; the three original pilot points are included.
CAPACITY_GRID = ((4, 64), (4, 128), (4, 256), (4, 512), (4, 1024),
                 (2, 256), (8, 256), (12, 256), (8, 512))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dataset-version", choices=("v1", "96k"), default="v1")
    parser.add_argument("--grid", choices=("pilot", "capacity"), default="pilot")
    args = parser.parse_args()
    experiment = EXPERIMENT_96K if args.dataset_version == "96k" else EXPERIMENT
    expected = 96000 if args.dataset_version == "96k" else 40000
    dataset = experiment / "dnn_dataset"
    manifest = json.loads((dataset / "split_manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((experiment / "data_audit.json").read_text(encoding="utf-8"))
    expected_counts = {"train": int(expected * .8), "validation": int(expected * .1), "test": int(expected * .1)}
    if (manifest["contract"] != CONTRACT or audit["status"] != "PASS" or audit["samples"] != expected
            or manifest["counts"] != expected_counts
            or (args.dataset_version == "96k" and manifest.get("version") != VERSION_96K)):
        raise ValueError(f"prepare and audit the complete {expected}-sample pilot first (not smoke data)")
    runs = experiment / "runs"
    grid = CAPACITY_GRID if args.grid == "capacity" else PILOT_GRID
    comparison_path = runs / ("comparison_capacity.json" if args.grid == "capacity" else "comparison.json")
    comparison = []
    for layers, width in grid:
        name = f"plain_mlp_h{layers}_w{width}_seed0"
        output = runs / name
        config = {"input_dim": 300, "output_dim": 4, "model_type": "plain_mlp", "activation": "relu",
                  "residual": False, "hidden_layers": layers, "hidden_width": width,
                  "seed": 0, "learning_rate": 1e-4, "batch_size": 512,
                  "max_epochs": 400, "min_epochs": 100, "patience": 80, "gradient_clip_norm": 1.,
                  "data_dir": str(dataset.relative_to(ROOT)), "evaluation_splits": ["train", "validation"],
                  "dataset_source_sha256": manifest["source_manifest_sha256"]}
        config_path = runs / "configs" / f"{name}.json"
        saved_config = output / "config.json"
        if saved_config.exists() and json.loads(saved_config.read_text(encoding="utf-8")) != config:
            raise ValueError(f"existing run configuration differs: {output}")
        completed = (output / "completion.json").exists()
        command = [sys.executable, "-m", "controller_architecture_study.train_controller",
                   "--config", str(config_path), "--output-dir", str(output), "--device", args.device]
        if saved_config.exists():
            command.append("--resume")
        print(f"{name}: {'skip complete' if completed else subprocess.list2cmdline(command)}", flush=True)
        if args.dry_run:
            continue
        if not completed:
            atomic_json(config_path, config)
            # One process/model. An error stops the queue, leaving saved epochs intact.
            subprocess.run(command, cwd=ROOT, check=True)
        metrics = json.loads((output / "offline_metrics.json").read_text(encoding="utf-8"))
        entry = {"model": name, "train": metrics["train"], "validation": metrics["validation"]}
        if args.grid == "capacity":
            cost = complexity(300, 4, layers, width)
            cost["fp32_weight_bytes"] = cost["parameter_count"] * 4
            completion = json.loads((output / "completion.json").read_text(encoding="utf-8"))
            entry.update(run=str(output), hidden_layers=layers, hidden_width=width, complexity=cost,
                         epochs_completed=completion["epochs_completed"], best_epoch=completion["best_epoch"])
        comparison.append(entry)
        atomic_json(comparison_path, {"test_split_evaluated": False, "models": comparison})
    if not args.dry_run:
        print(f"Comparison: {comparison_path}", flush=True)


if __name__ == "__main__":
    main()
