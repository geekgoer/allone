"""Local GPU pilot: train only the smallest and largest models for 300 epochs.

This is a pre-server smoke/performance run.  It uses the completed 160k v3
split, fixed 1e-4 learning rate, no early stopping and no LR scheduler.  The
two models run in separate processes so each writes its own recoverable
checkpoint without retaining the previous model's CUDA allocations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[1]
EXPECTED_CONTRACT = "falcon_track_p_trajectory_relative_12d_v1"
ARCHITECTURES = ((10, 512), (34, 1024))


def _complete(directory: Path) -> bool:
    try:
        return bool(json.loads((directory / "metrics.json").read_text(encoding="utf-8")).get("training_complete"))
    except (OSError, ValueError):
        return False


def _verify_dataset(data_dir: Path) -> None:
    manifest = json.loads((data_dir / "split_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("input_contract") != EXPECTED_CONTRACT or int(manifest.get("input_dimensions", 0)) != 12:
        raise ValueError("pilot requires the prepared 12-D hybrid-v3 split")
    for split in ("train", "validation", "test"):
        if not (data_dir / f"{split}.npz").is_file():
            raise FileNotFoundError(f"missing {split}.npz")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=HERE / "dnn_dataset_hybrid_160k_10ms_v4_environment_balanced")
    parser.add_argument("--output-root", type=Path, default=HERE / "runs_local_pilot_300_seed0")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cooldown-seconds", type=float, default=60.0)
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    data_dir, output_root = args.data_dir.resolve(), args.output_root.resolve()
    _verify_dataset(data_dir)

    for depth, width in ARCHITECTURES:
        label = f"seed_{args.seed}/depth{depth}_width{width}"
        output = output_root / label
        if _complete(output):
            print(f"SKIP complete: {label}", flush=True)
            continue
        command = [
            sys.executable, str(REPOSITORY_ROOT / "train_dnn_controller.py"),
            "--data-dir", str(data_dir), "--output-dir", str(output),
            "--depth", str(depth), "--width", str(width),
            "--epochs", "300", "--batch-size", str(args.batch_size),
            "--learning-rate", "1e-4", "--gradient-clip-norm", "1.0",
            "--workers", "0", "--disable-pin-memory",
            "--selection-metric", "normalized_mse",
            "--min-epochs", "300", "--patience", "0", "--lr-plateau-patience", "0",
            "--max-validation-baseline-ratio", "0.25", "--log-every", "10",
            "--seed", str(args.seed), "--device", args.device, "--skip-curve",
        ]
        if (output / "latest_checkpoint.pt").exists():
            command.append("--resume")
        print(f"{'RESUME' if '--resume' in command else 'START'} {label}; fixed 300 epochs, batch={args.batch_size}.", flush=True)
        result = subprocess.run(command, cwd=REPOSITORY_ROOT, check=False)
        if result.returncode:
            raise SystemExit(f"pilot failed for {label}; checkpoint retained in {output}")
        print(f"COMPLETE {label}: {output}", flush=True)
        if args.cooldown_seconds > 0:
            print(f"Cooling down CUDA for {args.cooldown_seconds:.0f}s.", flush=True)
            time.sleep(args.cooldown_seconds)

    print(f"Local 300-epoch pilot complete: {output_root}", flush=True)


if __name__ == "__main__":
    main()
