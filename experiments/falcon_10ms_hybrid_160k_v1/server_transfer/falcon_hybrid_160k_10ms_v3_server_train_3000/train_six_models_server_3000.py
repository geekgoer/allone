"""Portable Linux-server training runner for the six 10 m/s hybrid DNNs.

This is deliberately separate from the Windows 8000-epoch recovery runner:
it requires the completed v3 bucket-balanced 160k data, trains exactly 3000
epochs per model, keeps the learning rate fixed at 1e-4, and disables both
early stopping and ReduceLROnPlateau.  Each architecture runs in a fresh
child process so that a failure preserves ``latest_checkpoint.pt`` and does
not contaminate the next CUDA process.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
# In the repository this file is nested below ``experiments/``.  The transfer
# package intentionally flattens just the required training files, so fall
# back to the containing directory when used from that package.
_repository_root = HERE.parents[1]
ROOT = _repository_root if (_repository_root / "train_dnn_controller.py").is_file() else HERE
EXPECTED_CONTRACT = "falcon_track_p_trajectory_relative_12d_v1"

# Start with the smallest and the largest capacity; then run the remaining
# four configurations in increasing nominal size.
ARCHITECTURES = (
    (10, 512),
    (34, 1024),
    (10, 1024),
    (22, 512),
    (22, 1024),
    (34, 512),
)


def _complete(directory: Path) -> bool:
    try:
        return bool(json.loads((directory / "metrics.json").read_text(encoding="utf-8")).get("training_complete"))
    except (OSError, ValueError):
        return False


def _verify_dataset(data_dir: Path) -> None:
    manifest_path = data_dir / "split_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"prepared split dataset not found: {data_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("input_contract") != EXPECTED_CONTRACT:
        raise ValueError(f"wrong input contract: {manifest.get('input_contract')!r}")
    if int(manifest.get("input_dimensions", 0)) != 12:
        raise ValueError("server six-model run requires X=(N,12)")
    for split in ("train", "validation", "test"):
        if not (data_dir / f"{split}.npz").is_file():
            raise FileNotFoundError(f"missing {split}.npz in {data_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=HERE / "dnn_dataset_hybrid_160k_10ms_v3_bucket_balanced")
    parser.add_argument("--output-root", type=Path, default=HERE / "runs_server_3000_seed0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-retries-per-model", type=int, default=1)
    parser.add_argument("--cooldown-seconds", type=float, default=15.0)
    args = parser.parse_args()

    if args.epochs != 3000:
        raise ValueError("this server experiment is fixed at exactly 3000 epochs")
    if args.batch_size < 1 or args.workers < 0 or args.max_retries_per_model < 0:
        raise ValueError("batch size, workers and retries must be non-negative")

    data_dir = args.data_dir.resolve()
    output_root = args.output_root.resolve()
    _verify_dataset(data_dir)

    for depth, width in ARCHITECTURES:
        label = f"seed_{args.seed}/depth{depth}_width{width}"
        output_dir = output_root / label
        if _complete(output_dir):
            print(f"SKIP complete: {label}", flush=True)
            continue
        command = [
            sys.executable, str(ROOT / "train_dnn_controller.py"),
            "--data-dir", str(data_dir),
            "--output-dir", str(output_dir),
            "--depth", str(depth), "--width", str(width),
            "--epochs", "3000", "--batch-size", str(args.batch_size),
            "--workers", str(args.workers), "--learning-rate", str(args.learning_rate),
            "--gradient-clip-norm", "1.0",
            "--selection-metric", "normalized_mse",
            "--min-epochs", "3000", "--patience", "0",
            "--lr-plateau-patience", "0",
            "--max-validation-baseline-ratio", "0.25",
            "--log-every", "20", "--seed", str(args.seed),
            "--device", args.device, "--skip-curve",
        ]
        if (output_dir / "latest_checkpoint.pt").exists():
            command.append("--resume")

        completed = False
        for attempt in range(args.max_retries_per_model + 1):
            print(
                f"{'RESUME' if '--resume' in command else 'START'} {label}; "
                f"attempt={attempt + 1}/{args.max_retries_per_model + 1}; "
                f"batch={args.batch_size}, workers={args.workers}, fixed LR=1e-4, epochs=3000.",
                flush=True,
            )
            result = subprocess.run(command, cwd=ROOT, check=False)
            if result.returncode == 0:
                completed = True
                print(f"COMPLETE {label}: {output_dir}", flush=True)
                break
            if "--resume" not in command:
                command.append("--resume")
            if attempt < args.max_retries_per_model:
                print(f"Child failed; preserving checkpoint and cooling down {args.cooldown_seconds:.0f}s.", flush=True)
                time.sleep(args.cooldown_seconds)
        if not completed:
            raise SystemExit(f"FAILED {label}; checkpoint retained at {output_dir / 'latest_checkpoint.pt'}")
        if args.cooldown_seconds > 0:
            time.sleep(args.cooldown_seconds)

    print(f"All six fixed-3000 server runs completed: {output_root}", flush=True)


if __name__ == "__main__":
    main()
