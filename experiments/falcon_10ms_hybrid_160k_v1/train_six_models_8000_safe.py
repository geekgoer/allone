"""Train all six hybrid-160k FALCON MLPs in a conservative, resumable order.

The intended Windows policy is deliberately simple: exactly one CUDA process
at a time, a modest batch, no DataLoader worker processes, no pinned host
memory, gradient clipping, and a fresh Python/CUDA process for every model.
Every training epoch writes ``latest_checkpoint.pt`` atomically; after an
interruption or a recoverable CUDA failure the same command resumes it.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
EXPECTED_CONTRACT = "falcon_track_p_trajectory_relative_12d_v1"

# The user specifically wants an operational small-versus-largest comparison
# first.  The remaining four models then run from lower to higher complexity.
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
        raise ValueError(
            f"expected {EXPECTED_CONTRACT!r}, got {manifest.get('input_contract')!r}"
        )
    if int(manifest.get("input_dimensions", 0)) != 12:
        raise ValueError("this six-model sweep requires exactly 12 input dimensions")
    for name in ("train", "validation", "test"):
        if not (data_dir / f"{name}.npz").exists():
            raise FileNotFoundError(f"missing {name}.npz in {data_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path,
        default=HERE / "dnn_dataset_hybrid_160k_10ms_v3_bucket_balanced",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=HERE / "runs_8000_safe_seed0",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-retries-per-model", type=int, default=2)
    parser.add_argument("--cooldown-seconds", type=float, default=90.0)
    args = parser.parse_args()

    if args.epochs != 8000:
        raise ValueError("this requested experiment is fixed at exactly 8000 epochs")
    if args.batch_size < 1 or args.max_retries_per_model < 0 or args.cooldown_seconds < 0.0:
        raise ValueError("batch size, retry count and cooldown must be non-negative")

    data_dir = args.data_dir.resolve()
    output_root = args.output_root.resolve()
    _verify_dataset(data_dir)

    failures: list[str] = []
    for depth, width in ARCHITECTURES:
        label = f"seed_{args.seed}/depth{depth}_width{width}"
        output_dir = output_root / label
        if _complete(output_dir):
            print(f"SKIP complete: {label}", flush=True)
            continue

        command = [
            sys.executable,
            str(ROOT / "train_dnn_controller.py"),
            "--data-dir", str(data_dir),
            "--output-dir", str(output_dir),
            "--depth", str(depth),
            "--width", str(width),
            "--epochs", "8000",
            "--batch-size", str(args.batch_size),
            "--learning-rate", str(args.learning_rate),
            # A bounded gradient and conservative data pipeline reduce the
            # chance of a Windows WDDM/TDR CUDA context loss.
            "--gradient-clip-norm", "1.0",
            "--workers", "0",
            "--disable-pin-memory",
            "--selection-metric", "normalized_mse",
            # The requested experiment is a fixed 8000-epoch, fixed-LR run:
            # no early stopping and no ReduceLROnPlateau schedule.
            "--min-epochs", "8000",
            "--patience", "0",
            "--lr-plateau-patience", "0",
            "--max-validation-baseline-ratio", "0.25",
            "--log-every", "20",
            "--seed", str(args.seed),
            "--device", args.device,
            "--skip-curve",
        ]
        if (output_dir / "latest_checkpoint.pt").exists():
            command.append("--resume")

        for attempt in range(args.max_retries_per_model + 1):
            mode = "RESUME" if "--resume" in command else "START"
            print(
                f"{mode} {label}; attempt={attempt + 1}/{args.max_retries_per_model + 1}; "
                "serial CUDA mode, fixed 1e-4 LR, checkpoint each epoch.",
                flush=True,
            )
            result = subprocess.run(command, cwd=ROOT, check=False)
            if result.returncode == 0:
                print(f"COMPLETE {label}: {output_dir}", flush=True)
                break
            if attempt >= args.max_retries_per_model:
                failures.append(label)
                print(
                    f"FAILED {label} after {attempt + 1} attempt(s); "
                    "latest_checkpoint.pt was preserved. Stopping before later models.",
                    flush=True,
                )
                break
            # A failed child process releases its CUDA context.  Always resume
            # from its atomic checkpoint after a substantial driver cooldown.
            if "--resume" not in command:
                command.append("--resume")
            gc.collect()
            print(f"Cooling down CUDA for {args.cooldown_seconds:.0f}s before resume.", flush=True)
            time.sleep(args.cooldown_seconds)
        if failures:
            break
        # Ensure the finished model's CUDA process is gone before starting the
        # next architecture.  This is also useful when the desktop GPU is busy.
        gc.collect()
        if args.cooldown_seconds > 0.0:
            print(f"Cooling down CUDA for {args.cooldown_seconds:.0f}s before next model.", flush=True)
            time.sleep(args.cooldown_seconds)

    if failures:
        raise SystemExit(f"training failed: {failures}")
    print(f"All six fixed-8000 runs completed: {output_root}", flush=True)


if __name__ == "__main__":
    main()
