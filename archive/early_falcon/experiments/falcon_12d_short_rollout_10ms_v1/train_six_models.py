"""Train the six FALCON MLP sizes on the 10 m/s short-rollout dataset.

Runs are serial and resumable.  A completed model is never retrained, and an
interrupted model resumes from ``latest_checkpoint.pt`` with optimizer and
learning-rate-scheduler state intact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

# Ordering is intentionally small-to-large within the two widths.  It makes
# capacity comparisons easy while retaining the six architectures in Table II.
ARCHITECTURES = (
    (10, 512),
    (10, 1024),
    (22, 512),
    (22, 1024),
    (34, 512),
    (34, 1024),
)
PILOT_ARCHITECTURES = ((10, 512), (34, 1024))


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("--seeds must contain at least one integer")
    return seeds


def _complete(directory: Path) -> bool:
    path = directory / "metrics.json"
    if not path.exists():
        return False
    try:
        return bool(json.loads(path.read_text(encoding="utf-8")).get("training_complete"))
    except (OSError, ValueError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--seeds", type=_parse_seeds, default=(0,))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-dir", type=Path, default=HERE / "dnn_dataset_short_rollout_10ms_120k")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    # Windows WDDM shares the RTX 4060 with the desktop compositor.  512
    # keeps even the 34x1024 model well below the long-kernel/TDR regime while
    # retaining a sufficiently large batch for this fully connected workload.
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--min-epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=150)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-5)
    args = parser.parse_args()

    if not (args.data_dir / "split_manifest.json").exists():
        raise FileNotFoundError(
            f"prepared split dataset not found: {args.data_dir}. "
            "Run prepare_short_rollout_dataset.py first."
        )

    # Keep both the quick capacity pilot and full six-model sweep bounded by
    # the same practical 800-epoch budget.  Plateau scheduling and early
    # stopping may finish a model earlier.
    max_epochs = args.epochs if args.epochs is not None else 800
    architectures = PILOT_ARCHITECTURES if args.mode == "pilot" else ARCHITECTURES
    # Both modes share a root so a completed pilot checkpoint is reused by the
    # full sweep instead of being needlessly trained a second time.
    output_root = args.output_root or HERE / "runs"
    failures: list[str] = []

    for seed in args.seeds:
        for depth, width in architectures:
            label = f"seed_{seed}/depth{depth}_width{width}"
            output_dir = output_root / label
            if _complete(output_dir):
                print(f"SKIP complete: {label}", flush=True)
                continue

            command = [
                sys.executable,
                str(ROOT / "train_dnn_controller.py"),
                "--data-dir", str(args.data_dir),
                "--output-dir", str(output_dir),
                "--depth", str(depth),
                "--width", str(width),
                "--epochs", str(max_epochs),
                "--batch-size", str(args.batch_size),
                "--learning-rate", str(args.learning_rate),
                "--gradient-clip-norm", str(args.gradient_clip_norm),
                "--min-epochs", str(args.min_epochs),
                "--patience", str(args.patience),
                "--early-stopping-min-delta", str(args.early_stopping_min_delta),
                "--early-stop-after-min-lr",
                "--lr-plateau-patience", "40",
                "--lr-plateau-factor", "0.3",
                "--lr-plateau-threshold", "1e-4",
                "--lr-plateau-cooldown", "10",
                "--min-learning-rate", "1e-6",
                "--selection-metric", "normalized_mse",
                "--max-validation-baseline-ratio", "0.25",
                "--seed", str(seed),
                "--device", args.device,
                "--workers", str(args.workers),
                "--disable-pin-memory",
                "--skip-curve",
            ]
            if (output_dir / "latest_checkpoint.pt").exists():
                command.append("--resume")
                print(f"RESUME {label}", flush=True)
            else:
                print(f"START {label}", flush=True)

            completed = subprocess.run(command, cwd=ROOT, check=False)
            if completed.returncode != 0:
                failures.append(label)
                print(
                    f"FAILED {label} exit={completed.returncode}; latest checkpoint is preserved. "
                    "Stopping before the next model.",
                    flush=True,
                )
                break
            print(f"COMPLETE {label}: {output_dir}", flush=True)
        if failures:
            break

    if failures:
        raise SystemExit(f"training failed: {failures}")
    print(f"All requested {args.mode} runs completed: {output_root}", flush=True)


if __name__ == "__main__":
    main()
