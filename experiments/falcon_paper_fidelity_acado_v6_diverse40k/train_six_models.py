"""Train the v6 pilot or all six FALCON architectures safely and resumably."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
GENERIC = (
    ROOT
    / "experiments"
    / "falcon_paper_fidelity_acado_v5_phase_consistent"
    / "train_six_models_paper40k.py"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=500)
    parser.add_argument("--min-epochs", type=int, default=1000)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-5)
    args = parser.parse_args()
    epochs = args.epochs if args.epochs is not None else (1000 if args.mode == "pilot" else 20000)
    output_root = HERE / ("runs_pilot" if args.mode == "pilot" else "runs_full")
    command = [
        sys.executable,
        str(GENERIC),
        "--mode", args.mode,
        "--data-dir", str(HERE / "dnn_dataset_falcon12_v6_diverse40k"),
        "--output-root", str(output_root),
        "--seeds", args.seeds,
        "--device", args.device,
        "--epochs", str(epochs),
        "--batch-size", str(args.batch_size),
        "--workers", str(args.workers),
        "--gradient-clip-norm", str(args.gradient_clip_norm),
        "--patience", str(args.patience),
        "--min-epochs", str(args.min_epochs),
        "--early-stopping-min-delta", str(args.early_stopping_min_delta),
        "--disable-pin-memory",
    ]
    completed = subprocess.run(command, cwd=ROOT, check=False)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
