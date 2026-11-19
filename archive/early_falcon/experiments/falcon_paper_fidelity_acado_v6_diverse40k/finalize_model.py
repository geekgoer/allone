"""Finalize one interrupted v6 run from its existing best checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=int, choices=(10, 22, 34), required=True)
    parser.add_argument("--width", type=int, choices=(512, 1024), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output = (
        HERE
        / "runs_full"
        / f"seed_{args.seed}"
        / f"depth{args.depth}_width{args.width}"
    )
    required = (output / "best_model.pt", output / "latest_checkpoint.pt")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"cannot finalize; missing: {missing}")
    command = [
        sys.executable,
        str(ROOT / "train_dnn_controller.py"),
        "--data-dir", str(HERE / "dnn_dataset_falcon12_v6_diverse40k"),
        "--output-dir", str(output),
        "--depth", str(args.depth),
        "--width", str(args.width),
        "--epochs", "20000",
        "--batch-size", "2048",
        "--learning-rate", "1e-4",
        "--gradient-clip-norm", "1.0",
        "--patience", "0",
        "--lr-plateau-patience", "0",
        "--selection-metric", "normalized_mse",
        "--seed", str(args.seed),
        "--device", args.device,
        "--workers", "0",
        "--skip-curve",
        "--resume",
        "--finalize-best",
    ]
    completed = subprocess.run(command, cwd=ROOT, check=False)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
