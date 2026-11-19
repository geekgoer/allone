"""Train six 12-D DNN controllers on the 200k launch-augmented dataset.

The six models run serially from small to large.  Each model has a fixed
epoch budget, fixed learning rate, no early stopping, and an isolated output
directory.  Re-running the same command resumes the model's own checkpoint.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

EXPECTED_STRATEGY = "base_160k_plus_real_zero_speed_launch_rollouts_measured_start_v2"
EXPECTED_CONTRACT = "falcon_track_p_trajectory_relative_12d_v1"

RUNS = (
    (10, 512, 200),
    (22, 512, 300),
    (10, 1024, 400),
    (34, 512, 500),
    (22, 1024, 600),
    (34, 1024, 700),
)


def _completed(path: Path, epochs: int) -> bool:
    try:
        metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
        return bool(metrics.get("training_complete")) and int(
            metrics.get("epochs_completed", 0)
        ) >= epochs
    except (OSError, ValueError, TypeError):
        return False


def _verify_dataset(data_dir: Path) -> None:
    manifest_path = data_dir / "split_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing dataset manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("strategy") != EXPECTED_STRATEGY:
        raise ValueError(
            "unexpected dataset strategy: "
            f"{manifest.get('strategy')!r}; expected {EXPECTED_STRATEGY!r}"
        )
    if manifest.get("input_contract") != EXPECTED_CONTRACT:
        raise ValueError(
            f"expected input contract {EXPECTED_CONTRACT!r}, "
            f"got {manifest.get('input_contract')!r}"
        )
    if int(manifest.get("input_dimensions", 0)) != 12:
        raise ValueError("this training sweep requires exactly 12 input dimensions")
    if int(manifest.get("total_samples", 0)) != 200000:
        raise ValueError("this training sweep requires exactly 200000 samples")
    if not bool(manifest.get("environment_leakage") is False):
        raise ValueError("dataset manifest does not certify environment leakage=False")
    for split in ("train", "validation", "test"):
        if not (data_dir / f"{split}.npz").is_file():
            raise FileNotFoundError(f"missing {split}.npz in {data_dir}")
    if not (data_dir / "normalization.json").is_file():
        raise FileNotFoundError(f"missing normalization.json in {data_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=HERE / "dnn_dataset_v5_launch_augmented_200k_v1",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=HERE / "runs_launch_augmented_progressive_200_to_700_batch512_seed0",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cooldown-seconds", type=float, default=60.0)
    args = parser.parse_args()

    if args.cooldown_seconds < 0.0:
        raise ValueError("--cooldown-seconds must be non-negative")

    data_dir = args.data_dir.resolve()
    output_root = args.output_root.resolve()
    _verify_dataset(data_dir)

    for depth, width, epochs in RUNS:
        label = f"seed_{args.seed}/depth{depth}_width{width}"
        output_dir = output_root / label
        if _completed(output_dir, epochs):
            print(f"SKIP complete: {label}", flush=True)
            continue

        command = [
            sys.executable,
            str(ROOT / "train_dnn_controller.py"),
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(output_dir),
            "--depth",
            str(depth),
            "--width",
            str(width),
            "--epochs",
            str(epochs),
            "--batch-size",
            "512",
            "--learning-rate",
            "1e-4",
            "--gradient-clip-norm",
            "1.0",
            "--workers",
            "0",
            "--disable-pin-memory",
            "--selection-metric",
            "normalized_mse",
            "--min-epochs",
            str(epochs),
            "--patience",
            "0",
            "--lr-plateau-patience",
            "0",
            "--max-validation-baseline-ratio",
            "0.25",
            "--log-every",
            "20",
            "--seed",
            str(args.seed),
            "--device",
            args.device,
            "--skip-curve",
        ]
        if (output_dir / "latest_checkpoint.pt").is_file():
            command.append("--resume")

        mode = "RESUME" if "--resume" in command else "START"
        print(
            f"{mode} {label}; fixed epochs={epochs}, batch=512, "
            "seed={0}, LR=1e-4, early stopping disabled.".format(args.seed),
            flush=True,
        )
        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode != 0:
            raise SystemExit(
                f"FAILED {label}; checkpoint retained. "
                "Re-run the same command to resume."
            )

        gc.collect()
        if args.cooldown_seconds > 0.0:
            print(
                f"Cooling down CUDA for {args.cooldown_seconds:.0f}s.",
                flush=True,
            )
            time.sleep(args.cooldown_seconds)

    print(f"COMPLETE: {output_root}", flush=True)


if __name__ == "__main__":
    main()
