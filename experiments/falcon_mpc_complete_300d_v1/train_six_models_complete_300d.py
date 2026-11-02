"""Train the six DNNs on the deployable, MPC-complete 300-D contract.

The runner is deliberately conservative:

* raw 300-D solver inputs are used;
* the learning rate stays constant;
* early stopping and LR scheduling are disabled;
* models run serially to avoid CUDA memory fragmentation;
* an existing latest checkpoint is resumed automatically.

The labels are the first control returned by the deterministic reset-and-
refined ACADO teacher.  The 300-D input is built by collect_300d.py and does
not contain an MPC optimization result.
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
PROJECT_ROOT = HERE.parents[1]
TRAINER = PROJECT_ROOT / "train_dnn_controller.py"

# Increasing model complexity makes the comparison easy to read.
ARCHITECTURES = (
    (10, 512),
    (10, 1024),
    (22, 512),
    (22, 1024),
    (34, 512),
    (34, 1024),
)


def _complete(run_dir: Path, *, expected_input_dim: int) -> bool:
    metrics_path = run_dir / "metrics.json"
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError):
        return False
    return (
        payload.get("training_complete") is True
        and int(payload.get("input_dimensions", -1)) == expected_input_dim
    )


def _validate_dataset(data_dir: Path) -> None:
    manifest_path = data_dir / "split_manifest.json"
    normalization_path = data_dir / "normalization.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing split manifest: {manifest_path}")
    if not normalization_path.is_file():
        raise FileNotFoundError(f"missing normalization: {normalization_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    input_dim = int(manifest.get("input_dimensions", -1))
    if input_dim != 300:
        raise ValueError(
            f"this runner requires the raw MPC-complete 300-D dataset, got {input_dim}D"
        )
    if manifest.get("not_environment_disjoint") is not True:
        raise ValueError(
            "expected the within-environment randomized split; "
            "refusing an environment-disjoint split for this run"
        )
    required_contract = "falcon_mpc_complete_solver_window_300d_v1"
    if manifest.get("input_contract") != required_contract:
        raise ValueError(
            f"unexpected input contract: {manifest.get('input_contract')!r}; "
            f"expected {required_contract!r}"
        )
    for split in ("train", "validation", "test"):
        path = data_dir / f"{split}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"missing dataset split: {path}")


def _command(
    *,
    data_dir: Path,
    run_dir: Path,
    depth: int,
    width: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: str,
    max_validation_baseline_ratio: float | None,
) -> list[str]:
    command = [
        sys.executable,
        str(TRAINER),
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(run_dir),
        "--depth",
        str(depth),
        "--width",
        str(width),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--learning-rate",
        f"{learning_rate:.12g}",
        "--gradient-clip-norm",
        "1.0",
        "--workers",
        "0",
        "--disable-pin-memory",
        "--selection-metric",
        "normalized_mse",
        "--patience",
        "0",
        "--min-epochs",
        str(epochs),
        "--lr-plateau-patience",
        "0",
        "--log-every",
        "20",
        "--seed",
        str(seed),
        "--device",
        device,
        "--skip-curve",
    ]
    if max_validation_baseline_ratio is not None:
        command.extend(
            ["--max-validation-baseline-ratio", str(max_validation_baseline_ratio)]
        )
    if (run_dir / "latest_checkpoint.pt").is_file():
        command.append("--resume")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=HERE / "dnn_dataset_300d_140k_v1_within_environment",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=HERE / "runs_complete_300d_raw_seed0",
    )
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cooldown-seconds", type=float, default=20.0)
    parser.add_argument(
        "--max-validation-baseline-ratio",
        type=float,
        default=None,
        help="optional health check; omit to avoid stopping a completed run",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the six commands without starting training",
    )
    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be positive")
    if args.cooldown_seconds < 0.0:
        raise ValueError("--cooldown-seconds must be non-negative")

    data_dir = args.data_dir.resolve()
    output_root = args.output_root.resolve()
    _validate_dataset(data_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {data_dir}", flush=True)
    print(f"Output:  {output_root}", flush=True)
    print(
        f"Policy: raw 300-D input | epochs={args.epochs} | batch={args.batch_size} "
        f"| fixed lr={args.learning_rate:.3e} | seed={args.seed}",
        flush=True,
    )

    for depth, width in ARCHITECTURES:
        label = f"seed_{args.seed}/depth{depth}_width{width}"
        run_dir = output_root / f"seed_{args.seed}" / f"depth{depth}_width{width}"
        if _complete(run_dir, expected_input_dim=300):
            print(f"SKIP complete: {label}", flush=True)
            continue

        command = _command(
            data_dir=data_dir,
            run_dir=run_dir,
            depth=depth,
            width=width,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            device=args.device,
            max_validation_baseline_ratio=args.max_validation_baseline_ratio,
        )
        resumed = "--resume" in command
        print(
            f"{'RESUME' if resumed else 'START'} {label}; "
            f"fixed {args.epochs} epochs, fixed LR.",
            flush=True,
        )
        if args.dry_run:
            print(" ".join(f'"{item}"' if " " in item else item for item in command))
            continue

        result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
        if result.returncode != 0:
            raise SystemExit(
                f"FAILED {label} with exit code {result.returncode}; "
                f"checkpoint retained at {run_dir}. Re-run this same command to resume."
            )
        gc.collect()
        if args.cooldown_seconds:
            print(
                f"Cooling down CUDA for {args.cooldown_seconds:.0f}s.",
                flush=True,
            )
            time.sleep(args.cooldown_seconds)

    print(f"COMPLETE: {output_root}", flush=True)


if __name__ == "__main__":
    main()
