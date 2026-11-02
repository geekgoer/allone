"""Train the six FALCON Table-II architectures on the paper-only 40k dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ARCHITECTURES = (
    (10, 512),
    (10, 1024),
    (22, 512),
    (22, 1024),
    (34, 512),
    (34, 1024),
)
PILOT_ARCHITECTURES = ((10, 512), (34, 1024))
EXPECTED_CONTRACT = "falcon_paper_gate_relative_euler_12d_v1"


def _parse_seeds(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return result


def _complete(path: Path, target_epochs: int) -> bool:
    try:
        metrics = json.loads(path.read_text(encoding="utf-8"))
        if not bool(metrics.get("training_complete")):
            return False
        reason = str(metrics.get("termination_reason", ""))
        return int(metrics.get("epochs_completed", 0)) >= target_epochs or reason in {
            "early_stopping",
            "manual_finalize_best",
        }
    except (OSError, ValueError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "full"), default="pilot")
    parser.add_argument(
        "--data-dir", type=Path,
        default=HERE / "dnn_dataset_falcon12_v5_paper40k_rowsplit",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--seeds", type=_parse_seeds, default=(0,))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--gradient-clip-norm", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--min-epochs", type=int, default=0)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--disable-pin-memory", action="store_true")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    manifest = json.loads((data_dir / "split_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("input_contract") != EXPECTED_CONTRACT:
        raise ValueError(f"unexpected input contract: {manifest.get('input_contract')!r}")
    if int(manifest.get("total_samples", 0)) != 40000:
        raise ValueError("paper-facing training requires exactly 40,000 samples")
    if int(manifest.get("robustness_extension_samples", -1)) != 0:
        raise ValueError("paper-facing training must not include recovery rows")

    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else HERE / ("runs_paper40k_pilot" if args.mode == "pilot" else "runs_paper40k_full")
    )
    architectures = PILOT_ARCHITECTURES if args.mode == "pilot" else ARCHITECTURES
    for seed in args.seeds:
        for depth, width in architectures:
            name = f"seed_{seed}/depth{depth}_width{width}"
            output = output_root / name
            if _complete(output / "metrics.json", args.epochs):
                print(f"SKIP complete: {name}", flush=True)
                continue
            command = [
                sys.executable,
                str(ROOT / "train_dnn_controller.py"),
                "--data-dir", str(data_dir),
                "--output-dir", str(output),
                "--depth", str(depth),
                "--width", str(width),
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
                "--learning-rate", "1e-4",
                "--gradient-clip-norm", str(args.gradient_clip_norm),
                "--patience", str(args.patience),
                "--min-epochs", str(args.min_epochs),
                "--early-stopping-min-delta", str(args.early_stopping_min_delta),
                "--lr-plateau-patience", "0",
                "--selection-metric", "normalized_mse",
                "--seed", str(seed),
                "--device", args.device,
                "--workers", str(args.workers),
                "--skip-curve",
            ]
            if (output / "latest_checkpoint.pt").exists():
                command.append("--resume")
                print(f"RESUME {name}", flush=True)
            else:
                print(f"START {name}", flush=True)
            if args.disable_pin_memory:
                command.append("--disable-pin-memory")
            completed = subprocess.run(command, cwd=ROOT, check=False)
            if completed.returncode != 0:
                raise SystemExit(
                    f"training failed for {name} with exit code {completed.returncode}; "
                    "latest_checkpoint.pt was preserved"
                )
            print(f"COMPLETE {name}: {output}", flush=True)
    print(f"All requested {args.mode} paper40k runs are complete: {output_root}", flush=True)


if __name__ == "__main__":
    main()
