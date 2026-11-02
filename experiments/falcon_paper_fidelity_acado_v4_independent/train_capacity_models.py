"""Train the FALCON architecture pilot or all six models, serially and resumably."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
ARCHITECTURES = (
    (34, 1024),
    (22, 1024),
    (10, 1024),
    (34, 512),
    (22, 512),
    (10, 512),
)
PILOT_ARCHITECTURES = ((10, 512), (34, 1024))


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def _is_complete(output_dir: Path) -> bool:
    metrics_path = output_dir / "metrics.json"
    if not metrics_path.exists():
        return False
    try:
        return bool(json.loads(metrics_path.read_text(encoding="utf-8")).get("training_complete"))
    except (OSError, ValueError):
        return False


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--data-dir", type=Path, default=here / "dnn_dataset")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--seeds", type=_parse_seeds, default=(0, 1, 2))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--epochs", type=int, default=500,
        help="maximum epochs; large state-query datasets should not use legacy 20k-epoch budgets",
    )
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--min-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--skip-curve", action="store_true")
    args = parser.parse_args()

    output_root = args.output_root or here / ("runs_pilot" if args.mode == "pilot" else "runs_full")
    architectures = PILOT_ARCHITECTURES if args.mode == "pilot" else ARCHITECTURES
    failures: list[str] = []
    for seed in args.seeds:
        for depth, width in architectures:
            name = f"seed_{seed}/depth{depth}_width{width}"
            output_dir = output_root / name
            if _is_complete(output_dir):
                print(f"SKIP complete: {name}", flush=True)
                continue
            command = [
                sys.executable,
                str(ROOT / "train_dnn_controller.py"),
                "--data-dir", str(args.data_dir),
                "--output-dir", str(output_dir),
                "--depth", str(depth),
                "--width", str(width),
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
                "--learning-rate", str(args.learning_rate),
                "--patience", str(args.patience),
                "--min-epochs", str(args.min_epochs),
                "--early-stopping-min-delta", "1e-6",
                "--lr-plateau-patience", "25",
                "--lr-plateau-factor", "0.3",
                "--lr-plateau-threshold", "1e-4",
                "--min-learning-rate", "1e-6",
                "--early-stop-after-min-lr",
                "--selection-metric", "normalized_mse",
                "--max-validation-baseline-ratio", "0.25",
                "--seed", str(seed),
                "--device", args.device,
                "--workers", str(args.workers),
            ]
            if args.skip_curve:
                command.append("--skip-curve")
            if (output_dir / "latest_checkpoint.pt").exists():
                command.append("--resume")
                print(f"RESUME {name}", flush=True)
            else:
                print(f"START {name}", flush=True)
            result = subprocess.run(command, cwd=ROOT, check=False)
            if result.returncode != 0:
                failures.append(name)
                print(
                    f"FAILED {name} exit={result.returncode}; checkpoint was preserved. "
                    "Stop now so the problem can be fixed before later models.",
                    flush=True,
                )
                break
            print(f"COMPLETE {name}: {output_dir}", flush=True)
        if failures:
            break
    if failures:
        raise SystemExit(f"training failed: {failures}")
    print(f"All requested {args.mode} runs are complete: {output_root}", flush=True)


if __name__ == "__main__":
    main()
