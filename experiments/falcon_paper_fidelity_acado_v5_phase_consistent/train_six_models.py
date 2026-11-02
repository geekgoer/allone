"""Train FALCON Table-II FC/ReLU models serially and resumably."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
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


def _complete(path: Path) -> bool:
    try:
        return bool(json.loads(path.read_text(encoding="utf-8")).get("training_complete"))
    except (OSError, ValueError):
        return False


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--data-dir", type=Path, default=here / "dnn_dataset_falcon12_v5_stable")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--seeds", type=_parse_seeds, default=(0,))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--min-epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=180)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    manifest = json.loads((data_dir / "split_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("input_contract") != EXPECTED_CONTRACT:
        raise ValueError(
            f"expected {EXPECTED_CONTRACT!r}, got {manifest.get('input_contract')!r}"
        )
    if int(manifest.get("paper_core_samples", 0)) != 40000:
        raise ValueError("the FALCON paper core must contain exactly 40,000 samples")
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else here / ("runs_pilot_stable" if args.mode == "pilot" else "runs_full_stable")
    )
    architectures = PILOT_ARCHITECTURES if args.mode == "pilot" else ARCHITECTURES
    for seed in args.seeds:
        for depth, width in architectures:
            name = f"seed_{seed}/depth{depth}_width{width}"
            output = output_root / name
            if _complete(output / "metrics.json"):
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
                "--learning-rate", str(args.learning_rate),
                "--min-epochs", str(args.min_epochs),
                "--patience", str(args.patience),
                "--early-stopping-min-delta", "2e-6",
                "--early-stop-after-min-lr",
                "--lr-plateau-patience", "45",
                "--lr-plateau-factor", "0.3",
                "--lr-plateau-threshold", "1e-4",
                "--lr-plateau-cooldown", "10",
                "--min-learning-rate", "1e-6",
                "--selection-metric", "normalized_mse",
                "--max-validation-baseline-ratio", "0.25",
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
            completed = subprocess.run(command, cwd=ROOT, check=False)
            if completed.returncode != 0:
                raise SystemExit(
                    f"training failed for {name} with exit code {completed.returncode}; "
                    "its latest checkpoint was preserved"
                )
            print(f"COMPLETE {name}: {output}", flush=True)
    print(f"All requested {args.mode} runs are complete: {output_root}", flush=True)


if __name__ == "__main__":
    main()
