"""Train the six FC/ReLU architectures published in FALCON Table II.

Models run serially for GPU stability.  Every epoch writes a latest checkpoint,
and every validation improvement writes the best model.  Re-running this
command resumes an interrupted model and skips completed models.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARCHITECTURES = (
    (10, 512),
    (10, 1024),
    (22, 512),
    (22, 1024),
    (34, 512),
    (34, 1024),
)
EXPECTED_CONTRACT = "falcon_paper_gate_relative_euler_12d_v1"


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_manifest(data_dir: Path) -> dict:
    path = data_dir / "split_manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist; run prepare_paper_gate_12d.py first"
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("input_contract") != EXPECTED_CONTRACT:
        raise ValueError(
            f"expected input contract {EXPECTED_CONTRACT!r}, got "
            f"{manifest.get('input_contract')!r}"
        )
    for split, expected in (("train", 32000), ("validation", 4000), ("test", 4000)):
        with np.load(data_dir / f"{split}.npz", allow_pickle=False) as archive:
            if archive["X"].shape != (expected, 12) or archive["U"].shape != (expected, 4):
                raise ValueError(f"{split} split has unexpected X/U shapes")
    return manifest


def _metrics_complete(path: Path, epochs: int) -> bool:
    try:
        metrics = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return bool(metrics.get("training_complete", False)) or int(
        metrics.get("epochs_completed", 0)
    ) >= epochs


def _run(command: list[str]) -> None:
    print("\nRUN:", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=HERE / "dnn_dataset_paper_gate_12d_euler"
    )
    parser.add_argument(
        "--runs-dir", type=Path, default=HERE / "runs_paper_gate_12d_euler"
    )
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=1200)
    parser.add_argument("--min-epochs", type=int, default=1500)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-6)
    parser.add_argument("--lr-plateau-patience", type=int, default=400)
    parser.add_argument("--lr-plateau-factor", type=float, default=0.3)
    parser.add_argument("--lr-plateau-threshold", type=float, default=1e-4)
    parser.add_argument("--lr-plateau-cooldown", type=int, default=50)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--cooldown-seconds", type=float, default=60.0)
    parser.add_argument("--only", help="optional architecture such as 10x512")
    parser.add_argument("--no-export", action="store_true")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    runs_dir = args.runs_dir.resolve()
    manifest = _load_manifest(data_dir)
    selected = list(ARCHITECTURES)
    if args.only:
        depth_text, width_text = args.only.lower().split("x", 1)
        requested = (int(depth_text), int(width_text))
        if requested not in ARCHITECTURES:
            raise ValueError(f"unsupported FALCON architecture: {requested}")
        selected = [requested]

    runs_dir.mkdir(parents=True, exist_ok=True)
    status_path = runs_dir / "orchestration_status.json"
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        status = {
            "protocol": "falcon_table_ii_gate_relative_euler_12d_v1",
            "serial": True,
            "input_contract": EXPECTED_CONTRACT,
            "data_dir": str(data_dir),
            "split_unit": manifest.get("split_unit"),
            "models": {},
        }
    # Refresh the requested policy on every invocation.  This makes the status
    # file truthful when an older run directory is continued with early
    # stopping enabled.
    status.update(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        early_stopping={
            "patience": args.patience,
            "minimum_epochs": args.min_epochs,
            "minimum_absolute_improvement": args.early_stopping_min_delta,
            "requires_minimum_learning_rate": True,
        },
        learning_rate_schedule={
            "type": "reduce_on_plateau",
            "patience": args.lr_plateau_patience,
            "factor": args.lr_plateau_factor,
            "relative_threshold": args.lr_plateau_threshold,
            "cooldown": args.lr_plateau_cooldown,
            "minimum_learning_rate": args.min_learning_rate,
        },
    )

    for index, (depth, width) in enumerate(selected):
        name = f"depth{depth}_width{width}"
        output = runs_dir / name
        metrics_path = output / "metrics.json"
        record = status.setdefault("models", {}).setdefault(name, {})
        record.update(
            status="running",
            started_at=datetime.now(timezone.utc).isoformat(),
            output=str(output),
        )
        _write_json(status_path, status)

        if not _metrics_complete(metrics_path, args.epochs):
            command = [
                sys.executable,
                str(ROOT / "train_dnn_controller.py"),
                "--data-dir",
                str(data_dir),
                "--output-dir",
                str(output),
                "--depth",
                str(depth),
                "--width",
                str(width),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--learning-rate",
                str(args.learning_rate),
                "--patience",
                str(args.patience),
                "--min-epochs",
                str(args.min_epochs),
                "--early-stopping-min-delta",
                str(args.early_stopping_min_delta),
                "--early-stop-after-min-lr",
                "--lr-plateau-patience",
                str(args.lr_plateau_patience),
                "--lr-plateau-factor",
                str(args.lr_plateau_factor),
                "--lr-plateau-threshold",
                str(args.lr_plateau_threshold),
                "--lr-plateau-cooldown",
                str(args.lr_plateau_cooldown),
                "--min-learning-rate",
                str(args.min_learning_rate),
                "--device",
                args.device,
                "--workers",
                "0",
                "--seed",
                str(args.seed),
                "--skip-curve",
            ]
            if (output / "latest_checkpoint.pt").exists():
                command.append("--resume")
            try:
                _run(command)
            except subprocess.CalledProcessError as exc:
                record.update(status="failed", returncode=exc.returncode)
                _write_json(status_path, status)
                raise
        else:
            print(f"SKIP {name}: training is already complete", flush=True)

        if not args.no_export:
            export = output / "export"
            if not (export / "export_manifest.json").exists():
                _run(
                    [
                        sys.executable,
                        str(ROOT / "export_dnn_artifacts.py"),
                        "--model",
                        str(output / "best_model.pt"),
                        "--data-dir",
                        str(data_dir),
                        "--output-dir",
                        str(export),
                        "--golden-count",
                        "1000",
                    ]
                )
        record.update(
            status="completed",
            completed_at=datetime.now(timezone.utc).isoformat(),
            metrics=str(metrics_path),
            export=str(output / "export"),
        )
        _write_json(status_path, status)
        print(f"COMPLETE {name}: {output}", flush=True)
        if index + 1 < len(selected) and args.cooldown_seconds > 0.0:
            print(f"Cooling down CUDA for {args.cooldown_seconds:.0f}s.", flush=True)
            time.sleep(args.cooldown_seconds)

    print(f"All requested models completed. Status: {status_path}", flush=True)


if __name__ == "__main__":
    main()
