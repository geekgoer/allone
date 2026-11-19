"""Train and export all six FALCON FC/ReLU models, serially and resumably."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
ARCHITECTURES = ((34, 1024), (22, 1024), (10, 1024), (34, 512), (22, 512), (10, 512))


def _metrics_complete(path: Path, epochs: int) -> bool:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return int(value.get("epochs_completed", 0)) >= epochs


def _run(command: list[str]) -> None:
    print("\nRUN:", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=here / "dnn_dataset_paper_core_40k")
    parser.add_argument("--runs-dir", type=Path, default=here / "runs_paper_core")
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--cooldown-seconds", type=float, default=30.0)
    parser.add_argument("--only", default=None, help="optional DEPTHxWIDTH, for example 10x512")
    parser.add_argument("--no-export", action="store_true")
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    runs_dir = args.runs_dir.resolve()
    with __import__("numpy").load(data_dir / "train.npz", allow_pickle=False) as archive:
        if archive["X"].shape[1:] != (12,) or archive["U"].shape[1:] != (4,):
            raise ValueError("FALCON paper models require 12-D inputs and 4-D labels")
    selected = list(ARCHITECTURES)
    if args.only:
        depth_text, width_text = args.only.lower().split("x", 1)
        requested = (int(depth_text), int(width_text))
        if requested not in selected:
            raise ValueError(f"unsupported architecture: {requested}")
        selected = [requested]
    status_path = runs_dir / "orchestration_status.json"
    runs_dir.mkdir(parents=True, exist_ok=True)
    status = {
        "protocol": "falcon_paper_six_fc_relu_v1",
        "serial": True,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "models": {},
    }
    if status_path.exists():
        try:
            status.update(json.loads(status_path.read_text(encoding="utf-8")))
        except ValueError:
            pass
    for index, (depth, width) in enumerate(selected):
        name = f"depth{depth}_width{width}"
        output = runs_dir / name
        metrics_path = output / "metrics.json"
        record = status.setdefault("models", {}).setdefault(name, {})
        record.update(started_at=datetime.now(timezone.utc).isoformat(), output=str(output))
        status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        if not _metrics_complete(metrics_path, args.epochs):
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
                "--patience", "0",
                "--device", args.device,
                "--workers", "0",
                "--seed", str(args.seed),
                "--skip-curve",
                "--max-validation-baseline-ratio", "0.25",
            ]
            if (output / "latest_checkpoint.pt").exists():
                command.append("--resume")
            try:
                _run(command)
            except subprocess.CalledProcessError as exc:
                record.update(status="failed", returncode=exc.returncode)
                status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
                raise
        else:
            print(f"SKIP {name}: already completed {args.epochs} epochs", flush=True)
        if not args.no_export:
            export_dir = output / "export"
            export_manifest = export_dir / "export_manifest.json"
            if not export_manifest.exists():
                _run(
                    [
                        sys.executable,
                        str(ROOT / "export_dnn_artifacts.py"),
                        "--model", str(output / "best_model.pt"),
                        "--data-dir", str(data_dir),
                        "--output-dir", str(export_dir),
                        "--golden-count", "1000",
                    ]
                )
        record.update(
            status="completed",
            completed_at=datetime.now(timezone.utc).isoformat(),
            metrics=str(metrics_path),
            export=str(output / "export"),
        )
        status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        if index + 1 < len(selected) and args.cooldown_seconds > 0.0:
            print(f"Cooling CUDA for {args.cooldown_seconds:.0f}s before the next model.", flush=True)
            time.sleep(args.cooldown_seconds)
    print(f"All requested models completed. Status: {status_path}")


if __name__ == "__main__":
    main()
