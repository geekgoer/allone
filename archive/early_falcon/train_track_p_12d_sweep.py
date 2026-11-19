"""Train the six Track-P 12-D FALCON networks safely, one at a time.

Each architecture owns an independent output directory. Completed models are
skipped, partial models resume from their latest epoch checkpoint, and a CUDA
out-of-memory failure is retried with half the batch size. Progress is written
after every attempt so a later failure cannot invalidate earlier models.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ARCHITECTURES = (
    (10, 512),
    (10, 1024),
    (22, 512),
    (22, 1024),
    (34, 512),
    (34, 1024),
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _complete(output_dir: Path) -> bool:
    return all(
        (output_dir / name).exists()
        for name in ("best_model.pt", "controller_scripted.pt", "metrics.json")
    )


def _run_and_tee(command: list[str], log_path: Path) -> tuple[int, str]:
    captured: list[str] = []
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        quoted = subprocess.list2cmdline(command)
        header = f"\n[{datetime.now(timezone.utc).isoformat()}] {quoted}\n"
        print(header, end="", flush=True)
        log.write(header)
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            captured.append(line)
            # OOM text is near the end, so retaining the last 4000 lines is
            # sufficient without keeping a multi-hour log in RAM.
            if len(captured) > 4000:
                del captured[:2000]
        return process.wait(), "".join(captured)


def _selected_architectures(value: str) -> tuple[tuple[int, int], ...]:
    if value.strip().lower() == "all":
        return ARCHITECTURES
    selected: list[tuple[int, int]] = []
    for item in value.split(","):
        try:
            depth_text, width_text = item.strip().lower().split("x", 1)
            architecture = (int(depth_text), int(width_text))
        except ValueError as exc:
            raise ValueError(
                "--models must be 'all' or comma-separated values such as "
                "10x1024,22x512"
            ) from exc
        if architecture not in ARCHITECTURES:
            raise ValueError(f"unsupported architecture: {item!r}")
        if architecture not in selected:
            selected.append(architecture)
    if not selected:
        raise ValueError("--models selected no architectures")
    return tuple(selected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Python executable used for each isolated training process",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("dnn_dataset_track_p_12d_robust_v1"),
    )
    parser.add_argument("--runs-root", type=Path, default=Path("dnn_runs"))
    parser.add_argument("--run-prefix", default="track_p_12d_robust_v1")
    parser.add_argument("--models", default="all")
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--min-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=150)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    if not args.python.exists():
        raise FileNotFoundError(f"Python executable does not exist: {args.python}")
    for name in ("train.npz", "validation.npz", "test.npz", "normalization.json"):
        if not (args.data_dir / name).exists():
            raise FileNotFoundError(f"dataset file does not exist: {args.data_dir / name}")
    if args.batch_size < 1 or args.min_batch_size < 1:
        raise ValueError("batch sizes must be positive")

    architectures = _selected_architectures(args.models)
    args.runs_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.runs_root / f"{args.run_prefix}_sweep_status.json"
    status: dict[str, Any] = {
        "format_version": 1,
        "dataset": str(args.data_dir.resolve()),
        "python": str(args.python.resolve()),
        "architectures": [f"{depth}x{width}" for depth, width in architectures],
        "models": {},
    }
    if manifest_path.exists():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(previous.get("models"), dict):
                status["models"] = previous["models"]
        except (OSError, ValueError, TypeError):
            pass

    trainer = Path(__file__).with_name("train_dnn_controller.py")
    failures: list[str] = []
    for depth, width in architectures:
        key = f"depth{depth}_width{width}"
        output_dir = args.runs_root / f"{args.run_prefix}_{key}"
        output_dir.mkdir(parents=True, exist_ok=True)
        if _complete(output_dir):
            print(f"SKIP {key}: completed artifacts already exist in {output_dir}", flush=True)
            status["models"][key] = {
                "state": "complete",
                "output_dir": str(output_dir.resolve()),
                "skipped_existing": True,
            }
            _write_json(manifest_path, status)
            continue

        batch_size = args.batch_size
        while True:
            resume = (output_dir / "latest_checkpoint.pt").exists()
            command = [
                str(args.python),
                str(trainer),
                "--data-dir", str(args.data_dir),
                "--output-dir", str(output_dir),
                "--depth", str(depth),
                "--width", str(width),
                "--epochs", str(args.epochs),
                "--batch-size", str(batch_size),
                "--learning-rate", str(args.learning_rate),
                "--patience", str(args.patience),
                "--device", str(args.device),
                "--workers", str(args.workers),
            ]
            if resume:
                command.append("--resume")
            status["models"][key] = {
                "state": "running",
                "output_dir": str(output_dir.resolve()),
                "batch_size": batch_size,
                "resumed": resume,
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            }
            _write_json(manifest_path, status)
            return_code, tail = _run_and_tee(command, output_dir / "sweep_training.log")
            if return_code == 0 and _complete(output_dir):
                metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
                status["models"][key] = {
                    "state": "complete",
                    "output_dir": str(output_dir.resolve()),
                    "batch_size": batch_size,
                    "best_epoch": metrics.get("best_epoch"),
                    "test_rmse": metrics.get("test", {}).get("rmse"),
                    "cuda_memory": metrics.get("cuda_memory"),
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                }
                _write_json(manifest_path, status)
                print(f"COMPLETE {key}: artifacts saved in {output_dir}", flush=True)
                break

            is_oom = "out of memory" in tail.lower() and "cuda" in tail.lower()
            next_batch = batch_size // 2
            if is_oom and next_batch >= args.min_batch_size:
                print(
                    f"CUDA OOM for {key} at batch {batch_size}; retrying at "
                    f"batch {next_batch} from the latest completed epoch.",
                    flush=True,
                )
                batch_size = next_batch
                continue

            status["models"][key] = {
                "state": "failed",
                "output_dir": str(output_dir.resolve()),
                "batch_size": batch_size,
                "return_code": return_code,
                "cuda_oom": is_oom,
                "latest_checkpoint_available": (output_dir / "latest_checkpoint.pt").exists(),
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            }
            _write_json(manifest_path, status)
            failures.append(key)
            print(f"FAILED {key}; see {output_dir / 'sweep_training.log'}", flush=True)
            break

    status["finished_utc"] = datetime.now(timezone.utc).isoformat()
    status["status"] = "partial_failure" if failures else "complete"
    status["failures"] = failures
    _write_json(manifest_path, status)
    print(f"Sweep status: {manifest_path.resolve()}", flush=True)
    if failures:
        raise SystemExit(f"Training failed for: {', '.join(failures)}")


if __name__ == "__main__":
    main()
