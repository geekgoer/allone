"""Safely train the six Track-P 12-D FALCON networks one at a time.

The launcher can seed a new, isolated run from healthy partial checkpoints in
an earlier run root.  Checkpoints that failed to beat the mean-output baseline
are ignored, so collapsed deep models are initialized afresh.  No source file
or source checkpoint is modified.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

from train_track_p_12d_sweep import (
    ARCHITECTURES,
    _complete,
    _run_and_tee,
    _selected_architectures,
    _write_json,
)


CUDA_FATAL_MARKERS = (
    "cuda error",
    "illegal memory access",
    "device-side assert",
    "c10_cuda",
    "cublas",
    "cudnn",
    "out of memory",
)
WINDOWS_FATAL_EXIT_CODES = {
    0xC0000005,  # access violation, often returned by a failed CUDA DLL
    0xC0000409,  # fail-fast / stack-buffer-overrun reported by nvcuda64.dll
}


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required in the selected training environment") from exc
    return torch


def _validation_baseline_mse(data_dir: Path) -> float:
    with np.load(data_dir / "validation.npz", allow_pickle=False) as archive:
        target = np.asarray(archive["U"], dtype=np.float64)
    payload = json.loads((data_dir / "normalization.json").read_text(encoding="utf-8"))
    mean = np.asarray(payload["u_mean"], dtype=np.float64)
    return float(np.mean((target - mean) ** 2))


def _checkpoint_ratio(torch: Any, path: Path, baseline_mse: float) -> float | None:
    if not path.exists() or baseline_mse <= 0.0:
        return None
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        value = float(checkpoint["best_validation"])
        return value / baseline_mse if np.isfinite(value) else None
    except (OSError, KeyError, TypeError, ValueError, RuntimeError):
        return None


def _metrics_ratio(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        metrics = json.loads(path.read_text(encoding="utf-8"))
        health = metrics.get("health", {})
        if "validation_model_to_baseline_mse_ratio" in health:
            return float(health["validation_model_to_baseline_mse_ratio"])
        return None
    except (OSError, TypeError, ValueError):
        return None


def _status(
    *,
    state: str,
    output_dir: Path,
    batch_size: int,
    **extra: Any,
) -> dict[str, Any]:
    result = {
        "state": state,
        "output_dir": str(output_dir.resolve()),
        "batch_size": int(batch_size),
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }
    result.update(extra)
    return result


def _command(
    args: argparse.Namespace,
    *,
    trainer: Path,
    depth: int,
    width: int,
    output_dir: Path,
    batch_size: int,
    warm_start_from: Path | None,
) -> list[str]:
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
        "--max-validation-baseline-ratio", str(args.max_validation_baseline_ratio),
        "--skip-curve",
    ]
    local_latest = output_dir / "latest_checkpoint.pt"
    if local_latest.exists():
        command.append("--resume")
    elif warm_start_from is not None:
        command.extend(("--warm-start-from", str(warm_start_from)))
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--seed-runs-root", type=Path, default=None)
    parser.add_argument("--seed-run-prefix", default=None)
    parser.add_argument("--models", default="all")
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--min-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-validation-baseline-ratio", type=float, default=0.25)
    parser.add_argument(
        "--cooldown-seconds",
        type=float,
        default=30.0,
        help="idle time after each completed CUDA model before starting the next process",
    )
    args = parser.parse_args()

    if not args.python.exists():
        raise FileNotFoundError(f"Python executable does not exist: {args.python}")
    if args.batch_size < 1 or args.min_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    if args.cooldown_seconds < 0.0:
        raise ValueError("--cooldown-seconds cannot be negative")
    if not 0.0 < args.max_validation_baseline_ratio < 1.0:
        raise ValueError("--max-validation-baseline-ratio must be between zero and one")
    for name in ("train.npz", "validation.npz", "test.npz", "normalization.json"):
        if not (args.data_dir / name).exists():
            raise FileNotFoundError(f"dataset file does not exist: {args.data_dir / name}")
    if (args.seed_runs_root is None) != (args.seed_run_prefix is None):
        raise ValueError("--seed-runs-root and --seed-run-prefix must be supplied together")

    torch = _torch()
    baseline_mse = _validation_baseline_mse(args.data_dir)
    architectures = _selected_architectures(args.models)
    args.runs_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.runs_root / f"{args.run_prefix}_serial_status.json"
    manifest: dict[str, Any] = {
        "format_version": 1,
        "mode": "strict_serial",
        "dataset": str(args.data_dir.resolve()),
        "validation_mean_baseline_mse": baseline_mse,
        "maximum_allowed_baseline_ratio": args.max_validation_baseline_ratio,
        "architectures": [f"{depth}x{width}" for depth, width in architectures],
        "models": {},
    }
    if manifest_path.exists():
        try:
            old = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(old.get("models"), dict):
                manifest["models"] = old["models"]
        except (OSError, TypeError, ValueError):
            pass

    trainer = Path(__file__).with_name("train_dnn_controller.py")
    for depth, width in architectures:
        key = f"depth{depth}_width{width}"
        output_dir = args.runs_root / f"{args.run_prefix}_{key}"
        output_dir.mkdir(parents=True, exist_ok=True)
        existing_ratio = _metrics_ratio(output_dir / "metrics.json")
        if _complete(output_dir) and existing_ratio is not None and existing_ratio <= args.max_validation_baseline_ratio:
            print(f"SKIP {key}: healthy completed model already exists", flush=True)
            manifest["models"][key] = _status(
                state="complete",
                output_dir=output_dir,
                batch_size=args.batch_size,
                skipped_existing=True,
                validation_baseline_ratio=existing_ratio,
            )
            _write_json(manifest_path, manifest)
            continue
        if (output_dir / "metrics.json").exists() and existing_ratio is not None:
            raise RuntimeError(
                f"{output_dir} contains a completed but unhealthy model; use a new run prefix/root"
            )

        warm_start_from: Path | None = None
        source_ratio: float | None = None
        if not (output_dir / "latest_checkpoint.pt").exists() and args.seed_runs_root is not None:
            source_dir = args.seed_runs_root / f"{args.seed_run_prefix}_{key}"
            source_best = source_dir / "best_model.pt"
            source_ratio = _checkpoint_ratio(torch, source_best, baseline_mse)
            if source_ratio is not None and source_ratio <= args.max_validation_baseline_ratio:
                warm_start_from = source_best
                print(
                    f"SEED {key}: warm-start healthy best weights and reset optimizer "
                    f"(best/baseline={source_ratio:.6g})",
                    flush=True,
                )
            elif source_best.exists():
                print(
                    f"IGNORE {key}: old checkpoint is collapsed or unverifiable "
                    f"(best/baseline={source_ratio}); starting with Kaiming initialization",
                    flush=True,
                )

        batch_size = args.batch_size
        while True:
            manifest["models"][key] = _status(
                state="running",
                output_dir=output_dir,
                batch_size=batch_size,
                seeded_from=str(warm_start_from.resolve()) if warm_start_from is not None else None,
                source_validation_baseline_ratio=source_ratio,
            )
            _write_json(manifest_path, manifest)
            command = _command(
                args,
                trainer=trainer,
                depth=depth,
                width=width,
                output_dir=output_dir,
                batch_size=batch_size,
                warm_start_from=warm_start_from,
            )
            return_code, tail = _run_and_tee(command, output_dir / "serial_training.log")
            warm_start_from = None
            if return_code == 0 and _complete(output_dir):
                metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
                ratio = float(metrics["health"]["validation_model_to_baseline_mse_ratio"])
                manifest["models"][key] = _status(
                    state="complete",
                    output_dir=output_dir,
                    batch_size=batch_size,
                    best_epoch=metrics.get("best_epoch"),
                    test_rmse=metrics.get("test", {}).get("rmse"),
                    validation_baseline_ratio=ratio,
                    cuda_memory=metrics.get("cuda_memory"),
                )
                _write_json(manifest_path, manifest)
                print(f"COMPLETE {key}: {output_dir}", flush=True)
                if args.cooldown_seconds > 0.0:
                    print(
                        f"Cooling down CUDA for {args.cooldown_seconds:.0f}s before the next model.",
                        flush=True,
                    )
                    time.sleep(args.cooldown_seconds)
                break

            lowered = tail.lower()
            cuda_fatal = (
                any(marker in lowered for marker in CUDA_FATAL_MARKERS)
                or return_code in WINDOWS_FATAL_EXIT_CODES
                or (return_code & 0xFFFFFFFF) in WINDOWS_FATAL_EXIT_CODES
            )
            is_oom = "out of memory" in lowered and "cuda" in lowered
            if is_oom and batch_size // 2 >= args.min_batch_size:
                batch_size //= 2
                print(
                    f"CUDA OOM for {key}; retrying serially from latest checkpoint "
                    f"with batch_size={batch_size}",
                    flush=True,
                )
                continue

            manifest["models"][key] = _status(
                state="failed_cuda_restart_required" if cuda_fatal else "failed",
                output_dir=output_dir,
                batch_size=batch_size,
                return_code=return_code,
                cuda_fatal=cuda_fatal,
                log=str((output_dir / "serial_training.log").resolve()),
            )
            manifest["status"] = "stopped_on_failure"
            _write_json(manifest_path, manifest)
            if cuda_fatal:
                raise SystemExit(
                    f"Fatal CUDA failure in {key}. Progress is saved. Reboot Windows, then rerun "
                    "the same serial command; do not continue to another model in this GPU session."
                )
            raise SystemExit(f"Training failed for {key}; see {output_dir / 'serial_training.log'}")

    manifest["status"] = "complete"
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(manifest_path, manifest)
    print(f"All requested serial models completed: {manifest_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
