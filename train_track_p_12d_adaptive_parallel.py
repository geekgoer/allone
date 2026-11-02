"""Train a six-network Track-P matrix with adaptive safe concurrency.

The launcher begins with a conservative number of parallel CUDA processes
selected from currently free VRAM. Every architecture writes to an independent
directory through ``train_dnn_controller.py``. If any process reports CUDA
out-of-memory, no new parallel work is launched; after already-running jobs
finish, the failed job and all remaining jobs continue serially. A serial OOM
halves that job's batch size and resumes from its last completed epoch.
"""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from train_track_p_12d_sweep import (
    ARCHITECTURES,
    _complete,
    _run_and_tee,
    _selected_architectures,
    _write_json,
)


@dataclass
class Job:
    depth: int
    width: int
    output_dir: Path
    batch_size: int
    submitted_in_parallel: bool = False
    attempts: int = 0

    @property
    def key(self) -> str:
        return f"depth{self.depth}_width{self.width}"


def _free_vram_mib() -> int | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        first = result.stdout.strip().splitlines()[0]
        return int(first.strip())
    except (FileNotFoundError, IndexError, ValueError, subprocess.SubprocessError):
        return None


def _parallel_limit(value: str, job_count: int) -> tuple[int, int | None]:
    free = _free_vram_mib()
    if value.lower() != "auto":
        limit = int(value)
        if limit < 1:
            raise ValueError("--max-parallel must be 'auto' or a positive integer")
        return min(limit, job_count), free
    # The largest 34x1024 model was measured at about 814 MiB reserved with
    # batch 2048 on the target RTX 4060. Keep a large allowance for independent
    # CUDA contexts, Windows graphics use and transient validation tensors.
    if free is None or free < 3000:
        limit = 1
    elif free < 5200:
        limit = 2
    else:
        limit = 3
    return min(limit, job_count), free


def _command(args: argparse.Namespace, trainer: Path, job: Job) -> list[str]:
    command = [
        str(args.python),
        str(trainer),
        "--data-dir", str(args.data_dir),
        "--output-dir", str(job.output_dir),
        "--depth", str(job.depth),
        "--width", str(job.width),
        "--epochs", str(args.epochs),
        "--batch-size", str(job.batch_size),
        "--learning-rate", str(args.learning_rate),
        "--patience", str(args.patience),
        "--device", str(args.device),
        "--workers", str(args.workers),
    ]
    if (job.output_dir / "latest_checkpoint.pt").exists():
        command.append("--resume")
    return command


def _attempt(
    args: argparse.Namespace,
    trainer: Path,
    job: Job,
) -> tuple[int, str]:
    job.output_dir.mkdir(parents=True, exist_ok=True)
    job.attempts += 1
    return _run_and_tee(
        _command(args, trainer, job),
        job.output_dir / "adaptive_training.log",
    )


def _model_status(job: Job, state: str, **extra: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "state": state,
        "output_dir": str(job.output_dir.resolve()),
        "batch_size": job.batch_size,
        "attempts": job.attempts,
        "latest_checkpoint_available": (job.output_dir / "latest_checkpoint.pt").exists(),
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }
    result.update(extra)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--models", default="all")
    parser.add_argument("--max-parallel", default="auto")
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--min-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=200)
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
    manifest_path = args.runs_root / f"{args.run_prefix}_adaptive_status.json"
    manifest: dict[str, Any] = {
        "format_version": 1,
        "dataset": str(args.data_dir.resolve()),
        "python": str(args.python.resolve()),
        "run_prefix": args.run_prefix,
        "architectures": [f"{depth}x{width}" for depth, width in architectures],
        "models": {},
    }
    if manifest_path.exists():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(previous.get("models"), dict):
                manifest["models"] = previous["models"]
        except (OSError, TypeError, ValueError):
            pass

    jobs: deque[Job] = deque()
    for depth, width in architectures:
        output_dir = args.runs_root / f"{args.run_prefix}_depth{depth}_width{width}"
        job = Job(depth, width, output_dir, args.batch_size)
        if _complete(output_dir):
            print(f"SKIP {job.key}: completed artifacts already exist in {output_dir}")
            manifest["models"][job.key] = _model_status(
                job, "complete", skipped_existing=True
            )
        else:
            jobs.append(job)
    _write_json(manifest_path, manifest)
    if not jobs:
        print(f"All requested models are already complete. Status: {manifest_path.resolve()}")
        return

    parallel_limit, free_vram = _parallel_limit(args.max_parallel, len(jobs))
    print(
        f"Adaptive launch: free_vram={free_vram if free_vram is not None else 'unknown'} MiB, "
        f"initial_parallelism={parallel_limit}, batch_size={args.batch_size}",
        flush=True,
    )
    trainer = Path(__file__).with_name("train_dnn_controller.py")
    active: dict[Future[tuple[int, str]], Job] = {}
    failures: list[str] = []
    serial_mode = parallel_limit == 1

    with ThreadPoolExecutor(max_workers=max(1, parallel_limit)) as pool:
        while jobs or active:
            current_limit = 1 if serial_mode else parallel_limit
            while jobs and len(active) < current_limit:
                job = jobs.popleft()
                job.submitted_in_parallel = current_limit > 1
                manifest["models"][job.key] = _model_status(
                    job,
                    "running",
                    submitted_in_parallel=job.submitted_in_parallel,
                )
                _write_json(manifest_path, manifest)
                active[pool.submit(_attempt, args, trainer, job)] = job

            completed, _ = wait(tuple(active), return_when=FIRST_COMPLETED)
            for future in completed:
                job = active.pop(future)
                try:
                    return_code, tail = future.result()
                except Exception as exc:
                    return_code, tail = 1, f"launcher exception: {type(exc).__name__}: {exc}"
                if return_code == 0 and _complete(job.output_dir):
                    metrics = json.loads(
                        (job.output_dir / "metrics.json").read_text(encoding="utf-8")
                    )
                    manifest["models"][job.key] = _model_status(
                        job,
                        "complete",
                        best_epoch=metrics.get("best_epoch"),
                        test_rmse=metrics.get("test", {}).get("rmse"),
                        cuda_memory=metrics.get("cuda_memory"),
                    )
                    _write_json(manifest_path, manifest)
                    print(f"COMPLETE {job.key}: {job.output_dir}", flush=True)
                    continue

                is_oom = "out of memory" in tail.lower() and "cuda" in tail.lower()
                if is_oom:
                    if job.submitted_in_parallel:
                        serial_mode = True
                        job.submitted_in_parallel = False
                        jobs.appendleft(job)
                        manifest["models"][job.key] = _model_status(
                            job,
                            "queued_serial_after_parallel_oom",
                        )
                        print(
                            f"CUDA OOM in parallel for {job.key}; draining active jobs and "
                            "switching this and all remaining work to serial execution.",
                            flush=True,
                        )
                        _write_json(manifest_path, manifest)
                        continue
                    next_batch = job.batch_size // 2
                    if next_batch >= args.min_batch_size:
                        job.batch_size = next_batch
                        job.submitted_in_parallel = False
                        serial_mode = True
                        jobs.appendleft(job)
                        manifest["models"][job.key] = _model_status(
                            job,
                            "queued_smaller_batch_after_serial_oom",
                        )
                        print(
                            f"CUDA OOM in serial for {job.key}; retrying from the last "
                            f"completed epoch with batch_size={next_batch}.",
                            flush=True,
                        )
                        _write_json(manifest_path, manifest)
                        continue

                failures.append(job.key)
                manifest["models"][job.key] = _model_status(
                    job,
                    "failed",
                    return_code=return_code,
                    cuda_oom=is_oom,
                    log=str((job.output_dir / "adaptive_training.log").resolve()),
                )
                _write_json(manifest_path, manifest)
                print(f"FAILED {job.key}: see its adaptive_training.log", flush=True)

    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["status"] = "partial_failure" if failures else "complete"
    manifest["failures"] = failures
    manifest["serial_fallback_activated"] = serial_mode and parallel_limit > 1
    _write_json(manifest_path, manifest)
    print(f"Adaptive sweep status: {manifest_path.resolve()}", flush=True)
    if failures:
        raise SystemExit(f"Training failed for: {', '.join(failures)}")


if __name__ == "__main__":
    main()
