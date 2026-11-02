"""Run and resume the full FPGA closed-loop sweep for paper-style SVR data."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
DEFAULT_BENCHMARK = HERE.parent / "physical_benchmark_v1" / "benchmark_config.json"
DEFAULT_FPGA_CONFIG = HERE / "fpga_current5ms_config.json"
DEFAULT_MODEL_MANIFEST = (
    HERE.parent
    / "ubuntu_transfer"
    / "falcon_current_5ms_12d_six_20260920_202227"
    / "package_manifest.json"
)
COLLECTOR = HERE / "run_fpga_closed_loop.py"
BUILDER = HERE / "build_falcon_svr_dataset.py"
PAPER_METRIC_VERSION = "falcon_eq4_wx_identity_route_mean_v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _csv_numbers(value: str, *, integer: bool = False) -> list[float | int]:
    converter = int if integer else float
    values = [converter(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("numeric list must not be empty")
    if len(values) != len(set(values)):
        raise ValueError(f"numeric list contains duplicates: {values}")
    return values


def _number_tag(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _build_plan(periods_ms: list[int], delay_fractions: list[float]) -> list[dict[str, Any]]:
    plan = []
    seen = set()
    for period_ms in periods_ms:
        if period_ms <= 0:
            raise ValueError("periods must be positive")
        for fraction in delay_fractions:
            if not 0.0 <= fraction < 1.0:
                raise ValueError("delay fractions must satisfy 0 <= fraction < 1")
            delay_ms = round(period_ms * fraction, 6)
            run_id = f"T{period_ms:03d}ms_delay{_number_tag(delay_ms)}ms"
            if run_id in seen:
                raise ValueError(f"duplicate run generated: {run_id}")
            seen.add(run_id)
            plan.append(
                {
                    "run_id": run_id,
                    "period_ms": period_ms,
                    "delay_fraction": fraction,
                    "actuation_delay_ms": delay_ms,
                }
            )
    return plan


def _run_status(run_dir: Path, expected_episodes: int) -> dict[str, Any]:
    summary_path = run_dir / "fpga_closed_loop_summary.json"
    if not summary_path.exists():
        return {"complete": False, "reason": "summary_missing"}
    summary = _read_json(summary_path)
    trials = summary.get("trials", [])
    complete = bool(
        summary.get("paper_metric", {}).get("version") == PAPER_METRIC_VERSION
        and
        summary.get("episodes_requested") == expected_episodes
        and len(trials) == expected_episodes
        and all(bool(item.get("completed")) for item in trials)
    )
    return {
        "complete": complete,
        "episodes_requested": summary.get("episodes_requested"),
        "episodes_completed": summary.get("episodes_completed"),
        "fallback_count": int(summary.get("fallback_count", 0) or 0),
        "deadline_miss_count": int(summary.get("deadline_miss_count", 0) or 0),
        "timeout_count": int(summary.get("timeout_count", 0) or 0),
    }


def _configuration(args: argparse.Namespace, plan: list[dict[str, Any]], expected: int) -> dict[str, Any]:
    return {
        "python": str(Path(sys.executable).resolve()),
        "benchmark_config": str(args.benchmark_config.resolve()),
        "fpga_config": str(args.fpga_config.resolve()),
        "model_manifest": str(args.model_manifest.resolve()),
        "paper_metric_version": PAPER_METRIC_VERSION,
        "scenario_set": args.scenario_set,
        "model_ids": args.model_ids,
        "repeats": args.repeats,
        "expected_episodes_per_run": expected,
        "periods_ms": args.periods_ms,
        "delay_fractions": args.delay_fractions,
        "plan": plan,
    }


def _load_or_create_manifest(
    path: Path,
    configuration: dict[str, Any],
) -> dict[str, Any]:
    if path.exists():
        manifest = _read_json(path)
        if manifest.get("configuration") != configuration:
            raise ValueError(
                "sweep configuration differs from the existing manifest; "
                "use the original command or a new output root"
            )
        return manifest
    manifest = {
        "format_version": 1,
        "created_utc": _utc_now(),
        "updated_utc": _utc_now(),
        "status": "running",
        "active_run": None,
        "configuration": configuration,
        "runs": {},
    }
    _write_json(path, manifest)
    return manifest


def _collector_command(args: argparse.Namespace, run: dict[str, Any], run_dir: Path) -> list[str]:
    command = [
        sys.executable,
        str(COLLECTOR),
        "--enable-fpga-actuation",
        "--benchmark-config",
        str(args.benchmark_config.resolve()),
        "--fpga-config",
        str(args.fpga_config.resolve()),
        "--scenario-set",
        args.scenario_set,
        "--model-ids",
        args.model_ids,
        "--repeats",
        str(args.repeats),
        "--period-ms",
        str(run["period_ms"]),
        "--actuation-delay-ms",
        str(run["actuation_delay_ms"]),
        "--output-dir",
        str(run_dir),
    ]
    if (run_dir / "closed_loop_manifest.json").exists():
        command.append("--resume")
    return command


def _build_dataset(args: argparse.Namespace, plan: list[dict[str, Any]], runs_root: Path) -> Path:
    dataset_dir = args.output_root / "svr_dataset"
    command = [
        sys.executable,
        str(BUILDER),
        "--model-manifest",
        str(args.model_manifest.resolve()),
        "--output-dir",
        str(dataset_dir),
    ]
    for run in plan:
        command.extend(("--input-dir", str(runs_root / run["run_id"])))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    return dataset_dir / "paper_style_episode_dataset_train_ready.csv"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--benchmark-config", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--fpga-config", type=Path, default=DEFAULT_FPGA_CONFIG)
    parser.add_argument("--model-manifest", type=Path, default=DEFAULT_MODEL_MANIFEST)
    parser.add_argument("--scenario-set", default="full")
    parser.add_argument("--model-ids", default="1,2,3,4,5,6")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--periods-ms", default="30,40,50,60,70,80,90,100")
    parser.add_argument("--delay-fractions", default="0,0.2,0.4")
    args = parser.parse_args()

    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    args.output_root = args.output_root.resolve()
    args.benchmark_config = args.benchmark_config.resolve()
    args.fpga_config = args.fpga_config.resolve()
    args.model_manifest = args.model_manifest.resolve()
    args.periods_ms = [int(item) for item in _csv_numbers(args.periods_ms, integer=True)]
    args.delay_fractions = [float(item) for item in _csv_numbers(args.delay_fractions)]
    model_ids = [int(item) for item in _csv_numbers(args.model_ids, integer=True)]
    if any(model_id not in range(1, 7) for model_id in model_ids):
        raise ValueError("--model-ids must be a subset of 1,2,3,4,5,6")

    benchmark = _read_json(args.benchmark_config)
    scenario_sets = benchmark.get("scenario_sets", {})
    if args.scenario_set not in scenario_sets:
        raise ValueError(f"unknown scenario set: {args.scenario_set}")
    scenario_count = len(scenario_sets[args.scenario_set])
    expected_episodes = len(model_ids) * scenario_count * args.repeats
    plan = _build_plan(args.periods_ms, args.delay_fractions)

    args.output_root.mkdir(parents=True, exist_ok=True)
    runs_root = args.output_root / "runs"
    runs_root.mkdir(exist_ok=True)
    manifest_path = args.output_root / "sweep_manifest.json"
    configuration = _configuration(args, plan, expected_episodes)
    manifest = _load_or_create_manifest(manifest_path, configuration)

    try:
        for index, run in enumerate(plan, start=1):
            run_id = run["run_id"]
            run_dir = runs_root / run_id
            status = _run_status(run_dir, expected_episodes)
            if status["complete"]:
                manifest["runs"][run_id] = status
                print(f"SKIP complete {index}/{len(plan)} {run_id}", flush=True)
                continue

            if run_dir.exists() and any(run_dir.iterdir()) and not (run_dir / "closed_loop_manifest.json").exists():
                raise ValueError(f"nonempty run directory has no resumable manifest: {run_dir}")
            manifest["status"] = "running"
            manifest["active_run"] = run_id
            manifest["updated_utc"] = _utc_now()
            _write_json(manifest_path, manifest)
            print(
                f"\n=== SWEEP {index}/{len(plan)} {run_id}: "
                f"T={run['period_ms']}ms delay={run['actuation_delay_ms']}ms ===",
                flush=True,
            )
            subprocess.run(
                _collector_command(args, run, run_dir),
                cwd=PROJECT_ROOT,
                check=True,
            )
            status = _run_status(run_dir, expected_episodes)
            manifest["runs"][run_id] = status
            manifest["updated_utc"] = _utc_now()
            _write_json(manifest_path, manifest)
            if not status["complete"]:
                raise RuntimeError(
                    f"run {run_id} is incomplete; fix the reported failure and rerun the same sweep command"
                )

        manifest["active_run"] = None
        manifest["status"] = "building_dataset"
        manifest["updated_utc"] = _utc_now()
        _write_json(manifest_path, manifest)
        dataset = _build_dataset(args, plan, runs_root)
        dataset_manifest = _read_json(dataset.parent / "paper_style_episode_dataset_manifest.json")
        manifest["status"] = "complete"
        manifest["completed_utc"] = _utc_now()
        manifest["updated_utc"] = manifest["completed_utc"]
        manifest["active_run"] = None
        manifest["dataset"] = str(dataset)
        manifest["records_total"] = dataset_manifest["records_total"]
        manifest["records_valid_for_svr"] = dataset_manifest["records_valid_for_svr"]
        _write_json(manifest_path, manifest)
        print(f"\nSWEEP COMPLETE\nTraining dataset: {dataset}", flush=True)
        return 0
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        manifest["updated_utc"] = _utc_now()
        _write_json(manifest_path, manifest)
        print("\nInterrupted. Re-run the exact same command to resume.", flush=True)
        return 130
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        manifest["updated_utc"] = _utc_now()
        _write_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
