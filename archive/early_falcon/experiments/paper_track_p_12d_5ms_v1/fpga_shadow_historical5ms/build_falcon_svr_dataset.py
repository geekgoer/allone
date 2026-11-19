"""Build a paper-style episode-level SVR dataset from FPGA closed-loop runs.

The input is one or more directories produced by ``run_fpga_closed_loop.py``.
One completed episode becomes one row.  ``R`` is the measured total
state-to-command-dispatch latency, including the configured actuation delay.
The configured delay is retained separately.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
DEFAULT_MODEL_MANIFEST = (
    HERE.parent
    / "ubuntu_transfer"
    / "falcon_current_5ms_12d_six_20260920_202227"
    / "package_manifest.json"
)

CSV_FIELDS = [
    "record_id",
    "source_directory",
    "experiment_id",
    "trial_key",
    "model_id",
    "model_name",
    "depth",
    "width",
    "parameter_count",
    "beta_test_physical_mse",
    "beta_test_normalized_mse",
    "c_inverse_beta",
    "period_ms",
    "actuation_delay_ms",
    "R_max_ms",
    "R_p95_ms",
    "R_p99_ms",
    "measured_R_max_ms",
    "measured_R_p95_ms",
    "measured_R_p99_ms",
    "scenario_id",
    "epsilon",
    "seed",
    "initial_state_id",
    "repeat_index",
    "tracking_cost",
    "J_proxy",
    "paper_metric_version",
    "paper_horizon_K",
    "paper_window_samples",
    "paper_complete_windows",
    "paper_used_samples",
    "paper_discarded_samples",
    "paper_route_cost_mean",
    "J_paper_style",
    "reference_profile_version",
    "startup_duration_s",
    "completed",
    "pure_fpga_episode",
    "valid_for_svr",
    "samples",
    "crossings_count",
    "fallback_count",
    "deadline_miss_count",
    "timeout_count",
    "collision",
    "error",
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _number_or_blank(value: Any) -> Any:
    number = _finite_float(value)
    return number if number is not None else ""


def _int_or_blank(value: Any) -> Any:
    if value is None:
        return ""
    try:
        return int(value)
    except (TypeError, ValueError):
        return ""


def _model_dimensions(name: str) -> tuple[int | None, int | None]:
    parts = str(name).split("_")
    depth = width = None
    for part in parts:
        if part.startswith("depth"):
            try:
                depth = int(part.removeprefix("depth"))
            except ValueError:
                pass
        if part.startswith("width"):
            try:
                width = int(part.removeprefix("width"))
            except ValueError:
                pass
    return depth, width


def _model_map(manifest: dict[str, Any]) -> dict[int, dict[str, Any]]:
    models: dict[int, dict[str, Any]] = {}
    for item in manifest.get("models", []):
        model_id = int(item["model_id"])
        name = str(item["name"])
        depth, width = _model_dimensions(name)
        physical = _finite_float(item.get("test_physical_mse"))
        normalized = _finite_float(item.get("test_normalized_mse"))
        if physical is None or physical <= 0.0:
            raise ValueError(f"model {model_id} has no positive test_physical_mse")
        if normalized is None or normalized <= 0.0:
            raise ValueError(f"model {model_id} has no positive test_normalized_mse")
        models[model_id] = {
            "model_id": model_id,
            "model_name": name,
            "depth": depth,
            "width": width,
            "parameter_count": item.get("parameters"),
            "beta_test_physical_mse": physical,
            "beta_test_normalized_mse": normalized,
            "c_inverse_beta": 1.0 / normalized,
        }
    if not models:
        raise ValueError("model manifest contains no models")
    return models


def _summary_trials(input_dir: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    summary_path = input_dir / "fpga_closed_loop_summary.json"
    manifest_path = input_dir / "closed_loop_manifest.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing {summary_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing {manifest_path}")
    summary = _read_json(summary_path)
    run_manifest = _read_json(manifest_path)
    trials = summary.get("trials")
    if isinstance(trials, list) and trials:
        return summary, run_manifest, [dict(item) for item in trials]
    summary_dir = input_dir / "trial_summaries"
    trials = []
    if summary_dir.exists():
        for path in sorted(summary_dir.glob("*.json")):
            trials.append(_read_json(path))
    if not trials:
        raise ValueError(f"no trial summaries found in {input_dir}")
    return summary, run_manifest, trials


def _dispatch_values(trial: dict[str, Any]) -> list[float]:
    values = trial.get("state_to_command_dispatch_us", [])
    if not isinstance(values, list):
        return []
    result = []
    for value in values:
        number = _finite_float(value)
        if number is not None and number >= 0.0:
            result.append(number)
    return result


def _trial_key(trial: dict[str, Any]) -> str:
    if trial.get("trial_key"):
        return str(trial["trial_key"])
    return (
        f"m{int(trial['model_id']):02d}_{trial['scenario_id']}"
        f"_r{int(trial.get('repeat_index', 0)):02d}"
    )


def _make_row(
    *,
    input_dir: Path,
    run_manifest: dict[str, Any],
    trial: dict[str, Any],
    models: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    model_id = int(trial["model_id"])
    if model_id not in models:
        raise ValueError(f"model_id={model_id} is absent from the model manifest")
    model = models[model_id]
    reference_profile = run_manifest.get("reference_profile", {"version": "constant_speed_polyline_v1", "startup_duration_s": 0.0})
    if trial.get("reference_profile", reference_profile) != reference_profile:
        raise ValueError("trial reference profile disagrees with its run manifest")
    key = _trial_key(trial)
    dispatch_us = _dispatch_values(trial)

    measured_r_max = _finite_float(trial.get("measured_R_max_ms"))
    measured_r_p95 = _finite_float(trial.get("measured_R_p95_ms"))
    measured_r_p99 = _finite_float(trial.get("measured_R_p99_ms"))
    if measured_r_max is None:
        measured_r_max = _finite_float(trial.get("R_max_ms"))
    if measured_r_p95 is None:
        measured_r_p95 = _finite_float(trial.get("R_p95_ms"))
    if measured_r_p99 is None:
        measured_r_p99 = _finite_float(trial.get("R_p99_ms"))
    if dispatch_us:
        measured = [value / 1000.0 for value in dispatch_us]
        if measured_r_max is None:
            measured_r_max = max(measured)
        if measured_r_p95 is None:
            measured_r_p95 = _percentile(measured, 95.0)
        if measured_r_p99 is None:
            measured_r_p99 = _percentile(measured, 99.0)

    completed = bool(trial.get("completed", False))
    tracking_cost = _finite_float(trial.get("tracking_cost"))
    j_proxy = None
    if completed and tracking_cost is not None and tracking_cost > 0.0:
        j_proxy = 1.0 / tracking_cost

    paper_metric_version = str(trial.get("paper_metric_version", ""))
    paper_horizon_k = _int_or_blank(trial.get("paper_horizon_K"))
    paper_window_samples = _int_or_blank(trial.get("paper_window_samples"))
    paper_complete_windows = _int_or_blank(trial.get("paper_complete_windows"))
    paper_used_samples = _int_or_blank(trial.get("paper_used_samples"))
    paper_discarded_samples = _int_or_blank(trial.get("paper_discarded_samples"))
    reported_route_cost_mean = _finite_float(trial.get("paper_route_cost_mean"))
    reported_j_paper_style = _finite_float(trial.get("J_paper_style"))
    raw_window_costs = trial.get("paper_window_costs", [])
    paper_window_costs = []
    if isinstance(raw_window_costs, list):
        paper_window_costs = [
            value
            for item in raw_window_costs
            if (value := _finite_float(item)) is not None and value >= 0.0
        ]
    paper_route_cost_mean = (
        sum(paper_window_costs) / len(paper_window_costs)
        if paper_window_costs
        else None
    )
    j_paper_style = (
        1.0 / paper_route_cost_mean
        if paper_route_cost_mean is not None and paper_route_cost_mean > 0.0
        else None
    )
    paper_metric_valid = bool(
        paper_metric_version == "falcon_eq4_wx_identity_route_mean_v1"
        and paper_horizon_k == 20
        and paper_window_samples == 21
        and isinstance(paper_complete_windows, int)
        and paper_complete_windows > 0
        and len(paper_window_costs) == paper_complete_windows
        and paper_route_cost_mean is not None
        and paper_route_cost_mean > 0.0
        and j_paper_style is not None
        and j_paper_style > 0.0
        and reported_route_cost_mean is not None
        and math.isclose(reported_route_cost_mean, paper_route_cost_mean, rel_tol=1e-9, abs_tol=1e-12)
        and reported_j_paper_style is not None
        and math.isclose(reported_j_paper_style, j_paper_style, rel_tol=1e-9, abs_tol=1e-12)
    )

    period_ms = trial.get("period_ms", run_manifest.get("period_ms"))
    delay_ms = trial.get("actuation_delay_ms", run_manifest.get("actuation_delay_ms", 0.0))
    fallback_count = int(trial.get("fallback_count", 0) or 0)
    deadline_miss_count = int(trial.get("deadline_miss_count", 0) or 0)
    timeout_count = int(trial.get("timeout_count", 0) or 0)
    pure_fpga = bool(
        completed
        and not trial.get("error")
        and not trial.get("collision")
        and fallback_count == 0
        and deadline_miss_count == 0
        and timeout_count == 0
    )
    delay_ms = _finite_float(delay_ms)
    r_input = measured_r_max
    valid = bool(pure_fpga and paper_metric_valid and r_input is not None and r_input > 0.0)
    return {
        "record_id": f"{run_manifest.get('experiment_id', input_dir.name)}::{key}",
        "source_directory": input_dir.name,
        "experiment_id": run_manifest.get("experiment_id", input_dir.name),
        "trial_key": key,
        "model_id": model_id,
        "model_name": model["model_name"],
        "depth": model["depth"],
        "width": model["width"],
        "parameter_count": model["parameter_count"],
        "beta_test_physical_mse": model["beta_test_physical_mse"],
        "beta_test_normalized_mse": model["beta_test_normalized_mse"],
        "c_inverse_beta": model["c_inverse_beta"],
        "period_ms": _number_or_blank(period_ms),
        "actuation_delay_ms": _number_or_blank(delay_ms),
        "R_max_ms": _number_or_blank(r_input),
        "R_p95_ms": _number_or_blank(measured_r_p95),
        "R_p99_ms": _number_or_blank(measured_r_p99),
        "measured_R_max_ms": _number_or_blank(measured_r_max),
        "measured_R_p95_ms": _number_or_blank(measured_r_p95),
        "measured_R_p99_ms": _number_or_blank(measured_r_p99),
        "scenario_id": str(trial.get("scenario_id", "")),
        "epsilon": _number_or_blank(trial.get("epsilon")),
        "seed": _int_or_blank(trial.get("seed")),
        "initial_state_id": _int_or_blank(trial.get("initial_state_id")),
        "repeat_index": _int_or_blank(trial.get("repeat_index", 0)),
        "tracking_cost": _number_or_blank(tracking_cost),
        "J_proxy": _number_or_blank(j_proxy),
        "paper_metric_version": paper_metric_version,
        "paper_horizon_K": paper_horizon_k,
        "paper_window_samples": paper_window_samples,
        "paper_complete_windows": paper_complete_windows,
        "paper_used_samples": paper_used_samples,
        "paper_discarded_samples": paper_discarded_samples,
        "paper_route_cost_mean": _number_or_blank(paper_route_cost_mean),
        "J_paper_style": _number_or_blank(j_paper_style),
        "reference_profile_version": reference_profile["version"],
        "startup_duration_s": float(reference_profile["startup_duration_s"]),
        "completed": completed,
        "pure_fpga_episode": pure_fpga,
        "valid_for_svr": valid,
        "samples": _int_or_blank(trial.get("samples")),
        "crossings_count": len(trial.get("crossings", [])),
        "fallback_count": fallback_count,
        "deadline_miss_count": deadline_miss_count,
        "timeout_count": timeout_count,
        "collision": trial.get("collision") or "",
        "error": trial.get("error") or "",
    }


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    rows = list(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    rows = list(rows)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        action="append",
        type=Path,
        required=True,
        help="closed-loop result directory; repeat for each period/delay experiment",
    )
    parser.add_argument("--model-manifest", type=Path, default=DEFAULT_MODEL_MANIFEST)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    model_manifest_path = args.model_manifest.resolve()
    models = _model_map(_read_json(model_manifest_path))
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    input_dirs: list[str] = []
    for raw_dir in args.input_dir:
        input_dir = raw_dir.resolve()
        summary, run_manifest, trials = _summary_trials(input_dir)
        del summary
        input_dirs.append(str(input_dir))
        for trial in trials:
            row = _make_row(input_dir=input_dir, run_manifest=run_manifest, trial=trial, models=models)
            if row["record_id"] in seen:
                raise ValueError(f"duplicate experiment/trial record: {row['record_id']}")
            seen.add(row["record_id"])
            rows.append(row)

    rows.sort(key=lambda row: (str(row["experiment_id"]), str(row["trial_key"])))
    reference_profiles = {(row["reference_profile_version"], row["startup_duration_s"]) for row in rows}
    if len(reference_profiles) > 1:
        raise ValueError("cannot combine different reference profiles in one SVR dataset; build each profile separately")
    valid_rows = [row for row in rows if row["valid_for_svr"]]
    failed_rows = [row for row in rows if not row["valid_for_svr"]]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "paper_style_episode_dataset_all.csv", rows)
    _write_csv(args.output_dir / "paper_style_episode_dataset_train_ready.csv", valid_rows)
    _write_jsonl(args.output_dir / "paper_style_episode_dataset_all.jsonl", rows)
    manifest = {
        "format_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target_definition": "J_paper_style = 1 / mean(non-overlapping complete 21-sample Eq. (4) window costs), K=20, Wx=I_10, quaternion signs aligned",
        "route_aggregation_note": "The paper defines one K=20 Eq. (4) window but does not publish route-level aggregation; this reproduction averages complete window costs before taking the reciprocal.",
        "c_definition": "c = 1 / beta, beta = six-model test normalized MSE from package manifest",
        "R_definition": "R_max_ms = max(state_to_command_dispatch_us) / 1000 within one episode; this includes native response time and injected actuation_delay_ms",
        "measured_latency_definition": "R_max_ms/R_p95_ms/R_p99_ms are measured total state-to-command-dispatch latencies; measured_R_* are retained as explicit aliases for diagnostics",
        "delay_definition": "actuation_delay_ms is an injected sleep before AirSim command dispatch and is added to the native response path",
        "valid_for_svr_definition": "completed pure-FPGA episode with no fallback, deadline miss, timeout, collision, or error, a valid finite positive J_paper_style from at least one complete 21-sample window, and finite positive measured total R",
        "model_manifest": str(model_manifest_path),
        "input_directories": input_dirs,
        "records_total": len(rows),
        "records_valid_for_svr": len(valid_rows),
        "records_excluded_from_train_ready": len(failed_rows),
        "excluded_record_ids": [row["record_id"] for row in failed_rows],
        "columns": CSV_FIELDS,
        "reference_profiles": [{"version": version, "startup_duration_s": duration} for version, duration in sorted(reference_profiles)],
    }
    (args.output_dir / "paper_style_episode_dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "records_total": len(rows),
        "records_valid_for_svr": len(valid_rows),
        "records_excluded_from_train_ready": len(failed_rows),
        "output_dir": str(args.output_dir.resolve()),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
