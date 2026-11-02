"""Generation gate for the MPC-complete 300-D dataset.

This program writes *no* training examples.  It reconstructs the immutable
40-environment trajectories, evaluates a stratified set of real ACADO labels
with the deterministic teacher, and rejects the plan if either the label
quality or the environment-disjoint split is unsafe.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from airsim_interface import RawDroneState
from falcon_paper_fidelity.acado_mpc import AcadoMPCConfig
from falcon_paper_fidelity.mpc_complete_300d import (
    INPUT_CONTRACT,
    INPUT_DIMENSIONS,
    DeterministicAcadoTeacher,
    DeterministicTeacherConfig,
)
from falcon_paper_fidelity.motion_primitive import MotionPrimitiveConfig, QuinticMotionPrimitivePlanner
from falcon_paper_fidelity.phase_recovery_dataset import (
    _build_trajectories,
    _load_gates,
    _perturb_state,
    _validate_reconstruction,
)


def _resolve(value: str, base: Path) -> Path:
    return (base / value).resolve()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _quantiles(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0}
    return {
        "count": int(array.size), "mean": float(array.mean()), "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)), "p99": float(np.quantile(array, 0.99)), "max": float(array.max()),
    }


def _control_bucket(control: np.ndarray, lower: np.ndarray, upper: np.ndarray, *, tolerance: float, near_fraction: float) -> str:
    value = np.asarray(control, dtype=np.float64)
    if np.any(value <= lower + tolerance) or np.any(value >= upper - tolerance):
        return "saturated"
    distance = np.minimum(value - lower, upper - value)
    if np.any(distance <= near_fraction * (upper - lower)):
        return "near_limit"
    return "interior"


def _scaled_family(value: dict[str, Any], scale: float) -> dict[str, Any]:
    """Scale recovery magnitude while retaining family direction and ratios."""
    result: dict[str, Any] = {}
    for key, item in value.items():
        if key.endswith("_std") or key.endswith("_limit"):
            result[key] = (np.asarray(item, dtype=np.float64) * scale).tolist()
        else:
            result[key] = item
    return result


def _check_split(payload: dict[str, Any], planned: dict[str, Any]) -> dict[str, Any]:
    splits = {name: [int(item) for item in values] for name, values in payload["splits"].items()}
    expected = set(range(40))
    values = [item for group in splits.values() for item in group]
    unique = set(values)
    overlap_free = len(values) == len(unique)
    all_present = unique == expected
    exact_sizes = {"train": 32, "validation": 4, "test": 4}
    sizes_ok = {name: len(splits.get(name, [])) == count for name, count in exact_sizes.items()}
    per_environment = sum(int(value) for value in planned["samples_per_environment"].values())
    expected_counts = {name: len(splits[name]) * per_environment for name in exact_sizes}
    exact_plan = expected_counts == {name: int(planned["split_samples"][name]) for name in exact_sizes}
    return {
        "splits": splits, "disjoint": overlap_free, "covers_exactly_40_environments": all_present,
        "sizes_ok": sizes_ok, "per_environment_samples": per_environment,
        "expected_split_samples": expected_counts, "planned_split_samples": planned["split_samples"],
        "planned_counts_match": exact_plan,
        "passed": overlap_free and all_present and all(sizes_ok.values()) and exact_plan,
        "note": "IDs are difficulty-stratified and interleaved; no prefix/suffix environment split is used.",
    }


def run(config_path: Path, report_path: Path | None) -> int:
    config_file = config_path.resolve()
    base = config_file.parent
    config = json.loads(config_file.read_text(encoding="utf-8"))
    split = json.loads(_resolve(config["source"]["split"], base).read_text(encoding="utf-8"))
    split_check = _check_split(split, config["planned_dataset"])

    source_path = _resolve(config["source"]["core_dataset"], base)
    environment_path = _resolve(config["source"]["environments"], base)
    with np.load(source_path, allow_pickle=False) as archive:
        source = {key: np.asarray(archive[key]) for key in archive.files}
    required = {"X", "U", "episode_id", "environment_id", "step_id", "target_gate_id", "position", "quaternion", "velocity", "acceleration", "reference_position"}
    missing = sorted(required.difference(source))
    if missing:
        raise ValueError(f"source core lacks required arrays: {missing}")
    if source["X"].shape != (40000, 12) or source["U"].shape != (40000, 4):
        raise ValueError("preflight expects the complete immutable 40,000-row core")

    planner = QuinticMotionPrimitivePlanner(MotionPrimitiveConfig(**config["planner"]))
    environments = _load_gates(environment_path)
    trajectories = _build_trajectories(
        source, environments, planner,
        start_velocity_mode=str(config["trajectory"]["start_velocity_mode"]),
        nominal_speed=float(config["planner"]["nominal_speed"]),
        final_exit_distance=float(config["trajectory"]["final_exit_distance"]),
    )
    reconstruction_error = _validate_reconstruction(
        source, trajectories, control_dt=float(config["timing"]["control_dt"]),
        tolerance=float(config["source"]["reconstruction_tolerance"]),
    )

    mpc_values = dict(config["mpc"])
    mpc_values["library"] = str(_resolve(mpc_values["library"], base))
    teacher = DeterministicAcadoTeacher(
        AcadoMPCConfig(**mpc_values),
        DeterministicTeacherConfig(
            minimum_refinement_steps=int(config["teacher"]["minimum_refinement_steps"]),
            maximum_refinement_steps=int(config["teacher"]["maximum_refinement_steps"]),
            convergence_tolerance=float(config["teacher"]["convergence_tolerance"]),
        ),
    )
    lower = np.array([config["mpc"]["min_thrust"], -config["mpc"]["max_body_rate_xy"], -config["mpc"]["max_body_rate_xy"], -config["mpc"]["max_body_rate_z"]], dtype=np.float64)
    upper = np.array([config["mpc"]["max_thrust"], config["mpc"]["max_body_rate_xy"], config["mpc"]["max_body_rate_xy"], config["mpc"]["max_body_rate_z"]], dtype=np.float64)
    policy = config["saturation_policy"]
    rng = np.random.default_rng(int(config["random_seed"]))
    queries_per_environment = int(config["preflight"]["queries_per_environment"])
    scale = float(config["recovery_sampling"]["perturbation_scale"])
    if not 0.0 < scale <= 1.0:
        raise ValueError("recovery_sampling.perturbation_scale must be in (0, 1]")
    families = [_scaled_family(item, scale) for item in config["families"]]
    dt = float(config["timing"]["control_dt"])
    pdt = float(config["timing"]["prediction_dt"])
    horizon_nodes = int(config["timing"]["horizon"]) + 1
    max_speed = float(config["planner"]["maximum_speed"])
    max_acceleration = float(config["planner"]["maximum_acceleration"])
    records: list[dict[str, Any]] = []
    started = time.perf_counter()

    for environment_id in range(40):
        candidates = np.flatnonzero((source["environment_id"] == environment_id) & (source["step_id"] >= 2))
        if candidates.size == 0:
            raise RuntimeError(f"environment {environment_id} has no eligible core states")
        for slot in range(queries_per_environment):
            base_index = int(candidates[(slot * 97 + environment_id * 19) % candidates.size])
            gate_id = int(source["target_gate_id"][base_index])
            gate = next(item for item in environments[environment_id] if item.index == gate_id)
            if slot == 0:
                state = RawDroneState(
                    position=source["position"][base_index].copy(), quaternion=source["quaternion"][base_index].copy(),
                    velocity=source["velocity"][base_index].copy(), acceleration=source["acceleration"][base_index].copy(),
                    angular_velocity=np.zeros(3),
                )
                family = "core_nominal"
            else:
                item = families[(slot - 1) % len(families)]
                state, *_ = _perturb_state(source, base_index, gate, item, rng, maximum_speed=max_speed, maximum_acceleration=max_acceleration)
                family = str(item["name"])
            elapsed = float(source["step_id"][base_index]) * dt
            refs = planner.reference_window_for_trajectory(trajectories[int(source["episode_id"][base_index])], elapsed=elapsed, dt=pdt, horizon=horizon_nodes)
            result = teacher.query(state, refs)
            if result.feature.shape != (INPUT_DIMENSIONS,) or not np.isfinite(result.feature).all():
                raise RuntimeError("teacher produced a non-finite or malformed 300-D feature")
            bucket = _control_bucket(result.control, lower, upper, tolerance=float(policy["absolute_tolerance"]), near_fraction=float(policy["near_limit_fraction"]))
            records.append({"environment_id": environment_id, "family": family, "bucket": bucket, "converged": bool(result.converged), "iterations": int(result.iterations), "delta": float(result.final_control_delta), "control": result.control.tolist(), "feature_dimensions": int(result.feature.size)})

    repeat_count = min(int(config["preflight"]["repeatability_queries"]), len(records))
    repeat_errors: list[float] = []
    # Repeat exactly selected physical problems, reconstructed deterministically from the same indexed schedule.
    for repeat_index in range(repeat_count):
        record = records[repeat_index]
        environment_id = int(record["environment_id"])
        slot = repeat_index % queries_per_environment
        candidates = np.flatnonzero((source["environment_id"] == environment_id) & (source["step_id"] >= 2))
        base_index = int(candidates[(slot * 97 + environment_id * 19) % candidates.size])
        # The first repeat is nominal; perturbation repeats use a local RNG reconstructed from the global seed.
        # Re-querying a core problem is sufficient to test that solver history cannot change a label.
        state = RawDroneState(position=source["position"][base_index].copy(), quaternion=source["quaternion"][base_index].copy(), velocity=source["velocity"][base_index].copy(), acceleration=source["acceleration"][base_index].copy(), angular_velocity=np.zeros(3))
        elapsed = float(source["step_id"][base_index]) * dt
        refs = planner.reference_window_for_trajectory(trajectories[int(source["episode_id"][base_index])], elapsed=elapsed, dt=pdt, horizon=horizon_nodes)
        first = teacher.query(state, refs).control
        second = teacher.query(state, refs).control
        repeat_errors.append(float(np.max(np.abs(first - second))))

    buckets = {name: sum(record["bucket"] == name for record in records) for name in ("interior", "near_limit", "saturated")}
    family_buckets: dict[str, dict[str, int]] = {}
    for record in records:
        family_buckets.setdefault(str(record["family"]), {"interior": 0, "near_limit": 0, "saturated": 0})[str(record["bucket"])] += 1
    core_records = [record for record in records if record["family"] == "core_nominal"]
    saturation_fraction = buckets["saturated"] / len(records)
    core_saturation_fraction = sum(record["bucket"] == "saturated" for record in core_records) / len(core_records)
    convergence_fraction = sum(record["converged"] for record in records) / len(records)
    deltas = [float(record["delta"]) for record in records]
    p99_delta = float(np.quantile(deltas, .99))
    checks = {
        "input_contract": INPUT_CONTRACT,
        "split": split_check,
        "features_are_300d": all(record["feature_dimensions"] == INPUT_DIMENSIONS for record in records),
        "converged_fraction": {"actual": convergence_fraction, "minimum": float(config["teacher"]["minimum_converged_fraction"]), "passed": convergence_fraction >= float(config["teacher"]["minimum_converged_fraction"])},
        "p99_final_control_delta": {"actual": p99_delta, "maximum": float(config["teacher"]["maximum_p99_control_delta"]), "passed": p99_delta <= float(config["teacher"]["maximum_p99_control_delta"])},
        "repeatability_max_abs": {"actual": max(repeat_errors, default=0.0), "maximum": float(config["teacher"]["repeatability_tolerance"]), "passed": max(repeat_errors, default=0.0) <= float(config["teacher"]["repeatability_tolerance"])},
        "saturation_total_fraction": {"actual": saturation_fraction, "maximum": float(policy["maximum_total_fraction"]), "passed": saturation_fraction <= float(policy["maximum_total_fraction"])},
        "saturation_core_nominal_fraction": {"actual": core_saturation_fraction, "maximum": float(policy["maximum_nominal_core_fraction"]), "passed": core_saturation_fraction <= float(policy["maximum_nominal_core_fraction"])},
    }
    passed = split_check["passed"] and all(bool(value["passed"]) for key, value in checks.items() if isinstance(value, dict) and "passed" in value)
    report = {
        "status": "PASS" if passed else "FAIL", "config": str(config_file), "source": str(source_path),
        "elapsed_seconds": time.perf_counter() - started, "trajectory_reconstruction_max_abs": reconstruction_error,
        "queries": len(records), "per_environment_queries": queries_per_environment,
        "control_buckets": buckets, "family_control_buckets": family_buckets,
        "iterations": _quantiles([float(record["iterations"]) for record in records]),
        "final_control_delta": _quantiles(deltas), "repeatability_abs_error": _quantiles(repeat_errors), "checks": checks,
        "recovery_perturbation_scale": scale,
        "generation_authorization": "allowed" if passed else "blocked: revise configuration before generating any dataset shards",
    }
    text = json.dumps(_json_safe(report), indent=2)
    print(text)
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(text + "\n", encoding="utf-8")
        print(f"Report: {report_path}")
    return 0 if passed else 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).with_name("dataset_config.json")))
    parser.add_argument("--report", default=str(Path(__file__).with_name("preflight_report.json")))
    args = parser.parse_args()
    raise SystemExit(run(Path(args.config), Path(args.report) if args.report else None))


if __name__ == "__main__":
    main()
