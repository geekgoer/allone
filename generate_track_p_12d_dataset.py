"""Generate resumable, independent Track-P 12-D MPC teacher queries.

This is an offline state-space sampler, not an AirSim rollout.  Every row is
independent and receives a freshly recomputed MPC label.  Existing 25-D data
and model directories are never read or modified.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from dataset_store import atomic_write_json
from mpc_controller import MPCController
from track_p_12d_contract import (
    TRACK_P_12D_INPUT_FEATURES,
    solve_track_p_teacher,
)
from training_environment_generator import (
    TrainingEnvironment,
    generate_training_environments,
    save_training_environments,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "teacher_track_p_12d_robust_v1.json"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _environment_values(config: Mapping[str, Any]) -> list[TrainingEnvironment]:
    value = config["environment"]
    speeds = tuple(float(item) for item in value["speed_values"])
    if len(speeds) != 1:
        raise ValueError(
            "Track-P 12-D excludes reference speed, so speed_values must contain exactly one value"
        )
    return generate_training_environments(
        count=int(value["count"]),
        seed=int(value["seed"]),
        epsilon_values=value["epsilon_values"],
        gate_count=int(value["gate_count"]),
        x_start=float(value.get("x_start", 10.0)),
        x_spacing=float(value.get("x_spacing", 10.0)),
        x_start_values=value.get("x_start_values"),
        x_spacing_values=value.get("x_spacing_values"),
        base_y_range=value["base_y_range"],
        base_z_range=value["base_z_range"],
        max_z=float(value["max_z"]),
        speed_values=speeds,
    )


def _range(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3, 2) or not np.isfinite(result).all():
        raise ValueError(f"{name} must have shape (3, 2) with finite values")
    if np.any(result[:, 0] > result[:, 1]):
        raise ValueError(f"{name} lower bounds exceed upper bounds")
    return result


def _edge_biased_uniform(
    rng: np.random.Generator,
    ranges: np.ndarray,
    *,
    edge_probability: float,
    edge_band_fraction: float,
) -> np.ndarray:
    result = rng.uniform(ranges[:, 0], ranges[:, 1])
    widths = ranges[:, 1] - ranges[:, 0]
    for axis in range(3):
        if widths[axis] <= 0.0 or rng.random() >= edge_probability:
            continue
        band = widths[axis] * edge_band_fraction
        if rng.random() < 0.5:
            result[axis] = rng.uniform(ranges[axis, 0], ranges[axis, 0] + band)
        else:
            result[axis] = rng.uniform(ranges[axis, 1] - band, ranges[axis, 1])
    return result


def _bounded_norm(value: np.ndarray, maximum: float) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    return value if norm <= maximum else value * (maximum / norm)


def _families(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], np.ndarray]:
    families = [dict(value) for value in config["sampling"]["families"]]
    if not families:
        raise ValueError("sampling.families cannot be empty")
    probabilities = np.asarray([value["probability"] for value in families], dtype=np.float64)
    if np.any(probabilities < 0.0) or probabilities.sum() <= 0.0:
        raise ValueError("family probabilities must be non-negative with a positive sum")
    probabilities /= probabilities.sum()
    for family in families:
        for key in ("gate_position_error", "gate_attitude_error", "velocity", "acceleration"):
            family[key] = _range(family[key], f"{family.get('name', '<unnamed>')}.{key}")
        if family["gate_position_error"][0, 0] <= 0.0:
            raise ValueError("gate_position_error_x must remain positive (target gate is ahead)")
    return families, probabilities


def _generate_shard(
    environment: TrainingEnvironment,
    *,
    config: Mapping[str, Any],
    samples: int,
) -> dict[str, np.ndarray]:
    timing = config["timing"]
    mpc_config = config["mpc"]
    sampling = config["sampling"]
    dt = float(timing["dt"])
    horizon = int(mpc_config["horizon"])
    reference_speed = float(mpc_config["reference_speed"])
    if not np.isclose(environment.reference_speed, reference_speed):
        raise ValueError("environment speed and MPC reference_speed must be identical")
    controller = MPCController(
        dt=dt,
        horizon=horizon,
        max_horizontal_accel=float(mpc_config.get("max_horizontal_accel", 4.0)),
        max_thrust_accel=float(mpc_config.get("max_thrust_accel", 18.0)),
        max_body_rate=float(mpc_config.get("max_body_rate", 3.0)),
    )
    families, family_probabilities = _families(config)
    rng = np.random.default_rng(int(config["random_seed"]) + 104729 * environment.environment_id)
    x_values = np.empty((samples, 12), dtype=np.float32)
    controls = np.empty((samples, 4), dtype=np.float32)
    teacher_states = np.empty((samples, 12), dtype=np.float32)
    current_positions = np.empty((samples, 3), dtype=np.float32)
    gate_positions = np.empty((samples, 3), dtype=np.float32)
    gate_orientations = np.empty((samples, 4), dtype=np.float32)
    reference_first = np.empty((samples, 3), dtype=np.float32)
    reference_last = np.empty((samples, 3), dtype=np.float32)
    gate_ids = np.empty(samples, dtype=np.int16)
    family_ids = np.empty(samples, dtype=np.int8)
    solve_times = np.empty(samples, dtype=np.float32)
    edge_probability = float(sampling["edge_probability"])
    edge_band_fraction = float(sampling["edge_band_fraction"])
    if not 0.0 <= edge_probability <= 1.0 or not 0.0 < edge_band_fraction <= 0.5:
        raise ValueError("edge sampling probabilities are invalid")
    max_speed = float(sampling["max_total_speed"])
    max_acceleration = float(sampling["max_total_acceleration"])
    for index in range(samples):
        family_id = int(rng.choice(len(families), p=family_probabilities))
        family = families[family_id]
        gate_index = int(rng.integers(0, len(environment.gates)))
        gate = environment.gates[gate_index]
        position_error = _edge_biased_uniform(
            rng,
            family["gate_position_error"],
            edge_probability=edge_probability,
            edge_band_fraction=edge_band_fraction,
        )
        attitude_error = _edge_biased_uniform(
            rng,
            family["gate_attitude_error"],
            edge_probability=edge_probability,
            edge_band_fraction=edge_band_fraction,
        )
        velocity = _bounded_norm(
            _edge_biased_uniform(
                rng,
                family["velocity"],
                edge_probability=edge_probability,
                edge_band_fraction=edge_band_fraction,
            ),
            max_speed,
        )
        acceleration = _bounded_norm(
            _edge_biased_uniform(
                rng,
                family["acceleration"],
                edge_probability=edge_probability,
                edge_band_fraction=edge_band_fraction,
            ),
            max_acceleration,
        )
        started = time.perf_counter()
        control, query = solve_track_p_teacher(
            controller,
            gate_position=gate.center,
            gate_orientation=gate.orientation,
            gate_position_error=position_error,
            gate_attitude_error=attitude_error,
            velocity=velocity,
            acceleration=acceleration,
            dt=dt,
            horizon=horizon,
            reference_speed=reference_speed,
        )
        solve_times[index] = time.perf_counter() - started
        x_values[index] = query.policy_input
        controls[index] = control
        teacher_states[index] = query.teacher_state
        current_positions[index] = query.current_position
        gate_positions[index] = gate.center
        gate_orientations[index] = gate.orientation
        reference_first[index] = query.reference_window[0].position
        reference_last[index] = query.reference_window[-1].position
        gate_ids[index] = gate_index + 1
        family_ids[index] = family_id
    return {
        "X": x_values,
        "U": controls,
        "state_X": x_values.copy(),
        "teacher_state_X": teacher_states,
        "current_position": current_positions,
        "gate_position": gate_positions,
        "gate_orientation": gate_orientations,
        "reference_first_position": reference_first,
        "reference_last_position": reference_last,
        "reference_speed": np.full(samples, reference_speed, dtype=np.float32),
        "target_gate_id": gate_ids,
        "sample_family": family_ids,
        "environment_id": np.full(samples, environment.environment_id, dtype=np.int32),
        "episode_id": np.full(samples, environment.environment_id, dtype=np.int32),
        "initial_state_id": family_ids.astype(np.int32),
        "step_id": np.arange(samples, dtype=np.int32),
        "data_source": np.full(samples, 4, dtype=np.int8),
        "mpc_solve_time": solve_times,
    }


def _valid_shard(path: Path, *, samples: int, environment_id: int) -> bool:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return bool(
                archive["X"].shape == (samples, 12)
                and archive["U"].shape == (samples, 4)
                and np.all(archive["environment_id"] == environment_id)
                and np.isfinite(archive["X"]).all()
                and np.isfinite(archive["U"]).all()
            )
    except (OSError, ValueError, KeyError, EOFError):
        return False


def _merge_shards(paths: Sequence[Path], output: Path, family_names: Sequence[str]) -> dict[str, Any]:
    loaded: list[dict[str, np.ndarray]] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            loaded.append({key: np.asarray(archive[key]) for key in archive.files})
    keys = loaded[0].keys()
    if any(item.keys() != keys for item in loaded[1:]):
        raise ValueError("dataset shards do not contain identical arrays")
    merged = {key: np.concatenate([item[key] for item in loaded], axis=0) for key in keys}
    _atomic_npz(output, merged)
    families, family_counts = np.unique(merged["sample_family"], return_counts=True)
    gates, gate_counts = np.unique(merged["target_gate_id"], return_counts=True)
    return {
        "samples": int(merged["X"].shape[0]),
        "X_shape": list(merged["X"].shape),
        "U_shape": list(merged["U"].shape),
        "input_features": list(TRACK_P_12D_INPUT_FEATURES),
        "sample_family_counts": {
            family_names[int(value)]: int(count) for value, count in zip(families, family_counts)
        },
        "target_gate_counts": {str(int(value)): int(count) for value, count in zip(gates, gate_counts)},
        "reference_speed_values": sorted(float(value) for value in np.unique(merged["reference_speed"])),
        "X_min_per_dim": np.min(merged["X"], axis=0).astype(float).tolist(),
        "X_max_per_dim": np.max(merged["X"], axis=0).astype(float).tolist(),
        "X_std_per_dim": np.std(merged["X"], axis=0).astype(float).tolist(),
        "U_min_per_dim": np.min(merged["U"], axis=0).astype(float).tolist(),
        "U_max_per_dim": np.max(merged["U"], axis=0).astype(float).tolist(),
        "U_std_per_dim": np.std(merged["U"], axis=0).astype(float).tolist(),
        "average_mpc_solve_time_seconds": float(np.mean(merged["mpc_solve_time"])),
    }


def generate(
    config_path: Path,
    *,
    resume: bool,
    smoke_test: bool,
    limit_environments: int | None,
) -> Path:
    config = _load_json(config_path.resolve())
    if smoke_test:
        config = json.loads(json.dumps(config))
        config["environment"]["count"] = min(3, int(config["environment"]["count"]))
        config["dataset"]["samples_per_environment"] = min(
            128, int(config["dataset"]["samples_per_environment"])
        )
        config["dataset"]["root"] = f"{config['dataset']['root']}_smoke"
    output = (PROJECT_ROOT / config["dataset"]["root"]).resolve()
    samples = int(config["dataset"]["samples_per_environment"])
    if samples < 1:
        raise ValueError("samples_per_environment must be positive")
    environments = _environment_values(config)
    families, _ = _families(config)
    family_names = [str(value["name"]) for value in families]
    if len(family_names) != len(set(family_names)):
        raise ValueError("sampling family names must be unique")
    config_hash = _canonical_hash(config)
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and not resume:
        raise FileExistsError(f"output exists: {output}; use --resume to continue")
    output.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        manifest = _load_json(manifest_path)
        if manifest.get("config_hash") != config_hash:
            raise ValueError("resume configuration does not match the existing dataset")
    else:
        manifest = {
            "schema_version": 1,
            "status": "running",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "config_hash": config_hash,
            "config_file": str(config_path.resolve()),
            "strategy": "track_p_independent_gate_state_mpc_queries_v1",
            "input_contract": "[p_gate-p, r_gate-r, velocity, acceleration] in AirSim NED",
            "input_features": list(TRACK_P_12D_INPUT_FEATURES),
            "reference_rule": "fixed_speed_gate_axis_lane_recovery_v1",
            "reference_speed_is_student_input": False,
            "measured_acceleration_used_by_current_teacher": False,
            "input_dimensions": 12,
            "output_dimensions": 4,
            "environment_count": len(environments),
            "samples_per_environment": samples,
            "sample_family_names": family_names,
            "completed_environment_ids": [],
            "final_dataset": None,
        }
        atomic_write_json(output / "collection_config.json", config)
        save_training_environments(environments, output / "environments.json")
        atomic_write_json(manifest_path, manifest)
    completed = set(int(value) for value in manifest.get("completed_environment_ids", []))
    requested = len(environments) if limit_environments is None else min(limit_environments, len(environments))
    started = time.perf_counter()
    for environment in environments[:requested]:
        path = output / "shards" / f"environment_{environment.environment_id:04d}.npz"
        if environment.environment_id in completed and _valid_shard(
            path, samples=samples, environment_id=environment.environment_id
        ):
            continue
        arrays = _generate_shard(environment, config=config, samples=samples)
        _atomic_npz(path, arrays)
        if not _valid_shard(path, samples=samples, environment_id=environment.environment_id):
            raise RuntimeError(f"generated shard failed validation: {path}")
        completed.add(environment.environment_id)
        manifest["completed_environment_ids"] = sorted(completed)
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        atomic_write_json(manifest_path, manifest)
        elapsed = time.perf_counter() - started
        done = len(completed)
        rate = done / max(elapsed, 1e-9)
        remaining = (len(environments) - done) / max(rate, 1e-9)
        print(
            f"Environment: {environment.environment_id + 1:03d}/{len(environments):03d} | "
            f"Completed: {done} | Samples: {done * samples}/{len(environments) * samples} | "
            f"ETA: {remaining:.1f}s",
            flush=True,
        )
    if len(completed) == len(environments):
        shard_paths = [
            output / "shards" / f"environment_{index:04d}.npz"
            for index in range(len(environments))
        ]
        final_name = str(config["dataset"]["output_name"])
        final_path = output / "final" / final_name
        stats = _merge_shards(shard_paths, final_path, family_names)
        manifest["status"] = "completed"
        manifest["final_dataset"] = str(Path("final") / final_name)
        manifest["statistics"] = stats
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        atomic_write_json(manifest_path, manifest)
        print(json.dumps(stats, indent=2, ensure_ascii=False), flush=True)
        print(f"Track-P 12-D MPC dataset completed: {final_path}", flush=True)
    else:
        manifest["status"] = "incomplete"
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        atomic_write_json(manifest_path, manifest)
        print(
            f"Partial dataset saved ({len(completed)}/{len(environments)} environments). "
            "Run again with --resume.",
            flush=True,
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--limit-environments", type=int, default=None)
    args = parser.parse_args()
    generate(
        args.config,
        resume=args.resume,
        smoke_test=args.smoke_test,
        limit_environments=args.limit_environments,
    )


if __name__ == "__main__":
    main()
