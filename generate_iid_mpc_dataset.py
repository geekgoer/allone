"""Generate independent path-aware MPC teacher queries without flying AirSim.

Every row is sampled independently.  A random route location and a physically
bounded vehicle deviation define the current state; the existing path planner
provides a 20-step reference window; the MPC is then queried for the label.
Shards are written per environment so interrupted runs can resume safely.
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
from dnn_model_utils import (
    PATH_AWARE_INPUT_FEATURES,
    PATH_CONTEXT_OFFSETS,
    build_policy_input,
    input_feature_names,
)
from mpc_controller import MPCController, MPCReference
from reference_generator import gate_passage_waypoints
from training_environment_generator import (
    TrainingEnvironment,
    generate_training_environments,
    save_training_environments,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "teacher_iid_mpc_v1.json"
FAMILY_NAMES = ("nominal", "moderate", "recovery")


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
        speed_values=value["speed_values"],
    )


def _route_geometry(
    environment: TrainingEnvironment,
    *,
    start_x: float,
    approach_distance: float,
    exit_distance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    first = environment.gates[0].center
    start = np.array([start_x, first[1], first[2]], dtype=np.float64)
    points = [start]
    gate_progress: list[float] = []
    distance = 0.0
    previous = start
    for gate in environment.gates:
        before, after = gate_passage_waypoints(
            gate.center,
            gate.orientation,
            approach_distance=approach_distance,
            exit_distance=exit_distance,
        )
        distance += float(np.linalg.norm(before - previous))
        gate_progress.append(distance + float(np.linalg.norm(gate.center - before)))
        points.extend((before, after))
        distance += float(np.linalg.norm(after - before))
        previous = after
    waypoints = np.asarray(points, dtype=np.float64)
    segment_lengths = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    return waypoints, cumulative, np.asarray(gate_progress, dtype=np.float64), float(cumulative[-1])


def _reference_at_distance(
    waypoints: np.ndarray,
    cumulative: np.ndarray,
    distance: float,
    *,
    speed: float,
    orientation: np.ndarray,
) -> MPCReference:
    segment_lengths = np.diff(cumulative)
    if distance >= cumulative[-1]:
        return MPCReference(waypoints[-1], orientation, velocity=np.zeros(3))
    segment = min(
        int(np.searchsorted(cumulative, max(0.0, distance), side="right") - 1),
        len(segment_lengths) - 1,
    )
    segment = max(segment, 0)
    length = float(segment_lengths[segment])
    if length <= 1e-12:
        return MPCReference(waypoints[segment + 1], orientation, velocity=np.zeros(3))
    direction = (waypoints[segment + 1] - waypoints[segment]) / length
    local = np.clip(distance - cumulative[segment], 0.0, length)
    return MPCReference(
        waypoints[segment] + direction * local,
        orientation,
        velocity=direction * speed,
    )


def _reference_window(
    waypoints: np.ndarray,
    cumulative: np.ndarray,
    progress: float,
    *,
    speed: float,
    orientation: np.ndarray,
    dt: float,
    horizon: int,
) -> list[MPCReference]:
    return [
        _reference_at_distance(
            waypoints,
            cumulative,
            progress + index * speed * dt,
            speed=speed,
            orientation=orientation,
        )
        for index in range(horizon)
    ]


def _sample_progress(
    rng: np.random.Generator,
    *,
    total_length: float,
    gate_progress: np.ndarray,
    gate_focus_fraction: float,
    gate_focus_radius: float,
) -> float:
    if rng.random() < gate_focus_fraction:
        center = float(rng.choice(gate_progress))
        return float(np.clip(center + rng.uniform(-gate_focus_radius, gate_focus_radius), 0.0, total_length))
    return float(rng.uniform(0.0, total_length))


def _bounded_velocity(value: np.ndarray, max_speed: float) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm > max_speed:
        return value * (max_speed / norm)
    return value


def _generate_shard(
    environment: TrainingEnvironment,
    *,
    config: Mapping[str, Any],
    samples: int,
) -> dict[str, np.ndarray]:
    timing = config["timing"]
    route_config = config["route"]
    sampling = config["sampling"]
    mpc_config = config["mpc"]
    input_config = config["input"]
    dt = float(timing["dt"])
    horizon = int(mpc_config["horizon"])
    offsets = tuple(int(value) for value in input_config["path_context_offsets"])
    if offsets != PATH_CONTEXT_OFFSETS:
        raise ValueError(f"path_context_offsets must be {list(PATH_CONTEXT_OFFSETS)} for the 25-D model")
    controller = MPCController(
        dt=dt,
        horizon=horizon,
        max_horizontal_accel=float(mpc_config.get("max_horizontal_accel", 4.0)),
        max_thrust_accel=float(mpc_config.get("max_thrust_accel", 18.0)),
        max_body_rate=float(mpc_config.get("max_body_rate", 3.0)),
    )
    waypoints, cumulative, gate_progress, total_length = _route_geometry(
        environment,
        start_x=float(route_config["start_x"]),
        approach_distance=float(route_config["approach_distance"]),
        exit_distance=float(route_config["exit_distance"]),
    )
    rng = np.random.default_rng(int(config["random_seed"]) + 104729 * environment.environment_id)
    x_values = np.empty((samples, len(PATH_AWARE_INPUT_FEATURES)), dtype=np.float32)
    state_values = np.empty((samples, 12), dtype=np.float32)
    controls = np.empty((samples, 4), dtype=np.float32)
    contexts = np.empty((samples, len(offsets), 3), dtype=np.float32)
    progress_values = np.empty(samples, dtype=np.float32)
    gate_ids = np.empty(samples, dtype=np.int16)
    families = np.empty(samples, dtype=np.int8)
    reference_positions = np.empty((samples, 3), dtype=np.float32)
    reference_velocities = np.empty((samples, 3), dtype=np.float32)
    solve_times = np.empty(samples, dtype=np.float32)
    max_speed = float(sampling["max_total_speed"])
    family_probabilities = np.asarray(sampling["family_probabilities"], dtype=np.float64)
    if (
        family_probabilities.shape != (len(FAMILY_NAMES),)
        or np.any(family_probabilities < 0.0)
        or family_probabilities.sum() <= 0.0
    ):
        raise ValueError(f"family_probabilities must contain {len(FAMILY_NAMES)} non-negative values")
    family_probabilities /= family_probabilities.sum()
    family_limits = {
        key: np.asarray(sampling[key], dtype=np.float64)
        for key in (
            "position_error_limits",
            "attitude_error_limits",
            "velocity_error_limits",
            "acceleration_limits",
        )
    }
    for key, values in family_limits.items():
        if values.shape != (len(FAMILY_NAMES), 3) or np.any(values < 0.0):
            raise ValueError(f"{key} must have shape ({len(FAMILY_NAMES)}, 3) with non-negative values")
    for index in range(samples):
        family = int(rng.choice(len(FAMILY_NAMES), p=family_probabilities))
        progress = _sample_progress(
            rng,
            total_length=total_length,
            gate_progress=gate_progress,
            gate_focus_fraction=float(sampling["gate_focus_fraction"]),
            gate_focus_radius=float(sampling["gate_focus_radius"]),
        )
        gate_index = min(int(np.searchsorted(gate_progress, progress, side="left")), len(environment.gates) - 1)
        orientation = np.asarray(environment.gates[gate_index].orientation, dtype=np.float64)
        window = _reference_window(
            waypoints,
            cumulative,
            progress,
            speed=float(environment.reference_speed),
            orientation=orientation,
            dt=dt,
            horizon=horizon,
        )
        position_limit = family_limits["position_error_limits"][family]
        attitude_limit = family_limits["attitude_error_limits"][family]
        velocity_limit = family_limits["velocity_error_limits"][family]
        acceleration_limit = family_limits["acceleration_limits"][family]
        position_error = rng.uniform(-position_limit, position_limit)
        attitude_error = rng.uniform(-attitude_limit, attitude_limit)
        velocity_error = rng.uniform(-velocity_limit, velocity_limit)
        acceleration = rng.uniform(-acceleration_limit, acceleration_limit)
        velocity = _bounded_velocity(
            np.asarray(window[0].velocity, dtype=np.float64) + velocity_error,
            max_speed,
        )
        state = np.concatenate((position_error, attitude_error, velocity, acceleration))
        policy_input = build_policy_input(
            state,
            window,
            input_dim=len(PATH_AWARE_INPUT_FEATURES),
            reference_speed=float(environment.reference_speed),
            context_offsets=offsets,
        )
        started = time.perf_counter()
        control = controller.solve(state, window)
        solve_times[index] = time.perf_counter() - started
        x_values[index] = policy_input
        state_values[index] = state
        controls[index] = control
        contexts[index] = policy_input[12:24].reshape(len(offsets), 3)
        progress_values[index] = progress / max(total_length, 1e-12)
        gate_ids[index] = gate_index + 1
        families[index] = family
        reference_positions[index] = window[0].position
        reference_velocities[index] = window[0].velocity
    return {
        "X": x_values,
        "U": controls,
        "state_X": state_values,
        "path_context": contexts,
        "reference_speed": np.full(samples, environment.reference_speed, dtype=np.float32),
        "reference_position": reference_positions,
        "reference_velocity": reference_velocities,
        "route_progress": progress_values,
        "target_gate_id": gate_ids,
        "sample_family": families,
        "environment_id": np.full(samples, environment.environment_id, dtype=np.int32),
        "episode_id": np.full(samples, environment.environment_id, dtype=np.int32),
        "initial_state_id": families.astype(np.int32),
        "step_id": np.arange(samples, dtype=np.int32),
        "data_source": np.full(samples, 3, dtype=np.int8),
        "mpc_solve_time": solve_times,
    }


def _valid_shard(path: Path, *, samples: int, environment_id: int) -> bool:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if archive["X"].shape != (samples, len(PATH_AWARE_INPUT_FEATURES)):
                return False
            if archive["U"].shape != (samples, 4):
                return False
            if not np.all(archive["environment_id"] == environment_id):
                return False
            return bool(np.isfinite(archive["X"]).all() and np.isfinite(archive["U"]).all())
    except (OSError, ValueError, KeyError, EOFError):
        return False


def _merge_shards(paths: Sequence[Path], output: Path) -> dict[str, Any]:
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
        "input_features": list(input_feature_names(merged["X"].shape[1])),
        "sample_family_counts": {
            FAMILY_NAMES[int(value)]: int(count) for value, count in zip(families, family_counts)
        },
        "target_gate_counts": {str(int(value)): int(count) for value, count in zip(gates, gate_counts)},
        "reference_speed_values": sorted(float(value) for value in np.unique(merged["reference_speed"])),
        "X_min_per_dim": np.min(merged["X"], axis=0).astype(float).tolist(),
        "X_max_per_dim": np.max(merged["X"], axis=0).astype(float).tolist(),
        "U_min_per_dim": np.min(merged["U"], axis=0).astype(float).tolist(),
        "U_max_per_dim": np.max(merged["U"], axis=0).astype(float).tolist(),
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
        config["dataset"]["root"] = f"{config['dataset']['root']}_smoke_v2"
    output = (PROJECT_ROOT / config["dataset"]["root"]).resolve()
    samples = int(config["dataset"]["samples_per_environment"])
    if samples < 1:
        raise ValueError("samples_per_environment must be positive")
    config_hash = _canonical_hash(config)
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and not resume:
        raise FileExistsError(f"output exists: {output}; use --resume to continue")
    output.mkdir(parents=True, exist_ok=True)
    environments = _environment_values(config)
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
            "strategy": "independent_path_state_mpc_queries",
            "input_dimensions": len(PATH_AWARE_INPUT_FEATURES),
            "output_dimensions": 4,
            "environment_count": len(environments),
            "samples_per_environment": samples,
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
        shard_paths = [output / "shards" / f"environment_{index:04d}.npz" for index in range(len(environments))]
        final_name = str(config["dataset"]["output_name"])
        final_path = output / "final" / final_name
        stats = _merge_shards(shard_paths, final_path)
        manifest["status"] = "completed"
        manifest["final_dataset"] = str(Path("final") / final_name)
        manifest["statistics"] = stats
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        atomic_write_json(manifest_path, manifest)
        print(json.dumps(stats, indent=2, ensure_ascii=False), flush=True)
        print(f"Independent MPC dataset completed: {final_path}", flush=True)
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
