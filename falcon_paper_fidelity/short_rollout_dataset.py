"""Short closed-loop rollout queries for a 12-D gate-relative MPC student.

Each rollout starts from an independently sampled recoverable state.  At every
50 ms step the local trajectory is replanned, the N=20 ACADO MPC is solved
again, only its first four-value control is stored, and the quadrotor dynamics
advance one step.  This preserves receding-horizon feedback while adding
temporally coherent, physically reachable states to the imitation dataset.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from airsim_interface import RawDroneState
from dataset_store import atomic_write_json
from state_adapter import normalize_quaternion, quaternion_multiply

from .contracts import INPUT_CONTRACT, INPUT_FEATURES, OUTPUT_FEATURES, build_gate_input
from .motion_primitive import MotionPrimitiveConfig, MotionPrimitiveInfeasible, QuinticMotionPrimitivePlanner
from .recovery_dataset import (
    CONTROL_BUCKET_NAMES,
    _make_controller,
    classify_control_bucket,
    raw_state_from_input,
)


FAMILY_NAMES = (
    "nominal_cruise",
    "slow_speed_recovery",
    "overspeed_recovery",
    "lateral_recovery",
    "vertical_recovery",
    "combined_attitude_recovery",
)


def _config_hash(config: Mapping[str, Any]) -> str:
    text = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def _signed_uniform(
    rng: np.random.Generator, low: float, high: float
) -> float:
    value = float(rng.uniform(low, high))
    return value if rng.random() < 0.5 else -value


def sample_initial_input(
    rng: np.random.Generator,
    family: int,
    *,
    target_speed: float,
) -> np.ndarray:
    """Sample one moderate 10 m/s recovery state without forcing saturation."""

    if family < 0 or family >= len(FAMILY_NAMES):
        raise ValueError("invalid rollout family")
    position_error = np.array(
        [rng.uniform(7.0, 28.0), rng.normal(0.0, 0.22), rng.normal(0.0, 0.18)],
        dtype=np.float64,
    )
    attitude_error = rng.normal(0.0, 0.045, size=3)
    velocity = np.array(
        [rng.normal(target_speed, 0.8), rng.normal(0.0, 0.25), rng.normal(0.0, 0.20)],
        dtype=np.float64,
    )
    acceleration = rng.normal(0.0, 0.45, size=3)

    if family == 1:
        velocity[0] = rng.uniform(3.5, 7.5)
        position_error[0] = rng.uniform(10.0, 28.0)
    elif family == 2:
        velocity[0] = rng.uniform(11.0, 14.0)
        position_error[0] = rng.uniform(10.0, 28.0)
    elif family == 3:
        position_error[1] = _signed_uniform(rng, 0.45, 1.8)
        velocity[1] = _signed_uniform(rng, 0.25, 1.5)
    elif family == 4:
        position_error[2] = _signed_uniform(rng, 0.35, 1.4)
        velocity[2] = _signed_uniform(rng, 0.20, 1.1)
    elif family == 5:
        position_error[1] = _signed_uniform(rng, 0.35, 1.5)
        position_error[2] = _signed_uniform(rng, 0.30, 1.2)
        attitude_error = rng.uniform(-0.18, 0.18, size=3)
        velocity[0] = rng.uniform(6.0, 12.5)
        velocity[1:3] = rng.uniform(-1.2, 1.2, size=2)
        acceleration = rng.uniform(-1.5, 1.5, size=3)

    velocity[0] = float(np.clip(velocity[0], 0.0, 15.0))
    value = np.concatenate((position_error, attitude_error, velocity, acceleration))
    if value.shape != (12,) or not np.isfinite(value).all():
        raise RuntimeError("short-rollout sampler produced an invalid state")
    return value


def _rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = normalize_quaternion(quaternion)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _derivative(state: np.ndarray, control: np.ndarray, gravity: float) -> np.ndarray:
    quaternion = state[3:7]
    velocity = state[7:10]
    thrust = float(control[0])
    rates = np.asarray(control[1:4], dtype=np.float64)
    acceleration = np.array([0.0, 0.0, gravity], dtype=np.float64)
    acceleration += _rotation_matrix(quaternion) @ np.array([0.0, 0.0, -thrust])
    quaternion_dot = 0.5 * quaternion_multiply(quaternion, np.array([0.0, *rates]))
    return np.concatenate((velocity, quaternion_dot, acceleration))


def propagate_state(
    state: RawDroneState,
    control: np.ndarray,
    *,
    dt: float,
    gravity: float,
) -> RawDroneState:
    """Advance the published rigid-body model by one zero-order-hold step."""

    current = np.concatenate((state.position, normalize_quaternion(state.quaternion), state.velocity))
    value = np.asarray(control, dtype=np.float64).reshape(4)
    k1 = _derivative(current, value, gravity)
    k2 = _derivative(current + 0.5 * dt * k1, value, gravity)
    k3 = _derivative(current + 0.5 * dt * k2, value, gravity)
    k4 = _derivative(current + dt * k3, value, gravity)
    following = current + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    following[3:7] = normalize_quaternion(following[3:7])
    acceleration = _derivative(following, value, gravity)[7:10]
    return RawDroneState(
        position=following[0:3].copy(),
        quaternion=following[3:7].copy(),
        velocity=following[7:10].copy(),
        acceleration=acceleration.copy(),
        angular_velocity=value[1:4].copy(),
    )


def _limits(config: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    mpc = config["mpc"]
    lower = np.array(
        [mpc["min_thrust"], -mpc["max_body_rate_xy"], -mpc["max_body_rate_xy"], -mpc["max_body_rate_z"]],
        dtype=np.float64,
    )
    upper = np.array(
        [mpc["max_thrust"], mpc["max_body_rate_xy"], mpc["max_body_rate_xy"], mpc["max_body_rate_z"]],
        dtype=np.float64,
    )
    return lower, upper


def _existing_bucket_counts(output: Path, completed: list[int]) -> np.ndarray:
    counts = np.zeros(3, dtype=np.int64)
    for chunk_id in completed:
        path = output / "chunks" / f"chunk_{chunk_id:05d}.npz"
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as archive:
            values = np.asarray(archive["control_bucket"], dtype=np.int8)
        for bucket in range(3):
            counts[bucket] += int(np.count_nonzero(values == bucket))
    return counts


def generate(config_path: str | Path, *, resume: bool = False) -> Path:
    config_file = Path(config_path).expanduser().resolve()
    config = json.loads(config_file.read_text(encoding="utf-8"))
    output = (config_file.parent / config["dataset"]["root"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    signature = _config_hash(config)
    if manifest_path.exists() and not resume:
        raise FileExistsError(f"output exists: {output}; use --resume")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config_sha256") != signature:
            raise ValueError("resume config differs from recorded short-rollout config")
    else:
        manifest = {
            "schema_version": 1,
            "status": "running",
            "strategy": "falcon_acado_12d_receding_horizon_short_rollout_v1",
            "paper_core": False,
            "input_contract": INPUT_CONTRACT,
            "input_dimensions": 12,
            "input_features": list(INPUT_FEATURES),
            "output_dimensions": 4,
            "output_features": list(OUTPUT_FEATURES),
            "config_sha256": signature,
            "families": list(FAMILY_NAMES),
            "completed_chunks": [],
            "warning": "robustness extension; each row remains X_t -> first MPC control u0_t",
        }
        atomic_write_json(manifest_path, manifest)

    sample_count = int(config["dataset"]["sample_count"])
    rollout_steps = int(config["dataset"]["rollout_steps"])
    chunk_rollouts = int(config["dataset"]["chunk_rollouts"])
    if sample_count < rollout_steps or sample_count % rollout_steps:
        raise ValueError("sample_count must be a positive multiple of rollout_steps")
    if min(rollout_steps, chunk_rollouts) < 1:
        raise ValueError("rollout_steps and chunk_rollouts must be positive")
    rollout_count = sample_count // rollout_steps
    total_chunks = (rollout_count + chunk_rollouts - 1) // chunk_rollouts
    target_speed = float(config["planner"]["nominal_speed"])
    if not np.isclose(target_speed, float(config["planner"]["terminal_speed"])):
        raise ValueError("nominal and terminal speed must match for the 12-D fixed-speed contract")

    dt = float(config["timing"]["control_dt"])
    reference_dt = float(config["mpc"]["prediction_dt"])
    horizon_nodes = int(config["mpc"]["horizon"]) + 1
    gravity = float(config["mpc"]["gravity"])
    gate_position = np.array([0.0, 0.0, -3.0], dtype=np.float64)
    gate_orientation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    planner = QuinticMotionPrimitivePlanner(MotionPrimitiveConfig(**config["planner"]))
    controller, backend = _make_controller(config, config_file)
    lower, upper = _limits(config)

    sampling = config["sampling"]
    saturation_tolerance = float(sampling["saturation_tolerance"])
    near_limit_fraction = float(sampling["near_limit_fraction"])
    max_saturated_fraction = float(sampling["maximum_saturated_fraction"])
    max_near_limit_fraction = float(sampling["maximum_near_limit_fraction"])
    if not (0.0 <= max_saturated_fraction < 1.0 and 0.0 <= max_near_limit_fraction < 1.0):
        raise ValueError("control-bucket fraction limits must lie in [0,1)")
    saturated_budget = int(np.floor(sample_count * max_saturated_fraction))
    near_limit_budget = int(np.floor(sample_count * max_near_limit_fraction))
    max_attempts = int(config["dataset"]["max_candidate_attempts_per_rollout"])

    completed = sorted(int(value) for value in manifest.get("completed_chunks", []))
    bucket_counts = _existing_bucket_counts(output, completed)
    seed = int(config["random_seed"])
    for chunk_id in range(total_chunks):
        path = output / "chunks" / f"chunk_{chunk_id:05d}.npz"
        rollout_start = chunk_id * chunk_rollouts
        rollout_stop = min(rollout_count, rollout_start + chunk_rollouts)
        expected_rows = (rollout_stop - rollout_start) * rollout_steps
        if chunk_id in completed and path.exists():
            with np.load(path, allow_pickle=False) as archive:
                if archive["X"].shape == (expected_rows, 12):
                    continue

        rng = np.random.default_rng(seed + 104729 * chunk_id)
        rows: dict[str, list[Any]] = {
            key: []
            for key in (
                "X", "U", "sample_id", "query_group_id", "rollout_step",
                "sample_family", "control_bucket", "position", "quaternion",
                "velocity", "acceleration", "mpc_solve_time", "candidate_attempts",
            )
        }
        rejected_rollouts = 0
        for rollout_id in range(rollout_start, rollout_stop):
            family = rollout_id % len(FAMILY_NAMES)
            last_error: Exception | None = None
            for attempt in range(max_attempts):
                try:
                    initial_x = sample_initial_input(rng, family, target_speed=target_speed)
                    state = raw_state_from_input(initial_x, gate_position)
                    controller.reset()
                    candidate: list[dict[str, Any]] = []
                    candidate_buckets = np.zeros(3, dtype=np.int64)
                    for step in range(rollout_steps):
                        x = build_gate_input(state, gate_position, gate_orientation)
                        references = planner.reference_window(
                            position=state.position,
                            velocity=state.velocity,
                            acceleration=state.acceleration,
                            gate_position=gate_position,
                            gate_orientation=gate_orientation,
                            dt=reference_dt,
                            horizon=horizon_nodes,
                        )
                        started = time.perf_counter()
                        control = np.asarray(controller.solve(state, references), dtype=np.float64)
                        solve_time = time.perf_counter() - started
                        if control.shape != (4,) or not np.isfinite(control).all():
                            raise RuntimeError("MPC returned an invalid control")
                        bucket = classify_control_bucket(
                            control,
                            lower,
                            upper,
                            saturation_tolerance=saturation_tolerance,
                            near_limit_fraction=near_limit_fraction,
                        )
                        candidate_buckets[bucket] += 1
                        candidate.append(
                            {
                                "X": x.copy(), "U": control.copy(), "rollout_step": step,
                                "position": state.position.copy(), "quaternion": state.quaternion.copy(),
                                "velocity": state.velocity.copy(), "acceleration": state.acceleration.copy(),
                                "mpc_solve_time": solve_time, "control_bucket": bucket,
                            }
                        )
                        state = propagate_state(state, control, dt=dt, gravity=gravity)
                        if not all(
                            np.isfinite(value).all()
                            for value in (state.position, state.quaternion, state.velocity, state.acceleration)
                        ):
                            raise RuntimeError("rollout dynamics produced NaN or Inf")
                    if bucket_counts[2] + candidate_buckets[2] > saturated_budget:
                        raise RuntimeError("global saturated-control budget would be exceeded")
                    if bucket_counts[1] + candidate_buckets[1] > near_limit_budget:
                        raise RuntimeError("global near-limit-control budget would be exceeded")
                except (MotionPrimitiveInfeasible, RuntimeError, ValueError) as exc:
                    last_error = exc
                    rejected_rollouts += 1
                    continue
                bucket_counts += candidate_buckets
                for item in candidate:
                    sample_id = rollout_id * rollout_steps + int(item["rollout_step"])
                    rows["X"].append(item["X"])
                    rows["U"].append(item["U"])
                    rows["sample_id"].append(sample_id)
                    rows["query_group_id"].append(rollout_id)
                    rows["rollout_step"].append(item["rollout_step"])
                    rows["sample_family"].append(family)
                    rows["control_bucket"].append(item["control_bucket"])
                    rows["position"].append(item["position"])
                    rows["quaternion"].append(item["quaternion"])
                    rows["velocity"].append(item["velocity"])
                    rows["acceleration"].append(item["acceleration"])
                    rows["mpc_solve_time"].append(item["mpc_solve_time"])
                    rows["candidate_attempts"].append(attempt)
                break
            else:
                raise RuntimeError(
                    f"could not generate rollout {rollout_id} family={FAMILY_NAMES[family]} "
                    f"after {max_attempts} attempts: {last_error}"
                )

        arrays = {
            "X": np.asarray(rows["X"], dtype=np.float32),
            "U": np.asarray(rows["U"], dtype=np.float32),
            "sample_id": np.asarray(rows["sample_id"], dtype=np.int64),
            "query_group_id": np.asarray(rows["query_group_id"], dtype=np.int64),
            "rollout_step": np.asarray(rows["rollout_step"], dtype=np.int8),
            "sample_family": np.asarray(rows["sample_family"], dtype=np.int8),
            "control_bucket": np.asarray(rows["control_bucket"], dtype=np.int8),
            "position": np.asarray(rows["position"], dtype=np.float32),
            "quaternion": np.asarray(rows["quaternion"], dtype=np.float32),
            "velocity": np.asarray(rows["velocity"], dtype=np.float32),
            "acceleration": np.asarray(rows["acceleration"], dtype=np.float32),
            "mpc_solve_time": np.asarray(rows["mpc_solve_time"], dtype=np.float64),
            "candidate_attempts": np.asarray(rows["candidate_attempts"], dtype=np.int16),
            "reference_speed": np.full(expected_rows, target_speed, dtype=np.float32),
            "data_source": np.full(expected_rows, 2, dtype=np.int8),
        }
        if arrays["X"].shape != (expected_rows, 12) or arrays["U"].shape != (expected_rows, 4):
            raise RuntimeError("short-rollout chunk has an unexpected shape")
        _atomic_npz(path, arrays)
        completed = sorted(set(completed) | {chunk_id})
        manifest.update(
            completed_chunks=completed,
            completed_samples=min(sample_count, rollout_stop * rollout_steps),
            rejected_rollouts=int(manifest.get("rejected_rollouts", 0)) + rejected_rollouts,
            control_bucket_counts={
                CONTROL_BUCKET_NAMES[index]: int(bucket_counts[index]) for index in range(3)
            },
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        atomic_write_json(manifest_path, manifest)
        print(
            f"Short rollouts: {rollout_stop}/{rollout_count} | "
            f"samples={rollout_stop * rollout_steps}/{sample_count} | "
            f"near={bucket_counts[1]} sat={bucket_counts[2]}",
            flush=True,
        )

    merged_parts: dict[str, list[np.ndarray]] = {}
    for chunk_id in range(total_chunks):
        with np.load(output / "chunks" / f"chunk_{chunk_id:05d}.npz", allow_pickle=False) as archive:
            for key in archive.files:
                merged_parts.setdefault(key, []).append(np.asarray(archive[key]))
    merged = {key: np.concatenate(values, axis=0) for key, values in merged_parts.items()}
    final_path = output / "final" / str(config["dataset"]["output_name"])
    _atomic_npz(final_path, merged)
    actual_counts = np.bincount(merged["control_bucket"].astype(np.int64), minlength=3)
    saturated_fraction = float(actual_counts[2] / sample_count)
    near_limit_fraction_actual = float(actual_counts[1] / sample_count)
    if saturated_fraction > max_saturated_fraction + 1e-12:
        raise RuntimeError("final saturated-control fraction exceeds configured maximum")
    if near_limit_fraction_actual > max_near_limit_fraction + 1e-12:
        raise RuntimeError("final near-limit fraction exceeds configured maximum")
    manifest.update(
        status="completed",
        mpc_backend=backend,
        planner_signature=planner.signature(),
        sample_count=sample_count,
        rollout_count=rollout_count,
        rollout_steps=rollout_steps,
        control_dt_seconds=dt,
        prediction_dt_seconds=reference_dt,
        horizon=int(config["mpc"]["horizon"]),
        reference_speed_mps=target_speed,
        final_dataset=str(Path("final") / final_path.name),
        X_shape=list(merged["X"].shape),
        U_shape=list(merged["U"].shape),
        family_counts={
            FAMILY_NAMES[index]: int(np.count_nonzero(merged["sample_family"] == index))
            for index in range(len(FAMILY_NAMES))
        },
        control_bucket_counts={
            CONTROL_BUCKET_NAMES[index]: int(actual_counts[index]) for index in range(3)
        },
        saturated_fraction=saturated_fraction,
        near_limit_fraction=near_limit_fraction_actual,
        maximum_saturated_fraction=max_saturated_fraction,
        maximum_near_limit_fraction=max_near_limit_fraction,
        mean_mpc_solve_time_seconds=float(np.mean(merged["mpc_solve_time"])),
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
    atomic_write_json(manifest_path, manifest)
    print(
        f"Short-rollout dataset complete: X={merged['X'].shape} U={merged['U'].shape} "
        f"near={near_limit_fraction_actual:.3%} saturated={saturated_fraction:.3%}",
        flush=True,
    )
    return output
