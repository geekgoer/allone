"""Phase-consistent recovery queries for the FALCON 12-D controller.

Unlike :mod:`recovery_dataset`, this generator never creates a fresh trajectory
from each perturbed state.  Every query is attached to one time sample of a
previously collected continuous ACADO rollout.  The future MPC window is taken
from that rollout's immutable global trajectory, while only the measured drone
state is perturbed.  This preserves forward path progress and prevents the
student from learning the repeatedly-replanned ``t=0`` action.

The published FALCON input remains unchanged::

    [p_gate - p, wrap(Euler_gate - Euler_drone), velocity, acceleration]

This dataset is a disclosed robustness extension to the paper's 40k protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
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

from .acado_mpc import AcadoMPCConfig, AcadoRtiMPC
from .contracts import (
    OUTPUT_FEATURES,
    PAPER_GATE_EULER_INPUT_CONTRACT,
    PAPER_GATE_EULER_INPUT_FEATURES,
    build_paper_gate_euler_input,
)
from .motion_primitive import (
    ContinuousQuinticTrajectory,
    MotionPrimitiveConfig,
    QuinticMotionPrimitivePlanner,
)
from .recovery_dataset import CONTROL_BUCKET_NAMES, classify_control_bucket


FAMILY_NAMES = (
    "nominal_jitter",
    "lateral_recovery",
    "vertical_recovery",
    "attitude_recovery",
    "speed_recovery",
    "combined_recovery",
)


@dataclass(frozen=True)
class _Gate:
    index: int
    center: np.ndarray
    orientation: np.ndarray


def _canonical_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _resolve(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


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


def _quaternion_from_rotation_vector(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(value))
    if angle < 1e-12:
        return normalize_quaternion(np.array([1.0, *(0.5 * value)]))
    axis = value / angle
    return np.array([np.cos(0.5 * angle), *(axis * np.sin(0.5 * angle))])


def _rotate(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    q = normalize_quaternion(quaternion)
    inverse = np.array([q[0], -q[1], -q[2], -q[3]])
    pure = np.array([0.0, *np.asarray(vector, dtype=np.float64).reshape(3)])
    return quaternion_multiply(quaternion_multiply(q, pure), inverse)[1:]


def _sample_clipped(
    rng: np.random.Generator, standard_deviation: Any, absolute_limit: Any
) -> np.ndarray:
    sigma = np.asarray(standard_deviation, dtype=np.float64).reshape(3)
    limit = np.asarray(absolute_limit, dtype=np.float64).reshape(3)
    return np.clip(rng.normal(0.0, sigma), -limit, limit)


def _limit_norm(value: np.ndarray, maximum: float) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm <= maximum or norm < 1e-12:
        return value
    return value * (maximum / norm)


def _load_gates(path: Path) -> dict[int, tuple[_Gate, ...]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[int, tuple[_Gate, ...]] = {}
    for environment in payload:
        environment_id = int(environment["environment_id"])
        result[environment_id] = tuple(
            _Gate(
                index=int(gate["index"]),
                center=np.asarray(gate["center"], dtype=np.float64),
                orientation=normalize_quaternion(gate["orientation"]),
            )
            for gate in environment["gates"]
        )
    return result


def _build_trajectories(
    source: Mapping[str, np.ndarray],
    environments: Mapping[int, tuple[_Gate, ...]],
    planner: QuinticMotionPrimitivePlanner,
    *,
    start_velocity_mode: str,
    nominal_speed: float,
    final_exit_distance: float,
) -> dict[int, ContinuousQuinticTrajectory]:
    trajectories: dict[int, ContinuousQuinticTrajectory] = {}
    episode_id = source["episode_id"].astype(np.int64)
    for episode in np.unique(episode_id):
        first = int(np.flatnonzero(episode_id == episode)[0])
        gates = environments[int(source["environment_id"][first])]
        if start_velocity_mode not in {"measured", "nominal_gate_axis"}:
            raise ValueError(
                "trajectory.start_velocity_mode must be 'measured' or "
                f"'nominal_gate_axis', got {start_velocity_mode!r}"
            )
        # The controller reads one more AirSim sample after constructing the
        # trajectory. Consequently position[first] and velocity[first] are not
        # the exact polynomial boundary. The immutable trajectory-relative X
        # still records v_ref-v and a_ref, while reference_position records
        # p_ref, so recover the precise t=0 boundary from those arrays.
        start_position = source["reference_position"][first]
        start_velocity = source["X"][first, 6:9] + source["velocity"][first]
        start_acceleration = source["X"][first, 9:12]
        if start_velocity_mode == "nominal_gate_axis":
            configured = nominal_speed * planner.passage_axis(gates[0].orientation)
            if not np.allclose(start_velocity, configured, atol=1e-5, rtol=1e-5):
                raise ValueError(
                    "source trajectory does not use the configured nominal gate-axis "
                    f"start velocity: episode={int(episode)} "
                    f"stored={start_velocity.tolist()} configured={configured.tolist()}"
                )
        trajectories[int(episode)] = planner.plan_continuous_route(
            start_position,
            start_velocity,
            start_acceleration,
            gates,
            final_exit_distance=final_exit_distance,
        )
    return trajectories


def _validate_reconstruction(
    source: Mapping[str, np.ndarray],
    trajectories: Mapping[int, ContinuousQuinticTrajectory],
    *,
    control_dt: float,
    tolerance: float,
) -> float:
    largest = 0.0
    for index in range(len(source["X"])):
        trajectory = trajectories[int(source["episode_id"][index])]
        expected, _, _, _ = trajectory.evaluate(float(source["step_id"][index]) * control_dt)
        error = float(np.max(np.abs(expected - source["reference_position"][index])))
        largest = max(largest, error)
    if largest > tolerance:
        raise ValueError(
            "continuous trajectory reconstruction does not match the immutable core: "
            f"max_abs={largest:.3e}, tolerance={tolerance:.3e}. Check the start-velocity mode."
        )
    return largest


def _controller(config: Mapping[str, Any], config_file: Path) -> AcadoRtiMPC:
    values = dict(config["mpc"])
    backend = str(values.pop("backend", "official_rpg_mpc_acado_qpoases")).lower()
    if backend not in {"official_rpg_mpc_acado_qpoases", "acado", "acado_qpoases"}:
        raise ValueError("phase-consistent FALCON recovery requires the ACADO backend")
    library = _resolve(values["library"], config_file.parent)
    values["library"] = str(library)
    return AcadoRtiMPC(AcadoMPCConfig(**values))


def _perturb_state(
    source: Mapping[str, np.ndarray],
    base_index: int,
    gate: _Gate,
    family: Mapping[str, Any],
    rng: np.random.Generator,
    *,
    maximum_speed: float,
    maximum_acceleration: float,
) -> tuple[RawDroneState, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    position_delta_local = _sample_clipped(
        rng, family["position_std"], family["position_limit"]
    )
    velocity_delta_local = _sample_clipped(
        rng, family["velocity_std"], family["velocity_limit"]
    )
    acceleration_delta_local = _sample_clipped(
        rng, family["acceleration_std"], family["acceleration_limit"]
    )
    attitude_delta = _sample_clipped(
        rng, family["attitude_std"], family["attitude_limit"]
    )
    position = source["position"][base_index] + _rotate(
        gate.orientation, position_delta_local
    )
    velocity = _limit_norm(
        source["velocity"][base_index]
        + _rotate(gate.orientation, velocity_delta_local),
        maximum_speed,
    )
    acceleration = _limit_norm(
        source["acceleration"][base_index]
        + _rotate(gate.orientation, acceleration_delta_local),
        maximum_acceleration,
    )
    quaternion = normalize_quaternion(
        quaternion_multiply(
            _quaternion_from_rotation_vector(attitude_delta),
            source["quaternion"][base_index],
        )
    )
    state = RawDroneState(
        position=position,
        quaternion=quaternion,
        velocity=velocity,
        acceleration=acceleration,
        angular_velocity=np.zeros(3),
    )
    return state, position_delta_local, attitude_delta, velocity_delta_local, acceleration_delta_local


def generate(
    config_path: str | Path, *, resume: bool = False, smoke_test: bool = False
) -> Path:
    config_file = Path(config_path).expanduser().resolve()
    config = json.loads(config_file.read_text(encoding="utf-8"))
    if smoke_test:
        dataset = config["dataset"]
        dataset["root"] = str(dataset["root"]) + "_smoke"
        output_name = Path(str(dataset["output_name"]))
        dataset["output_name"] = f"{output_name.stem}_smoke{output_name.suffix}"
        dataset["sample_count"] = 1200
        dataset["chunk_size"] = 1200
    base = config_file.parent
    source_path = _resolve(config["source"]["dataset"], base)
    environments_path = _resolve(config["source"]["environments"], base)
    output = _resolve(config["dataset"]["root"], base)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    signature = _canonical_hash(config)
    if manifest_path.exists() and not resume:
        raise FileExistsError(f"output exists: {output}; use --resume")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config_sha256") != signature:
            raise ValueError("resume config differs from the recorded phase-recovery config")
    else:
        manifest = {
            "schema_version": 1,
            "status": "running",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "strategy": "falcon_acado_12d_phase_consistent_recovery_v1",
            "paper_core": False,
            "paper_relationship": "disclosed robustness extension to the continuous 40k protocol",
            "input_contract": PAPER_GATE_EULER_INPUT_CONTRACT,
            "input_dimensions": 12,
            "input_features": list(PAPER_GATE_EULER_INPUT_FEATURES),
            "output_dimensions": 4,
            "output_features": list(OUTPUT_FEATURES),
            "config_sha256": signature,
            "sample_count": int(config["dataset"]["sample_count"]),
            "chunk_size": int(config["dataset"]["chunk_size"]),
            "completed_chunks": [],
            "families": [item["name"] for item in config["sampling"]["families"]],
            "source_dataset": str(source_path),
            "source_environments": str(environments_path),
            "procrastination_guard": (
                "future references stay attached to source trajectory phase; "
                "no trajectory is replanned from a perturbed state"
            ),
        }
        atomic_write_json(manifest_path, manifest)

    with np.load(source_path, allow_pickle=False) as archive:
        source = {key: np.asarray(archive[key]) for key in archive.files}
    required = {
        "X", "U", "episode_id", "initial_state_id", "environment_id", "step_id", "target_gate_id",
        "position", "quaternion", "velocity", "acceleration", "reference_position",
    }
    missing = sorted(required.difference(source))
    if missing:
        raise ValueError(f"continuous source is missing arrays: {missing}")
    if source["X"].shape != (40000, 12) or source["U"].shape != (40000, 4):
        raise ValueError("phase recovery requires the immutable 40000-row continuous core")

    planner = QuinticMotionPrimitivePlanner(MotionPrimitiveConfig(**config["planner"]))
    environments = _load_gates(environments_path)
    trajectory_cfg = config["trajectory"]
    trajectories = _build_trajectories(
        source,
        environments,
        planner,
        start_velocity_mode=str(trajectory_cfg["start_velocity_mode"]).lower(),
        nominal_speed=float(config["planner"]["nominal_speed"]),
        final_exit_distance=float(trajectory_cfg["final_exit_distance"]),
    )
    reconstruction_error = _validate_reconstruction(
        source,
        trajectories,
        control_dt=float(config["timing"]["dt"]),
        tolerance=float(config["source"].get("reconstruction_tolerance", 1e-5)),
    )
    manifest["trajectory_reconstruction_max_abs"] = reconstruction_error
    manifest["planner_signature"] = planner.signature()
    atomic_write_json(manifest_path, manifest)

    controller = _controller(config, config_file)
    count = int(config["dataset"]["sample_count"])
    chunk_size = int(config["dataset"]["chunk_size"])
    seed = int(config["random_seed"])
    control_dt = float(config["timing"]["dt"])
    prediction_dt = float(config["mpc"]["prediction_dt"])
    reference_nodes = int(config["mpc"]["horizon"]) + 1
    min_step = int(config["sampling"].get("minimum_source_step", 2))
    eligible = np.flatnonzero(source["step_id"].astype(np.int64) >= min_step)
    if eligible.size == 0:
        raise ValueError("no source rows satisfy sampling.minimum_source_step")
    selector = np.random.default_rng(seed + 17)
    base_indices = np.concatenate(
        [selector.permutation(eligible) for _ in range((count + len(eligible) - 1) // len(eligible))]
    )[:count]
    families = list(config["sampling"]["families"])
    if tuple(item["name"] for item in families) != FAMILY_NAMES:
        raise ValueError(f"sampling family order must be {FAMILY_NAMES}")
    family_index_by_name = {
        str(family["name"]): index for index, family in enumerate(families)
    }
    maximum_speed = float(config["sampling"]["maximum_total_speed"])
    maximum_acceleration = float(config["sampling"]["maximum_total_acceleration"])
    bucket_slots = config["sampling"].get(
        "control_bucket_slots", {"interior": 12, "near_limit": 5, "saturated": 3}
    )
    bucket_pattern = [
        bucket
        for bucket, name in enumerate(CONTROL_BUCKET_NAMES)
        for _ in range(int(bucket_slots[name]))
    ]
    if not bucket_pattern:
        raise ValueError("sampling.control_bucket_slots must not be empty")
    default_eligible_families = {
        "interior": list(FAMILY_NAMES),
        "near_limit": list(FAMILY_NAMES[1:]),
        "saturated": ["attitude_recovery", "speed_recovery", "combined_recovery"],
    }
    configured_eligible = config["sampling"].get(
        "control_bucket_eligible_families", default_eligible_families
    )
    eligible_family_indices: dict[int, tuple[int, ...]] = {}
    for bucket_index, bucket_name in enumerate(CONTROL_BUCKET_NAMES):
        names = tuple(str(name) for name in configured_eligible.get(bucket_name, ()))
        if not names:
            raise ValueError(
                f"sampling.control_bucket_eligible_families.{bucket_name} must not be empty"
            )
        unknown = [name for name in names if name not in family_index_by_name]
        if unknown:
            raise ValueError(
                f"unknown families for control bucket {bucket_name}: {unknown}"
            )
        eligible_family_indices[bucket_index] = tuple(
            family_index_by_name[name] for name in names
        )
    manifest["control_bucket_names"] = list(CONTROL_BUCKET_NAMES)
    manifest["control_bucket_slots"] = {
        name: int(bucket_slots[name]) for name in CONTROL_BUCKET_NAMES
    }
    manifest["control_bucket_eligible_families"] = {
        name: [families[index]["name"] for index in eligible_family_indices[bucket]]
        for bucket, name in enumerate(CONTROL_BUCKET_NAMES)
    }
    maximum_attempts = int(config["sampling"].get("maximum_attempts_per_sample", 200))
    if maximum_attempts < 1:
        raise ValueError("sampling.maximum_attempts_per_sample must be positive")
    saturation_tolerance = float(config["sampling"].get("saturation_tolerance", 1e-3))
    near_limit_fraction = float(config["sampling"].get("near_limit_fraction", 0.15))
    lower = np.array(
        [
            float(config["mpc"]["min_thrust"]),
            -float(config["mpc"]["max_body_rate_xy"]),
            -float(config["mpc"]["max_body_rate_xy"]),
            -float(config["mpc"]["max_body_rate_z"]),
        ]
    )
    upper = np.array(
        [
            float(config["mpc"]["max_thrust"]),
            float(config["mpc"]["max_body_rate_xy"]),
            float(config["mpc"]["max_body_rate_xy"]),
            float(config["mpc"]["max_body_rate_z"]),
        ]
    )
    total_chunks = (count + chunk_size - 1) // chunk_size
    completed = set(int(item) for item in manifest.get("completed_chunks", []))
    for chunk_id in range(total_chunks):
        start = chunk_id * chunk_size
        stop = min(count, start + chunk_size)
        expected = stop - start
        chunk_path = output / "chunks" / f"chunk_{chunk_id:05d}.npz"
        if chunk_id in completed and chunk_path.exists():
            with np.load(chunk_path, allow_pickle=False) as archive:
                if archive["X"].shape == (expected, 12) and archive["U"].shape == (expected, 4):
                    continue
        rows: dict[str, list[np.ndarray | float | int]] = {
            key: []
            for key in (
                "X", "U", "sample_id", "base_row_id", "base_episode_id",
                "initial_state_id", "environment_id", "base_step_id", "target_gate_id",
                "sample_family", "position", "quaternion", "velocity", "acceleration",
                "position_delta_gate_frame", "attitude_delta_rotvec",
                "velocity_delta_gate_frame", "acceleration_delta_gate_frame",
                "reference_position", "reference_velocity", "reference_acceleration",
                "mpc_solve_time", "control_bucket", "control_saturated",
                "rejected_attempts", "data_source",
            )
        }
        for global_index in range(start, stop):
            # Select the desired output-space bucket first.  Do not demand an
            # intrinsically mild family such as nominal_jitter to manufacture
            # a saturated command: that cross-product is physically
            # contradictory and can make rejection sampling non-terminating.
            target_bucket = bucket_pattern[global_index % len(bucket_pattern)]
            eligible = eligible_family_indices[target_bucket]
            for attempt in range(maximum_attempts):
                family_index = eligible[
                    (global_index // len(bucket_pattern) + attempt) % len(eligible)
                ]
                family = families[family_index]
                # Vary the source phase as well as the perturbation when a
                # requested control bucket cannot be reached. The accepted
                # query remains attached to exactly one immutable phase.
                base_slot = (global_index + attempt * 7919) % count
                base_index = int(base_indices[base_slot])
                rng = np.random.default_rng(
                    seed + 1_000_003 * global_index + 65_537 * attempt
                )
                environment_id = int(source["environment_id"][base_index])
                gate_id = int(source["target_gate_id"][base_index])
                gates = environments[environment_id]
                gate = next(item for item in gates if item.index == gate_id)
                state, dp, dr, dv, da = _perturb_state(
                    source,
                    base_index,
                    gate,
                    family,
                    rng,
                    maximum_speed=maximum_speed,
                    maximum_acceleration=maximum_acceleration,
                )
                elapsed = float(source["step_id"][base_index]) * control_dt
                references = planner.reference_window_for_trajectory(
                    trajectories[int(source["episode_id"][base_index])],
                    elapsed=elapsed,
                    dt=prediction_dt,
                    horizon=reference_nodes,
                )
                controller.reset()
                started = time.perf_counter()
                control = controller.solve(state, references)
                solve_time = time.perf_counter() - started
                actual_bucket = classify_control_bucket(
                    control,
                    lower,
                    upper,
                    saturation_tolerance=saturation_tolerance,
                    near_limit_fraction=near_limit_fraction,
                )
                if actual_bucket == target_bucket:
                    break
            else:
                raise RuntimeError(
                    f"could not sample control bucket {CONTROL_BUCKET_NAMES[target_bucket]} "
                    f"for sample={global_index} using eligible_families="
                    f"{[families[index]['name'] for index in eligible]!r} "
                    f"after {maximum_attempts} attempts"
                )
            x = build_paper_gate_euler_input(
                state, gate.center, gate.orientation
            )
            saturated = actual_bucket == 2
            values: dict[str, np.ndarray | float | int] = {
                "X": x,
                "U": control,
                "sample_id": global_index,
                "base_row_id": base_index,
                "base_episode_id": int(source["episode_id"][base_index]),
                "initial_state_id": int(source["initial_state_id"][base_index]),
                "environment_id": environment_id,
                "base_step_id": int(source["step_id"][base_index]),
                "target_gate_id": gate_id,
                "sample_family": family_index,
                "position": state.position,
                "quaternion": state.quaternion,
                "velocity": state.velocity,
                "acceleration": state.acceleration,
                "position_delta_gate_frame": dp,
                "attitude_delta_rotvec": dr,
                "velocity_delta_gate_frame": dv,
                "acceleration_delta_gate_frame": da,
                "reference_position": references[0].position,
                "reference_velocity": references[0].velocity,
                "reference_acceleration": references[0].acceleration,
                "mpc_solve_time": solve_time,
                "control_bucket": actual_bucket,
                "control_saturated": int(saturated),
                "rejected_attempts": attempt,
                "data_source": 1,
            }
            for key, value in values.items():
                rows[key].append(value)
        arrays = {key: np.asarray(value) for key, value in rows.items()}
        _atomic_npz(chunk_path, arrays)
        completed.add(chunk_id)
        manifest["completed_chunks"] = sorted(completed)
        manifest["completed_samples"] = min(count, len(completed) * chunk_size)
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        atomic_write_json(manifest_path, manifest)
        print(
            f"Phase recovery chunks: {len(completed)}/{total_chunks} | "
            f"samples through {stop}/{count}",
            flush=True,
        )

    merged: dict[str, list[np.ndarray]] = {}
    for chunk_id in range(total_chunks):
        with np.load(output / "chunks" / f"chunk_{chunk_id:05d}.npz", allow_pickle=False) as archive:
            for key in archive.files:
                merged.setdefault(key, []).append(np.asarray(archive[key]))
    final_arrays = {key: np.concatenate(value, axis=0) for key, value in merged.items()}
    if final_arrays["X"].shape != (count, 12) or final_arrays["U"].shape != (count, 4):
        raise RuntimeError("merged phase-recovery dataset has unexpected X/U shapes")
    final_path = output / "final" / str(config["dataset"]["output_name"])
    _atomic_npz(final_path, final_arrays)
    manifest.update(
        status="complete",
        completed_samples=count,
        final_dataset=str(final_path.relative_to(output)),
        statistics={
            "X_shape": list(final_arrays["X"].shape),
            "U_shape": list(final_arrays["U"].shape),
            "family_counts": {
                name: int(np.count_nonzero(final_arrays["sample_family"] == index))
                for index, name in enumerate(FAMILY_NAMES)
            },
            "saturated_fraction": float(np.mean(final_arrays["control_saturated"])),
            "control_bucket_counts": {
                name: int(np.count_nonzero(final_arrays["control_bucket"] == index))
                for index, name in enumerate(CONTROL_BUCKET_NAMES)
            },
            "mean_rejected_attempts": float(np.mean(final_arrays["rejected_attempts"])),
            "mean_mpc_solve_ms": float(np.mean(final_arrays["mpc_solve_time"]) * 1e3),
            "reference_speed_values": [float(config["planner"]["nominal_speed"])],
            "minimum_source_step": min_step,
        },
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
    atomic_write_json(manifest_path, manifest)
    print(f"Phase-consistent recovery complete: X={final_arrays['X'].shape} U={final_arrays['U'].shape}")
    print(f"Dataset: {final_path}")
    return final_path
