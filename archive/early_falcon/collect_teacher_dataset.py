"""Collect the FALCON-compatible MPC teacher dataset in one AirSim session."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import numpy as np

from airsim_interface import AirSimDrone, require_airsim, vehicle_local_to_world_offset
from airsim_process import AirSimLaunchConfig, AirSimProcessError, AirSimProcessManager
from config import ThrustCalibration, load_calibration
from control_adapter import send_control
from dataset_store import (
    EpisodeSamples,
    atomic_write_json,
    is_valid_episode,
    merge_episodes,
    save_episode_atomic,
    validate_merged_dataset,
    write_summary,
)
from environment import Gate
from gate_route import ContinuousGateRoute, GatePassageError
from gate_visualization import cleanup_generated_gates, spawn_gate_sequence
from initial_state_generator import InitialState, InitialStateGenerator
from mpc_controller import MPCController
from state_adapter import build_controller_state
from training_environment_generator import (
    TrainingEnvironment,
    generate_training_environments,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "teacher_collection.json"
TEACHER_POLICY_VERSION = "continuous_gate_linear_mpc_v2"


class EpisodeFailure(RuntimeError):
    """Raised when an episode cannot produce a valid fixed-length sample set."""


def _resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def load_collection_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("collection config must contain a JSON object")
    return config, config_path


def _initial_states(config: Mapping[str, Any]) -> list[InitialState]:
    spec = config["initial_state"]
    return InitialStateGenerator(
        count=int(spec["count"]),
        seed=int(spec["seed"]),
        position_range=spec["position_range"],
        attitude_range=spec["attitude_range"],
        velocity_range=spec["velocity_range"],
        angular_velocity_range=spec["angular_velocity_range"],
    ).generate()


def _training_environments(config: Mapping[str, Any]) -> list[TrainingEnvironment]:
    spec = config["environment"]
    return generate_training_environments(
        count=int(spec["count"]),
        seed=int(spec["seed"]),
        epsilon_values=spec["epsilon_values"],
        gate_count=int(spec["gate_count"]),
        x_start=float(spec["x_start"]),
        x_spacing=float(spec["x_spacing"]),
        base_y_range=spec["base_y_range"],
        base_z_range=spec["base_z_range"],
        max_z=float(spec["max_z"]),
        speed_values=spec.get("speed_values", (float(config["mpc"]["speed"]),)),
        x_spacing_values=spec.get("x_spacing_values"),
        x_start_values=spec.get("x_start_values"),
        x_spacing_overrides=spec.get("x_spacing_overrides"),
    )


def _write_json_list(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(values, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _save_generation_artifacts(root: Path, states: list[InitialState], environments: list[TrainingEnvironment]) -> None:
    _write_json_list(root / "initial_states.json", [state.to_dict() for state in states])
    _write_json_list(root / "environments.json", [environment.to_dict() for environment in environments])


def _episode_key(episode_id: int) -> str:
    return f"episode_{episode_id:04d}"


def _episode_path(root: Path, episode_id: int) -> Path:
    return root / "episodes" / f"{_episode_key(episode_id)}.npz"


def _new_manifest(
    *,
    config_path: Path,
    config: Mapping[str, Any],
    states: list[InitialState],
    environments: list[TrainingEnvironment],
    steps: int,
    vehicle_name: str,
) -> dict[str, Any]:
    episode_count = len(states) * len(environments)
    episodes = {}
    for episode_id in range(episode_count):
        state_id = episode_id // len(environments)
        environment_id = episode_id % len(environments)
        episodes[_episode_key(episode_id)] = {
            "episode_id": episode_id,
            "initial_state_id": state_id,
            "environment_id": environment_id,
            "status": "pending",
            "attempts": 0,
            "sample_count": 0,
            "file": str(Path("episodes") / f"{_episode_key(episode_id)}.npz"),
            "last_error": None,
        }
    return {
        "schema_version": 2,
        "teacher_policy_version": TEACHER_POLICY_VERSION,
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "config_file": str(config_path),
        "vehicle_name": vehicle_name,
        "initial_state_count": len(states),
        "environment_count": len(environments),
        "initial_states": [state.to_dict() for state in states],
        "environments": [environment.to_dict() for environment in environments],
        "episode_count": episode_count,
        "steps_per_episode": steps,
        "expected_samples": episode_count * steps,
        "rpc_restarts": 0,
        "episodes": episodes,
    }


def _update_manifest(path: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(path, manifest)


def _environment_resume_signature(value: Mapping[str, Any]) -> tuple[Any, ...]:
    """Return stable route parameters used to protect completed episodes."""

    gates = tuple(
        (
            int(gate["index"]),
            tuple(float(item) for item in gate["center"]),
            tuple(float(item) for item in gate["orientation"]),
        )
        for gate in value.get("gates", [])
    )
    default_spacing = (
        float(value["gates"][1]["center"][0] - value["gates"][0]["center"][0])
        if len(value.get("gates", [])) > 1
        else 10.0
    )
    default_start = (
        float(value["gates"][0]["center"][0]) if value.get("gates") else 10.0
    )
    return (
        int(value.get("seed", 0)),
        float(value.get("epsilon", 0.0)),
        str(value.get("difficulty", "")),
        float(value.get("reference_speed", 2.0)),
        float(value.get("x_start", default_start)),
        float(value.get("x_spacing", default_spacing)),
        gates,
    )


def _prepare_dataset(
    root: Path,
    config_path: Path,
    config: dict[str, Any],
    *,
    resume: bool,
    smoke_test: bool,
) -> tuple[list[InitialState], list[TrainingEnvironment], dict[str, Any], Path]:
    effective = deepcopy(config)
    if smoke_test:
        effective["initial_state"]["count"] = 2
        effective["environment"]["count"] = 2
        effective["timing"]["steps_per_episode"] = 20
        effective["timing"]["episode_seconds"] = 1.0
        effective["timing"]["episode_timeout_seconds"] = 3.0
        # A 20-sample smoke run is only a lifecycle/RPC/storage check; it is
        # intentionally too short to validate a five-gate route.
        effective["dataset"]["require_all_gates"] = False
        effective["dataset"]["required_gate_ids"] = []
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if manifest_path.exists() and not resume:
        raise FileExistsError(
            f"Dataset output already exists: {root}. Use --resume to continue it."
        )
    states = _initial_states(effective)
    environments = _training_environments(effective)
    steps = int(effective["timing"]["steps_per_episode"])
    if float(effective["timing"]["episode_seconds"]) != steps * float(effective["timing"]["dt"]):
        raise ValueError("episode_seconds must equal steps_per_episode * dt")
    vehicle_name = str(effective["vehicle_name"])
    if not vehicle_name:
        raise ValueError("vehicle_name must not be empty")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        policy_version = manifest.get("teacher_policy_version")
        if policy_version != TEACHER_POLICY_VERSION:
            raise ValueError(
                "resume dataset was generated by a different teacher policy: "
                f"expected={TEACHER_POLICY_VERSION!r}, actual={policy_version!r}. "
                "Use a new --output-root instead of mixing old and new episodes."
            )
        expected = (len(states), len(environments), steps, vehicle_name)
        actual = (
            int(manifest["initial_state_count"]),
            int(manifest["environment_count"]),
            int(manifest["steps_per_episode"]),
            str(manifest["vehicle_name"]),
        )
        if actual != expected:
            raise ValueError(f"resume config does not match manifest: expected={expected}, actual={actual}")
        old_environments = manifest.get("environments")
        if isinstance(old_environments, list) and len(old_environments) == len(environments):
            completed_environment_ids = {
                int(entry["environment_id"])
                for entry in manifest.get("episodes", {}).values()
                if entry.get("status") == "completed"
            }
            for environment_id in completed_environment_ids:
                old_signature = _environment_resume_signature(old_environments[environment_id])
                new_signature = _environment_resume_signature(environments[environment_id].to_dict())
                if old_signature != new_signature:
                    raise ValueError(
                        "resume would change an environment used by a completed episode: "
                        f"environment_id={environment_id}. Use a new output root instead."
                    )
    else:
        manifest = _new_manifest(
            config_path=config_path,
            config=effective,
            states=states,
            environments=environments,
            steps=steps,
            vehicle_name=vehicle_name,
        )
    # Keep the compact JSON artifacts and a self-contained manifest in sync;
    # this also upgrades manifests created by an earlier collector revision.
    manifest["initial_states"] = [state.to_dict() for state in states]
    manifest["environments"] = [environment.to_dict() for environment in environments]
    _save_generation_artifacts(root, states, environments)
    (root / "collection_config.json").write_text(
        json.dumps(effective, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for episode_id in range(len(states) * len(environments)):
        entry = manifest["episodes"][_episode_key(episode_id)]
        path = _episode_path(root, episode_id)
        if entry.get("status") == "completed" and not is_valid_episode(
            path, expected_steps=steps, expected_episode_id=episode_id
        ):
            entry["status"] = "pending"
            entry["sample_count"] = 0
            entry["last_error"] = "completed entry had a missing or invalid episode file"
    manifest["status"] = "running"
    _update_manifest(manifest_path, manifest)
    return states, environments, manifest, manifest_path


def _airsim_pose(airsim: Any, state: InitialState) -> Any:
    return airsim.Pose(
        position_val=airsim.Vector3r(*[float(value) for value in state.position]),
        orientation_val=airsim.Quaternionr(
            x_val=float(state.orientation[1]),
            y_val=float(state.orientation[2]),
            z_val=float(state.orientation[3]),
            w_val=float(state.orientation[0]),
        ),
    )


def _join(future: Any) -> None:
    join = getattr(future, "join", None)
    if callable(join):
        join()


def _reset_vehicle_for_episode(
    client: Any,
    initial_state: InitialState,
    vehicle_name: str,
    *,
    settle_seconds: float = 0.0,
) -> AirSimDrone:
    client.reset()
    if settle_seconds > 0.0:
        time.sleep(settle_seconds)
    try:
        client.simPause(True)
    except Exception:
        pass
    airsim = require_airsim()
    client.simSetVehiclePose(
        _airsim_pose(airsim, initial_state),
        ignore_collision=True,
        vehicle_name=vehicle_name,
    )
    drone = AirSimDrone(client, vehicle_name)
    drone.enable_api_control(True)
    drone.arm()
    return drone


class GateTargetTracker(ContinuousGateRoute):
    """Backward-compatible name for the continuous gate-route tracker."""

    def __init__(
        self,
        gates: tuple[Gate, ...],
        start_position: np.ndarray,
        *,
        approach_distance: float,
        exit_distance: float,
        threshold: float,
        stabilization_speed: float,
        reference_speed: float,
        start_time: float,
        gate_width: float = 4.0,
        gate_height: float = 4.0,
        gate_thickness: float = 0.15,
        clearance: float = 0.25,
    ) -> None:
        # Kept for old configuration files; continuous tracking no longer
        # waits for this speed before advancing to the next route segment.
        self.stabilization_speed = float(stabilization_speed)
        super().__init__(
            gates,
            start_position,
            approach_distance=approach_distance,
            exit_distance=exit_distance,
            reference_speed=reference_speed,
            start_time=start_time,
            gate_width=gate_width,
            gate_height=gate_height,
            gate_thickness=gate_thickness,
            clearance=clearance,
            completion_radius=threshold,
        )


def _check_state_and_collision(client: Any, drone: AirSimDrone, state: Any, config: Mapping[str, Any]) -> None:
    if not all(
        np.isfinite(value).all()
        for value in (state.position, state.quaternion, state.velocity, state.acceleration, state.angular_velocity)
    ):
        raise EpisodeFailure("AirSim returned NaN or Inf in the drone state")
    bounds = config.get("bounds", {})
    if bounds:
        lower = np.asarray(bounds["position_min"], dtype=np.float64)
        upper = np.asarray(bounds["position_max"], dtype=np.float64)
        if np.any(state.position < lower) or np.any(state.position > upper):
            raise EpisodeFailure(f"drone left configured position bounds: {state.position.tolist()}")
    collision = client.simGetCollisionInfo(vehicle_name=drone.vehicle_name)
    if bool(getattr(collision, "has_collided", False)):
        object_name = getattr(collision, "object_name", "unknown")
        raise EpisodeFailure(f"drone collision with {object_name}")


def run_episode(
    client: Any,
    initial_state: InitialState,
    environment: TrainingEnvironment,
    *,
    episode_id: int,
    config: Mapping[str, Any],
    calibration: ThrustCalibration,
) -> EpisodeSamples:
    vehicle_name = str(config["vehicle_name"])
    timing = config["timing"]
    dt = float(timing["dt"])
    steps = int(timing["steps_per_episode"])
    episode_timeout = float(
        timing.get("episode_timeout_seconds", max(2.0 * steps * dt, steps * dt + 5.0))
    )
    if episode_timeout <= steps * dt:
        raise ValueError("episode_timeout_seconds must exceed the nominal episode duration")
    gate_config = config["gate"]
    mpc_config = config["mpc"]
    dataset_config = config.get("dataset", {})
    record_environment_metadata = bool(dataset_config.get("record_environment_metadata", False))
    record_path_context = bool(dataset_config.get("record_path_context", False))
    context_offsets = tuple(
        int(value) for value in dataset_config.get("path_context_offsets", (0, 5, 10, 19))
    )
    if record_path_context and (
        not context_offsets or any(value < 0 for value in context_offsets)
    ):
        raise ValueError("path_context_offsets must contain non-negative offsets")
    drone = _reset_vehicle_for_episode(
        client,
        initial_state,
        vehicle_name,
        settle_seconds=float(timing.get("reset_settle_seconds", 0.0)),
    )
    scene = None
    simulation_unpaused = False
    try:
        # Remove gates from both this collector and the interactive run
        # scripts before spawning the current environment.
        if bool(gate_config.get("spawn_in_simulation", True)):
            cleanup_generated_gates(client, prefix="FalconGate_")
            cleanup_generated_gates(client, prefix="FalconDatasetGate_")
            scene = spawn_gate_sequence(
                client,
                environment.gates,
                width=float(gate_config["width"]),
                height=float(gate_config["height"]),
                thickness=float(gate_config["thickness"]),
                asset_name=str(gate_config["asset"]),
                object_prefix="FalconDatasetGate_",
                world_offset=vehicle_local_to_world_offset(client, vehicle_name),
            )
        client.simPause(False)
        simulation_unpaused = True
        controller = MPCController(
            dt=dt,
            horizon=int(mpc_config["horizon"]),
            max_horizontal_accel=float(mpc_config.get("max_horizontal_accel", 4.0)),
            max_thrust_accel=float(mpc_config.get("max_thrust_accel", 18.0)),
            max_body_rate=float(mpc_config.get("max_body_rate", 3.0)),
        )
        controller.reset()
        first_state = drone.get_raw_state()
        tracker = GateTargetTracker(
            environment.gates,
            first_state.position,
            approach_distance=float(gate_config["approach_distance"]),
            exit_distance=float(gate_config["exit_distance"]),
            threshold=float(gate_config["threshold"]),
            stabilization_speed=float(gate_config.get("stabilization_speed", 0.0)),
            reference_speed=float(environment.reference_speed),
            start_time=time.perf_counter(),
            gate_width=float(gate_config["width"]),
            gate_height=float(gate_config["height"]),
            gate_thickness=float(gate_config["thickness"]),
            clearance=float(gate_config.get("clearance", 0.25)),
        )
        x_values: list[np.ndarray] = []
        u_values: list[np.ndarray] = []
        gate_ids: list[int] = []
        timestamps: list[float] = []
        positions: list[np.ndarray] = []
        quaternions: list[np.ndarray] = []
        velocities: list[np.ndarray] = []
        accelerations: list[np.ndarray] = []
        reference_positions: list[np.ndarray] = []
        reference_orientations: list[np.ndarray] = []
        solve_times: list[float] = []
        loop_periods: list[float] = []
        deadline_misses: list[int] = []
        path_contexts: list[np.ndarray] = []
        episode_start = time.perf_counter()
        next_deadline = episode_start
        last_start = episode_start
        max_loop_period = float(timing.get("catastrophic_loop_period", 0.5))
        for step_id in range(steps):
            if time.perf_counter() - episode_start > episode_timeout:
                raise EpisodeFailure(
                    f"episode exceeded timeout of {episode_timeout:.2f}s "
                    f"before completing {steps} samples"
                )
            period_start = time.perf_counter()
            loop_period = dt if step_id == 0 else period_start - last_start
            if loop_period > max_loop_period:
                raise EpisodeFailure(f"catastrophic control-loop overrun: {loop_period:.3f}s")
            raw = drone.get_raw_state()
            _check_state_and_collision(client, drone, raw, config)
            try:
                tracker.update(raw.position, period_start)
            except GatePassageError as exc:
                raise EpisodeFailure(str(exc)) from exc
            progress_every = int(config.get("dataset", {}).get("progress_every_steps", 0))
            if progress_every > 0 and step_id % progress_every == 0:
                print(
                    f"episode={episode_id} step={step_id:04d}/{steps} "
                    f"gate={tracker.target_gate_id} phase={tracker.phase} "
                    f"pos={np.array2string(raw.position, precision=2)}",
                    flush=True,
                )
            window = tracker.reference_window(period_start, dt=dt, horizon=controller.horizon)
            reference = window[0]
            x = build_controller_state(raw, reference.position, reference.orientation)
            solve_start = time.perf_counter()
            control = controller.solve(x, window)
            solve_time = time.perf_counter() - solve_start
            if not np.isfinite(control).all():
                raise EpisodeFailure("MPC returned NaN or Inf")
            send_control(drone, control, calibration, duration=dt)
            next_deadline += dt
            deadline_miss = int(time.perf_counter() > next_deadline)
            x_values.append(x.copy())
            u_values.append(np.asarray(control, dtype=np.float64).copy())
            gate_ids.append(tracker.target_gate_id)
            timestamps.append(time.time())
            positions.append(raw.position.copy())
            quaternions.append(raw.quaternion.copy())
            velocities.append(raw.velocity.copy())
            accelerations.append(raw.acceleration.copy())
            reference_positions.append(reference.position.copy())
            reference_orientations.append(reference.orientation.copy())
            if record_path_context:
                context = np.asarray(
                    [
                        np.asarray(window[min(offset, len(window) - 1)].position, dtype=np.float64)
                        - raw.position
                        for offset in context_offsets
                    ],
                    dtype=np.float64,
                )
                path_contexts.append(context)
            solve_times.append(solve_time)
            loop_periods.append(loop_period)
            deadline_misses.append(deadline_miss)
            last_start = period_start
            sleep_time = next_deadline - time.perf_counter()
            if sleep_time > 0.0:
                time.sleep(sleep_time)
        if bool(config.get("dataset", {}).get("require_all_gates", False)) and not tracker.completed:
            raise EpisodeFailure(
                f"episode ended before all {len(environment.gates)} gates were reached "
                f"(last target gate={tracker.target_gate_id}, phase={tracker.phase})"
            )
        return EpisodeSamples(
            X=np.asarray(x_values),
            U=np.asarray(u_values),
            episode_id=episode_id,
            initial_state_id=initial_state.initial_state_id,
            environment_id=environment.environment_id,
            step_id=np.arange(steps),
            target_gate_id=np.asarray(gate_ids),
            timestamp=np.asarray(timestamps),
            position=np.asarray(positions),
            quaternion=np.asarray(quaternions),
            velocity=np.asarray(velocities),
            acceleration=np.asarray(accelerations),
            reference_position=np.asarray(reference_positions),
            reference_orientation=np.asarray(reference_orientations),
            mpc_solve_time=np.asarray(solve_times),
            actual_loop_period=np.asarray(loop_periods),
            deadline_miss=np.asarray(deadline_misses),
            path_context=np.asarray(path_contexts) if record_path_context else None,
            reference_speed=(
                np.full(steps, float(environment.reference_speed), dtype=np.float64)
                if record_environment_metadata
                else None
            ),
            data_source=(
                np.full(steps, int(dataset_config.get("data_source_code", 0)), dtype=np.int8)
                if record_environment_metadata
                else None
            ),
        )
    finally:
        active_error = sys.exc_info()[1]
        if simulation_unpaused:
            try:
                client.simPause(False)
            except Exception:
                pass
        # If the RPC endpoint died, destroying 20 gate bars one by one would
        # wait for one client timeout per bar.  The restarted UE4 process has
        # a fresh world, so skip remote cleanup in that failure case.
        if scene is not None and not (
            active_error is not None and _looks_like_rpc_failure(active_error)
        ):
            scene.destroy()


def _looks_like_rpc_failure(exc: BaseException) -> bool:
    if isinstance(exc, (ConnectionError, OSError, AirSimProcessError)):
        return True
    text = str(exc).lower()
    return any(
        token in text
        for token in ("rpc", "connection", "refused", "socket", "msgpack", "timeout")
    )


def _connect_without_launch(*, rpc_host: str = "127.0.0.1", rpc_port: int = 41451, rpc_timeout: float = 10.0) -> Any:
    airsim = require_airsim()
    client = airsim.MultirotorClient(
        ip=rpc_host,
        port=int(rpc_port),
        timeout_value=float(rpc_timeout),
    )
    client.confirmConnection()
    return client


def _print_progress(
    *,
    episode_id: int,
    total_episodes: int,
    states: list[InitialState],
    environments: list[TrainingEnvironment],
    manifest: Mapping[str, Any],
    steps: int,
    started_at: float,
) -> None:
    completed = sum(entry.get("status") == "completed" for entry in manifest["episodes"].values())
    failed = sum(entry.get("status") == "failed" for entry in manifest["episodes"].values())
    elapsed = max(0.0, time.perf_counter() - started_at)
    average = elapsed / completed if completed else 0.0
    remaining = max(0, total_episodes - completed)
    eta = average * remaining
    environment_count = len(environments)
    state_id = episode_id // environment_count
    environment_id = episode_id % environment_count
    print(
        f"Episode: {episode_id + 1:04d} / {total_episodes:04d} | "
        f"Initial state: {state_id + 1} / {len(states)} | "
        f"Environment: {environment_id + 1} / {environment_count} | "
        f"Step: {steps} / {steps} | Completed: {completed} | Failed: {failed} | "
        f"Samples: {completed * steps} / {total_episodes * steps} | "
        f"Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s"
    )


def collect(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    auto_launch: bool = False,
    resume: bool = False,
    smoke_test: bool = False,
    output_root: str | Path | None = None,
    render: bool = False,
    progress_every_steps: int | None = None,
) -> Path:
    config, resolved_config_path = load_collection_config(config_path)
    if render:
        config["airsim"]["null_rhi"] = False
    if progress_every_steps is not None:
        if progress_every_steps < 0:
            raise ValueError("progress_every_steps must be non-negative")
        config["dataset"]["progress_every_steps"] = progress_every_steps
    base_dir = resolved_config_path.parent.parent
    configured_root = _resolve_path(
        output_root if output_root is not None else config["dataset"]["root"],
        base_dir,
    )
    root = configured_root.with_name(f"{configured_root.name}_smoke") if smoke_test and output_root is None else configured_root
    states, environments, manifest, manifest_path = _prepare_dataset(
        root,
        resolved_config_path,
        config,
        resume=resume,
        smoke_test=smoke_test,
    )
    effective_config = json.loads((root / "collection_config.json").read_text(encoding="utf-8"))
    total_episodes = len(states) * len(environments)
    steps = int(effective_config["timing"]["steps_per_episode"])
    calibration_value = effective_config.get("calibration")
    calibration = load_calibration(
        _resolve_path(calibration_value, base_dir) if calibration_value else None
    )
    airsim_config = effective_config["airsim"]
    manager: AirSimProcessManager | None = None
    client: Any = None
    collection_start = time.perf_counter()
    try:
        if auto_launch:
            manager_config = AirSimLaunchConfig.from_paths(
                airsim_config["editor"],
                airsim_config["uproject"],
                map_asset=airsim_config.get("map_asset"),
                log_dir=_resolve_path(effective_config.get("log_dir", "logs"), base_dir),
                startup_timeout=float(airsim_config["startup_timeout"]),
                retry_interval=float(airsim_config["retry_interval"]),
                rpc_host=str(airsim_config.get("rpc_host", "127.0.0.1")),
                rpc_port=int(airsim_config.get("rpc_port", 41451)),
                rpc_timeout=float(airsim_config.get("rpc_timeout", 10.0)),
                res_x=int(airsim_config["res_x"]),
                res_y=int(airsim_config["res_y"]),
                null_rhi=bool(airsim_config.get("null_rhi", False)),
            )
            manager = AirSimProcessManager(manager_config)
            client = manager.start_and_wait()
            print(f"AirSim RPC ready; managed pid={manager.pid}, log={manager.log_path}")
        else:
            client = _connect_without_launch(
                rpc_host=str(airsim_config.get("rpc_host", "127.0.0.1")),
                rpc_port=int(airsim_config.get("rpc_port", 41451)),
                rpc_timeout=float(airsim_config.get("rpc_timeout", 10.0)),
            )
        retry_limit = int(effective_config["dataset"]["retry_count"])
        for episode_id in range(total_episodes):
            key = _episode_key(episode_id)
            entry = manifest["episodes"][key]
            episode_path = _episode_path(root, episode_id)
            if entry.get("status") == "completed" and is_valid_episode(
                episode_path, expected_steps=steps, expected_episode_id=episode_id
            ):
                continue
            state = states[int(entry["initial_state_id"])]
            environment = environments[int(entry["environment_id"])]
            completed = False
            for attempt in range(1, retry_limit + 2):
                entry["status"] = "running" if attempt == 1 else "retrying"
                entry["attempts"] = attempt
                _update_manifest(manifest_path, manifest)
                try:
                    samples = run_episode(
                        client,
                        state,
                        environment,
                        episode_id=episode_id,
                        config=effective_config,
                        calibration=calibration,
                    )
                    save_episode_atomic(episode_path, samples)
                    entry["status"] = "completed"
                    entry["sample_count"] = steps
                    entry["last_error"] = None
                    _update_manifest(manifest_path, manifest)
                    completed = True
                    _print_progress(
                        episode_id=episode_id,
                        total_episodes=total_episodes,
                        states=states,
                        environments=environments,
                        manifest=manifest,
                        steps=steps,
                        started_at=collection_start,
                    )
                    break
                except KeyboardInterrupt:
                    entry["status"] = "interrupted"
                    entry["last_error"] = "keyboard interrupt"
                    _update_manifest(manifest_path, manifest)
                    raise
                except Exception as exc:
                    entry["last_error"] = f"{type(exc).__name__}: {exc}"
                    if attempt > retry_limit:
                        entry["status"] = "failed"
                        entry["sample_count"] = 0
                        _update_manifest(manifest_path, manifest)
                        print(f"{key} failed after {attempt} attempts: {entry['last_error']}")
                        break
                    entry["status"] = "retrying"
                    _update_manifest(manifest_path, manifest)
                    print(f"{key} attempt {attempt} failed; retrying: {entry['last_error']}")
                    if _looks_like_rpc_failure(exc):
                        if manager is not None:
                            client = manager.restart()
                            manifest["rpc_restarts"] = int(manifest.get("rpc_restarts", 0)) + 1
                            _update_manifest(manifest_path, manifest)
                        else:
                            client = _connect_without_launch(
                                rpc_host=str(airsim_config.get("rpc_host", "127.0.0.1")),
                                rpc_port=int(airsim_config.get("rpc_port", 41451)),
                                rpc_timeout=float(airsim_config.get("rpc_timeout", 10.0)),
                            )
            if not completed and entry.get("status") == "failed":
                continue
        completed_paths = [
            _episode_path(root, episode_id)
            for episode_id in range(total_episodes)
            if manifest["episodes"][_episode_key(episode_id)].get("status") == "completed"
        ]
        failed_ids = [
            episode_id
            for episode_id in range(total_episodes)
            if manifest["episodes"][_episode_key(episode_id)].get("status") != "completed"
        ]
        if not failed_ids and len(completed_paths) == total_episodes:
            final_dir = root / "final"
            final_path = final_dir / str(effective_config["dataset"]["output_name"])
            metadata_path = final_dir / "teacher_metadata.csv"
            merge_stats = merge_episodes(
                completed_paths,
                final_path,
                expected_steps=steps,
                expected_episode_count=total_episodes,
                metadata_csv_path=metadata_path,
            )
            validation_stats = validate_merged_dataset(
                final_path,
                expected_samples=total_episodes * steps,
                expected_episodes=total_episodes,
                expected_steps=steps,
                expected_initial_states=len(states),
                expected_environments=len(environments),
            )
            manifest["status"] = "completed"
            manifest["final_dataset"] = str(Path("final") / final_path.name)
            _update_manifest(manifest_path, manifest)
            from validate_teacher_dataset import validate_dataset

            try:
                validation_report = validate_dataset(root)
            except Exception:
                manifest["status"] = "incomplete"
                manifest["last_error"] = "final dataset validation failed"
                _update_manifest(manifest_path, manifest)
                raise
            summary = {
                "status": "completed",
                "episodes": f"{total_episodes} / {total_episodes}",
                "samples": validation_stats["samples"],
                "initial_states": len(states),
                "environments": len(environments),
                "sampling_period": effective_config["timing"]["dt"],
                "samples_per_episode": steps,
                "failed_episodes": 0,
                "rpc_restarts": manifest.get("rpc_restarts", 0),
                "X_shape": validation_stats["X_shape"],
                "U_shape": validation_stats["U_shape"],
                "X_mean": validation_stats["X_mean"],
                "X_std": validation_stats["X_std"],
                "U_mean": validation_stats["U_mean"],
                "U_std": validation_stats["U_std"],
                "deadline_misses": validation_stats["deadline_misses"],
                "average_mpc_solve_time": validation_stats["average_mpc_solve_time"],
                "reference_speed_values": validation_stats.get("reference_speed_values", []),
                "data_source_counts": validation_stats.get("data_source_counts", {}),
                "path_context_shape": validation_stats.get("path_context_shape"),
                "validation": validation_report["status"],
                "validation_warnings": validation_report["warnings"],
            }
            write_summary(root / "summary.txt", summary)
            manifest["summary"] = summary
            _update_manifest(manifest_path, manifest)
            print(f"Collection completed: X.shape={tuple(validation_stats['X_shape'])}, U.shape={tuple(validation_stats['U_shape'])}")
        else:
            summary = {
                "status": "incomplete",
                "episodes_completed": total_episodes - len(failed_ids),
                "episodes_expected": total_episodes,
                "samples_completed": (total_episodes - len(failed_ids)) * steps,
                "samples_expected": total_episodes * steps,
                "failed_episode_ids": failed_ids,
                "rpc_restarts": manifest.get("rpc_restarts", 0),
            }
            write_summary(root / "summary.txt", summary)
            manifest["status"] = "incomplete"
            manifest["failed_episode_ids"] = failed_ids
            manifest["summary"] = summary
            _update_manifest(manifest_path, manifest)
            print(f"Collection incomplete; failed episodes: {failed_ids}")
    finally:
        if manager is not None:
            manager.stop()
            print("Managed UE4/AirSim process stopped.")
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--auto-launch", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--output-root", default=None)
    parser.add_argument(
        "--render",
        action="store_true",
        help="launch UE4 with graphics instead of the headless -nullrhi mode",
    )
    parser.add_argument(
        "--progress-every-steps",
        type=int,
        default=None,
        help="print in-episode progress every N samples (0 disables it)",
    )
    args = parser.parse_args()
    collect(
        args.config,
        auto_launch=args.auto_launch,
        resume=args.resume,
        smoke_test=args.smoke_test,
        output_root=args.output_root,
        render=args.render,
        progress_every_steps=args.progress_every_steps,
    )


if __name__ == "__main__":
    main()
