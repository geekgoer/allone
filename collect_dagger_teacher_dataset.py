"""Collect DAgger-style MPC labels on the student's actual AirSim states.

The DNN supplies the applied action whenever the state is inside a configurable
safety envelope.  The MPC is evaluated at the same measured state on every
step and is always written as the training label.  Outside the envelope a
simple MPC shield is applied so a weak early student does not destroy the
rollout before useful recovery states are recorded.

This script intentionally uses a fixed manifest and atomic per-episode files.
Pressing Ctrl+C, closing the terminal, or losing an episode only leaves that
episode incomplete; rerunning with ``--resume`` skips valid files and
continues from the remaining episode IDs.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

from airsim_interface import AirSimDrone, require_airsim, vehicle_local_to_world_offset
from airsim_process import AirSimLaunchConfig, AirSimProcessError, AirSimProcessManager
from collect_teacher_dataset import (
    _airsim_pose,
    _looks_like_rpc_failure,
    _resolve_path,
    _join,
)
from config import ThrustCalibration, load_calibration
from control_adapter import send_control
from dataset_store import (
    EpisodeSamples,
    atomic_write_json,
    is_valid_episode,
    load_episode,
    merge_episodes,
    save_episode_atomic,
    validate_merged_dataset,
    write_summary,
)
from dnn_controller import DNNController
from dnn_model_utils import build_policy_input
from gate_route import ContinuousGateRoute, GatePassageError
from gate_visualization import cleanup_generated_gates, spawn_gate_sequence
from initial_state_generator import InitialState, load_initial_states
from mpc_controller import MPCController
from state_adapter import build_controller_state
from training_environment_generator import TrainingEnvironment, load_training_environments


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "teacher_dataset_robust_v1"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "teacher_dataset_dagger_v1"


class EpisodeFailure(RuntimeError):
    """Raised when a fixed-length DAgger episode cannot be completed."""


def _episode_key(episode_id: int) -> str:
    return f"episode_{episode_id:04d}"


def _episode_path(root: Path, episode_id: int) -> Path:
    return root / "episodes" / f"{_episode_key(episode_id)}.npz"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_state_and_collision(
    client: Any,
    drone: AirSimDrone,
    state: Any,
    config: dict[str, Any],
    *,
    first_sample: bool,
) -> bool:
    values = (state.position, state.quaternion, state.velocity, state.acceleration, state.angular_velocity)
    if not all(np.isfinite(value).all() for value in values):
        raise EpisodeFailure("AirSim returned NaN or Inf in DAgger rollout")
    bounds = config.get("bounds", {})
    if bounds:
        lower = np.asarray(bounds["position_min"], dtype=np.float64)
        upper = np.asarray(bounds["position_max"], dtype=np.float64)
        if np.any(state.position < lower) or np.any(state.position > upper):
            raise EpisodeFailure(f"drone left configured position bounds: {state.position.tolist()}")
    collision = client.simGetCollisionInfo(vehicle_name=drone.vehicle_name)
    if bool(getattr(collision, "has_collided", False)):
        object_name = getattr(collision, "object_name", "unknown")
        if first_sample:
            print(
                f"Ignoring reset collision flag on first DAgger sample: "
                f"object={object_name} position={np.array2string(state.position, precision=3)}",
                flush=True,
            )
            return True
        raise EpisodeFailure(f"DAgger drone collision with {object_name}")
    return False


def _episode_from_arrays(arrays: dict[str, np.ndarray], *, episode_id: int) -> EpisodeSamples:
    return EpisodeSamples(
        X=np.asarray(arrays["X"]),
        U=np.asarray(arrays["U"]),
        episode_id=episode_id,
        initial_state_id=int(np.asarray(arrays["initial_state_id"])[0]),
        environment_id=int(np.asarray(arrays["environment_id"])[0]),
        step_id=np.asarray(arrays["step_id"]),
        target_gate_id=np.asarray(arrays["target_gate_id"]),
        timestamp=np.asarray(arrays["timestamp"]),
        position=np.asarray(arrays["position"]),
        quaternion=np.asarray(arrays["quaternion"]),
        velocity=np.asarray(arrays["velocity"]),
        acceleration=np.asarray(arrays["acceleration"]),
        reference_position=np.asarray(arrays["reference_position"]),
        reference_orientation=np.asarray(arrays["reference_orientation"]),
        mpc_solve_time=np.asarray(arrays["mpc_solve_time"]),
        actual_loop_period=np.asarray(arrays["actual_loop_period"]),
        deadline_miss=np.asarray(arrays["deadline_miss"]),
        path_context=np.asarray(arrays["path_context"]),
        reference_speed=np.asarray(arrays["reference_speed"]),
        data_source=np.asarray(arrays["data_source"]),
        student_control=np.asarray(arrays["student_control"]),
        shield_active=np.asarray(arrays["shield_active"]),
    )


def _new_manifest(
    *,
    source_root: Path,
    output_root: Path,
    source_manifest: dict[str, Any],
    model_path: Path,
    model_hash: str,
) -> dict[str, Any]:
    states = int(source_manifest["initial_state_count"])
    environments = int(source_manifest["environment_count"])
    episode_count = states * environments
    episodes = {}
    for episode_id in range(episode_count):
        state_id, environment_id = divmod(episode_id, environments)
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
        "schema_version": 1,
        "teacher_policy_version": "dagger_mpc_labels_v1",
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root),
        "output_root": str(output_root),
        "model_path": str(model_path),
        "model_sha256": model_hash,
        "initial_state_count": states,
        "environment_count": environments,
        "episode_count": episode_count,
        "steps_per_episode": int(source_manifest["steps_per_episode"]),
        "episodes": episodes,
    }


def _update_manifest(path: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(path, manifest)


def run_episode(
    client: Any,
    initial_state: InitialState,
    environment: TrainingEnvironment,
    *,
    episode_id: int,
    config: dict[str, Any],
    calibration: ThrustCalibration,
    student: DNNController,
    shield_position_error: float,
    shield_velocity: float,
) -> EpisodeSamples:
    vehicle_name = str(config["vehicle_name"])
    timing = config["timing"]
    dt = float(timing["dt"])
    steps = int(timing["steps_per_episode"])
    episode_timeout = float(timing.get("episode_timeout_seconds", max(2.0 * steps * dt, steps * dt + 5.0)))
    gate_config = config["gate"]
    mpc_config = config["mpc"]
    drone = None
    scene = None
    simulation_unpaused = False
    try:
        client.reset()
        try:
            client.simPause(True)
        except Exception:
            pass
        client.simSetVehiclePose(
            _airsim_pose(require_airsim(), initial_state),
            ignore_collision=True,
            vehicle_name=vehicle_name,
        )
        drone = AirSimDrone(client, vehicle_name)
        drone.enable_api_control(True)
        drone.arm()
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
        # Pose placement does not itself create an active thrust command.
        # Start hover immediately so DAgger does not collect a ground-fall
        # transient as if it were a student recovery state.
        _join(drone.hover())
        time.sleep(0.1)
        mpc = MPCController(
            dt=dt,
            horizon=int(mpc_config["horizon"]),
            max_horizontal_accel=float(mpc_config.get("max_horizontal_accel", 4.0)),
            max_thrust_accel=float(mpc_config.get("max_thrust_accel", 18.0)),
            max_body_rate=float(mpc_config.get("max_body_rate", 3.0)),
        )
        first_state = drone.get_raw_state()
        route_start = time.perf_counter()
        route = ContinuousGateRoute(
            environment.gates,
            first_state.position,
            approach_distance=float(gate_config["approach_distance"]),
            exit_distance=float(gate_config["exit_distance"]),
            reference_speed=float(environment.reference_speed),
            start_time=route_start,
            gate_width=float(gate_config["width"]),
            gate_height=float(gate_config["height"]),
            gate_thickness=float(gate_config["thickness"]),
            clearance=float(gate_config.get("clearance", 0.25)),
            completion_radius=float(gate_config["threshold"]),
        )
        x_values: list[np.ndarray] = []
        teacher_controls: list[np.ndarray] = []
        student_controls: list[np.ndarray] = []
        shield_values: list[int] = []
        gate_ids: list[int] = []
        timestamps: list[float] = []
        positions: list[np.ndarray] = []
        quaternions: list[np.ndarray] = []
        velocities: list[np.ndarray] = []
        accelerations: list[np.ndarray] = []
        reference_positions: list[np.ndarray] = []
        reference_orientations: list[np.ndarray] = []
        path_contexts: list[np.ndarray] = []
        solve_times: list[float] = []
        loop_periods: list[float] = []
        deadline_misses: list[int] = []
        episode_start = time.perf_counter()
        next_deadline = episode_start
        last_start = episode_start
        shield_count = 0
        for step_id in range(steps):
            if time.perf_counter() - episode_start > episode_timeout:
                raise EpisodeFailure(f"DAgger episode exceeded timeout {episode_timeout:.1f}s")
            period_start = time.perf_counter()
            loop_period = dt if step_id == 0 else period_start - last_start
            if loop_period > float(timing.get("catastrophic_loop_period", 0.5)):
                raise EpisodeFailure(f"catastrophic control-loop overrun: {loop_period:.3f}s")
            raw = drone.get_raw_state()
            _check_state_and_collision(client, drone, raw, config, first_sample=step_id == 0)
            try:
                crossing = route.update(raw.position, period_start)
            except GatePassageError as exc:
                raise EpisodeFailure(str(exc)) from exc
            window = route.reference_window(period_start, dt=dt, horizon=mpc.horizon)
            reference = window[0]
            x = build_controller_state(raw, reference.position, reference.orientation)
            solve_start = time.perf_counter()
            teacher_control = mpc.solve(x, window)
            solve_time = time.perf_counter() - solve_start
            student_input = build_policy_input(
                x,
                window,
                input_dim=student.input_dim,
                reference_speed=float(environment.reference_speed),
            )
            student_control = student.solve(student_input)
            lateral_vertical_error = float(np.max(np.abs(x[1:3])))
            student_velocity = float(np.linalg.norm(raw.velocity))
            use_shield = lateral_vertical_error > shield_position_error or student_velocity > shield_velocity
            applied_control = teacher_control if use_shield else student_control
            if use_shield:
                shield_count += 1
            if crossing is not None:
                print(
                    f"DAgger episode={episode_id} gate={crossing.gate_index} crossed "
                    f"lateral={crossing.lateral_offset:.3f} vertical={crossing.vertical_offset:.3f} "
                    f"shield_steps={shield_count}",
                    flush=True,
                )
            if not np.isfinite(teacher_control).all() or not np.isfinite(student_control).all():
                raise EpisodeFailure("MPC or DNN returned NaN or Inf")
            send_control(drone, applied_control, calibration, duration=dt)
            next_deadline += dt
            deadline_miss = int(time.perf_counter() > next_deadline)
            x_values.append(student_input.copy())
            teacher_controls.append(np.asarray(teacher_control, dtype=np.float64).copy())
            student_controls.append(np.asarray(student_control, dtype=np.float64).copy())
            shield_values.append(int(use_shield))
            gate_ids.append(route.target_gate_id)
            timestamps.append(time.time())
            positions.append(raw.position.copy())
            quaternions.append(raw.quaternion.copy())
            velocities.append(raw.velocity.copy())
            accelerations.append(raw.acceleration.copy())
            reference_positions.append(reference.position.copy())
            reference_orientations.append(reference.orientation.copy())
            path_contexts.append(
                np.asarray(
                    [np.asarray(window[index].position, dtype=np.float64) - raw.position for index in (0, 5, 10, 19)],
                    dtype=np.float64,
                )
            )
            solve_times.append(solve_time)
            loop_periods.append(loop_period)
            deadline_misses.append(deadline_miss)
            if step_id % 20 == 0:
                print(
                    f"dagger episode={episode_id} step={step_id:04d}/{steps} "
                    f"gate={route.target_gate_id} shield={int(use_shield)} "
                    f"pos_err={np.linalg.norm(x[:3]):.2f}m",
                    flush=True,
                )
            last_start = period_start
            sleep_time = next_deadline - time.perf_counter()
            if sleep_time > 0.0:
                time.sleep(sleep_time)
        return EpisodeSamples(
            X=np.asarray(x_values),
            U=np.asarray(teacher_controls),
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
            path_context=np.asarray(path_contexts),
            reference_speed=np.full(steps, float(environment.reference_speed)),
            data_source=np.full(steps, 2, dtype=np.int8),
            student_control=np.asarray(student_controls),
            shield_active=np.asarray(shield_values, dtype=np.int8),
        )
    finally:
        active_error = sys.exc_info()[1]
        if simulation_unpaused:
            try:
                client.simPause(False)
            except Exception:
                pass
        if scene is not None and not (active_error is not None and _looks_like_rpc_failure(active_error)):
            scene.destroy()
        if drone is not None:
            try:
                _join(drone.hover())
                _join(drone.land())
                drone.disarm()
                drone.enable_api_control(False)
            except Exception:
                pass


def _connect_without_launch(config: dict[str, Any]) -> Any:
    airsim = require_airsim()
    client = airsim.MultirotorClient(
        ip=str(config.get("rpc_host", "127.0.0.1")),
        port=int(config.get("rpc_port", 41451)),
        timeout_value=float(config.get("rpc_timeout", 10.0)),
    )
    client.confirmConnection()
    return client


def collect(
    *,
    model_path: str | Path,
    source_root: str | Path = DEFAULT_SOURCE_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    normalization_path: str | Path | None = None,
    auto_launch: bool = False,
    resume: bool = False,
    render: bool = False,
    shield_position_error: float = 0.9,
    shield_velocity: float = 6.0,
    progress_every_episodes: int = 1,
) -> Path:
    source = Path(source_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    model = Path(model_path).expanduser().resolve()
    if not model.exists():
        raise FileNotFoundError(f"DNN model was not found: {model}")
    source_manifest = _load_json(source / "manifest.json")
    if source_manifest.get("status") != "completed":
        raise ValueError("DAgger source root must be a completed nominal/augmented dataset")
    config = _load_json(source / "collection_config.json")
    states = load_initial_states(source / "initial_states.json")
    environments = load_training_environments(source / "environments.json")
    if len(states) != int(source_manifest["initial_state_count"]) or len(environments) != int(source_manifest["environment_count"]):
        raise ValueError("source generation artifacts do not match source manifest")
    if shield_position_error <= 0.0 or shield_velocity <= 0.0:
        raise ValueError("shield limits must be positive")
    if render:
        config["airsim"]["null_rhi"] = False
    manifest_path = output / "manifest.json"
    model_hash = _file_sha256(model)
    if manifest_path.exists() and not resume:
        raise FileExistsError(f"output exists: {output}; use --resume to continue")
    output.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        manifest = _load_json(manifest_path)
        if manifest.get("model_sha256") != model_hash:
            raise ValueError("resume model does not match the model recorded in the DAgger manifest")
    else:
        manifest = _new_manifest(
            source_root=source,
            output_root=output,
            source_manifest=source_manifest,
            model_path=model,
            model_hash=model_hash,
        )
        (output / "initial_states.json").write_text(
            (source / "initial_states.json").read_text(encoding="utf-8"), encoding="utf-8"
        )
        (output / "environments.json").write_text(
            (source / "environments.json").read_text(encoding="utf-8"), encoding="utf-8"
        )
        config["dataset"] = dict(config.get("dataset", {}))
        config["dataset"]["root"] = output.name
        config["dataset"]["output_name"] = "teacher_dataset_dagger_v1.npz"
        config["dataset"]["strategy"] = "dagger_student_rollout_with_mpc_labels"
        config["dataset"]["require_all_gates"] = False
        (output / "collection_config.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        manifest["shield_position_error"] = shield_position_error
        manifest["shield_velocity"] = shield_velocity
        _update_manifest(manifest_path, manifest)
    config = _load_json(output / "collection_config.json")
    calibration_value = config.get("calibration")
    base_dir = Path(__file__).resolve().parent
    calibration = load_calibration(_resolve_path(calibration_value, base_dir) if calibration_value else None)
    student = DNNController(
        model,
        normalization_path=normalization_path,
        device="cpu",
        max_thrust_accel=float(config["mpc"].get("max_thrust_accel", 18.0)),
        max_body_rate=float(config["mpc"].get("max_body_rate", 3.0)),
    )
    airsim_config = config["airsim"]
    manager: AirSimProcessManager | None = None
    client: Any = None
    try:
        if auto_launch:
            launch_config = AirSimLaunchConfig.from_paths(
                airsim_config["editor"],
                airsim_config["uproject"],
                map_asset=airsim_config.get("map_asset"),
                log_dir=_resolve_path(config.get("log_dir", "logs"), base_dir),
                startup_timeout=float(airsim_config["startup_timeout"]),
                retry_interval=float(airsim_config["retry_interval"]),
                rpc_host=str(airsim_config.get("rpc_host", "127.0.0.1")),
                rpc_port=int(airsim_config.get("rpc_port", 41451)),
                rpc_timeout=float(airsim_config.get("rpc_timeout", 10.0)),
                res_x=int(airsim_config["res_x"]),
                res_y=int(airsim_config["res_y"]),
                null_rhi=bool(airsim_config.get("null_rhi", False)),
            )
            manager = AirSimProcessManager(launch_config)
            client = manager.start_and_wait()
            print(f"AirSim RPC ready; managed pid={manager.pid}, log={manager.log_path}")
        else:
            client = _connect_without_launch(airsim_config)
        total = int(manifest["episode_count"])
        steps = int(manifest["steps_per_episode"])
        retry_limit = int(config.get("dataset", {}).get("retry_count", 1))
        started = time.perf_counter()
        for episode_id in range(total):
            key = _episode_key(episode_id)
            entry = manifest["episodes"][key]
            path = _episode_path(output, episode_id)
            if entry.get("status") == "completed" and is_valid_episode(path, expected_steps=steps, expected_episode_id=episode_id):
                continue
            state_id = int(entry["initial_state_id"])
            environment_id = int(entry["environment_id"])
            for attempt in range(1, retry_limit + 2):
                entry["status"] = "running" if attempt == 1 else "retrying"
                entry["attempts"] = attempt
                _update_manifest(manifest_path, manifest)
                try:
                    samples = run_episode(
                        client,
                        states[state_id],
                        environments[environment_id],
                        episode_id=episode_id,
                        config=config,
                        calibration=calibration,
                        student=student,
                        shield_position_error=float(manifest["shield_position_error"]),
                        shield_velocity=float(manifest["shield_velocity"]),
                    )
                    save_episode_atomic(path, samples)
                    entry["status"] = "completed"
                    entry["sample_count"] = steps
                    entry["last_error"] = None
                    _update_manifest(manifest_path, manifest)
                    completed = sum(item.get("status") == "completed" for item in manifest["episodes"].values())
                    if progress_every_episodes > 0 and completed % progress_every_episodes == 0:
                        elapsed = time.perf_counter() - started
                        average = elapsed / completed if completed else 0.0
                        print(
                            f"DAgger episode {episode_id + 1:04d}/{total:04d} | "
                            f"completed={completed} | samples={completed * steps}/{total * steps} | "
                            f"ETA={average * (total - completed):.1f}s",
                            flush=True,
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
                        _update_manifest(manifest_path, manifest)
                        print(f"{key} failed after {attempt} attempts: {entry['last_error']}", flush=True)
                        break
                    entry["status"] = "retrying"
                    _update_manifest(manifest_path, manifest)
                    print(f"{key} attempt {attempt} failed; retrying: {entry['last_error']}", flush=True)
                    if _looks_like_rpc_failure(exc):
                        if manager is not None:
                            client = manager.restart()
                            manifest["rpc_restarts"] = int(manifest.get("rpc_restarts", 0)) + 1
                            _update_manifest(manifest_path, manifest)
                        else:
                            client = _connect_without_launch(airsim_config)
            
        failed = [
            episode_id for episode_id in range(total)
            if manifest["episodes"][_episode_key(episode_id)].get("status") != "completed"
        ]
        if not failed:
            paths = [_episode_path(output, episode_id) for episode_id in range(total)]
            final_path = output / "final" / "teacher_dataset_dagger_v1.npz"
            stats = merge_episodes(
                paths,
                final_path,
                expected_steps=steps,
                expected_episode_count=total,
                metadata_csv_path=output / "final" / "teacher_metadata.csv",
            )
            validate_merged_dataset(
                final_path,
                expected_samples=total * steps,
                expected_episodes=total,
                expected_steps=steps,
                expected_initial_states=len(states),
                expected_environments=len(environments),
            )
            manifest["status"] = "completed"
            manifest["final_dataset"] = str(Path("final") / final_path.name)
            manifest["statistics"] = stats
            _update_manifest(manifest_path, manifest)
            write_summary(output / "summary.txt", {"status": "completed", **stats})
            print(f"DAgger collection completed: X.shape={tuple(stats['X_shape'])}, U.shape={tuple(stats['U_shape'])}")
        else:
            manifest["status"] = "incomplete"
            manifest["failed_episode_ids"] = failed
            _update_manifest(manifest_path, manifest)
            print(f"DAgger collection incomplete; rerun with --resume. Remaining episodes: {failed}")
    finally:
        if manager is not None:
            manager.stop()
            print("Managed UE4/AirSim process stopped.")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--normalization", type=Path, default=None)
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--auto-launch", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--shield-position-error", type=float, default=0.9)
    parser.add_argument("--shield-velocity", type=float, default=6.0)
    parser.add_argument("--progress-every-episodes", type=int, default=1)
    args = parser.parse_args()
    collect(
        model_path=args.model,
        source_root=args.source_root,
        output_root=args.output_root,
        normalization_path=args.normalization,
        auto_launch=args.auto_launch,
        resume=args.resume,
        render=args.render,
        shield_position_error=args.shield_position_error,
        shield_velocity=args.shield_velocity,
        progress_every_episodes=args.progress_every_episodes,
    )


if __name__ == "__main__":
    main()
