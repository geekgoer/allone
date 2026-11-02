"""Resumable AirSim collector for the isolated FALCON paper protocol."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import numpy as np

from airsim_interface import AirSimDrone, RawDroneState, require_airsim, vehicle_local_to_world_offset
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
from gate_route import ContinuousGateRoute, GatePassageError
from gate_visualization import cleanup_generated_gates, spawn_gate_sequence
from initial_state_generator import InitialState, InitialStateGenerator
from mpc_controller import MPCReference
from training_environment_generator import TrainingEnvironment, generate_training_environments

from .contracts import (
    INPUT_CONTRACT,
    INPUT_FEATURES,
    OUTPUT_FEATURES,
    TRAJECTORY_INPUT_CONTRACT,
    TRAJECTORY_INPUT_FEATURES,
    build_gate_input,
    build_trajectory_input,
)
from .acado_mpc import AcadoMPCConfig, AcadoRtiMPC
from .motion_primitive import (
    ContinuousQuinticTrajectory,
    MotionPrimitiveConfig,
    MotionPrimitiveInfeasible,
    QuinticMotionPrimitivePlanner,
    QuinticPolynomial,
)
from .nonlinear_mpc import CasadiTrackingMPC, NonlinearMPCConfig


CASADI_POLICY_VERSION = "falcon_paper_core_quintic_casadi_nmpc_v1"
ACADO_POLICY_VERSION = "falcon_paper_core_segment_quintic_full_reference_acado_qpoases_v2"
ACADO_CONTINUOUS_POLICY_VERSION = (
    "falcon_paper_core_global_continuous_quintic_acado_qpoases_v3"
)
ACADO_MARKOV_CONTINUOUS_POLICY_VERSION = (
    "falcon_paper_core_global_continuous_nominal_start_acado_qpoases_v5"
)
MANAGED_AIRSIM_RESTART_INTERVAL_EPISODES = 100


def _mpc_backend(config: Mapping[str, Any]) -> str:
    backend = str(config["mpc"].get("backend", "casadi_ipopt")).strip().lower()
    aliases = {
        "casadi": "casadi_ipopt",
        "casadi_ipopt": "casadi_ipopt",
        "acado": "official_rpg_mpc_acado_qpoases",
        "rpg_mpc": "official_rpg_mpc_acado_qpoases",
        "official_rpg_mpc_acado_qpoases": "official_rpg_mpc_acado_qpoases",
    }
    if backend not in aliases:
        raise ValueError(f"unsupported MPC backend: {backend!r}")
    return aliases[backend]


def _policy_version(config: Mapping[str, Any]) -> str:
    explicit = str(config.get("trajectory", {}).get("policy_version", "")).strip()
    if explicit:
        return explicit
    if (
        _mpc_backend(config) == "official_rpg_mpc_acado_qpoases"
        and str(config.get("trajectory", {}).get("mode", "per_gate"))
        == "global_continuous"
        and str(
            config.get("trajectory", {}).get("start_velocity_mode", "measured")
        ).lower()
        == "nominal_gate_axis"
    ):
        return ACADO_MARKOV_CONTINUOUS_POLICY_VERSION
    if (
        _mpc_backend(config) == "official_rpg_mpc_acado_qpoases"
        and str(config.get("trajectory", {}).get("mode", "per_gate"))
        == "global_continuous"
    ):
        return ACADO_CONTINUOUS_POLICY_VERSION
    return (
        ACADO_POLICY_VERSION
        if _mpc_backend(config) == "official_rpg_mpc_acado_qpoases"
        else CASADI_POLICY_VERSION
    )


def _uses_continuous_trajectory(config: Mapping[str, Any]) -> bool:
    return (
        str(config.get("trajectory", {}).get("mode", "per_gate"))
        == "global_continuous"
    )


def _input_contract(config: Mapping[str, Any]) -> tuple[str, tuple[str, ...]]:
    if _uses_continuous_trajectory(config):
        return TRAJECTORY_INPUT_CONTRACT, TRAJECTORY_INPUT_FEATURES
    return INPUT_CONTRACT, INPUT_FEATURES


def _fidelity_tier(config: Mapping[str, Any]) -> str:
    if _mpc_backend(config) == "official_rpg_mpc_acado_qpoases":
        return "official_public_rpg_mpc_generated_acado_qpoases_solver"
    return "paper_equations_and_public_rpg_mpc_defaults_not_author_acado_binary"


class EpisodeFailure(RuntimeError):
    """One episode failed a physical or numerical safety contract."""


def _is_rpc_transport_failure(exc: BaseException) -> bool:
    """Return whether an exception means that the AirSim RPC channel died.

    Physical episode failures and solver failures must remain ordinary dataset
    failures. In contrast, msgpack-rpc timeout/transport failures invalidate
    the shared client used by every later episode and require simulator
    recovery before collection can continue.
    """

    current: BaseException | None = exc
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        module = type(current).__module__.lower()
        name = type(current).__name__.lower()
        message = str(current).lower()
        if module.startswith("msgpackrpc") and name in {"timeouterror", "transporterror"}:
            return True
        if isinstance(current, (ConnectionError, BrokenPipeError, TimeoutError)):
            return True
        if any(
            marker in message
            for marker in (
                "request timed out",
                "retry connection over the limit",
                "connection reset by peer",
                "connection refused",
                "broken pipe",
            )
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def _record_simulator_restart(
    manifest: dict[str, Any], *, episode_id: int, reason: str
) -> None:
    manifest["rpc_restarts"] = int(manifest.get("rpc_restarts", 0)) + 1
    events = manifest.setdefault("rpc_restart_events", [])
    events.append(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "episode_id": int(episode_id),
            "reason": str(reason),
        }
    )


def _canonical_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    resolved = Path(path).expanduser().resolve()
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("collection config must contain one JSON object")
    return value, resolved


def _resolve(path: str | Path, base: Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (base / value).resolve()


def _states(config: Mapping[str, Any]) -> list[InitialState]:
    spec = config["initial_state"]
    return InitialStateGenerator(
        count=int(spec["count"]),
        seed=int(spec["seed"]),
        position_range=spec["position_range"],
        attitude_range=spec["attitude_range"],
        velocity_range=spec["velocity_range"],
        angular_velocity_range=spec["angular_velocity_range"],
    ).generate()


def _environments(config: Mapping[str, Any]) -> list[TrainingEnvironment]:
    spec = config["environment"]
    speed = float(config["planner"]["nominal_speed"])
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
        speed_values=(speed,),
    )


def _episode_key(episode_id: int) -> str:
    return f"episode_{episode_id:04d}"


def _episode_path(root: Path, episode_id: int) -> Path:
    return root / "episodes" / f"{_episode_key(episode_id)}.npz"


def _write_artifacts(root: Path, states: list[InitialState], environments: list[TrainingEnvironment]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(root / "initial_states.json", [item.to_dict() for item in states])  # type: ignore[arg-type]
    atomic_write_json(root / "environments.json", [item.to_dict() for item in environments])  # type: ignore[arg-type]


def _manifest(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    states: list[InitialState],
    environments: list[TrainingEnvironment],
) -> dict[str, Any]:
    policy_version = _policy_version(config)
    input_contract, input_features = _input_contract(config)
    steps = int(config["timing"]["steps_per_episode"])
    episodes: dict[str, Any] = {}
    for state_id in range(len(states)):
        for environment_id in range(len(environments)):
            episode_id = state_id * len(environments) + environment_id
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
    now = datetime.now(timezone.utc).isoformat()
    return {
        "schema_version": 1,
        "status": "running",
        "created_at": now,
        "updated_at": now,
        "strategy": policy_version,
        "teacher_policy_version": policy_version,
        "fidelity_tier": _fidelity_tier(config),
        "paper_protocol": {
            "initial_states": len(states),
            "environmental_conditions": len(environments),
            "seconds_per_episode": float(config["timing"]["episode_seconds"]),
            "sample_period_seconds": float(config["timing"]["dt"]),
            "prediction_period_seconds": float(
                config["mpc"].get("prediction_dt", config["timing"]["dt"])
            ),
            "horizon": int(config["mpc"]["horizon"]),
            "expected_pairs": len(states) * len(environments) * steps,
            "published_pairs": 40000,
        },
        "input_dimensions": 12,
        "input_features": list(input_features),
        "input_contract": input_contract,
        "output_dimensions": 4,
        "output_features": list(OUTPUT_FEATURES),
        "planner": dict(config["planner"]),
        "trajectory": dict(config.get("trajectory", {"mode": "per_gate"})),
        "mpc": dict(config["mpc"]),
        "fixed_hidden_parameters": {
            "planner_config_is_constant": True,
            "independently_randomized_reference_speed": False,
            "reason": "prevents contradictory labels for the fixed 12-D input",
        },
        "config_file": str(config_path),
        "config_sha256": _canonical_hash(config),
        "vehicle_name": str(config["vehicle_name"]),
        "initial_state_count": len(states),
        "environment_count": len(environments),
        "episode_count": len(episodes),
        "steps_per_episode": steps,
        "expected_samples": len(episodes) * steps,
        "rpc_restarts": 0,
        "episodes": episodes,
    }


def _prepare(
    root: Path,
    config: dict[str, Any],
    config_path: Path,
    *,
    resume: bool,
    smoke_test: bool,
) -> tuple[dict[str, Any], list[InitialState], list[TrainingEnvironment], Path]:
    effective = deepcopy(config)
    if smoke_test:
        effective["initial_state"]["count"] = 2
        effective["environment"]["count"] = 2
        # A 0.5 s smoke run previously stopped before a systematic instability
        # became visible at roughly 1--1.5 s.  Keep the complete 5 s control
        # horizon while reducing only the state/environment combinations.
        effective["timing"]["steps_per_episode"] = 100
        effective["timing"]["episode_seconds"] = 5.0
        effective["timing"]["episode_timeout_seconds"] = 15.0
    dt = float(effective["timing"]["dt"])
    steps = int(effective["timing"]["steps_per_episode"])
    if not np.isclose(float(effective["timing"]["episode_seconds"]), dt * steps):
        raise ValueError("episode_seconds must equal dt * steps_per_episode")
    if int(effective["mpc"]["horizon"]) != 20 or not np.isclose(dt, 0.05):
        raise ValueError("paper core requires horizon=20 and dt=0.05")
    states = _states(effective)
    environments = _environments(effective)
    manifest_path = root / "manifest.json"
    if manifest_path.exists() and not resume:
        raise FileExistsError(f"output exists: {root}; use --resume to continue")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("teacher_policy_version") != _policy_version(effective):
            raise ValueError("resume root belongs to another teacher policy")
        if manifest.get("config_sha256") != _canonical_hash(effective):
            raise ValueError("resume config differs from the immutable collection config")
    else:
        manifest = _manifest(
            config=effective,
            config_path=config_path,
            states=states,
            environments=environments,
        )
    root.mkdir(parents=True, exist_ok=True)
    (root / "collection_config.json").write_text(
        json.dumps(effective, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_artifacts(root, states, environments)
    for episode_id in range(len(states) * len(environments)):
        entry = manifest["episodes"][_episode_key(episode_id)]
        if entry.get("status") == "completed" and not is_valid_episode(
            _episode_path(root, episode_id),
            expected_steps=steps,
            expected_episode_id=episode_id,
        ):
            entry.update(status="pending", sample_count=0, last_error="invalid episode file")
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(manifest_path, manifest)
    return effective, states, environments, manifest_path


def _set_initial_kinematics(client: Any, state: InitialState, vehicle_name: str) -> AirSimDrone:
    airsim = require_airsim()
    client.reset()
    try:
        client.simPause(True)
    except Exception:
        pass
    drone = AirSimDrone(client, vehicle_name)
    drone.enable_api_control(True)
    drone.arm()
    kinematics = airsim.KinematicsState()
    kinematics.position = airsim.Vector3r(*[float(value) for value in state.position])
    kinematics.orientation = airsim.Quaternionr(
        x_val=float(state.orientation[1]),
        y_val=float(state.orientation[2]),
        z_val=float(state.orientation[3]),
        w_val=float(state.orientation[0]),
    )
    kinematics.linear_velocity = airsim.Vector3r(*[float(value) for value in state.velocity])
    kinematics.angular_velocity = airsim.Vector3r(*[float(value) for value in state.angular_velocity])
    kinematics.linear_acceleration = airsim.Vector3r(0.0, 0.0, 0.0)
    kinematics.angular_acceleration = airsim.Vector3r(0.0, 0.0, 0.0)
    client.simSetKinematics(kinematics, ignore_collision=True, vehicle_name=vehicle_name)
    return drone


def _check_state(client: Any, drone: AirSimDrone, state: RawDroneState, config: Mapping[str, Any], step: int) -> None:
    values = (state.position, state.quaternion, state.velocity, state.acceleration, state.angular_velocity)
    if not all(np.isfinite(value).all() for value in values):
        raise EpisodeFailure("AirSim returned NaN or Inf")
    bounds = config["bounds"]
    lower = np.asarray(bounds["position_min"], dtype=np.float64)
    upper = np.asarray(bounds["position_max"], dtype=np.float64)
    if np.any(state.position < lower) or np.any(state.position > upper):
        raise EpisodeFailure(f"vehicle left position bounds: {state.position.tolist()}")
    collision = client.simGetCollisionInfo(vehicle_name=drone.vehicle_name)
    if bool(getattr(collision, "has_collided", False)):
        object_name = str(getattr(collision, "object_name", "unknown"))
        reset_ground_flag = step == 0 and "ground" in object_name.lower() and state.position[2] < -1.0
        if not reset_ground_flag:
            raise EpisodeFailure(f"vehicle collision with {object_name}")


def _hold_reference(position: np.ndarray, orientation: np.ndarray, horizon: int) -> list[MPCReference]:
    return [
        MPCReference(position=position, orientation=orientation, velocity=np.zeros(3), acceleration=np.zeros(3))
        for _ in range(horizon)
    ]


def run_episode(
    client: Any,
    initial_state: InitialState,
    environment: TrainingEnvironment,
    *,
    episode_id: int,
    config: Mapping[str, Any],
    calibration: ThrustCalibration,
    planner: QuinticMotionPrimitivePlanner,
    controller: Any,
) -> EpisodeSamples:
    timing = config["timing"]
    dt = float(timing["dt"])
    steps = int(timing["steps_per_episode"])
    vehicle_name = str(config["vehicle_name"])
    gate_cfg = config["gate"]
    controller.reset()
    drone = _set_initial_kinematics(client, initial_state, vehicle_name)
    scene = None
    simulation_unpaused = False
    try:
        if bool(gate_cfg.get("spawn_in_simulation", False)):
            cleanup_generated_gates(client, prefix="FalconPaperGate_")
            scene = spawn_gate_sequence(
                client,
                environment.gates,
                width=float(gate_cfg["width"]),
                height=float(gate_cfg["height"]),
                thickness=float(gate_cfg["thickness"]),
                asset_name=str(gate_cfg["asset"]),
                object_prefix="FalconPaperGate_",
                world_offset=vehicle_local_to_world_offset(client, vehicle_name),
            )
            preflight_hold = float(gate_cfg.get("preflight_hold_seconds", 0.0))
            if preflight_hold > 0.0:
                print(
                    f"Visible gates spawned; holding the paused simulation for "
                    f"{preflight_hold:.1f}s before flight.",
                    flush=True,
                )
                time.sleep(preflight_hold)
        client.simPause(False)
        simulation_unpaused = True
        first = drone.get_raw_state()
        route = ContinuousGateRoute(
            environment.gates,
            first.position,
            approach_distance=float(gate_cfg["approach_distance"]),
            exit_distance=float(gate_cfg["exit_distance"]),
            reference_speed=float(config["planner"]["nominal_speed"]),
            start_time=time.perf_counter(),
            gate_width=float(gate_cfg["width"]),
            gate_height=float(gate_cfg["height"]),
            gate_thickness=float(gate_cfg["thickness"]),
            clearance=float(gate_cfg["clearance"]),
            completion_radius=float(gate_cfg["threshold"]),
        )
        continuous_mode = _uses_continuous_trajectory(config)
        continuous_trajectory: ContinuousQuinticTrajectory | None = None
        trajectory_elapsed = 0.0
        if continuous_mode:
            trajectory_config = config.get("trajectory", {})
            start_velocity_mode = str(
                trajectory_config.get("start_velocity_mode", "measured")
            ).lower()
            if start_velocity_mode == "measured":
                reference_start_velocity = first.velocity
            elif start_velocity_mode == "nominal_gate_axis":
                reference_start_velocity = (
                    float(config["planner"]["nominal_speed"])
                    * planner.passage_axis(environment.gates[0].orientation)
                )
            else:
                raise EpisodeFailure(
                    "trajectory.start_velocity_mode must be 'measured' or "
                    f"'nominal_gate_axis', got {start_velocity_mode!r}"
                )
            continuous_trajectory = planner.plan_continuous_route(
                first.position,
                reference_start_velocity,
                np.zeros(3),
                environment.gates,
                final_exit_distance=float(
                    trajectory_config.get("final_exit_distance", 5.0)
                ),
            )
        storage: dict[str, list[Any]] = {
            key: [] for key in (
                "X", "U", "gate", "timestamp", "position", "quaternion", "velocity",
                "acceleration", "reference_position", "reference_orientation", "solve",
                "loop", "miss", "context",
            )
        }
        started = time.perf_counter()
        next_deadline = started
        last_start = started
        active_gate_id: int | None = None
        active_primitive: QuinticPolynomial | None = None
        primitive_elapsed = 0.0
        for step in range(steps):
            period_start = time.perf_counter()
            if period_start - started > float(timing["episode_timeout_seconds"]):
                raise EpisodeFailure("episode wall-clock timeout")
            loop_period = dt if step == 0 else period_start - last_start
            if loop_period > float(timing.get("catastrophic_loop_period", 0.5)):
                raise EpisodeFailure(f"catastrophic loop overrun: {loop_period:.3f}s")
            raw = drone.get_raw_state()
            _check_state(client, drone, raw, config, step)
            try:
                route.update(raw.position, period_start)
            except GatePassageError as exc:
                raise EpisodeFailure(str(exc)) from exc
            gate = route.current_gate
            reference_dt = float(getattr(controller.config, "prediction_dt", dt))
            reference_nodes = int(controller.config.horizon) + 1
            if continuous_mode:
                assert continuous_trajectory is not None
                references = planner.reference_window_for_trajectory(
                    continuous_trajectory,
                    elapsed=trajectory_elapsed,
                    dt=reference_dt,
                    horizon=reference_nodes,
                )
                x = build_trajectory_input(raw, references[0])
            elif route.completed:
                references = _hold_reference(
                    route.exit_points[-1], gate.orientation, reference_nodes
                )
                x = build_gate_input(raw, gate.center, gate.orientation)
            else:
                if active_gate_id != route.target_gate_id or active_primitive is None:
                    # AirSim may report one free-fall/gravity acceleration sample
                    # immediately after simSetKinematics even though the protocol's
                    # randomized initial state specifies p/q/v (not acceleration).
                    # Do not bake that simulator reset transient into the first
                    # motion primitive.  Later gate transitions use the measured
                    # closed-loop acceleration as their true segment boundary.
                    planning_acceleration = (
                        np.zeros(3) if active_gate_id is None else raw.acceleration
                    )
                    try:
                        active_primitive = planner.plan(
                            raw.position,
                            raw.velocity,
                            planning_acceleration,
                            gate.center,
                            gate.orientation,
                        )
                    except MotionPrimitiveInfeasible as exc:
                        raise EpisodeFailure(
                            f"Gate {route.target_gate_id} motion primitive infeasible: {exc}"
                        ) from exc
                    active_gate_id = route.target_gate_id
                    primitive_elapsed = 0.0
                references = planner.reference_window_for_primitive(
                    active_primitive,
                    gate_orientation=gate.orientation,
                    elapsed=primitive_elapsed,
                    dt=reference_dt,
                    horizon=reference_nodes,
                )
                x = build_gate_input(raw, gate.center, gate.orientation)
            solve_start = time.perf_counter()
            control = controller.solve(raw, references)
            solve_time = time.perf_counter() - solve_start
            if control.shape != (4,) or not np.isfinite(control).all():
                raise EpisodeFailure("MPC returned an invalid control")
            send_control(drone, control, calibration, duration=dt)
            if continuous_mode:
                trajectory_elapsed += dt
            elif not route.completed:
                primitive_elapsed += dt
            next_deadline += dt
            miss = int(time.perf_counter() > next_deadline)
            storage["X"].append(x)
            storage["U"].append(control)
            storage["gate"].append(route.target_gate_id)
            storage["timestamp"].append(time.time())
            storage["position"].append(raw.position.copy())
            storage["quaternion"].append(raw.quaternion.copy())
            storage["velocity"].append(raw.velocity.copy())
            storage["acceleration"].append(raw.acceleration.copy())
            storage["reference_position"].append(references[0].position.copy())
            storage["reference_orientation"].append(references[0].orientation.copy())
            storage["solve"].append(solve_time)
            storage["loop"].append(loop_period)
            storage["miss"].append(miss)
            storage["context"].append(
                np.asarray([references[index].position - raw.position for index in (0, 5, 10, 19)])
            )
            progress = int(config["dataset"].get("progress_every_steps", 0))
            if progress > 0 and step % progress == 0:
                print(
                    f"paper episode={episode_id:03d} step={step:03d}/{steps} "
                    f"gate={route.target_gate_id} speed={np.linalg.norm(raw.velocity):.2f} "
                    f"solve={solve_time * 1e3:.2f}ms",
                    flush=True,
                )
            last_start = period_start
            sleep_time = next_deadline - time.perf_counter()
            if sleep_time > 0.0:
                time.sleep(sleep_time)
        return EpisodeSamples(
            X=np.asarray(storage["X"]),
            U=np.asarray(storage["U"]),
            episode_id=episode_id,
            initial_state_id=initial_state.initial_state_id,
            environment_id=environment.environment_id,
            step_id=np.arange(steps),
            target_gate_id=np.asarray(storage["gate"]),
            timestamp=np.asarray(storage["timestamp"]),
            position=np.asarray(storage["position"]),
            quaternion=np.asarray(storage["quaternion"]),
            velocity=np.asarray(storage["velocity"]),
            acceleration=np.asarray(storage["acceleration"]),
            reference_position=np.asarray(storage["reference_position"]),
            reference_orientation=np.asarray(storage["reference_orientation"]),
            mpc_solve_time=np.asarray(storage["solve"]),
            actual_loop_period=np.asarray(storage["loop"]),
            deadline_miss=np.asarray(storage["miss"]),
            path_context=np.asarray(storage["context"]),
            reference_speed=np.full(steps, float(config["planner"]["nominal_speed"])),
            data_source=np.zeros(steps, dtype=np.int8),
        )
    finally:
        active_error = sys.exc_info()[1]
        if simulation_unpaused:
            try:
                client.simPause(False)
            except Exception:
                pass
        if (
            scene is not None
            and active_error is None
            and not bool(gate_cfg.get("keep_spawned_gates", False))
        ):
            scene.destroy()
        try:
            drone.hover()
        except Exception:
            pass


def _connect(config: Mapping[str, Any]) -> Any:
    airsim = require_airsim()
    client = airsim.MultirotorClient(
        ip=str(config.get("rpc_host", "127.0.0.1")),
        port=int(config.get("rpc_port", 41451)),
        timeout_value=float(config.get("rpc_timeout", 10.0)),
    )
    client.confirmConnection()
    return client


def _build_controller(config: Mapping[str, Any], base: Path) -> Any:
    values = dict(config["mpc"])
    backend = _mpc_backend(config)
    values.pop("backend", None)
    if backend == "official_rpg_mpc_acado_qpoases":
        if "library" in values:
            values["library"] = str(_resolve(values["library"], base))
        return AcadoRtiMPC(AcadoMPCConfig(**values))
    values.pop("prediction_dt", None)
    values.pop("library", None)
    values["dt"] = float(config["timing"]["dt"])
    return CasadiTrackingMPC(NonlinearMPCConfig(**values))


def collect(
    config_path: str | Path,
    *,
    output_root: str | Path | None = None,
    resume: bool = False,
    smoke_test: bool = False,
    auto_launch: bool = False,
    render: bool = False,
) -> Path:
    config, resolved_config = _load_config(config_path)
    base = resolved_config.parent
    root = _resolve(output_root or config["dataset"]["root"], base)
    if smoke_test and output_root is None:
        root = root.with_name(root.name + "_smoke")
    config, states, environments, manifest_path = _prepare(
        root, config, resolved_config, resume=resume, smoke_test=smoke_test
    )
    # Rendering is an execution preference, not part of the immutable teacher
    # protocol.  A resumed dataset may therefore switch between visible and
    # headless AirSim without invalidating its config hash.
    if render:
        config["airsim"]["null_rhi"] = False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    steps = int(config["timing"]["steps_per_episode"])
    airsim_cfg = config["airsim"]
    calibration_path = config.get("calibration")
    calibration = load_calibration(_resolve(calibration_path, base) if calibration_path else None)
    manager: AirSimProcessManager | None = None
    client: Any = None
    collection_start = time.perf_counter()
    try:
        if auto_launch:
            launch = AirSimLaunchConfig.from_paths(
                airsim_cfg["editor"],
                airsim_cfg["uproject"],
                map_asset=airsim_cfg.get("map_asset"),
                log_dir=_resolve(config.get("log_dir", "logs"), base),
                startup_timeout=float(airsim_cfg["startup_timeout"]),
                retry_interval=float(airsim_cfg["retry_interval"]),
                rpc_host=str(airsim_cfg.get("rpc_host", "127.0.0.1")),
                rpc_port=int(airsim_cfg.get("rpc_port", 41451)),
                rpc_timeout=float(airsim_cfg.get("rpc_timeout", 10.0)),
                res_x=int(airsim_cfg["res_x"]),
                res_y=int(airsim_cfg["res_y"]),
                null_rhi=bool(airsim_cfg.get("null_rhi", True)),
            )
            manager = AirSimProcessManager(launch)
            client = manager.start_and_wait()
        else:
            client = _connect(airsim_cfg)
        planner = QuinticMotionPrimitivePlanner(MotionPrimitiveConfig(**config["planner"]))
        controller = _build_controller(config, base)
        total = len(states) * len(environments)
        retry_count = int(config["dataset"].get("retry_count", 1))
        abort_after_consecutive_failures = int(
            config["dataset"].get("abort_after_consecutive_failures", 0)
        )
        consecutive_failed_episodes = 0
        completed_since_restart = 0
        for episode_id in range(total):
            entry = manifest["episodes"][_episode_key(episode_id)]
            path = _episode_path(root, episode_id)
            if entry.get("status") == "completed" and is_valid_episode(
                path, expected_steps=steps, expected_episode_id=episode_id
            ):
                continue
            # AirSim 1.x can leave its RPC thread alive but permanently
            # unresponsive after hundreds of reset/simPause cycles. Every
            # episode starts from a fully specified state, so periodically
            # recycling only the process is safe and does not alter the data
            # protocol or the immutable collection configuration.
            if (
                manager is not None
                and completed_since_restart >= MANAGED_AIRSIM_RESTART_INTERVAL_EPISODES
            ):
                print(
                    f"Recycling managed AirSim before {_episode_key(episode_id)} "
                    f"after {completed_since_restart} completed episodes.",
                    flush=True,
                )
                client = manager.restart()
                _record_simulator_restart(
                    manifest,
                    episode_id=episode_id,
                    reason="periodic preventive restart",
                )
                completed_since_restart = 0
                manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
                atomic_write_json(manifest_path, manifest)
            state = states[int(entry["initial_state_id"])]
            environment = environments[int(entry["environment_id"])]
            episode_completed = False
            for attempt in range(1, retry_count + 2):
                entry.update(status="running", attempts=attempt)
                manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
                atomic_write_json(manifest_path, manifest)
                try:
                    episode = run_episode(
                        client,
                        state,
                        environment,
                        episode_id=episode_id,
                        config=config,
                        calibration=calibration,
                        planner=planner,
                        controller=controller,
                    )
                    save_episode_atomic(path, episode)
                    entry.update(status="completed", sample_count=steps, last_error=None)
                    episode_completed = True
                    completed_since_restart += 1
                    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
                    atomic_write_json(manifest_path, manifest)
                    completed = sum(item["status"] == "completed" for item in manifest["episodes"].values())
                    elapsed = time.perf_counter() - collection_start
                    eta = elapsed / max(completed, 1) * (total - completed)
                    print(
                        f"Paper core: {completed:03d}/{total:03d} episodes | "
                        f"{completed * steps}/{total * steps} samples | ETA={eta:.1f}s",
                        flush=True,
                    )
                    break
                except KeyboardInterrupt:
                    entry.update(status="interrupted", last_error="KeyboardInterrupt")
                    atomic_write_json(manifest_path, manifest)
                    raise
                except Exception as exc:
                    entry["last_error"] = f"{type(exc).__name__}: {exc}"
                    if _is_rpc_transport_failure(exc):
                        if attempt > retry_count:
                            entry["status"] = "interrupted"
                            manifest.update(
                                status="interrupted",
                                updated_at=datetime.now(timezone.utc).isoformat(),
                            )
                            atomic_write_json(manifest_path, manifest)
                            raise AirSimProcessError(
                                f"AirSim RPC failed {attempt} times at "
                                f"{_episode_key(episode_id)}; collection stopped so later "
                                "episodes are not incorrectly marked as data failures"
                            ) from exc
                        print(
                            f"{_episode_key(episode_id)} RPC failure on attempt {attempt}: "
                            f"{entry['last_error']}; restarting AirSim before retry.",
                            flush=True,
                        )
                        try:
                            client = manager.restart() if manager is not None else _connect(airsim_cfg)
                        except Exception as recovery_exc:
                            entry.update(
                                status="interrupted",
                                last_error=(
                                    f"RPC recovery failed after {type(exc).__name__}: {exc}; "
                                    f"{type(recovery_exc).__name__}: {recovery_exc}"
                                ),
                            )
                            manifest.update(
                                status="interrupted",
                                updated_at=datetime.now(timezone.utc).isoformat(),
                            )
                            atomic_write_json(manifest_path, manifest)
                            raise AirSimProcessError(
                                f"Could not recover AirSim at {_episode_key(episode_id)}"
                            ) from recovery_exc
                        _record_simulator_restart(
                            manifest,
                            episode_id=episode_id,
                            reason=entry["last_error"],
                        )
                        completed_since_restart = 0
                        entry["status"] = "pending"
                        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
                        atomic_write_json(manifest_path, manifest)
                        print(
                            f"AirSim RPC recovered; retrying {_episode_key(episode_id)}.",
                            flush=True,
                        )
                        continue
                    if attempt > retry_count:
                        entry["status"] = "failed"
                        atomic_write_json(manifest_path, manifest)
                        print(f"{_episode_key(episode_id)} failed: {entry['last_error']}", flush=True)
                        break
                    print(f"{_episode_key(episode_id)} retry {attempt}: {entry['last_error']}", flush=True)
            if episode_completed:
                consecutive_failed_episodes = 0
            else:
                consecutive_failed_episodes += 1
                if (
                    abort_after_consecutive_failures > 0
                    and consecutive_failed_episodes >= abort_after_consecutive_failures
                ):
                    manifest.update(
                        status="interrupted",
                        updated_at=datetime.now(timezone.utc).isoformat(),
                    )
                    atomic_write_json(manifest_path, manifest)
                    raise EpisodeFailure(
                        "aborting collection after "
                        f"{consecutive_failed_episodes} consecutive failed episodes; "
                        "fix the physical/configuration failure before resuming"
                    )
        failed = [
            index for index in range(total)
            if manifest["episodes"][_episode_key(index)]["status"] != "completed"
        ]
        if failed:
            manifest.update(status="incomplete", failed_episode_ids=failed)
            atomic_write_json(manifest_path, manifest)
            write_summary(root / "summary.txt", {"status": "incomplete", "failed_episode_ids": failed})
            print(f"Collection incomplete. Resume after fixing failures: {failed}")
            return root
        final = root / "final" / str(config["dataset"]["output_name"])
        stats = merge_episodes(
            [_episode_path(root, index) for index in range(total)],
            final,
            expected_steps=steps,
            expected_episode_count=total,
            metadata_csv_path=root / "final" / "teacher_metadata.csv",
        )
        validate_merged_dataset(
            final,
            expected_samples=total * steps,
            expected_episodes=total,
            expected_steps=steps,
            expected_initial_states=len(states),
            expected_environments=len(environments),
            expected_input_dimensions=12,
        )
        manifest.update(
            status="completed",
            final_dataset=str(Path("final") / final.name),
            statistics=stats,
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        atomic_write_json(manifest_path, manifest)
        write_summary(root / "summary.txt", {"status": "completed", **stats})
        print(f"Paper core complete: X={stats['X_shape']} U={stats['U_shape']}")
        return root
    finally:
        if manager is not None:
            manager.stop()
