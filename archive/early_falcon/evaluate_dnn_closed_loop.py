"""Automatically evaluate the DNN controller on held-out AirSim layouts."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from airsim_interface import AirSimDrone, require_airsim, vehicle_local_to_world_offset
from airsim_process import AirSimLaunchConfig, AirSimProcessManager
from config import ThrustCalibration, ensure_runtime_directories, load_calibration
from control_adapter import send_control
from dnn_controller import DNNController
from dnn_model_utils import build_policy_input
from gate_route import ContinuousGateRoute, GatePassageError
from gate_visualization import (
    DEFAULT_GATE_ASSET,
    DEFAULT_GATE_HEIGHT,
    DEFAULT_GATE_THICKNESS,
    DEFAULT_GATE_WIDTH,
    cleanup_generated_gates,
    spawn_gate_sequence,
)
from initial_state_generator import InitialState, load_initial_states
from state_adapter import build_controller_state
from training_environment_generator import TrainingEnvironment, load_training_environments
from mpc_controller import MPCController


def _join(future: object) -> None:
    join = getattr(future, "join", None)
    if callable(join):
        join()


def _pose(airsim: object, state: InitialState) -> object:
    return airsim.Pose(
        position_val=airsim.Vector3r(*[float(value) for value in state.position]),
        orientation_val=airsim.Quaternionr(
            x_val=float(state.orientation[1]),
            y_val=float(state.orientation[2]),
            z_val=float(state.orientation[3]),
            w_val=float(state.orientation[0]),
        ),
    )


def _check_state_and_collision(client: object, drone: AirSimDrone, *, first_sample: bool = False) -> object:
    raw = drone.get_raw_state()
    values = (raw.position, raw.quaternion, raw.velocity, raw.acceleration, raw.angular_velocity)
    if not all(np.isfinite(value).all() for value in values):
        raise RuntimeError("AirSim returned NaN or Inf")
    collision = client.simGetCollisionInfo(vehicle_name=drone.vehicle_name)
    if bool(getattr(collision, "has_collided", False)):
        object_name = getattr(collision, "object_name", "unknown")
        if not first_sample:
            raise RuntimeError(f"collision with {object_name}")
        print(f"Ignoring AirSim reset collision flag on first sample: object={object_name}", flush=True)
    return raw


def _run_episode(
    client: object,
    controller: object,
    environment: TrainingEnvironment,
    initial_state: InitialState,
    *,
    vehicle_name: str,
    world_offset: np.ndarray,
    calibration: ThrustCalibration,
    gate_width: float,
    gate_height: float,
    gate_thickness: float,
    gate_asset: str,
    gate_clearance: float,
    speed: float,
    threshold: float,
    per_gate_timeout: float,
) -> dict[str, object]:
    airsim = require_airsim()
    drone: AirSimDrone | None = None
    scene = None
    crossings: list[dict[str, float]] = []
    samples = 0
    misses = 0
    inference_times: list[float] = []
    started = time.perf_counter()
    try:
        client.reset()
        client.simPause(True)
        client.simSetVehiclePose(_pose(airsim, initial_state), ignore_collision=True, vehicle_name=vehicle_name)
        drone = AirSimDrone(client, vehicle_name)
        drone.enable_api_control(True)
        drone.arm()
        cleanup_generated_gates(client, prefix="FalconGate_")
        cleanup_generated_gates(client, prefix="FalconDatasetGate_")
        cleanup_generated_gates(client, prefix="FalconDnnEvalGate_")
        scene = spawn_gate_sequence(
            client,
            environment.gates,
            width=gate_width,
            height=gate_height,
            thickness=gate_thickness,
            asset_name=gate_asset,
            object_prefix="FalconDnnEvalGate_",
            world_offset=world_offset,
        )
        client.simPause(False)
        first = drone.get_raw_state()
        route_start = time.perf_counter()
        route = ContinuousGateRoute(
            environment.gates,
            first.position,
            approach_distance=2.0,
            exit_distance=1.0,
            reference_speed=speed,
            start_time=route_start,
            gate_width=gate_width,
            gate_height=gate_height,
            gate_thickness=gate_thickness,
            clearance=gate_clearance,
            completion_radius=threshold,
        )
        next_deadline = route_start
        last_start = route_start
        while not route.completed:
            period_start = time.perf_counter()
            if period_start - route.gate_start_time > per_gate_timeout:
                raise TimeoutError(f"gate {route.target_gate_id} timeout {per_gate_timeout:.1f}s")
            raw = _check_state_and_collision(client, drone, first_sample=(samples == 0))
            crossing = route.update(raw.position, period_start)
            if crossing is not None:
                crossings.append({
                    "gate": float(crossing.gate_index),
                    "lateral_offset": float(crossing.lateral_offset),
                    "vertical_offset": float(crossing.vertical_offset),
                })
            if route.completed:
                break
            window = route.reference_window(period_start, dt=0.05, horizon=20)
            reference = window[0]
            state = build_controller_state(raw, reference.position, reference.orientation)
            inference_start = time.perf_counter()
            if isinstance(controller, MPCController):
                control = controller.solve(state, window)
            else:
                policy_input = build_policy_input(
                    state,
                    window,
                    input_dim=controller.input_dim,
                    reference_speed=speed,
                )
                control = controller.solve(policy_input)
            inference_times.append(time.perf_counter() - inference_start)
            send_control(drone, control, calibration, duration=0.05)
            next_deadline += 0.05
            loop_period = 0.05 if samples == 0 else period_start - last_start
            misses += int(time.perf_counter() > next_deadline)
            samples += 1
            last_start = period_start
            sleep_time = next_deadline - time.perf_counter()
            if sleep_time > 0.0:
                time.sleep(sleep_time)
        return {
            "environment_id": environment.environment_id,
            "initial_state_id": initial_state.initial_state_id,
            "epsilon": environment.epsilon,
            "difficulty": environment.difficulty,
            "success": True,
            "samples": samples,
            "deadline_misses": misses,
            "crossings": crossings,
            "mean_inference_ms": float(np.mean(inference_times) * 1e3) if inference_times else None,
            "max_inference_ms": float(np.max(inference_times) * 1e3) if inference_times else None,
            "elapsed_seconds": time.perf_counter() - started,
            "error": None,
        }
    except Exception as exc:
        return {
            "environment_id": environment.environment_id,
            "initial_state_id": initial_state.initial_state_id,
            "epsilon": environment.epsilon,
            "difficulty": environment.difficulty,
            "success": False,
            "samples": samples,
            "deadline_misses": misses,
            "crossings": crossings,
            "mean_inference_ms": float(np.mean(inference_times) * 1e3) if inference_times else None,
            "max_inference_ms": float(np.max(inference_times) * 1e3) if inference_times else None,
            "elapsed_seconds": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        if drone is not None:
            try:
                _join(drone.hover())
                _join(drone.land())
                drone.disarm()
                drone.enable_api_control(False)
            except Exception:
                pass
        if scene is not None:
            scene.destroy()


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    ensure_runtime_directories()
    environments = {item.environment_id: item for item in load_training_environments(args.layout_root / "environments.json")}
    states = {item.initial_state_id: item for item in load_initial_states(args.layout_root / "initial_states.json")}
    environment_ids = [int(value) for value in args.environment_ids.split(",") if value.strip()]
    state_ids = [int(value) for value in args.initial_state_ids.split(",") if value.strip()]
    missing_environments = [environment_id for environment_id in environment_ids if environment_id not in environments]
    missing_states = [state_id for state_id in state_ids if state_id not in states]
    if missing_environments or missing_states:
        raise ValueError(
            f"requested IDs are not available: environments={missing_environments}, states={missing_states}"
        )
    selected = [(environments[environment_id], states[state_id]) for state_id in state_ids for environment_id in environment_ids]
    if args.max_episodes > 0:
        selected = selected[:args.max_episodes]
    controller = DNNController(args.model, args.normalization, device=args.device)
    manager = args.process_manager
    owns_manager = manager is not None
    if manager is not None:
        client = manager.start_and_wait()
    else:
        airsim = require_airsim()
        client = airsim.MultirotorClient()
        client.confirmConnection()
    client.reset()
    time.sleep(0.5)
    vehicle_name = args.vehicle_name
    if not vehicle_name:
        names = list(client.listVehicles())
        if not names:
            raise RuntimeError("AirSim did not report any vehicles")
        vehicle_name = str(names[0])
    world_offset = vehicle_local_to_world_offset(client, vehicle_name)
    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    try:
        for index, (environment, initial_state) in enumerate(selected, start=1):
            print(
                f"heldout={index:02d}/{len(selected)} environment={environment.environment_id} "
                f"initial_state={initial_state.initial_state_id}", flush=True
            )
            result = _run_episode(
                client,
                controller,
                environment,
                initial_state,
                vehicle_name=vehicle_name,
                world_offset=world_offset,
                calibration=args.calibration,
                gate_width=args.gate_width,
                gate_height=args.gate_height,
                gate_thickness=args.gate_thickness,
                gate_asset=args.gate_asset,
                gate_clearance=args.gate_clearance,
                speed=args.speed,
                threshold=args.threshold,
                per_gate_timeout=args.per_gate_timeout,
            )
            rows.append(result)
            print(
                f"  success={result['success']} crossings={len(result['crossings'])}/"
                f"{environment.gates.__len__()} samples={result['samples']} error={result['error']}",
                flush=True,
            )
    finally:
        if owns_manager and manager is not None:
            manager.stop()
    report_path = args.report or (args.model.parent / "heldout_closed_loop.json")
    csv_path = report_path.with_suffix(".csv")
    report = {
        "controller": "DNN",
        "model": str(args.model.resolve()),
        "normalization": str((args.normalization or args.model.parent / "normalization.json").resolve()),
        "vehicle_name": vehicle_name,
        "environment_ids": environment_ids,
        "initial_state_ids": state_ids,
        "episodes_requested": len(selected),
        "episodes_completed": len(rows),
        "successes": int(sum(bool(row["success"]) for row in rows)),
        "failures": int(sum(not bool(row["success"]) for row in rows)),
        "success_rate": float(np.mean([bool(row["success"]) for row in rows])) if rows else 0.0,
        "elapsed_seconds": time.perf_counter() - started,
        "results": rows,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("environment_id", "initial_state_id", "epsilon", "difficulty", "success", "samples", "deadline_misses", "mean_inference_ms", "max_inference_ms", "elapsed_seconds", "error"))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in writer.fieldnames})
    print(json.dumps({key: report[key] for key in ("episodes_requested", "episodes_completed", "successes", "failures", "success_rate", "elapsed_seconds")}, indent=2), flush=True)
    print(f"Report: {report_path.resolve()}")
    print(f"CSV: {csv_path.resolve()}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, default=None)
    parser.add_argument("--layout-root", type=Path, default=Path("teacher_dataset_full_route_continuous_v2"))
    parser.add_argument("--environment-ids", default="36,37,38,39")
    parser.add_argument("--initial-state-ids", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--vehicle-name", default="Drone1")
    parser.add_argument("--epsilon", type=float, default=None, help="retained for command compatibility; layouts come from metadata")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--per-gate-timeout", type=float, default=30.0)
    parser.add_argument("--speed", type=float, default=2.0)
    parser.add_argument("--gate-width", type=float, default=DEFAULT_GATE_WIDTH)
    parser.add_argument("--gate-height", type=float, default=DEFAULT_GATE_HEIGHT)
    parser.add_argument("--gate-thickness", type=float, default=DEFAULT_GATE_THICKNESS)
    parser.add_argument("--gate-clearance", type=float, default=0.25)
    parser.add_argument("--gate-asset", default=DEFAULT_GATE_ASSET)
    parser.add_argument("--calibration", type=load_calibration, default=load_calibration())
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--auto-launch", action="store_true")
    parser.add_argument("--render", action="store_true", help="use a visible UE4 process instead of -nullrhi")
    parser.add_argument("--ue4-editor", type=Path, default=Path(r"E:\epicGamesStrg\UE_4.27\Engine\Binaries\Win64\UE4Editor.exe"))
    parser.add_argument("--uproject", type=Path, default=Path(r"F:\Airsim\AirSim\Unreal\Environments\Blocks\Blocks.uproject"))
    parser.add_argument("--startup-timeout", type=float, default=120.0)
    args = parser.parse_args()
    args.process_manager = None
    if args.auto_launch:
        args.process_manager = AirSimProcessManager(
            AirSimLaunchConfig.from_paths(
                args.ue4_editor,
                args.uproject,
                log_dir=Path("logs"),
                startup_timeout=args.startup_timeout,
                null_rhi=not args.render,
            )
        )
    evaluate(args)


if __name__ == "__main__":
    main()
