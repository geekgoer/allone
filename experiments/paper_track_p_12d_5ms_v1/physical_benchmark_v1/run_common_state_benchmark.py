"""Fly one MPC teacher and compare six DNNs on exactly the same AirSim states.

The DNN inference pass runs after each flight.  It therefore cannot delay or
perturb the 20 Hz teacher loop.  Only complete, collision-free five-gate
teacher episodes contribute to the aggregate benchmark.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from airsim_interface import AirSimDrone, require_airsim, vehicle_local_to_world_offset
from benchmark_utils import load_config, regression_metrics, resolve_project_path, select_scenarios
from config import load_calibration
from control_adapter import send_control
from dnn_model_utils import denormalized_output, load_normalization, normalized_input
from environment import GateEnvironment
from gate_route import ContinuousGateRoute, GatePassageError
from gate_visualization import cleanup_generated_gates, spawn_gate_sequence
from initial_state_generator import InitialState, load_initial_states
from mpc_controller import MPCController
from run_single_drone_dnn_gates import _pose
from state_adapter import build_controller_state
from track_p_12d_contract import solve_track_p_teacher


OUTPUT_NAMES = ("F", "wx", "wy", "wz")


def _join(future: object) -> None:
    join = getattr(future, "join", None)
    if callable(join):
        join()


def _predict(model_item: dict[str, Any], x: np.ndarray, device: str, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    import torch

    model_path = resolve_project_path(model_item["model"])
    normalization = load_normalization(resolve_project_path(model_item["normalization"]), expected_input_dim=12)
    model = torch.jit.load(str(model_path), map_location=torch.device(device)).eval()
    x_normalized = normalized_input(x, normalization)
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            tensor = torch.from_numpy(x_normalized[start : start + batch_size]).to(device)
            chunks.append(model(tensor).detach().cpu().numpy())
    prediction_normalized = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
    prediction = denormalized_output(prediction_normalized, normalization)
    return prediction, prediction_normalized


def _collision(client: Any, vehicle_name: str, *, first_sample: bool) -> str | None:
    info = client.simGetCollisionInfo(vehicle_name=vehicle_name)
    if not bool(getattr(info, "has_collided", False)):
        return None
    name = str(getattr(info, "object_name", "unknown"))
    if first_sample:
        print(f"Ignoring reset/pose collision flag on first sample: {name}", flush=True)
        return None
    return name


def _find_initial_state(states: list[InitialState], state_id: int) -> InitialState:
    for state in states:
        if state.initial_state_id == state_id:
            return state
    raise ValueError(f"initial_state_id {state_id} was not found")


def _run_teacher_episode(
    client: Any,
    vehicle_name: str,
    scenario: dict[str, Any],
    config: dict[str, Any],
    initial_state: InitialState,
    output_dir: Path,
    *,
    keep_gates: bool,
) -> dict[str, Any]:
    timing = config["timing"]
    gate_config = config["gate"]
    mpc_config = config["mpc"]
    dt = float(timing["dt"])
    horizon = int(timing["horizon"])
    environment = GateEnvironment(
        epsilon=float(scenario["epsilon"]),
        seed=int(scenario["seed"]),
        base_y=float(gate_config["base_y"]),
        base_z=float(gate_config["base_z"]),
        max_z=float(gate_config["max_z"]),
        gate_count=int(gate_config["count"]),
        x_start=float(gate_config["x_start"]),
        x_spacing=float(gate_config["x_spacing"]),
    )
    gates = environment.generate()
    controller = MPCController(
        dt=dt,
        horizon=horizon,
        max_horizontal_accel=float(mpc_config["max_horizontal_accel"]),
        max_thrust_accel=float(mpc_config["max_thrust_accel"]),
        max_body_rate=float(mpc_config["max_body_rate"]),
    )
    calibration = load_calibration()
    drone = AirSimDrone(client, vehicle_name)
    client.reset()
    time.sleep(0.4)
    world_offset = vehicle_local_to_world_offset(client, vehicle_name)
    cleanup_generated_gates(client, "FalconGate_")
    cleanup_generated_gates(client, "FalconBenchmarkGate_")
    scene = spawn_gate_sequence(
        client,
        gates,
        width=float(gate_config["width"]),
        height=float(gate_config["height"]),
        thickness=float(gate_config["thickness"]),
        asset_name=str(gate_config["asset"]),
        object_prefix="FalconBenchmarkGate_",
        world_offset=world_offset,
    )
    x_rows: list[np.ndarray] = []
    u_rows: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    gate_ids: list[int] = []
    solve_times: list[float] = []
    deadline_misses = 0
    crossings: list[dict[str, float]] = []
    failure: str | None = None
    started = time.perf_counter()
    try:
        client.simPause(True)
        client.simSetVehiclePose(
            _pose(require_airsim(), initial_state),
            ignore_collision=True,
            vehicle_name=vehicle_name,
        )
        drone.enable_api_control(True)
        drone.arm()
        client.simPause(False)
        _join(drone.hover())
        time.sleep(0.1)
        first = drone.get_raw_state()
        route_start = time.perf_counter()
        route = ContinuousGateRoute(
            gates,
            first.position,
            approach_distance=float(gate_config["approach_distance"]),
            exit_distance=float(gate_config["exit_distance"]),
            reference_speed=float(mpc_config["reference_speed"]),
            start_time=route_start,
            gate_width=float(gate_config["width"]),
            gate_height=float(gate_config["height"]),
            gate_thickness=float(gate_config["thickness"]),
            clearance=float(gate_config["clearance"]),
            completion_radius=float(gate_config["completion_radius"]),
        )
        next_deadline = route_start
        step = 0
        while not route.completed:
            period_start = time.perf_counter()
            if period_start - route.gate_start_time > float(timing["per_gate_timeout_seconds"]):
                raise TimeoutError(f"gate {route.target_gate_id} timed out")
            raw = drone.get_raw_state()
            collision_name = _collision(client, vehicle_name, first_sample=(step == 0))
            if collision_name is not None:
                raise RuntimeError(f"teacher collision with {collision_name}")
            crossing = route.update(raw.position, period_start)
            if crossing is not None:
                crossings.append({
                    "gate": int(crossing.gate_index),
                    "lateral_offset": float(crossing.lateral_offset),
                    "vertical_offset": float(crossing.vertical_offset),
                })
                print(
                    f"{scenario['id']} gate {crossing.gate_index}: PASS "
                    f"lateral={crossing.lateral_offset:.3f}m vertical={crossing.vertical_offset:.3f}m",
                    flush=True,
                )
            if route.completed:
                break
            gate = route.current_gate
            policy_input = build_controller_state(raw, gate.center, gate.orientation)
            solve_started = time.perf_counter()
            control, query = solve_track_p_teacher(
                controller,
                gate_position=gate.center,
                gate_orientation=gate.orientation,
                gate_position_error=policy_input[0:3],
                gate_attitude_error=policy_input[3:6],
                velocity=policy_input[6:9],
                acceleration=policy_input[9:12],
                dt=dt,
                horizon=horizon,
                reference_speed=float(mpc_config["reference_speed"]),
            )
            solve_times.append(time.perf_counter() - solve_started)
            send_control(drone, control, calibration, duration=dt)
            x_rows.append(query.policy_input.astype(np.float32))
            u_rows.append(np.asarray(control, dtype=np.float32))
            positions.append(raw.position.astype(np.float32))
            gate_ids.append(route.target_gate_id)
            next_deadline += dt
            deadline_misses += int(time.perf_counter() > next_deadline)
            if step % 20 == 0:
                print(
                    f"{scenario['id']} step={step:04d} gate={route.target_gate_id} "
                    f"speed={np.linalg.norm(raw.velocity):.2f}m/s "
                    f"solve={solve_times[-1] * 1e3:.2f}ms",
                    flush=True,
                )
            remaining = next_deadline - time.perf_counter()
            if remaining > 0.0:
                time.sleep(remaining)
            step += 1
    except (GatePassageError, RuntimeError, TimeoutError) as exc:
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            _join(drone.hover())
            _join(drone.land())
            drone.disarm()
            drone.enable_api_control(False)
        except Exception:
            pass
        if not keep_gates:
            scene.destroy()
    x = np.asarray(x_rows, dtype=np.float32).reshape(-1, 12)
    u = np.asarray(u_rows, dtype=np.float32).reshape(-1, 4)
    npz_path = output_dir / f"common_state_{scenario['id']}.npz"
    np.savez_compressed(
        npz_path,
        X=x,
        U_teacher=u,
        position=np.asarray(positions, dtype=np.float32).reshape(-1, 3),
        gate_id=np.asarray(gate_ids, dtype=np.int16),
    )
    success = failure is None and len(crossings) == int(gate_config["count"])
    return {
        "scenario_id": str(scenario["id"]),
        "epsilon": float(scenario["epsilon"]),
        "seed": int(scenario["seed"]),
        "initial_state_id": int(initial_state.initial_state_id),
        "success": bool(success),
        "failure": failure,
        "samples": int(x.shape[0]),
        "crossings": crossings,
        "deadline_misses": int(deadline_misses),
        "mean_mpc_solve_ms": float(np.mean(solve_times) * 1e3) if solve_times else None,
        "elapsed_seconds": float(time.perf_counter() - started),
        "dataset": str(npz_path),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    scenarios = select_scenarios(config, args.scenario_set)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    initial_states = load_initial_states(resolve_project_path(config["initial_states"]))
    airsim = require_airsim()
    client = airsim.MultirotorClient()
    client.confirmConnection()
    vehicle_name = args.vehicle_name
    if not vehicle_name:
        vehicles = list(client.listVehicles())
        if not vehicles:
            raise RuntimeError("AirSim reported no vehicles")
        vehicle_name = str(vehicles[0])
        print(f"No --vehicle-name supplied; using {vehicle_name}.", flush=True)
    episode_reports: list[dict[str, Any]] = []
    model_errors: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = {
        str(item["name"]): [] for item in config["models"]
    }
    for scenario in scenarios:
        state = _find_initial_state(initial_states, int(scenario["initial_state_id"]))
        print(
            f"\n=== Common-state teacher {scenario['id']}: epsilon={scenario['epsilon']} "
            f"seed={scenario['seed']} initial={state.initial_state_id} ===",
            flush=True,
        )
        episode = _run_teacher_episode(
            client,
            vehicle_name,
            scenario,
            config,
            state,
            output_dir,
            keep_gates=args.keep_gates,
        )
        episode_reports.append(episode)
        if not episode["success"]:
            print(f"INVALID teacher episode {scenario['id']}: {episode['failure']}", flush=True)
            continue
        values = np.load(episode["dataset"])
        x = values["X"].astype(np.float32)
        u = values["U_teacher"].astype(np.float32)
        for model_item in config["models"]:
            prediction, prediction_normalized = _predict(
                model_item, x, args.device, args.batch_size
            )
            normalization = load_normalization(
                resolve_project_path(model_item["normalization"]), expected_input_dim=12
            )
            target_normalized = (u - normalization["u_mean"]) / normalization["u_std"]
            name = str(model_item["name"])
            model_errors[name].append((prediction, u, prediction_normalized, target_normalized))
            physical = regression_metrics(prediction, u)
            print(
                f"  {name}: physical_mse={physical['mse']:.9g} "
                f"rmse={physical['rmse']:.6g}",
                flush=True,
            )
    model_reports: list[dict[str, Any]] = []
    for model_item in config["models"]:
        name = str(model_item["name"])
        chunks = model_errors[name]
        if not chunks:
            continue
        prediction = np.concatenate([item[0] for item in chunks], axis=0)
        target = np.concatenate([item[1] for item in chunks], axis=0)
        prediction_normalized = np.concatenate([item[2] for item in chunks], axis=0)
        target_normalized = np.concatenate([item[3] for item in chunks], axis=0)
        model_reports.append({
            "model": name,
            "model_path": str(resolve_project_path(model_item["model"])),
            "physical": regression_metrics(prediction, target),
            "normalized": regression_metrics(prediction_normalized, target_normalized),
        })
    payload = {
        "format_version": 1,
        "benchmark": config["benchmark_name"],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scenario_set": args.scenario_set,
        "vehicle_name": vehicle_name,
        "teacher_episodes_requested": len(scenarios),
        "teacher_episodes_valid": sum(int(item["success"]) for item in episode_reports),
        "teacher_episodes": episode_reports,
        "models": model_reports,
        "important_note": "This is a new held-out common-state AirSim benchmark, not a replacement for the historical training-set or paper Table II metric.",
    }
    report_path = output_dir / "common_state_summary.json"
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (output_dir / "common_state_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("model", "samples", "physical_mse", "physical_rmse", "normalized_mse", "normalized_rmse"),
        )
        writer.writeheader()
        for item in model_reports:
            writer.writerow({
                "model": item["model"],
                "samples": item["physical"]["samples"],
                "physical_mse": item["physical"]["mse"],
                "physical_rmse": item["physical"]["rmse"],
                "normalized_mse": item["normalized"]["mse"],
                "normalized_rmse": item["normalized"]["rmse"],
            })
    print(f"\nReport: {report_path}", flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "benchmark_config.json")
    parser.add_argument("--scenario-set", default="pilot", help="pilot, full, or comma-separated scenario ids")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results" / "common_state_pilot")
    parser.add_argument("--vehicle-name", default="")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--keep-gates", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    run(args)


if __name__ == "__main__":
    main()
