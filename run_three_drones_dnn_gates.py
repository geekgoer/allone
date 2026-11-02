"""Fly three AirSim drones through independent gates with trained DNN policies."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from airsim_interface import vehicle_local_to_world_offset
from config import ThrustCalibration, ensure_runtime_directories, load_calibration
from dnn_controller import DNNController
from dnn_model_utils import build_policy_input
from environment import generate_gate_sequences
from fpga_shadow_backend import BufferedJsonlLogger, FalconFpgaControlBackend, load_fpga_config
from fpga_udp_client import FalconFpgaUdpClient
from gate_route import ContinuousGateRoute, GatePassageError
from gate_visualization import DEFAULT_GATE_ASSET, DEFAULT_GATE_HEIGHT, DEFAULT_GATE_THICKNESS, spawn_gate_sequences
from multi_drone_manager import MultiDroneManager
from state_adapter import build_controller_state


DEFAULT_VEHICLES = ("Drone1", "Drone2", "Drone3")
DEFAULT_GATE_WIDTH = 4.0


def _write_report(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _check_vehicle(client: object, name: str, raw: object, *, first_sample: bool) -> bool:
    values = (raw.position, raw.quaternion, raw.velocity, raw.acceleration, raw.angular_velocity)
    if not all(np.isfinite(value).all() for value in values):
        raise RuntimeError(f"AirSim returned NaN or Inf for {name}")
    collision = client.simGetCollisionInfo(vehicle_name=name)
    if bool(getattr(collision, "has_collided", False)):
        object_name = str(getattr(collision, "object_name", "unknown"))
        if first_sample:
            print(f"Ignoring initial collision flag for {name}: object={object_name}", flush=True)
            return True
        raise RuntimeError(f"{name} collision with {object_name}")
    return False


def run(
    *,
    model_path: Path,
    normalization_path: Path | None,
    device: str,
    epsilon: float,
    seed: int,
    threshold: float,
    per_gate_timeout: float,
    speed: float,
    group_spacing: float,
    vehicle_names: tuple[str, ...],
    calibration: ThrustCalibration,
    gate_width: float,
    gate_height: float,
    gate_thickness: float,
    gate_asset: str,
    keep_gates: bool,
    gate_count: int,
    x_start: float,
    x_spacing: float,
    base_z: float,
    approach_distance: float,
    exit_distance: float,
    gate_clearance: float,
    max_z: float | None,
    report_path: Path,
    input_target: str = "reference",
    controller_backend: str = "cpu_dnn",
    fpga_config_path: Path = Path("fpga_closed_loop_config.json"),
    fpga_log_path: Path | None = None,
) -> dict[str, object]:
    if len(vehicle_names) != 3:
        raise ValueError("exactly three vehicle names are required")
    if input_target not in ("reference", "gate"):
        raise ValueError("input_target must be 'reference' or 'gate'")
    if min(threshold, per_gate_timeout, speed, group_spacing) <= 0.0:
        raise ValueError("threshold, timeout, speed and group spacing must be positive")
    if gate_count < 1 or x_spacing <= 0.0:
        raise ValueError("gate count and x spacing must be positive")
    if gate_width >= group_spacing:
        raise ValueError("gate width must be smaller than group spacing")
    ensure_runtime_directories()
    gates_by_drone = generate_gate_sequences(
        vehicle_names,
        epsilon=epsilon,
        seed=seed,
        group_spacing=group_spacing,
        base_z=base_z,
        gate_count=gate_count,
        x_start=x_start,
        x_spacing=x_spacing,
        max_z=max_z,
    )
    manager = MultiDroneManager(list(vehicle_names))
    fpga_config = load_fpga_config(fpga_config_path) if controller_backend != "cpu_dnn" else {}
    fpga_log_path = fpga_log_path or report_path.with_suffix(".jsonl")
    fpga_logger = BufferedJsonlLogger(fpga_log_path)
    shared_fpga_client: FalconFpgaUdpClient | None = None
    backends: dict[str, FalconFpgaControlBackend] = {}
    scene = None
    started = time.perf_counter()
    crossings: dict[str, list[dict[str, float]]] = {name: [] for name in vehicle_names}
    inference_times: dict[str, list[float]] = {name: [] for name in vehicle_names}
    initial_collision_flags = {name: False for name in vehicle_names}
    deadline_misses = 0
    samples = 0
    try:
        manager.client.reset()
        time.sleep(0.5)
        world_offsets = {
            name: vehicle_local_to_world_offset(manager.client, name)
            for name in vehicle_names
        }
        print(
            "Vehicle local-to-world offsets: "
            + " ".join(f"{name}={np.array2string(world_offsets[name], precision=3)}" for name in vehicle_names)
        )
        scene = spawn_gate_sequences(
            manager.client,
            gates_by_drone,
            width=gate_width,
            height=gate_height,
            thickness=gate_thickness,
            asset_name=gate_asset,
            vehicle_world_offsets=world_offsets,
        )
        print(
            f"Spawned {len(scene.object_names)} visible gate bars "
            f"({len(vehicle_names) * gate_count} independent gates, width={gate_width:.1f}m, asset={gate_asset})."
        )
        controllers = {
            name: DNNController(model_path, normalization_path, device=device)
            for name in vehicle_names
        }
        input_dims = {controller.input_dim for controller in controllers.values()}
        if len(input_dims) != 1:
            raise RuntimeError("DNN instances have inconsistent input dimensions")
        input_dim = input_dims.pop()
        if input_target == "gate" and input_dim != 12:
            raise ValueError("gate-target input is defined only for the Track-P 12-D model")
        print(f"Loaded 3 independent DNN instances: input_dim={input_dim}, device={device}")
        if controller_backend != "cpu_dnn":
            # FLCN/v1 has no vehicle ID.  One shared socket guarantees a
            # single global sequence space across the three serial requests.
            shared_fpga_client = FalconFpgaUdpClient(
                host=str(fpga_config["host"]),
                port=int(fpga_config["port"]),
                timeout_ms=float(fpga_config["udp_timeout_ms"]),
                retries=int(fpga_config["retries"]),
            )
        backends = {
            name: FalconFpgaControlBackend(
                controllers[name], controller_backend, fpga_config, client=shared_fpga_client
            )
            for name in vehicle_names
        }
        manager.enable_api_control_all(True)
        manager.arm_all()
        manager.wait_all(manager.takeoff_all())
        time.sleep(0.1)
        route_start = time.perf_counter()
        initial_states = manager.get_all_states()
        routes = {
            name: ContinuousGateRoute(
                gates_by_drone[name],
                initial_states[name].position,
                approach_distance=approach_distance,
                exit_distance=exit_distance,
                reference_speed=speed,
                start_time=route_start,
                gate_width=gate_width,
                gate_height=gate_height,
                gate_thickness=gate_thickness,
                clearance=gate_clearance,
                completion_radius=threshold,
            )
            for name in vehicle_names
        }
        if controller_backend != "cpu_dnn":
            print(
                f"Warming CPU DNN and shared FPGA path with "
                f"{int(fpga_config['warmup_count'])} requests per vehicle before deadline accounting...",
                flush=True,
            )
            for name in vehicle_names:
                window = routes[name].reference_window(route_start, dt=0.05, horizon=20)
                reference = window[0]
                state_target = routes[name].current_gate if input_target == "gate" else reference
                state = build_controller_state(
                    initial_states[name],
                    state_target.center if input_target == "gate" else state_target.position,
                    state_target.orientation,
                )
                policy_input = build_policy_input(
                    state,
                    window,
                    input_dim=backends[name].input_dim,
                    reference_speed=speed,
                )
                backends[name].warmup(policy_input)
            route_start = time.perf_counter()
            routes = {
                name: ContinuousGateRoute(
                    gates_by_drone[name],
                    initial_states[name].position,
                    approach_distance=approach_distance,
                    exit_distance=exit_distance,
                    reference_speed=speed,
                    start_time=route_start,
                    gate_width=gate_width,
                    gate_height=gate_height,
                    gate_thickness=gate_thickness,
                    clearance=gate_clearance,
                    completion_radius=threshold,
                )
                for name in vehicle_names
            }
        next_deadline = route_start
        last_start = route_start
        iteration = 0
        while not all(route.completed for route in routes.values()):
            period_start = time.perf_counter()
            states = manager.get_all_states()
            crossing_events: dict[str, object | None] = {name: None for name in vehicle_names}
            for name in vehicle_names:
                initial_collision_flags[name] = _check_vehicle(
                    manager.client, name, states[name], first_sample=(samples == 0)
                ) or initial_collision_flags[name]
                route = routes[name]
                if not route.completed and period_start - route.gate_start_time > per_gate_timeout:
                    raise TimeoutError(
                        f"{name} did not cross gate {route.target_gate_id} within {per_gate_timeout:.1f}s"
                    )
                try:
                    crossing = route.update(states[name].position, period_start)
                except GatePassageError as exc:
                    raise GatePassageError(f"{name}: {exc}") from exc
                if crossing is not None:
                    crossing_events[name] = crossing
                    crossings[name].append({
                        "gate": float(crossing.gate_index),
                        "lateral_offset": float(crossing.lateral_offset),
                        "vertical_offset": float(crossing.vertical_offset),
                    })
                    print(
                        f"Gate {crossing.gate_index}: {name} crossed inside opening "
                        f"lateral={crossing.lateral_offset:.3f}m vertical={crossing.vertical_offset:.3f}m"
                    )
            if all(route.completed for route in routes.values()):
                break
            controls: dict[str, np.ndarray] = {}
            distances: dict[str, float] = {}
            policy_inputs: dict[str, np.ndarray] = {}
            cycle_results = {}
            for name in vehicle_names:
                route = routes[name]
                window = route.reference_window(period_start, dt=0.05, horizon=20)
                reference = window[0]
                state_target = route.current_gate if input_target == "gate" else reference
                state = build_controller_state(
                    states[name],
                    state_target.center if input_target == "gate" else state_target.position,
                    state_target.orientation,
                )
                policy_input = build_policy_input(
                    state,
                    window,
                    input_dim=controllers[name].input_dim,
                    reference_speed=speed,
                )
                policy_inputs[name] = policy_input
                inference_start = time.perf_counter()
                cycle_results[name] = backends[name].infer(
                    policy_input, vehicle_name=name, cycle_index=iteration
                )
                controls[name] = cycle_results[name].applied_control
                inference_times[name].append(time.perf_counter() - inference_start)
                distances[name] = route.distance_to_current_gate(states[name].position)
            manager.send_control_all(controls, calibration, duration=0.05)
            next_deadline += 0.05
            loop_period = 0.05 if iteration == 0 else period_start - last_start
            miss = time.perf_counter() > next_deadline
            deadline_misses += int(miss)
            for name in vehicle_names:
                crossing = crossing_events[name]
                result = cycle_results[name]
                fpga_logger.append({
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "vehicle_name": name,
                    "cycle_index": iteration,
                    "controller_backend": controller_backend,
                    "gate_id": routes[name].target_gate_id,
                    "phase": routes[name].phase,
                    "crossing": None if crossing is None else {
                        "gate": crossing.gate_index,
                        "lateral_offset": crossing.lateral_offset,
                        "vertical_offset": crossing.vertical_offset,
                    },
                    "x_physical": policy_inputs[name].tolist(),
                    "x_normalized": result.x_normalized.tolist(),
                    **result.to_log_dict(),
                    "inference_total_us": int(inference_times[name][-1] * 1e6),
                    "loop_period_us": int(loop_period * 1e6),
                    "deadline_miss": bool(miss),
                    "collision": False,
                    "timeout": bool(result.fpga is not None and not result.fpga.fresh),
                })
            if iteration % 20 == 0:
                summary = " ".join(
                    f"{name}:g={routes[name].target_gate_id},d={distances[name]:.2f}m,"
                    f"v={np.linalg.norm(states[name].velocity):.2f}m/s,"
                    f"dnn={inference_times[name][-1] * 1e3:.2f}ms"
                    for name in vehicle_names
                )
                print(f"k={iteration:04d} loop={loop_period * 1e3:.2f}ms miss={miss} {summary}")
            samples += 1
            iteration += 1
            last_start = period_start
            sleep_time = next_deadline - time.perf_counter()
            if sleep_time > 0.0:
                time.sleep(sleep_time)
        manager.wait_all(manager.hover_all())
        report: dict[str, object] = {
            "controller": controller_backend,
            "input_target": input_target,
            "model": str(model_path.resolve()),
            "normalization": str((normalization_path or model_path.parent / "normalization.json").resolve()),
            "vehicles": list(vehicle_names),
            "epsilon": epsilon,
            "seed": seed,
            "speed": speed,
            "gate_count_per_vehicle": gate_count,
            "completed": True,
            "samples": samples,
            "deadline_misses": deadline_misses,
            "initial_collision_flags": initial_collision_flags,
            "crossings": crossings,
            "inference_ms": {
                name: {
                    "mean": float(np.mean(inference_times[name])) * 1e3,
                    "max": float(np.max(inference_times[name])) * 1e3,
                }
                for name in vehicle_names
            },
            "fpga_jsonl": str(fpga_log_path.resolve()),
            "backends": {name: backends[name].summary() for name in vehicle_names},
            "fpga_shared_client": None if shared_fpga_client is None else dict(shared_fpga_client.stats),
            "elapsed_seconds": time.perf_counter() - started,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_report(report_path, report)
        print(f"All {gate_count} gates crossed by all three DNN-controlled drones; hovering.")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"Report: {report_path.resolve()}")
        return report
    finally:
        try:
            manager.wait_all(manager.hover_all())
            manager.wait_all(manager.land_all())
            manager.disarm_all()
            manager.enable_api_control_all(False)
        finally:
            fpga_logger.flush()
            for backend in backends.values():
                backend.close()
            if shared_fpga_client is not None:
                shared_fpga_client.close()
            if scene is not None:
                if keep_gates:
                    print("Keeping visible gates in the UE4 world (--keep-gates).")
                else:
                    scene.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--input-target",
        choices=("reference", "gate"),
        default="reference",
        help="use gate for Track-P [p_gate-p,...] 12-D models; reference preserves legacy/25-D behavior",
    )
    parser.add_argument(
        "--controller-backend",
        choices=("cpu_dnn", "shadow_fpga", "fpga"),
        default="cpu_dnn",
        help="shadow_fpga logs FPGA but applies CPU DNN; fpga explicitly enables FPGA control",
    )
    parser.add_argument("--fpga-config", type=Path, default=Path("fpga_closed_loop_config.json"))
    parser.add_argument("--fpga-log", type=Path, default=None)
    parser.add_argument("--epsilon", type=float, choices=(0.0, 0.8, 1.6), default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--per-gate-timeout", type=float, default=30.0)
    parser.add_argument("--speed", type=float, default=2.0)
    parser.add_argument("--group-spacing", type=float, default=6.0)
    parser.add_argument("--gate-count", type=int, default=5)
    parser.add_argument("--x-start", type=float, default=10.0)
    parser.add_argument("--x-spacing", type=float, default=10.0)
    parser.add_argument("--base-z", type=float, default=-2.0)
    parser.add_argument("--max-z", type=float, default=None)
    parser.add_argument("--approach-distance", type=float, default=2.0)
    parser.add_argument("--exit-distance", type=float, default=1.0)
    parser.add_argument("--gate-clearance", type=float, default=0.25)
    parser.add_argument("--gate-width", type=float, default=DEFAULT_GATE_WIDTH)
    parser.add_argument("--gate-height", type=float, default=DEFAULT_GATE_HEIGHT)
    parser.add_argument("--gate-thickness", type=float, default=DEFAULT_GATE_THICKNESS)
    parser.add_argument("--gate-asset", default=DEFAULT_GATE_ASSET)
    parser.add_argument("--keep-gates", action="store_true")
    parser.add_argument("--vehicles", nargs=3, default=list(DEFAULT_VEHICLES))
    parser.add_argument("--calibration", default=None)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("dnn_runs/iid_path_v1_depth10_width512/three_drone_dnn.json"),
    )
    args = parser.parse_args()
    run(
        model_path=args.model,
        normalization_path=args.normalization,
        device=args.device,
        input_target=args.input_target,
        epsilon=args.epsilon,
        seed=args.seed,
        threshold=args.threshold,
        per_gate_timeout=args.per_gate_timeout,
        speed=args.speed,
        group_spacing=args.group_spacing,
        vehicle_names=tuple(args.vehicles),
        calibration=load_calibration(args.calibration),
        gate_width=args.gate_width,
        gate_height=args.gate_height,
        gate_thickness=args.gate_thickness,
        gate_asset=args.gate_asset,
        keep_gates=args.keep_gates,
        gate_count=args.gate_count,
        x_start=args.x_start,
        x_spacing=args.x_spacing,
        base_z=args.base_z,
        approach_distance=args.approach_distance,
        exit_distance=args.exit_distance,
        gate_clearance=args.gate_clearance,
        max_z=args.max_z,
        report_path=args.report,
        controller_backend=args.controller_backend,
        fpga_config_path=args.fpga_config,
        fpga_log_path=args.fpga_log,
    )


if __name__ == "__main__":
    main()
