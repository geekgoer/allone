"""Navigate one AirSim drone through five paper-compatible gates."""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone

import numpy as np

from airsim_interface import AirSimDrone, require_airsim, vehicle_local_to_world_offset
from config import DATA_DIR, ThrustCalibration, ensure_runtime_directories, load_calibration
from control_adapter import send_control
from environment import GateEnvironment
from gate_route import ContinuousGateRoute
from gate_visualization import (
    DEFAULT_GATE_ASSET,
    DEFAULT_GATE_HEIGHT,
    DEFAULT_GATE_THICKNESS,
    DEFAULT_GATE_WIDTH,
    spawn_gate_sequence,
)
from logger import TeacherDataLogger
from mpc_controller import MPCController
from state_adapter import build_controller_state


def _join(future: object) -> None:
    join = getattr(future, "join", None)
    if callable(join):
        join()


def run(
    epsilon: float,
    seed: int | None,
    threshold: float,
    per_gate_timeout: float,
    speed: float,
    base_y: float,
    base_z: float,
    vehicle_name: str,
    calibration: ThrustCalibration,
    gate_width: float = DEFAULT_GATE_WIDTH,
    gate_height: float = DEFAULT_GATE_HEIGHT,
    gate_thickness: float = DEFAULT_GATE_THICKNESS,
    gate_asset: str = DEFAULT_GATE_ASSET,
    keep_gates: bool = False,
    gate_count: int = 5,
    x_start: float = 10.0,
    x_spacing: float = 10.0,
    approach_distance: float = 2.0,
    exit_distance: float = 1.0,
    stabilization_speed: float = 0.25,
    gate_clearance: float = 0.25,
    max_z: float | None = None,
) -> list[float]:
    if threshold <= 0.0 or per_gate_timeout <= 0.0 or speed <= 0.0:
        raise ValueError("threshold, per_gate_timeout and speed must be positive")
    if approach_distance <= 0.0 or exit_distance <= 0.0:
        raise ValueError("approach_distance and exit_distance must be positive")
    if gate_clearance < 0.0:
        raise ValueError("gate_clearance must be non-negative")
    _ = stabilization_speed
    environment = GateEnvironment(
        epsilon=epsilon,
        seed=seed,
        base_y=base_y,
        base_z=base_z,
        gate_count=gate_count,
        x_start=x_start,
        x_spacing=x_spacing,
        max_z=max_z,
    )
    gates = environment.generate()
    ensure_runtime_directories()
    airsim = require_airsim()
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.reset()
    time.sleep(0.5)
    resolved_vehicle_name = vehicle_name
    if not resolved_vehicle_name:
        available = list(client.listVehicles())
        if not available:
            raise RuntimeError("AirSim did not report any available vehicles")
        resolved_vehicle_name = str(available[0])
        print(f"No --vehicle-name supplied; using {resolved_vehicle_name}.")
    world_offset = vehicle_local_to_world_offset(client, resolved_vehicle_name)
    gate_scene = spawn_gate_sequence(
        client,
        gates,
        width=gate_width,
        height=gate_height,
        thickness=gate_thickness,
        asset_name=gate_asset,
        world_offset=world_offset,
    )
    print(
        f"Spawned {len(gate_scene.object_names)} visible gate bars "
        f"({len(gates)} gates, asset={gate_asset})."
    )
    print(
        "Vehicle local-to-world offset: "
        f"{np.array2string(world_offset, precision=3)}"
    )
    drone = AirSimDrone(client, resolved_vehicle_name)
    arrivals: list[float] = []
    try:
        drone.enable_api_control(True)
        drone.arm()
        _join(drone.takeoff())
        controller = MPCController(dt=0.05, horizon=20)
        route_start = time.perf_counter()
        initial_state = drone.get_raw_state()
        route = ContinuousGateRoute(
            gates,
            initial_state.position,
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
        run_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        with TeacherDataLogger(
            DATA_DIR / f"gates_epsilon_{epsilon}_{resolved_vehicle_name}_{run_tag}.csv",
            resolved_vehicle_name,
        ) as logger:
            next_deadline = route_start
            last_start = route_start
            iteration = 0
            while not route.completed:
                period_start = time.perf_counter()
                if period_start - route.gate_start_time > per_gate_timeout:
                    raise TimeoutError(
                        f"Gate {route.target_gate_id} was not crossed within "
                        f"{per_gate_timeout:.1f}s"
                    )
                raw = drone.get_raw_state()
                crossing = route.update(raw.position, period_start)
                if crossing is not None:
                    error = float(np.hypot(crossing.lateral_offset, crossing.vertical_offset))
                    arrivals.append(error)
                    print(
                        f"Gate {crossing.gate_index} crossed inside opening: "
                        f"lateral={crossing.lateral_offset:.3f}m "
                        f"vertical={crossing.vertical_offset:.3f}m"
                    )
                if route.completed:
                    break
                window = route.reference_window(
                    period_start, dt=controller.dt, horizon=controller.horizon
                )
                reference = window[0]
                state = build_controller_state(raw, reference.position, reference.orientation)
                solve_start = time.perf_counter()
                control = controller.solve(state, window)
                solve_time = time.perf_counter() - solve_start
                send_control(drone, control, calibration, controller.dt)
                next_deadline += controller.dt
                loop_period = period_start - last_start if iteration else controller.dt
                miss = time.perf_counter() > next_deadline
                logger.write_sample(
                    timestamp=datetime.now(timezone.utc).timestamp(),
                    monotonic_timestamp=period_start,
                    state=state,
                    control=control,
                    raw_state=raw,
                    reference=reference,
                    mpc_solve_time=solve_time,
                    actual_loop_period=loop_period,
                    deadline_miss=miss,
                )
                if iteration % 20 == 0:
                    print(
                        f"gate={route.target_gate_id} k={iteration:04d} "
                        f"distance={route.distance_to_current_gate(raw.position):.2f}m "
                        f"pos_err={np.linalg.norm(state[:3]):.2f}m "
                        f"vel={np.linalg.norm(raw.velocity):.2f}m/s "
                        f"solve={solve_time * 1e3:.2f}ms loop={loop_period * 1e3:.2f}ms "
                        f"miss={miss} u={np.array2string(control, precision=3)}"
                    )
                iteration += 1
                last_start = period_start
                sleep_time = next_deadline - time.perf_counter()
                if sleep_time > 0.0:
                    time.sleep(sleep_time)
            _join(drone.hover())
            print(f"All {gate_count} gates crossed inside their clear openings; hovering.")
    finally:
        _join(drone.hover())
        _join(drone.land())
        drone.disarm()
        drone.enable_api_control(False)
        if not keep_gates:
            gate_scene.destroy()
        else:
            print("Keeping visible gates in the UE4 world (--keep-gates).")
    return arrivals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epsilon", type=float, choices=[0.0, 0.8, 1.6], default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--per-gate-timeout", type=float, default=20.0)
    parser.add_argument("--speed", type=float, default=2.0)
    parser.add_argument("--base-y", type=float, default=0.0)
    parser.add_argument("--base-z", type=float, default=-2.0)
    parser.add_argument("--max-z", type=float, default=None)
    parser.add_argument("--gate-width", type=float, default=DEFAULT_GATE_WIDTH)
    parser.add_argument("--gate-height", type=float, default=DEFAULT_GATE_HEIGHT)
    parser.add_argument("--gate-thickness", type=float, default=DEFAULT_GATE_THICKNESS)
    parser.add_argument("--gate-asset", default=DEFAULT_GATE_ASSET)
    parser.add_argument("--gate-count", type=int, default=5)
    parser.add_argument("--x-start", type=float, default=10.0)
    parser.add_argument("--x-spacing", type=float, default=10.0)
    parser.add_argument("--approach-distance", type=float, default=2.0)
    parser.add_argument("--exit-distance", type=float, default=1.0)
    parser.add_argument("--stabilization-speed", type=float, default=0.25)
    parser.add_argument("--gate-clearance", type=float, default=0.25)
    parser.add_argument(
        "--keep-gates",
        action="store_true",
        help="keep generated gate bars after the flight finishes",
    )
    parser.add_argument("--vehicle-name", default="")
    parser.add_argument("--calibration", default=None)
    args = parser.parse_args()
    run(
        args.epsilon,
        args.seed,
        args.threshold,
        args.per_gate_timeout,
        args.speed,
        args.base_y,
        args.base_z,
        args.vehicle_name,
        load_calibration(args.calibration),
        gate_width=args.gate_width,
        gate_height=args.gate_height,
        gate_thickness=args.gate_thickness,
        gate_asset=args.gate_asset,
        keep_gates=args.keep_gates,
        gate_count=args.gate_count,
        x_start=args.x_start,
        x_spacing=args.x_spacing,
        approach_distance=args.approach_distance,
        exit_distance=args.exit_distance,
        stabilization_speed=args.stabilization_speed,
        gate_clearance=args.gate_clearance,
        max_z=args.max_z,
    )


if __name__ == "__main__":
    main()
