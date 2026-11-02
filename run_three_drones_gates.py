"""Fly three independently controlled AirSim drones through five visible gates."""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone

import numpy as np

from airsim_interface import vehicle_local_to_world_offset
from config import DATA_DIR, ThrustCalibration, ensure_runtime_directories, load_calibration
from environment import generate_gate_sequences
from gate_route import ContinuousGateRoute, GatePassageError
from gate_visualization import (
    DEFAULT_GATE_ASSET,
    DEFAULT_GATE_HEIGHT,
    DEFAULT_GATE_THICKNESS,
    spawn_gate_sequences,
)
from logger import TeacherDataLogger
from mpc_controller import MPCController
from multi_drone_manager import MultiDroneManager
from state_adapter import build_controller_state


DEFAULT_VEHICLES = ("Drone1", "Drone2", "Drone3")
DEFAULT_GATE_WIDTH = 4.0


def run(
    epsilon: float,
    seed: int | None,
    threshold: float,
    per_gate_timeout: float,
    speed: float,
    group_spacing: float,
    vehicle_names: tuple[str, ...],
    calibration: ThrustCalibration,
    gate_width: float = DEFAULT_GATE_WIDTH,
    gate_height: float = DEFAULT_GATE_HEIGHT,
    gate_thickness: float = DEFAULT_GATE_THICKNESS,
    gate_asset: str = DEFAULT_GATE_ASSET,
    keep_gates: bool = False,
    gate_count: int = 5,
    x_start: float = 10.0,
    x_spacing: float = 10.0,
    base_z: float = -2.0,
    approach_distance: float = 2.0,
    exit_distance: float = 1.0,
    stabilization_speed: float = 0.25,
    gate_clearance: float = 0.25,
    max_z: float | None = None,
) -> None:
    if len(vehicle_names) != 3:
        raise ValueError("run_three_drones_gates requires exactly three vehicles")
    if threshold <= 0.0 or per_gate_timeout <= 0.0 or speed <= 0.0 or group_spacing <= 0.0:
        raise ValueError("threshold, timeout, speed and group_spacing must be positive")
    if gate_count < 1 or x_spacing <= 0.0:
        raise ValueError("gate_count must be positive and x_spacing must be positive")
    if gate_width >= group_spacing:
        raise ValueError(
            "gate_width must be smaller than group_spacing to keep independent "
            "gate groups from overlapping; try --gate-width 4 --group-spacing 6"
        )
    if approach_distance <= 0.0 or exit_distance <= 0.0:
        raise ValueError("approach_distance and exit_distance must be positive")
    if gate_clearance < 0.0:
        raise ValueError("gate_clearance must be non-negative")
    # Retained as an API/CLI compatibility parameter.  Continuous routes no
    # longer require the vehicle to settle below a stabilization speed.
    _ = stabilization_speed
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
    gate_scene = None
    loggers: dict[str, TeacherDataLogger] = {}
    try:
        # A visible demonstration is deterministic only when every vehicle
        # begins at its configured spawn origin.
        manager.client.reset()
        time.sleep(0.5)
        world_offsets = {
            name: vehicle_local_to_world_offset(manager.client, name)
            for name in vehicle_names
        }
        print(
            "Vehicle local-to-world offsets: "
            + " ".join(
                f"{name}={np.array2string(world_offsets[name], precision=3)}"
                for name in vehicle_names
            )
        )
        gate_scene = spawn_gate_sequences(
            manager.client,
            gates_by_drone,
            width=gate_width,
            height=gate_height,
            thickness=gate_thickness,
            asset_name=gate_asset,
            vehicle_world_offsets=world_offsets,
        )
        print(
            f"Spawned {len(gate_scene.object_names)} visible gate bars "
            f"({len(vehicle_names) * gate_count} independent gates, "
            f"width={gate_width:.1f}m, asset={gate_asset})."
        )
        manager.enable_api_control_all(True)
        manager.arm_all()
        manager.wait_all(manager.takeoff_all())
        controllers = {name: MPCController(dt=0.05, horizon=20) for name in vehicle_names}
        run_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        loggers = {
            name: TeacherDataLogger(DATA_DIR / f"{name}_gates_{run_tag}.csv", name)
            for name in vehicle_names
        }
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
        next_deadline = route_start
        last_start = route_start
        iteration = 0
        while not all(route.completed for route in routes.values()):
            period_start = time.perf_counter()
            states = manager.get_all_states()
            for name in vehicle_names:
                route = routes[name]
                if not route.completed and period_start - route.gate_start_time > per_gate_timeout:
                    raise TimeoutError(
                        f"{name} did not cross gate {route.target_gate_id} within "
                        f"{per_gate_timeout:.1f}s"
                    )
                try:
                    crossing = route.update(states[name].position, period_start)
                except GatePassageError as exc:
                    raise GatePassageError(f"{name}: {exc}") from exc
                if crossing is not None:
                    print(
                        f"Gate {crossing.gate_index}: {name} crossed inside opening "
                        f"lateral={crossing.lateral_offset:.3f}m "
                        f"vertical={crossing.vertical_offset:.3f}m"
                    )
            if all(route.completed for route in routes.values()):
                break

            controls: dict[str, np.ndarray] = {}
            references: dict[str, object] = {}
            controller_states: dict[str, np.ndarray] = {}
            solve_times: dict[str, float] = {}
            distances: dict[str, float] = {}
            for name in vehicle_names:
                route = routes[name]
                window = route.reference_window(
                    period_start,
                    dt=controllers[name].dt,
                    horizon=controllers[name].horizon,
                )
                reference = window[0]
                state = build_controller_state(
                    states[name], reference.position, reference.orientation
                )
                solve_start = time.perf_counter()
                controls[name] = controllers[name].solve(state, window)
                solve_times[name] = time.perf_counter() - solve_start
                references[name] = reference
                controller_states[name] = state
                distances[name] = route.distance_to_current_gate(states[name].position)
            manager.send_control_all(controls, calibration, duration=0.05)
            next_deadline += 0.05
            loop_period = period_start - last_start if iteration else 0.05
            miss = time.perf_counter() > next_deadline
            for name in vehicle_names:
                loggers[name].write_sample(
                    timestamp=datetime.now(timezone.utc).timestamp(),
                    monotonic_timestamp=period_start,
                    state=controller_states[name],
                    control=controls[name],
                    raw_state=states[name],
                    reference=references[name],
                    mpc_solve_time=solve_times[name],
                    actual_loop_period=loop_period,
                    deadline_miss=miss,
                )
            if iteration % 20 == 0:
                summary = " ".join(
                    f"{name}:g={routes[name].target_gate_id},d={distances[name]:.2f}m,"
                    f"v={np.linalg.norm(states[name].velocity):.2f}m/s"
                    for name in vehicle_names
                )
                print(
                    f"k={iteration:04d} loop={loop_period * 1e3:.2f}ms "
                    f"miss={miss} {summary}"
                )
            iteration += 1
            last_start = period_start
            sleep_time = next_deadline - time.perf_counter()
            if sleep_time > 0.0:
                time.sleep(sleep_time)
        manager.wait_all(manager.hover_all())
        print(
            f"All {gate_count} gates were crossed inside their clear openings "
            "by all three drones; hovering."
        )
    finally:
        for logger in loggers.values():
            logger.close()
        try:
            manager.wait_all(manager.hover_all())
            manager.wait_all(manager.land_all())
            manager.disarm_all()
            manager.enable_api_control_all(False)
        finally:
            if gate_scene is not None:
                if keep_gates:
                    print("Keeping visible gates in the UE4 world (--keep-gates).")
                else:
                    gate_scene.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epsilon", type=float, choices=[0.0, 0.8, 1.6], default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--per-gate-timeout", type=float, default=15.0)
    parser.add_argument("--speed", type=float, default=2.0)
    parser.add_argument("--group-spacing", type=float, default=6.0)
    parser.add_argument("--gate-count", type=int, default=5)
    parser.add_argument("--x-start", type=float, default=10.0)
    parser.add_argument("--x-spacing", type=float, default=10.0)
    parser.add_argument("--base-z", type=float, default=-2.0)
    parser.add_argument("--max-z", type=float, default=None)
    parser.add_argument("--approach-distance", type=float, default=2.0)
    parser.add_argument("--exit-distance", type=float, default=1.0)
    parser.add_argument("--stabilization-speed", type=float, default=0.25)
    parser.add_argument("--gate-clearance", type=float, default=0.25)
    parser.add_argument("--gate-width", type=float, default=DEFAULT_GATE_WIDTH)
    parser.add_argument("--gate-height", type=float, default=DEFAULT_GATE_HEIGHT)
    parser.add_argument("--gate-thickness", type=float, default=DEFAULT_GATE_THICKNESS)
    parser.add_argument("--gate-asset", default=DEFAULT_GATE_ASSET)
    parser.add_argument("--keep-gates", action="store_true")
    parser.add_argument(
        "--vehicles",
        nargs=3,
        default=list(DEFAULT_VEHICLES),
        metavar=("DRONE1", "DRONE2", "DRONE3"),
    )
    parser.add_argument("--calibration", default=None)
    args = parser.parse_args()
    run(
        args.epsilon,
        args.seed,
        args.threshold,
        args.per_gate_timeout,
        args.speed,
        args.group_spacing,
        tuple(args.vehicles),
        load_calibration(args.calibration),
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
        stabilization_speed=args.stabilization_speed,
        gate_clearance=args.gate_clearance,
        max_z=args.max_z,
    )


if __name__ == "__main__":
    main()
