"""Run three independent 20 Hz MPC teachers against three AirSim vehicles."""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone

import numpy as np

from config import DATA_DIR, ThrustCalibration, ensure_runtime_directories, load_calibration
from environment import generate_gate_sequences
from gate_visualization import (
    DEFAULT_GATE_ASSET,
    DEFAULT_GATE_HEIGHT,
    DEFAULT_GATE_THICKNESS,
    DEFAULT_GATE_WIDTH,
    spawn_gate_sequences,
)
from logger import TeacherDataLogger
from mpc_controller import MPCController, MPCReference
from multi_drone_manager import MultiDroneManager
from reference_generator import linear_reference_window_at_time
from state_adapter import build_controller_state


DEFAULT_VEHICLES = ("Drone1", "Drone2", "Drone3")


def run(
    duration: float,
    target_offset: np.ndarray,
    vehicle_names: tuple[str, ...],
    calibration: ThrustCalibration,
    speed: float = 2.0,
    epsilon: float = 0.8,
    seed: int | None = 0,
    gate_width: float = DEFAULT_GATE_WIDTH,
    gate_height: float = DEFAULT_GATE_HEIGHT,
    gate_thickness: float = DEFAULT_GATE_THICKNESS,
    gate_asset: str = DEFAULT_GATE_ASSET,
    spawn_gates: bool = True,
    keep_gates: bool = False,
    group_spacing: float = 6.0,
    gate_count: int = 5,
    x_start: float = 10.0,
    x_spacing: float = 10.0,
    base_z: float = -2.0,
    max_z: float | None = None,
) -> None:
    if duration <= 0.0 or speed <= 0.0:
        raise ValueError("duration and speed must be positive")
    target_offset = np.asarray(target_offset, dtype=np.float64).reshape(-1)
    if target_offset.size != 3:
        raise ValueError("target_offset must have shape (3,)")
    if spawn_gates and (group_spacing <= 0.0 or gate_count < 1 or x_spacing <= 0.0):
        raise ValueError("group_spacing, gate_count, and x_spacing must be positive")
    if spawn_gates and gate_width >= group_spacing:
        raise ValueError("gate_width must be smaller than group_spacing")
    ensure_runtime_directories()
    manager = MultiDroneManager(list(vehicle_names))
    gate_scene = None
    try:
        if spawn_gates:
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
            gate_scene = spawn_gate_sequences(
                manager.client,
                gates_by_drone,
                width=gate_width,
                height=gate_height,
                thickness=gate_thickness,
                asset_name=gate_asset,
            )
            print(
                f"Spawned {len(gate_scene.object_names)} visible gate bars "
                f"({len(vehicle_names) * gate_count} independent gates, asset={gate_asset})."
            )
        manager.enable_api_control_all(True)
        manager.arm_all()
        manager.wait_all(manager.takeoff_all())
        initial_states = manager.get_all_states()
        path_starts = {
            name: state.position.copy() for name, state in initial_states.items()
        }
        targets = {name: start + target_offset for name, start in path_starts.items()}
        controllers = {name: MPCController(dt=0.05, horizon=20) for name in vehicle_names}
        loggers = {
            name: TeacherDataLogger(DATA_DIR / f"{name}.csv", name)
            for name in vehicle_names
        }
        try:
            start_time = time.perf_counter()
            next_deadline = start_time
            last_start = start_time
            iteration = 0
            while time.perf_counter() - start_time < duration:
                period_start = time.perf_counter()
                states = manager.get_all_states()
                controls: dict[str, np.ndarray] = {}
                references: dict[str, MPCReference] = {}
                controller_states: dict[str, np.ndarray] = {}
                solve_times: dict[str, float] = {}
                elapsed = period_start - start_time
                for name in vehicle_names:
                    controller = controllers[name]
                    window = linear_reference_window_at_time(
                        path_starts[name],
                        targets[name],
                        elapsed=elapsed,
                        dt=controller.dt,
                        horizon=controller.horizon,
                        speed=speed,
                    )
                    reference = window[0]
                    state = build_controller_state(
                        states[name], reference.position, reference.orientation
                    )
                    solve_start = time.perf_counter()
                    control = controller.solve(state, window)
                    solve_times[name] = time.perf_counter() - solve_start
                    references[name] = reference
                    controller_states[name] = state
                    controls[name] = control

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
                        f"{name}:err={np.linalg.norm(states[name].position - targets[name]):.2f}m"
                        f",v={np.linalg.norm(states[name].velocity):.2f}m/s"
                        f",solve={solve_times[name] * 1e3:.2f}ms"
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
        finally:
            for logger in loggers.values():
                logger.close()
    finally:
        manager.wait_all(manager.hover_all())
        manager.wait_all(manager.land_all())
        manager.disarm_all()
        manager.enable_api_control_all(False)
        if gate_scene is not None:
            if keep_gates:
                print("Keeping visible gates in the UE4 world (--keep-gates).")
            else:
                gate_scene.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--dx", type=float, default=3.0)
    parser.add_argument("--dy", type=float, default=0.0)
    parser.add_argument("--dz", type=float, default=0.0)
    parser.add_argument("--speed", type=float, default=2.0)
    parser.add_argument("--epsilon", type=float, choices=[0.0, 0.8, 1.6], default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gate-width", type=float, default=DEFAULT_GATE_WIDTH)
    parser.add_argument("--gate-height", type=float, default=DEFAULT_GATE_HEIGHT)
    parser.add_argument("--gate-thickness", type=float, default=DEFAULT_GATE_THICKNESS)
    parser.add_argument("--gate-asset", default=DEFAULT_GATE_ASSET)
    parser.add_argument("--group-spacing", type=float, default=6.0)
    parser.add_argument("--gate-count", type=int, default=5)
    parser.add_argument("--x-start", type=float, default=10.0)
    parser.add_argument("--x-spacing", type=float, default=10.0)
    parser.add_argument("--base-z", type=float, default=-2.0)
    parser.add_argument("--max-z", type=float, default=None)
    parser.add_argument(
        "--no-gates",
        action="store_true",
        help="run the three-drone point-tracking test without visible gates",
    )
    parser.add_argument(
        "--keep-gates",
        action="store_true",
        help="keep generated gate bars after the flight finishes",
    )
    parser.add_argument(
        "--vehicles",
        nargs=3,
        default=list(DEFAULT_VEHICLES),
        metavar=("DRONE1", "DRONE2", "DRONE3"),
    )
    parser.add_argument("--calibration", default=None)
    args = parser.parse_args()
    run(
        args.duration,
        np.array([args.dx, args.dy, args.dz], dtype=np.float64),
        tuple(args.vehicles),
        load_calibration(args.calibration),
        speed=args.speed,
        epsilon=args.epsilon,
        seed=args.seed,
        gate_width=args.gate_width,
        gate_height=args.gate_height,
        gate_thickness=args.gate_thickness,
        gate_asset=args.gate_asset,
        spawn_gates=not args.no_gates,
        keep_gates=args.keep_gates,
        group_spacing=args.group_spacing,
        gate_count=args.gate_count,
        x_start=args.x_start,
        x_spacing=args.x_spacing,
        base_z=args.base_z,
        max_z=args.max_z,
    )


if __name__ == "__main__":
    main()
