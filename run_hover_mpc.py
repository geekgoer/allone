"""Run the single-drone closed-loop MPC hover experiment."""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone

import numpy as np

from airsim_interface import AirSimDrone, require_airsim
from config import DATA_DIR, ThrustCalibration, ensure_runtime_directories, load_calibration
from control_adapter import send_control
from logger import TeacherDataLogger
from mpc_controller import MPCController
from reference_generator import constant_reference
from state_adapter import build_controller_state


def _join(future: object) -> None:
    join = getattr(future, "join", None)
    if callable(join):
        join()


def _connect() -> object:
    airsim = require_airsim()
    client = airsim.MultirotorClient()
    client.confirmConnection()
    return client


def run(duration: float, vehicle_name: str, target_z: float, calibration: ThrustCalibration) -> None:
    if duration <= 0.0:
        raise ValueError("duration must be positive")
    client = _connect()
    drone = AirSimDrone(client, vehicle_name)
    drone.enable_api_control(True)
    drone.arm()
    try:
        _join(drone.takeoff())
        initial = drone.get_raw_state()
        target = np.array([initial.position[0], initial.position[1], target_z], dtype=np.float64)
        reference = constant_reference(target)
        controller = MPCController(dt=0.05, horizon=20)
        ensure_runtime_directories()
        log_name = f"hover_{vehicle_name or 'default'}.csv"
        with TeacherDataLogger(DATA_DIR / log_name, vehicle_name or "default") as logger:
            start = time.perf_counter()
            next_deadline = start
            previous_start = start
            iteration = 0
            while time.perf_counter() - start < duration:
                period_start = time.perf_counter()
                raw = drone.get_raw_state()
                state = build_controller_state(raw, target, reference.orientation)
                solve_start = time.perf_counter()
                control = controller.solve(state, reference)
                solve_time = time.perf_counter() - solve_start
                send_control(drone, control, calibration, controller.dt)
                loop_period = period_start - previous_start if iteration else controller.dt
                next_deadline += controller.dt
                deadline_miss = time.perf_counter() > next_deadline
                logger.write_sample(
                    timestamp=datetime.now(timezone.utc).timestamp(),
                    monotonic_timestamp=period_start,
                    state=state,
                    control=control,
                    raw_state=raw,
                    reference=reference,
                    mpc_solve_time=solve_time,
                    actual_loop_period=loop_period,
                    deadline_miss=deadline_miss,
                )
                if iteration % 20 == 0:
                    print(
                        f"k={iteration:04d} p_err={np.linalg.norm(state[:3]):.3f}m "
                        f"v={np.linalg.norm(state[6:9]):.3f}m/s "
                        f"solve={solve_time * 1e3:.2f}ms loop={loop_period * 1e3:.2f}ms "
                        f"u={np.array2string(control, precision=3)} miss={deadline_miss}"
                    )
                previous_start = period_start
                iteration += 1
                sleep_time = next_deadline - time.perf_counter()
                if sleep_time > 0.0:
                    time.sleep(sleep_time)
    finally:
        _join(drone.hover())
        _join(drone.land())
        drone.disarm()
        drone.enable_api_control(False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--vehicle-name", default="")
    parser.add_argument("--target-z", type=float, default=-2.0)
    parser.add_argument("--calibration", default=None)
    args = parser.parse_args()
    run(args.duration, args.vehicle_name, args.target_z, load_calibration(args.calibration))


if __name__ == "__main__":
    main()
