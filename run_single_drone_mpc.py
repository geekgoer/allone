"""Run hover, move-to-point, and short linear-reference tracking."""

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
from reference_generator import MPCReference, linear_reference_window_at_time
from state_adapter import build_controller_state


def _join(future: object) -> None:
    join = getattr(future, "join", None)
    if callable(join):
        join()


def run(
    duration: float,
    target_offset: np.ndarray,
    vehicle_name: str,
    calibration: ThrustCalibration,
) -> None:
    airsim = require_airsim()
    client = airsim.MultirotorClient()
    client.confirmConnection()
    drone = AirSimDrone(client, vehicle_name)
    drone.enable_api_control(True)
    drone.arm()
    try:
        _join(drone.takeoff())
        initial = drone.get_raw_state()
        target = initial.position + target_offset
        controller = MPCController(dt=0.05, horizon=20)
        path_start = initial.position.copy()
        ensure_runtime_directories()
        with TeacherDataLogger(
            DATA_DIR / f"single_{vehicle_name or 'default'}.csv", vehicle_name or "default"
        ) as logger:
            start_time = time.perf_counter()
            next_deadline = start_time
            last_start = start_time
            iteration = 0
            while time.perf_counter() - start_time < duration:
                period_start = time.perf_counter()
                raw = drone.get_raw_state()
                elapsed = time.perf_counter() - start_time if iteration else 0.0
                window = linear_reference_window_at_time(
                    path_start,
                    target,
                    elapsed=elapsed,
                    dt=controller.dt,
                    horizon=controller.horizon,
                    speed=2.0,
                )
                reference: MPCReference = window[0]
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
                        f"k={iteration:04d} goal_err={np.linalg.norm(raw.position - target):.3f}m "
                        f"vel={np.linalg.norm(raw.velocity):.3f}m/s "
                        f"solve={solve_time * 1e3:.2f}ms loop={loop_period * 1e3:.2f}ms "
                        f"miss={miss} u={np.array2string(control, precision=3)}"
                    )
                iteration += 1
                last_start = period_start
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
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--dx", type=float, default=3.0)
    parser.add_argument("--dy", type=float, default=0.0)
    parser.add_argument("--dz", type=float, default=0.0)
    parser.add_argument("--vehicle-name", default="")
    parser.add_argument("--calibration", default=None)
    args = parser.parse_args()
    run(
        args.duration,
        np.array([args.dx, args.dy, args.dz], dtype=np.float64),
        args.vehicle_name,
        load_calibration(args.calibration),
    )


if __name__ == "__main__":
    main()
