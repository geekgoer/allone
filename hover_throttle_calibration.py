"""Measure AirSim hover throttle and save a vehicle-specific calibration.

The live procedure intentionally records the observed NED vertical velocity:
positive velocity means descending, negative means ascending.  A successful
bracket should therefore show lower throttle -> positive ``v_z`` and higher
throttle -> negative ``v_z`` around the selected hover point.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import time
from typing import Any, Iterable

import numpy as np

from airsim_interface import AirSimDrone, require_airsim
from config import ThrustCalibration, save_calibration


def _measure_vertical_response(
    drone: AirSimDrone,
    throttle: float,
    hold_time: float,
    sample_period: float = 0.05,
) -> dict[str, float]:
    future = drone.send_angle_rates_throttle(0.0, 0.0, 0.0, throttle, hold_time)
    values: list[float] = []
    positions: list[float] = []
    start = time.perf_counter()
    while time.perf_counter() - start < hold_time:
        raw = drone.get_raw_state()
        values.append(float(raw.velocity[2]))
        positions.append(float(raw.position[2]))
        time.sleep(min(sample_period, max(0.0, hold_time - (time.perf_counter() - start))))
    join = getattr(future, "join", None)
    if callable(join):
        join()
    if not values:
        raise RuntimeError("No AirSim samples were collected")
    # Prefer the latter half after the command has settled.
    tail = values[max(0, len(values) // 2) :]
    return {
        "mean_vz_ned": float(np.mean(tail)),
        "start_z_ned": float(positions[0]),
        "end_z_ned": float(positions[-1]),
        "drift_vz_ned": float((positions[-1] - positions[0]) / hold_time),
    }


def calibrate_hover_throttle(
    drone: AirSimDrone,
    candidates: Iterable[float],
    hold_time: float = 1.0,
) -> tuple[ThrustCalibration, list[dict[str, float]]]:
    candidates = [float(value) for value in candidates]
    if len(candidates) < 3:
        raise ValueError("At least three throttle candidates are required")
    if any(value < 0.0 or value > 1.0 for value in candidates):
        raise ValueError("Throttle candidates must lie in [0, 1]")

    observations: list[dict[str, float]] = []
    for throttle in candidates:
        # Every candidate starts from a fresh takeoff.  A low-throttle test
        # can otherwise touch the ground; a following hover command would then
        # report zero velocity and falsely look like a good hover point.
        land_future = drone.land()
        join = getattr(land_future, "join", None)
        if callable(join):
            join()
        time.sleep(0.2)
        takeoff_future = drone.takeoff()
        join = getattr(takeoff_future, "join", None)
        if callable(join):
            join()
        response = _measure_vertical_response(drone, throttle, hold_time)
        observations.append({"throttle": throttle, **response})
        hover_future = drone.hover()
        join = getattr(hover_future, "join", None)
        if callable(join):
            join()

    # Mean velocity alone can report zero after the vehicle hits the ground.
    # Use the measured altitude trend as the primary metric so that contact is
    # not mistaken for a hover equilibrium.
    best = min(observations, key=lambda row: abs(row["drift_vz_ned"]))
    ordered = sorted(observations, key=lambda row: row["throttle"])
    monotonic_sign = ordered[0]["drift_vz_ned"] > ordered[-1]["drift_vz_ned"]
    bracketed = ordered[0]["drift_vz_ned"] > 0.0 and ordered[-1]["drift_vz_ned"] < 0.0
    calibration = ThrustCalibration(
        hover_throttle=float(best["throttle"]),
        calibration_complete=bool(monotonic_sign and bracketed),
        notes=(
            "Measured NED velocity bracket; slope remains an engineering assumption."
            if monotonic_sign and bracketed
            else "No complete descent/hover/ascent bracket; repeat with a wider range."
        ),
    )
    return calibration, observations


def _connect() -> Any:
    airsim = require_airsim()
    client = airsim.MultirotorClient()
    client.confirmConnection()
    return client


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vehicle-name", default="")
    parser.add_argument("--throttle-min", type=float, default=0.35)
    parser.add_argument("--throttle-max", type=float, default=0.65)
    parser.add_argument("--steps", type=int, default=7)
    parser.add_argument("--hold-time", type=float, default=1.0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.steps < 3:
        parser.error("--steps must be at least 3")
    candidates = np.linspace(args.throttle_min, args.throttle_max, args.steps)
    client = _connect()
    drone = AirSimDrone(client, args.vehicle_name)
    drone.enable_api_control(True)
    drone.arm()
    takeoff_future = drone.takeoff()
    join = getattr(takeoff_future, "join", None)
    if callable(join):
        join()
    try:
        calibration, observations = calibrate_hover_throttle(
            drone, candidates, hold_time=args.hold_time
        )
        calibration.vehicle_name = args.vehicle_name
        output_path = save_calibration(calibration, args.output)
        print(f"Saved calibration: {output_path}")
        print(asdict(calibration))
        print("Observations:")
        for row in observations:
            direction = "descending" if row["drift_vz_ned"] > 0 else "ascending"
            print(
                f"  throttle={row['throttle']:.3f} "
                f"drift_vz={row['drift_vz_ned']:.3f} "
                f"mean_vz={row['mean_vz_ned']:.3f} ({direction})"
            )
    finally:
        land_future = drone.land()
        join = getattr(land_future, "join", None)
        if callable(join):
            join()
        drone.disarm()
        drone.enable_api_control(False)


if __name__ == "__main__":
    main()
