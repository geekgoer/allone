"""Convert the controller's fixed four-value output into AirSim commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from airsim_interface import AirSimDrone
from config import ThrustCalibration, load_calibration


@dataclass(frozen=True)
class AirSimCommand:
    """AirSim command after conversion from ``[F, wx, wy, wz]``."""

    thrust_accel: float
    # These are the rate values as passed to AirSim.  AirSim's NED aircraft
    # command uses the opposite sign for its pitch/yaw rate arguments relative
    # to the right-handed quaternion rotation-vector convention used by the
    # controller.  The conversion is kept here, rather than hidden in MPC.
    roll_rate: float
    pitch_rate: float
    yaw_rate: float
    throttle: float
    duration: float


def validate_control(control: Any) -> np.ndarray:
    value = np.asarray(control, dtype=np.float64).reshape(-1)
    if value.size != 4:
        raise ValueError("MPC control must have shape (4,): [F, wx, wy, wz]")
    if not np.all(np.isfinite(value)):
        raise ValueError("MPC control contains a non-finite value")
    return value


def thrust_to_throttle(
    thrust_accel: float,
    calibration: ThrustCalibration | None = None,
) -> float:
    """Map physical collective thrust acceleration to AirSim's [0, 1] input.

    The map is anchored at the measured hover throttle.  It is deliberately
    configurable because AirSim's normalized throttle is not a physical force
    value and the mapping is vehicle/physics-model dependent.
    """

    calibration = calibration or load_calibration()
    calibration.validate()
    throttle = calibration.hover_throttle + (
        float(thrust_accel) - calibration.hover_thrust_accel
    ) / calibration.thrust_accel_per_throttle
    return float(np.clip(throttle, calibration.min_throttle, calibration.max_throttle))


def throttle_to_thrust(
    throttle: float,
    calibration: ThrustCalibration | None = None,
) -> float:
    calibration = calibration or load_calibration()
    calibration.validate()
    value = float(np.clip(throttle, calibration.min_throttle, calibration.max_throttle))
    return calibration.hover_thrust_accel + (
        value - calibration.hover_throttle
    ) * calibration.thrust_accel_per_throttle


def control_to_airsim_command(
    control: Any,
    calibration: ThrustCalibration | None = None,
    duration: float = 0.05,
) -> AirSimCommand:
    value = validate_control(control)
    if duration <= 0.0:
        raise ValueError("duration must be positive")
    return AirSimCommand(
        thrust_accel=float(value[0]),
        roll_rate=float(value[1]),
        pitch_rate=float(-value[2]),
        yaw_rate=float(-value[3]),
        throttle=thrust_to_throttle(value[0], calibration),
        duration=float(duration),
    )


def send_control(
    drone: AirSimDrone,
    control: Any,
    calibration: ThrustCalibration | None = None,
    duration: float = 0.05,
) -> Any:
    """Send one converted command without waiting for the duration to elapse."""

    command = control_to_airsim_command(control, calibration, duration)
    return drone.send_angle_rates_throttle(
        command.roll_rate,
        command.pitch_rate,
        command.yaw_rate,
        command.throttle,
        command.duration,
    )
