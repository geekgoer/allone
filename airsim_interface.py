"""Small adapter around the AirSim Python API.

Controller code should depend on :class:`RawDroneState` and simple methods in
:class:`AirSimDrone`, rather than on AirSim's nested RPC objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:  # Keep offline state-adapter tests usable without AirSim installed.
    import airsim  # type: ignore
except ImportError:  # pragma: no cover - exercised only on non-AirSim hosts.
    airsim = None  # type: ignore


class AirSimDependencyError(RuntimeError):
    """Raised when a live-AirSim operation is attempted without the client."""


def require_airsim() -> Any:
    if airsim is None:
        raise AirSimDependencyError(
            "AirSim is not importable. Activate F:\\Airsim\\AirSim\\PythonClient\\.venv."
        )
    return airsim


def _vector3_to_array(value: Any) -> np.ndarray:
    return np.array([value.x_val, value.y_val, value.z_val], dtype=np.float64)


def _quaternion_to_array(value: Any) -> np.ndarray:
    return np.array([value.w_val, value.x_val, value.y_val, value.z_val], dtype=np.float64)


def _pose_position_to_array(pose: Any) -> np.ndarray:
    return _vector3_to_array(pose.position)


def vehicle_local_to_world_offset(client: Any, vehicle_name: str) -> np.ndarray:
    """Return the translation from one vehicle's local NED frame to UE world NED.

    AirSim vehicle-state and vehicle-pose APIs report coordinates relative to
    each vehicle's configured spawn origin.  Scene objects spawned with
    ``simSpawnObject`` use the level/world NED origin.  Comparing the vehicle
    actor pose with its local pose provides the translation needed to place a
    visible object at the same physical location as a local controller target.
    """

    if not vehicle_name:
        return np.zeros(3, dtype=np.float64)
    local_pose = client.simGetVehiclePose(vehicle_name=vehicle_name)
    world_pose = client.simGetObjectPose(vehicle_name)
    offset = _pose_position_to_array(world_pose) - _pose_position_to_array(local_pose)
    if not np.all(np.isfinite(offset)):
        raise RuntimeError(
            f"Could not resolve a finite local-to-world offset for vehicle {vehicle_name!r}"
        )
    return offset


@dataclass(frozen=True)
class RawDroneState:
    """Flat, numeric state returned by the AirSim state adapter.

    Positions, velocities and accelerations are in the vehicle's local AirSim
    NED frame.  With multiple vehicles, each frame is translated by the X/Y/Z
    spawn offset configured for that vehicle.  Angular velocity is in the
    vehicle body frame, matching AirSim's API.
    """

    position: np.ndarray
    quaternion: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    angular_velocity: np.ndarray
    timestamp_ns: int | None = None

    def __post_init__(self) -> None:
        for name, size in (
            ("position", 3),
            ("quaternion", 4),
            ("velocity", 3),
            ("acceleration", 3),
            ("angular_velocity", 3),
        ):
            value = np.asarray(getattr(self, name), dtype=np.float64).reshape(-1)
            if value.size != size:
                raise ValueError(f"{name} must have {size} elements")
            object.__setattr__(self, name, value.copy())


class AirSimDrone:
    """Typed façade for one AirSim multirotor vehicle."""

    def __init__(self, client: Any, vehicle_name: str = "") -> None:
        if client is None:
            raise ValueError("client must be an AirSim MultirotorClient instance")
        self.client = client
        self.vehicle_name = vehicle_name

    def get_raw_state(self) -> RawDroneState:
        state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        kin = state.kinematics_estimated
        timestamp_ns = getattr(state, "timestamp", None)
        return RawDroneState(
            position=_vector3_to_array(kin.position),
            quaternion=_quaternion_to_array(kin.orientation),
            velocity=_vector3_to_array(kin.linear_velocity),
            acceleration=_vector3_to_array(kin.linear_acceleration),
            angular_velocity=_vector3_to_array(kin.angular_velocity),
            timestamp_ns=timestamp_ns,
        )

    def enable_api_control(self, enabled: bool = True) -> None:
        self.client.enableApiControl(enabled, vehicle_name=self.vehicle_name)

    def arm(self) -> None:
        self.client.armDisarm(True, vehicle_name=self.vehicle_name)

    def disarm(self) -> None:
        self.client.armDisarm(False, vehicle_name=self.vehicle_name)

    def takeoff(self) -> Any:
        return self.client.takeoffAsync(vehicle_name=self.vehicle_name)

    def hover(self) -> Any:
        return self.client.hoverAsync(vehicle_name=self.vehicle_name)

    def land(self) -> Any:
        return self.client.landAsync(vehicle_name=self.vehicle_name)

    def send_angle_rates_throttle(
        self,
        roll_rate: float,
        pitch_rate: float,
        yaw_rate: float,
        throttle: float,
        duration: float,
    ) -> Any:
        """Send AirSim's body-rate plus normalized-throttle command."""

        return self.client.moveByAngleRatesThrottleAsync(
            float(roll_rate),
            float(pitch_rate),
            float(yaw_rate),
            float(throttle),
            float(duration),
            vehicle_name=self.vehicle_name,
        )
