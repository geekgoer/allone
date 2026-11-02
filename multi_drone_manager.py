"""Small coordinator for independent AirSim multirotors.

This module deliberately does not implement FALCON scheduling or priority
logic.  It only submits the same lifecycle and control operation to each
named AirSim vehicle, leaving one MPC controller per drone to the caller.
"""

from __future__ import annotations

from typing import Any, Mapping

from airsim_interface import AirSimDrone, require_airsim
from config import ThrustCalibration
from control_adapter import send_control


class MultiDroneManager:
    """Coordinate a fixed set of named AirSim multirotors."""

    def __init__(self, vehicle_names: list[str] | tuple[str, ...], client: Any = None) -> None:
        names = tuple(str(name) for name in vehicle_names)
        if not names or any(not name for name in names):
            raise ValueError("vehicle_names must contain at least one non-empty name")
        if len(set(names)) != len(names):
            raise ValueError("vehicle_names must not contain duplicates")
        if client is None:
            airsim = require_airsim()
            client = airsim.MultirotorClient()
            client.confirmConnection()
        self.client = client
        self.vehicle_names = names
        self.drones = {name: AirSimDrone(client, name) for name in names}

    def get_all_states(self) -> dict[str, Any]:
        """Return one flattened :class:`RawDroneState` per vehicle."""

        return {name: drone.get_raw_state() for name, drone in self.drones.items()}

    def enable_api_control_all(self, enabled: bool = True) -> None:
        for drone in self.drones.values():
            drone.enable_api_control(enabled)

    def arm_all(self) -> None:
        for drone in self.drones.values():
            drone.arm()

    def disarm_all(self) -> None:
        for drone in self.drones.values():
            drone.disarm()

    def takeoff_all(self) -> dict[str, Any]:
        """Submit every takeoff before waiting, so vehicles start together."""

        return {name: drone.takeoff() for name, drone in self.drones.items()}

    def hover_all(self) -> dict[str, Any]:
        return {name: drone.hover() for name, drone in self.drones.items()}

    def land_all(self) -> dict[str, Any]:
        """Submit every landing before waiting, so vehicles land together."""

        return {name: drone.land() for name, drone in self.drones.items()}

    @staticmethod
    def wait_all(futures: Mapping[str, Any]) -> None:
        for future in futures.values():
            join = getattr(future, "join", None)
            if callable(join):
                join()

    def send_control_all(
        self,
        controls: Mapping[str, Any],
        calibration: ThrustCalibration,
        duration: float = 0.05,
    ) -> dict[str, Any]:
        """Submit one independent ``[F, wx, wy, wz]`` command per vehicle."""

        missing = set(self.vehicle_names) - set(controls)
        if missing:
            raise KeyError(f"Missing controls for vehicles: {sorted(missing)}")
        return {
            name: send_control(self.drones[name], controls[name], calibration, duration)
            for name in self.vehicle_names
        }
