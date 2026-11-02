"""Continuous gate-route references and physical gate-plane validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from environment import Gate
from reference_generator import (
    constant_reference,
    gate_passage_waypoints,
    polyline_reference_at_time,
    polyline_reference_window_at_time,
)
from state_adapter import normalize_quaternion, quaternion_multiply


@dataclass(frozen=True)
class GateCrossing:
    gate_index: int
    point: np.ndarray
    lateral_offset: float
    vertical_offset: float


class GatePassageError(RuntimeError):
    """Raised when a vehicle crosses a gate plane outside the clear opening."""


def _rotate_vector(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    q = normalize_quaternion(quaternion)
    pure = np.array([0.0, vector[0], vector[1], vector[2]], dtype=np.float64)
    inverse = np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)
    return quaternion_multiply(quaternion_multiply(q, pure), inverse)[1:]


def _gate_local(point: np.ndarray, gate: Gate) -> np.ndarray:
    q = normalize_quaternion(gate.orientation)
    inverse = np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)
    return _rotate_vector(inverse, np.asarray(point, dtype=np.float64) - gate.center)


def segment_gate_crossing(
    previous_position: Iterable[float],
    current_position: Iterable[float],
    gate: Gate,
) -> GateCrossing | None:
    """Return the forward intersection with a gate plane, if one occurred."""

    previous = np.asarray(previous_position, dtype=np.float64).reshape(-1)
    current = np.asarray(current_position, dtype=np.float64).reshape(-1)
    if previous.size != 3 or current.size != 3:
        raise ValueError("gate-crossing positions must have shape (3,)")
    local_previous = _gate_local(previous, gate)
    local_current = _gate_local(current, gate)
    if not (local_previous[0] <= 0.0 < local_current[0]):
        return None
    denominator = local_current[0] - local_previous[0]
    if denominator <= 1e-12:
        return None
    fraction = float(-local_previous[0] / denominator)
    point = previous + fraction * (current - previous)
    local_point = _gate_local(point, gate)
    return GateCrossing(
        gate_index=gate.index,
        point=point,
        lateral_offset=float(local_point[1]),
        vertical_offset=float(local_point[2]),
    )


class ContinuousGateRoute:
    """Track one non-stopping polyline through a sequence of gate openings."""

    def __init__(
        self,
        gates: Iterable[Gate],
        start_position: Iterable[float],
        *,
        approach_distance: float,
        exit_distance: float,
        reference_speed: float,
        start_time: float,
        gate_width: float = 4.0,
        gate_height: float = 4.0,
        gate_thickness: float = 0.15,
        clearance: float = 0.25,
        completion_radius: float = 0.8,
    ) -> None:
        self.gates = tuple(gates)
        if not self.gates:
            raise ValueError("a route must contain at least one gate")
        if min(approach_distance, exit_distance, reference_speed, completion_radius) <= 0.0:
            raise ValueError("route distances and reference speed must be positive")
        if min(gate_width, gate_height, gate_thickness) <= 0.0 or clearance < 0.0:
            raise ValueError("gate dimensions must be positive and clearance non-negative")
        self.reference_speed = float(reference_speed)
        self.exit_distance = float(exit_distance)
        self.completion_radius = float(completion_radius)
        self.start_time = float(start_time)
        self.gate_start_time = float(start_time)
        self.index = 0
        self.completed = False
        self.phase = "route"
        self.previous_position = np.asarray(start_position, dtype=np.float64).reshape(-1).copy()
        if self.previous_position.size != 3:
            raise ValueError("start_position must have shape (3,)")
        self.clear_half_width = gate_width / 2.0 - gate_thickness / 2.0 - clearance
        self.clear_half_height = gate_height / 2.0 - gate_thickness / 2.0 - clearance
        if self.clear_half_width <= 0.0 or self.clear_half_height <= 0.0:
            raise ValueError("gate opening is eliminated by thickness and clearance")
        self.approach_points: list[np.ndarray] = []
        self.exit_points: list[np.ndarray] = []
        route_points = [self.previous_position.copy()]
        for gate in self.gates:
            before, after = gate_passage_waypoints(
                gate.center,
                gate.orientation,
                approach_distance=approach_distance,
                exit_distance=exit_distance,
            )
            self.approach_points.append(before)
            self.exit_points.append(after)
            route_points.extend((before, after))
        self.waypoints = np.asarray(route_points, dtype=np.float64)
        self.crossings: list[GateCrossing] = []
        self._last_gate_crossed = False

    @property
    def target_gate_id(self) -> int:
        return min(self.index + 1, len(self.gates))

    @property
    def current_gate(self) -> Gate:
        return self.gates[min(self.index, len(self.gates) - 1)]

    def distance_to_current_gate(self, position: Iterable[float]) -> float:
        return float(
            np.linalg.norm(np.asarray(position, dtype=np.float64) - self.current_gate.center)
        )

    def update(self, position: Iterable[float], now: float) -> GateCrossing | None:
        current = np.asarray(position, dtype=np.float64).reshape(-1)
        if current.size != 3:
            raise ValueError("position must have shape (3,)")
        if self.completed:
            self.previous_position = current.copy()
            return None
        if self._last_gate_crossed:
            local = _gate_local(current, self.gates[-1])
            self.previous_position = current.copy()
            if local[0] >= self.exit_distance - self.completion_radius:
                self.completed = True
                self.phase = "done"
            return None
        crossing = segment_gate_crossing(self.previous_position, current, self.current_gate)
        self.previous_position = current.copy()
        if crossing is None:
            return None
        if (
            abs(crossing.lateral_offset) > self.clear_half_width
            or abs(crossing.vertical_offset) > self.clear_half_height
        ):
            raise GatePassageError(
                f"Gate {crossing.gate_index} plane was crossed outside its clear opening: "
                f"lateral={crossing.lateral_offset:.3f}m "
                f"(limit={self.clear_half_width:.3f}m), "
                f"vertical={crossing.vertical_offset:.3f}m "
                f"(limit={self.clear_half_height:.3f}m)"
            )
        self.crossings.append(crossing)
        if self.index + 1 >= len(self.gates):
            self._last_gate_crossed = True
            self.phase = "exit"
        else:
            self.index += 1
            self.gate_start_time = float(now)
        return crossing

    def reference_window(self, now: float, *, dt: float, horizon: int) -> list[object]:
        if self.completed:
            orientation = self.gates[-1].orientation
            return [
                constant_reference(self.exit_points[-1], orientation)
                for _ in range(horizon)
            ]
        return polyline_reference_window_at_time(
            self.waypoints,
            elapsed=max(0.0, float(now - self.start_time)),
            dt=dt,
            horizon=horizon,
            speed=self.reference_speed,
            orientation=self.current_gate.orientation,
        )

    def reference_at_time(self, now: float) -> object:
        """Return the route reference aligned with the current sample time."""

        if self.completed:
            return constant_reference(self.exit_points[-1], self.gates[-1].orientation)
        return polyline_reference_at_time(
            self.waypoints,
            elapsed=max(0.0, float(now - self.start_time)),
            speed=self.reference_speed,
            orientation=self.current_gate.orientation,
        )
