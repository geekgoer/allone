"""Future reference generation kept separate from flight control."""

from __future__ import annotations

from typing import Any

import numpy as np

from mpc_controller import MPCReference
from state_adapter import normalize_quaternion, quaternion_multiply


def constant_reference(
    position: Any,
    orientation: Any = (1.0, 0.0, 0.0, 0.0),
) -> MPCReference:
    return MPCReference(np.asarray(position, dtype=np.float64), np.asarray(orientation, dtype=np.float64))


def linear_reference_window(
    start: Any,
    target: Any,
    dt: float = 0.05,
    horizon: int = 20,
    speed: float = 2.0,
    orientation: Any = (1.0, 0.0, 0.0, 0.0),
) -> list[MPCReference]:
    """Generate a finite future position window; this function sends no control."""

    if dt <= 0.0 or horizon < 1 or speed <= 0.0:
        raise ValueError("dt, horizon and speed must be positive")
    return linear_reference_window_at_time(
        start,
        target,
        elapsed=0.0,
        dt=dt,
        horizon=horizon,
        speed=speed,
        orientation=orientation,
    )


def linear_reference_window_at_time(
    start: Any,
    target: Any,
    elapsed: float,
    dt: float = 0.05,
    horizon: int = 20,
    speed: float = 2.0,
    orientation: Any = (1.0, 0.0, 0.0, 0.0),
) -> list[MPCReference]:
    """Sample a fixed linear path at ``elapsed + (1..horizon)*dt``."""

    if dt <= 0.0 or horizon < 1 or speed <= 0.0 or elapsed < 0.0:
        raise ValueError("dt, horizon, speed and elapsed must be valid")
    start = np.asarray(start, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if start.size != 3 or target.size != 3:
        raise ValueError("start and target must be 3-D")
    delta = target - start
    distance = float(np.linalg.norm(delta))
    if distance < 1e-12:
        return [constant_reference(target, orientation) for _ in range(horizon)]
    direction = delta / distance
    samples: list[MPCReference] = []
    for index in range(horizon):
        distance_at_sample = min(distance, speed * (elapsed + dt * (index + 1)))
        position = start + direction * distance_at_sample
        velocity = direction * speed if distance_at_sample < distance else np.zeros(3)
        samples.append(
            MPCReference(position, np.asarray(orientation), velocity=velocity)
        )
    return samples


def polyline_reference_at_time(
    waypoints: Any,
    elapsed: float,
    speed: float = 2.0,
    orientation: Any = (1.0, 0.0, 0.0, 0.0),
) -> MPCReference:
    """Sample one constant-speed reference at an exact elapsed time."""

    if speed <= 0.0 or elapsed < 0.0:
        raise ValueError("speed and elapsed must be valid")
    points = np.asarray(waypoints, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 2:
        raise ValueError("waypoints must have shape (N, 3) with N >= 2")
    deltas = np.diff(points, axis=0)
    lengths = np.linalg.norm(deltas, axis=1)
    keep = lengths > 1e-12
    if not np.any(keep):
        return constant_reference(points[-1], orientation)
    starts = points[:-1][keep]
    deltas = deltas[keep]
    lengths = lengths[keep]
    directions = deltas / lengths[:, None]
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    distance = speed * elapsed
    if distance >= float(cumulative[-1]):
        return constant_reference(points[-1], orientation)
    segment = min(
        int(np.searchsorted(cumulative, distance, side="right") - 1),
        len(lengths) - 1,
    )
    local_distance = distance - cumulative[segment]
    return MPCReference(
        starts[segment] + directions[segment] * local_distance,
        np.asarray(orientation),
        velocity=directions[segment] * speed,
    )


def polyline_reference_window_at_time(
    waypoints: Any,
    elapsed: float,
    dt: float = 0.05,
    horizon: int = 20,
    speed: float = 2.0,
    orientation: Any = (1.0, 0.0, 0.0, 0.0),
) -> list[MPCReference]:
    """Sample a constant-speed future window along a waypoint polyline.

    Intermediate waypoints are traversed with non-zero reference velocity;
    only the final waypoint is a stopping point.  This is suitable for a gate
    route made from each gate's pre-plane and post-plane passage points.
    """

    if dt <= 0.0 or horizon < 1 or speed <= 0.0 or elapsed < 0.0:
        raise ValueError("dt, horizon, speed and elapsed must be valid")
    return [
        polyline_reference_at_time(
            waypoints,
            elapsed=elapsed + dt * (index + 1),
            speed=speed,
            orientation=orientation,
        )
        for index in range(horizon)
    ]


def gate_reference_window(
    current_position: Any,
    gate_position: Any,
    dt: float = 0.05,
    horizon: int = 20,
    speed: float = 2.0,
) -> list[MPCReference]:
    return linear_reference_window(current_position, gate_position, dt, horizon, speed)


def _rotate_vector(quaternion: Any, vector: Any) -> np.ndarray:
    """Rotate a vector by a [w, x, y, z] quaternion."""

    q = normalize_quaternion(quaternion)
    value = np.asarray(vector, dtype=np.float64).reshape(-1)
    if value.size != 3:
        raise ValueError("vector must have shape (3,)")
    pure = np.array([0.0, value[0], value[1], value[2]], dtype=np.float64)
    inverse = np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)
    return quaternion_multiply(quaternion_multiply(q, pure), inverse)[1:]


def gate_passage_waypoints(
    gate_position: Any,
    gate_orientation: Any = (1.0, 0.0, 0.0, 0.0),
    approach_distance: float = 2.0,
    exit_distance: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a stable pre-gate point and a post-gate crossing point.

    Gate local +x is the passage direction.  With the default identity
    orientation this is AirSim world +x.  Requiring the post-gate point makes
    arrival mean that the vehicle has actually crossed the gate plane rather
    than merely getting close to its center from the front.
    """

    if approach_distance <= 0.0 or exit_distance <= 0.0:
        raise ValueError("approach_distance and exit_distance must be positive")
    center = np.asarray(gate_position, dtype=np.float64).reshape(-1)
    if center.size != 3:
        raise ValueError("gate_position must have shape (3,)")
    direction = _rotate_vector(gate_orientation, [1.0, 0.0, 0.0])
    return center - approach_distance * direction, center + exit_distance * direction


def gate_alignment_waypoint(current_position: Any, gate_position: Any) -> np.ndarray:
    """Return a staging point aligned with a gate before moving forward.

    The staging point keeps the vehicle's current x coordinate and adopts the
    gate center's lateral and vertical coordinates.  This separates a large
    lateral/height correction from the final approach, so a vehicle does not
    reach a narrow gate frame while still cutting across it diagonally.
    """

    current = np.asarray(current_position, dtype=np.float64).reshape(-1)
    center = np.asarray(gate_position, dtype=np.float64).reshape(-1)
    if current.size != 3 or center.size != 3:
        raise ValueError("current_position and gate_position must have shape (3,)")
    return np.array([current[0], center[1], center[2]], dtype=np.float64)
