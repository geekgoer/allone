"""Convert AirSim state into the controller's explicit 12-D interface.

The controller frame is intentionally NED, the same world frame used by
AirSim and by the gravity convention in the cited quadrotor model:
``x=North, y=East, z=Down``.  Therefore the conversion functions below are
explicit identity mappings for the default frame; no sign change is hidden in
the controller.  A Z-up conversion is provided for any future controller that
chooses that convention.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from airsim_interface import RawDroneState


CONTROLLER_FRAME: Literal["ned"] = "ned"


def _vector3(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray) or isinstance(value, (list, tuple)):
        result = np.asarray(value, dtype=np.float64).reshape(-1)
    else:
        result = np.array([value.x_val, value.y_val, value.z_val], dtype=np.float64)
    if result.size != 3:
        raise ValueError("Expected a 3-D vector")
    return result


def _quaternion(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray) or isinstance(value, (list, tuple)):
        result = np.asarray(value, dtype=np.float64).reshape(-1)
    else:
        result = np.array([value.w_val, value.x_val, value.y_val, value.z_val], dtype=np.float64)
    if result.size != 4:
        raise ValueError("Expected a quaternion in [w, x, y, z] order")
    return result


def airsim_to_controller_frame(vector: Any, frame: str = CONTROLLER_FRAME) -> np.ndarray:
    """Convert a world vector from AirSim NED to the selected controller frame."""

    value = _vector3(vector)
    if frame == "ned":
        return value.copy()
    if frame == "z_up":
        return np.array([value[0], value[1], -value[2]], dtype=np.float64)
    raise ValueError(f"Unsupported controller frame: {frame}")


def controller_to_airsim_frame(vector: Any, frame: str = CONTROLLER_FRAME) -> np.ndarray:
    """Convert a vector from the selected controller frame to AirSim NED."""

    value = _vector3(vector)
    if frame == "ned":
        return value.copy()
    if frame == "z_up":
        return np.array([value[0], value[1], -value[2]], dtype=np.float64)
    raise ValueError(f"Unsupported controller frame: {frame}")


def normalize_quaternion(quaternion: Any) -> np.ndarray:
    value = _quaternion(quaternion)
    norm = np.linalg.norm(value)
    if norm < 1e-12:
        raise ValueError("Quaternion norm is too small")
    return value / norm


def quaternion_conjugate(quaternion: Any) -> np.ndarray:
    q = _quaternion(quaternion)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quaternion_multiply(left: Any, right: Any) -> np.ndarray:
    """Hamilton product for quaternions stored as [w, x, y, z]."""

    w1, x1, y1, z1 = _quaternion(left)
    w2, x2, y2, z2 = _quaternion(right)
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def quaternion_inverse(quaternion: Any) -> np.ndarray:
    q = normalize_quaternion(quaternion)
    return quaternion_conjugate(q)


def quaternion_relative_rotation(target: Any, current: Any) -> np.ndarray:
    """Return ``q_target * q_current^-1``; quaternion subtraction is invalid."""

    return normalize_quaternion(quaternion_multiply(target, quaternion_inverse(current)))


def quaternion_to_rotation_vector(quaternion: Any) -> np.ndarray:
    """Convert a unit quaternion to the shortest continuous 3-D rotation vector."""

    q = normalize_quaternion(quaternion)
    # q and -q encode the same rotation; choosing w >= 0 avoids a 2*pi jump.
    if q[0] < 0.0:
        q = -q
    vector_norm = np.linalg.norm(q[1:])
    if vector_norm < 1e-10:
        return 2.0 * q[1:]
    angle = 2.0 * np.arctan2(vector_norm, np.clip(q[0], -1.0, 1.0))
    return q[1:] * (angle / vector_norm)


def build_controller_state(
    drone_state: RawDroneState,
    target_position: Any,
    target_orientation: Any,
) -> np.ndarray:
    """Build ``x=[p_target-p, r_target-r, v, a]`` with shape ``(12,)``.

    ``r_target-r`` is represented by the rotation vector of the relative
    quaternion ``q_target * q_current^-1``.  It is not a component-wise
    quaternion subtraction.
    """

    if not isinstance(drone_state, RawDroneState):
        raise TypeError("drone_state must be a RawDroneState")
    target_p = _vector3(target_position)
    position = airsim_to_controller_frame(drone_state.position)
    velocity = airsim_to_controller_frame(drone_state.velocity)
    acceleration = airsim_to_controller_frame(drone_state.acceleration)
    position_error = target_p - position
    attitude_error = quaternion_to_rotation_vector(
        quaternion_relative_rotation(target_orientation, drone_state.quaternion)
    )
    return np.concatenate((position_error, attitude_error, velocity, acceleration)).astype(
        np.float64, copy=False
    )


def split_controller_state(state: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    value = np.asarray(state, dtype=np.float64).reshape(-1)
    if value.size != 12:
        raise ValueError("Controller state must have shape (12,)")
    return value[0:3], value[3:6], value[6:9], value[9:12]
