"""Exact 12-D/4-D interface published by FALCON."""

from __future__ import annotations

from typing import Any

import numpy as np

from airsim_interface import RawDroneState
from state_adapter import (
    airsim_to_controller_frame,
    build_controller_state,
    quaternion_relative_rotation,
    quaternion_to_rotation_vector,
)
from mpc_controller import MPCReference


INPUT_CONTRACT = "falcon_track_p_gate_relative_12d_v1"
INPUT_FEATURES = (
    "gate_position_error_x",
    "gate_position_error_y",
    "gate_position_error_z",
    "gate_attitude_error_x",
    "gate_attitude_error_y",
    "gate_attitude_error_z",
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "acceleration_x",
    "acceleration_y",
    "acceleration_z",
)
OUTPUT_FEATURES = (
    "collective_thrust_acceleration",
    "body_rate_x",
    "body_rate_y",
    "body_rate_z",
)

# Literal paper-style contract. Keep it separate from INPUT_CONTRACT because
# the latter uses a relative-quaternion rotation vector and is already used by
# deployed FLCN/v2 artifacts. FALCON writes [p_g-p, r_g-r, v, vdot]; here r is
# represented by wrapped XYZ Euler angles.
PAPER_GATE_EULER_INPUT_CONTRACT = "falcon_paper_gate_relative_euler_12d_v1"
PAPER_GATE_EULER_INPUT_FEATURES = (
    "gate_position_error_x",
    "gate_position_error_y",
    "gate_position_error_z",
    "gate_euler_error_roll",
    "gate_euler_error_pitch",
    "gate_euler_error_yaw",
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "acceleration_x",
    "acceleration_y",
    "acceleration_z",
)

TRAJECTORY_INPUT_CONTRACT = "falcon_track_p_trajectory_relative_12d_v1"
TRAJECTORY_INPUT_FEATURES = (
    "reference_position_error_x",
    "reference_position_error_y",
    "reference_position_error_z",
    "reference_attitude_error_x",
    "reference_attitude_error_y",
    "reference_attitude_error_z",
    "reference_velocity_error_x",
    "reference_velocity_error_y",
    "reference_velocity_error_z",
    "reference_acceleration_x",
    "reference_acceleration_y",
    "reference_acceleration_z",
)


def build_gate_input(
    state: RawDroneState,
    gate_position: Any,
    gate_orientation: Any,
) -> np.ndarray:
    """Return ``[p_g-p, r_g-r, v, v_dot]`` with shape ``(12,)``.

    The attitude difference is the shortest rotation vector of the relative
    quaternion, not component-wise Euler or quaternion subtraction.
    """

    value = build_controller_state(state, gate_position, gate_orientation)
    if value.shape != (12,) or not np.isfinite(value).all():
        raise ValueError("FALCON paper input must be one finite 12-D vector")
    return value


def quaternion_to_euler_xyz(quaternion: Any) -> np.ndarray:
    """Convert one ``[w,x,y,z]`` quaternion to XYZ roll/pitch/yaw."""

    q = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    if q.size != 4 or not np.isfinite(q).all():
        raise ValueError("quaternion must be one finite [w,x,y,z] vector")
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        raise ValueError("quaternion cannot be zero")
    w, x, y, z = q / norm
    return np.array(
        [
            np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)),
            np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)),
            np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)),
        ],
        dtype=np.float64,
    )


def build_paper_gate_euler_input(
    state: RawDroneState,
    gate_position: Any,
    gate_orientation: Any,
) -> np.ndarray:
    """Return the literal paper vector ``[p_g-p, r_g-r, v, v_dot]``.

    ``r`` is wrapped XYZ Euler roll/pitch/yaw in AirSim's NED convention.
    This matches the earlier strict-paper converter and is intentionally not
    interchangeable with :func:`build_gate_input`.
    """

    if not isinstance(state, RawDroneState):
        raise TypeError("state must be a RawDroneState")
    gate_p = np.asarray(gate_position, dtype=np.float64).reshape(-1)
    if gate_p.size != 3 or not np.isfinite(gate_p).all():
        raise ValueError("gate_position must be one finite 3-D vector")
    attitude_error = (
        quaternion_to_euler_xyz(gate_orientation)
        - quaternion_to_euler_xyz(state.quaternion)
    )
    attitude_error = (attitude_error + np.pi) % (2.0 * np.pi) - np.pi
    value = np.concatenate(
        (
            gate_p - state.position,
            attitude_error,
            state.velocity,
            state.acceleration,
        )
    ).astype(np.float64, copy=False)
    if value.shape != (12,) or not np.isfinite(value).all():
        raise ValueError("FALCON paper Euler input must be one finite 12-D vector")
    return value


def build_trajectory_input(
    state: RawDroneState,
    reference: MPCReference,
) -> np.ndarray:
    """Return the gate-transparent vector ``[ep, er, ev, a_ref]``."""

    if not isinstance(state, RawDroneState):
        raise TypeError("state must be a RawDroneState")
    position_error = (
        airsim_to_controller_frame(reference.position)
        - airsim_to_controller_frame(state.position)
    )
    attitude_error = quaternion_to_rotation_vector(
        quaternion_relative_rotation(reference.orientation, state.quaternion)
    )
    velocity_error = (
        airsim_to_controller_frame(reference.velocity)
        - airsim_to_controller_frame(state.velocity)
    )
    reference_acceleration = airsim_to_controller_frame(reference.acceleration)
    value = np.concatenate(
        (position_error, attitude_error, velocity_error, reference_acceleration)
    ).astype(np.float64, copy=False)
    if value.shape != (12,) or not np.isfinite(value).all():
        raise ValueError("trajectory-tracking input must be one finite 12-D vector")
    return value
