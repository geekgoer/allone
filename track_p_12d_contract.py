"""Paper-track 12-D student contract and deterministic MPC teacher query.

The student sees only the current gate pose error and measured vehicle state::

    [p_gate - p, r_gate - r, velocity, acceleration]

The MPC teacher still needs a 20-sample reference.  For identifiability, that
hidden reference is not sampled independently: it is generated deterministically
from the same gate and state by advancing at a fixed speed along the gate's
passage axis while correcting lateral/vertical lane error.  Consequently one
12-D input has one teacher label under a fixed configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mpc_controller import MPCController, MPCReference
from state_adapter import normalize_quaternion, quaternion_multiply


TRACK_P_12D_INPUT_FEATURES = (
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


def _vec3(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.size != 3 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite vector with shape (3,)")
    return result


def _rotate_vector(quaternion: Any, vector: Any) -> np.ndarray:
    q = normalize_quaternion(quaternion)
    value = _vec3(vector, "vector")
    pure = np.array([0.0, *value], dtype=np.float64)
    inverse = np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)
    return quaternion_multiply(quaternion_multiply(q, pure), inverse)[1:]


def build_track_p_12d_input(
    gate_position_error: Any,
    gate_attitude_error: Any,
    velocity: Any,
    acceleration: Any,
) -> np.ndarray:
    """Return the exact 12-D physical input used by the Track-P student."""

    result = np.concatenate(
        (
            _vec3(gate_position_error, "gate_position_error"),
            _vec3(gate_attitude_error, "gate_attitude_error"),
            _vec3(velocity, "velocity"),
            _vec3(acceleration, "acceleration"),
        )
    )
    if result.shape != (12,):
        raise AssertionError("Track-P input construction produced the wrong shape")
    return result


@dataclass(frozen=True)
class TrackPTeacherQuery:
    """All reproducible values for one independent 12-D -> 4-D query."""

    policy_input: np.ndarray
    teacher_state: np.ndarray
    current_position: np.ndarray
    reference_window: tuple[MPCReference, ...]


def build_track_p_teacher_query(
    *,
    gate_position: Any,
    gate_orientation: Any,
    gate_position_error: Any,
    gate_attitude_error: Any,
    velocity: Any,
    acceleration: Any,
    dt: float,
    horizon: int,
    reference_speed: float,
) -> TrackPTeacherQuery:
    """Build a deterministic local gate-passage reference for an MPC label.

    The ideal lane is the line through the gate center along gate-local +x.
    Its first future sample is one ``speed * dt`` step ahead of the vehicle's
    longitudinal projection.  Lateral and vertical recovery therefore enter
    the MPC state immediately, while the reference continues through the gate
    instead of stopping at its center.
    """

    if dt <= 0.0 or horizon < 1 or reference_speed <= 0.0:
        raise ValueError("dt, horizon and reference_speed must be positive")
    gate = _vec3(gate_position, "gate_position")
    orientation = normalize_quaternion(gate_orientation)
    position_error = _vec3(gate_position_error, "gate_position_error")
    attitude_error = _vec3(gate_attitude_error, "gate_attitude_error")
    velocity_value = _vec3(velocity, "velocity")
    acceleration_value = _vec3(acceleration, "acceleration")
    current = gate - position_error
    passage_axis = _rotate_vector(orientation, [1.0, 0.0, 0.0])
    passage_axis /= np.linalg.norm(passage_axis)
    longitudinal = float(np.dot(current - gate, passage_axis))
    projected_lane_position = gate + longitudinal * passage_axis
    window = tuple(
        MPCReference(
            projected_lane_position
            + passage_axis * reference_speed * dt * float(index + 1),
            orientation,
            velocity=passage_axis * reference_speed,
            acceleration=np.zeros(3, dtype=np.float64),
        )
        for index in range(horizon)
    )
    teacher_position_error = window[0].position - current
    teacher_state = np.concatenate(
        (teacher_position_error, attitude_error, velocity_value, acceleration_value)
    )
    policy_input = build_track_p_12d_input(
        position_error, attitude_error, velocity_value, acceleration_value
    )
    return TrackPTeacherQuery(
        policy_input=policy_input,
        teacher_state=teacher_state,
        current_position=current,
        reference_window=window,
    )


def solve_track_p_teacher(
    controller: MPCController,
    **query_arguments: Any,
) -> tuple[np.ndarray, TrackPTeacherQuery]:
    """Build one query and return the existing MPC's four-value label."""

    query = build_track_p_teacher_query(**query_arguments)
    return controller.solve(query.teacher_state, query.reference_window), query
