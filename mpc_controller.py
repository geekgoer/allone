"""FALCON-compatible model-predictive teacher for AirSim.

The FALCON paper specifies the 12-D input, four-value output, 20-step
horizon, and 50 ms sample period, but it does not publish a complete set of
AirSim actuator parameters or an executable Python solver.  The implementation
here is therefore an explicit engineering-compatible teacher: a finite-horizon
linear MPC/LQR position model feeds a bounded attitude/body-rate command and a
physical collective-thrust acceleration.  It is not claimed to be the exact
FALCON or PAMPC solver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from state_adapter import normalize_quaternion, split_controller_state


def _vec3(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.size != 3:
        raise ValueError(f"{name} must have shape (3,)")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result.copy()


@dataclass
class MPCReference:
    """One future reference sample in the controller's NED world frame."""

    position: np.ndarray
    orientation: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    body_rates: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self) -> None:
        self.position = _vec3(self.position, "position")
        self.orientation = normalize_quaternion(self.orientation)
        self.velocity = _vec3(self.velocity, "velocity")
        self.acceleration = _vec3(self.acceleration, "acceleration")
        self.body_rates = _vec3(self.body_rates, "body_rates")


def _first_reference(reference: Any) -> MPCReference:
    if reference is None:
        return MPCReference(np.zeros(3))
    if isinstance(reference, MPCReference):
        return reference
    if isinstance(reference, np.ndarray):
        value = np.asarray(reference, dtype=np.float64)
        if value.shape == (12,):
            # A raw 12-D reference is accepted for interface convenience.  Its
            # first three entries are interpreted as position only when used
            # by a caller that has not already built the error state.
            return MPCReference(value[0:3], velocity=value[6:9], acceleration=value[9:12])
        if value.ndim == 2 and value.shape[1] == 12 and value.shape[0] >= 1:
            return _first_reference(value[0])
    if isinstance(reference, Iterable) and not isinstance(reference, (str, bytes)):
        values = list(reference)
        if not values:
            return MPCReference(np.zeros(3))
        return _first_reference(values[0])
    raise TypeError("reference must be MPCReference, a (12,) array, or a future reference sequence")


def _reference_sequence(reference: Any, horizon: int) -> list[MPCReference]:
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if isinstance(reference, MPCReference) or reference is None:
        first = _first_reference(reference)
        return [first for _ in range(horizon)]
    if isinstance(reference, np.ndarray):
        value = np.asarray(reference, dtype=np.float64)
        if value.shape == (12,):
            first = _first_reference(value)
            return [first for _ in range(horizon)]
        if value.ndim == 2 and value.shape[1] == 12:
            values = [_first_reference(row) for row in value]
        else:
            raise TypeError("reference array must have shape (12,) or (N, 12)")
    elif isinstance(reference, Iterable) and not isinstance(reference, (str, bytes)):
        values = [_first_reference(item) for item in list(reference)]
    else:
        raise TypeError("reference must be an MPCReference or a future reference sequence")
    if not values:
        values = [MPCReference(np.zeros(3))]
    if len(values) < horizon:
        values.extend([values[-1]] * (horizon - len(values)))
    return values[:horizon]


class MPCController:
    """Linear receding-horizon trajectory tracker with bounded AirSim output."""

    def __init__(
        self,
        dt: float = 0.05,
        horizon: int = 20,
        gravity: float = 9.81,
        position_weight: float = 4.0,
        velocity_weight: float = 1.0,
        input_weight: float = 0.5,
        max_horizontal_accel: float = 4.0,
        max_thrust_accel: float = 18.0,
        max_body_rate: float = 3.0,
        attitude_gain: float = 3.5,
    ) -> None:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if horizon < 1:
            raise ValueError("horizon must be at least 1")
        self.dt = float(dt)
        self.horizon = int(horizon)
        self.gravity = float(gravity)
        self.max_horizontal_accel = float(max_horizontal_accel)
        self.max_thrust_accel = float(max_thrust_accel)
        self.max_body_rate = float(max_body_rate)
        self.attitude_gain = float(attitude_gain)
        (
            self._state_prediction,
            self._reference_tracking_gain,
        ) = self._finite_horizon_tracking_matrices(
            position_weight, velocity_weight, input_weight
        )

    def _finite_horizon_tracking_matrices(
        self,
        position_weight: float,
        velocity_weight: float,
        input_weight: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if min(position_weight, velocity_weight, input_weight) <= 0.0:
            raise ValueError("MPC weights must be positive")
        identity = np.eye(3)
        # Absolute NED double-integrator prediction:
        # p[k+1] = p[k] + dt*v[k] + 0.5*dt^2*a[k]
        # v[k+1] = v[k] + dt*a[k].
        A = np.block([[identity, self.dt * identity], [np.zeros((3, 3)), identity]])
        B = np.vstack([0.5 * self.dt**2 * identity, self.dt * identity])
        state_size = 6
        input_size = 3
        state_prediction = np.zeros((state_size * self.horizon, state_size))
        control_prediction = np.zeros(
            (state_size * self.horizon, input_size * self.horizon)
        )
        for row in range(self.horizon):
            state_prediction[
                row * state_size : (row + 1) * state_size
            ] = np.linalg.matrix_power(A, row + 1)
            for column in range(row + 1):
                control_prediction[
                    row * state_size : (row + 1) * state_size,
                    column * input_size : (column + 1) * input_size,
                ] = np.linalg.matrix_power(A, row - column) @ B
        state_weight = np.diag(
            [position_weight] * 3 + [velocity_weight] * 3
        )
        state_weights = np.kron(np.eye(self.horizon), state_weight)
        input_weights = input_weight * np.eye(input_size * self.horizon)
        hessian = (
            control_prediction.T @ state_weights @ control_prediction
            + input_weights
        )
        tracking_gain = np.linalg.solve(
            hessian,
            control_prediction.T @ state_weights,
        )
        return state_prediction, tracking_gain

    def solve(self, state: Any, reference: Any = None) -> np.ndarray:
        """Return exactly ``np.array([F, wx, wy, wz])``.

        ``state`` is the 12-D adapter output.  The position component is
        target-minus-current in NED, attitude is the relative-rotation vector,
        velocity is current NED velocity, and acceleration is current NED
        acceleration.  ``reference`` supplies future target velocity and
        acceleration when tracking a moving trajectory.
        """

        position_error, attitude_error, velocity, _acceleration = split_controller_state(state)
        references = _reference_sequence(reference, self.horizon)
        first = references[0]
        current_position = first.position - position_error
        current_state = np.concatenate((current_position, velocity))
        reference_states = np.concatenate(
            [np.concatenate((item.position, item.velocity)) for item in references]
        )
        predicted_without_control = self._state_prediction @ current_state
        control_sequence = self._reference_tracking_gain @ (
            reference_states - predicted_without_control
        )
        desired_net_accel = control_sequence[:3] + first.acceleration
        desired_net_accel = desired_net_accel.astype(np.float64, copy=False)
        desired_net_accel[0:2] = np.clip(
            desired_net_accel[0:2],
            -self.max_horizontal_accel,
            self.max_horizontal_accel,
        )
        desired_net_accel[2] = np.clip(
            desired_net_accel[2],
            -self.max_horizontal_accel,
            self.max_horizontal_accel,
        )

        # In AirSim NED, positive z is down and collective thrust accelerates
        # the vehicle upward.  The required thrust vector is therefore
        # [a_x, a_y, a_z - gravity].  Taking its norm compensates the vertical
        # thrust loss introduced by roll/pitch during horizontal flight.
        required_thrust_vector = np.array(
            [desired_net_accel[0], desired_net_accel[1], desired_net_accel[2] - self.gravity],
            dtype=np.float64,
        )
        thrust_accel = float(
            np.clip(np.linalg.norm(required_thrust_vector), 0.0, self.max_thrust_accel)
        )

        # Small-angle mapping from desired horizontal NED acceleration to the
        # desired roll/pitch around the reference attitude.  AirSim's body-rate
        # command is roll, pitch, yaw in rad/s.
        desired_attitude = np.array(
            [
                desired_net_accel[1] / max(thrust_accel, self.gravity),
                -desired_net_accel[0] / max(thrust_accel, self.gravity),
                0.0,
            ],
            dtype=np.float64,
        )
        desired_attitude[0:2] = np.clip(desired_attitude[0:2], -0.45, 0.45)
        # q_target*q_current^-1 is stored in the state, so approximately
        # current-minus-target is -attitude_error.  The sum below is the
        # desired-minus-current error for the body-rate inner loop.
        attitude_command_error = desired_attitude + attitude_error
        rates = np.clip(
            self.attitude_gain * attitude_command_error,
            -self.max_body_rate,
            self.max_body_rate,
        )
        return np.array([thrust_accel, rates[0], rates[1], rates[2]], dtype=np.float64)

    def reset(self) -> None:
        """Reset per-episode state.

        This teacher currently has no warm-start memory, so reset is a
        deliberate no-op.  The explicit hook keeps the collector correct if a
        solver with internal state is introduced later.
        """

        return None
