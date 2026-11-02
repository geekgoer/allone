"""Nonlinear quadrotor tracking MPC matching the equations published by FALCON.

FALCON reports ACADO/qpOASES but does not release its generated solver or all
AirSim tuning parameters.  This is therefore an equation-faithful CasADi
single-shooting port, not a bit-identical copy of the authors' executable.
Weights and bounds default to the public ``uzh-rpg/rpg_mpc`` configuration and
are all recorded in the dataset manifest.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

from airsim_interface import RawDroneState
from mpc_controller import MPCReference
from state_adapter import normalize_quaternion


@dataclass(frozen=True)
class NonlinearMPCConfig:
    dt: float = 0.05
    horizon: int = 20
    gravity: float = 9.81
    q_position_xy: float = 200.0
    q_position_z: float = 500.0
    q_attitude: float = 50.0
    q_velocity: float = 10.0
    r_thrust: float = 1.0
    r_pitch_roll: float = 1.0
    r_yaw: float = 1.0
    min_thrust: float = 2.0
    max_thrust: float = 20.0
    max_body_rate_xy: float = 3.0
    max_body_rate_z: float = 2.0
    maximum_iterations: int = 50
    tolerance: float = 1e-5

    def validate(self) -> None:
        if self.dt <= 0.0 or self.horizon < 1:
            raise ValueError("MPC dt and horizon must be positive")
        if self.min_thrust < 0.0 or self.max_thrust <= self.min_thrust:
            raise ValueError("MPC thrust bounds are invalid")
        if min(self.max_body_rate_xy, self.max_body_rate_z) <= 0.0:
            raise ValueError("MPC body-rate bounds must be positive")
        weights = (
            self.q_position_xy,
            self.q_position_z,
            self.q_attitude,
            self.q_velocity,
            self.r_thrust,
            self.r_pitch_roll,
            self.r_yaw,
        )
        if any(value < 0.0 or not np.isfinite(value) for value in weights):
            raise ValueError("MPC weights must be finite and non-negative")


def _reference_vector(reference: MPCReference, gravity: float) -> tuple[np.ndarray, np.ndarray]:
    q = normalize_quaternion(reference.orientation)
    state = np.concatenate((reference.position, q, reference.velocity)).astype(np.float64)
    # In NED, gravity is +z and collective thrust acts along body -z.
    # The feed-forward magnitude is exact for level/vertical references and a
    # useful neutral input for the nonlinear optimizer on curved references.
    thrust = float(np.linalg.norm(np.array([0.0, 0.0, gravity]) - reference.acceleration))
    control = np.array([thrust, 0.0, 0.0, 0.0], dtype=np.float64)
    return state, control


class CasadiTrackingMPC:
    """Warm-started nonlinear MPC with output ``[F, wx, wy, wz]``."""

    def __init__(self, config: NonlinearMPCConfig | None = None) -> None:
        self.config = config or NonlinearMPCConfig()
        self.config.validate()
        try:
            import casadi as ca  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on user environment.
            raise RuntimeError(
                "CasADi is required for the paper-fidelity teacher. Install it in the "
                "AirSim Python environment with: python -m pip install casadi==3.6.7"
            ) from exc
        self.ca = ca
        self._last_solution = np.tile(
            np.array([self.config.gravity, 0.0, 0.0, 0.0], dtype=np.float64),
            self.config.horizon,
        )
        self._solver, self._lower, self._upper = self._build_solver()

    def _quat_multiply(self, left: Any, right: Any) -> Any:
        ca = self.ca
        w1, x1, y1, z1 = [left[index] for index in range(4)]
        w2, x2, y2, z2 = [right[index] for index in range(4)]
        return ca.vertcat(
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )

    def _rotation_matrix(self, q: Any) -> Any:
        ca = self.ca
        w, x, y, z = [q[index] for index in range(4)]
        return ca.vertcat(
            ca.horzcat(1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            ca.horzcat(2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            ca.horzcat(2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        )

    def _dynamics(self, state: Any, control: Any) -> Any:
        ca = self.ca
        q = state[3:7]
        velocity = state[7:10]
        thrust = control[0]
        rates = control[1:4]
        world_acceleration = ca.vertcat(0.0, 0.0, self.config.gravity) + self._rotation_matrix(q) @ ca.vertcat(0.0, 0.0, -thrust)
        q_dot = 0.5 * self._quat_multiply(q, ca.vertcat(0.0, rates))
        return ca.vertcat(velocity, q_dot, world_acceleration)

    def _integrate(self, state: Any, control: Any) -> Any:
        dt = self.config.dt
        k1 = self._dynamics(state, control)
        k2 = self._dynamics(state + 0.5 * dt * k1, control)
        k3 = self._dynamics(state + 0.5 * dt * k2, control)
        k4 = self._dynamics(state + dt * k3, control)
        next_state = state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        q = next_state[3:7]
        normalized_q = q / self.ca.sqrt(self.ca.sumsqr(q) + 1e-12)
        return self.ca.vertcat(next_state[0:3], normalized_q, next_state[7:10])

    def _build_solver(self) -> tuple[Any, np.ndarray, np.ndarray]:
        ca = self.ca
        cfg = self.config
        N = cfg.horizon
        controls = ca.MX.sym("controls", 4, N)
        # Parameters: x0 plus N state references and N neutral inputs.
        parameters = ca.MX.sym("parameters", 10 + N * 14)
        state = parameters[0:10]
        objective = 0
        q_position = ca.DM([cfg.q_position_xy, cfg.q_position_xy, cfg.q_position_z])
        q_velocity = ca.DM([cfg.q_velocity] * 3)
        r_control = ca.DM([cfg.r_thrust, cfg.r_pitch_roll, cfg.r_pitch_roll, cfg.r_yaw])
        offset = 10
        for index in range(N):
            reference_state = parameters[offset : offset + 10]
            reference_control = parameters[offset + 10 : offset + 14]
            offset += 14
            state = self._integrate(state, controls[:, index])
            position_error = state[0:3] - reference_state[0:3]
            velocity_error = state[7:10] - reference_state[7:10]
            q_ref = reference_state[3:7]
            q_state = state[3:7]
            q_ref_inverse = ca.vertcat(q_ref[0], -q_ref[1], -q_ref[2], -q_ref[3])
            q_error = self._quat_multiply(q_ref_inverse, q_state)
            attitude_error = 2.0 * q_error[1:4]
            control_error = controls[:, index] - reference_control
            objective += ca.dot(q_position, position_error * position_error)
            objective += cfg.q_attitude * ca.sumsqr(attitude_error)
            objective += ca.dot(q_velocity, velocity_error * velocity_error)
            objective += ca.dot(r_control, control_error * control_error)
        problem = {
            "x": ca.reshape(controls, 4 * N, 1),
            "p": parameters,
            "f": objective,
        }
        options = {
            "print_time": False,
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "ipopt.max_iter": cfg.maximum_iterations,
            "ipopt.tol": cfg.tolerance,
            "ipopt.acceptable_tol": max(cfg.tolerance * 10.0, 1e-4),
            "ipopt.warm_start_init_point": "yes",
        }
        solver = ca.nlpsol("falcon_tracking_mpc", "ipopt", problem, options)
        lower = np.tile(
            np.array([cfg.min_thrust, -cfg.max_body_rate_xy, -cfg.max_body_rate_xy, -cfg.max_body_rate_z]),
            N,
        )
        upper = np.tile(
            np.array([cfg.max_thrust, cfg.max_body_rate_xy, cfg.max_body_rate_xy, cfg.max_body_rate_z]),
            N,
        )
        return solver, lower, upper

    def solve(self, state: RawDroneState, references: Sequence[MPCReference]) -> np.ndarray:
        if not isinstance(state, RawDroneState):
            raise TypeError("state must be RawDroneState")
        if not references:
            raise ValueError("references must not be empty")
        values = list(references[: self.config.horizon])
        values.extend([values[-1]] * (self.config.horizon - len(values)))
        x0 = np.concatenate((state.position, normalize_quaternion(state.quaternion), state.velocity))
        parameters = [x0]
        for reference in values:
            reference_state, reference_control = _reference_vector(reference, self.config.gravity)
            parameters.extend((reference_state, reference_control))
        solution = self._solver(
            x0=self._last_solution,
            p=np.concatenate(parameters),
            lbx=self._lower,
            ubx=self._upper,
        )
        statistics = self._solver.stats()
        if not bool(statistics.get("success", False)):
            raise RuntimeError(
                "nonlinear MPC did not converge: "
                f"{statistics.get('return_status', 'unknown status')}"
            )
        controls = np.asarray(solution["x"], dtype=np.float64).reshape(-1)
        if controls.shape != (4 * self.config.horizon,) or not np.isfinite(controls).all():
            raise RuntimeError("nonlinear MPC returned an invalid solution")
        # Shift the previous optimum for the next receding-horizon solve.
        self._last_solution = np.concatenate((controls[4:], controls[-4:]))
        return controls[:4].copy()

    def reset(self) -> None:
        self._last_solution = np.tile(
            np.array([self.config.gravity, 0.0, 0.0, 0.0], dtype=np.float64),
            self.config.horizon,
        )

    def signature(self) -> dict[str, Any]:
        return {
            "backend": "casadi_nonlinear_tracking_mpc_v1",
            "fidelity": "published_dynamics_and_horizon; not author ACADO binary",
            **asdict(self.config),
        }
