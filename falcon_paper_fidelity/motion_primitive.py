"""Deterministic quintic motion primitives for the FALCON paper protocol.

The paper says that the trajectory planner polynomially fits a reference path
but does not publish its boundary-condition implementation.  This module keeps
the legacy per-gate primitive for reproducibility and also provides the v3
global C2 route, whose prediction windows continue across gate boundaries.

The planner has no independently sampled hidden target speed.  Given the
published 12-D gate-relative input, a fixed configuration, and an identity-
oriented gate, its reference window is deterministic.  That prevents a single
12-D input from acquiring contradictory labels solely because an unobserved
speed command changed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from mpc_controller import MPCReference
from state_adapter import normalize_quaternion, quaternion_multiply


def _vec3(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.size != 3 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite vector with shape (3,)")
    return result.copy()


def _rotate(quaternion: Any, vector: Any) -> np.ndarray:
    q = normalize_quaternion(quaternion)
    v = _vec3(vector, "vector")
    pure = np.array([0.0, *v], dtype=np.float64)
    inverse = np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)
    return quaternion_multiply(quaternion_multiply(q, pure), inverse)[1:]


@dataclass(frozen=True)
class MotionPrimitiveConfig:
    """Explicit assumptions not fully specified in the FALCON paper."""

    nominal_speed: float = 5.0
    terminal_speed: float = 5.0
    exit_distance: float = 1.0
    minimum_duration: float = 0.8
    maximum_duration: float = 8.0
    maximum_speed: float = 8.0
    maximum_acceleration: float = 6.0
    feasibility_samples: int = 81
    gravity: float = 9.8066
    body_rate_difference_dt: float = 1e-3

    def validate(self) -> None:
        values = (
            self.nominal_speed,
            self.terminal_speed,
            self.exit_distance,
            self.minimum_duration,
            self.maximum_duration,
            self.maximum_speed,
            self.maximum_acceleration,
            self.gravity,
            self.body_rate_difference_dt,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("motion-primitive scalar settings must be positive and finite")
        if self.minimum_duration > self.maximum_duration:
            raise ValueError("minimum_duration must not exceed maximum_duration")
        if self.feasibility_samples < 5:
            raise ValueError("feasibility_samples must be at least 5")


class QuinticPolynomial:
    """Three-axis fifth-order polynomial with p/v/a boundary conditions."""

    def __init__(
        self,
        start_position: Any,
        start_velocity: Any,
        start_acceleration: Any,
        end_position: Any,
        end_velocity: Any,
        end_acceleration: Any,
        duration: float,
    ) -> None:
        self.duration = float(duration)
        if not np.isfinite(self.duration) or self.duration <= 0.0:
            raise ValueError("duration must be positive and finite")
        p0 = _vec3(start_position, "start_position")
        v0 = _vec3(start_velocity, "start_velocity")
        a0 = _vec3(start_acceleration, "start_acceleration")
        p1 = _vec3(end_position, "end_position")
        v1 = _vec3(end_velocity, "end_velocity")
        a1 = _vec3(end_acceleration, "end_acceleration")
        T = self.duration
        coefficients = np.empty((3, 6), dtype=np.float64)
        coefficients[:, 0] = p0
        coefficients[:, 1] = v0
        coefficients[:, 2] = 0.5 * a0
        system = np.array(
            [
                [T**3, T**4, T**5],
                [3.0 * T**2, 4.0 * T**3, 5.0 * T**4],
                [6.0 * T, 12.0 * T**2, 20.0 * T**3],
            ],
            dtype=np.float64,
        )
        rhs = np.column_stack(
            (
                p1 - (p0 + v0 * T + 0.5 * a0 * T**2),
                v1 - (v0 + a0 * T),
                a1 - a0,
            )
        )
        coefficients[:, 3:6] = np.linalg.solve(system, rhs.T).T
        self.coefficients = coefficients

    def evaluate(self, time_value: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        t = float(np.clip(time_value, 0.0, self.duration))
        c = self.coefficients
        position = c[:, 0] + c[:, 1] * t + c[:, 2] * t**2 + c[:, 3] * t**3 + c[:, 4] * t**4 + c[:, 5] * t**5
        velocity = c[:, 1] + 2.0 * c[:, 2] * t + 3.0 * c[:, 3] * t**2 + 4.0 * c[:, 4] * t**3 + 5.0 * c[:, 5] * t**4
        acceleration = 2.0 * c[:, 2] + 6.0 * c[:, 3] * t + 12.0 * c[:, 4] * t**2 + 20.0 * c[:, 5] * t**3
        return position, velocity, acceleration


class MotionPrimitiveInfeasible(RuntimeError):
    """No candidate duration satisfied the configured trajectory limits."""


@dataclass(frozen=True)
class TimedQuinticSegment:
    """One segment in a globally timed, gate-transparent reference path."""

    primitive: QuinticPolynomial
    heading_orientation: np.ndarray
    start_time: float

    @property
    def end_time(self) -> float:
        return self.start_time + self.primitive.duration


class ContinuousQuinticTrajectory:
    """C2 piecewise-quintic trajectory sampled across segment boundaries."""

    def __init__(self, segments: Iterable[TimedQuinticSegment]) -> None:
        self.segments = tuple(segments)
        if not self.segments:
            raise ValueError("continuous trajectory needs at least one segment")
        expected_start = 0.0
        for segment in self.segments:
            if not np.isclose(segment.start_time, expected_start, atol=1e-10):
                raise ValueError("trajectory segments must be contiguous")
            expected_start = segment.end_time
        self.duration = float(expected_start)
        self.boundary_times = tuple(segment.end_time for segment in self.segments[:-1])
        final_p, final_v, final_a = self.segments[-1].primitive.evaluate(
            self.segments[-1].primitive.duration
        )
        self.final_position = final_p
        self.final_velocity = final_v
        self.final_acceleration = final_a
        self.final_orientation = normalize_quaternion(
            self.segments[-1].heading_orientation
        )

    def evaluate(
        self, time_value: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return p/v/a and heading at global trajectory time."""

        t = max(0.0, float(time_value))
        if t >= self.duration:
            return (
                self.final_position.copy(),
                self.final_velocity.copy(),
                self.final_acceleration.copy(),
                self.final_orientation.copy(),
            )
        for segment in self.segments:
            if t < segment.end_time or segment is self.segments[-1]:
                p, v, a = segment.primitive.evaluate(t - segment.start_time)
                return p, v, a, normalize_quaternion(segment.heading_orientation)
        raise AssertionError("trajectory segment lookup failed")


def _rotation_matrix(quaternion: Any) -> np.ndarray:
    w, x, y, z = normalize_quaternion(quaternion)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _quaternion_from_rotation_matrix(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(value))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quaternion = np.array(
            [
                0.25 * scale,
                (value[2, 1] - value[1, 2]) / scale,
                (value[0, 2] - value[2, 0]) / scale,
                (value[1, 0] - value[0, 1]) / scale,
            ]
        )
    else:
        diagonal = np.diag(value)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = 2.0 * np.sqrt(max(1.0 + value[0, 0] - value[1, 1] - value[2, 2], 0.0))
            quaternion = np.array(
                [
                    (value[2, 1] - value[1, 2]) / scale,
                    0.25 * scale,
                    (value[0, 1] + value[1, 0]) / scale,
                    (value[0, 2] + value[2, 0]) / scale,
                ]
            )
        elif index == 1:
            scale = 2.0 * np.sqrt(max(1.0 + value[1, 1] - value[0, 0] - value[2, 2], 0.0))
            quaternion = np.array(
                [
                    (value[0, 2] - value[2, 0]) / scale,
                    (value[0, 1] + value[1, 0]) / scale,
                    0.25 * scale,
                    (value[1, 2] + value[2, 1]) / scale,
                ]
            )
        else:
            scale = 2.0 * np.sqrt(max(1.0 + value[2, 2] - value[0, 0] - value[1, 1], 0.0))
            quaternion = np.array(
                [
                    (value[1, 0] - value[0, 1]) / scale,
                    (value[0, 2] + value[2, 0]) / scale,
                    (value[1, 2] + value[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    return normalize_quaternion(quaternion)


def _zup_flu_attitude(acceleration_ned: np.ndarray, gate_orientation_ned: np.ndarray, gravity: float) -> np.ndarray:
    """Return the z-up/FLU attitude implied by translational acceleration."""

    acceleration_zup = acceleration_ned * np.array([1.0, -1.0, -1.0])
    gate_zup = normalize_quaternion(
        gate_orientation_ned * np.array([1.0, 1.0, -1.0, -1.0])
    )
    gate_rotation = _rotation_matrix(gate_zup)
    heading = gate_rotation[:, 0].copy()
    heading[2] = 0.0
    if np.linalg.norm(heading) < 1e-9:
        heading = np.array([1.0, 0.0, 0.0])
    heading /= np.linalg.norm(heading)
    body_z = acceleration_zup + np.array([0.0, 0.0, gravity])
    body_z /= max(np.linalg.norm(body_z), 1e-12)
    body_y = np.cross(body_z, heading)
    if np.linalg.norm(body_y) < 1e-9:
        heading = np.array([0.0, 1.0, 0.0])
        body_y = np.cross(body_z, heading)
    body_y /= np.linalg.norm(body_y)
    body_x = np.cross(body_y, body_z)
    body_x /= np.linalg.norm(body_x)
    return _quaternion_from_rotation_matrix(np.column_stack((body_x, body_y, body_z)))


def _zup_flu_body_rates(current: np.ndarray, following: np.ndarray, dt: float) -> np.ndarray:
    """Finite-difference body rates from two body-to-world quaternions."""

    q0 = normalize_quaternion(current)
    q1 = normalize_quaternion(following)
    if np.dot(q0, q1) < 0.0:
        q1 = -q1
    delta = normalize_quaternion(
        quaternion_multiply(np.array([q0[0], -q0[1], -q0[2], -q0[3]]), q1)
    )
    if delta[0] < 0.0:
        delta = -delta
    vector_norm = float(np.linalg.norm(delta[1:]))
    if vector_norm < 1e-12:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(vector_norm, float(np.clip(delta[0], -1.0, 1.0)))
    return delta[1:] / vector_norm * (angle / dt)


class QuinticMotionPrimitivePlanner:
    """Generate a future reference window through the active gate."""

    def __init__(self, config: MotionPrimitiveConfig | None = None) -> None:
        self.config = config or MotionPrimitiveConfig()
        self.config.validate()

    @staticmethod
    def passage_axis(gate_orientation: Any) -> np.ndarray:
        """Return the unit gate-passage axis in AirSim NED coordinates."""

        normal = _rotate(gate_orientation, [1.0, 0.0, 0.0])
        return normal / max(np.linalg.norm(normal), 1e-12)

    def plan(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        acceleration: np.ndarray,
        gate_position: np.ndarray,
        gate_orientation: np.ndarray,
    ) -> QuinticPolynomial:
        normal = self.passage_axis(gate_orientation)
        end_position = gate_position + self.config.exit_distance * normal
        end_velocity = self.config.terminal_speed * normal
        distance = float(np.linalg.norm(end_position - position))
        base_duration = np.clip(
            distance / self.config.nominal_speed,
            self.config.minimum_duration,
            self.config.maximum_duration,
        )
        candidates = np.linspace(
            float(base_duration),
            self.config.maximum_duration,
            17,
            dtype=np.float64,
        )
        last_limits: tuple[float, float] | None = None
        for duration in np.unique(candidates):
            primitive = QuinticPolynomial(
                position,
                velocity,
                acceleration,
                end_position,
                end_velocity,
                np.zeros(3),
                float(duration),
            )
            samples = [
                primitive.evaluate(value)
                for value in np.linspace(0.0, float(duration), self.config.feasibility_samples)
            ]
            max_speed = max(float(np.linalg.norm(item[1])) for item in samples)
            max_acceleration = max(float(np.linalg.norm(item[2])) for item in samples)
            last_limits = (max_speed, max_acceleration)
            if (
                max_speed <= self.config.maximum_speed + 1e-9
                and max_acceleration <= self.config.maximum_acceleration + 1e-9
            ):
                return primitive
        assert last_limits is not None
        raise MotionPrimitiveInfeasible(
            "no feasible quintic candidate: "
            f"max_speed={last_limits[0]:.3f}m/s "
            f"max_acceleration={last_limits[1]:.3f}m/s^2"
        )

    def _fit_boundary_segment(
        self,
        start_position: Any,
        start_velocity: Any,
        start_acceleration: Any,
        end_position: Any,
        end_velocity: Any,
        end_acceleration: Any,
    ) -> QuinticPolynomial:
        p0 = _vec3(start_position, "start_position")
        v0 = _vec3(start_velocity, "start_velocity")
        a0 = _vec3(start_acceleration, "start_acceleration")
        p1 = _vec3(end_position, "end_position")
        v1 = _vec3(end_velocity, "end_velocity")
        a1 = _vec3(end_acceleration, "end_acceleration")
        distance = float(np.linalg.norm(p1 - p0))
        base_duration = np.clip(
            distance / self.config.nominal_speed,
            self.config.minimum_duration,
            self.config.maximum_duration,
        )
        candidates = np.linspace(
            float(base_duration), self.config.maximum_duration, 33, dtype=np.float64
        )
        last_limits: tuple[float, float] | None = None
        for duration in np.unique(candidates):
            primitive = QuinticPolynomial(p0, v0, a0, p1, v1, a1, float(duration))
            samples = [
                primitive.evaluate(value)
                for value in np.linspace(
                    0.0, float(duration), self.config.feasibility_samples
                )
            ]
            max_speed = max(float(np.linalg.norm(item[1])) for item in samples)
            max_acceleration = max(float(np.linalg.norm(item[2])) for item in samples)
            last_limits = (max_speed, max_acceleration)
            if (
                max_speed <= self.config.maximum_speed + 1e-9
                and max_acceleration <= self.config.maximum_acceleration + 1e-9
            ):
                return primitive
        assert last_limits is not None
        raise MotionPrimitiveInfeasible(
            "no feasible continuous-route segment: "
            f"max_speed={last_limits[0]:.3f}m/s "
            f"max_acceleration={last_limits[1]:.3f}m/s^2"
        )

    def plan_continuous_route(
        self,
        position: Any,
        velocity: Any,
        acceleration: Any,
        gates: Iterable[Any],
        *,
        final_exit_distance: float,
    ) -> ContinuousQuinticTrajectory:
        """Plan one C2 trajectory through every gate before control starts."""

        gate_list = tuple(gates)
        if not gate_list:
            raise ValueError("continuous route needs at least one gate")
        if not np.isfinite(final_exit_distance) or final_exit_distance <= 0.0:
            raise ValueError("final_exit_distance must be positive and finite")
        points = [_vec3(position, "position")]
        velocities = [_vec3(velocity, "velocity")]
        accelerations = [_vec3(acceleration, "acceleration")]
        headings: list[np.ndarray] = []
        for gate in gate_list:
            center = _vec3(gate.center, "gate.center")
            orientation = normalize_quaternion(gate.orientation)
            normal = _rotate(orientation, [1.0, 0.0, 0.0])
            normal /= max(np.linalg.norm(normal), 1e-12)
            points.append(center)
            velocities.append(self.config.terminal_speed * normal)
            accelerations.append(np.zeros(3))
            headings.append(orientation)
        last_orientation = normalize_quaternion(gate_list[-1].orientation)
        last_normal = _rotate(last_orientation, [1.0, 0.0, 0.0])
        last_normal /= max(np.linalg.norm(last_normal), 1e-12)
        points.append(points[-1] + float(final_exit_distance) * last_normal)
        velocities.append(np.zeros(3))
        accelerations.append(np.zeros(3))
        headings.append(last_orientation)

        segments: list[TimedQuinticSegment] = []
        start_time = 0.0
        for index in range(len(points) - 1):
            primitive = self._fit_boundary_segment(
                points[index],
                velocities[index],
                accelerations[index],
                points[index + 1],
                velocities[index + 1],
                accelerations[index + 1],
            )
            segment = TimedQuinticSegment(
                primitive=primitive,
                heading_orientation=headings[index].copy(),
                start_time=start_time,
            )
            segments.append(segment)
            start_time = segment.end_time
        return ContinuousQuinticTrajectory(segments)

    def reference_window_for_trajectory(
        self,
        trajectory: ContinuousQuinticTrajectory,
        *,
        elapsed: float,
        dt: float,
        horizon: int,
    ) -> list[MPCReference]:
        """Sample a future window continuously across route segment boundaries."""

        if elapsed < 0.0 or dt <= 0.0 or horizon < 1:
            raise ValueError("elapsed must be non-negative and dt/horizon positive")
        references: list[MPCReference] = []
        for index in range(horizon):
            sample_time = float(elapsed) + index * float(dt)
            ref_p, ref_v, ref_a, heading = trajectory.evaluate(sample_time)
            following_time = sample_time + self.config.body_rate_difference_dt
            _, _, following_acceleration, following_heading = trajectory.evaluate(
                following_time
            )
            q_zup = _zup_flu_attitude(ref_a, heading, self.config.gravity)
            q_following_zup = _zup_flu_attitude(
                following_acceleration, following_heading, self.config.gravity
            )
            rates_zup = _zup_flu_body_rates(
                q_zup, q_following_zup, self.config.body_rate_difference_dt
            )
            q_ned = normalize_quaternion(
                q_zup * np.array([1.0, 1.0, -1.0, -1.0])
            )
            rates_ned = rates_zup * np.array([1.0, -1.0, -1.0])
            references.append(
                MPCReference(
                    position=ref_p,
                    orientation=q_ned,
                    velocity=ref_v,
                    acceleration=ref_a,
                    body_rates=rates_ned,
                )
            )
        return references

    def reference_window_for_primitive(
        self,
        primitive: QuinticPolynomial,
        *,
        gate_orientation: Any,
        elapsed: float,
        dt: float,
        horizon: int,
    ) -> list[MPCReference]:
        if elapsed < 0.0 or dt <= 0.0 or horizon < 1:
            raise ValueError("elapsed must be non-negative and dt/horizon positive")
        gate_q = normalize_quaternion(gate_orientation)
        references: list[MPCReference] = []
        for index in range(horizon):
            sample_time = min(float(elapsed) + index * float(dt), primitive.duration)
            ref_p, ref_v, ref_a = primitive.evaluate(sample_time)
            following_time = min(
                sample_time + self.config.body_rate_difference_dt,
                primitive.duration,
            )
            _, _, following_acceleration = primitive.evaluate(following_time)
            q_zup = _zup_flu_attitude(ref_a, gate_q, self.config.gravity)
            q_following_zup = _zup_flu_attitude(
                following_acceleration, gate_q, self.config.gravity
            )
            if following_time > sample_time:
                rates_zup = _zup_flu_body_rates(
                    q_zup, q_following_zup, following_time - sample_time
                )
            else:
                rates_zup = np.zeros(3)
            # Both conversions are involutions: z-up/FLU -> NED/FRD.
            q_ned = normalize_quaternion(q_zup * np.array([1.0, 1.0, -1.0, -1.0]))
            rates_ned = rates_zup * np.array([1.0, -1.0, -1.0])
            references.append(
                MPCReference(
                    position=ref_p,
                    orientation=q_ned,
                    velocity=ref_v,
                    acceleration=ref_a,
                    body_rates=rates_ned,
                )
            )
        return references

    def reference_window(
        self,
        *,
        position: Any,
        velocity: Any,
        acceleration: Any,
        gate_position: Any,
        gate_orientation: Any,
        dt: float,
        horizon: int,
    ) -> list[MPCReference]:
        if dt <= 0.0 or horizon < 1:
            raise ValueError("dt and horizon must be positive")
        p = _vec3(position, "position")
        v = _vec3(velocity, "velocity")
        a = _vec3(acceleration, "acceleration")
        gate_p = _vec3(gate_position, "gate_position")
        gate_q = normalize_quaternion(gate_orientation)
        primitive = self.plan(p, v, a, gate_p, gate_q)
        return self.reference_window_for_primitive(
            primitive,
            gate_orientation=gate_q,
            elapsed=0.0,
            dt=dt,
            horizon=horizon,
        )

    def signature(self) -> dict[str, float | int | str]:
        return {
            "planner": "segment_time_quintic_pva_boundary_full_reference_v2",
            **self.config.__dict__,
        }
