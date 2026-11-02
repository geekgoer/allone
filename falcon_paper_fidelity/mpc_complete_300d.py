"""Deployable, MPC-complete 300-D feature contract and deterministic teacher.

The ACADO wrapper in :mod:`falcon_paper_fidelity.acado_mpc` solves one
trajectory-tracking problem from a 10-D current state, 21 reference states,
and the first 20 reference control vectors.  This module serializes precisely
those online-available quantities for a DNN.  It deliberately excludes every
MPC optimization result from the feature vector.

The feature is expressed in the same Z-up/FLU convention supplied to the
generated ACADO solver.  At deployment a state estimator and the existing
trajectory planner are sufficient to construct it; no ACADO object is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from airsim_interface import RawDroneState
from mpc_controller import MPCReference
from state_adapter import normalize_quaternion

from .acado_mpc import AcadoMPCConfig, AcadoRtiMPC


INPUT_CONTRACT = "falcon_mpc_complete_solver_window_300d_v1"
OUTPUT_CONTRACT = "falcon_acado_first_control_4d_v1"
HORIZON = 20
NODES = HORIZON + 1
STATE_DIMENSIONS = 10
REFERENCE_INPUT_DIMENSIONS = 4
INPUT_DIMENSIONS = STATE_DIMENSIONS + NODES * STATE_DIMENSIONS + HORIZON * REFERENCE_INPUT_DIMENSIONS
OUTPUT_DIMENSIONS = 4

INPUT_FEATURES = (
    tuple(f"state_{field}" for field in (
        "position_x", "position_y", "position_z", "quaternion_w", "quaternion_x", "quaternion_y", "quaternion_z", "velocity_x", "velocity_y", "velocity_z",
    ))
    + tuple(
        f"reference_{node:02d}_{field}"
        for node in range(NODES)
        for field in (
            "position_x", "position_y", "position_z", "quaternion_w", "quaternion_x", "quaternion_y", "quaternion_z", "velocity_x", "velocity_y", "velocity_z",
        )
    )
    + tuple(
        f"reference_control_{node:02d}_{field}"
        for node in range(HORIZON)
        for field in ("collective_thrust_acceleration", "body_rate_x", "body_rate_y", "body_rate_z")
    )
)
OUTPUT_FEATURES = (
    "collective_thrust_acceleration",
    "body_rate_x",
    "body_rate_y",
    "body_rate_z",
)


def _zup_flu_vector(value: Any) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(3)
    if not np.isfinite(vector).all():
        raise ValueError("vector must be finite")
    return vector * np.array([1.0, -1.0, -1.0], dtype=np.float64)


def _zup_flu_quaternion(value: Any) -> np.ndarray:
    quaternion = normalize_quaternion(value)
    return normalize_quaternion(quaternion * np.array([1.0, 1.0, -1.0, -1.0], dtype=np.float64))


def _copy_state_with_quaternion(state: RawDroneState, quaternion: np.ndarray) -> RawDroneState:
    return RawDroneState(
        position=state.position,
        quaternion=quaternion,
        velocity=state.velocity,
        acceleration=state.acceleration,
        angular_velocity=state.angular_velocity,
        timestamp_ns=state.timestamp_ns,
    )


def _copy_reference_with_quaternion(reference: MPCReference, quaternion: np.ndarray) -> MPCReference:
    return MPCReference(
        position=reference.position,
        orientation=quaternion,
        velocity=reference.velocity,
        acceleration=reference.acceleration,
        body_rates=reference.body_rates,
    )


def canonicalize_problem(
    state: RawDroneState,
    references: Sequence[MPCReference],
    *,
    horizon: int = HORIZON,
) -> tuple[RawDroneState, list[MPCReference]]:
    """Pad one reference window and remove equivalent quaternion sign choices.

    A quaternion and its negative describe the same orientation.  Without a
    fixed convention they create duplicate DNN inputs for the same physical
    MPC problem.  Reference signs are made continuous across the horizon, and
    the measured-state sign is aligned to the first reference.  The returned
    state/reference values are still AirSim NED/FRD values, ready for either
    feature construction or :class:`AcadoRtiMPC`.
    """

    if not isinstance(state, RawDroneState):
        raise TypeError("state must be RawDroneState")
    if horizon != HORIZON:
        raise ValueError(f"this contract requires horizon={HORIZON}")
    values = list(references)
    if not values or any(not isinstance(value, MPCReference) for value in values):
        raise TypeError("references must contain at least one MPCReference")
    values = values[: horizon + 1]
    values.extend([values[-1]] * (horizon + 1 - len(values)))

    canonical_references: list[MPCReference] = []
    previous: np.ndarray | None = None
    for reference in values:
        orientation = normalize_quaternion(reference.orientation)
        if previous is None:
            # Anchor the first node so every feature vector receives the same
            # sign convention even if a planner independently normalizes q.
            if orientation[0] < 0.0:
                orientation = -orientation
        elif float(np.dot(orientation, previous)) < 0.0:
            orientation = -orientation
        canonical_references.append(_copy_reference_with_quaternion(reference, orientation))
        previous = orientation

    state_orientation = normalize_quaternion(state.quaternion)
    if float(np.dot(state_orientation, canonical_references[0].orientation)) < 0.0:
        state_orientation = -state_orientation
    return _copy_state_with_quaternion(state, state_orientation), canonical_references


def reference_control(reference: MPCReference, *, gravity: float) -> np.ndarray:
    """Return the planner-provided ACADO reference input ``[F, wx, wy, wz]``.

    This is a feed-forward value derived from planned acceleration and planned
    body rates.  It is not an MPC optimization output and remains available
    after the MPC is removed from the online controller.
    """

    acceleration = _zup_flu_vector(reference.acceleration)
    thrust = np.linalg.norm(acceleration + np.array([0.0, 0.0, float(gravity)]))
    rates = _zup_flu_vector(reference.body_rates)
    value = np.concatenate(([thrust], rates)).astype(np.float64, copy=False)
    if value.shape != (4,) or not np.isfinite(value).all():
        raise ValueError("reference control must be one finite 4-D vector")
    return value


def build_mpc_complete_300d(
    state: RawDroneState,
    references: Sequence[MPCReference],
    *,
    gravity: float = 9.8066,
) -> np.ndarray:
    """Serialize all varying, online-available ACADO problem inputs.

    Layout: current 10-D state, 21 x 10-D reference states, and first 20 x
    4-D reference inputs.  The terminal reference input is omitted because the
    generated solver only reads reference inputs for stages 0 through 19.
    """

    state, values = canonicalize_problem(state, references)
    current = np.concatenate(
        (_zup_flu_vector(state.position), _zup_flu_quaternion(state.quaternion), _zup_flu_vector(state.velocity))
    )
    states = np.asarray(
        [
            np.concatenate(
                (_zup_flu_vector(reference.position), _zup_flu_quaternion(reference.orientation), _zup_flu_vector(reference.velocity))
            )
            for reference in values
        ],
        dtype=np.float64,
    )
    controls = np.asarray([reference_control(reference, gravity=gravity) for reference in values[:HORIZON]], dtype=np.float64)
    feature = np.concatenate((current, states.reshape(-1), controls.reshape(-1))).astype(np.float64, copy=False)
    if feature.shape != (INPUT_DIMENSIONS,) or not np.isfinite(feature).all():
        raise ValueError(f"complete MPC feature must be one finite {INPUT_DIMENSIONS}-D vector")
    return feature


@dataclass(frozen=True)
class DeterministicTeacherConfig:
    """Fixed label-generation procedure for one ACADO trajectory problem."""

    minimum_refinement_steps: int = 4
    maximum_refinement_steps: int = 16
    convergence_tolerance: float = 1e-3

    def validate(self) -> None:
        if self.minimum_refinement_steps < 2:
            raise ValueError("minimum_refinement_steps must be at least two")
        if self.maximum_refinement_steps < self.minimum_refinement_steps:
            raise ValueError("maximum_refinement_steps must not be smaller than minimum_refinement_steps")
        if self.convergence_tolerance <= 0.0:
            raise ValueError("convergence_tolerance must be positive")


@dataclass(frozen=True)
class DeterministicTeacherResult:
    feature: np.ndarray
    control: np.ndarray
    iterations: int
    final_control_delta: float
    converged: bool
    diagnostics: Mapping[str, float]


class DeterministicAcadoTeacher:
    """Repeatedly solve a reset ACADO problem to make labels reproducible.

    Each query resets the solver once, then performs a fixed number of RTI
    iterations on exactly the same state and reference window.  This defines a
    history-free teacher mapping suitable for supervised learning.
    """

    def __init__(
        self,
        mpc_config: AcadoMPCConfig | None = None,
        teacher_config: DeterministicTeacherConfig | None = None,
    ) -> None:
        self.mpc_config = mpc_config or AcadoMPCConfig()
        self.mpc_config.validate()
        self.teacher_config = teacher_config or DeterministicTeacherConfig()
        self.teacher_config.validate()
        self.controller = AcadoRtiMPC(self.mpc_config)

    def query(self, state: RawDroneState, references: Sequence[MPCReference]) -> DeterministicTeacherResult:
        canonical_state, canonical_references = canonicalize_problem(
            state, references, horizon=self.mpc_config.horizon
        )
        feature = build_mpc_complete_300d(
            canonical_state, canonical_references, gravity=self.mpc_config.gravity
        )
        self.controller.reset()
        previous: np.ndarray | None = None
        control = np.empty(OUTPUT_DIMENSIONS, dtype=np.float64)
        final_delta = float("inf")
        iterations = 0
        for index in range(self.teacher_config.maximum_refinement_steps):
            control = np.asarray(self.controller.solve(canonical_state, canonical_references), dtype=np.float64)
            iterations = index + 1
            if previous is not None:
                final_delta = float(np.max(np.abs(control - previous)))
            previous = control.copy()
            if (
                iterations >= self.teacher_config.minimum_refinement_steps
                and final_delta <= self.teacher_config.convergence_tolerance
            ):
                break
        return DeterministicTeacherResult(
            feature=feature.astype(np.float32),
            control=control.astype(np.float32),
            iterations=iterations,
            final_control_delta=final_delta,
            converged=bool(final_delta <= self.teacher_config.convergence_tolerance),
            diagnostics=dict(self.controller.last_diagnostics),
        )

    def reproducibility_check(
        self, state: RawDroneState, references: Sequence[MPCReference]
    ) -> float:
        """Return max absolute output difference from two independent queries."""

        first = self.query(state, references)
        second = self.query(state, references)
        return float(np.max(np.abs(first.control - second.control)))

    def signature(self) -> dict[str, object]:
        return {
            "input_contract": INPUT_CONTRACT,
            "input_dimensions": INPUT_DIMENSIONS,
            "output_contract": OUTPUT_CONTRACT,
            "output_dimensions": OUTPUT_DIMENSIONS,
            "mpc": self.controller.signature(),
            "teacher": {
                "minimum_refinement_steps": self.teacher_config.minimum_refinement_steps,
                "maximum_refinement_steps": self.teacher_config.maximum_refinement_steps,
                "convergence_tolerance": self.teacher_config.convergence_tolerance,
                "reset_before_each_query": True,
            },
        }
