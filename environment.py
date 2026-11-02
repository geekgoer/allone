"""Paper-compatible Gate environment generation.

The FALCON paper specifies the five x coordinates and the epsilon metric, but
does not publish a unique algorithm for reconstructing the original y/z
coordinates.  This module therefore generates deterministic
paper-compatible environments, not the authors' exact gate layouts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


SUPPORTED_EPSILONS = (0.0, 0.8, 1.6)


def _validate_epsilon(epsilon: float) -> float:
    value = float(epsilon)
    if not any(np.isclose(value, candidate) for candidate in SUPPORTED_EPSILONS):
        raise ValueError(f"epsilon must be one of {SUPPORTED_EPSILONS}")
    return value


@dataclass(frozen=True)
class Gate:
    index: int
    center: np.ndarray
    orientation: np.ndarray

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=np.float64).reshape(-1)
        orientation = np.asarray(self.orientation, dtype=np.float64).reshape(-1)
        if center.size != 3 or orientation.size != 4:
            raise ValueError("Gate center must be (3,) and orientation must be (4,)")
        norm = np.linalg.norm(orientation)
        if norm < 1e-12:
            raise ValueError("Gate orientation quaternion cannot be zero")
        object.__setattr__(self, "center", center.copy())
        object.__setattr__(self, "orientation", orientation / norm)


def generate_gate_sequence(
    epsilon: float,
    seed: int | None = None,
    base_y: float = 0.0,
    base_z: float = -2.0,
    gate_count: int = 5,
    x_start: float = 10.0,
    x_spacing: float = 10.0,
    max_z: float | None = None,
) -> list[Gate]:
    """Generate a configurable gate sequence with epsilon y/z steps.

    ``max_z`` is an optional NED safety bound.  By default it equals
    ``base_z``, so generated gates never move downward toward the ground.
    """

    epsilon = _validate_epsilon(epsilon)
    if gate_count < 1:
        raise ValueError("gate_count must be at least 1")
    if x_spacing <= 0.0:
        raise ValueError("x_spacing must be positive")
    max_z = float(base_z if max_z is None else max_z)
    if float(base_z) > max_z:
        raise ValueError("base_z must not be below max_z in NED coordinates")
    rng = np.random.default_rng(seed)
    y, z = float(base_y), float(base_z)
    gates: list[Gate] = []
    # Each step changes exactly one of y or z by +/- epsilon.  Therefore the
    # requested Manhattan displacement is exact up to floating-point roundoff.
    directions = np.array(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]], dtype=np.float64
    )
    for index in range(1, gate_count + 1):
        if index > 1 and epsilon > 0.0:
            allowed = directions[
                [z + epsilon * direction[1] <= max_z + 1e-12 for direction in directions]
            ]
            direction = allowed[int(rng.integers(0, len(allowed)))]
            y += epsilon * direction[0]
            z += epsilon * direction[1]
        gates.append(
            Gate(
                index=index,
                center=np.array(
                    [x_start + x_spacing * (index - 1), y, z], dtype=np.float64
                ),
                orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            )
        )
    return gates


class GateEnvironment:
    """Configuration object for a five-gate paper-compatible scenario."""

    def __init__(
        self,
        epsilon: float = 0.8,
        seed: int | None = None,
        base_y: float = 0.0,
        base_z: float = -2.0,
        gate_count: int = 5,
        x_start: float = 10.0,
        x_spacing: float = 10.0,
        max_z: float | None = None,
    ) -> None:
        self.epsilon = _validate_epsilon(epsilon)
        self.seed = seed
        self.base_y = float(base_y)
        self.base_z = float(base_z)
        self.gate_count = int(gate_count)
        self.x_start = float(x_start)
        self.x_spacing = float(x_spacing)
        self.max_z = float(self.base_z if max_z is None else max_z)
        if self.gate_count < 1:
            raise ValueError("gate_count must be at least 1")
        if self.x_spacing <= 0.0:
            raise ValueError("x_spacing must be positive")
        if self.base_z > self.max_z:
            raise ValueError("base_z must not be below max_z in NED coordinates")

    def generate(self) -> list[Gate]:
        return generate_gate_sequence(
            self.epsilon,
            self.seed,
            self.base_y,
            self.base_z,
            gate_count=self.gate_count,
            x_start=self.x_start,
            x_spacing=self.x_spacing,
            max_z=self.max_z,
        )


def generate_gate_sequences(
    vehicle_names: Iterable[str],
    epsilon: float = 0.8,
    seed: int | None = None,
    group_spacing: float = 6.0,
    base_z: float = -2.0,
    gate_count: int = 5,
    x_start: float = 10.0,
    x_spacing: float = 10.0,
    max_z: float | None = None,
) -> dict[str, list[Gate]]:
    """Generate one independent gate sequence for each named drone.

    The groups are translated along y by ``group_spacing`` and use distinct
    random-generator seeds.  Thus each returned list owns its own Gate
    objects and can be changed or flown independently of the other lists.
    """

    names = tuple(str(name) for name in vehicle_names)
    if not names or any(not name for name in names):
        raise ValueError("vehicle_names must contain non-empty names")
    if len(set(names)) != len(names):
        raise ValueError("vehicle_names must not contain duplicates")
    if group_spacing <= 0.0:
        raise ValueError("group_spacing must be positive")
    center_index = (len(names) - 1) / 2.0
    # A sequence can move in y by at most (gate_count - 1) * epsilon.  Add
    # that worst-case excursion on both sides so ``group_spacing`` remains a
    # lower bound on the separation of corresponding gate centers.
    y_excursion = max(0, int(gate_count) - 1) * float(epsilon)
    effective_group_spacing = float(group_spacing) + 2.0 * y_excursion
    sequences: dict[str, list[Gate]] = {}
    for index, name in enumerate(names):
        sequence_seed = None if seed is None else int(seed) + 1009 * index
        base_y = (index - center_index) * effective_group_spacing
        sequences[name] = generate_gate_sequence(
            epsilon,
            sequence_seed,
            base_y=base_y,
            base_z=base_z,
            gate_count=gate_count,
            x_start=x_start,
            x_spacing=x_spacing,
            max_z=max_z,
        )
    return sequences


def gate_manhattan_displacements(gates: list[Gate]) -> list[float]:
    return [
        float(np.abs(gates[index].center[1] - gates[index - 1].center[1])
              + np.abs(gates[index].center[2] - gates[index - 1].center[2]))
        for index in range(1, len(gates))
    ]


def distance_to_gate(position: Any, gate: Gate) -> float:
    value = np.asarray(position, dtype=np.float64).reshape(-1)
    if value.size != 3:
        raise ValueError("position must have shape (3,)")
    return float(np.linalg.norm(value - gate.center))
