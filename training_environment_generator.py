"""Generate reproducible, diverse paper-compatible training environments."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from environment import Gate, SUPPORTED_EPSILONS, generate_gate_sequence, gate_manhattan_displacements


@dataclass(frozen=True)
class TrainingEnvironment:
    environment_id: int
    seed: int
    epsilon: float
    difficulty: str
    gates: tuple[Gate, ...]
    # The route/controller parameters are part of the environment identity.
    # Keeping them here makes a resumed dataset self-describing and prevents
    # downstream code from silently assuming that every episode used 2 m/s.
    reference_speed: float = 2.0
    x_start: float = 10.0
    x_spacing: float = 10.0
    base_y: float | None = None
    base_z: float | None = None

    @property
    def adjacent_yz_displacement(self) -> tuple[float, ...]:
        return tuple(gate_manhattan_displacements(list(self.gates)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "environment_id": int(self.environment_id),
            "seed": int(self.seed),
            "epsilon": float(self.epsilon),
            "difficulty": self.difficulty,
            "reference_speed": float(self.reference_speed),
            "x_start": float(self.x_start),
            "x_spacing": float(self.x_spacing),
            "base_y": None if self.base_y is None else float(self.base_y),
            "base_z": None if self.base_z is None else float(self.base_z),
            "gates": [
                {
                    "index": int(gate.index),
                    "center": gate.center.tolist(),
                    "orientation": gate.orientation.tolist(),
                }
                for gate in self.gates
            ],
            "adjacent_yz_displacement": list(self.adjacent_yz_displacement),
        }


def _difficulty(epsilon: float) -> str:
    if epsilon <= 0.0:
        return "easy"
    if epsilon <= 0.8:
        return "medium"
    return "hard"


def generate_training_environments(
    count: int = 40,
    seed: int = 0,
    *,
    epsilon_values: Sequence[float] = SUPPORTED_EPSILONS,
    gate_count: int = 5,
    x_start: float = 10.0,
    x_spacing: float = 10.0,
    base_y_range: Sequence[float] = (-1.0, 1.0),
    base_z_range: Sequence[float] = (-3.0, -2.2),
    max_z: float = -2.0,
    speed_values: Sequence[float] = (2.0,),
    x_spacing_values: Sequence[float] | None = None,
    x_start_values: Sequence[float] | None = None,
    x_spacing_overrides: Mapping[int | str, float] | None = None,
) -> list[TrainingEnvironment]:
    """Create ``count`` different layouts with deterministic random seeds.

    The paper specifies 40 environmental conditions but not the concrete
    layout generator.  Each layout therefore records its engineering seed,
    base offsets, epsilon, gates and measured difficulty statistics.
    """

    if count < 1:
        raise ValueError("count must be at least 1")
    if gate_count < 1 or x_spacing <= 0.0:
        raise ValueError("gate_count must be positive and x_spacing must be positive")
    epsilons = tuple(float(value) for value in epsilon_values)
    if not epsilons or any(not any(np.isclose(value, supported) for supported in SUPPORTED_EPSILONS) for value in epsilons):
        raise ValueError(f"epsilon_values must use values from {SUPPORTED_EPSILONS}")
    speeds = tuple(float(value) for value in speed_values)
    if not speeds or any(not np.isfinite(value) or value <= 0.0 for value in speeds):
        raise ValueError("speed_values must contain positive finite values")
    spacings = tuple(
        float(value) for value in (x_spacing_values if x_spacing_values is not None else (x_spacing,))
    )
    starts = tuple(
        float(value) for value in (x_start_values if x_start_values is not None else (x_start,))
    )
    if not spacings or any(not np.isfinite(value) or value <= 0.0 for value in spacings):
        raise ValueError("x_spacing_values must contain positive finite values")
    if not starts or any(not np.isfinite(value) for value in starts):
        raise ValueError("x_start_values must contain finite values")
    spacing_overrides = {
        int(key): float(value) for key, value in (x_spacing_overrides or {}).items()
    }
    if any(key < 0 or key >= count for key in spacing_overrides):
        raise ValueError("x_spacing_overrides contains an environment ID outside the requested count")
    if any(not np.isfinite(value) or value <= 0.0 for value in spacing_overrides.values()):
        raise ValueError("x_spacing_overrides must contain positive finite values")
    y_low, y_high = float(base_y_range[0]), float(base_y_range[1])
    z_low, z_high = float(base_z_range[0]), float(base_z_range[1])
    if y_low > y_high or z_low > z_high or z_high > max_z:
        raise ValueError("base ranges must be ordered and base_z_range must stay at/below max_z")
    rng = np.random.default_rng(seed)
    environments: list[TrainingEnvironment] = []
    for environment_id in range(count):
        environment_seed = int(seed) + 7919 * environment_id
        epsilon = epsilons[environment_id % len(epsilons)]
        reference_speed = speeds[environment_id % len(speeds)]
        environment_x_spacing = spacing_overrides.get(
            environment_id,
            spacings[environment_id % len(spacings)],
        )
        environment_x_start = starts[environment_id % len(starts)]
        # Small random offsets make even epsilon=0 layouts distinct while
        # keeping the first target close enough for the conservative MPC.
        base_y = float(rng.uniform(y_low, y_high))
        base_z = float(rng.uniform(z_low, z_high))
        gates = tuple(
            generate_gate_sequence(
                epsilon,
                seed=environment_seed,
                base_y=base_y,
                base_z=base_z,
                gate_count=gate_count,
                x_start=environment_x_start,
                x_spacing=environment_x_spacing,
                max_z=max_z,
            )
        )
        environments.append(
            TrainingEnvironment(
                environment_id=environment_id,
                seed=environment_seed,
                epsilon=float(epsilon),
                difficulty=_difficulty(float(epsilon)),
                gates=gates,
                reference_speed=reference_speed,
                x_start=environment_x_start,
                x_spacing=environment_x_spacing,
                base_y=base_y,
                base_z=base_z,
            )
        )
    return environments


def save_training_environments(environments: Sequence[TrainingEnvironment], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps([environment.to_dict() for environment in environments], indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def load_training_environments(path: str | Path) -> list[TrainingEnvironment]:
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    environments: list[TrainingEnvironment] = []
    for item in values:
        gates = tuple(
            Gate(
                index=int(gate["index"]),
                center=gate["center"],
                orientation=gate["orientation"],
            )
            for gate in item["gates"]
        )
        environments.append(
            TrainingEnvironment(
                environment_id=int(item["environment_id"]),
                seed=int(item["seed"]),
                epsilon=float(item["epsilon"]),
                difficulty=str(item["difficulty"]),
                gates=gates,
                reference_speed=float(item.get("reference_speed", 2.0)),
                x_start=float(item.get("x_start", gates[0].center[0] if gates else 10.0)),
                x_spacing=float(
                    item.get(
                        "x_spacing",
                        gates[1].center[0] - gates[0].center[0] if len(gates) > 1 else 10.0,
                    )
                ),
                base_y=None if item.get("base_y") is None else float(item["base_y"]),
                base_z=None if item.get("base_z") is None else float(item["base_z"]),
            )
        )
    return environments
