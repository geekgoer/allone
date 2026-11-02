"""Deterministic initial plant-state generation for teacher collection."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _range_pair(value: Sequence[float], name: str) -> tuple[float, float]:
    values = np.asarray(value, dtype=np.float64).reshape(-1)
    if values.size != 2 or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain two finite values")
    low, high = float(values[0]), float(values[1])
    if low > high:
        raise ValueError(f"{name} lower bound must not exceed upper bound")
    return low, high


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert XYZ Euler angles to a [w, x, y, z] unit quaternion."""

    cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
    cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
    cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class InitialState:
    """One reproducible initial state in AirSim's NED world frame."""

    initial_state_id: int
    position: np.ndarray
    orientation: np.ndarray
    velocity: np.ndarray
    angular_velocity: np.ndarray

    def __post_init__(self) -> None:
        position = np.asarray(self.position, dtype=np.float64).reshape(-1)
        orientation = np.asarray(self.orientation, dtype=np.float64).reshape(-1)
        velocity = np.asarray(self.velocity, dtype=np.float64).reshape(-1)
        angular_velocity = np.asarray(self.angular_velocity, dtype=np.float64).reshape(-1)
        if position.size != 3 or velocity.size != 3 or angular_velocity.size != 3:
            raise ValueError("position, velocity and angular_velocity must have shape (3,)")
        if orientation.size != 4:
            raise ValueError("orientation must have shape (4,)")
        if not all(np.all(np.isfinite(value)) for value in (position, orientation, velocity, angular_velocity)):
            raise ValueError("initial state contains a non-finite value")
        norm = np.linalg.norm(orientation)
        if norm < 1e-12:
            raise ValueError("orientation quaternion cannot be zero")
        object.__setattr__(self, "position", position.copy())
        object.__setattr__(self, "orientation", (orientation / norm).copy())
        object.__setattr__(self, "velocity", velocity.copy())
        object.__setattr__(self, "angular_velocity", angular_velocity.copy())

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_state_id": int(self.initial_state_id),
            "position": self.position.tolist(),
            "orientation": self.orientation.tolist(),
            "velocity": self.velocity.tolist(),
            "angular_velocity": self.angular_velocity.tolist(),
        }


class InitialStateGenerator:
    """Generate a deterministic set of conservative random initial states.

    The paper specifies ten random initial plant states but not their ranges.
    All ranges are therefore explicit engineering assumptions supplied by the
    caller/configuration rather than paper claims.
    """

    def __init__(
        self,
        count: int = 10,
        seed: int = 0,
        *,
        position_range: Mapping[str, Sequence[float]] | None = None,
        attitude_range: Mapping[str, Sequence[float]] | None = None,
        velocity_range: Mapping[str, Sequence[float]] | None = None,
        angular_velocity_range: Mapping[str, Sequence[float]] | None = None,
    ) -> None:
        if count < 1:
            raise ValueError("count must be at least 1")
        self.count = int(count)
        self.seed = int(seed)
        self.position_range = self._mapping_ranges(
            position_range or {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-2.2, -1.8)},
            "position_range",
        )
        self.attitude_range = self._mapping_ranges(
            attitude_range or {
                "roll": (-0.08, 0.08),
                "pitch": (-0.08, 0.08),
                "yaw": (-0.2, 0.2),
            },
            "attitude_range",
        )
        self.velocity_range = self._mapping_ranges(
            velocity_range or {axis: (0.0, 0.0) for axis in "xyz"},
            "velocity_range",
        )
        self.angular_velocity_range = self._mapping_ranges(
            angular_velocity_range or {axis: (0.0, 0.0) for axis in "xyz"},
            "angular_velocity_range",
        )

    @staticmethod
    def _mapping_ranges(
        value: Mapping[str, Sequence[float]], name: str
    ) -> dict[str, tuple[float, float]]:
        result = {str(key): _range_pair(pair, f"{name}.{key}") for key, pair in value.items()}
        return result

    def _sample_vector(
        self,
        rng: np.random.Generator,
        ranges: Mapping[str, tuple[float, float]],
        axes: Sequence[str],
    ) -> np.ndarray:
        missing = [axis for axis in axes if axis not in ranges]
        if missing:
            raise ValueError(f"missing {axes} range(s): {missing}")
        return np.array([rng.uniform(*ranges[axis]) for axis in axes], dtype=np.float64)

    def generate(self) -> list[InitialState]:
        rng = np.random.default_rng(self.seed)
        states: list[InitialState] = []
        for state_id in range(self.count):
            position = self._sample_vector(rng, self.position_range, "xyz")
            angles = self._sample_vector(rng, self.attitude_range, ("roll", "pitch", "yaw"))
            velocity = self._sample_vector(rng, self.velocity_range, "xyz")
            angular_velocity = self._sample_vector(rng, self.angular_velocity_range, "xyz")
            states.append(
                InitialState(
                    initial_state_id=state_id,
                    position=position,
                    orientation=euler_to_quaternion(*angles),
                    velocity=velocity,
                    angular_velocity=angular_velocity,
                )
            )
        return states


def save_initial_states(states: Sequence[InitialState], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = [state.to_dict() for state in states]
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output


def load_initial_states(path: str | Path) -> list[InitialState]:
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        InitialState(
            initial_state_id=int(item["initial_state_id"]),
            position=item["position"],
            orientation=item["orientation"],
            velocity=item["velocity"],
            angular_velocity=item["angular_velocity"],
        )
        for item in values
    ]
