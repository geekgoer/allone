"""CSV logger for reproducible MPC teacher input-output pairs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _flat(value: Any, size: int, name: str) -> list[float]:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.size != size:
        raise ValueError(f"{name} must have {size} values")
    return [float(item) for item in result]


class TeacherDataLogger:
    """One CSV file per drone, with stable column ordering."""

    def __init__(self, path: str | Path, drone_id: str) -> None:
        self.path = Path(path)
        self.drone_id = str(drone_id)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=self.fieldnames())
        self._writer.writeheader()
        self._handle.flush()

    @staticmethod
    def fieldnames() -> list[str]:
        return (
            ["timestamp", "monotonic_timestamp", "drone_id"]
            + [f"x{i}" for i in range(12)]
            + ["F", "wx", "wy", "wz"]
            + [f"raw_p_{axis}" for axis in "xyz"]
            + [f"raw_q_{axis}" for axis in "wxyz"]
            + [f"raw_v_{axis}" for axis in "xyz"]
            + [f"raw_a_{axis}" for axis in "xyz"]
            + [f"raw_w_{axis}" for axis in "xyz"]
            + [f"reference_p_{axis}" for axis in "xyz"]
            + [f"reference_q_{axis}" for axis in "wxyz"]
            + ["mpc_solve_time", "actual_loop_period", "deadline_miss"]
        )

    def write_sample(
        self,
        *,
        timestamp: float,
        monotonic_timestamp: float,
        state: Any,
        control: Any,
        raw_state: Any,
        reference: Any,
        mpc_solve_time: float,
        actual_loop_period: float,
        deadline_miss: bool,
    ) -> None:
        control = _flat(control, 4, "control")
        x = _flat(state, 12, "state")
        row: dict[str, Any] = {
            "timestamp": float(timestamp),
            "monotonic_timestamp": float(monotonic_timestamp),
            "drone_id": self.drone_id,
            **{f"x{i}": value for i, value in enumerate(x)},
            "F": control[0],
            "wx": control[1],
            "wy": control[2],
            "wz": control[3],
            "mpc_solve_time": float(mpc_solve_time),
            "actual_loop_period": float(actual_loop_period),
            "deadline_miss": int(bool(deadline_miss)),
        }
        for prefix, value, axes in (
            ("raw_p_", raw_state.position, "xyz"),
            ("raw_q_", raw_state.quaternion, "wxyz"),
            ("raw_v_", raw_state.velocity, "xyz"),
            ("raw_a_", raw_state.acceleration, "xyz"),
            ("raw_w_", raw_state.angular_velocity, "xyz"),
            ("reference_p_", reference.position, "xyz"),
            ("reference_q_", reference.orientation, "wxyz"),
        ):
            values = _flat(value, len(axes), prefix)
            row.update({f"{prefix}{axis}": value for axis, value in zip(axes, values)})
        self._writer.writerow(row)
        self._handle.flush()

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    def __enter__(self) -> "TeacherDataLogger":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
