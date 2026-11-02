"""Project paths and persisted control calibration.

The values in this module are deliberately small and explicit so that the
controller can be run from any working directory.  Values that depend on a
particular AirSim vehicle belong in ``configs/thrust_calibration.json``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "logs"
DEFAULT_CALIBRATION_PATH = CONFIG_DIR / "thrust_calibration.json"


@dataclass
class ThrustCalibration:
    """Mapping from collective thrust acceleration to AirSim throttle.

    ``thrust_accel_per_throttle`` is an engineering slope around hover.  The
    only value that can be inferred robustly from a short AirSim experiment is
    the hover point; the slope remains configurable and is recorded as an
    assumption until measured on the chosen vehicle.
    """

    hover_throttle: float = 0.5
    hover_thrust_accel: float = 9.81
    thrust_accel_per_throttle: float = 19.62
    min_throttle: float = 0.0
    max_throttle: float = 1.0
    calibration_complete: bool = False
    vehicle_name: str = ""
    notes: str = "Fallback values; run hover_throttle_calibration.py."

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ThrustCalibration":
        fields = {
            "hover_throttle",
            "hover_thrust_accel",
            "thrust_accel_per_throttle",
            "min_throttle",
            "max_throttle",
            "calibration_complete",
            "vehicle_name",
            "notes",
        }
        return cls(**{key: value[key] for key in fields if key in value})

    def validate(self) -> None:
        if not 0.0 <= self.min_throttle < self.max_throttle <= 1.0:
            raise ValueError("Throttle limits must satisfy 0 <= min < max <= 1")
        if not self.min_throttle <= self.hover_throttle <= self.max_throttle:
            raise ValueError("hover_throttle must lie within throttle limits")
        if self.hover_thrust_accel <= 0.0:
            raise ValueError("hover_thrust_accel must be positive")
        if self.thrust_accel_per_throttle <= 0.0:
            raise ValueError("thrust_accel_per_throttle must be positive")


def load_calibration(path: str | Path | None = None) -> ThrustCalibration:
    """Load calibration JSON, falling back to explicit engineering defaults."""

    path = Path(path) if path is not None else DEFAULT_CALIBRATION_PATH
    if not path.exists():
        calibration = ThrustCalibration()
        calibration.validate()
        return calibration
    with path.open("r", encoding="utf-8") as handle:
        calibration = ThrustCalibration.from_mapping(json.load(handle))
    calibration.validate()
    return calibration


def save_calibration(calibration: ThrustCalibration, path: str | Path | None = None) -> Path:
    """Persist calibration and return the resolved output path."""

    calibration.validate()
    path = Path(path) if path is not None else DEFAULT_CALIBRATION_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(calibration), handle, indent=2)
        handle.write("\n")
    return path


def ensure_runtime_directories() -> None:
    """Create only the project-owned output directories used by the scripts."""

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
