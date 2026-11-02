"""Python adapter for the official rpg_mpc ACADO/qpOASES generated solver.

The upstream solver uses a z-up world/body convention.  AirSim exposes NED
world coordinates and FRD body rates, so this module performs the proper
180-degree x-axis frame change on states, references, and rate commands.
"""

from __future__ import annotations

import ctypes
from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from airsim_interface import RawDroneState
from mpc_controller import MPCReference
from state_adapter import normalize_quaternion


_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_LIBRARY = (
    _ROOT / "native" / "acado_rpg_mpc" / "build" / "bin" / "libfalcon_acado_mpc.dll"
)
_DOUBLE_POINTER = ctypes.POINTER(ctypes.c_double)


@dataclass(frozen=True)
class AcadoMPCConfig:
    library: str = str(_DEFAULT_LIBRARY)
    horizon: int = 20
    prediction_dt: float = 0.1
    gravity: float = 9.8066
    q_position_xy: float = 200.0
    q_position_z: float = 500.0
    q_attitude: float = 50.0
    q_velocity: float = 10.0
    q_perception: float = 0.0
    r_thrust: float = 1.0
    r_pitch_roll: float = 1.0
    r_yaw: float = 1.0
    min_thrust: float = 2.0
    max_thrust: float = 20.0
    max_body_rate_xy: float = 3.0
    max_body_rate_z: float = 2.0

    def validate(self) -> None:
        if self.horizon != 20:
            raise ValueError("the checked-in rpg_mpc generated solver requires horizon=20")
        if not np.isclose(self.prediction_dt, 0.1):
            raise ValueError("the checked-in rpg_mpc solver has a fixed 0.1 s prediction step")
        if self.min_thrust < 0.0 or self.max_thrust <= self.min_thrust:
            raise ValueError("invalid thrust limits")
        if min(self.max_body_rate_xy, self.max_body_rate_z) <= 0.0:
            raise ValueError("invalid body-rate limits")


def _ned_frd_vector_to_zup_flu(value: Any) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(3)
    return vector * np.array([1.0, -1.0, -1.0], dtype=np.float64)


def _ned_frd_quaternion_to_zup_flu(value: Any) -> np.ndarray:
    quaternion = normalize_quaternion(value)
    converted = quaternion * np.array([1.0, 1.0, -1.0, -1.0], dtype=np.float64)
    return normalize_quaternion(converted)


def _state_vector(state: RawDroneState) -> np.ndarray:
    return np.ascontiguousarray(
        np.concatenate(
            (
                _ned_frd_vector_to_zup_flu(state.position),
                _ned_frd_quaternion_to_zup_flu(state.quaternion),
                _ned_frd_vector_to_zup_flu(state.velocity),
            )
        ),
        dtype=np.float64,
    )


def _reference_state(reference: MPCReference) -> np.ndarray:
    return np.concatenate(
        (
            _ned_frd_vector_to_zup_flu(reference.position),
            _ned_frd_quaternion_to_zup_flu(reference.orientation),
            _ned_frd_vector_to_zup_flu(reference.velocity),
        )
    )


class AcadoRtiMPC:
    """N=20 ACADO RTI controller with output ``[F, wx, wy, wz]`` in AirSim axes."""

    def __init__(self, config: AcadoMPCConfig | None = None) -> None:
        self.config = config or AcadoMPCConfig()
        self.config.validate()
        library_path = Path(self.config.library).expanduser().resolve()
        if not library_path.exists():
            raise FileNotFoundError(
                f"ACADO solver library not found: {library_path}. "
                "Build it with cmake before collecting data."
            )
        self.library_path = library_path
        # Python 3.8+ uses the secure Windows DLL search path.  Register the
        # solver directory explicitly so the bundled libwinpthread dependency
        # can be resolved even when MinGW is absent from PATH.
        self._dll_directory_handle = None
        if os.name == "nt" and hasattr(os, "add_dll_directory"):
            self._dll_directory_handle = os.add_dll_directory(str(library_path.parent))
        self._dll = ctypes.CDLL(str(library_path))
        self._bind()
        status = int(self._dll.falcon_acado_initialize())
        self._raise_status(status, "initialize")
        state_weights = np.ascontiguousarray(
            [
                self.config.q_position_xy,
                self.config.q_position_z,
                self.config.q_attitude,
                self.config.q_velocity,
                self.config.q_perception,
            ],
            dtype=np.float64,
        )
        input_weights = np.ascontiguousarray(
            [self.config.r_thrust, self.config.r_pitch_roll, self.config.r_yaw],
            dtype=np.float64,
        )
        limits = np.ascontiguousarray(
            [
                self.config.min_thrust,
                self.config.max_thrust,
                self.config.max_body_rate_xy,
                self.config.max_body_rate_z,
            ],
            dtype=np.float64,
        )
        status = int(
            self._dll.falcon_acado_configure(
                state_weights.ctypes.data_as(_DOUBLE_POINTER),
                input_weights.ctypes.data_as(_DOUBLE_POINTER),
                limits.ctypes.data_as(_DOUBLE_POINTER),
            )
        )
        self._raise_status(status, "configure")
        compiled_dt = float(self._dll.falcon_acado_prediction_dt())
        if not np.isclose(compiled_dt, self.config.prediction_dt):
            raise RuntimeError(
                f"ACADO DLL prediction dt {compiled_dt} disagrees with config "
                f"{self.config.prediction_dt}"
            )
        self._reset_pending = True
        self.last_diagnostics: dict[str, float] = {}

    def _bind(self) -> None:
        self._dll.falcon_acado_prediction_dt.restype = ctypes.c_double
        self._dll.falcon_acado_initialize.restype = ctypes.c_int
        self._dll.falcon_acado_configure.argtypes = [
            _DOUBLE_POINTER,
            _DOUBLE_POINTER,
            _DOUBLE_POINTER,
        ]
        self._dll.falcon_acado_configure.restype = ctypes.c_int
        self._dll.falcon_acado_reset.argtypes = [_DOUBLE_POINTER]
        self._dll.falcon_acado_reset.restype = ctypes.c_int
        self._dll.falcon_acado_solve.argtypes = [
            _DOUBLE_POINTER,
            _DOUBLE_POINTER,
            _DOUBLE_POINTER,
            _DOUBLE_POINTER,
            _DOUBLE_POINTER,
        ]
        self._dll.falcon_acado_solve.restype = ctypes.c_int
        self._dll.falcon_acado_status_string.argtypes = [ctypes.c_int]
        self._dll.falcon_acado_status_string.restype = ctypes.c_char_p

    def _raise_status(self, status: int, operation: str) -> None:
        if status == 0:
            return
        message = self._dll.falcon_acado_status_string(status)
        detail = message.decode("utf-8", errors="replace") if message else "unknown error"
        raise RuntimeError(f"ACADO {operation} failed with status {status}: {detail}")

    def solve(self, state: RawDroneState, references: Sequence[MPCReference]) -> np.ndarray:
        if not isinstance(state, RawDroneState):
            raise TypeError("state must be RawDroneState")
        values = list(references)
        if not values:
            raise ValueError("references must not be empty")
        values = values[: self.config.horizon + 1]
        values.extend([values[-1]] * (self.config.horizon + 1 - len(values)))
        state_vector = _state_vector(state)
        reference_states = np.ascontiguousarray(
            np.asarray([_reference_state(item) for item in values], dtype=np.float64).reshape(-1)
        )
        reference_inputs = np.zeros((self.config.horizon + 1, 4), dtype=np.float64)
        for index, reference in enumerate(values):
            acceleration = _ned_frd_vector_to_zup_flu(reference.acceleration)
            reference_inputs[index, 0] = np.linalg.norm(
                acceleration + np.array([0.0, 0.0, self.config.gravity])
            )
            reference_inputs[index, 1:4] = _ned_frd_vector_to_zup_flu(
                reference.body_rates
            )
        reference_inputs = np.ascontiguousarray(reference_inputs.reshape(-1))
        if self._reset_pending:
            status = int(
                self._dll.falcon_acado_reset(state_vector.ctypes.data_as(_DOUBLE_POINTER))
            )
            self._raise_status(status, "reset")
            self._reset_pending = False
        output = np.empty(4, dtype=np.float64)
        diagnostics = np.empty(3, dtype=np.float64)
        status = int(
            self._dll.falcon_acado_solve(
                state_vector.ctypes.data_as(_DOUBLE_POINTER),
                reference_states.ctypes.data_as(_DOUBLE_POINTER),
                reference_inputs.ctypes.data_as(_DOUBLE_POINTER),
                output.ctypes.data_as(_DOUBLE_POINTER),
                diagnostics.ctypes.data_as(_DOUBLE_POINTER),
            )
        )
        self._raise_status(status, "solve")
        # Convert body rates from the solver's FLU frame back to AirSim FRD.
        output[2] = -output[2]
        output[3] = -output[3]
        if not np.isfinite(output).all():
            raise RuntimeError("ACADO returned NaN or Inf")
        self.last_diagnostics = {
            "kkt": float(diagnostics[0]),
            "objective": float(diagnostics[1]),
            "qp_nwsr": float(diagnostics[2]),
        }
        return output

    def reset(self) -> None:
        self._reset_pending = True

    def signature(self) -> dict[str, Any]:
        return {
            "backend": "official_rpg_mpc_acado_qpoases_generated",
            "fidelity": "official generated solver; prediction_dt ambiguity recorded",
            **asdict(self.config),
        }
