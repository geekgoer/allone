"""Shared model, normalization and framework-neutral export helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


STATE_INPUT_FEATURES = (
    "position_error_x", "position_error_y", "position_error_z",
    "attitude_error_x", "attitude_error_y", "attitude_error_z",
    "velocity_x", "velocity_y", "velocity_z",
    "acceleration_x", "acceleration_y", "acceleration_z",
)
PATH_CONTEXT_OFFSETS = (1, 5, 10, 19)
PATH_CONTEXT_FEATURES = tuple(
    f"path_offset_{offset}_{axis}"
    for offset in PATH_CONTEXT_OFFSETS
    for axis in ("x", "y", "z")
)
PATH_AWARE_INPUT_FEATURES = STATE_INPUT_FEATURES + PATH_CONTEXT_FEATURES + (
    "reference_speed",
)
# Backward-compatible name used by older reports and imports.
INPUT_FEATURES = STATE_INPUT_FEATURES
OUTPUT_FEATURES = (
    "collective_thrust_acceleration", "body_rate_x", "body_rate_y", "body_rate_z",
)


def input_feature_names(input_dim: int) -> tuple[str, ...]:
    if input_dim == len(STATE_INPUT_FEATURES):
        return STATE_INPUT_FEATURES
    if input_dim == len(PATH_AWARE_INPUT_FEATURES):
        return PATH_AWARE_INPUT_FEATURES
    return tuple(f"input_{index}" for index in range(input_dim))


def build_policy_input(
    state: Any,
    reference_window: Any,
    *,
    input_dim: int,
    reference_speed: float | None = None,
    context_offsets: tuple[int, ...] = PATH_CONTEXT_OFFSETS,
) -> np.ndarray:
    """Compose the legacy 12-D or path-aware 25-D controller input.

    The path-aware representation contains the original state, four future
    reference positions relative to the current vehicle position, and the
    scalar reference speed.  This gives the student the same future-path
    information used by the MPC teacher.
    """

    state_value = np.asarray(state, dtype=np.float64).reshape(-1)
    if state_value.size != len(STATE_INPUT_FEATURES):
        raise ValueError("controller state must have shape (12,)")
    if input_dim == len(STATE_INPUT_FEATURES):
        return state_value.copy()
    if input_dim != len(PATH_AWARE_INPUT_FEATURES):
        raise ValueError(f"unsupported controller input dimension: {input_dim}")
    references = list(reference_window)
    if not references:
        raise ValueError("reference_window must contain at least one item")
    if len(context_offsets) * 3 + len(STATE_INPUT_FEATURES) + 1 != input_dim:
        raise ValueError("context offsets do not match the requested input dimension")
    first_position = np.asarray(references[0].position, dtype=np.float64).reshape(3)
    current_position = first_position - state_value[0:3]
    context = np.asarray(
        [
            np.asarray(references[min(offset, len(references) - 1)].position, dtype=np.float64).reshape(3)
            - current_position
            for offset in context_offsets
        ],
        dtype=np.float64,
    )
    if reference_speed is None:
        speeds = [
            float(np.linalg.norm(np.asarray(item.velocity, dtype=np.float64)))
            for item in references
        ]
        reference_speed = max(speeds, default=0.0)
    result = np.concatenate((state_value, context.reshape(-1), [float(reference_speed)]))
    if result.shape != (input_dim,) or not np.isfinite(result).all():
        raise ValueError("constructed controller input is invalid")
    return result


def build_model(torch: Any, *, depth: int, width: int, input_dim: int = 12) -> Any:
    """Build the FC/ReLU model with stable initialization for deep ReLU stacks.

    PyTorch ``Linear`` defaults attenuate variance through long plain-ReLU
    networks.  That is especially damaging for the 22- and 34-layer FALCON
    variants, where activations and gradients can collapse to nearly zero.
    Kaiming initialization preserves hidden-layer variance; the linear output
    uses Xavier initialization because it has no following ReLU.
    """
    if depth not in (10, 22, 34):
        raise ValueError("depth must be one of 10, 22, 34")
    if width not in (512, 1024):
        raise ValueError("width must be 512 or 1024")
    layers = []
    if input_dim < 1:
        raise ValueError("input_dim must be positive")
    in_features = int(input_dim)
    for _ in range(depth - 1):
        layers.extend((torch.nn.Linear(in_features, width), torch.nn.ReLU()))
        in_features = width
    layers.append(torch.nn.Linear(in_features, 4))
    model = torch.nn.Sequential(*layers)
    linear_layers = [module for module in model.modules() if isinstance(module, torch.nn.Linear)]
    for module in linear_layers[:-1]:
        torch.nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    output = linear_layers[-1]
    torch.nn.init.xavier_uniform_(output.weight)
    if output.bias is not None:
        torch.nn.init.zeros_(output.bias)
    return model


def load_normalization(
    path: str | Path,
    *,
    expected_input_dim: int | None = None,
) -> dict[str, np.ndarray]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = ("x_mean", "x_std", "u_mean", "u_std")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"normalization is missing {missing}")
    result = {key: np.asarray(payload[key], dtype=np.float32) for key in required}
    if result["x_mean"].ndim != 1 or result["x_std"].shape != result["x_mean"].shape:
        raise ValueError("input normalization arrays must be matching 1-D vectors")
    if expected_input_dim is not None and result["x_mean"].shape != (expected_input_dim,):
        raise ValueError(
            f"input normalization arrays must have shape ({expected_input_dim},)"
        )
    if result["u_mean"].shape != (4,) or result["u_std"].shape != (4,):
        raise ValueError("output normalization arrays must have shape (4,)")
    if np.any(result["x_std"] <= 0) or np.any(result["u_std"] <= 0):
        raise ValueError("normalization standard deviations must be positive")
    return result


def model_spec(
    *,
    depth: int,
    width: int,
    normalization_path: str | Path,
    input_dim: int = 12,
    input_features: tuple[str, ...] | list[str] | None = None,
    teacher_policy: str | None = None,
    input_contract: str | None = None,
) -> dict[str, Any]:
    input_dim = int(input_dim)
    path_aware = input_dim == len(PATH_AWARE_INPUT_FEATURES)
    feature_names = tuple(input_features) if input_features is not None else input_feature_names(input_dim)
    if len(feature_names) != input_dim:
        raise ValueError("input_features length does not match input_dim")
    result = {
        "format_version": 1,
        "model_name": f"falcon_dnn_input{input_dim}_depth{depth}_width{width}",
        "dtype": "float32",
        "input_shape": [1, input_dim],
        "output_shape": [1, 4],
        "input_features": list(feature_names),
        "output_features": list(OUTPUT_FEATURES),
        "frame": "AirSim NED",
        "sample_period_seconds": 0.05,
        "depth": int(depth),
        "hidden_layers": int(depth - 1),
        "width": int(width),
        "activation": "ReLU",
        "output_activation": None,
        "normalization_inside_model": False,
        "batch_size": 1,
        "normalization_file": str(normalization_path),
        "teacher_policy": teacher_policy or (
            "independent_path_aware_linear_mpc_queries_v1"
            if path_aware
            else "continuous_gate_linear_mpc_v2"
        ),
        "path_context_offsets": list(PATH_CONTEXT_OFFSETS) if path_aware else [],
    }
    if input_contract is not None:
        result["input_contract"] = str(input_contract)
    return result


def normalized_input(x: np.ndarray, normalization: Mapping[str, np.ndarray]) -> np.ndarray:
    return (np.asarray(x, dtype=np.float32) - normalization["x_mean"]) / normalization["x_std"]


def denormalized_output(u: np.ndarray, normalization: Mapping[str, np.ndarray]) -> np.ndarray:
    return np.asarray(u, dtype=np.float32) * normalization["u_std"] + normalization["u_mean"]


def state_dict_cpu(model: Any) -> dict[str, Any]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def linear_layers(model: Any) -> list[Any]:
    return [module for module in model.modules() if module.__class__.__name__ == "Linear"]


def numpy_forward(weights: Mapping[str, np.ndarray], x: np.ndarray, *, depth: int) -> np.ndarray:
    """Run the exported FC/ReLU graph without importing PyTorch."""
    value = np.asarray(x, dtype=np.float32)
    if value.ndim == 1:
        value = value[None, :]
    expected_input_dim = int(np.asarray(weights["linear_00_weight"]).shape[1])
    if value.shape[1:] != (expected_input_dim,):
        raise ValueError(f"numpy_forward expects shape (N, {expected_input_dim})")
    for index in range(depth):
        value = value @ np.asarray(weights[f"linear_{index:02d}_weight"], dtype=np.float32).T
        value = value + np.asarray(weights[f"linear_{index:02d}_bias"], dtype=np.float32)
        if index < depth - 1:
            value = np.maximum(value, 0.0)
    if value.shape[1:] != (4,):
        raise ValueError("exported model output must have shape (N, 4)")
    return value
