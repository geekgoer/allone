"""Immutable wrapper around an existing fixed FALCON controller."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FixedModelInfo:
    depth: int
    width: int
    input_dim: int
    output_dim: int
    hidden_layers: int
    checkpoint: str


class FixedController:
    """Raw physical input -> physical control output wrapper."""

    def __init__(self, torch: Any, model: Any, normalization: dict[str, np.ndarray], info: FixedModelInfo):
        self.torch = torch
        self.model = model
        self.info = info
        self.x_mean = torch.as_tensor(normalization["x_mean"], dtype=torch.float32)
        self.x_std = torch.as_tensor(normalization["x_std"], dtype=torch.float32)
        self.u_mean = torch.as_tensor(normalization["u_mean"], dtype=torch.float32)
        self.u_std = torch.as_tensor(normalization["u_std"], dtype=torch.float32)

    def to(self, device: str) -> "FixedController":
        self.model.to(device)
        self.x_mean = self.x_mean.to(device)
        self.x_std = self.x_std.to(device)
        self.u_mean = self.u_mean.to(device)
        self.u_std = self.u_std.to(device)
        return self

    def eval(self) -> "FixedController":
        self.model.eval()
        return self

    def normalized_forward(self, normalized_x: Any) -> Any:
        return self.model(normalized_x)

    def __call__(self, physical_x: Any) -> Any:
        normalized_x = (physical_x - self.x_mean) / self.x_std
        normalized_u = self.normalized_forward(normalized_x)
        return normalized_u * self.u_std + self.u_mean


def _extract_state_dict(checkpoint: Any) -> dict[str, Any]:
    if isinstance(checkpoint, dict):
        for key in ("model_state", "model_state_dict", "state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if checkpoint and all(hasattr(value, "shape") for value in checkpoint.values()):
            return checkpoint
    raise ValueError("checkpoint does not contain a supported model state_dict")


def _linear_weight_items(state_dict: dict[str, Any]) -> list[tuple[str, Any]]:
    numbered: list[tuple[int, str, Any]] = []
    for key, value in state_dict.items():
        match = re.fullmatch(r"(\d+)\.weight", key)
        if match and len(value.shape) == 2:
            numbered.append((int(match.group(1)), key, value))
    numbered.sort()
    if not numbered:
        raise ValueError("no Sequential Linear weights found in checkpoint")
    return [(key, value) for _, key, value in numbered]


def load_fixed_controller(
    torch: Any,
    checkpoint_path: str | Path,
    normalization_path: str | Path,
    *,
    device: str = "cpu",
) -> FixedController:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = _extract_state_dict(checkpoint)
    weights = _linear_weight_items(state_dict)
    depth = len(weights)
    input_dim = int(weights[0][1].shape[1])
    output_dim = int(weights[-1][1].shape[0])
    hidden_widths = [int(value.shape[0]) for _, value in weights[:-1]]
    if not hidden_widths or len(set(hidden_widths)) != 1:
        raise ValueError("first prototype requires equal-width hidden layers")
    width = hidden_widths[0]

    layers: list[Any] = []
    in_features = input_dim
    for _ in range(depth - 1):
        layers.extend((torch.nn.Linear(in_features, width), torch.nn.ReLU()))
        in_features = width
    layers.append(torch.nn.Linear(in_features, output_dim))
    model = torch.nn.Sequential(*layers)
    model.load_state_dict(state_dict, strict=True)

    payload = json.loads(Path(normalization_path).read_text(encoding="utf-8"))
    normalization = {
        key: np.asarray(payload[key], dtype=np.float32)
        for key in ("x_mean", "x_std", "u_mean", "u_std")
    }
    if normalization["x_mean"].shape != (input_dim,):
        raise ValueError("normalization input dimension does not match checkpoint")
    if normalization["u_mean"].shape != (output_dim,):
        raise ValueError("normalization output dimension does not match checkpoint")
    info = FixedModelInfo(
        depth=depth,
        width=width,
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_layers=depth - 1,
        checkpoint=str(checkpoint_path.resolve()),
    )
    return FixedController(torch, model, normalization, info).to(device).eval()
