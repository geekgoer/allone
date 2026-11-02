"""Select a controller architecture without changing the frozen plain MLP."""

from __future__ import annotations

from .plain_mlp import build_plain_mlp
from .residual_mlp import ResidualMLP


def build_model(torch, *, model_type: str, input_dim: int, output_dim: int, hidden_layers: int, hidden_width: int):
    if model_type == "plain_mlp":
        return build_plain_mlp(torch, input_dim=input_dim, output_dim=output_dim,
                               hidden_layers=hidden_layers, hidden_width=hidden_width)
    if model_type == "residual_mlp":
        return ResidualMLP(input_dim, output_dim, hidden_layers, hidden_width)
    raise ValueError(f"unsupported model_type: {model_type}")
