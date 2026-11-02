"""FC model cost at batch size one; bias additions are not MACs."""

from __future__ import annotations

from pathlib import Path


def complexity(input_dim: int, output_dim: int, hidden_layers: int, hidden_width: int) -> dict:
    if min(input_dim, output_dim, hidden_layers, hidden_width) < 1:
        raise ValueError("all model dimensions must be positive")
    macs = input_dim * hidden_width + (hidden_layers - 1) * hidden_width**2 + hidden_width * output_dim
    parameters = macs + hidden_layers * hidden_width + output_dim
    return {"parameter_count": parameters, "mac_count": macs}


def checkpoint_size(path: str | Path) -> int:
    return Path(path).stat().st_size
