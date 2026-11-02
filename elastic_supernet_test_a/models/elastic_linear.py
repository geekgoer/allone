"""Prefix-sliced linear operation used by the elastic MLP."""

from __future__ import annotations

from typing import Any


def elastic_linear(
    torch: Any,
    x: Any,
    weight: Any,
    bias: Any | None,
    *,
    active_in: int,
    active_out: int,
) -> Any:
    if active_in < 1 or active_out < 1:
        raise ValueError("active dimensions must be positive")
    if active_in > weight.shape[1] or active_out > weight.shape[0]:
        raise ValueError("active dimensions exceed maximum tensor dimensions")
    if x.shape[-1] < active_in:
        raise ValueError("input tensor is narrower than active_in")
    active_bias = None if bias is None else bias[:active_out]
    return torch.nn.functional.linear(
        x[..., :active_in],
        weight[:active_out, :active_in],
        active_bias,
    )

