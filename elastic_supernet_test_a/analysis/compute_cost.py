"""Theoretical active cost for a width-elastic MLP."""

from __future__ import annotations

from typing import Iterable


def mlp_macs(input_dim: int, hidden_widths: Iterable[int], output_dim: int) -> int:
    dimensions = [int(input_dim), *(int(value) for value in hidden_widths), int(output_dim)]
    if any(value < 1 for value in dimensions):
        raise ValueError("all dimensions must be positive")
    return sum(left * right for left, right in zip(dimensions, dimensions[1:]))


def mlp_active_parameters(input_dim: int, hidden_widths: Iterable[int], output_dim: int) -> int:
    dimensions = [int(input_dim), *(int(value) for value in hidden_widths), int(output_dim)]
    if any(value < 1 for value in dimensions):
        raise ValueError("all dimensions must be positive")
    weights = sum(left * right for left, right in zip(dimensions, dimensions[1:]))
    biases = sum(dimensions[1:])
    return weights + biases

