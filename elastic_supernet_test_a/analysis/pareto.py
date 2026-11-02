"""Pareto frontier for lower-cost/lower-error experiments."""

from __future__ import annotations

from typing import Any, Iterable


def lower_is_better_frontier(
    rows: Iterable[dict[str, Any]],
    *,
    cost_key: str = "macs",
    error_key: str = "mse",
) -> list[dict[str, Any]]:
    candidates = list(rows)
    frontier: list[dict[str, Any]] = []
    for candidate in candidates:
        cost = float(candidate[cost_key])
        error = float(candidate[error_key])
        dominated = any(
            float(other[cost_key]) <= cost
            and float(other[error_key]) <= error
            and (
                float(other[cost_key]) < cost
                or float(other[error_key]) < error
            )
            for other in candidates
            if other is not candidate
        )
        if not dominated:
            frontier.append(candidate)
    return sorted(frontier, key=lambda row: (float(row[cost_key]), float(row[error_key])))

