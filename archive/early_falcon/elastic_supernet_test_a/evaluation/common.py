"""Shared loading and regression metric helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np


def load_split(path: str | Path, *, max_samples: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as archive:
        x = np.asarray(archive["X"], dtype=np.float32)
        u = np.asarray(archive["U"], dtype=np.float32)
    if max_samples is not None:
        x = x[:max_samples]
        u = u[:max_samples]
    if x.ndim != 2 or u.ndim != 2 or x.shape[0] != u.shape[0]:
        raise ValueError("dataset must contain matching 2-D X and U arrays")
    return x, u


def regression_metrics(
    torch: Any,
    predict: Callable[[Any], Any],
    x: np.ndarray,
    target: np.ndarray,
    *,
    batch_size: int,
    device: str,
    reference_predict: Callable[[Any], Any] | None = None,
) -> dict[str, Any]:
    squared = np.zeros(target.shape[1], dtype=np.float64)
    absolute = np.zeros(target.shape[1], dtype=np.float64)
    reference_squared = 0.0
    maximum = 0.0
    count = 0
    with torch.inference_mode():
        for start in range(0, x.shape[0], batch_size):
            stop = min(start + batch_size, x.shape[0])
            batch_x = torch.from_numpy(x[start:stop]).to(device)
            prediction = predict(batch_x).detach().cpu().numpy().astype(np.float64)
            difference = prediction - target[start:stop].astype(np.float64)
            squared += np.square(difference).sum(axis=0)
            absolute += np.abs(difference).sum(axis=0)
            maximum = max(maximum, float(np.abs(difference).max(initial=0.0)))
            if reference_predict is not None:
                reference = reference_predict(batch_x).detach().cpu().numpy().astype(np.float64)
                reference_squared += float(np.square(prediction - reference).sum())
            count += stop - start
    mse_per_output = squared / count
    mae_per_output = absolute / count
    result: dict[str, Any] = {
        "samples": count,
        "mse": float(mse_per_output.mean()),
        "rmse": float(np.sqrt(mse_per_output.mean())),
        "mae": float(mae_per_output.mean()),
        "max_abs_error": maximum,
        "mse_per_output": mse_per_output.tolist(),
        "mae_per_output": mae_per_output.tolist(),
    }
    if reference_predict is not None:
        result["mse_vs_fixed_full"] = reference_squared / (count * target.shape[1])
    return result

