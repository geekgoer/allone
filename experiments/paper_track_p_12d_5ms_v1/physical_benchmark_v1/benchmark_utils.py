"""Shared helpers for the frozen historical 12-D physical benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if int(payload.get("format_version", 0)) != 1:
        raise ValueError("benchmark config format_version must be 1")
    names = [str(item["name"]) for item in payload["models"]]
    if len(names) != 6 or len(set(names)) != 6:
        raise ValueError("benchmark must contain exactly six uniquely named models")
    scenarios = [str(item["id"]) for item in payload["scenarios"]]
    if len(scenarios) != len(set(scenarios)):
        raise ValueError("scenario ids must be unique")
    for model in payload["models"]:
        for key in ("model", "normalization"):
            if not resolve_project_path(model[key]).exists():
                raise FileNotFoundError(f"missing {key} for {model['name']}: {model[key]}")
    initial_states = resolve_project_path(payload["initial_states"])
    if not initial_states.exists():
        raise FileNotFoundError(f"missing initial-state file: {initial_states}")
    return payload


def select_scenarios(config: dict[str, Any], scenario_set: str) -> list[dict[str, Any]]:
    by_id = {str(item["id"]): item for item in config["scenarios"]}
    if scenario_set in config["scenario_sets"]:
        ids = [str(value) for value in config["scenario_sets"][scenario_set]]
    else:
        ids = [value.strip() for value in scenario_set.split(",") if value.strip()]
    missing = [value for value in ids if value not in by_id]
    if missing:
        raise ValueError(f"unknown scenario ids: {missing}")
    return [by_id[value] for value in ids]


def regression_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if pred.shape != truth.shape or pred.ndim != 2 or pred.shape[1] != 4:
        raise ValueError("prediction and target must have matching shape (N, 4)")
    error = pred - truth
    squared = error * error
    absolute = np.abs(error)
    return {
        "samples": int(pred.shape[0]),
        "mse": float(np.mean(squared)),
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(squared))),
        "max_abs_error": float(np.max(absolute)),
        "mse_per_output": [float(value) for value in np.mean(squared, axis=0)],
        "mae_per_output": [float(value) for value in np.mean(absolute, axis=0)],
    }

