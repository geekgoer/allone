from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


BENCHMARK_DIR = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "paper_track_p_12d_5ms_v1"
    / "physical_benchmark_v1"
)
sys.path.insert(0, str(BENCHMARK_DIR))

from benchmark_utils import regression_metrics, select_scenarios


def test_regression_metrics_exact_values() -> None:
    target = np.zeros((2, 4), dtype=np.float32)
    prediction = np.array([[1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 2.0, -2.0]])
    metrics = regression_metrics(prediction, target)
    assert metrics["samples"] == 2
    assert np.isclose(metrics["mse"], 1.25)
    assert np.isclose(metrics["mae"], 0.75)
    assert np.allclose(metrics["mse_per_output"], [0.5, 0.5, 2.0, 2.0])


def test_select_named_and_explicit_scenarios() -> None:
    config = {
        "scenario_sets": {"pilot": ["A"]},
        "scenarios": [{"id": "A"}, {"id": "B"}],
    }
    assert [item["id"] for item in select_scenarios(config, "pilot")] == ["A"]
    assert [item["id"] for item in select_scenarios(config, "B,A")] == ["B", "A"]

