"""Benchmark the Python ACADO adapter on changing states and references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from airsim_interface import RawDroneState  # noqa: E402
from falcon_paper_fidelity.acado_mpc import AcadoMPCConfig, AcadoRtiMPC  # noqa: E402
from falcon_paper_fidelity.motion_primitive import (  # noqa: E402
    MotionPrimitiveConfig,
    QuinticMotionPrimitivePlanner,
)


def _state(iteration: int) -> RawDroneState:
    phase = 0.013 * iteration
    return RawDroneState(
        position=np.array(
            [0.15 * np.sin(phase), 0.10 * np.cos(0.7 * phase), -2.5 + 0.05 * np.sin(0.5 * phase)]
        ),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        velocity=np.array([0.20 * np.cos(phase), 0.10 * np.sin(0.7 * phase), 0.0]),
        acceleration=np.zeros(3),
        angular_velocity=np.zeros(3),
    )


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=here / "paper_core_config.json")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--deadline-ms", type=float, default=50.0)
    parser.add_argument("--report", type=Path, default=here / "solver_benchmark.json")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    values = dict(config["mpc"])
    values.pop("backend", None)
    library = Path(values["library"])
    values["library"] = str(
        library.resolve() if library.is_absolute() else (config_path.parent / library).resolve()
    )
    controller = AcadoRtiMPC(AcadoMPCConfig(**values))
    planner = QuinticMotionPrimitivePlanner(MotionPrimitiveConfig(**config["planner"]))
    gate_position = np.array([10.0, 0.0, -2.5])
    gate_orientation = np.array([1.0, 0.0, 0.0, 0.0])
    def references(state: RawDroneState):
        return planner.reference_window(
            position=state.position,
            velocity=state.velocity,
            acceleration=state.acceleration,
            gate_position=gate_position,
            gate_orientation=gate_orientation,
            dt=controller.config.prediction_dt,
            horizon=controller.config.horizon + 1,
        )
    for index in range(args.warmup):
        state = _state(index)
        controller.solve(state, references(state))
    planner_durations: list[float] = []
    solver_durations: list[float] = []
    total_durations: list[float] = []
    outputs: list[np.ndarray] = []
    for index in range(args.iterations):
        state = _state(index + args.warmup)
        total_started = time.perf_counter_ns()
        planner_started = time.perf_counter_ns()
        reference_window = references(state)
        solver_started = time.perf_counter_ns()
        planner_durations.append((solver_started - planner_started) / 1e6)
        outputs.append(controller.solve(state, reference_window))
        finished = time.perf_counter_ns()
        solver_durations.append((finished - solver_started) / 1e6)
        total_durations.append((finished - total_started) / 1e6)
    planner_elapsed = np.asarray(planner_durations)
    solver_elapsed = np.asarray(solver_durations)
    elapsed = np.asarray(total_durations)
    controls = np.asarray(outputs)
    def timing(values: np.ndarray) -> dict[str, float]:
        return {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
            "max": float(values.max()),
        }
    result = {
        "status": "PASS" if float(np.percentile(elapsed, 99)) < args.deadline_ms else "FAIL",
        "backend": controller.signature(),
        "iterations": args.iterations,
        "deadline_ms": args.deadline_ms,
        "timing_ms": {
            "planner": timing(planner_elapsed),
            "solver_adapter": timing(solver_elapsed),
            "planner_plus_solver": timing(elapsed),
        },
        "all_outputs_finite": bool(np.isfinite(controls).all()),
        "last_output": controls[-1].tolist(),
        "last_diagnostics": controller.last_diagnostics,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Report: {args.report.resolve()}")
    if result["status"] != "PASS" or not result["all_outputs_finite"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
