"""Audit all 400 v6 initial-state/environment starts without launching AirSim."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from airsim_interface import RawDroneState  # noqa: E402
from falcon_paper_fidelity.collector import (  # noqa: E402
    _build_controller,
    _environments,
    _load_config,
    _states,
)
from falcon_paper_fidelity.contracts import build_paper_gate_euler_input  # noqa: E402
from falcon_paper_fidelity.motion_primitive import (  # noqa: E402
    MotionPrimitiveConfig,
    QuinticMotionPrimitivePlanner,
)


HERE = Path(__file__).resolve().parent


def _summary(values: np.ndarray) -> dict[str, object]:
    return {
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "paper_core_config.json")
    parser.add_argument("--report", type=Path, default=HERE / "preflight_audit.json")
    args = parser.parse_args()

    config, config_path = _load_config(args.config)
    states = _states(config)
    environments = _environments(config)
    planner = QuinticMotionPrimitivePlanner(MotionPrimitiveConfig(**config["planner"]))
    controller = _build_controller(config, config_path.parent)
    inputs: list[np.ndarray] = []
    controls: list[np.ndarray] = []
    durations: list[float] = []

    for state in states:
        raw = RawDroneState(
            position=state.position,
            quaternion=state.orientation,
            velocity=state.velocity,
            acceleration=np.zeros(3),
            angular_velocity=state.angular_velocity,
        )
        for environment in environments:
            trajectory = planner.plan_continuous_route(
                state.position,
                state.velocity,
                np.zeros(3),
                environment.gates,
                final_exit_distance=float(config["trajectory"]["final_exit_distance"]),
            )
            references = planner.reference_window_for_trajectory(
                trajectory,
                elapsed=0.0,
                dt=float(config["mpc"]["prediction_dt"]),
                horizon=int(config["mpc"]["horizon"]) + 1,
            )
            controller.reset()
            control = np.asarray(controller.solve(raw, references), dtype=np.float64)
            if not np.all(np.isfinite(control)):
                raise RuntimeError("ACADO returned a non-finite first action")
            inputs.append(
                build_paper_gate_euler_input(
                    raw, environment.gates[0].center, environment.gates[0].orientation
                )
            )
            controls.append(control)
            durations.append(float(trajectory.duration))

    x = np.asarray(inputs)
    u = np.asarray(controls)
    limits = np.array(
        [
            float(config["mpc"]["max_thrust"]),
            float(config["mpc"]["max_body_rate_xy"]),
            float(config["mpc"]["max_body_rate_xy"]),
            float(config["mpc"]["max_body_rate_z"]),
        ]
    )
    lower = np.array(
        [float(config["mpc"]["min_thrust"]), -limits[1], -limits[2], -limits[3]]
    )
    upper = limits
    tolerance = 1e-3
    saturated = np.any((u <= lower + tolerance) | (u >= upper - tolerance), axis=1)
    saturation_fraction = float(np.mean(saturated))
    status = "PASS" if saturation_fraction <= 0.10 else "FAIL"
    payload = {
        "status": status,
        "config": str(config_path),
        "paper_protocol": {
            "initial_states": len(states),
            "environments": len(environments),
            "episodes": len(states) * len(environments),
            "samples_after_collection": len(states) * len(environments) * 100,
            "input_dimensions": 12,
            "output_dimensions": 4,
            "nominal_speed_mps": float(config["planner"]["nominal_speed"]),
        },
        "trajectory_start_velocity_mode": config["trajectory"]["start_velocity_mode"],
        "initial_positions": _summary(np.asarray([item.position for item in states])),
        "initial_velocities": _summary(np.asarray([item.velocity for item in states])),
        "first_inputs": _summary(x),
        "first_controls": _summary(u),
        "first_control_saturation_fraction": saturation_fraction,
        "maximum_allowed_first_control_saturation_fraction": 0.10,
        "trajectory_duration_seconds": {
            "min": float(np.min(durations)),
            "mean": float(np.mean(durations)),
            "max": float(np.max(durations)),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    if status != "PASS":
        raise SystemExit("preflight failed: too many first actions are saturated")


if __name__ == "__main__":
    main()
