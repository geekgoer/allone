"""Offline validation for the v3 global continuous trajectory protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from falcon_paper_fidelity.collector import (  # noqa: E402
    ACADO_CONTINUOUS_POLICY_VERSION,
    _environments,
    _input_contract,
    _load_config,
    _policy_version,
    _states,
)
from falcon_paper_fidelity.contracts import TRAJECTORY_INPUT_CONTRACT  # noqa: E402
from falcon_paper_fidelity.motion_primitive import (  # noqa: E402
    MotionPrimitiveConfig,
    QuinticMotionPrimitivePlanner,
)


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=here / "paper_core_config.json")
    parser.add_argument("--report", type=Path, default=here / "protocol_validation.json")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    config, _ = _load_config(args.config)
    states = _states(config)
    environments = _environments(config)
    planner = QuinticMotionPrimitivePlanner(MotionPrimitiveConfig(**config["planner"]))
    final_exit_distance = float(config["trajectory"]["final_exit_distance"])
    durations: list[float] = []
    segment_durations: list[float] = []
    max_p_jump = 0.0
    max_v_jump = 0.0
    max_a_jump = 0.0
    frozen_moving_pairs = 0
    checked_boundaries = 0

    for state in states:
        for environment in environments:
            trajectory = planner.plan_continuous_route(
                state.position,
                state.velocity,
                np.zeros(3),
                environment.gates,
                final_exit_distance=final_exit_distance,
            )
            durations.append(trajectory.duration)
            segment_durations.extend(
                segment.primitive.duration for segment in trajectory.segments
            )
            if len(trajectory.boundary_times) != len(environment.gates):
                raise AssertionError("each gate must be one trajectory boundary")
            for index, (gate, boundary) in enumerate(
                zip(environment.gates, trajectory.boundary_times)
            ):
                end = trajectory.segments[index].primitive.evaluate(
                    trajectory.segments[index].primitive.duration
                )
                start = trajectory.segments[index + 1].primitive.evaluate(0.0)
                max_p_jump = max(max_p_jump, float(np.linalg.norm(end[0] - start[0])))
                max_v_jump = max(max_v_jump, float(np.linalg.norm(end[1] - start[1])))
                max_a_jump = max(max_a_jump, float(np.linalg.norm(end[2] - start[2])))
                np.testing.assert_allclose(end[0], gate.center, atol=1e-8)
                references = planner.reference_window_for_trajectory(
                    trajectory,
                    elapsed=max(0.0, boundary - 0.5),
                    dt=float(config["mpc"]["prediction_dt"]),
                    horizon=int(config["mpc"]["horizon"]) + 1,
                )
                for left, right in zip(references[:-1], references[1:]):
                    stationary = np.linalg.norm(right.position - left.position) <= 1e-8
                    moving = max(np.linalg.norm(left.velocity), np.linalg.norm(right.velocity)) > 1e-6
                    frozen_moving_pairs += int(stationary and moving)
                checked_boundaries += 1
            _, final_velocity, final_acceleration, _ = trajectory.evaluate(
                trajectory.duration + 1.0
            )
            np.testing.assert_allclose(final_velocity, 0.0, atol=1e-8)
            np.testing.assert_allclose(final_acceleration, 0.0, atol=1e-8)

    contract, features = _input_contract(config)
    report = {
        "status": "PASS",
        "policy_version": _policy_version(config),
        "input_contract": contract,
        "input_dimensions": len(features),
        "initial_states": len(states),
        "environments": len(environments),
        "trajectories_checked": len(durations),
        "gate_boundaries_checked": checked_boundaries,
        "trajectory_duration_seconds": {
            "min": min(durations),
            "mean": float(np.mean(durations)),
            "max": max(durations),
        },
        "segment_duration_seconds": {
            "min": min(segment_durations),
            "mean": float(np.mean(segment_durations)),
            "max": max(segment_durations),
        },
        "maximum_boundary_discontinuity": {
            "position": max_p_jump,
            "velocity": max_v_jump,
            "acceleration": max_a_jump,
        },
        "frozen_position_with_nonzero_velocity_pairs": frozen_moving_pairs,
        "gate_data_visible_to_controller": False,
        "final_stop_only_after_last_gate": True,
    }
    if report["policy_version"] != ACADO_CONTINUOUS_POLICY_VERSION:
        raise AssertionError("v3 config did not select the continuous ACADO policy")
    if contract != TRAJECTORY_INPUT_CONTRACT or len(features) != 12:
        raise AssertionError("v3 did not select the trajectory-relative 12-D contract")
    if max(max_p_jump, max_v_jump, max_a_jump) > 1e-8:
        raise AssertionError("trajectory is not C2 at a gate boundary")
    if frozen_moving_pairs:
        raise AssertionError("a reference position froze while velocity stayed nonzero")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2) if args.json else report)
    print(f"Report: {args.report.resolve()}")


if __name__ == "__main__":
    main()
