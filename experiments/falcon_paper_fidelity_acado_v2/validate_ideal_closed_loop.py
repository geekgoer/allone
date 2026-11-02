"""Validate all 400 protocol combinations against ideal matching dynamics."""

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
    _hold_reference,
    _states,
)
from falcon_paper_fidelity.motion_primitive import (  # noqa: E402
    MotionPrimitiveConfig,
    QuinticMotionPrimitivePlanner,
)
from gate_route import ContinuousGateRoute  # noqa: E402
from state_adapter import normalize_quaternion, quaternion_multiply  # noqa: E402


def _rotation_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=here / "paper_core_config.json")
    parser.add_argument("--report", type=Path, default=here / "ideal_closed_loop_report.json")
    parser.add_argument("--max-episodes", type=int, default=None)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    states = _states(config)
    environments = _environments(config)
    planner = QuinticMotionPrimitivePlanner(MotionPrimitiveConfig(**config["planner"]))
    controller = _build_controller(config, config_path.parent)
    dt = float(config["timing"]["dt"])
    steps = int(config["timing"]["steps_per_episode"])
    gate_cfg = config["gate"]
    failures: list[dict[str, object]] = []
    maximum_speed = 0.0
    maximum_ned_z = -np.inf
    episodes = 0

    for initial in states:
        for environment in environments:
            if args.max_episodes is not None and episodes >= args.max_episodes:
                break
            episode_id = initial.initial_state_id * len(environments) + environment.environment_id
            episodes += 1
            controller.reset()
            position = initial.position.copy()
            quaternion = initial.orientation.copy()
            velocity = initial.velocity.copy()
            acceleration = np.zeros(3)
            route = ContinuousGateRoute(
                environment.gates,
                position,
                approach_distance=float(gate_cfg["approach_distance"]),
                exit_distance=float(gate_cfg["exit_distance"]),
                reference_speed=float(config["planner"]["nominal_speed"]),
                start_time=0.0,
                gate_width=float(gate_cfg["width"]),
                gate_height=float(gate_cfg["height"]),
                gate_thickness=float(gate_cfg["thickness"]),
                clearance=float(gate_cfg["clearance"]),
                completion_radius=float(gate_cfg["threshold"]),
            )
            active_gate_id = None
            active_primitive = None
            primitive_elapsed = 0.0
            try:
                for step in range(steps):
                    route.update(position, step * dt)
                    gate = route.current_gate
                    if route.completed:
                        references = _hold_reference(
                            route.exit_points[-1], gate.orientation, controller.config.horizon + 1
                        )
                    else:
                        if active_gate_id != route.target_gate_id or active_primitive is None:
                            active_primitive = planner.plan(
                                position,
                                velocity,
                                acceleration,
                                gate.center,
                                gate.orientation,
                            )
                            active_gate_id = route.target_gate_id
                            primitive_elapsed = 0.0
                        references = planner.reference_window_for_primitive(
                            active_primitive,
                            gate_orientation=gate.orientation,
                            elapsed=primitive_elapsed,
                            dt=controller.config.prediction_dt,
                            horizon=controller.config.horizon + 1,
                        )
                    raw = RawDroneState(
                        position=position.copy(),
                        quaternion=quaternion.copy(),
                        velocity=velocity.copy(),
                        acceleration=acceleration.copy(),
                        angular_velocity=np.zeros(3),
                    )
                    control = controller.solve(raw, references)
                    rates = control[1:4]
                    q_dot = 0.5 * quaternion_multiply(quaternion, np.r_[0.0, rates])
                    quaternion = normalize_quaternion(quaternion + dt * q_dot)
                    acceleration = np.array([0.0, 0.0, controller.config.gravity]) + _rotation_matrix(
                        quaternion
                    ) @ np.array([0.0, 0.0, -control[0]])
                    velocity = velocity + dt * acceleration
                    position = position + dt * velocity
                    primitive_elapsed += dt
                    speed = float(np.linalg.norm(velocity))
                    maximum_speed = max(maximum_speed, speed)
                    maximum_ned_z = max(maximum_ned_z, float(position[2]))
                    if not np.isfinite(np.r_[position, velocity, acceleration, quaternion, control]).all():
                        raise RuntimeError("non-finite ideal state")
                    if position[2] >= 0.0:
                        raise RuntimeError(f"ideal vehicle reached ground plane at step {step}")
            except Exception as exc:
                failures.append(
                    {
                        "episode_id": episode_id,
                        "initial_state_id": initial.initial_state_id,
                        "environment_id": environment.environment_id,
                        "error": f"{type(exc).__name__}: {exc}",
                        "position": position.tolist(),
                        "velocity": velocity.tolist(),
                    }
                )
        if args.max_episodes is not None and episodes >= args.max_episodes:
            break

    report = {
        "status": "PASS" if not failures else "FAIL",
        "episodes": episodes,
        "failures": len(failures),
        "maximum_speed": maximum_speed,
        "maximum_ned_z": maximum_ned_z,
        "failure_details": failures,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "failure_details"}, indent=2))
    if failures:
        print(json.dumps(failures[:5], indent=2))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
