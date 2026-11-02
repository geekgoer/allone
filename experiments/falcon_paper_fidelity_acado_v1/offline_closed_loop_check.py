"""Run the ACADO controller against its matching ideal quadrotor dynamics."""

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
from falcon_paper_fidelity.acado_mpc import AcadoMPCConfig, AcadoRtiMPC  # noqa: E402
from falcon_paper_fidelity.motion_primitive import (  # noqa: E402
    MotionPrimitiveConfig,
    QuinticMotionPrimitivePlanner,
)
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
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--planner-zero-acceleration", action="store_true")
    parser.add_argument("--maximum-acceleration", type=float, default=None)
    parser.add_argument(
        "--planner-mode",
        choices=("segment", "replan"),
        default="segment",
        help="segment is the corrected policy; replan reproduces the failed v1 policy",
    )
    parser.add_argument("--report", type=Path, default=here / "offline_closed_loop.json")
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
    planner_values = dict(config["planner"])
    if args.maximum_acceleration is not None:
        planner_values["maximum_acceleration"] = args.maximum_acceleration
    planner = QuinticMotionPrimitivePlanner(MotionPrimitiveConfig(**planner_values))
    dt = float(config["timing"]["dt"])
    position = np.array([0.0, 0.0, -2.5])
    quaternion = np.array([1.0, 0.0, 0.0, 0.0])
    velocity = np.zeros(3)
    acceleration = np.zeros(3)
    gates = [np.array([10.0 * index, 0.0, -2.5]) for index in range(1, 6)]
    gate_index = 0
    active_primitive = None
    primitive_elapsed = 0.0
    controls: list[list[float]] = []
    states: list[list[float]] = []
    for step in range(args.steps):
        raw = RawDroneState(
            position=position.copy(),
            quaternion=quaternion.copy(),
            velocity=velocity.copy(),
            acceleration=acceleration.copy(),
            angular_velocity=np.zeros(3),
        )
        while gate_index < len(gates) and position[0] > gates[gate_index][0]:
            gate_index += 1
            active_primitive = None
            primitive_elapsed = 0.0
        if gate_index >= len(gates):
            references = [
                MPCReference(
                    position=gates[-1] + np.array([1.0, 0.0, 0.0]),
                    orientation=np.array([1.0, 0.0, 0.0, 0.0]),
                )
                for _ in range(controller.config.horizon + 1)
            ]
        elif args.planner_mode == "replan":
            references = planner.reference_window(
                position=position,
                velocity=velocity,
                acceleration=(np.zeros(3) if args.planner_zero_acceleration else acceleration),
                gate_position=gates[gate_index],
                gate_orientation=np.array([1.0, 0.0, 0.0, 0.0]),
                dt=controller.config.prediction_dt,
                horizon=controller.config.horizon + 1,
            )
        else:
            if active_primitive is None:
                active_primitive = planner.plan(
                    position,
                    velocity,
                    np.zeros(3) if args.planner_zero_acceleration else acceleration,
                    gates[gate_index],
                    np.array([1.0, 0.0, 0.0, 0.0]),
                )
            references = planner.reference_window_for_primitive(
                active_primitive,
                gate_orientation=np.array([1.0, 0.0, 0.0, 0.0]),
                elapsed=primitive_elapsed,
                dt=controller.config.prediction_dt,
                horizon=controller.config.horizon + 1,
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
        if gate_index < len(gates) and args.planner_mode == "segment":
            primitive_elapsed += dt
        controls.append(control.tolist())
        states.append(np.r_[position, velocity, acceleration, quaternion].tolist())
        if step % 10 == 0:
            print(
                f"step={step:03d} p={np.round(position, 3)} "
                f"v={np.round(velocity, 3)} u={np.round(control, 3)}"
            )
    result = {
        "steps": args.steps,
        "dt": dt,
        "final_position": position.tolist(),
        "final_velocity": velocity.tolist(),
        "maximum_speed": float(max(np.linalg.norm(np.asarray(item)[3:6]) for item in states)),
        "minimum_ned_z": float(min(item[2] for item in states)),
        "maximum_ned_z": float(max(item[2] for item in states)),
        "gates_crossed": gate_index,
        "planner_mode": args.planner_mode,
        "finite": bool(np.isfinite(states).all() and np.isfinite(controls).all()),
        "states": states,
        "controls": controls,
    }
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in {"states", "controls"}}, indent=2))


if __name__ == "__main__":
    main()
