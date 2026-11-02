"""Isolate reference-clock and RTI-initialization effects without AirSim.

This is an ideal rigid-body replay, not evidence of live AirSim flight or DNN
performance. All cases apply only u[0] and then re-solve after 50 ms.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from airsim_interface import RawDroneState
from falcon_paper_fidelity.acado_mpc import AcadoMPCConfig, AcadoRtiMPC
from falcon_paper_fidelity.contracts import build_trajectory_input
from falcon_paper_fidelity.motion_primitive import MotionPrimitiveConfig, QuinticMotionPrimitivePlanner
from falcon_paper_fidelity.short_rollout_dataset import propagate_state


ROOT = Path(__file__).resolve().parents[1]


def reference_alias_case(controller, planner):
    """Two feasible futures can share exactly the same instantaneous 12D."""
    state = RawDroneState(position=np.array([0., 0., -3.]),
                          quaternion=np.array([1., 0., 0., 0.]),
                          velocity=np.zeros(3), acceleration=np.zeros(3), angular_velocity=np.zeros(3))
    inputs, controls, kkt = [], [], []
    for lateral in (-3., 3.):
        references = planner.reference_window(
            position=state.position, velocity=state.velocity, acceleration=state.acceleration,
            gate_position=np.array([20., lateral, -3.]), gate_orientation=np.array([1., 0., 0., 0.]),
            dt=controller.config.prediction_dt, horizon=controller.config.horizon + 1)
        inputs.append(build_trajectory_input(state, references[0]))
        controller.reset()
        for _ in range(16):
            control = controller.solve(state, references)
        controls.append(control.copy())
        kkt.append(controller.last_diagnostics["kkt"])
    return {"contract": "trajectory_relative_12d_not_gate_relative_12d",
            "input_max_abs_difference": float(np.max(np.abs(inputs[0] - inputs[1]))),
            "control_max_abs_difference": float(np.max(np.abs(controls[0] - controls[1]))),
            "inputs": [value.tolist() for value in inputs],
            "controls": [value.tolist() for value in controls], "kkt": kkt,
            "interpretation": "Instantaneous trajectory 12D omits future curvature/feedforward changes."}


def run_case(controller, planner, *, reference_mode, solve_mode, steps, dt):
    state = RawDroneState(position=np.array([0., 0., -3.]),
                          quaternion=np.array([1., 0., 0., 0.]),
                          velocity=np.zeros(3), acceleration=np.zeros(3), angular_velocity=np.zeros(3))
    gate = np.array([20., 0., -3.])
    gate_q = np.array([1., 0., 0., 0.])
    primitive = planner.plan(state.position, state.velocity, state.acceleration, gate, gate_q)
    controller.reset()
    trace = []
    completed_steps = 0
    started = time.perf_counter()
    error = None
    maximum_kkt = 0.
    final_control_delta = None
    try:
        for step in range(steps):
            if reference_mode == "replan_zero_clock":
                references = planner.reference_window(
                    position=state.position, velocity=state.velocity, acceleration=state.acceleration,
                    gate_position=gate, gate_orientation=gate_q,
                    dt=controller.config.prediction_dt, horizon=controller.config.horizon + 1)
            else:
                references = planner.reference_window_for_primitive(
                    primitive, gate_orientation=gate_q, elapsed=step * dt,
                    dt=controller.config.prediction_dt, horizon=controller.config.horizon + 1)
            if solve_mode != "warm_rti":
                controller.reset()
            iterations = 16 if solve_mode == "cold_refine16" else 1
            previous = None
            for _ in range(iterations):
                control = controller.solve(state, references)
                if previous is not None:
                    final_control_delta = float(np.max(np.abs(control - previous)))
                previous = control.copy()
            maximum_kkt = max(maximum_kkt, controller.last_diagnostics["kkt"])
            following = propagate_state(state, control, dt=dt, gravity=controller.config.gravity)
            if step < 5 or step % 20 == 0 or step == steps - 1:
                trace.append({"t": step * dt, "position": state.position.tolist(),
                              "speed_mps": float(np.linalg.norm(state.velocity)),
                              "u0": control.tolist(), "kkt": controller.last_diagnostics["kkt"],
                              "first_reference_error": (references[0].position - state.position).tolist()})
            state = following
            completed_steps = step + 1
            if not np.isfinite(state.position).all() or np.linalg.norm(state.velocity) > 30:
                raise RuntimeError("ideal replay diverged")
    except RuntimeError as exc:
        error = str(exc)
    return {"reference_mode": reference_mode, "solve_mode": solve_mode,
            "steps_completed": completed_steps, "simulated_seconds": completed_steps * dt,
            "initial_primitive_duration": primitive.duration, "final_position": state.position.tolist(),
            "forward_displacement_m": float(state.position[0]),
            "final_speed_mps": float(np.linalg.norm(state.velocity)),
            "gate_plane_reached": bool(state.position[0] >= gate[0]),
            "maximum_recorded_kkt": maximum_kkt, "last_refinement_control_delta": final_control_delta,
            "error": error, "elapsed_wall_seconds": time.perf_counter() - started, "trace": trace}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "controller_architecture_study/results/acado_progress_audit/report.json")
    args = parser.parse_args()
    config = AcadoMPCConfig()
    controller = AcadoRtiMPC(config)
    planner = QuinticMotionPrimitivePlanner(MotionPrimitiveConfig())
    cases = []
    # The DLL owns global solver state. Deliberately run cases serially.
    for reference in ("replan_zero_clock", "committed_clock"):
        for solve in ("cold_rti", "warm_rti", "cold_refine16"):
            result = run_case(controller, planner, reference_mode=reference,
                              solve_mode=solve, steps=200, dt=0.05)
            cases.append(result)
            print(f"{reference}/{solve}: x={result['forward_displacement_m']:.3f}m, "
                  f"v={result['final_speed_mps']:.3f}m/s, error={result['error']}", flush=True)
    payload = {"model": "ideal_rigid_body_RK4_no_AirSim_actuator_or_collision",
               "control_dt": 0.05, "prediction_dt": config.prediction_dt, "horizon": config.horizon,
               "target_speed_mps": planner.config.nominal_speed,
               "all_cases_execute_first_control_only": True,
               "live_AirSim_validated": False, "training_started": False, "cases": cases,
               "instantaneous_12d_alias": reference_alias_case(controller, planner)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.report)
    print(f"Report: {args.report.resolve()}")


if __name__ == "__main__":
    main()
