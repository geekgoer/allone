"""Compare fixed ACADO RTI iteration counts before authorizing 300-D labels."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from airsim_interface import RawDroneState
from falcon_paper_fidelity.acado_mpc import AcadoMPCConfig, AcadoRtiMPC
from falcon_paper_fidelity.motion_primitive import MotionPrimitiveConfig, QuinticMotionPrimitivePlanner
from falcon_paper_fidelity.phase_recovery_dataset import _build_trajectories, _load_gates


def resolve(value: str, base: Path) -> Path:
    return (base / value).resolve()


def main() -> None:
    config_file = Path(__file__).with_name("dataset_config.json").resolve()
    base = config_file.parent
    config = json.loads(config_file.read_text(encoding="utf-8"))
    with np.load(resolve(config["source"]["core_dataset"], base), allow_pickle=False) as data:
        source = {key: np.asarray(data[key]) for key in data.files}
    planner = QuinticMotionPrimitivePlanner(MotionPrimitiveConfig(**config["planner"]))
    environments = _load_gates(resolve(config["source"]["environments"], base))
    trajectories = _build_trajectories(source, environments, planner, start_velocity_mode=config["trajectory"]["start_velocity_mode"], nominal_speed=float(config["planner"]["nominal_speed"]), final_exit_distance=float(config["trajectory"]["final_exit_distance"]))
    values = dict(config["mpc"])
    values["library"] = str(resolve(values["library"], base))
    controller = AcadoRtiMPC(AcadoMPCConfig(**values))
    lower = np.array([2.0, -3.0, -3.0, -2.0])
    upper = np.array([20.0, 3.0, 3.0, 2.0])
    # One representative nominal core state from every environment.
    indices = []
    for env in range(40):
        rows = np.flatnonzero((source["environment_id"] == env) & (source["step_id"] == 30))
        indices.append(int(rows[0]))
    rows_by_count = {count: [] for count in (1, 2, 4, 8, 16)}
    repeat_errors = {count: [] for count in rows_by_count}
    for index in indices:
        state = RawDroneState(source["position"][index], source["quaternion"][index], source["velocity"][index], source["acceleration"][index], np.zeros(3))
        elapsed = float(source["step_id"][index]) * float(config["timing"]["control_dt"])
        refs = planner.reference_window_for_trajectory(trajectories[int(source["episode_id"][index])], elapsed=elapsed, dt=float(config["timing"]["prediction_dt"]), horizon=21)
        for count in rows_by_count:
            controller.reset()
            for _ in range(count):
                control = controller.solve(state, refs)
            rows_by_count[count].append(control)
            controller.reset()
            for _ in range(count):
                duplicate = controller.solve(state, refs)
            repeat_errors[count].append(float(np.max(np.abs(control - duplicate))))
    report = {"samples": len(indices), "iterations": {}}
    for count, controls in rows_by_count.items():
        array = np.asarray(controls)
        saturated = np.any((array <= lower + 1e-3) | (array >= upper - 1e-3), axis=1)
        report["iterations"][str(count)] = {
            "saturated_fraction": float(np.mean(saturated)),
            "repeatability_max_abs": float(max(repeat_errors[count])),
            "mean_control": array.mean(axis=0).tolist(),
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
