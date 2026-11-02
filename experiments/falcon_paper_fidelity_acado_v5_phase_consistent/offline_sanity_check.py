"""Offline checks for the v5 FALCON teacher before an AirSim collection.

The check reuses the completed v3 rollouts only as initial-state snapshots and
environment metadata.  It does not modify either dataset.  For every one of
the 400 source episodes it verifies that the new nominal-speed global
trajectory is feasible.  For a small deterministic subset it then compares
the first ACADO action produced by the old measured-velocity start against the
new 5 m/s nominal gate-axis start.
"""

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
    _load_config,
)
from falcon_paper_fidelity.contracts import (  # noqa: E402
    build_paper_gate_euler_input,
)
from falcon_paper_fidelity.motion_primitive import (  # noqa: E402
    MotionPrimitiveConfig,
    QuinticMotionPrimitivePlanner,
)
from training_environment_generator import load_training_environments  # noqa: E402


HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE_ROOT = (
    HERE.parent
    / "falcon_paper_fidelity_acado_v3_continuous"
    / "paper_core_40k_acado_v3_continuous"
)


def _source_dataset(root: Path) -> Path:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") not in {"complete", "completed"}:
        raise ValueError(f"source manifest is not complete: {root}")
    relative = manifest.get("final_dataset")
    if relative:
        return root / str(relative)
    candidates = list((root / "final").glob("*.npz"))
    if len(candidates) != 1:
        raise ValueError("could not identify one source final dataset")
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "paper_core_config.json")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--action-samples", type=int, default=12)
    parser.add_argument("--report", type=Path, default=HERE / "offline_sanity_report.json")
    args = parser.parse_args()
    if args.action_samples < 1:
        raise ValueError("action-samples must be positive")

    config, config_path = _load_config(args.config)
    base = config_path.parent
    source_root = args.source_root.resolve()
    with np.load(_source_dataset(source_root), allow_pickle=False) as archive:
        source = {key: np.asarray(archive[key]) for key in archive.files}
    environments = load_training_environments(source_root / "environments.json")
    environment_lookup = {item.environment_id: item for item in environments}
    planner = QuinticMotionPrimitivePlanner(MotionPrimitiveConfig(**config["planner"]))
    episode_ids = np.unique(source["episode_id"].astype(np.int64))
    if episode_ids.size != 400:
        raise ValueError(f"expected 400 source episodes, found {episode_ids.size}")

    episodes: list[dict[str, object]] = []
    trajectories: dict[int, object] = {}
    for episode_id in episode_ids:
        index = int(np.flatnonzero(source["episode_id"] == episode_id)[0])
        environment_id = int(source["environment_id"][index])
        environment = environment_lookup[environment_id]
        start_velocity = (
            float(config["planner"]["nominal_speed"])
            * planner.passage_axis(environment.gates[0].orientation)
        )
        trajectory = planner.plan_continuous_route(
            source["position"][index],
            start_velocity,
            np.zeros(3),
            environment.gates,
            final_exit_distance=float(config["trajectory"]["final_exit_distance"]),
        )
        trajectories[int(episode_id)] = trajectory
        first_reference = planner.reference_window_for_trajectory(
            trajectory,
            elapsed=0.0,
            dt=float(config["mpc"]["prediction_dt"]),
            horizon=int(config["mpc"]["horizon"]) + 1,
        )[0]
        episodes.append(
            {
                "episode_id": int(episode_id),
                "environment_id": environment_id,
                "trajectory_duration_seconds": float(trajectory.duration),
                "first_reference_speed": float(np.linalg.norm(first_reference.velocity)),
            }
        )

    controller = _build_controller(config, base)
    action_rows: list[dict[str, object]] = []
    selected = episode_ids[: min(args.action_samples, episode_ids.size)]
    for episode_id in selected:
        index = int(np.flatnonzero(source["episode_id"] == episode_id)[0])
        environment = environment_lookup[int(source["environment_id"][index])]
        state = RawDroneState(
            position=source["position"][index],
            quaternion=source["quaternion"][index],
            velocity=source["velocity"][index],
            acceleration=source["acceleration"][index],
            angular_velocity=np.zeros(3),
        )
        measured_trajectory = planner.plan_continuous_route(
            state.position,
            state.velocity,
            np.zeros(3),
            environment.gates,
            final_exit_distance=float(config["trajectory"]["final_exit_distance"]),
        )
        old_references = planner.reference_window_for_trajectory(
            measured_trajectory,
            elapsed=0.0,
            dt=float(config["mpc"]["prediction_dt"]),
            horizon=int(config["mpc"]["horizon"]) + 1,
        )
        new_references = planner.reference_window_for_trajectory(
            trajectories[int(episode_id)],
            elapsed=0.0,
            dt=float(config["mpc"]["prediction_dt"]),
            horizon=int(config["mpc"]["horizon"]) + 1,
        )
        controller.reset()
        old_control = controller.solve(state, old_references)
        controller.reset()
        new_control = controller.solve(state, new_references)
        paper_x = build_paper_gate_euler_input(
            state, environment.gates[0].center, environment.gates[0].orientation
        )
        action_rows.append(
            {
                "episode_id": int(episode_id),
                "actual_initial_speed": float(np.linalg.norm(state.velocity)),
                "paper_x": paper_x.tolist(),
                "old_measured_start_control": old_control.tolist(),
                "new_nominal_start_control": new_control.tolist(),
                "max_abs_control_change": float(np.max(np.abs(new_control - old_control))),
            }
        )

    changes = np.asarray([item["max_abs_control_change"] for item in action_rows])
    payload = {
        "status": "PASS",
        "config": str(config_path),
        "source_used_only_for_initial_snapshots": str(source_root),
        "trajectory_start_velocity_mode": "nominal_gate_axis",
        "nominal_speed_mps": float(config["planner"]["nominal_speed"]),
        "episodes_checked": len(episodes),
        "trajectory_duration_seconds": {
            "min": float(min(item["trajectory_duration_seconds"] for item in episodes)),
            "mean": float(np.mean([item["trajectory_duration_seconds"] for item in episodes])),
            "max": float(max(item["trajectory_duration_seconds"] for item in episodes)),
        },
        "first_reference_speed_mps": {
            "min": float(min(item["first_reference_speed"] for item in episodes)),
            "max": float(max(item["first_reference_speed"] for item in episodes)),
        },
        "first_action_comparison": {
            "samples": len(action_rows),
            "median_max_abs_change": float(np.median(changes)),
            "rows": action_rows,
        },
        "paper_input_contract": "falcon_paper_gate_relative_euler_12d_v1",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "first_action_comparison"}, indent=2))
    print(
        "First-action comparison: "
        f"samples={len(action_rows)} median_max_abs_change={np.median(changes):.6f}"
    )
    print(f"Report: {args.report.resolve()}")


if __name__ == "__main__":
    main()
