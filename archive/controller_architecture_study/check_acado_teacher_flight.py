"""Manual-AirSim ACADO teacher flight from rest, using the existing v3 route.

Writes only a standalone report. No training data or trained model is changed.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from config import load_calibration
from experiments.falcon_paper_fidelity_acado_v3_continuous.demo_acado_five_gates import _crossing_results
from falcon_paper_fidelity.collector import _build_controller, _connect, _load_config, _resolve, run_episode
from falcon_paper_fidelity.motion_primitive import MotionPrimitiveConfig, QuinticMotionPrimitivePlanner
from initial_state_generator import InitialState
from training_environment_generator import load_training_environments


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment-id", type=int, default=38)
    args = parser.parse_args()
    config, config_path = _load_config(ROOT / "experiments/falcon_paper_fidelity_acado_v3_continuous/paper_core_config.json")
    config = deepcopy(config)
    environments = load_training_environments(ROOT / "experiments/falcon_paper_fidelity_acado_v2/paper_core_40k_acado_v2/environments.json")
    environment = next(item for item in environments if item.environment_id == args.environment_id)
    config["timing"].update(steps_per_episode=400, episode_seconds=20., episode_timeout_seconds=35.)
    config["trajectory"]["start_velocity_mode"] = "measured"
    config["gate"].update(spawn_in_simulation=True, keep_spawned_gates=True)
    initial = InitialState(0, np.array([0., 0., -3.]), np.array([1., 0., 0., 0.]),
                           np.zeros(3), np.zeros(3))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    report_path = ROOT / f"controller_architecture_study/results/acado_teacher_flight/rest_env{args.environment_id}_{stamp}.json"
    report = {"status": "FAIL", "controller": "ACADO_MPC_not_DNN", "initial_speed_mps": 0.,
              "nominal_speed_mps": config["planner"]["nominal_speed"], "environment_id": args.environment_id,
              "trajectory": "committed_global_C2_clock", "dataset_written": False}
    started = time.perf_counter()
    try:
        # Connect to the manually opened Play session; never launch another UE4.
        client = _connect(config["airsim"])
        controller = _build_controller(config, config_path.parent)
        planner = QuinticMotionPrimitivePlanner(MotionPrimitiveConfig(**config["planner"]))
        episode = run_episode(client, initial, environment, episode_id=-1, config=config,
                              calibration=load_calibration(_resolve(config["calibration"], config_path.parent)),
                              planner=planner, controller=controller)
        crossings = _crossing_results(episode.position, episode.velocity, environment, config["gate"])
        # A ground/collision flag or invalid gate passage makes run_episode fail.
        progress = float(np.linalg.norm(episode.position[100] - episode.position[0]))
        valid = int(sum(item[3] for item in crossings))
        report.update(status="PASS" if valid == len(crossings) and progress >= 1. else "FAIL",
                      samples=len(episode.X), displacement_first_5s_m=progress,
                      valid_crossings=valid, gate_count=len(crossings),
                      deadline_misses=int(np.sum(episode.deadline_miss)),
                      mean_solve_ms=float(np.mean(episode.mpc_solve_time) * 1000),
                      crossings=[{"gate": index, "valid": ok, "speed_mps": speed,
                                  "lateral_m": None if crossing is None else crossing.lateral_offset,
                                  "vertical_m": None if crossing is None else crossing.vertical_offset}
                                 for index, crossing, speed, ok in crossings], error=None)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
        print(f"Report: {report_path}", flush=True)
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
