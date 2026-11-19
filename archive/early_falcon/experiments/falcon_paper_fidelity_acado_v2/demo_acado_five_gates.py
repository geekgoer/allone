"""Render the corrected ACADO/qpOASES teacher flying five physical gates.

This is an isolated demonstration. It reads the immutable initial-state and
environment artifacts from the completed paper-core dataset, but never writes
to that dataset or changes its manifest.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from airsim_process import AirSimLaunchConfig, AirSimProcessManager  # noqa: E402
from config import load_calibration  # noqa: E402
from falcon_paper_fidelity.collector import (  # noqa: E402
    _build_controller,
    _connect,
    _load_config,
    _resolve,
    run_episode,
)
from falcon_paper_fidelity.motion_primitive import (  # noqa: E402
    MotionPrimitiveConfig,
    QuinticMotionPrimitivePlanner,
)
from gate_route import segment_gate_crossing  # noqa: E402
from gate_visualization import cleanup_generated_gates  # noqa: E402
from initial_state_generator import load_initial_states  # noqa: E402
from training_environment_generator import load_training_environments  # noqa: E402


HERE = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = HERE / "paper_core_40k_acado_v2"


def _count_valid_crossings(positions, environment, gate_config):
    half_width = (
        float(gate_config["width"]) / 2.0
        - float(gate_config["thickness"]) / 2.0
        - float(gate_config["clearance"])
    )
    half_height = (
        float(gate_config["height"]) / 2.0
        - float(gate_config["thickness"]) / 2.0
        - float(gate_config["clearance"])
    )
    results = []
    trajectory = np.asarray(positions, dtype=np.float64)
    for gate in environment.gates:
        crossing = None
        for previous, current in zip(trajectory[:-1], trajectory[1:]):
            candidate = segment_gate_crossing(previous, current, gate)
            if candidate is not None:
                crossing = candidate
                break
        valid = bool(
            crossing is not None
            and abs(crossing.lateral_offset) <= half_width
            and abs(crossing.vertical_offset) <= half_height
        )
        results.append((gate.index, crossing, valid))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "paper_core_config.json")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--initial-state-id", type=int, default=8)
    parser.add_argument("--environment-id", type=int, default=38)
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--auto-launch", action="store_true")
    args = parser.parse_args()

    config, config_path = _load_config(args.config)
    config = deepcopy(config)
    base = config_path.parent
    dataset_root = args.dataset_root.expanduser().resolve()
    states = load_initial_states(dataset_root / "initial_states.json")
    environments = load_training_environments(dataset_root / "environments.json")
    if not 0 <= args.initial_state_id < len(states):
        raise ValueError(f"initial-state-id must be in 0..{len(states) - 1}")
    if not 0 <= args.environment_id < len(environments):
        raise ValueError(f"environment-id must be in 0..{len(environments) - 1}")
    if args.duration <= 0.0:
        raise ValueError("duration must be positive")

    dt = float(config["timing"]["dt"])
    steps = int(round(args.duration / dt))
    config["timing"].update(
        steps_per_episode=steps,
        episode_seconds=steps * dt,
        episode_timeout_seconds=max(float(config["timing"]["episode_timeout_seconds"]), args.duration + 15.0),
    )
    config["gate"]["spawn_in_simulation"] = True
    config["airsim"]["null_rhi"] = False

    state = states[args.initial_state_id]
    environment = environments[args.environment_id]
    manager = None
    try:
        if args.auto_launch:
            airsim_cfg = config["airsim"]
            launch = AirSimLaunchConfig.from_paths(
                airsim_cfg["editor"],
                airsim_cfg["uproject"],
                map_asset=airsim_cfg.get("map_asset"),
                log_dir=_resolve(config.get("log_dir", "logs"), base),
                startup_timeout=float(airsim_cfg["startup_timeout"]),
                retry_interval=float(airsim_cfg["retry_interval"]),
                rpc_host=str(airsim_cfg.get("rpc_host", "127.0.0.1")),
                rpc_port=int(airsim_cfg.get("rpc_port", 41451)),
                rpc_timeout=float(airsim_cfg.get("rpc_timeout", 10.0)),
                res_x=int(airsim_cfg["res_x"]),
                res_y=int(airsim_cfg["res_y"]),
                null_rhi=False,
            )
            manager = AirSimProcessManager(launch)
            client = manager.start_and_wait()
        else:
            client = _connect(config["airsim"])

        # Remove only gate objects generated by this repository. This avoids
        # repeating the contaminated-scene failure diagnosed during resume.
        cleanup_generated_gates(client, prefix="FalconGate_")
        cleanup_generated_gates(client, prefix="FalconPaperGate_")
        controller = _build_controller(config, base)
        planner = QuinticMotionPrimitivePlanner(MotionPrimitiveConfig(**config["planner"]))
        print(
            f"Teacher demo: ACADO/qpOASES, initial_state={args.initial_state_id}, "
            f"environment={args.environment_id}, difficulty={environment.difficulty}, "
            f"epsilon={environment.epsilon}, speed={config['planner']['nominal_speed']:.1f}m/s, "
            f"gates={len(environment.gates)}, duration={steps * dt:.1f}s",
            flush=True,
        )
        episode = run_episode(
            client,
            state,
            environment,
            episode_id=-1,
            config=config,
            calibration=load_calibration(_resolve(config["calibration"], base)),
            planner=planner,
            controller=controller,
        )
        crossings = _count_valid_crossings(episode.position, environment, config["gate"])
        for gate_index, crossing, valid in crossings:
            if crossing is None:
                print(f"gate {gate_index}: NOT CROSSED")
            else:
                print(
                    f"gate {gate_index}: {'PASS' if valid else 'FAIL'} "
                    f"lateral={crossing.lateral_offset:.3f}m "
                    f"vertical={crossing.vertical_offset:.3f}m"
                )
        print(
            f"Demo complete: valid_crossings={sum(item[2] for item in crossings)}/"
            f"{len(crossings)}, samples={len(episode.X)}, "
            f"deadline_misses={int(np.sum(episode.deadline_miss))}, "
            f"mean_solve={np.mean(episode.mpc_solve_time) * 1e3:.3f}ms"
        )
        time.sleep(1.0)
    finally:
        if manager is not None:
            manager.stop()


if __name__ == "__main__":
    main()
