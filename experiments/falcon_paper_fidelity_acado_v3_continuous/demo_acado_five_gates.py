"""Render the v3 ACADO teacher following one global trajectory through five gates."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
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
DEFAULT_LAYOUT_ROOT = (
    HERE.parent / "falcon_paper_fidelity_acado_v2" / "paper_core_40k_acado_v2"
)


def _crossing_results(positions, velocities, environment, gate_config):
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
    trajectory = np.asarray(positions, dtype=np.float64)
    speed = np.linalg.norm(np.asarray(velocities, dtype=np.float64), axis=1)
    results = []
    for gate in environment.gates:
        crossing = None
        crossing_speed = None
        for index, (previous, current) in enumerate(zip(trajectory[:-1], trajectory[1:])):
            candidate = segment_gate_crossing(previous, current, gate)
            if candidate is not None:
                crossing = candidate
                crossing_speed = float(0.5 * (speed[index] + speed[index + 1]))
                break
        valid = bool(
            crossing is not None
            and abs(crossing.lateral_offset) <= half_width
            and abs(crossing.vertical_offset) <= half_height
        )
        results.append((gate.index, crossing, crossing_speed, valid))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "paper_core_config.json")
    parser.add_argument("--layout-root", type=Path, default=DEFAULT_LAYOUT_ROOT)
    parser.add_argument("--initial-state-id", type=int, default=8)
    parser.add_argument("--environment-id", type=int, default=38)
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--auto-launch", action="store_true")
    parser.add_argument(
        "--preflight-hold-seconds",
        type=float,
        default=0.0,
        help="Keep the simulation paused after spawning visible gates, before flight.",
    )
    parser.add_argument(
        "--postflight-hold-seconds",
        type=float,
        default=1.0,
        help="Keep the script connected after the flight so the final view can be inspected.",
    )
    parser.add_argument(
        "--keep-gates",
        action="store_true",
        help="Leave generated gates in the connected Unreal world after a successful demo.",
    )
    parser.add_argument("--report", type=Path, default=HERE / "hard_demo_report.json")
    args = parser.parse_args()

    config, config_path = _load_config(args.config)
    config = deepcopy(config)
    base = config_path.parent
    layout_root = args.layout_root.expanduser().resolve()
    states = load_initial_states(layout_root / "initial_states.json")
    environments = load_training_environments(layout_root / "environments.json")
    if not 0 <= args.initial_state_id < len(states):
        raise ValueError(f"initial-state-id must be in 0..{len(states) - 1}")
    if not 0 <= args.environment_id < len(environments):
        raise ValueError(f"environment-id must be in 0..{len(environments) - 1}")
    if args.duration <= 0.0:
        raise ValueError("duration must be positive")
    if args.preflight_hold_seconds < 0.0:
        raise ValueError("preflight-hold-seconds must be non-negative")
    if args.postflight_hold_seconds < 0.0:
        raise ValueError("postflight-hold-seconds must be non-negative")

    dt = float(config["timing"]["dt"])
    steps = int(round(args.duration / dt))
    config["timing"].update(
        steps_per_episode=steps,
        episode_seconds=steps * dt,
        episode_timeout_seconds=max(
            float(config["timing"]["episode_timeout_seconds"]), args.duration + 15.0
        ),
    )
    config["gate"]["spawn_in_simulation"] = True
    config["gate"]["preflight_hold_seconds"] = args.preflight_hold_seconds
    config["gate"]["keep_spawned_gates"] = args.keep_gates
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
        cleanup_generated_gates(client, prefix="FalconGate_")
        cleanup_generated_gates(client, prefix="FalconPaperGate_")
        controller = _build_controller(config, base)
        planner = QuinticMotionPrimitivePlanner(MotionPrimitiveConfig(**config["planner"]))
        print(
            f"V3 continuous teacher demo: initial_state={args.initial_state_id}, "
            f"environment={args.environment_id}, difficulty={environment.difficulty}, "
            f"speed={config['planner']['nominal_speed']:.1f}m/s, "
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
        results = _crossing_results(
            episode.position, episode.velocity, environment, config["gate"]
        )
        for gate_index, crossing, crossing_speed, valid in results:
            if crossing is None:
                print(f"gate {gate_index}: NOT CROSSED")
            else:
                print(
                    f"gate {gate_index}: {'PASS' if valid else 'FAIL'} "
                    f"speed={crossing_speed:.2f}m/s "
                    f"lateral={crossing.lateral_offset:.3f}m "
                    f"vertical={crossing.vertical_offset:.3f}m"
                )
        report = {
            "status": "PASS" if all(item[3] for item in results) else "FAIL",
            "controller": "ACADO/qpOASES",
            "trajectory": "global_continuous_C2_quintic",
            "initial_state_id": args.initial_state_id,
            "environment_id": args.environment_id,
            "difficulty": environment.difficulty,
            "nominal_speed_mps": float(config["planner"]["nominal_speed"]),
            "valid_crossings": int(sum(item[3] for item in results)),
            "gate_count": len(results),
            "crossings": [
                {
                    "gate": gate_index,
                    "valid": valid,
                    "speed_mps": crossing_speed,
                    "lateral_m": None if crossing is None else crossing.lateral_offset,
                    "vertical_m": None if crossing is None else crossing.vertical_offset,
                }
                for gate_index, crossing, crossing_speed, valid in results
            ],
            "samples": len(episode.X),
            "deadline_misses": int(np.sum(episode.deadline_miss)),
            "mean_solve_ms": float(np.mean(episode.mpc_solve_time) * 1e3),
            "max_solve_ms": float(np.max(episode.mpc_solve_time) * 1e3),
            "final_speed_mps": float(np.linalg.norm(episode.velocity[-1])),
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(
            f"Demo complete: valid_crossings={sum(item[3] for item in results)}/"
            f"{len(results)}, samples={len(episode.X)}, "
            f"deadline_misses={int(np.sum(episode.deadline_miss))}, "
            f"mean_solve={np.mean(episode.mpc_solve_time) * 1e3:.3f}ms"
        )
        print(f"Report: {args.report.resolve()}")
        if args.keep_gates:
            print("Keeping visible gates in the connected Unreal world (--keep-gates).")
        if args.postflight_hold_seconds > 0.0:
            print(
                f"Holding the final view for {args.postflight_hold_seconds:.1f}s.",
                flush=True,
            )
            time.sleep(args.postflight_hold_seconds)
    finally:
        if manager is not None:
            manager.stop()


if __name__ == "__main__":
    main()
