"""Compare the DNN policy with the MPC teacher offline and in AirSim.

Offline comparison uses the fixed held-out test split.  Optional AirSim
comparison executes both controllers on the same held-out layouts while
retaining the same ContinuousGateRoute and reference generator.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from dnn_controller import DNNController
from dnn_model_utils import load_normalization, normalized_input, denormalized_output
from evaluate_dnn_closed_loop import _run_episode
from airsim_interface import require_airsim, vehicle_local_to_world_offset
from airsim_process import AirSimLaunchConfig, AirSimProcessManager
from config import load_calibration
from initial_state_generator import load_initial_states
from mpc_controller import MPCController
from training_environment_generator import load_training_environments


def _offline_without_batch_api(model: Path, data_dir: Path, device: str) -> dict[str, object]:
    """Use one vectorized PyTorch call for the offline report."""
    import torch

    normalization = load_normalization(data_dir / "normalization.json")
    input_dim = int(normalization["x_mean"].size)
    with np.load(data_dir / "test.npz", allow_pickle=False) as archive:
        x = np.asarray(archive["X"], dtype=np.float32)
        target = np.asarray(archive["U"], dtype=np.float32)
        gates = np.asarray(archive["target_gate_id"], dtype=np.int64)
    try:
        network = torch.jit.load(str(model), map_location=device)
    except RuntimeError:
        checkpoint = torch.load(model, map_location=device, weights_only=False)
        from dnn_model_utils import build_model

        network = build_model(
            torch,
            depth=int(checkpoint["depth"]),
            width=int(checkpoint["width"]),
            input_dim=int(checkpoint.get("input_dim", 12)),
        ).to(device).eval()
        network.load_state_dict(checkpoint["model_state"])
    network = network.to(device).eval()
    x_norm = normalized_input(x, normalization)
    if x.shape[1:] != (input_dim,):
        raise ValueError(f"offline data input shape {x.shape} does not match normalization")
    with torch.no_grad():
        predicted_norm = network(torch.from_numpy(x_norm).to(device)).cpu().numpy()
    prediction = denormalized_output(predicted_norm, normalization)
    error = prediction - target
    names = ("collective_thrust_acceleration", "body_rate_x", "body_rate_y", "body_rate_z")
    per_gate = {}
    for gate in sorted(np.unique(gates).astype(int).tolist()):
        mask = gates == gate
        per_gate[str(gate)] = {"samples": int(mask.sum()), "mse": float(np.mean(error[mask] ** 2)), "mae": float(np.mean(np.abs(error[mask])))}
    return {
        "samples": int(len(x)),
        "mse": float(np.mean(error ** 2)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "max_abs_error": float(np.max(np.abs(error))),
        "mse_per_output": {name: float(value) for name, value in zip(names, np.mean(error ** 2, axis=0))},
        "mae_per_output": {name: float(value) for name, value in zip(names, np.mean(np.abs(error), axis=0))},
        "per_gate": per_gate,
    }


def _closed_loop(args: argparse.Namespace) -> dict[str, object]:
    environments = {item.environment_id: item for item in load_training_environments(args.layout_root / "environments.json")}
    states = {item.initial_state_id: item for item in load_initial_states(args.layout_root / "initial_states.json")}
    env_ids = [int(value) for value in args.environment_ids.split(",") if value.strip()]
    state_ids = [int(value) for value in args.initial_state_ids.split(",") if value.strip()]
    selected = [(environments[e], states[s]) for s in state_ids for e in env_ids]
    if args.max_episodes > 0:
        selected = selected[:args.max_episodes]
    manager = args.process_manager
    if manager is not None:
        client = manager.start_and_wait()
    else:
        airsim = require_airsim()
        client = airsim.MultirotorClient()
        client.confirmConnection()
    client.reset()
    time.sleep(0.5)
    vehicle_name = args.vehicle_name
    if not vehicle_name:
        names = list(client.listVehicles())
        vehicle_name = str(names[0])
    offset = vehicle_local_to_world_offset(client, vehicle_name)
    controllers = {
        "MPC": MPCController(dt=0.05, horizon=20),
        "DNN": DNNController(args.model, args.normalization, device=args.device),
    }
    rows = []
    started = time.perf_counter()
    try:
        for environment, state in selected:
            for name, controller in controllers.items():
                print(f"controller={name} environment={environment.environment_id} initial_state={state.initial_state_id}", flush=True)
                row = _run_episode(
                    client,
                    controller,
                    environment,
                    state,
                    vehicle_name=vehicle_name,
                    world_offset=offset,
                    calibration=args.calibration,
                    gate_width=args.gate_width,
                    gate_height=args.gate_height,
                    gate_thickness=args.gate_thickness,
                    gate_asset=args.gate_asset,
                    gate_clearance=args.gate_clearance,
                    speed=args.speed,
                    threshold=args.threshold,
                    per_gate_timeout=args.per_gate_timeout,
                )
                row["controller"] = name
                rows.append(row)
                print(f"  success={row['success']} crossings={len(row['crossings'])}/{len(environment.gates)} error={row['error']}", flush=True)
    finally:
        if manager is not None:
            manager.stop()
    result = {
        "controllers": {name: {
            "episodes": int(sum(row["controller"] == name for row in rows)),
            "successes": int(sum(row["controller"] == name and row["success"] for row in rows)),
            "success_rate": float(np.mean([row["success"] for row in rows if row["controller"] == name])) if any(row["controller"] == name for row in rows) else 0.0,
        } for name in controllers},
        "episodes_requested_per_controller": len(selected),
        "elapsed_seconds": time.perf_counter() - started,
        "results": rows,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("dnn_dataset_full_route_v2"))
    parser.add_argument("--normalization", type=Path, default=None)
    parser.add_argument("--offline-only", action="store_true")
    parser.add_argument("--layout-root", type=Path, default=Path("teacher_dataset_full_route_continuous_v2"))
    parser.add_argument("--environment-ids", default="36,37,38,39")
    parser.add_argument("--initial-state-ids", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--vehicle-name", default="Drone1")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--per-gate-timeout", type=float, default=30.0)
    parser.add_argument("--speed", type=float, default=2.0)
    parser.add_argument("--gate-width", type=float, default=4.0)
    parser.add_argument("--gate-height", type=float, default=4.0)
    parser.add_argument("--gate-thickness", type=float, default=0.15)
    parser.add_argument("--gate-clearance", type=float, default=0.25)
    parser.add_argument("--gate-asset", default="1M_Cube")
    parser.add_argument("--calibration", type=load_calibration, default=load_calibration())
    parser.add_argument("--report", type=Path, default=Path("dnn_runs/depth10_width512/mpc_dnn_comparison.json"))
    parser.add_argument("--auto-launch", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--ue4-editor", type=Path, default=Path(r"E:\epicGamesStrg\UE_4.27\Engine\Binaries\Win64\UE4Editor.exe"))
    parser.add_argument("--uproject", type=Path, default=Path(r"F:\Airsim\AirSim\Unreal\Environments\Blocks\Blocks.uproject"))
    args = parser.parse_args()
    args.process_manager = None
    offline = _offline_without_batch_api(args.model, args.data_dir, args.device)
    result: dict[str, object] = {"offline_test_split": offline}
    if not args.offline_only:
        if args.auto_launch:
            args.process_manager = AirSimProcessManager(
                AirSimLaunchConfig.from_paths(args.ue4_editor, args.uproject, log_dir=Path("logs"), null_rhi=not args.render)
            )
        result["closed_loop"] = _closed_loop(args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Report: {args.report.resolve()}")


if __name__ == "__main__":
    main()
