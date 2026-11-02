"""Run safe MPC-actuated, six-model FPGA shadow trials for historical 5 m/s Track-P.

Every 50-ms cycle applies only the fast linear MPC teacher to AirSim.  All six
FLCN/v2 FPGA models receive the exact same gate-relative 12-D state in serial,
using each model's own frozen normalization.  FPGA replies are logged but are
never used as an actuator command in this phase.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
BENCHMARK_ROOT = HERE.parent / "physical_benchmark_v1"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from airsim_interface import AirSimDrone, require_airsim, vehicle_local_to_world_offset
from config import ThrustCalibration, load_calibration
from control_adapter import send_control
from dnn_model_utils import load_normalization, normalized_input
from environment import GateEnvironment
from falcon12_fpga_udp_client import Falcon12FpgaUdpClient
from gate_route import ContinuousGateRoute, GatePassageError
from gate_visualization import cleanup_generated_gates, spawn_gate_sequence
from initial_state_generator import InitialState, load_initial_states
from mpc_controller import MPCController
from state_adapter import build_controller_state
from benchmark_utils import load_config, regression_metrics, resolve_project_path, select_scenarios


MODEL_IDS = {
    "depth10_width512": 1,
    "depth10_width1024": 2,
    "depth22_width512": 3,
    "depth22_width1024": 4,
    "depth34_width512": 5,
    "depth34_width1024": 6,
}


def _join(future: object) -> None:
    join = getattr(future, "join", None)
    if callable(join):
        join()


def _timestamp_us(raw: object | None = None) -> int:
    timestamp_ns = getattr(raw, "timestamp_ns", None) if raw is not None else None
    if timestamp_ns is not None and int(timestamp_ns) > 0:
        return int(timestamp_ns) // 1_000
    return time.perf_counter_ns() // 1_000


def _distribution(values: list[float | int]) -> dict[str, float | int] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
    }


def _pose(airsim: object, state: InitialState) -> object:
    return airsim.Pose(
        position_val=airsim.Vector3r(*[float(value) for value in state.position]),
        orientation_val=airsim.Quaternionr(
            x_val=float(state.orientation[1]),
            y_val=float(state.orientation[2]),
            z_val=float(state.orientation[3]),
            w_val=float(state.orientation[0]),
        ),
    )


def _load_shadow_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "format_version", "protocol", "host", "port", "control_period_s",
        "udp_timeout_ms", "retries", "warmup_requests_per_model", "teacher_policy",
        "actuator_source", "input_mode", "output_mode", "fallback_policy",
    }
    missing = sorted(required.difference(value))
    if missing:
        raise ValueError(f"shadow configuration misses keys: {missing}")
    if int(value["format_version"]) != 1 or value["protocol"] != "FLCN/v2":
        raise ValueError("only format_version=1 / protocol=FLCN/v2 is supported")
    if float(value["control_period_s"]) != 0.05:
        raise ValueError("frozen historical Track-P shadow period must be exactly 0.05 s")
    if value["actuator_source"] != "linear_mpc_teacher_only":
        raise ValueError("shadow mode must apply only the linear MPC teacher")
    return value


def _check_collision(client: object, drone: AirSimDrone, *, first: bool) -> tuple[bool, str | None]:
    info = client.simGetCollisionInfo(vehicle_name=drone.vehicle_name)
    if not bool(getattr(info, "has_collided", False)):
        return False, None
    name = str(getattr(info, "object_name", "unknown"))
    # AirSim commonly retains ground contact from reset/pose placement for one
    # sample.  Record it, but do not abort a powered airborne initial state.
    return (False, name) if first else (True, name)


def _model_table(benchmark: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in benchmark["models"]:
        name = str(item["name"])
        if name not in MODEL_IDS:
            raise ValueError(f"unknown historical model name: {name}")
        normalization_path = resolve_project_path(item["normalization"])
        rows.append({
            "model_id": MODEL_IDS[name],
            "model_name": name,
            # "normalization_path": normalization_path,
            "normalization_path": str(normalization_path),
            "normalization": load_normalization(normalization_path, expected_input_dim=12),
        })
    if [item["model_id"] for item in rows] != list(range(1, 7)):
        raise ValueError("benchmark model order must match frozen FLCN/v2 model IDs 1..6")
    return rows


def _scenario_run(
    *,
    client: object,
    airsim: object,
    scenario: dict[str, Any],
    benchmark: dict[str, Any],
    models: list[dict[str, Any]],
    udp_client: Falcon12FpgaUdpClient,
    shadow: dict[str, Any],
    calibration: ThrustCalibration,
    vehicle_name: str,
    initial_states: dict[int, InitialState],
    output_handle: Any,
    experiment_id: str,
    keep_gates: bool,
) -> dict[str, Any]:
    gate_config = benchmark["gate"]
    timing = benchmark["timing"]
    state_id = int(scenario["initial_state_id"])
    initial = initial_states[state_id]
    client.reset()
    time.sleep(0.5)
    world_offset = vehicle_local_to_world_offset(client, vehicle_name)
    cleanup_generated_gates(client, prefix="FalconGate_")
    cleanup_generated_gates(client, prefix="FalconDatasetGate_")
    environment = GateEnvironment(
        epsilon=float(scenario["epsilon"]),
        seed=int(scenario["seed"]),
        base_y=float(gate_config["base_y"]),
        base_z=float(gate_config["base_z"]),
        max_z=float(gate_config["max_z"]),
        gate_count=int(gate_config["count"]),
        x_start=float(gate_config["x_start"]),
        x_spacing=float(gate_config["x_spacing"]),
    )
    gates = environment.generate()
    scene = spawn_gate_sequence(
        client,
        gates,
        width=float(gate_config["width"]),
        height=float(gate_config["height"]),
        thickness=float(gate_config["thickness"]),
        asset_name=str(gate_config["asset"]),
        world_offset=world_offset,
    )
    drone = AirSimDrone(client, vehicle_name)
    teacher = MPCController(
        dt=float(timing["dt"]),
        horizon=int(timing["horizon"]),
        max_horizontal_accel=float(benchmark["mpc"]["max_horizontal_accel"]),
        max_thrust_accel=float(benchmark["mpc"]["max_thrust_accel"]),
        max_body_rate=float(benchmark["mpc"]["max_body_rate"]),
    )
    period_s = float(shadow["control_period_s"])
    report: dict[str, Any] = {
        "scenario_id": str(scenario["id"]), "epsilon": float(scenario["epsilon"]),
        "seed": int(scenario["seed"]), "initial_state_id": state_id,
        "samples": 0, "crossings": [], "completed": False, "error": None,
        "collision": None, "full_sweep_us": [], "late_cycles": 0,
    }
    try:
        client.simPause(True)
        client.simSetVehiclePose(_pose(airsim, initial), ignore_collision=True, vehicle_name=vehicle_name)
        drone.enable_api_control(True)
        drone.arm()
        client.simPause(False)
        _join(drone.hover())
        time.sleep(0.1)
        started = time.perf_counter()
        initial_raw = drone.get_raw_state()
        route = ContinuousGateRoute(
            gates, initial_raw.position,
            approach_distance=float(gate_config["approach_distance"]),
            exit_distance=float(gate_config["exit_distance"]),
            reference_speed=float(benchmark["mpc"]["reference_speed"]),
            start_time=started,
            gate_width=float(gate_config["width"]), gate_height=float(gate_config["height"]),
            gate_thickness=float(gate_config["thickness"]), clearance=float(gate_config["clearance"]),
            completion_radius=float(gate_config["completion_radius"]),
        )
        # Warm every board kernel before counting 50-ms cycle timing.
        warm_x = build_controller_state(initial_raw, route.current_gate.center, route.current_gate.orientation)
        for model in models:
            x_norm = normalized_input(warm_x[None, :], model["normalization"])[0].astype(np.float32, copy=False)
            results = udp_client.warmup(x_norm, count=int(shadow["warmup_requests_per_model"]), model_id=int(model["model_id"]))
            if any(not item.fresh for item in results):
                raise RuntimeError(f"FPGA warm-up failed for {model['model_name']}")
        started = time.perf_counter()
        route.start_time = started
        route.gate_start_time = started
        deadline = started
        cycle = 0
        while not route.completed:
            cycle_start = time.perf_counter()
            if cycle_start - route.gate_start_time > float(timing["per_gate_timeout_seconds"]):
                raise TimeoutError(f"gate {route.target_gate_id} was not crossed within timeout")
            raw = drone.get_raw_state()
            collided, object_name = _check_collision(client, drone, first=(cycle == 0))
            if collided:
                report["collision"] = object_name
                raise RuntimeError(f"AirSim collision with {object_name}")
            crossing = route.update(raw.position, cycle_start)
            if crossing is not None:
                report["crossings"].append({"gate": crossing.gate_index, "lateral": crossing.lateral_offset, "vertical": crossing.vertical_offset})
            if route.completed:
                break
            window = route.reference_window(cycle_start, dt=teacher.dt, horizon=teacher.horizon)
            reference = window[0]
            teacher_state = build_controller_state(raw, reference.position, reference.orientation)
            teacher_control = np.asarray(teacher.solve(teacher_state, window), dtype=np.float64)
            x_physical = build_controller_state(raw, route.current_gate.center, route.current_gate.orientation).astype(np.float32)
            state_timestamp_us = _timestamp_us(raw)
            sweep_start = time.perf_counter_ns()
            pending_rows: list[dict[str, Any]] = []
            for model in models:
                x_normalized = normalized_input(x_physical[None, :], model["normalization"])[0].astype(np.float32, copy=False)
                result = udp_client.infer_normalized(x_normalized, model_id=int(model["model_id"]), cycle_index=cycle)
                pending_rows.append({
                    "experiment_id": experiment_id,
                    "scenario_id": str(scenario["id"]),
                    "epsilon": float(scenario["epsilon"]),
                    "seed": int(scenario["seed"]),
                    "initial_state_id": state_id,
                    "cycle_index": cycle,
                    "state_timestamp_us": state_timestamp_us,
                    "X_physical": x_physical.tolist(),
                    "X_normalized": x_normalized.tolist(),
                    "U_teacher": teacher_control.tolist(),
                    "model_id": int(model["model_id"]),
                    "model_name": model["model_name"],
                    "sequence": int(result.sequence),
                    "request_timestamp_us": int(result.request_send_timestamp_us),
                    "reply_timestamp_us": result.response_receive_timestamp_us,
                    "U_fpga_physical": None if result.outputs_physical is None else result.outputs_physical.tolist(),
                    "rtt_us": result.rtt_us,
                    "board_dpu_plus_io_us": result.dpu_time_us,
                    "board_server_total_us": result.server_total_us,
                    "retry_count": int(result.retries),
                    "fresh_reply": bool(result.fresh),
                    "timeout": bool(not result.fresh and (result.error or "").startswith("timeout")),
                    "protocol_error": bool(not result.fresh and not (result.error or "").startswith("timeout")),
                    "error": result.error,
                    "period_ms": 50,
                    "teacher_applied": True,
                })
            sweep_us = (time.perf_counter_ns() - sweep_start) // 1_000
            deadline += period_s
            late = time.perf_counter() > deadline
            for row in pending_rows:
                row["full_six_request_sweep_us"] = int(sweep_us)
                row["late_after_period"] = bool(late)
                output_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            output_handle.flush()
            send_control(drone, teacher_control, calibration, duration=period_s)
            report["samples"] += 1
            report["full_sweep_us"].append(int(sweep_us))
            report["late_cycles"] += int(late)
            if cycle % 20 == 0:
                print(f"shadow {scenario['id']} cycle={cycle:04d} gate={route.target_gate_id} sweep={sweep_us / 1000:.2f}ms late={late}", flush=True)
            cycle += 1
            delay = deadline - time.perf_counter()
            if delay > 0.0:
                time.sleep(delay)
        report["completed"] = True
        report["elapsed_seconds"] = time.perf_counter() - started
        return report
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["elapsed_seconds"] = None
        return report
    finally:
        try:
            _join(drone.hover())
            _join(drone.land())
            drone.disarm()
            drone.enable_api_control(False)
        finally:
            if not keep_gates:
                scene.destroy()


def _summary(rows_path: Path, scenario_reports: list[dict[str, Any]], models: list[dict[str, Any]], client: Falcon12FpgaUdpClient) -> dict[str, Any]:
    by_model: dict[int, list[dict[str, Any]]] = defaultdict(list)
    sweeps: list[int] = []
    with rows_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            by_model[int(row["model_id"])].append(row)
            if int(row["model_id"]) == 1:
                sweeps.append(int(row["full_six_request_sweep_us"]))
    per_model: list[dict[str, Any]] = []
    for model in models:
        rows = by_model[int(model["model_id"])]
        fresh = [row for row in rows if row["fresh_reply"] and row["U_fpga_physical"] is not None]
        metrics = None
        if fresh:
            metrics = regression_metrics(
                np.asarray([row["U_fpga_physical"] for row in fresh]),
                np.asarray([row["U_teacher"] for row in fresh]),
            )
        per_model.append({
            "model_id": model["model_id"], "model_name": model["model_name"],
            "cycles": len(rows), "fresh_replies": len(fresh),
            "timeouts": sum(int(row["timeout"]) for row in rows),
            "retries": sum(int(row["retry_count"]) for row in rows),
            "protocol_errors": sum(int(row["protocol_error"]) for row in rows),
            "late_after_period": sum(int(row["late_after_period"]) for row in rows),
            "fpga_vs_teacher": metrics,
            "rtt_us": _distribution([row["rtt_us"] for row in fresh if row["rtt_us"] is not None]),
            "board_dpu_plus_io_us": _distribution([row["board_dpu_plus_io_us"] for row in fresh if row["board_dpu_plus_io_us"] is not None]),
        })
    return {
        "format_version": 1,
        "phase": "MPC-actuated FPGA shadow only",
        "rows": sum(len(value) for value in by_model.values()),
        "scenarios": scenario_reports,
        "per_model": per_model,
        "full_six_request_sweep_us": _distribution(sweeps),
        "client_stats": dict(client.stats),
        "important_interpretation": "FPGA-vs-teacher compares INT8 DNN outputs against MPC outputs. It is not a golden-vector accuracy result and is expected to be nonzero.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-config", type=Path, default=BENCHMARK_ROOT / "benchmark_config.json")
    parser.add_argument("--fpga-config", type=Path, default=HERE / "fpga_shadow_config.json")
    parser.add_argument("--scenario-set", default="pilot", help="pilot, full, or comma-separated IDs")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--vehicle-name", default="Drone1")
    parser.add_argument("--calibration", default=None)
    parser.add_argument("--keep-gates", action="store_true")
    args = parser.parse_args()

    benchmark = load_config(args.benchmark_config)
    shadow = _load_shadow_config(args.fpga_config)
    scenarios = select_scenarios(benchmark, args.scenario_set)
    models = _model_table(benchmark)
    run_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = (args.output_dir or HERE / "results" / f"shadow_{args.scenario_set}_{run_tag}").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "shadow_cycles.jsonl"
    states = {item.initial_state_id: item for item in load_initial_states(resolve_project_path(benchmark["initial_states"]))}
    missing = [int(item["initial_state_id"]) for item in scenarios if int(item["initial_state_id"]) not in states]
    if missing:
        raise ValueError(f"benchmark initial states are missing: {missing}")
    airsim = require_airsim()
    client = airsim.MultirotorClient()
    client.confirmConnection()
    if args.vehicle_name not in client.listVehicles():
        raise ValueError(f"AirSim vehicle {args.vehicle_name!r} is unavailable: {list(client.listVehicles())}")
    manifest = {
        "format_version": 1, "experiment_id": f"historical5ms_shadow_{run_tag}",
        "created_utc": datetime.now(timezone.utc).isoformat(), "benchmark_config": str(args.benchmark_config.resolve()),
        "fpga_config": str(args.fpga_config.resolve()), "endpoint": {"host": shadow["host"], "port": shadow["port"]},
        "protocol": "FLCN/v2", "period_ms": 50, "teacher_applied": True,
        "actuator_source": shadow["actuator_source"], "fallback_policy": shadow["fallback_policy"],
        "scenario_set": args.scenario_set, "scenarios": scenarios,
        "models": [{key: value for key, value in item.items() if key != "normalization"} for item in models],
        "deviation": "Serial six-request sweep; its measured duration and late cycles are retained in the log.",
    }
    (output_dir / "shadow_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    reports: list[dict[str, Any]] = []
    with Falcon12FpgaUdpClient(host=str(shadow["host"]), port=int(shadow["port"]), timeout_ms=float(shadow["udp_timeout_ms"]), retries=int(shadow["retries"]), model_id=1) as udp_client:
        # Close the writer before reopening the JSONL for summary aggregation;
        # this avoids Windows file-sharing ambiguity and ensures every row is
        # durable even when a scenario reports an error.
        with rows_path.open("w", encoding="utf-8") as output_handle:
            for scenario in scenarios:
                print(f"\n=== FPGA shadow {scenario['id']} / epsilon={scenario['epsilon']} ===", flush=True)
                reports.append(_scenario_run(
                    client=client, airsim=airsim, scenario=scenario, benchmark=benchmark, models=models,
                    udp_client=udp_client, shadow=shadow, calibration=load_calibration(args.calibration),
                    vehicle_name=args.vehicle_name, initial_states=states, output_handle=output_handle,
                    experiment_id=manifest["experiment_id"], keep_gates=args.keep_gates,
                ))
        summary = _summary(rows_path, reports, models, udp_client)
    (output_dir / "shadow_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"Output directory: {output_dir}", flush=True)
    if any(not item["completed"] for item in reports):
        raise SystemExit("one or more shadow scenarios failed; inspect shadow_summary.json")


if __name__ == "__main__":
    main()
