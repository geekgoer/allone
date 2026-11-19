"""Run explicitly enabled single-model FPGA closed-loop pilot trials.

Unlike ``run_fpga_shadow.py``, this program may actuate an FPGA reply.  One
model is selected per trial, and only a fresh reply received within the current
control period is applied.  Every missing, stale, late, or invalid reply falls
back to the same-cycle linear-MPC teacher command.  Optional delay injection is
applied immediately before AirSim command dispatch.  Alternatively, a target
response time schedules send initiation relative to state acquisition, while
the measured response still includes the AirSim send-call overhead. Invocation requires the
explicit ``--enable-fpga-actuation`` acknowledgement.
"""

from __future__ import annotations

import argparse
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
from dnn_model_utils import normalized_input
from environment import Gate, GateEnvironment
from falcon12_fpga_udp_client import Falcon12FpgaUdpClient
from gate_route import ContinuousGateRoute, GatePassageError
from gate_visualization import cleanup_generated_gates, spawn_gate_sequence
from initial_state_generator import InitialState, load_initial_states
from mpc_controller import MPCController
from state_adapter import build_controller_state, normalize_quaternion
from benchmark_utils import load_config, resolve_project_path, select_scenarios
from run_fpga_shadow import _check_collision, _distribution, _load_shadow_config, _model_table, _pose, _join
from startup_reference import SmoothStartupGateRoute, reference_contract


PAPER_HORIZON_K = 20
PAPER_WINDOW_SAMPLES = PAPER_HORIZON_K + 1
PAPER_METRIC_VERSION = "falcon_eq4_wx_identity_route_mean_v1"


def _paper_state_pair(raw: Any, reference: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Eq. (4) actual/reference states and their squared error."""

    actual_quaternion = normalize_quaternion(raw.quaternion)
    reference_quaternion = normalize_quaternion(reference.orientation)
    if float(np.dot(actual_quaternion, reference_quaternion)) < 0.0:
        reference_quaternion = -reference_quaternion
    actual = np.concatenate((raw.position, raw.velocity, actual_quaternion)).astype(
        np.float64, copy=False
    )
    target = np.concatenate(
        (reference.position, reference.velocity, reference_quaternion)
    ).astype(np.float64, copy=False)
    if actual.shape != (10,) or target.shape != (10,):
        raise ValueError("paper metric states must have shape (10,)")
    if not np.isfinite(actual).all() or not np.isfinite(target).all():
        raise ValueError("paper metric states contain NaN or Inf")
    error = actual - target
    return actual, target, error * error


def _paper_metric_summary(error_squared_samples: list[float]) -> dict[str, Any]:
    """Aggregate complete, non-overlapping K=20 windows into route-level J."""

    values = np.asarray(error_squared_samples, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("paper metric samples must be finite non-negative scalars")
    complete_windows = int(values.size // PAPER_WINDOW_SAMPLES)
    used_samples = complete_windows * PAPER_WINDOW_SAMPLES
    discarded_samples = int(values.size - used_samples)
    window_costs = (
        values[:used_samples].reshape(complete_windows, PAPER_WINDOW_SAMPLES).sum(axis=1)
        if complete_windows
        else np.empty(0, dtype=np.float64)
    )
    route_cost_mean = float(np.mean(window_costs)) if window_costs.size else None
    j_value = (
        float(1.0 / route_cost_mean)
        if route_cost_mean is not None and route_cost_mean > 0.0
        else None
    )
    return {
        "paper_metric_version": PAPER_METRIC_VERSION,
        "paper_horizon_K": PAPER_HORIZON_K,
        "paper_window_samples": PAPER_WINDOW_SAMPLES,
        "paper_state_order": ["px", "py", "pz", "vx", "vy", "vz", "qw", "qx", "qy", "qz"],
        "paper_Wx_diagonal": [1.0] * 10,
        "paper_complete_windows": complete_windows,
        "paper_used_samples": used_samples,
        "paper_discarded_samples": discarded_samples,
        "paper_window_costs": window_costs.tolist(),
        "paper_route_cost_mean": route_cost_mean,
        "J_paper_style": j_value,
    }


def _clip(control: Any) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(control, dtype=np.float64).reshape(4)
    if not np.isfinite(value).all():
        raise ValueError("FPGA reply contains NaN or Inf")
    clipped = value.copy()
    clipped[0] = np.clip(clipped[0], 2.0, 20.0)
    clipped[1:3] = np.clip(clipped[1:3], -3.0, 3.0)
    clipped[3] = np.clip(clipped[3], -2.0, 2.0)
    return clipped, clipped != value


def _trial_key(model_id: int, scenario_id: str, repeat_index: int) -> str:
    safe_scenario = str(scenario_id).replace("/", "_").replace("\\", "_")
    return f"m{int(model_id):02d}_{safe_scenario}_r{int(repeat_index):02d}"


def _response_timing_contract(period_ms, delay_ms, target_ms=None, tolerance_ms=3.0):
    if not np.isfinite([period_ms, delay_ms]).all() or period_ms <= 0 or delay_ms < 0:
        raise ValueError("period must be finite and positive; delay must be finite and nonnegative")
    if target_ms is None:
        return {"mode": "native_plus_added_delay", "actuation_delay_ms": float(delay_ms)}
    if (not np.isfinite([target_ms, tolerance_ms]).all() or target_ms <= 0 or
            tolerance_ms <= 0 or target_ms + tolerance_ms >= period_ms):
        raise ValueError("target response and tolerance must be positive, finite, and sum to less than the period")
    if delay_ms != 0:
        raise ValueError("target response cannot be combined with additive actuation delay")
    return {"mode": "fixed_send_release", "target_response_ms": float(target_ms),
            "tolerance_ms": float(tolerance_ms)}


def _wait_until_us(target_us):
    remaining_us = target_us - time.perf_counter_ns() // 1_000
    while remaining_us > 0:
        time.sleep(remaining_us / 1_000_000.0)
        remaining_us = target_us - time.perf_counter_ns() // 1_000


def _response_target_violation(elapsed_us, target_ms, tolerance_ms):
    return target_ms is not None and abs(elapsed_us / 1000.0 - target_ms) > tolerance_ms + 1e-9


def _scenario_gates(scenario, gate_config):
    if "gate_centers" in scenario:
        centers = np.asarray(scenario["gate_centers"], dtype=np.float64)
        if (centers.shape != (int(gate_config["count"]), 3) or not np.isfinite(centers).all()
                or np.any(np.diff(centers[:, 0]) <= 0)):
            raise ValueError("explicit gate centers must be finite, match gate count, and increase in x")
        return [Gate(i + 1, center, np.array([1., 0., 0., 0.])) for i, center in enumerate(centers)]
    return GateEnvironment(
        epsilon=float(scenario["epsilon"]), seed=int(scenario["seed"]),
        base_y=float(gate_config["base_y"]), base_z=float(gate_config["base_z"]),
        max_z=float(gate_config["max_z"]), gate_count=int(gate_config["count"]),
        x_start=float(gate_config["x_start"]), x_spacing=float(gate_config["x_spacing"]),
    ).generate()


def _latency_summary(trial: dict[str, Any]) -> None:
    values = np.asarray(trial["state_to_command_dispatch_us"], dtype=np.float64)
    if values.size:
        trial["measured_R_max_ms"] = float(np.max(values) / 1000.0)
        trial["measured_R_p95_ms"] = float(np.percentile(values, 95) / 1000.0)
        trial["measured_R_p99_ms"] = float(np.percentile(values, 99) / 1000.0)
    else:
        trial["measured_R_max_ms"] = None
        trial["measured_R_p95_ms"] = None
        trial["measured_R_p99_ms"] = None


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_trial_summaries(summary_dir: Path) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for path in summary_dir.glob("*.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        key = str(value.get("trial_key", path.stem))
        values[key] = value
    return values


def _rebuild_cycle_log(output_dir: Path, plan: list[str]) -> None:
    aggregate = output_dir / "fpga_closed_loop_trials.jsonl"
    temporary = aggregate.with_suffix(aggregate.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as destination:
        for key in plan:
            path = output_dir / "trial_logs" / f"{key}.jsonl"
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as source:
                for line in source:
                    destination.write(line)
    temporary.replace(aggregate)


def _run_trial(
    *,
    client: object,
    airsim: object,
    scenario: dict[str, Any],
    repeat_index: int,
    benchmark: dict[str, Any],
    model: dict[str, Any],
    udp_client: Falcon12FpgaUdpClient,
    shadow: dict[str, Any],
    period_s: float,
    actuation_delay_s: float,
    calibration: ThrustCalibration,
    vehicle_name: str,
    initial_states: dict[int, InitialState],
    output_handle: Any,
    experiment_id: str,
    keep_gates: bool,
    reference_profile: str = "constant",
    startup_duration_s: float = 2.0,
    target_response_ms: float | None = None,
    response_tolerance_ms: float = 3.0,
    max_episode_seconds: float | None = None,
    cycle_log_stride: int = 20,
    reset_after_trial: bool = False,
) -> dict[str, Any]:
    """Run one scenario with one model, applying fresh FPGA replies only."""

    gate_config = dict(benchmark["gate"])
    gate_config.update(scenario.get("gate_overrides", {}))
    timing = benchmark["timing"]
    state_id = int(scenario["initial_state_id"])
    initial = initial_states[state_id]
    client.reset()
    time.sleep(0.5)
    world_offset = vehicle_local_to_world_offset(client, vehicle_name)
    cleanup_generated_gates(client, prefix="FalconGate_")
    cleanup_generated_gates(client, prefix="FalconDatasetGate_")
    gates = _scenario_gates(scenario, gate_config)
    scene = spawn_gate_sequence(
        client, gates, width=float(gate_config["width"]), height=float(gate_config["height"]),
        thickness=float(gate_config["thickness"]), asset_name=str(gate_config["asset"]),
        world_offset=world_offset,
    )
    drone = AirSimDrone(client, vehicle_name)
    teacher = MPCController(
        dt=period_s, horizon=int(timing["horizon"]),
        max_horizontal_accel=float(benchmark["mpc"]["max_horizontal_accel"]),
        max_thrust_accel=float(benchmark["mpc"]["max_thrust_accel"]),
        max_body_rate=float(benchmark["mpc"]["max_body_rate"]),
    )
    trial: dict[str, Any] = {
        "model_id": int(model["model_id"]), "model_name": model["model_name"],
        "scenario_id": str(scenario["id"]), "epsilon": float(scenario["epsilon"]),
        "seed": int(scenario["seed"]), "initial_state_id": state_id, "repeat_index": repeat_index,
        "period_ms": period_s * 1000.0,
        "actuation_delay_ms": actuation_delay_s * 1000.0,
        "samples": 0, "crossings": [], "completed": False, "collision": None, "error": None,
        "fallback_count": 0, "deadline_miss_count": 0, "timeout_count": 0,
        "tracking_cost": 0.0, "max_state_norm": 0.0, "response_us": [],
        "state_to_command_dispatch_us": [],
        "reference_profile": reference_contract(reference_profile, startup_duration_s),
        "response_timing": _response_timing_contract(period_s * 1000, actuation_delay_s * 1000, target_response_ms, response_tolerance_ms),
        "response_target_violation_count": 0,
        "gate_centers": [gate.center.tolist() for gate in gates],
    }
    paper_error_squared_samples: list[float] = []
    raw = None
    trial_started = time.perf_counter()
    try:
        client.simPause(True)
        client.simSetVehiclePose(_pose(airsim, initial), ignore_collision=True, vehicle_name=vehicle_name)
        drone.enable_api_control(True)
        drone.arm()
        client.simPause(False)
        _join(drone.hover())
        time.sleep(0.1)
        started = time.perf_counter()
        raw_initial = drone.get_raw_state()
        route = ContinuousGateRoute(
            gates, raw_initial.position, approach_distance=float(gate_config["approach_distance"]),
            exit_distance=float(gate_config["exit_distance"]), reference_speed=float(benchmark["mpc"]["reference_speed"]),
            start_time=started, gate_width=float(gate_config["width"]),
            gate_height=float(gate_config["height"]),
            gate_thickness=float(gate_config["thickness"]), clearance=float(gate_config["clearance"]),
            completion_radius=float(gate_config["completion_radius"]),
        )
        warm_x = build_controller_state(raw_initial, route.current_gate.center, route.current_gate.orientation)
        warm_norm = normalized_input(warm_x[None, :], model["normalization"])[0].astype(np.float32, copy=False)
        warm = udp_client.warmup(warm_norm, count=int(shadow["warmup_requests_per_model"]), model_id=int(model["model_id"]))
        if any(not item.fresh for item in warm):
            raise RuntimeError("FPGA warm-up did not produce fresh replies")
        if reference_profile == "smooth-start":
            reference_initial = drone.get_raw_state()
            route = SmoothStartupGateRoute(
                gates, reference_initial.position,
                initial_velocity=reference_initial.velocity, startup_duration=startup_duration_s,
                approach_distance=float(gate_config["approach_distance"]),
                exit_distance=float(gate_config["exit_distance"]),
                reference_speed=float(benchmark["mpc"]["reference_speed"]), start_time=started,
                gate_width=float(gate_config["width"]), gate_height=float(gate_config["height"]),
                gate_thickness=float(gate_config["thickness"]), clearance=float(gate_config["clearance"]),
                completion_radius=float(gate_config["completion_radius"]),
            )
            trial["reference_initial_velocity"] = reference_initial.velocity.tolist()
        trial["reference_waypoints"] = route.waypoints.tolist()
        started = time.perf_counter()
        route.start_time = started
        route.gate_start_time = started
        deadline = started
        cycle = 0
        while not route.completed:
            cycle_start = time.perf_counter()
            if max_episode_seconds is not None and cycle_start - started > max_episode_seconds:
                raise TimeoutError(f"episode exceeded {max_episode_seconds}s")
            if cycle_start - route.gate_start_time > float(timing["per_gate_timeout_seconds"]):
                raise TimeoutError(f"gate {route.target_gate_id} was not crossed within timeout")
            raw = drone.get_raw_state()
            state_acquired_us = time.perf_counter_ns() // 1_000
            collided, object_name = _check_collision(client, drone, first=(cycle == 0))
            if collided:
                trial["collision"] = object_name or "unknown"
                raise RuntimeError(f"AirSim collision with {object_name}")
            crossing = route.update(raw.position, cycle_start)
            if crossing is not None:
                trial["crossings"].append({"gate": crossing.gate_index, "lateral": crossing.lateral_offset, "vertical": crossing.vertical_offset})
            if route.completed:
                break
            window = route.reference_window(cycle_start, dt=teacher.dt, horizon=teacher.horizon)
            reference = window[0]
            paper_reference_now = route.reference_at_time(cycle_start)
            paper_actual, paper_reference, paper_error_squared = _paper_state_pair(
                raw, paper_reference_now
            )
            paper_error_squared_sum = float(np.sum(paper_error_squared))
            paper_error_squared_samples.append(paper_error_squared_sum)
            teacher_state = build_controller_state(raw, reference.position, reference.orientation)
            teacher_control = np.asarray(teacher.solve(teacher_state, window), dtype=np.float64)
            x_physical = build_controller_state(raw, route.current_gate.center, route.current_gate.orientation).astype(np.float32)
            x_normalized = normalized_input(x_physical[None, :], model["normalization"])[0].astype(np.float32, copy=False)
            deadline += period_s
            result = udp_client.infer_normalized(x_normalized, model_id=int(model["model_id"]), cycle_index=cycle)
            received_on_time = bool(result.fresh and result.outputs_physical is not None and time.perf_counter() <= deadline)
            fallback = None
            clip_flags = np.zeros(4, dtype=bool)
            if received_on_time:
                command, clip_flags = _clip(result.outputs_physical)
                command_source = "fpga_fresh"
            else:
                command = teacher_control
                command_source = "teacher_fallback"
                fallback = "timeout_or_late_fpga_reply"
                trial["fallback_count"] += 1
            native_ready_us = time.perf_counter_ns() // 1_000
            target_send_us = None
            if target_response_ms is not None:
                target_send_us = state_acquired_us + round(target_response_ms * 1000)
                _wait_until_us(target_send_us)
            elif actuation_delay_s > 0.0:
                time.sleep(actuation_delay_s)
            send_started_us = time.perf_counter_ns() // 1_000
            send_control(drone, command, calibration, duration=period_s)
            dispatch_us = time.perf_counter_ns() // 1_000
            target_violation = _response_target_violation(dispatch_us - state_acquired_us, target_response_ms, response_tolerance_ms)
            trial["response_target_violation_count"] += int(target_violation)
            late = time.perf_counter() > deadline
            trial["deadline_miss_count"] += int(late)
            trial["timeout_count"] += int(not result.fresh)
            if result.rtt_us is not None:
                trial["response_us"].append(int(result.rtt_us))
            trial["state_to_command_dispatch_us"].append(int(dispatch_us - state_acquired_us))
            # Declared engineering tracking cost, retained rather than presented
            # as the paper's undisclosed exact objective.
            position_error = np.asarray(teacher_state[0:3], dtype=np.float64)
            velocity_error = np.asarray(reference.velocity, dtype=np.float64) - raw.velocity
            state_norm = float(np.linalg.norm(np.concatenate((position_error, velocity_error))))
            trial["tracking_cost"] += float((np.dot(position_error, position_error) + 0.25 * np.dot(velocity_error, velocity_error)) * period_s)
            trial["max_state_norm"] = max(float(trial["max_state_norm"]), state_norm)
            row = {
                "experiment_id": experiment_id, "model_id": int(model["model_id"]), "model_name": model["model_name"],
                "scenario_id": str(scenario["id"]), "epsilon": float(scenario["epsilon"]), "seed": int(scenario["seed"]),
                "initial_state_id": state_id, "repeat_index": repeat_index, "cycle_index": cycle,
                "period_ms": period_s * 1000.0, "state_acquired_us": int(state_acquired_us), "X_physical": x_physical.tolist(),
                "airsim_state_timestamp_ns": getattr(raw, "timestamp_ns", None),
                "actuation_delay_ms": actuation_delay_s * 1000.0,
                "X_normalized": x_normalized.tolist(), "U_teacher": teacher_control.tolist(),
                "sequence": int(result.sequence), "request_timestamp_us": int(result.request_send_timestamp_us),
                "reply_timestamp_us": result.response_receive_timestamp_us, "U_fpga_physical": None if result.outputs_physical is None else result.outputs_physical.tolist(),
                "rtt_us": result.rtt_us, "board_dpu_plus_io_us": result.dpu_time_us, "board_server_total_us": result.server_total_us,
                "fresh_reply": bool(result.fresh), "fresh_on_time": received_on_time, "timeout": bool(not result.fresh),
                "retry_count": int(result.retries), "error": result.error, "command_source": command_source,
                "fallback": fallback, "U_applied": command.tolist(), "clip_flags": clip_flags.tolist(),
                "command_dispatch_us": int(dispatch_us), "state_to_command_dispatch_us": int(dispatch_us - state_acquired_us),
                "R_cycle_ms": float(dispatch_us - state_acquired_us) / 1000.0,
                "deadline_miss": late, "tracking_cost_increment": float((np.dot(position_error, position_error) + 0.25 * np.dot(velocity_error, velocity_error)) * period_s),
                "paper_psi_actual": paper_actual.tolist(),
                "paper_psi_reference": paper_reference.tolist(),
                "paper_state_error_squared": paper_error_squared.tolist(),
                "paper_state_error_squared_sum": paper_error_squared_sum,
                "reference_profile": trial["reference_profile"],
                "reference_elapsed_s": float(cycle_start-route.start_time),
                "target_response_ms": target_response_ms,
                "response_tolerance_ms": response_tolerance_ms if target_response_ms is not None else None,
                "native_ready_us": int(native_ready_us), "target_send_us": target_send_us,
                "command_send_started_us": int(send_started_us),
                "response_target_violation": bool(target_violation),
            }
            output_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            output_handle.flush()
            if cycle_log_stride > 0 and cycle % cycle_log_stride == 0:
                print(f"closed-loop {model['model_name']} {scenario['id']} r={repeat_index} cycle={cycle:04d} source={command_source} rtt={result.rtt_us}us late={late}", flush=True)
            trial["samples"] += 1
            cycle += 1
            pause = deadline - time.perf_counter()
            if pause > 0.0:
                time.sleep(pause)
        trial["completed"] = True
        trial["elapsed_seconds"] = time.perf_counter() - started
        trial.update(_paper_metric_summary(paper_error_squared_samples))
        _latency_summary(trial)
        return trial
    except Exception as exc:
        trial["error"] = f"{type(exc).__name__}: {exc}"
        if hasattr(exc, "input_range_violation"):
            trial["input_range_violation"] = exc.input_range_violation
        trial["elapsed_seconds"] = time.perf_counter() - trial_started
        trial.update(_paper_metric_summary(paper_error_squared_samples))
        _latency_summary(trial)
        return trial
    finally:
        if raw is not None:
            trial["terminal_state"] = {"position": raw.position.tolist(), "velocity": raw.velocity.tolist(), "quaternion": raw.quaternion.tolist()}
        try:
            if reset_after_trial:
                client.reset()
            else:
                _join(drone.hover())
                _join(drone.land())
                drone.disarm()
                drone.enable_api_control(False)
        except Exception as exc:
            trial["cleanup_error"] = f"{type(exc).__name__}: {exc}"
        finally:
            if not keep_gates:
                scene.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enable-fpga-actuation", action="store_true", help="required acknowledgement before any FPGA output can command AirSim")
    parser.add_argument("--benchmark-config", type=Path, default=BENCHMARK_ROOT / "benchmark_config.json")
    parser.add_argument("--fpga-config", type=Path, default=HERE / "fpga_shadow_config.json")
    parser.add_argument("--scenario-set", default="pilot")
    parser.add_argument("--model-ids", default="1,2,3,4,5,6")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--period-ms", type=int, default=None, help="override the control period; default is the FPGA config value")
    parser.add_argument("--actuation-delay-ms", type=float, default=0.0, help="extra delay before sending the selected command to AirSim")
    parser.add_argument("--target-response-ms", type=float, default=None, help="schedule send initiation at state-acquisition + target; measured R still includes the send call")
    parser.add_argument("--response-tolerance-ms", type=float, default=3.0, help="allowed measured-R deviation from the target, recorded per cycle")
    parser.add_argument("--resume", action="store_true", help="resume completed trials in an existing output directory")
    parser.add_argument("--vehicle-name", default="Drone1")
    parser.add_argument("--calibration", default=None)
    parser.add_argument("--keep-gates", action="store_true")
    parser.add_argument("--reference-profile", choices=("constant", "smooth-start"), default="constant")
    parser.add_argument("--startup-duration-s", type=float, default=2.0)
    args = parser.parse_args()
    reference_profile_manifest = reference_contract(args.reference_profile, args.startup_duration_s)
    if not args.enable_fpga_actuation:
        raise SystemExit("refusing FPGA actuation: add --enable-fpga-actuation after reviewing the shadow acceptance")
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    selected_ids = tuple(int(value.strip()) for value in args.model_ids.split(",") if value.strip())
    if not selected_ids or any(value not in range(1, 7) for value in selected_ids):
        raise ValueError("--model-ids must be a nonempty subset of 1,2,3,4,5,6")
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("--model-ids must not contain duplicates")
    benchmark = load_config(args.benchmark_config)
    shadow = _load_shadow_config(args.fpga_config)
    period_ms = int(round(float(shadow["control_period_s"]) * 1000.0)) if args.period_ms is None else int(args.period_ms)
    if period_ms <= 0:
        raise ValueError("--period-ms must be positive")
    if args.actuation_delay_ms < 0.0:
        raise ValueError("--actuation-delay-ms must be non-negative")
    period_s = period_ms / 1000.0
    actuation_delay_s = float(args.actuation_delay_ms) / 1000.0
    response_timing = _response_timing_contract(period_ms, args.actuation_delay_ms, args.target_response_ms, args.response_tolerance_ms)
    scenarios = select_scenarios(benchmark, args.scenario_set)
    all_models = _model_table(benchmark)
    models_by_id = {int(item["model_id"]): item for item in all_models}
    models = [models_by_id[model_id] for model_id in selected_ids]
    run_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = (args.output_dir or HERE / "results" / f"closed_loop_{args.scenario_set}_{run_tag}").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trial_logs_dir = output_dir / "trial_logs"
    trial_summaries_dir = output_dir / "trial_summaries"
    trial_logs_dir.mkdir(exist_ok=True)
    trial_summaries_dir.mkdir(exist_ok=True)
    states = {item.initial_state_id: item for item in load_initial_states(resolve_project_path(benchmark["initial_states"]))}
    airsim = require_airsim()
    client = airsim.MultirotorClient()
    client.confirmConnection()
    if args.vehicle_name not in client.listVehicles():
        raise ValueError(f"AirSim vehicle {args.vehicle_name!r} is unavailable")
    paper_metric_manifest = {
        "version": PAPER_METRIC_VERSION,
        "paper_equation": "J = 1 / sum_{k=0}^{K} ||psi(k)-psi_ref(k)||^2_Wx",
        "K": PAPER_HORIZON_K,
        "window_samples": PAPER_WINDOW_SAMPLES,
        "state_order": ["px", "py", "pz", "vx", "vy", "vz", "qw", "qx", "qy", "qz"],
        "Wx_diagonal": [1.0] * 10,
        "quaternion_rule": "normalize actual/reference quaternions and flip reference sign when dot(actual, reference) < 0",
        "sample_alignment": "actual state and route reference evaluated at the same cycle timestamp",
        "route_aggregation": "non-overlapping complete 21-sample window costs are averaged, then J_paper_style is their reciprocal; trailing incomplete samples are discarded",
    }
    manifest = {
        "format_version": 2, "created_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "FPGA actuated", "protocol": "FLCN/v2", "endpoint": {"host": shadow["host"], "port": shadow["port"]},
        "period_ms": period_ms, "actuation_delay_ms": float(args.actuation_delay_ms),
        "models": [{"model_id": item["model_id"], "model_name": item["model_name"]} for item in models],
        "scenarios": scenarios, "repeats": args.repeats,
        "fallback_policy": "same-cycle linear MPC teacher whenever FPGA response is missing or late",
        "latency_definition": "R_cycle = host state-acquisition to host AirSim-command dispatch; AirSim exposes no exact physics-application timestamp",
        "episode_R_definition": "R_max_ms in the SVR dataset is the measured maximum state-to-command-dispatch latency over one episode",
        "actuation_delay_definition": "sleep after command selection and before AirSim command dispatch; it is added to the native response path and included in R_max_ms",
        "tracking_cost_definition": f"sum((||position_error||^2 + 0.25||velocity_error||^2)*{period_s:g})",
        "paper_metric": paper_metric_manifest,
        "reference_profile": reference_profile_manifest,
        "response_timing": response_timing,
    }
    manifest_path = output_dir / "closed_loop_manifest.json"
    if manifest_path.exists():
        if not args.resume:
            raise ValueError(f"output directory already contains a manifest; use --resume: {output_dir}")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        for field in ("period_ms", "actuation_delay_ms", "models", "scenarios", "repeats", "paper_metric"):
            if existing.get(field) != manifest.get(field):
                raise ValueError(f"resume configuration mismatch for {field}")
        if existing.get("reference_profile", reference_contract()) != reference_profile_manifest:
            raise ValueError("resume configuration mismatch for reference_profile; use a new output directory")
        old_timing = existing.get("response_timing", _response_timing_contract(existing["period_ms"], existing.get("actuation_delay_ms", 0.0)))
        if old_timing != response_timing:
            raise ValueError("resume configuration mismatch for response_timing; use a new output directory")
        manifest = existing
        experiment_id = str(existing["experiment_id"])
    else:
        if args.resume:
            raise ValueError(f"cannot resume without an existing manifest: {output_dir}")
        run_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        experiment_id = f"historical5ms_fpga_closed_loop_{period_ms}ms_delay{str(args.actuation_delay_ms).replace('.', 'p')}ms_{run_tag}"
        manifest["experiment_id"] = experiment_id
        _write_json(manifest_path, manifest)
    plan = [_trial_key(int(model["model_id"]), str(scenario["id"]), repeat) for model in models for scenario in scenarios for repeat in range(args.repeats)]
    summaries = _load_trial_summaries(trial_summaries_dir)
    calibration = load_calibration(args.calibration)
    with Falcon12FpgaUdpClient(host=str(shadow["host"]), port=int(shadow["port"]), timeout_ms=float(shadow["udp_timeout_ms"]), retries=int(shadow["retries"]), model_id=1) as udp_client:
        for model in models:
            for scenario in scenarios:
                for repeat in range(args.repeats):
                    key = _trial_key(int(model["model_id"]), str(scenario["id"]), repeat)
                    previous = summaries.get(key)
                    final_path = trial_logs_dir / f"{key}.jsonl"
                    if args.resume and previous is not None and bool(previous.get("completed")) and final_path.exists():
                        print(f"SKIP completed {key}", flush=True)
                        continue
                    partial_path = trial_logs_dir / f"{key}.jsonl.partial"
                    partial_path.unlink(missing_ok=True)
                    final_path.unlink(missing_ok=True)
                    print(f"\n=== FPGA closed loop {model['model_name']} / {scenario['id']} / repeat {repeat} ({key}) ===", flush=True)
                    with partial_path.open("w", encoding="utf-8") as handle:
                        trial = _run_trial(client=client, airsim=airsim, scenario=scenario, repeat_index=repeat, benchmark=benchmark, model=model, udp_client=udp_client, shadow=shadow, period_s=period_s, actuation_delay_s=actuation_delay_s, calibration=calibration, vehicle_name=args.vehicle_name, initial_states=states, output_handle=handle, experiment_id=experiment_id, keep_gates=args.keep_gates, reference_profile=args.reference_profile, startup_duration_s=args.startup_duration_s, target_response_ms=args.target_response_ms, response_tolerance_ms=args.response_tolerance_ms)
                    partial_path.replace(final_path)
                    trial["trial_key"] = key
                    _write_json(trial_summaries_dir / f"{key}.json", trial)
                    summaries[key] = trial
                    print(
                        f"RESULT completed={trial['completed']} crossings={len(trial['crossings'])}/5 "
                        f"fallback={trial['fallback_count']} delay_ms={trial['actuation_delay_ms']} "
                        f"measured_R_max_ms={trial.get('measured_R_max_ms')} error={trial['error']}",
                        flush=True,
                    )
    _rebuild_cycle_log(output_dir, plan)
    trials = [summaries[key] for key in plan if key in summaries]
    client_stats = dict(udp_client.stats)
    summary = {
        "format_version": 2, "phase": "FPGA actuated", "experiment_id": experiment_id,
        "paper_metric": paper_metric_manifest,
        "reference_profile": reference_profile_manifest,
        "response_timing": response_timing,
        "trials": trials,
        "episodes_requested": len(models) * len(scenarios) * args.repeats,
        "episodes_completed": sum(int(item["completed"]) for item in trials),
        "success_rate": float(np.mean([bool(item["completed"]) for item in trials])) if trials else None,
        "response_observed_us": _distribution([value for item in trials for value in item["response_us"]]),
        "state_to_command_dispatch_us": _distribution([value for item in trials for value in item["state_to_command_dispatch_us"]]),
        "fallback_count": sum(int(item["fallback_count"]) for item in trials),
        "deadline_miss_count": sum(int(item["deadline_miss_count"]) for item in trials),
        "timeout_count": sum(int(item["timeout_count"]) for item in trials),
        "response_target_violation_count": sum(int(item.get("response_target_violation_count", 0)) for item in trials),
        "client_stats": client_stats,
    }
    _write_json(output_dir / "fpga_closed_loop_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"Output directory: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
