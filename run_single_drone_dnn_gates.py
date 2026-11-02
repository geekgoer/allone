"""Fly one AirSim drone through five gates with the trained DNN policy.

The gate route and reference generator are the same as the MPC runner.  Only
the MPC ``solve`` call is replaced by the learned controller.  Legacy
reference-target 12-D, Track-P gate-target 12-D, and path-aware 25-D models
are supported through an explicit input-target selection.
"""

from __future__ import annotations

import argparse
import csv
import io
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from airsim_interface import AirSimDrone, require_airsim, vehicle_local_to_world_offset
from airsim_process import AirSimLaunchConfig, AirSimProcessManager
from config import DATA_DIR, ThrustCalibration, ensure_runtime_directories, load_calibration
from control_adapter import send_control
from dnn_controller import DNNController
from dnn_model_utils import build_policy_input
from environment import GateEnvironment
from fpga_shadow_backend import BufferedJsonlLogger, FalconFpgaControlBackend, load_fpga_config
from falcon12_fpga_shadow import Falcon12FpgaControlBackend, load_falcon12_fpga_config
from gate_route import ContinuousGateRoute, GatePassageError
from gate_visualization import (
    DEFAULT_GATE_ASSET,
    DEFAULT_GATE_HEIGHT,
    DEFAULT_GATE_THICKNESS,
    DEFAULT_GATE_WIDTH,
    cleanup_generated_gates,
    spawn_gate_sequence,
)
from initial_state_generator import InitialState, load_initial_states
from falcon_paper_fidelity.contracts import (
    PAPER_GATE_EULER_INPUT_CONTRACT,
    TRAJECTORY_INPUT_CONTRACT,
    build_trajectory_input,
    build_paper_gate_euler_input,
)
from state_adapter import build_controller_state


# Models produced before input contracts received stable symbolic names stored
# this descriptive string in model_spec.json.  Those datasets and the runtime
# state adapter both used the relative-quaternion rotation vector, so this is
# an alias of ``falcon_track_p_gate_relative_12d_v1`` rather than an unknown
# contract.  Keeping the alias here lets historical 12-D checkpoints be
# retested without weakening validation for genuinely incompatible models.
LEGACY_ROTATION_VECTOR_GATE_CONTRACTS = {
    "[p_gate-p, r_gate-r, velocity, acceleration] in AirSim NED",
}


def _join(future: object) -> None:
    join = getattr(future, "join", None)
    if callable(join):
        join()


def _write_report(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


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


def _check_state_and_collision(client: object, drone: AirSimDrone, raw: object, *, first_sample: bool = False) -> bool:
    values = (raw.position, raw.quaternion, raw.velocity, raw.acceleration, raw.angular_velocity)
    if not all(np.isfinite(value).all() for value in values):
        raise RuntimeError("AirSim returned NaN or Inf during DNN closed-loop flight")
    collision = client.simGetCollisionInfo(vehicle_name=drone.vehicle_name)
    if bool(getattr(collision, "has_collided", False)):
        object_name = getattr(collision, "object_name", "unknown")
        if first_sample:
            print(
                f"Ignoring AirSim reset/takeoff collision flag on first sample: "
                f"object={object_name} position={np.array2string(raw.position, precision=3)}",
                flush=True,
            )
            return True
        raise RuntimeError(f"DNN drone collision with {object_name}")
    return False


def run(
    *,
    model_path: Path,
    normalization_path: Path | None,
    epsilon: float,
    seed: int,
    threshold: float,
    per_gate_timeout: float,
    speed: float,
    base_y: float,
    base_z: float,
    vehicle_name: str,
    calibration: ThrustCalibration,
    gate_width: float,
    gate_height: float,
    gate_thickness: float,
    gate_asset: str,
    keep_gates: bool,
    gate_count: int,
    x_start: float,
    x_spacing: float,
    approach_distance: float,
    exit_distance: float,
    gate_clearance: float,
    device: str,
    report_path: Path | None,
    input_target: str = "reference",
    gate_attitude_encoding: str = "auto",
    process_manager: AirSimProcessManager | None = None,
    initial_state: InitialState | None = None,
    use_takeoff: bool = False,
    controller_backend: str = "cpu_dnn",
    fpga_config_path: Path = Path("fpga_closed_loop_config.json"),
    fpga_log_path: Path | None = None,
) -> dict[str, object]:
    if threshold <= 0.0 or per_gate_timeout <= 0.0 or speed <= 0.0:
        raise ValueError("threshold, per_gate_timeout and speed must be positive")
    if input_target not in ("reference", "gate"):
        raise ValueError("input_target must be 'reference' or 'gate'")
    if gate_attitude_encoding not in ("auto", "rotation_vector", "euler"):
        raise ValueError(
            "gate_attitude_encoding must be 'auto', 'rotation_vector', or 'euler'"
        )
    environment = GateEnvironment(
        epsilon=epsilon,
        seed=seed,
        base_y=base_y,
        base_z=base_z,
        gate_count=gate_count,
        x_start=x_start,
        x_spacing=x_spacing,
    )
    gates = environment.generate()
    ensure_runtime_directories()
    manager = process_manager
    owns_manager = False
    if manager is not None:
        client = manager.start_and_wait()
        owns_manager = True
    else:
        airsim = require_airsim()
        client = airsim.MultirotorClient()
        client.confirmConnection()
    client.reset()
    time.sleep(0.5)
    resolved_name = vehicle_name
    if not resolved_name:
        names = list(client.listVehicles())
        if not names:
            raise RuntimeError("AirSim did not report any vehicles")
        resolved_name = str(names[0])
        print(f"No --vehicle-name supplied; using {resolved_name}.")
    world_offset = vehicle_local_to_world_offset(client, resolved_name)
    cleanup_generated_gates(client, prefix="FalconGate_")
    cleanup_generated_gates(client, prefix="FalconDatasetGate_")
    scene = spawn_gate_sequence(
        client,
        gates,
        width=gate_width,
        height=gate_height,
        thickness=gate_thickness,
        asset_name=gate_asset,
        world_offset=world_offset,
    )
    print(
        f"Spawned {len(scene.object_names)} visible gate bars "
        f"({len(gates)} gates, width={gate_width:.1f}m, asset={gate_asset})."
    )
    print(f"Vehicle local-to-world offset: {np.array2string(world_offset, precision=3)}")
    drone = AirSimDrone(client, resolved_name)
    controller = DNNController(model_path, normalization_path, device=device)
    if input_target == "gate" and controller.input_dim != 12:
        raise ValueError("gate-target input is defined only for the Track-P 12-D model")
    resolved_gate_encoding = gate_attitude_encoding
    if input_target == "gate" and resolved_gate_encoding == "auto":
        resolved_gate_encoding = (
            "euler"
            if controller.input_contract == PAPER_GATE_EULER_INPUT_CONTRACT
            else "rotation_vector"
        )
    if input_target == "gate":
        expected_contract = {
            "euler": PAPER_GATE_EULER_INPUT_CONTRACT,
            "rotation_vector": "falcon_track_p_gate_relative_12d_v1",
        }[resolved_gate_encoding]
        compatible_contracts = {expected_contract}
        if resolved_gate_encoding == "rotation_vector":
            compatible_contracts.update(LEGACY_ROTATION_VECTOR_GATE_CONTRACTS)
        if (
            controller.input_contract is not None
            and controller.input_contract not in compatible_contracts
        ):
            raise ValueError(
                "requested gate attitude encoding disagrees with model_spec.json: "
                f"encoding={resolved_gate_encoding!r}, "
                f"model input_contract={controller.input_contract!r}"
            )
        if controller.input_contract in LEGACY_ROTATION_VECTOR_GATE_CONTRACTS:
            print(
                "Legacy 12-D gate contract recognized as rotation-vector "
                f"contract: {controller.input_contract}",
                flush=True,
            )
        print(
            f"Gate-target input contract: {expected_contract} "
            f"(attitude={resolved_gate_encoding}).",
            flush=True,
        )

    def state_for_policy(raw_state: object, target: object) -> np.ndarray:
        if input_target == "gate" and resolved_gate_encoding == "euler":
            return build_paper_gate_euler_input(
                raw_state, target.center, target.orientation
            )
        if (
            input_target == "reference"
            and controller.input_contract == TRAJECTORY_INPUT_CONTRACT
        ):
            return build_trajectory_input(raw_state, target)
        return build_controller_state(
            raw_state,
            target.center if input_target == "gate" else target.position,
            target.orientation,
        )
    trained_speeds = controller.reference_speed_values
    if (
        input_target == "gate"
        and len(trained_speeds) == 1
        and not np.isclose(speed, trained_speeds[0])
    ):
        print(
            "WARNING: --speed does not command a new cruise speed for the "
            "Track-P 12-D gate-target model. Its input has no reference-speed "
            "feature and its MPC labels were generated at "
            f"{trained_speeds[0]:.3f} m/s; the value {speed:.3f} only changes "
            f"hidden route timing. Use {trained_speeds[0]:.3f} for a "
            "matched test, or train a separate fixed-speed/path-aware model.",
            flush=True,
        )
    fpga_config: dict[str, object] = {}
    if controller_backend == "cpu_dnn":
        backend = FalconFpgaControlBackend(controller, controller_backend, fpga_config)
    else:
        config_preview = json.loads(fpga_config_path.read_text(encoding="utf-8"))
        if config_preview.get("protocol") == "FLCN/v2":
            if input_target != "gate" or controller.input_dim != 12:
                raise ValueError(
                    "FLCN/v2 requires --input-target gate and a 12-D Track-P CPU model"
                )
            fpga_config = load_falcon12_fpga_config(fpga_config_path)
            backend = Falcon12FpgaControlBackend(controller, controller_backend, fpga_config)
        else:
            fpga_config = load_fpga_config(fpga_config_path)
            backend = FalconFpgaControlBackend(controller, controller_backend, fpga_config)
    run_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path = DATA_DIR / f"dnn_{resolved_name or 'default'}_gates_{run_tag}.csv"
    report_path = report_path or (model_path.parent / f"closed_loop_{resolved_name}_{run_tag}.json")
    fpga_log_path = fpga_log_path or (model_path.parent / f"{controller_backend}_{resolved_name}_{run_tag}.jsonl")
    fpga_logger = BufferedJsonlLogger(fpga_log_path)
    columns = (
        ["timestamp", "monotonic_timestamp", "gate_id", "phase", "inference_time", "loop_period", "deadline_miss"]
        + [f"x{i}" for i in range(controller.input_dim)] + ["F", "wx", "wy", "wz"]
        + [f"raw_p_{axis}" for axis in "xyz"]
        + [f"reference_p_{axis}" for axis in "xyz"]
    )
    arrivals: list[dict[str, float]] = []
    samples = 0
    misses = 0
    initial_collision_flag = False
    started = time.perf_counter()
    try:
        if initial_state is not None and not use_takeoff:
            client.simPause(True)
            client.simSetVehiclePose(
                _pose(require_airsim(), initial_state),
                ignore_collision=True,
                vehicle_name=resolved_name,
            )
        drone.enable_api_control(True)
        drone.arm()
        if use_takeoff:
            _join(drone.takeoff())
        else:
            client.simPause(False)
            # A recorded initial state is already airborne, but simply
            # unpausing after simSetVehiclePose leaves SimpleFlight without
            # an active thrust/hover command.  The vehicle can fall for the
            # old 0.5 s settling interval and collide with the ground before
            # the first DNN action is sent.  Issue hover immediately so the
            # initial-state test starts from the requested airborne pose.
            _join(drone.hover())
        # AirSim can report the ground contact from reset/pose placement for
        # a short interval.  Keep this short: the vehicle must remain under
        # an active hover or takeoff command while the flag settles.
        time.sleep(0.1)
        route_start = time.perf_counter()
        initial = drone.get_raw_state()
        route = ContinuousGateRoute(
            gates,
            initial.position,
            approach_distance=approach_distance,
            exit_distance=exit_distance,
            reference_speed=speed,
            start_time=route_start,
            gate_width=gate_width,
            gate_height=gate_height,
            gate_thickness=gate_thickness,
            clearance=gate_clearance,
            completion_radius=threshold,
        )
        if controller_backend != "cpu_dnn":
            warmup_window = route.reference_window(route_start, dt=0.05, horizon=20)
            warmup_reference = warmup_window[0]
            warmup_target = route.current_gate if input_target == "gate" else warmup_reference
            warmup_state = state_for_policy(initial, warmup_target)
            warmup_input = build_policy_input(
                warmup_state,
                warmup_window,
                input_dim=backend.input_dim,
                reference_speed=speed,
            )
            print(
                f"Warming CPU DNN and FPGA with {int(fpga_config['warmup_count'])} requests "
                "before deadline accounting...",
                flush=True,
            )
            backend.warmup(warmup_input)
            route_start = time.perf_counter()
            route = ContinuousGateRoute(
                gates,
                initial.position,
                approach_distance=approach_distance,
                exit_distance=exit_distance,
                reference_speed=speed,
                start_time=route_start,
                gate_width=gate_width,
                gate_height=gate_height,
                gate_thickness=gate_thickness,
                clearance=gate_clearance,
                completion_radius=threshold,
            )
        next_deadline = route_start
        last_start = route_start
        # Buffer telemetry in memory so filesystem latency cannot disturb the
        # 20 Hz controller.  It is committed after the flight (also in finally
        # for the FPGA JSONL audit log).
        with io.StringIO(newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            iteration = 0
            while not route.completed:
                period_start = time.perf_counter()
                if period_start - route.gate_start_time > per_gate_timeout:
                    raise TimeoutError(
                        f"Gate {route.target_gate_id} was not crossed within {per_gate_timeout:.1f}s"
                    )
                raw = drone.get_raw_state()
                initial_collision_flag = _check_state_and_collision(
                    client, drone, raw, first_sample=(samples == 0)
                ) or initial_collision_flag
                try:
                    crossing = route.update(raw.position, period_start)
                except GatePassageError as exc:
                    raise GatePassageError(f"DNN route: {exc}") from exc
                if crossing is not None:
                    arrival = {
                        "gate": float(crossing.gate_index),
                        "lateral_offset": float(crossing.lateral_offset),
                        "vertical_offset": float(crossing.vertical_offset),
                    }
                    arrivals.append(arrival)
                    print(
                        f"Gate {crossing.gate_index} crossed inside opening: "
                        f"lateral={crossing.lateral_offset:.3f}m "
                        f"vertical={crossing.vertical_offset:.3f}m"
                    )
                if route.completed:
                    break
                window = route.reference_window(period_start, dt=0.05, horizon=20)
                reference = window[0]
                state_target = route.current_gate if input_target == "gate" else reference
                state = state_for_policy(raw, state_target)
                policy_input = build_policy_input(
                    state,
                    window,
                    input_dim=controller.input_dim,
                    reference_speed=speed,
                )
                inference_start = time.perf_counter()
                cycle_result = backend.infer(
                    policy_input, vehicle_name=resolved_name, cycle_index=iteration
                )
                control = cycle_result.applied_control
                inference_time = time.perf_counter() - inference_start
                send_control(drone, control, calibration, duration=0.05)
                next_deadline += 0.05
                loop_period = 0.05 if iteration == 0 else period_start - last_start
                miss = time.perf_counter() > next_deadline
                misses += int(miss)
                row = {
                    "timestamp": datetime.now(timezone.utc).timestamp(),
                    "monotonic_timestamp": period_start,
                    "gate_id": route.target_gate_id,
                    "phase": route.phase,
                    "inference_time": inference_time,
                    "loop_period": loop_period,
                    "deadline_miss": int(miss),
                    **{f"x{i}": float(value) for i, value in enumerate(policy_input)},
                    **{name: float(value) for name, value in zip(("F", "wx", "wy", "wz"), control)},
                    **{f"raw_p_{axis}": float(value) for axis, value in zip("xyz", raw.position)},
                    **{f"reference_p_{axis}": float(value) for axis, value in zip("xyz", reference.position)},
                }
                writer.writerow(row)
                fpga_logger.append({
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "vehicle_name": resolved_name,
                    "cycle_index": iteration,
                    "controller_backend": controller_backend,
                    "gate_id": route.target_gate_id,
                    "phase": route.phase,
                    "crossing": None if crossing is None else {
                        "gate": crossing.gate_index,
                        "lateral_offset": crossing.lateral_offset,
                        "vertical_offset": crossing.vertical_offset,
                    },
                    "x_physical": np.asarray(policy_input).tolist(),
                    "x_normalized": cycle_result.x_normalized.tolist(),
                    **cycle_result.to_log_dict(),
                    "inference_total_us": int(inference_time * 1e6),
                    "loop_period_us": int(loop_period * 1e6),
                    "deadline_miss": bool(miss),
                    "collision": False,
                    "timeout": bool(cycle_result.fpga is not None and not cycle_result.fpga.fresh),
                })
                if iteration % 20 == 0:
                    print(
                        f"gate={route.target_gate_id} k={iteration:04d} "
                        f"distance={route.distance_to_current_gate(raw.position):.2f}m "
                        f"pos_err={np.linalg.norm(state[:3]):.2f}m "
                        f"vel={np.linalg.norm(raw.velocity):.2f}m/s "
                        f"inference={inference_time * 1e3:.3f}ms "
                        f"loop={loop_period * 1e3:.2f}ms miss={miss}"
                    )
                samples += 1
                iteration += 1
                last_start = period_start
                sleep_time = next_deadline - time.perf_counter()
                if sleep_time > 0.0:
                    time.sleep(sleep_time)
            csv_text = handle.getvalue()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.write_text(csv_text, encoding="utf-8")
        _join(drone.hover())
        print(f"All {gate_count} gates crossed inside their clear openings with DNN; hovering.")
        report = {
            "controller": controller_backend,
            "model": str(model_path.resolve()),
            "normalization": str(controller.normalization_path),
            "input_target": input_target,
            "gate_attitude_encoding": resolved_gate_encoding,
            "input_contract": controller.input_contract,
            "vehicle_name": resolved_name,
            "epsilon": epsilon,
            "seed": seed,
            "gate_count": gate_count,
            "samples": samples,
            "deadline_misses": misses,
            "initial_collision_flag": initial_collision_flag,
            "completed": route.completed,
            "crossings": arrivals,
            "csv": str(csv_path.resolve()),
            "fpga_jsonl": str(fpga_log_path.resolve()),
            "backend": backend.summary(),
            "elapsed_seconds": time.perf_counter() - started,
        }
        _write_report(report_path, report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return report
    finally:
        try:
            _join(drone.hover())
            _join(drone.land())
            drone.disarm()
            drone.enable_api_control(False)
        finally:
            fpga_logger.flush()
            backend.close()
            if keep_gates:
                print("Keeping visible gates in the UE4 world (--keep-gates).")
            else:
                scene.destroy()
            if owns_manager and manager is not None:
                manager.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("dnn_runs/depth10_width512/controller_scripted.pt"))
    parser.add_argument("--normalization", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--input-target",
        choices=("reference", "gate"),
        default="reference",
        help="use gate for Track-P [p_gate-p,...] 12-D models; reference preserves legacy/25-D behavior",
    )
    parser.add_argument(
        "--gate-attitude-encoding",
        choices=("auto", "rotation_vector", "euler"),
        default="auto",
        help=(
            "12-D gate attitude representation. 'auto' reads model_spec.json; "
            "legacy models without a contract default to rotation_vector."
        ),
    )
    parser.add_argument(
        "--controller-backend",
        choices=("cpu_dnn", "shadow_fpga", "fpga"),
        default="cpu_dnn",
        help="shadow_fpga logs FPGA but applies CPU DNN; fpga explicitly enables FPGA control",
    )
    parser.add_argument("--fpga-config", type=Path, default=Path("fpga_closed_loop_config.json"))
    parser.add_argument("--fpga-log", type=Path, default=None)
    parser.add_argument("--epsilon", type=float, choices=(0.0, 0.8, 1.6), default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--per-gate-timeout", type=float, default=30.0)
    parser.add_argument("--speed", type=float, default=2.0)
    parser.add_argument("--base-y", type=float, default=0.0)
    parser.add_argument("--base-z", type=float, default=-2.0)
    parser.add_argument("--gate-width", type=float, default=DEFAULT_GATE_WIDTH)
    parser.add_argument("--gate-height", type=float, default=DEFAULT_GATE_HEIGHT)
    parser.add_argument("--gate-thickness", type=float, default=DEFAULT_GATE_THICKNESS)
    parser.add_argument("--gate-asset", default=DEFAULT_GATE_ASSET)
    parser.add_argument("--gate-count", type=int, default=5)
    parser.add_argument("--x-start", type=float, default=10.0)
    parser.add_argument("--x-spacing", type=float, default=10.0)
    parser.add_argument("--approach-distance", type=float, default=2.0)
    parser.add_argument("--exit-distance", type=float, default=1.0)
    parser.add_argument("--gate-clearance", type=float, default=0.25)
    parser.add_argument("--vehicle-name", default="")
    parser.add_argument("--calibration", default=None)
    parser.add_argument("--keep-gates", action="store_true")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--layout-root", type=Path, default=Path("teacher_dataset_full_route_continuous_v2"))
    parser.add_argument("--initial-state-id", type=int, default=0, help="use this recorded initial state; use -1 to use AirSim takeoff")
    parser.add_argument("--takeoff", action="store_true", help="start with AirSim takeoff instead of a recorded airborne initial state")
    parser.add_argument("--auto-launch", action="store_true", help="launch a managed UE4 game process")
    parser.add_argument("--render", action="store_true", help="keep UE4 graphics enabled when using --auto-launch")
    parser.add_argument("--ue4-editor", type=Path, default=Path(r"E:\epicGamesStrg\UE_4.27\Engine\Binaries\Win64\UE4Editor.exe"))
    parser.add_argument("--uproject", type=Path, default=Path(r"F:\Airsim\AirSim\Unreal\Environments\Blocks\Blocks.uproject"))
    parser.add_argument("--startup-timeout", type=float, default=120.0)
    args = parser.parse_args()
    initial_state = None
    if not args.takeoff and args.initial_state_id >= 0:
        initial_states = load_initial_states(args.layout_root / "initial_states.json")
        matching = [state for state in initial_states if state.initial_state_id == args.initial_state_id]
        if not matching:
            raise ValueError(f"initial state ID {args.initial_state_id} was not found under {args.layout_root}")
        initial_state = matching[0]
    manager = None
    if args.auto_launch:
        manager = AirSimProcessManager(
            AirSimLaunchConfig.from_paths(
                args.ue4_editor,
                args.uproject,
                log_dir=Path("logs"),
                startup_timeout=args.startup_timeout,
                null_rhi=not args.render,
            )
        )
    run(
        model_path=args.model,
        normalization_path=args.normalization,
        epsilon=args.epsilon,
        seed=args.seed,
        threshold=args.threshold,
        per_gate_timeout=args.per_gate_timeout,
        speed=args.speed,
        base_y=args.base_y,
        base_z=args.base_z,
        vehicle_name=args.vehicle_name,
        calibration=load_calibration(args.calibration),
        gate_width=args.gate_width,
        gate_height=args.gate_height,
        gate_thickness=args.gate_thickness,
        gate_asset=args.gate_asset,
        keep_gates=args.keep_gates,
        gate_count=args.gate_count,
        x_start=args.x_start,
        x_spacing=args.x_spacing,
        approach_distance=args.approach_distance,
        exit_distance=args.exit_distance,
        gate_clearance=args.gate_clearance,
        device=args.device,
        input_target=args.input_target,
        gate_attitude_encoding=args.gate_attitude_encoding,
        report_path=args.report,
        process_manager=manager,
        initial_state=initial_state,
        use_takeoff=args.takeoff,
        controller_backend=args.controller_backend,
        fpga_config_path=args.fpga_config,
        fpga_log_path=args.fpga_log,
    )


if __name__ == "__main__":
    main()
