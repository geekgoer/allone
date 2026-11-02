"""Manual-Play three-drone 237D resource probe: MPC shadow or explicit DNN control.

One CPU producer/RPC client per drone and one shared FCFS inference worker.
The simulator runs continuously during measured flight. Only initial placement
uses simSetKinematics; global trajectory time never restarts at a gate.
"""

import argparse
from concurrent.futures import TimeoutError as FutureTimeout
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
import threading
import time

import numpy as np

from airsim_interface import AirSimDrone, require_airsim, vehicle_local_to_world_offset
from config import load_calibration
from control_adapter import send_control
from environment import Gate
from gate_visualization import spawn_gate_sequences
from .convex_features import encode
from .convex_teacher import ConvexTeacher
from .decision_dataset import atomic_json, sha256
from .dynamic_probe_common import (BASE, RUNS, DEFAULT_MODELS, InferenceService, ModelPool, describe, new_output,
                                   provenance, read_json, release_schedule, save_timing)
from .fly_convex_policy import load_airsim_client
from .fly_decision_dnn import StoredRoutePlanner
from .fast_convex_preprocess import FastReferenceWindow, encode as fast_encode
from experiments.falcon_paper_fidelity_acado_v3_continuous.demo_acado_five_gates import _crossing_results


VEHICLES = ("Drone1", "Drone2", "Drone3")
GATE = {"width": 4., "height": 4., "thickness": .15, "clearance": .25}


def translate_layout(layout, shift):
    value = deepcopy(layout)
    for segment in value["segments"]:
        c = np.asarray(segment["coefficients"], dtype=float)
        c[:, 0] += shift
        segment["coefficients"] = c.tolist()
    for gate in value["gates"]:
        gate["center"] = (np.asarray(gate["center"]) + shift).tolist()
    return value


def load_layout(environment, route):
    manifest = read_json(BASE / "manifest.json")
    key = f"env{environment:02d}_route{route:02d}"
    path = BASE / "routes" / f"{key}.json"
    if not manifest["complete"] or sha256(path) != manifest["chunks"][key]["route_sha256"]:
        raise ValueError("route provenance mismatch")
    return read_json(path), sha256(path)


def await_control(future, row, deadline):
    if future is None:
        raise RuntimeError("DNN request busy; no stale output will be applied")
    try:
        control = future.result(timeout=max(0., deadline-time.perf_counter()))
    except FutureTimeout as exc:
        row["control_outcome"] = "deadline_abort"
        raise RuntimeError("DNN deadline expired; flight aborted without applying a stale output") from exc
    if time.perf_counter() > deadline:
        row["control_outcome"] = "late_output_not_applied"
        raise RuntimeError("late DNN output not applied")
    return control


def check_online_dispatch(row, stop, deadline):
    now, stopped = time.perf_counter(), stop.is_set()
    expired = now >= deadline
    row.update(dispatch_guard_s=now, dispatch_guard_stop_requested=stopped,
               dispatch_guard_deadline_expired=expired)
    if expired:
        row["control_outcome"] = "late_output_not_applied"
        raise RuntimeError("committed deadline expired before dispatch")
    if stopped:
        row["control_outcome"] = "peer_abort"
        raise RuntimeError("another flight worker requested stop; output not applied")


def run(args):
    switch_plan = getattr(args, "switch_plan", None)
    joint_shadow = getattr(args, "joint_selection_shadow", False)
    joint_online = getattr(args, "joint_online", False)
    if joint_online and (switch_plan is not None or joint_shadow or args.mode != "dnn"):
        raise ValueError("online joint control cannot be mixed with scripted/MPC/shadow control")
    dynamic_control = switch_plan is not None or joint_online
    if joint_shadow and switch_plan is None:
        raise ValueError("joint shadow replay requires the validated scripted switching harness")
    budget_config_path = getattr(args, "state_budget_config", None)
    budget_config = None
    if budget_config_path is not None:
        from .state_budget_shadow import load_config
        if args.mode != "dnn":
            raise ValueError("budget replay requires DNN actuation, not MPC-controlled shadow mode")
        budget_config = load_config(budget_config_path)
    if joint_shadow and budget_config is None:
        raise ValueError("joint shadow replay requires an explicit state-budget configuration")
    if joint_online and budget_config is None:
        raise ValueError("online joint control requires an explicit state-budget configuration")
    if switch_plan is not None and (args.mode != "dnn" or args.model == "cycle"):
        raise ValueError("scripted switching requires explicit DNN control")
    names = list(DEFAULT_MODELS)
    if args.model != "cycle" and args.model not in names:
        names.append(args.model)
    preprocess_policy = getattr(args, "preprocess_policy", "parallel")
    # Small NumPy calls in three Python producer threads can contend heavily.
    # This changes execution order, not the planner, horizon, features or labels.
    preprocess_lock = threading.Lock()
    evidence = provenance(names)
    pool = ModelPool(names, args.device)
    if joint_shadow or joint_online:
        from .joint_controller_sweep import RUNTIME_KEYS
        actual = pool.runtime_info()
        if any(actual.get(k) != args.joint_profile["runtime"].get(k) for k in RUNTIME_KEYS):
            raise ValueError("joint selection runtime differs from candidate profile")
    if switch_plan is not None:
        from .joint_controller_sweep import RUNTIME_KEYS
        actual = pool.runtime_info()
        if any(actual.get(k) != args.switch_runtime.get(k) for k in RUNTIME_KEYS):
            raise ValueError("runtime/device/precision differs from the candidate flight pilot")
    layout, layout_hash = load_layout(args.environment_id, args.route_index)
    original = StoredRoutePlanner(layout)
    duration = max(20., original.trajectory.duration+2.)
    samples = original.reference_window_for_trajectory(original.trajectory, elapsed=0.,
        dt=duration/200, horizon=201)
    y = np.array([r.position[1] for r in samples] + [g["center"][1] for g in layout["gates"]])
    lane_spacing = max(12., float(y.max()-y.min())+10.)
    path = getattr(args, "output_dir", None)
    if path is None:
        path = new_output("airsim_joint_online" if joint_online else
                          "airsim_joint_switch" if switch_plan is not None else f"airsim_{args.mode}")
    else:
        path.mkdir(parents=True, exist_ok=False)
    metadata = {"provenance": evidence, "mode": args.mode, "device": args.device,
        "runtime": pool.runtime_info(),
        "model_selection": args.model, "phase": args.phase, "period_ms": args.period_ms,
        "environment_id": args.environment_id, "route_index": args.route_index,
        "saved_route_sha256": layout_hash, "world_lane_spacing_m": lane_spacing,
        "nominal_speed_mps": 5., "trajectory_clock": "elapsed_host_monotonic_time",
        "duration_seconds": duration, "resident_parameter_bytes": pool.parameter_bytes,
        "resident_models": names, "preprocess_policy": preprocess_policy,
        "preprocess_source_sha256": sha256(Path(__file__).with_name("fast_convex_preprocess.py"))
            if preprocess_policy == "vectorized" else None,
        "server": {"host": args.host, "port": args.port},
        "controller": "CPU convex MPC; DNN output is NOT applied" if args.mode == "shadow" else "shared GPU/CPU DNN; no MPC fallback",
        "simulation_requirement": "ClockSpeed=1, manual Play, no concurrent training; verify sim/wall ratio below"}
    if switch_plan is not None:
        metadata.update(model_selection="scripted_plan", period_ms=None,
            experiment="scripted_joint_switch_mechanism", online_resource_feedback=False,
            state_dependent_T_max=None, switch_plan=switch_plan.steps,
            switch_plan_sha256=args.switch_plan_sha256,
            switch_profile_sha256=args.switch_profile_sha256,
            deadline_policy="each request D=T; later switches never modify it")
    if joint_shadow:
        metadata.update(shadow_resource_feedback=False,
            joint_selection_shadow={"enabled": True, "control_changed": False,
                "execution_mode": "post_flight_causal_replay",
                "flight_critical_path": "position/velocity/timestamp capture only; no policy evaluation",
                "queue_feedback_source": "causally reconstructed from recorded enqueue/ready/dispatch events",
                "profile_sha256": args.joint_profile["summary_sha256"],
                "selector_source_sha256": sha256(Path(__file__).with_name("state_joint_policy.py")),
                "observer_source_sha256": sha256(Path(__file__).with_name("joint_selection_shadow.py"))})
    if joint_online:
        metadata.update(model_selection="online_next_request", period_ms=None,
            experiment="online_state_resource_joint_control", online_resource_feedback=True,
            policy=args.policy, profile_sha256=args.joint_profile["summary_sha256"],
            quality_scope=args.joint_profile["quality_scope"],
            online_controller_sha256=sha256(Path(__file__).with_name("online_joint_flight.py")),
            selector_source_sha256=sha256(Path(__file__).with_name("state_joint_policy.py")),
            deadline_policy="committed request model/D=T/next release immutable; select next request only",
            online_selection={"enabled": True, "recommendations_applied": True,
                "failure_policy": "abort all flights and submit hover cleanup; no MPC/stale-output fallback",
                "is_safety_certificate": False})
    atomic_json(path / "setup.json", metadata)
    rows, errors, traces = [], [], {v: [] for v in VEHICLES}
    stop = threading.Event()
    service = scene = client = online = None
    threads = []
    start_time = [None]
    planners, gates, fast_windows = {}, {}, {}
    period = args.period_ms/1000
    simulation_touched = False
    try:
        def warmup():
            for name in names:
                for _ in range(20):
                    pool.predict(name, pool.norm["x_mean"])
        service = InferenceService(pool.predict, warmup, feedback=joint_online)
        if joint_online:
            from .online_joint_flight import OnlineJointController
            online = OnlineJointController(args.joint_profile, budget_config, service, VEHICLES, mode=args.policy)
        load_airsim_client()
        airsim = require_airsim()
        def connect():
            c = airsim.MultirotorClient(ip=args.host, port=args.port, timeout_value=3.)
            c.confirmConnection()
            return c
        client = connect()
        if not set(VEHICLES) <= set(client.listVehicles()):
            raise RuntimeError("manual Play session must contain Drone1, Drone2 and Drone3")
        foreign = [n for n in client.simListSceneObjects("Falcon.*Gate.*") if not n.startswith("FalconProbeGate_")]
        if foreign:
            raise RuntimeError("Other generated gates remain. Stop/Play once to clear runtime gates before this probe; no foreign gates deleted.")
        print("This probe resets simulation ONCE and places three drones at aerial starts; no takeoff animation.", flush=True)
        simulation_touched = True
        client.reset()
        client.simPause(True)
        offsets = {v: vehicle_local_to_world_offset(client, v) for v in VEHICLES}
        for i, vehicle in enumerate(VEHICLES):
            shift = np.array([0., (i-1)*lane_spacing, 0.]) - offsets[vehicle]
            translated = translate_layout(layout, shift)
            planner = planners[vehicle] = StoredRoutePlanner(translated)
            if preprocess_policy == "vectorized":
                fast_windows[vehicle] = FastReferenceWindow(planner.trajectory, planner.config)
            gates[vehicle] = [Gate(g["index"], g["center"], g["orientation"]) for g in translated["gates"]]
            first = planner.reference_window_for_trajectory(planner.trajectory, elapsed=0., dt=.1, horizon=21)[0]
            drone = AirSimDrone(client, vehicle)
            drone.enable_api_control(True)
            drone.arm()
            kinematics = airsim.KinematicsState()
            kinematics.position = airsim.Vector3r(*[float(v) for v in first.position])
            q = first.orientation
            kinematics.orientation = airsim.Quaternionr(float(q[1]), float(q[2]), float(q[3]), float(q[0]))
            kinematics.linear_velocity = airsim.Vector3r(0., 0., 0.)
            kinematics.angular_velocity = airsim.Vector3r(0., 0., 0.)
            kinematics.linear_acceleration = airsim.Vector3r(0., 0., 0.)
            kinematics.angular_acceleration = airsim.Vector3r(0., 0., 0.)
            client.simSetKinematics(kinematics, ignore_collision=True, vehicle_name=vehicle)
            drone.hover()
        if budget_config is not None:
            metadata.update(gate_geometry=dict(GATE), vehicle_local_gates={v: [
                {"index": g.index, "center": g.center.tolist(), "orientation": g.orientation.tolist()}
                for g in gates[v]] for v in VEHICLES})
        scene = spawn_gate_sequences(client, gates, width=4., height=4., thickness=.15,
            asset_name="1M_Cube", object_prefix="FalconProbeGate_", vehicle_world_offsets=offsets)
        calibration = load_calibration()
        startup = threading.Barrier(4, timeout=30.)
        launch = threading.Event()
        def fly(index):
            vehicle = VEHICLES[index]
            drone = None
            try:
                c = connect()  # RPC/Tornado clients MUST NOT cross thread ownership.
                drone = AirSimDrone(c, vehicle)
                planner = planners[vehicle]
                teacher = ConvexTeacher() if args.mode == "shadow" else None
                initial_collision = c.simGetCollisionInfo(vehicle_name=vehicle)
                ignored_stamp = int(getattr(initial_collision, "time_stamp", -1))
                initial_raw = drone.get_raw_state()
                references = planner.reference_window_for_trajectory(planner.trajectory, elapsed=0., dt=.1, horizon=21)
                if teacher:
                    teacher.solve(initial_raw, references)  # CPU solver setup is outside flight timing.
                startup.wait()
                launch.wait()
                if stop.is_set():
                    return
                phase = index*period/3 if args.phase == "staggered" else 0.
                previous_choice = previous_control = None
                if joint_online:
                    releases = online.releases(start_time[0], duration, phase, stop, service, vehicle)
                elif switch_plan is not None:
                    releases = switch_plan.releases(start_time[0], duration, phase, stop, service, vehicle)
                else:
                    releases = ((k, scheduled, {"model": args.model if args.model != "cycle" else names[(k+index) % len(names)],
                        "period_ms": args.period_ms, "deadline_ms": args.deadline_ms})
                        for k, scheduled in release_schedule(start_time[0], duration, period, stop, service, vehicle, phase))
                for k, scheduled, choice in releases:
                    request_period = choice["period_ms"]/1000
                    request_deadline = choice["deadline_ms"]/1000
                    sample_start = time.perf_counter()
                    raw = drone.get_raw_state()
                    state_ready = time.perf_counter()
                    if not all(np.isfinite(v).all() for v in (raw.position, raw.velocity, raw.quaternion, raw.acceleration)):
                        raise RuntimeError("nonfinite AirSim state")
                    collision_start = time.perf_counter()
                    collision = c.simGetCollisionInfo(vehicle_name=vehicle)
                    collision_ready = time.perf_counter()
                    if collision.has_collided and int(getattr(collision, "time_stamp", -2)) != ignored_stamp:
                        raise RuntimeError(f"collision with {collision.object_name}")
                    preprocess_requested = time.perf_counter()
                    with preprocess_lock if preprocess_policy == "serialized" else nullcontext():
                        preprocess_start = time.perf_counter()
                        elapsed = sample_start-start_time[0]
                        if preprocess_policy == "vectorized":
                            references = fast_windows[vehicle].window(elapsed)
                        else:
                            references = planner.reference_window_for_trajectory(planner.trajectory,
                                elapsed=elapsed, dt=.1, horizon=21)
                        reference_ready = time.perf_counter()
                        feature = fast_encode(raw, references) if preprocess_policy == "vectorized" else encode(raw, references)
                        feature_ready = time.perf_counter()
                    error = float(np.linalg.norm(raw.position-references[0].position))
                    if raw.position[2]+offsets[vehicle][2] > -.3 or error > 8. or np.linalg.norm(raw.velocity) > 15.:
                        raise RuntimeError(f"flight guard: tracking_error={error:.3f}m")
                    name = choice["model"]
                    row = {"vehicle": vehicle, "cycle": k, "scheduled_s": scheduled,
                           "sample_start_s": sample_start, "state_ready_s": state_ready,
                           "collision_start_s": collision_start, "collision_ready_s": collision_ready,
                           "preprocess_requested_s": preprocess_requested, "preprocess_start_s": preprocess_start,
                           "reference_ready_s": reference_ready, "feature_ready_s": feature_ready,
                           "period_ms": choice["period_ms"], "deadline_ms": choice["deadline_ms"],
                           "sim_timestamp_ns": int(raw.timestamp_ns or 0)}
                    if joint_shadow:
                        # Keep failed requests too. Queue/dispatch history and all
                        # policy comparisons are reconstructed AFTER flight.
                        row["joint_shadow_capture_start_s"] = time.perf_counter()
                        row["joint_shadow_state"] = {"position": raw.position.tolist(),
                                                     "velocity": raw.velocity.tolist()}
                        row["joint_shadow_snapshot_s"] = time.perf_counter()
                    if joint_online:
                        row["model"] = name  # Preserve rejected samples before service.submit.
                        online.select_next(row, raw, gates[vehicle], GATE)
                    future = service.submit(name, feature, row)
                    if teacher:
                        row["mpc_start_s"] = time.perf_counter()
                        control = teacher.solve(raw, references)
                        row["mpc_ready_s"] = time.perf_counter()
                        row["mpc_output"] = control.tolist()
                    else:
                        control = await_control(future, row, scheduled+request_deadline)
                    applied = np.clip(control, [2., -3., -3., -2.], [20., 3., 3., 2.])
                    if joint_online:
                        check_online_dispatch(row, stop, scheduled+request_deadline)
                    row["command_start_s"] = time.perf_counter()
                    send_control(drone, applied, calibration, duration=request_period)
                    row["command_ready_s"] = time.perf_counter()
                    row["mpc_dispatch_s" if teacher else "dnn_dispatch_s"] = row["command_ready_s"]
                    row["control_outcome"] = "mpc_shadow" if teacher else "dnn_dispatched"
                    row["clipped"] = bool(np.any(applied != control))
                    if dynamic_control:
                        row["model_changed"] = previous_choice is not None and name != previous_choice["model"]
                        row["period_changed"] = previous_choice is not None and choice["period_ms"] != previous_choice["period_ms"]
                        row["command_duration_ms"] = choice["period_ms"]
                        row["applied_control"] = applied.tolist()
                        row["control_delta_from_previous"] = None if previous_control is None else (applied-previous_control).tolist()
                        previous_choice, previous_control = choice, applied.copy()
                    if joint_online:
                        online.commit_dispatch(row)
                    traces[vehicle].append({"position": raw.position.tolist(), "velocity": raw.velocity.tolist(),
                        "reference_position": references[0].position.tolist(), "sample_s": sample_start,
                        "sim_timestamp_ns": int(raw.timestamp_ns or 0), "tracking_error_m": error,
                        "model": name, "period_ms": choice["period_ms"], "reference_elapsed_s": elapsed})
                    if k % 40 == 0:
                        print(f"{vehicle} k={k:04d} mode={args.mode} error={error:.3f}m speed={np.linalg.norm(raw.velocity):.2f}m/s", flush=True)
                        if joint_online:
                            print(f"  current={name}/T{choice['period_ms']:g} "
                                  f"next={row['next_model']}/T{row['next_period_ms']:g} "
                                  f"selection={1000*(row['joint_online_ready_s']-row['joint_online_start_s']):.3f}ms", flush=True)
            except BaseException as exc:
                errors.append(f"{vehicle}: {type(exc).__name__}: {exc}")
                stop.set()
                startup.abort()
            finally:
                if drone is not None:
                    try:
                        drone.hover()
                    except Exception as exc:
                        errors.append(f"{vehicle} hover cleanup: {exc}")
        threads = [threading.Thread(target=fly, args=(i,), name=v) for i, v in enumerate(VEHICLES)]
        for thread in threads:
            thread.start()
        startup.wait()
        client.simPause(False)
        # Do not leave placed drones without their measured controller for 200 ms.
        start_time[0] = time.perf_counter()
        launch.set()
        while any(t.is_alive() for t in threads):
            time.sleep(.1)
    except BaseException as exc:
        errors.append(f"coordinator: {type(exc).__name__}: {exc}")
        stop.set()
    finally:
        if "launch" in locals():
            launch.set()
        if "startup" in locals():
            startup.abort()
        for thread in threads:
            if thread.ident is not None:
                thread.join()
        if service is not None:
            service.close()
            rows = service.rows
        if client is not None and simulation_touched:
            try:
                client.simPause(False)
                for v in VEHICLES:
                    client.hoverAsync(vehicle_name=v)
                client.ping()  # Pump pending async RPC submissions before returning.
                if scene is not None and not args.keep_gates:
                    scene.destroy()
            except Exception as exc:
                errors.append(f"cleanup: {exc}")
    flight = {}
    for vehicle, trace in traces.items():
        if not trace:
            flight[vehicle] = {"samples": 0, "crossings": 0}
            continue
        p = np.asarray([r["position"] for r in trace])
        v = np.asarray([r["velocity"] for r in trace])
        from types import SimpleNamespace
        crossings = _crossing_results(p, v, SimpleNamespace(gates=gates[vehicle]), GATE)
        wall = trace[-1]["sample_s"]-trace[0]["sample_s"]
        flight[vehicle] = {"samples": len(trace), "crossings": sum(c[3] for c in crossings),
            "tracking_error_m": describe([r["tracking_error_m"] for r in trace]),
            "sim_seconds_per_host_second": (trace[-1]["sim_timestamp_ns"]-trace[0]["sim_timestamp_ns"])/1e9/wall if wall > 0 else None}
    atomic_json(path / "flight_trace.json", traces)
    planned_releases = int(np.ceil(duration/period))*3
    if dynamic_control:
        from .joint_controller_sweep import time_mean_squared_error
        from .replay_online_joint import output_age, summarize
        if switch_plan is not None:
            planned_releases = sum(sum(1 for _ in switch_plan.slots(0., duration,
                i*period/3 if args.phase == "staggered" else 0.)) for i in range(3))
        else:
            from .online_joint_flight import audit_schedule
            initial = {"model": args.model, "period_ms": args.period_ms, "deadline_ms": args.deadline_ms}
            audit = audit_schedule(rows, VEHICLES, start_time[0], duration, args.phase, initial)
            planned_releases = audit["realized_releases"]
            metadata.update(schedule_accounting_passed=audit["passed"], dynamic_schedule_audit=audit,
                release_count_scope="realized online schedule; full duration and every transition independently audited")
        costs = {v: time_mean_squared_error(trace, start_time[0], original.trajectory.duration)
                 if trace and start_time[0] is not None else None for v, trace in traces.items()}
        mse = float(np.mean(list(costs.values()))) if all(c is not None for c in costs.values()) else None
        switched = [r for r in rows if r.get("control_delta_from_previous") is not None
                    and (r.get("model_changed") or r.get("period_changed"))]
        deltas = np.abs([r["control_delta_from_previous"] for r in switched])
        dispatched = [{**r, "ready_s": r["dnn_dispatch_s"]} for r in rows if "dnn_dispatch_s" in r]
        metadata.update(tracking_mse_m2=mse, tracking_rmse_m=None if mse is None else float(np.sqrt(mse)),
            tracking_window_seconds=original.trajectory.duration,
            clock_ratio_passed=all(f.get("sim_seconds_per_host_second") is not None and
                .95 <= f["sim_seconds_per_host_second"] <= 1.05 for f in flight.values()),
            tracking_window_covered=mse is not None,
            command_input_age=output_age(dispatched, start_time[0], [3], duration, duration,
                                        vehicle_names=VEHICLES)
                if start_time[0] is not None else None,
            common_response_limit_misses=summarize(rows)["common_response_limit_misses"],
            switch_control_delta_max_abs_per_output=deltas.max(axis=0).tolist() if switched else None,
            model_switches=sum(bool(r.get("model_changed")) for r in rows),
            period_switches=sum(bool(r.get("period_changed")) for r in rows))
    metadata.update(status="INCOMPLETE" if errors else "COMPLETE", errors=errors, flight=flight,
        planned_releases=planned_releases,
        all_15_crossings_passed=not errors and all(f.get("crossings") == 5 for f in flight.values()),
        scope="three-drone live shadow; DNN ready latency, not DNN actuation" if args.mode == "shadow" else "three-drone DNN closed loop; async command dispatch timing")
    if budget_config is not None and "vehicle_local_gates" in metadata:
        from .state_budget_shadow import save_analysis
        budget = save_analysis(path, traces, rows, metadata["vehicle_local_gates"], GATE, budget_config)
        budget.update(config_sha256=sha256(budget_config_path),
                      source_sha256=sha256(Path(__file__).with_name("state_budget_shadow.py")))
        atomic_json(path / "state_budget_report.json", budget)
        metadata["state_dependent_T_max"] = {"mode": "online_selection_with_posthoc_audit" if joint_online else "causal_shadow_replay",
            "control_changed": joint_online,
            "is_safety_certificate": False, "report": "state_budget_report.json",
            "applicable_samples": budget["applicable_samples"],
            "posthoc_over_budget_samples": budget["posthoc_over_budget_samples"]}
    if joint_shadow:
        from .joint_selection_shadow import replay
        joint_monitor = replay(args.joint_profile, budget_config, metadata, rows, traces,
                               metadata.get("vehicle_local_gates", {}), GATE)
        summary = joint_monitor.save(path, execution_mode="post_flight_causal_replay", source_status=metadata["status"])
        capture_ms = describe([1000*(r["joint_shadow_snapshot_s"]-r["joint_shadow_capture_start_s"])
                               for r in rows if "joint_shadow_snapshot_s" in r])
        summary["capture_ms"] = capture_ms
        atomic_json(path / "joint_shadow_report.json", summary)
        metadata["joint_selection_shadow"].update(report="joint_shadow_report.json",
            samples=summary["samples"], capture_ms=capture_ms, decision_ms=summary["decision_ms"])
    if joint_online:
        from collections import Counter
        decisions = [r for r in rows if "joint_online_decision" in r]
        summary = {"execution_mode": "online_next_request", "policy": args.policy,
            "decisions": len(decisions), "recommendations_applied": True,
            "status_counts": dict(Counter(r["joint_online_decision"]["status"] for r in decisions)),
            "next_choices_committed": sum(bool(r.get("next_choice_committed")) for r in rows),
            "selection_ms": describe([1000*(r["joint_online_ready_s"]-r["joint_online_start_s"])
                                      for r in rows if "joint_online_ready_s" in r]),
            "actual_model_counts": dict(Counter(r["model"] for r in rows if "dnn_dispatch_s" in r)),
            "actual_period_counts_ms": dict(Counter(str(r["period_ms"]) for r in rows if "dnn_dispatch_s" in r)),
            "is_safety_certificate": False, "quality_scope": args.joint_profile["quality_scope"]}
        atomic_json(path / "online_selection_report.json", summary)
        metadata["online_selection"].update(report="online_selection_report.json", **summary)
    report = save_timing(path, rows, None if dynamic_control else args.deadline_ms, metadata)
    print(f"flight_passed={report['flight_passed']} realtime_passed={report['realtime_passed']} "
          f"acceptance_passed={report['acceptance_passed']} errors={errors}", flush=True)
    if joint_online:
        print(f"Online policy={args.policy}; next-request recommendations ARE applied; no MPC fallback.", flush=True)
        print(f"model_switches={report['model_switches']} period_switches={report['period_switches']} "
              f"policy_aborts={report['timing']['policy_aborts']} "
              f"peer_aborts={report['timing']['peer_aborts']} "
              f"schedule_accounting_passed={report['schedule_accounting_passed']}", flush=True)
    if joint_shadow:
        print("Joint policies were evaluated AFTER flight; recommendations were NOT applied.", flush=True)
        print(f"Joint shadow report: {path / 'joint_shadow_report.json'}", flush=True)
        if capture_ms:
            print(f"shadow_capture_ms: mean={capture_ms['mean']:.3f} p99={capture_ms['p99']:.3f} "
                  f"max={capture_ms['max_observed']:.3f}", flush=True)
    for name in ("state_rpc_ms", "collision_rpc_ms", "preprocess_wait_ms", "reference_ms", "feature_ms",
                 "online_selection_ms", "mpc_solve_ms", "queue_ms", "service_ms", "sample_to_output_ms",
                 "release_jitter_ms", "release_to_output_ms"):
        value = report["timing"][name]
        if value:
            print(f"{name}: mean={value['mean']:.3f} p99={value['p99']:.3f} max={value['max_observed']:.3f}", flush=True)
    if errors or not metadata["all_15_crossings_passed"] or report["timing"]["inference_errors"]:
        raise SystemExit(1)
    if not report["realtime_passed"]:
        print("Flight geometry passed, but timing/schedule accounting FAILED; inspect report.json.", flush=True)
        raise SystemExit(2)
    if dynamic_control and not report["acceptance_passed"]:
        print("Switch trial FAILED: reference window coverage or sim/wall clock ratio.", flush=True)
        raise SystemExit(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("shadow", "dnn"), default="shadow")
    parser.add_argument("--model", choices=(*RUNS, "cycle"), default="residual21x1024")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--phase", choices=("sync", "staggered"), default="sync")
    parser.add_argument("--preprocess-policy", choices=("serialized", "parallel", "vectorized"), default="parallel",
                        help="Original parallel/serialized baseline, or equivalent batched reference and direct 237D encoding")
    parser.add_argument("--period-ms", type=float, default=50.)
    parser.add_argument("--deadline-ms", type=float, default=50.)
    parser.add_argument("--environment-id", type=int, choices=range(48, 54), default=48)
    parser.add_argument("--route-index", type=int, choices=range(20), default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=41451)
    parser.add_argument("--keep-gates", action="store_true")
    parser.add_argument("--output-dir", type=Path, help="New report directory; existing directories are refused")
    parser.add_argument("--state-budget-config", type=Path,
                        help="Post-flight causal T_max shadow replay only; does not change control or acceptance")
    args = parser.parse_args()
    if not all(np.isfinite(v) and v > 0 for v in (args.period_ms, args.deadline_ms)):
        parser.error("period/deadline must be positive and finite")
    if args.mode == "dnn" and args.model == "cycle":
        parser.error("cycle is an open-loop/shadow scheduling probe, not a validated switching controller")
    run(args)


if __name__ == "__main__":
    main()
