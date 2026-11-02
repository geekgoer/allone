"""Live AirSim portion of E1; common delivery slots with continuous physics."""

import time
from types import SimpleNamespace

import numpy as np

from .decision_dataset import atomic_json
from .state_mode_probe import PROTOCOL, recovery_metrics, risk_features


class TrialFailure(RuntimeError):
    def __init__(self, message, kind):
        super().__init__(message)
        self.kind = kind


def checked_dispatch(action, row, now, tolerance_s):
    """A missed comparison target is recorded; an expired job is never sent."""
    row["delivery_target_missed"] = bool(now > row["target_s"]+tolerance_s)
    row["deadline_missed"] = bool(now >= row["deadline_s"])
    row["timing_violation"] = row["delivery_target_missed"] or row["deadline_missed"]
    if row["deadline_missed"]:
        row["timing_violation"] = True
        raise TrialFailure("control deadline expired before dispatch; output NOT applied", "timing")
    clipped = np.clip(action, [2., -3., -3., -2.], [20., 3., 3., 2.])
    row.update(control_unclipped=np.asarray(action).tolist(), control=clipped.tolist(),
        clipped=bool(np.any(clipped != action)), command_send_s=now)
    return clipped


def flight(client, pool, layout, case, trial, args, directory):
    from airsim_interface import AirSimDrone, require_airsim, vehicle_local_to_world_offset
    from config import load_calibration
    from control_adapter import send_control, control_to_airsim_command
    from environment import Gate
    from gate_visualization import spawn_gate_sequences
    from .fast_convex_preprocess import FastReferenceWindow, encode
    from .fly_decision_dnn import StoredRoutePlanner
    from .probe_airsim_three import translate_layout, GATE
    from experiments.falcon_paper_fidelity_acado_v3_continuous.demo_acado_five_gates import _crossing_results

    api = require_airsim()
    trace, commands, scene, drone = [], [], None, None
    error = error_kind = None
    phase = "setup"
    period, delay, tolerance = args.period_ms/1000., args.delivery_ms/1000., args.delivery_tolerance_ms/1000.
    try:
        client.reset()
        client.simPause(True)
        offset = vehicle_local_to_world_offset(client, args.vehicle)
        local = translate_layout(layout, -offset)
        planner = StoredRoutePlanner(local)
        window = FastReferenceWindow(planner.trajectory, planner.config)
        gates = [Gate(g["index"], g["center"], g["orientation"]) for g in local["gates"]]
        # Arc-length progress is measured by nearest points on the fixed saved path.
        arc_times = np.linspace(0., planner.trajectory.duration, int(np.ceil(planner.trajectory.duration/.01))+1)
        arc_points = window._evaluate(arc_times)[0]
        arc_lengths = np.r_[0., np.cumsum(np.linalg.norm(np.diff(arc_points, axis=0), axis=1))]
        drone = AirSimDrone(client, args.vehicle)
        drone.enable_api_control(True)
        drone.arm()
        kin = api.KinematicsState()
        desired_p = np.asarray(case["position"])-offset
        q = case["quaternion"]
        kin.position = api.Vector3r(*map(float, desired_p))
        kin.orientation = api.Quaternionr(float(q[1]), float(q[2]), float(q[3]), float(q[0]))
        kin.linear_velocity = api.Vector3r(*map(float, case["velocity"]))
        kin.linear_acceleration = api.Vector3r(0., 0., 0.)
        kin.angular_velocity = api.Vector3r(0., 0., 0.)
        kin.angular_acceleration = api.Vector3r(0., 0., 0.)
        # Initial placement is a teleport, not a physical flight from the spawn.
        # setKinematics alone updates the physics body before UE renders it.
        client.simSetKinematics(kin, ignore_collision=True, vehicle_name=args.vehicle)
        world_pose = api.Pose(position_val=api.Vector3r(*map(float, case["position"])),
            orientation_val=kin.orientation)
        if not client.simSetObjectPose(args.vehicle, world_pose, teleport=True):
            raise TrialFailure("could not teleport the vehicle's UE actor to the initial state", "setup")
        scene = spawn_gate_sequences(client, {args.vehicle: gates}, width=PROTOCOL["gate_width_m"],
            height=PROTOCOL["gate_height_m"], thickness=PROTOCOL["gate_thickness_m"],
            asset_name="1M_Cube", object_prefix="FalconModeGate_", vehicle_world_offsets={args.vehicle: offset})
        calibration = load_calibration()
        # Identical, explicit old command, queued before physics resumes.
        send_control(drone, case["held_control"], calibration, duration=PROTOCOL["command_lease_s"])
        client.ping()
        raw = drone.get_raw_state()
        readback = [float(np.linalg.norm(raw.position-desired_p)),
            float(np.linalg.norm(raw.velocity-case["velocity"])),
            float(2*np.arccos(np.clip(abs(np.dot(raw.quaternion, q)), 0., 1.)))]
        matched = bool(np.isfinite(readback).all() and np.all(np.asarray(readback) <= [.02, .05, .02]))
        actor = client.simGetVehiclePose(vehicle_name=args.vehicle)
        actor_position = np.array([actor.position.x_val, actor.position.y_val, actor.position.z_val])
        actor_q = np.array([actor.orientation.w_val, actor.orientation.x_val, actor.orientation.y_val, actor.orientation.z_val])
        actor_error = [float(np.linalg.norm(actor_position-desired_p)),
            float(2*np.arccos(np.clip(abs(np.dot(actor_q, q)), 0., 1.)))]
        matched = bool(matched and np.isfinite(actor_error).all() and np.all(np.asarray(actor_error) <= [.02, .02]))
        atomic_json(directory/"initial_state.json", {"requested_world": case, "offset": offset.tolist(),
            "actual_local_position": raw.position.tolist(), "actual_velocity": raw.velocity.tolist(),
            "actual_quaternion": raw.quaternion.tolist(), "readback_position_velocity_rotation": readback,
            "actor_local_position": actor_position.tolist(), "actor_local_quaternion": actor_q.tolist(),
            "actor_readback_position_rotation": actor_error, "placement_teleport": True,
            "matched": matched, "held_airsim_command": vars(control_to_airsim_command(
                case["held_control"], calibration, PROTOCOL["command_lease_s"]))})
        if not matched or raw.timestamp_ns is None:
            raise TrialFailure("physics/actor initial-state readback mismatch or missing simulation timestamp", "setup")
        # Exercise the exact state/width/depth before starting the trial clock.
        warm_refs = window.window(case["reference_phase_s"])
        warm_feature = encode(raw, warm_refs)
        warm_times = [pool.predict(trial["mode"], warm_feature)[1]["inference_ms"] for _ in range(5)]
        atomic_json(directory/"warmup.json", {"actual_initial_state": True, "inference_ms": warm_times,
            "during_placement_pause": True, "control_not_applied": True})
        initial_sim = raw.timestamp_ns
        initial_collision_stamp = client.simGetCollisionInfo(vehicle_name=args.vehicle).time_stamp

        def observe(state, wall):
            elapsed = (state.timestamp_ns-initial_sim)/1e9
            if elapsed < 0. or (trace and elapsed < trace[-1]["t"]):
                raise TrialFailure("simulation clock moved backwards", "infrastructure")
            refs = window.window(case["reference_phase_s"]+elapsed)
            collision = client.simGetCollisionInfo(vehicle_name=args.vehicle)
            collided = bool(collision.has_collided and collision.time_stamp != initial_collision_stamp)
            values = np.r_[state.position, state.velocity, state.quaternion]
            if not np.isfinite(values).all():
                raise TrialFailure("nonfinite state", "flight")
            nearest = int(np.argmin(np.sum((arc_points-state.position)**2, axis=1)))
            row = {"t": float(elapsed), "wall_s": wall, "sim_ns": int(state.timestamp_ns),
                "position": state.position.tolist(), "velocity": state.velocity.tolist(),
                "quaternion": state.quaternion.tolist(), "reference_position": refs[0].position.tolist(),
                "reference_velocity": refs[0].velocity.tolist(), "reference_quaternion": refs[0].orientation.tolist(),
                "reference_phase_s": case["reference_phase_s"]+elapsed,
                "position_error_m": float(np.linalg.norm(state.position-refs[0].position)),
                "velocity_error_mps": float(np.linalg.norm(state.velocity-refs[0].velocity)),
                "path_arc_m": float(arc_lengths[nearest]),
                "reference_arc_m": float(np.interp(case["reference_phase_s"]+elapsed, arc_times, arc_lengths)),
                "collision": collided, "collision_object": str(collision.object_name), **risk_features(state, refs[0])}
            if collided:
                row["collision_timestamp_ns"] = int(collision.time_stamp)
                row["collision_position"] = [float(getattr(collision.position, key)) for key in ("x_val", "y_val", "z_val")]
                row["collision_impact_point"] = [float(getattr(collision.impact_point, key)) for key in ("x_val", "y_val", "z_val")]
            if trace and elapsed == trace[-1]["t"]:
                trace[-1] = row
            else:
                trace.append(row)
            if collided or row["h_m"] < 0.:
                raise TrialFailure(f"collision/boundary violation: {collision.object_name}; h={row['h_m']:.3f}", "flight")
            if state.position[2]+offset[2] > -.3 or np.linalg.norm(state.velocity) > 15. or row["position_error_m"] > 8.:
                raise TrialFailure("position/speed flight guard exceeded", "flight")
            return refs

        observe(raw, 0.)
        client.simPause(False)
        origin = time.perf_counter()
        phase, slot, next_observation, pending = "flight", 0, 0., None
        last_send = 0.
        while True:
            wall = time.perf_counter()-origin
            if wall > max(args.duration+2., args.duration*1.5):
                raise TrialFailure("simulation did not cover the fixed window in wall-time limit", "timing")
            if wall >= next_observation or (pending is None and wall >= slot*period):
                sample_start = time.perf_counter()-origin
                raw = drone.get_raw_state()
                refs = observe(raw, time.perf_counter()-origin)
                next_observation = time.perf_counter()-origin+PROTOCOL["observation_period_s"]
                if trace[-1]["t"] >= args.duration:
                    break
                if pending is None and sample_start >= slot*period:
                    row = {"slot": slot, "sample_id": f"{trial['id']}:{slot}", "mode": trial["mode"],
                        "release_s": slot*period, "sample_request_s": sample_start,
                        "sample_sim_ns": int(raw.timestamp_ns), "target_s": slot*period+delay,
                        "deadline_s": (slot+1)*period,
                        "reference_phase_s": case["reference_phase_s"]+trace[-1]["t"], "timing_violation": False}
                    commands.append(row)
                    action, timing = pool.predict(trial["mode"], encode(raw, refs))
                    row.update(timing, inference_ready_s=time.perf_counter()-origin)
                    pending = action, row
            if pending is not None and time.perf_counter()-origin >= pending[1]["target_s"]:
                action, row = pending
                now = time.perf_counter()-origin
                clipped = checked_dispatch(action, row, now, tolerance)
                send_control(drone, clipped, calibration, duration=PROTOCOL["command_lease_s"])
                client.ping()  # RPC pump return is a proxy, not an actuator acknowledgement.
                end = time.perf_counter()-origin
                row.update(rpc_return_s=end, command_send_lateness_ms=(now-row["target_s"])*1000,
                    sample_to_send_ms=(now-row["sample_request_s"])*1000,
                    sample_to_rpc_return_ms=(end-row["sample_request_s"])*1000,
                    command_send_gap_ms=(now-last_send)*1000, rpc_after_send_ms=(end-now)*1000,
                    airsim_command=vars(control_to_airsim_command(clipped, calibration, PROTOCOL["command_lease_s"])))
                row["rpc_return_after_deadline"] = bool(end >= row["deadline_s"])
                # A late RPC return cannot prove when the command took effect.
                # Keep the observation, but do not call its timing comparable.
                if row["rpc_return_after_deadline"]:
                    row["timing_violation"] = True
                last_send, slot, pending = now, slot+1, None
                skipped = max(0, int(np.floor((end-slot*period)/period)))
                row["skipped_releases"] = skipped
                if skipped:
                    row["timing_violation"] = True
                    slot += skipped
            target = min(next_observation, pending[1]["target_s"] if pending else slot*period)
            time.sleep(max(0., target-(time.perf_counter()-origin)))
    except KeyboardInterrupt:
        error, error_kind = "KeyboardInterrupt", "interrupted"
    except TrialFailure as exc:
        error, error_kind = str(exc), exc.kind
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        error_kind = "infrastructure" if type(exc).__module__.startswith(("msgpackrpc", "tornado")) else phase
    finally:
        try:
            client.simPause(False)
            if drone is not None:
                drone.hover()
                client.ping()
            if scene is not None:
                scene.destroy()
        except Exception as exc:
            error, error_kind = f"{error or ''}; cleanup: {exc}", "infrastructure"
    crossings = []
    if len(trace) > 1 and "gates" in locals():
        crossings = [{"gate": g, "crossed": c is not None, "valid": valid, "speed_mps": speed}
            for g, c, speed, valid in _crossing_results([r["position"] for r in trace],
                [r["velocity"] for r in trace], SimpleNamespace(gates=gates),
                {**GATE, "clearance": PROTOCOL["clearance_m"]+PROTOCOL["vehicle_radius_m"]})]
        if any(c["crossed"] and not c["valid"] for c in crossings) and error_kind is None:
            error, error_kind = "gate plane crossed outside the declared clear opening", "flight"
    report = {"trial": trial, "error": error, "error_kind": error_kind, "crossings": crossings,
        "status": "interrupted" if error_kind in ("interrupted", "infrastructure", "setup") else "finished",
        **recovery_metrics(trace, commands, args.duration, error_kind=error_kind)}
    atomic_json(directory/"trace.json", trace)
    atomic_json(directory/"commands.json", commands)
    atomic_json(directory/"report.json", report)
    return report
