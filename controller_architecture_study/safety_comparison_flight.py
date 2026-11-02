"""Actual simultaneous AirSim plants sharing one segmented CPU worker."""

from copy import deepcopy
from types import SimpleNamespace
import time

import numpy as np

from airsim_interface import AirSimDrone, require_airsim, vehicle_local_to_world_offset
from config import load_calibration
from control_adapter import send_control
from environment import Gate
from gate_route import segment_gate_crossing
from gate_visualization import spawn_gate_sequences
from state_adapter import normalize_quaternion
from .decision_dataset import atomic_json
from .fast_convex_preprocess import FastReferenceWindow, encode
from .fly_convex_policy import load_airsim_client
from .fly_decision_dnn import StoredRoutePlanner
from .probe_airsim_three import translate_layout
from .safety_comparison_core import (BODY_RADIUS, CLEARANCE, ProcessSegmentServer, joint_release_plan,
                                      paper_score, require_startup_vehicles, safe_horizon)


def crossing_candidates(points, gate):
    """Cheap plane filter; original crossing test still decides every candidate."""
    points=np.asarray(points,dtype=float)
    if len(points)<2:return []
    w,x,y,z=normalize_quaternion(gate.orientation)
    normal=np.array([1.-2.*(y*y+z*z),2.*(x*y+w*z),2.*(x*z-w*y)])
    local_x=(points-gate.center)@normal
    tolerance=1e-9*(1.+np.max(np.abs(local_x)))
    return np.flatnonzero((local_x[:-1]<=tolerance)&(local_x[1:]>=-tolerance)
        &(local_x[1:]-local_x[:-1]>0.))


def reference_crossings(points, gates):
    """Task gate IDs come from the saved reference, never from observed success."""
    return [g.index for g in gates if any(segment_gate_crossing(points[i],points[i+1],g) is not None
        for i in crossing_candidates(points,g))]


class SafetyWorld:
    def __init__(self, layout, phase, count, net_width, phase_span=0., *, task_duration=3., layouts=None, spawn_gates=True):
        load_airsim_client()
        api = require_airsim()
        self.client = api.MultirotorClient(timeout_value=3.)
        self.client.confirmConnection()
        self.vehicles = [f"Drone{i+1}" for i in range(count)]
        require_startup_vehicles(self.client, self.vehicles)
        if net_width <= 2*(BODY_RADIUS+CLEARANCE):
            raise ValueError("clear aperture is no larger than the assumed vehicle envelope plus clearance")
        self.phase, self.net_width = phase, net_width
        self.phases = {v: phase-phase_span*i/max(1, count-1) for i,v in enumerate(self.vehicles)}
        self.calibration = load_calibration()
        self.layout = deepcopy(layout)
        self.layouts = deepcopy(layouts) if layouts is not None else {v:self.layout for v in self.vehicles}
        if set(self.layouts)!=set(self.vehicles):
            raise ValueError("one layout required for each vehicle")
        self.scene = None
        self.offsets, self.windows, self.gates, self.drones, self.expected = {}, {}, {}, {}, {}
        if self.client.simIsPause():
            raise RuntimeError("manual Play must be active and unpaused")
        foreign = self.client.simListSceneObjects("Falcon.*Gate.*")
        if foreign:
            raise RuntimeError(f"generated gates already present; inspect first: {foreign[:5]}")
        self.client.reset()
        self.client.simPause(True)
        try:
            for i, vehicle in enumerate(self.vehicles):
                offset = vehicle_local_to_world_offset(self.client, vehicle)
                self.offsets[vehicle] = offset
                shift = np.array([0., (i-(count-1)/2)*12., 0.])-offset
                translated = translate_layout(self.layouts[vehicle], shift)
                planner = StoredRoutePlanner(translated)
                self.windows[vehicle] = FastReferenceWindow(planner.trajectory, planner.config)
                self.gates[vehicle] = [Gate(g["index"], g["center"], g["orientation"]) for g in translated["gates"]]
                points = self.windows[vehicle]._evaluate(self.phases[vehicle]+np.arange(0.,task_duration+.001,.02))[0]
                self.expected[vehicle] = reference_crossings(points, self.gates[vehicle])
                self.gates[vehicle] = [g for g in self.gates[vehicle] if g.index in self.expected[vehicle]]
                if not self.gates[vehicle]:
                    raise ValueError(f"fixed task has no reference gate crossings: {vehicle}")
                self.drones[vehicle] = AirSimDrone(self.client, vehicle)
            # width/height in the old helper are POST-CENTRE distances, not clear apertures.
            if spawn_gates:
                self.scene = spawn_gate_sequences(self.client, self.gates, width=net_width+.15,
                    height=4.+.15, thickness=.15, asset_name="1M_Cube", object_prefix="FalconSafetyGate_",
                    vehicle_world_offsets=self.offsets)
        finally:
            self.client.simPause(False)

    def close(self):
        # A dead RPC server must not mask the original exception with another timeout.
        try:
            self.client.simPause(False)
            for vehicle in self.vehicles:
                self.client.hoverAsync(vehicle_name=vehicle)
            self.client.ping()
            if self.scene is not None:
                self.scene.destroy()
                for name in self.client.simListSceneObjects("FalconSafetyGate_.*"):
                    if name in self.scene.object_names:
                        self.client.simDestroyObject(name)
        except Exception as exc:
            return {"status": "CLEANUP_FAILED", "error": f"{type(exc).__name__}: {exc}",
                "manual_restart_may_be_required": True}
        return {"status": "CLEANUP_COMPLETE"}

    def trial(self, pool, configuration, scenario, policy, directory, duration=3.):
        api = require_airsim()
        directory.mkdir(parents=True, exist_ok=False)
        self.client.reset()
        self.client.simPause(True)
        states, traces, failures, initial, requests, decisions = {}, {}, {}, {}, [], []
        models = configuration["models"]
        periods = configuration["periods_ms"]
        ordered = configuration["priority_order"]
        placements = {}
        for i, vehicle in enumerate(self.vehicles):
            print(f"  initial placement {vehicle}", flush=True)
            drone, window = self.drones[vehicle], self.windows[vehicle]
            ref = window.window(self.phases[vehicle])[0]
            p, v = ref.position.copy(), ref.velocity.copy()
            if i == scenario["critical_index"]:
                p[1] += scenario["position_y_m"]
                v[1] += scenario["velocity_y_mps"]
            drone.enable_api_control(True)
            drone.arm()
            kin = api.KinematicsState()
            kin.position, kin.linear_velocity = api.Vector3r(*map(float, p)), api.Vector3r(*map(float, v))
            q = ref.orientation
            kin.orientation = api.Quaternionr(float(q[1]), float(q[2]), float(q[3]), float(q[0]))
            kin.linear_acceleration = api.Vector3r(0., 0., 0.)
            kin.angular_velocity, kin.angular_acceleration = api.Vector3r(0., 0., 0.), api.Vector3r(0., 0., 0.)
            self.client.simSetKinematics(kin, ignore_collision=True, vehicle_name=vehicle)
            world_pose = api.Pose(api.Vector3r(*map(float, p+self.offsets[vehicle])), kin.orientation)
            if not self.client.simSetObjectPose(vehicle, world_pose, teleport=True):
                raise RuntimeError(f"could not teleport startup vehicle actor: {vehicle}")
            held = np.r_[np.linalg.norm(ref.acceleration-[0., 0., 9.80665]), 0., 0., 0.]
            send_control(drone, held, self.calibration, duration=1.)
            placements[vehicle] = (kin, p, v, q, held)
        # All pawns exist at startup: their actor poses are available without a
        # rendered initialization frame, which would otherwise change the state.
        for vehicle, (kin, p, v, q, held) in placements.items():
            drone = self.drones[vehicle]
            raw = drone.get_raw_state()
            matched = bool(np.linalg.norm(raw.position-p) <= .02 and np.linalg.norm(raw.velocity-v) <= .05
                           and abs(float(np.dot(raw.quaternion, q))) >= np.cos(.02/2))
            if not matched:
                raise RuntimeError(f"initial state mismatch: {vehicle}")
            initial[vehicle] = {"position": p.tolist(), "velocity": v.tolist(), "quaternion": q.tolist(),
                "held_control": held.tolist(), "matched": matched, "sim_ns": raw.timestamp_ns}
            states[vehicle] = raw
            traces[vehicle] = []
            failures[vehicle] = None
        initial_stamps = {v: int(self.client.simGetCollisionInfo(vehicle_name=v).time_stamp) for v in self.vehicles}
        atomic_json(directory/"initial_state.json", initial)
        atomic_json(directory/"trial_setup.json", {"configuration": configuration, "scenario": scenario,
            "policy": policy, "actual_vehicle_count": len(self.vehicles), "all_vehicles_created_at_startup": True,
            "world_offsets": {v: o.tolist() for v,o in self.offsets.items()},
            "initial_pose_method": "paused full kinematics and startup UE actor teleport, no unmeasured dynamics frame",
            "net_width_m": self.net_width, "net_height_m": 4., "post_center_width_m": self.net_width+.15,
            "body_radius_assumption_m": BODY_RADIUS, "extra_clearance_m": CLEARANCE,
            "reference_phase_s": self.phase, "reference_phases_s": self.phases,
            "expected_gate_ids": self.expected, "duration_s": duration,
            "no_MPC_fallback": True, "is_safety_certificate": False})
        server = ProcessSegmentServer(policy)
        if server.provenance != pool.provenance():
            server.close()
            raise RuntimeError("frontend and inference process artifact identities differ")
        atomic_json(directory/"worker_profile.json", server.profile)
        scheduled, pending, sequence, latest = {}, {}, {}, {}
        origin = None
        error = None
        try:
            self.client.simPause(False)
            origin = time.perf_counter()
            for i, vehicle in enumerate(self.vehicles):
                offset = i*periods[vehicle]/1000/len(self.vehicles) if configuration["phase"] == "staggered" else 0.
                sequence[vehicle], pending[vehicle] = 0, False
                scheduled[vehicle] = {"planned_release_s": origin+offset,
                    "deadline_s": origin+offset+periods[vehicle]/1000,
                    "vehicle": vehicle, "service_bound_s": configuration.get("job_costs_s", {}).get(vehicle,
                        .012+server.profile[models[vehicle]]["max_observed_ms"]/1000)}

            def observe(vehicle, raw, sample_s):
                elapsed = (raw.timestamp_ns-initial[vehicle]["sim_ns"])/1e9
                ref = self.windows[vehicle].window(self.phases[vehicle]+elapsed, horizon=1)[0]
                q, qr = normalize_quaternion(raw.quaternion), normalize_quaternion(ref.orientation)
                if np.dot(q, qr) < 0:
                    qr = -qr
                e = np.r_[raw.position-ref.position, raw.velocity-ref.velocity, q-qr]
                horizon, constraints = safe_horizon(raw.position, raw.velocity, self.gates[vehicle], self.net_width)
                row = {"t": float(elapsed), "sample_s": sample_s, "sim_ns": int(raw.timestamp_ns),
                    "position": raw.position.tolist(), "velocity": raw.velocity.tolist(), "quaternion": q.tolist(),
                    "reference_position": ref.position.tolist(), "reference_velocity": ref.velocity.tolist(),
                    "reference_quaternion": qr.tolist(), "state_cost": float(e@e),
                    "position_error_m": float(np.linalg.norm(e[:3])), "budget_s": horizon,
                    "gate_constraints": constraints}
                if traces[vehicle] and elapsed <= traces[vehicle][-1]["t"]:
                    return
                traces[vehicle].append(row)
                latest[vehicle] = row
                if traces[vehicle][:-1]:
                    for gate in self.gates[vehicle]:
                        cross = segment_gate_crossing(traces[vehicle][-2]["position"], row["position"], gate)
                        if cross is not None and (abs(cross.lateral_offset) > self.net_width/2-BODY_RADIUS-CLEARANCE
                                                or abs(cross.vertical_offset) > 2.-BODY_RADIUS-CLEARANCE):
                            row["invalid_crossing"] = gate.index
                if not np.isfinite(np.r_[raw.position, raw.velocity, raw.quaternion]).all():
                    raise RuntimeError("nonfinite simulation state")
                if raw.position[2]+self.offsets[vehicle][2] > -.3 or np.linalg.norm(raw.velocity) > 15. or np.linalg.norm(e[:3]) > 6.:
                    failures[vehicle] = {"kind": "flight_guard", "t": elapsed}

            for vehicle in self.vehicles:
                observe(vehicle, states[vehicle], origin)
                if policy != "static":
                    h = latest[vehicle]["budget_s"]
                    scheduled[vehicle]["deadline_s"] = origin+h
                    scheduled[vehicle]["planned_release_s"] = origin  # bootstrap has no prior neural command
            def sample(vehicle):
                sample_s = time.perf_counter()
                states[vehicle] = self.drones[vehicle].get_raw_state()
                observe(vehicle, states[vehicle], sample_s)
                collision = self.client.simGetCollisionInfo(vehicle_name=vehicle)
                if collision.has_collided and int(collision.time_stamp) != initial_stamps[vehicle]:
                    failures[vehicle] = {"kind": "collision", "object": str(collision.object_name),
                        "t": traces[vehicle][-1]["t"], "position": traces[vehicle][-1]["position"],
                        "collision_timestamp": int(collision.time_stamp),
                        "collision_position_raw": [float(collision.position.x_val), float(collision.position.y_val), float(collision.position.z_val)],
                        "impact_point_raw": [float(collision.impact_point.x_val), float(collision.impact_point.y_val), float(collision.impact_point.z_val)]}
                if failures[vehicle]:
                    self.drones[vehicle].hover()

            def replan_all(now, trigger):
                began = time.perf_counter()
                active = {v: scheduled[v] for v in self.vehicles if not failures[v] and traces[v][-1]["t"] < duration}
                revised, decision = joint_release_plan(now, active)
                scheduled.update(revised)
                decision.update(vehicle=trigger, deadline_s=scheduled[trigger]["deadline_s"] if trigger in scheduled else None,
                    overhead_ms=1000*(time.perf_counter()-began))
                decisions.append(decision)

            if policy != "static":
                replan_all(time.perf_counter(), "bootstrap")

            while True:
                now = time.perf_counter()
                if now-origin > duration+3.:
                    error = "fixed simulation window not covered within wall-time limit"
                    break
                for action, row in server.drain():
                    vehicle = row["vehicle"]
                    pending[vehicle] = False
                    now = time.perf_counter()
                    row["handled_s"] = now
                    if failures[vehicle] or traces[vehicle][-1]["t"] >= duration:
                        row["status"] = "cancelled_after_flight"
                        continue
                    if action is not None and now < row["deadline_s"]:
                        clipped = np.clip(action, [2., -3., -3., -2.], [20., 3., 3., 2.])
                        row.update(command_send_s=now, control=clipped.tolist())
                        send_control(self.drones[vehicle], clipped, self.calibration, duration=1.)
                        self.client.ping()
                        row.update(rpc_return_s=time.perf_counter(), status="applied")
                        now = row["rpc_return_s"]
                    else:
                        row["status"] = "expired_not_applied"
                        if policy != "static":
                            failures[vehicle] = {"kind": "dynamic_deadline_expired", "t": traces[vehicle][-1]["t"]}
                            self.drones[vehicle].hover()
                            continue
                    if policy == "static":
                        release = row["planned_release_s"]+periods[vehicle]/1000
                        skipped = max(0, int(np.floor((now-release)/(periods[vehicle]/1000))))
                        release += skipped*periods[vehicle]/1000
                        row["skipped_periodic_releases"] = skipped
                        scheduled[vehicle] = {"planned_release_s": release, "deadline_s": release+periods[vehicle]/1000,
                            "vehicle": vehicle, "service_bound_s": row["service_bound_s"]}
                    else:
                        observation = latest[vehicle]
                        scheduled[vehicle] = {"planned_release_s": now,
                            "deadline_s": min(observation["sample_s"]+observation["budget_s"], now+.150),
                            "vehicle": vehicle, "service_bound_s": row["service_bound_s"]}
                        replan_all(now, vehicle)
                # Acquire state as part of a released control job, not an unrelated
                # always-on N-vehicle polling workload. Final samples cover the task window.
                if not any(pending.values()) and now-origin >= duration:
                    unfinished = [v for v in self.vehicles if not failures[v] and traces[v][-1]["t"] < duration]
                    if unfinished:
                        sample(unfinished[0])
                for vehicle in sorted(self.vehicles, key=lambda v: (ordered.index(v) if policy == "static"
                        else scheduled[v]["deadline_s"], scheduled[v]["planned_release_s"])):
                    if failures[vehicle] or traces[vehicle][-1]["t"] >= duration or pending[vehicle]:
                        continue
                    if any(pending.values()):
                        break  # One exclusive full job for both policies, including the frontend.
                    now = time.perf_counter()
                    slot = scheduled[vehicle]
                    if now < slot["planned_release_s"]:
                        continue
                    job_started_s = time.perf_counter()
                    sample(vehicle)
                    now = time.perf_counter()
                    if failures[vehicle] or traces[vehicle][-1]["t"] >= duration:
                        continue
                    if policy != "static" and slot["deadline_s"] <= now:
                        # Never extend a physical deadline simply because the planner was late.
                        failures[vehicle] = {"kind": "unreleased_dynamic_deadline_expired", "t": traces[vehicle][-1]["t"]}
                        decisions.append({"vehicle": vehicle, "status": "deadline_expired_before_release",
                            "deadline_s": slot["deadline_s"], "observed_s": now})
                        self.drones[vehicle].hover()
                        continue
                    refs = self.windows[vehicle].window(self.phases[vehicle]+traces[vehicle][-1]["t"])
                    row = {**slot, "sample_id": f"{vehicle}:{sequence[vehicle]}", "sample_s": latest[vehicle]["sample_s"],
                        "job_started_s": job_started_s, "state_done_s": now,
                        "sample_age_at_dispatch_ms": 1000*(now-latest[vehicle]["sample_s"]),
                        "model": models[vehicle], "priority": ordered.index(vehicle), "status": "submitted"}
                    requests.append(row)
                    feature = encode(states[vehicle], refs)
                    row["feature_done_s"] = time.perf_counter()
                    server.submit(feature, row)
                    pending[vehicle] = True
                    sequence[vehicle] += 1
                    break  # Interleave the next ready-control drain with feature construction.
                if all(failures[v] or traces[v][-1]["t"] >= duration for v in self.vehicles):
                    break
                time.sleep(.0005)
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            server.close()
            atomic_json(directory/"traces.json", traces)
            atomic_json(directory/"requests.json", requests)
            atomic_json(directory/"segments.json", server.segments)
            atomic_json(directory/"decisions.json", decisions)
            try:
                self.client.simPause(False)
                for vehicle in self.vehicles:
                    self.client.hoverAsync(vehicle_name=vehicle)
                self.client.ping()
            except Exception as exc:
                error = error or f"cleanup {type(exc).__name__}: {exc}"
        from experiments.falcon_paper_fidelity_acado_v3_continuous.demo_acado_five_gates import _crossing_results
        reports = {}
        for vehicle in self.vehicles:
            trace = traces[vehicle]
            covered = bool(trace and trace[-1]["t"] >= duration)
            crossings = _crossing_results([r["position"] for r in trace], [r["velocity"] for r in trace],
                SimpleNamespace(gates=self.gates[vehicle]), {"width": self.net_width+.15, "height": 4.+.15,
                    "thickness": .15, "clearance": BODY_RADIUS+CLEARANCE}) if len(trace) >= 2 else []
            valid = sum(c[3] for c in crossings)
            ratio = ((trace[-1]["sim_ns"]-trace[0]["sim_ns"])/1e9/(trace[-1]["sample_s"]-trace[0]["sample_s"])) if len(trace) >= 2 else None
            passed = bool(covered and not failures[vehicle] and valid == len(self.expected[vehicle]))
            reports[vehicle] = {"failure": failures[vehicle], "window_covered": covered,
                "valid_crossings": valid, "expected_crossings": len(self.expected[vehicle]), "success": passed,
                "clock_ratio": ratio, "score": paper_score(trace, duration) if passed else None}
        scores = [r["score"]["J"] for r in reports.values() if r["score"] and r["score"]["J"] is not None]
        result = {"policy": policy, "scenario": scenario, "configuration": configuration, "error": error,
            "actual_vehicles": len(self.vehicles), "vehicles": reports,
            "mission_passed": not error and all(r["success"] for r in reports.values()),
            "clock_valid": all(r["clock_ratio"] is not None and .95 <= r["clock_ratio"] <= 1.05 for r in reports.values()),
            "total_J": float(sum(scores)) if len(scores) == len(self.vehicles) and not error else None,
            "collisions": sum(r["failure"] is not None and r["failure"]["kind"] == "collision" for r in reports.values()),
            "requests": len(requests), "applied": sum(r["status"] == "applied" for r in requests),
            "worker_compute_ms": sum(1000*(s["end_s"]-s["start_s"]) for s in server.segments),
            "worker_occupied_fraction_of_task_window": sum(s["end_s"]-s["start_s"] for s in server.segments)/duration,
            "expired_not_applied": sum(r["status"].startswith("expired") for r in requests),
            "rpc_after_deadline": sum(r.get("rpc_return_s", -np.inf) > r["deadline_s"] for r in requests),
            "infeasible_estimates": sum(not r["estimated_feasible"] for r in decisions if "estimated_feasible" in r),
            "selection_overhead_max_ms": max((r["overhead_ms"] for r in decisions if "overhead_ms" in r), default=0.),
            "command_timing_semantics": "asynchronous command send plus ping return is an RPC proxy, not a hardware actuation timestamp",
            "inference_runtime": "one shared segmented CPU worker in a separate Python process; IPC included in response time",
            "frontend_inflight_limit": 1, "layer_preemption_exercised_in_flight": False,
            "scheduling_model": "one conservative serialized full-job resource; predictive costs, not WCET",
            "scope": "FP32 CPU framework-component pilot, no SVM/FPGA/CBF guarantee; same modes and layer segmentation"}
        atomic_json(directory/"report.json", result)
        return result
