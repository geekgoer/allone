"""Shared placement, authoritative collision snapshots and physical scoring."""

import time

import numpy as np

from airsim_interface import RawDroneState, require_airsim
from control_adapter import send_control
from gate_route import segment_gate_crossing, _gate_local
from state_adapter import normalize_quaternion
from .decision_dataset import atomic_json
from .framework_runtime import route_risk
from .framework_runtime_features import reference_point
from .safety_comparison_core import BODY_RADIUS, CLEARANCE, paper_score
from .safety_comparison_flight import SafetyWorld, crossing_candidates


class FrameworkWorld(SafetyWorld):
    def initialize(self, scenario, directory):
        api = require_airsim()
        self.client.reset()
        self.client.simPause(True)
        initial,states = {},{}
        for i,v in enumerate(self.vehicles):
            ref = self.windows[v].window(self.phases[v],horizon=1)[0]
            p,velocity,q = ref.position.copy(),ref.velocity.copy(),ref.orientation.copy()
            if i == scenario.get("critical_index",-1):
                p[1] += scenario.get("position_y_m",0.)
                velocity[1] += scenario.get("velocity_y_mps",0.)
            if "profile_risk" in scenario:
                from .state_mode_probe import normal_frame
                normal,derivative = normal_frame(ref)
                h,hdot = scenario["profile_risk"]
                p += normal*(self.net_width/2-BODY_RADIUS-CLEARANCE-h)
                velocity += normal*(-hdot-float((p-ref.position)@derivative))
            if "relative_initial" in scenario:
                override = scenario["relative_initial"]
                p = ref.position+np.asarray(override["position_error"],dtype=float)
                velocity = ref.velocity+np.asarray(override["velocity_error"],dtype=float)
                if "quaternion" in override:
                    q = normalize_quaternion(override["quaternion"])
            drone = self.drones[v]
            drone.enable_api_control(True)
            drone.arm()
            kin = api.KinematicsState()
            kin.position = api.Vector3r(*map(float,p))
            kin.linear_velocity = api.Vector3r(*map(float,velocity))
            kin.orientation = api.Quaternionr(float(q[1]),float(q[2]),float(q[3]),float(q[0]))
            derivatives = scenario.get("relative_initial",{})
            kin.linear_acceleration = api.Vector3r(*map(float,derivatives.get("acceleration",[0.,0.,0.])))
            kin.angular_velocity = api.Vector3r(*map(float,derivatives.get("angular_velocity",[0.,0.,0.])))
            kin.angular_acceleration = api.Vector3r(0.,0.,0.)
            self.client.simSetKinematics(kin,ignore_collision=True,vehicle_name=v)
            if not self.client.simSetObjectPose(v,api.Pose(api.Vector3r(*map(float,p+self.offsets[v])),kin.orientation),teleport=True):
                raise RuntimeError(f"could not place startup actor {v}")
            held = np.r_[np.linalg.norm(ref.acceleration-[0.,0.,9.80665]),0.,0.,0.]
            if "relative_initial" in scenario and "held_control" in scenario["relative_initial"]:
                held = np.asarray(scenario["relative_initial"]["held_control"],dtype=float)
            send_control(drone,held,self.calibration,duration=1.)
            raw,collision,_ = self.snapshot(v)
            if (np.linalg.norm(raw.position-p) > .02 or np.linalg.norm(raw.velocity-velocity) > .05
                    or abs(float(raw.quaternion@q)) < np.cos(.02/2)):
                raise RuntimeError(f"initial state mismatch {v}")
            initial[v] = {"position":p.tolist(),"velocity":velocity.tolist(),"quaternion":q.tolist(),
                "held_control":held.tolist(),"sim_ns":int(raw.timestamp_ns),
                "collision_stamp":int(collision.time_stamp),"matched":True}
            states[v] = raw
        self.initial,self.traces,self.failures = initial,{v:[] for v in self.vehicles},{v:None for v in self.vehicles}
        atomic_json(directory/"initial_state.json",initial)
        return states

    def snapshot(self, vehicle):
        return self.snapshot_from(self.client,vehicle)

    def park_vehicle(self, vehicle, state):
        """Failure-only stop: retire neural actuation and hover this vehicle."""
        api=require_airsim()
        self.client.cancelLastTask(vehicle_name=vehicle)
        kin=api.KinematicsState()
        kin.position=api.Vector3r(*map(float,state.position))
        q=state.quaternion
        kin.orientation=api.Quaternionr(float(q[1]),float(q[2]),float(q[3]),float(q[0]))
        kin.linear_velocity=api.Vector3r(0.,0.,0.)
        kin.linear_acceleration=api.Vector3r(0.,0.,0.)
        kin.angular_velocity=api.Vector3r(0.,0.,0.)
        kin.angular_acceleration=api.Vector3r(0.,0.,0.)
        self.client.simSetKinematics(kin,ignore_collision=False,vehicle_name=vehicle)
        self.client.hoverAsync(vehicle_name=vehicle)
        self.client.ping()

    @staticmethod
    def snapshot_from(client, vehicle):
        began = time.perf_counter()
        packet = client.getMultirotorState(vehicle_name=vehicle)
        # This AirSim build never populates MultirotorState.collision.
        # The dedicated API also retains its event stamp after has_collided resets.
        collision = client.simGetCollisionInfo(vehicle_name=vehicle)
        kin = packet.kinematics_estimated
        xyz = lambda a: np.array([a.x_val,a.y_val,a.z_val],dtype=float)
        q = kin.orientation
        state = RawDroneState(xyz(kin.position),np.array([q.w_val,q.x_val,q.y_val,q.z_val]),
            xyz(kin.linear_velocity),xyz(kin.linear_acceleration),xyz(kin.angular_velocity),int(packet.timestamp))
        return state,collision,began

    def observe(self, vehicle, raw, collision, sampled, *, record=True):
        elapsed = (raw.timestamp_ns-self.initial[vehicle]["sim_ns"])/1e9
        if elapsed<0.:raise RuntimeError("simulation clock moved before trial origin")
        ref = reference_point(self.windows[vehicle],self.phases[vehicle]+elapsed)
        q,qr = normalize_quaternion(raw.quaternion),normalize_quaternion(ref.orientation)
        if q@qr < 0:
            qr = -qr
        e = np.r_[raw.position-ref.position,raw.velocity-ref.velocity,q-qr]
        collision_stamp = int(collision.time_stamp)
        collided = collision_stamp > max(0,self.initial[vehicle]["collision_stamp"])
        row = {"t":float(elapsed),"sample_s":sampled,"sim_ns":int(raw.timestamp_ns),
            "position":raw.position.tolist(),"velocity":raw.velocity.tolist(),"quaternion":q.tolist(),
            "acceleration":raw.acceleration.tolist(),"angular_velocity":raw.angular_velocity.tolist(),
            "reference_position":ref.position.tolist(),"reference_velocity":ref.velocity.tolist(),
            "reference_quaternion":qr.tolist(),"position_error_m":float(np.linalg.norm(e[:3])),
            "velocity_error_mps":float(np.linalg.norm(e[3:6])),"state_cost":float(e@e),
            "collision":collided,"collision_object":str(collision.object_name),
            "collision_stamp":collision_stamp,"collision_flag":bool(collision.has_collided),
            "risk":route_risk(raw,ref,self.net_width)}
        trace = self.traces[vehicle]
        if record and trace and elapsed < trace[-1]["t"]:
            raise RuntimeError("simulation clock moved backwards")
        if record:
            if trace and elapsed == trace[-1]["t"]:
                trace[-1] = row
            else:
                trace.append(row)
        if collided and (self.failures[vehicle] is None or self.failures[vehicle]["kind"] != "collision"):
            self.failures[vehicle] = {"kind":"collision","object":str(collision.object_name),"t":elapsed,
                "time_stamp":collision_stamp,"source":"simGetCollisionInfo"}
        if not np.isfinite(np.r_[raw.position,raw.velocity,raw.quaternion]).all():
            raise RuntimeError("nonfinite state")
        if raw.position[2]+self.offsets[vehicle][2] > -.3 or np.linalg.norm(raw.velocity) > 15. or np.linalg.norm(e[:3]) > 6.:
            self.failures[vehicle] = self.failures[vehicle] or {"kind":"flight_guard","t":elapsed}
        return row

    def physical_report(self, duration, *, enforce_tube=False, realtime=True):
        reports = {}
        for v,trace in self.traces.items():
            crossings = []
            positions=[r["position"] for r in trace]
            candidates=sorted((i,j) for j,gate in enumerate(self.gates[v])
                for i in crossing_candidates(positions,gate))
            for i,j in candidates:
                a,b,gate=trace[i],trace[i+1],self.gates[v][j]
                cross = segment_gate_crossing(a["position"],b["position"],gate)
                if cross is None:
                    continue
                x0,x1 = _gate_local(np.array(a["position"]),gate)[0],_gate_local(np.array(b["position"]),gate)[0]
                t = a["t"]+(b["t"]-a["t"])*(-x0)/(x1-x0)
                if t <= duration:
                    crossings.append({"gate":gate.index,"t":t,
                        "valid":bool(abs(cross.lateral_offset) <= self.net_width/2-BODY_RADIUS-CLEARANCE
                            and abs(cross.vertical_offset) <= 2.-BODY_RADIUS-CLEARANCE),
                        "lateral":cross.lateral_offset,"vertical":cross.vertical_offset})
            covered = bool(trace and trace[-1]["t"] >= duration)
            first_tail = max(0,int(np.searchsorted([r["t"] for r in trace],duration-.25,side="right"))-1)
            # Later cleanup snapshots are outside the task; keep only its first covering tick.
            last_tail = int(np.searchsorted([r["t"] for r in trace],duration,side="left"))
            tail = trace[first_tail:last_tail+1]
            tail_gap = float(max(np.diff([r["t"] for r in tail]),default=0.))
            tail_coverage = bool(tail and tail[0]["t"] <= duration-.25 and tail[-1]["t"] >= duration and tail_gap <= .10)
            tail_bounds = bool(tail and all(r["position_error_m"] <= .3 and r["velocity_error_mps"] <= .6 for r in tail))
            tail_ok = tail_coverage and tail_bounds
            tube_ok = bool(trace and all(r["risk"]["h"] >= 0 for r in trace))
            reference_progress = trace[-1]["reference_position"][0]-trace[0]["reference_position"][0] if len(trace)>1 else 0.
            progress = (trace[-1]["position"][0]-trace[0]["position"][0])/reference_progress if reference_progress>1e-6 else None
            crossed = {r["gate"] for r in crossings if r["valid"]}
            ratio = ((trace[-1]["sim_ns"]-trace[0]["sim_ns"])/1e9/(trace[-1]["sample_s"]-trace[0]["sample_s"])) if len(trace)>1 else None
            physical = bool(covered and not self.failures[v] and crossed == set(self.expected[v]) and progress is not None and progress>=.8)
            reports[v] = {"failure":self.failures[v],"window_covered":covered,"crossings":crossings,
                "expected_gate_ids":self.expected[v],"progress_ratio":progress,"tail_tracking_passed":tail_ok,
                "tail_observation_coverage_passed":tail_coverage,"tail_state_bounds_observed_passed":tail_bounds,
                "max_tail_observation_gap_s":tail_gap,
                "tube_observed_passed":tube_ok,"minimum_h":min((r["risk"]["h"] for r in trace),default=None),
                "physical_task_passed":physical,"recovery_passed":physical and tail_ok and (tube_ok or not enforce_tube),
                "clock_ratio":ratio,"clock_valid":bool(not realtime or (ratio is not None and .95<=ratio<=1.05)),
                "score":paper_score(trace,duration) if physical else None}
        return reports

    def hover_all(self):
        self.client.simPause(False)
        for v in self.vehicles:
            self.drones[v].hover()
        self.client.ping()
