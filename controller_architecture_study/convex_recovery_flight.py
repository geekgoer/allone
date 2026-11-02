"""Physical AirSim state kicks; no sensor-only noise and no trajectory replanning."""

from copy import deepcopy
import time

import numpy as np

from state_adapter import quaternion_inverse, quaternion_multiply, quaternion_to_rotation_vector


PROFILES = {
    # Position/velocity are forward, lateral, NED-down; attitude is a body rotation vector.
    "combined_left": {"position": [0., -.6, .3], "velocity": [-.8, -.8, .3], "rotation": [-.12, -.12, -.12]},
    "combined_right": {"position": [0., .6, -.3], "velocity": [-.8, .8, -.3], "rotation": [.12, -.12, .12]},
}
PROTOCOL = {"kick_time_seconds": 3., "position_tolerance_m": .3,
            "velocity_tolerance_mps": .6, "hold_seconds": .25, "recovery_timeout_seconds": 3.}


def vector(value):
    return np.array([value.x_val, value.y_val, value.z_val], dtype=float)


def quaternion(value):
    return np.array([value.w_val, value.x_val, value.y_val, value.z_val], dtype=float)


def snapshot(kinematics):
    return {"position": vector(kinematics.position).tolist(),
            "velocity": vector(kinematics.linear_velocity).tolist(),
            "quaternion_wxyz": quaternion(kinematics.orientation).tolist()}


class RecoveryRecorder:
    """Add reference velocity without changing the controller or frozen training code."""
    def __init__(self, controller):
        self.controller = controller

    def __getattr__(self, name):
        return getattr(self.controller, name)

    def solve(self, state, references):
        action = self.controller.solve(state, references)
        self.records[-1]["reference_velocity"] = references[0].velocity.copy()
        return action


class PerturbedClient:
    def __init__(self, client, controller, *, profile, heading_yaw, dt, vehicle_name, kick_time=3.):
        if profile not in PROFILES or not np.isfinite(kick_time) or kick_time <= 0 or dt <= 0:
            raise ValueError("use a known disturbance and a positive kick time/control period")
        self.client, self.controller = client, controller
        self.profile, self.vehicle_name, self.dt = profile, vehicle_name, dt
        self.kick_step = int(np.ceil(kick_time/dt))
        c, s = np.cos(heading_yaw), np.sin(heading_yaw)
        self.rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        self.event = None

    def __getattr__(self, name):
        return getattr(self.client, name)

    def getMultirotorState(self, vehicle_name=""):
        state = self.client.getMultirotorState(vehicle_name=vehicle_name)
        step = len(self.controller.records)
        if vehicle_name != self.vehicle_name or self.event is not None or step < self.kick_step:
            return state
        before = state.kinematics_estimated
        requested = deepcopy(before)
        profile = PROFILES[self.profile]
        for field, delta in (("position", profile["position"]), ("linear_velocity", profile["velocity"])):
            value = vector(getattr(before, field))+self.rotation @ np.asarray(delta)
            target = getattr(requested, field)
            target.x_val, target.y_val, target.z_val = map(float, value)
        rotation = np.asarray(profile["rotation"])
        angle = np.linalg.norm(rotation)
        dq = np.r_[np.cos(angle/2), rotation*np.sin(angle/2)/angle]
        q = quaternion_multiply(quaternion(before.orientation), dq)
        requested.orientation.w_val, requested.orientation.x_val, requested.orientation.y_val, requested.orientation.z_val = map(float, q)
        self.event = {"profile": self.profile, "step": step, "simulation_time_seconds": step*self.dt,
                      "before": snapshot(before), "requested": snapshot(requested), "applied": False}
        paused = self.client.simIsPause()
        started = time.perf_counter()
        try:
            self.client.simPause(True)
            # New collisions remain enabled. The updated state is read before collector safety checks.
            self.client.simSetKinematics(requested, ignore_collision=False, vehicle_name=vehicle_name)
            state = self.client.getMultirotorState(vehicle_name=vehicle_name)
            actual = state.kinematics_estimated
            errors = [np.linalg.norm(vector(actual.position)-vector(requested.position)),
                      np.linalg.norm(vector(actual.linear_velocity)-vector(requested.linear_velocity)),
                      np.linalg.norm(quaternion_to_rotation_vector(quaternion_multiply(
                          quaternion(actual.orientation), quaternion_inverse(quaternion(requested.orientation)))))]
            self.event.update(actual=snapshot(actual), readback_error_position_velocity_rotation=list(map(float, errors)),
                              applied=bool(np.isfinite(errors).all() and np.all(np.asarray(errors) <= [.02, .05, .02])))
            if not self.event["applied"]:
                raise RuntimeError("AirSim physical kick readback mismatch; do not count this as a recovery test")
        finally:
            self.client.simPause(paused)
            self.event["injection_wall_ms"] = (time.perf_counter()-started)*1000
        print(f"Physical kick at t={step*self.dt:.2f}s: {self.profile}; path clock unchanged", flush=True)
        return state


def recovery_metrics(records, event, dt):
    protocol = {**PROTOCOL, "kick_time_seconds": None if not event else event.get("simulation_time_seconds")}
    result = {"protocol": protocol, "event": event, "recovered": False,
              "confirmation_time_after_kick_seconds": None}
    if not event or not event["applied"] or len(records) <= event["step"]:
        result["reason"] = "kick not applied/read back or no post-kick samples"
        return result
    start = event["step"]
    errors = np.asarray([[np.linalg.norm(row["position"]-row["reference_position"]),
                          np.linalg.norm(row["velocity"]-row["reference_velocity"])] for row in records[start:]])
    result.update(post_kick_peak_position_error_m=float(errors[:, 0].max()),
                  post_kick_peak_velocity_error_mps=float(errors[:, 1].max()))
    hold_steps = int(np.ceil(PROTOCOL["hold_seconds"]/dt))+1
    good = np.isfinite(errors).all(axis=1) & (errors[:, 0] <= PROTOCOL["position_tolerance_m"]) & (
        errors[:, 1] <= PROTOCOL["velocity_tolerance_mps"])
    for end in range(hold_steps-1, len(good)):
        if end*dt > PROTOCOL["recovery_timeout_seconds"]+1e-9:
            break
        if good[end-hold_steps+1:end+1].all():
            result.update(recovered=True, confirmation_time_after_kick_seconds=end*dt)
            return result
    result["reason"] = "position/velocity did not remain within limits for the required hold within 3s"
    return result
