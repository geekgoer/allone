"""Versioned, initial-velocity-matched startup for the frozen gate polyline."""
from __future__ import annotations

import numpy as np

from gate_route import ContinuousGateRoute
from mpc_controller import MPCReference
from reference_generator import polyline_reference_at_time


def reference_contract(profile="constant", startup_duration=2.0):
    if profile == "constant":
        return {"version": "constant_speed_polyline_v1", "startup_duration_s": 0.0}
    if profile != "smooth-start" or not np.isfinite(startup_duration) or startup_duration <= 0:
        raise ValueError("smooth-start needs a finite positive startup duration")
    return {"version": "initial_velocity_quintic_start_polyline_v1", "startup_duration_s": float(startup_duration)}


class StartupPolylineReference:
    """Quintic startup with measured initial velocity and zero endpoint acceleration.

    The endpoint lies on the first straight segment. After the startup, the
    reference traverses the original polyline at the declared cruise speed.
    The ramp duration is fixed before trials and is never fitted to flight errors.
    """

    def __init__(self, waypoints, initial_velocity, cruise_speed=5.0, duration=2.0):
        self.points = np.asarray(waypoints, dtype=float).copy()
        self.initial_velocity = np.asarray(initial_velocity, dtype=float).reshape(3).copy()
        self.speed, self.duration = float(cruise_speed), float(duration)
        if (self.points.ndim != 2 or self.points.shape[1] != 3 or len(self.points) < 2 or
                not np.isfinite(self.points).all() or not np.isfinite(self.initial_velocity).all() or
                not np.isfinite([self.speed, self.duration]).all() or min(self.speed, self.duration) <= 0):
            raise ValueError("invalid startup reference arguments")
        deltas = np.diff(self.points, axis=0)
        lengths = np.linalg.norm(deltas, axis=1)
        nonzero = np.flatnonzero(lengths > 1e-12)
        if not len(nonzero):
            raise ValueError("startup needs a nonzero first route segment")
        direction = deltas[nonzero[0]] / lengths[nonzero[0]]
        self.initial_along_speed = float(self.initial_velocity @ direction)
        self.startup_distance = .5 * (self.initial_along_speed + self.speed) * self.duration
        if not 0 < self.startup_distance < lengths[nonzero[0]]:
            raise ValueError("startup endpoint must be inside the first straight segment")
        displacement = direction * self.startup_distance
        terminal_velocity = direction * self.speed
        d = displacement - self.initial_velocity * self.duration
        v = (terminal_velocity - self.initial_velocity) * self.duration
        self.c3, self.c4, self.c5 = 10*d-4*v, -15*d+7*v, 6*d-3*v

    def at(self, elapsed, orientation=(1., 0., 0., 0.)):
        elapsed = float(elapsed)
        if not np.isfinite(elapsed) or elapsed < 0:
            raise ValueError("elapsed time must be finite and nonnegative")
        if elapsed >= self.duration:
            return polyline_reference_at_time(self.points,
                elapsed=elapsed-self.duration+self.startup_distance/self.speed,
                speed=self.speed, orientation=orientation)
        u = elapsed / self.duration
        position = (self.points[0] + self.initial_velocity*elapsed +
                    self.c3*u**3 + self.c4*u**4 + self.c5*u**5)
        velocity = self.initial_velocity + (3*self.c3*u**2+4*self.c4*u**3+5*self.c5*u**4)/self.duration
        acceleration = (6*self.c3*u+12*self.c4*u**2+20*self.c5*u**3)/self.duration**2
        return MPCReference(position, orientation=np.asarray(orientation), velocity=velocity, acceleration=acceleration)


class SmoothStartupGateRoute(ContinuousGateRoute):
    def __init__(self, *args, initial_velocity, startup_duration=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.startup = StartupPolylineReference(self.waypoints, initial_velocity, self.reference_speed, startup_duration)

    def reference_at_time(self, now):
        if self.completed:
            return super().reference_at_time(now)
        return self.startup.at(max(0.0, float(now-self.start_time)), self.current_gate.orientation)

    def reference_window(self, now, *, dt, horizon):
        if dt <= 0 or horizon < 1:
            raise ValueError("dt and horizon must be positive")
        return [self.reference_at_time(now+dt*(index+1)) for index in range(horizon)]
