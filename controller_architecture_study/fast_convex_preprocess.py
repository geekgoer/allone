"""Batched, numerically equivalent preprocessing for the live 237D probe.

No sampled lookup table, path refitting, horizon change or model/data writes.
The original scalar planner/encoder remain the reference implementation.
"""

import numpy as np

from airsim_interface import RawDroneState
from mpc_controller import MPCReference


VECTOR_FLIP = np.array([1., -1., -1.])
QUATERNION_FLIP = np.array([1., 1., -1., -1.])


def _normalize(q):
    q = np.asarray(q, dtype=np.float64)
    length = np.linalg.norm(q, axis=-1, keepdims=True)
    if not np.isfinite(q).all() or np.any(length < 1e-12):
        raise ValueError("expected finite nonzero quaternions")
    return q / length


def _multiply(a, b):
    w, x, y, z = a.T
    v, i, j, k = b.T
    return np.column_stack((w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j,
                            w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v))


def _matrix_quaternions(matrix):
    """Same trace/diagonal branches as the scalar planner, applied in batches."""
    diagonal = np.diagonal(matrix, axis1=1, axis2=2)
    trace = diagonal.sum(axis=1)
    branch = np.where(trace > 0., -1, np.argmax(diagonal, axis=1))
    result = np.empty((len(matrix), 4))
    for index in (-1, 0, 1, 2):
        mask = branch == index
        m = matrix[mask]
        if not len(m):
            continue
        if index == -1:
            s = 2.*np.sqrt(trace[mask]+1.)
            result[mask] = np.column_stack((.25*s, (m[:, 2, 1]-m[:, 1, 2])/s,
                (m[:, 0, 2]-m[:, 2, 0])/s, (m[:, 1, 0]-m[:, 0, 1])/s))
        elif index == 0:
            s = 2.*np.sqrt(np.maximum(1.+m[:, 0, 0]-m[:, 1, 1]-m[:, 2, 2], 0.))
            result[mask] = np.column_stack(((m[:, 2, 1]-m[:, 1, 2])/s, .25*s,
                (m[:, 0, 1]+m[:, 1, 0])/s, (m[:, 0, 2]+m[:, 2, 0])/s))
        elif index == 1:
            s = 2.*np.sqrt(np.maximum(1.+m[:, 1, 1]-m[:, 0, 0]-m[:, 2, 2], 0.))
            result[mask] = np.column_stack(((m[:, 0, 2]-m[:, 2, 0])/s,
                (m[:, 0, 1]+m[:, 1, 0])/s, .25*s, (m[:, 1, 2]+m[:, 2, 1])/s))
        else:
            s = 2.*np.sqrt(np.maximum(1.+m[:, 2, 2]-m[:, 0, 0]-m[:, 1, 1], 0.))
            result[mask] = np.column_stack(((m[:, 1, 0]-m[:, 0, 1])/s,
                (m[:, 0, 2]+m[:, 2, 0])/s, (m[:, 1, 2]+m[:, 2, 1])/s, .25*s))
    return _normalize(result)


def _attitudes(acceleration, heading, gravity):
    gate = _normalize(heading * QUATERNION_FLIP)
    w, x, y, z = gate.T
    forward = np.column_stack((1.-2.*(y*y+z*z), 2.*(x*y+z*w), np.zeros(len(gate))))
    length = np.linalg.norm(forward, axis=1)
    forward[length < 1e-9] = [1., 0., 0.]
    forward /= np.linalg.norm(forward, axis=1, keepdims=True)
    body_z = acceleration * VECTOR_FLIP + [0., 0., gravity]
    body_z /= np.maximum(np.linalg.norm(body_z, axis=1, keepdims=True), 1e-12)
    body_y = np.cross(body_z, forward)
    fallback = np.linalg.norm(body_y, axis=1) < 1e-9
    body_y[fallback] = np.cross(body_z[fallback], [0., 1., 0.])
    body_y /= np.linalg.norm(body_y, axis=1, keepdims=True)
    body_x = np.cross(body_y, body_z)
    body_x /= np.linalg.norm(body_x, axis=1, keepdims=True)
    return _matrix_quaternions(np.stack((body_x, body_y, body_z), axis=2))


def _body_rates(current, following, dt):
    q0, q1 = _normalize(current), _normalize(following)
    q1[np.sum(q0*q1, axis=1) < 0.] *= -1.
    inverse = q0 * [1., -1., -1., -1.]
    delta = _normalize(_multiply(inverse, q1))
    delta[delta[:, 0] < 0.] *= -1.
    length = np.linalg.norm(delta[:, 1:], axis=1)
    rates = np.zeros((len(delta), 3))
    active = length >= 1e-12
    angle = 2.*np.arctan2(length[active], np.clip(delta[active, 0], -1., 1.))
    rates[active] = delta[active, 1:] / length[active, None] * (angle[:, None]/dt)
    return rates


class FastReferenceWindow:
    """Evaluate the saved continuous polynomial at the actual requested times."""
    def __init__(self, trajectory, config):
        self.trajectory, self.config = trajectory, config
        self.coefficients = np.stack([s.primitive.coefficients for s in trajectory.segments])
        self.starts = np.array([s.start_time for s in trajectory.segments])
        self.ends = np.array([s.end_time for s in trajectory.segments])
        self.durations = np.array([s.primitive.duration for s in trajectory.segments])
        self.headings = _normalize(np.stack([s.heading_orientation for s in trajectory.segments]))

    def _evaluate(self, times):
        indices = np.minimum(np.searchsorted(self.ends, times, side="right"), len(self.ends)-1)
        t = np.clip(times-self.starts[indices], 0., self.durations[indices])[:, None]
        c = self.coefficients[indices]
        p = c[:, :, 0]+c[:, :, 1]*t+c[:, :, 2]*t**2+c[:, :, 3]*t**3+c[:, :, 4]*t**4+c[:, :, 5]*t**5
        v = c[:, :, 1]+2.*c[:, :, 2]*t+3.*c[:, :, 3]*t**2+4.*c[:, :, 4]*t**3+5.*c[:, :, 5]*t**4
        a = 2.*c[:, :, 2]+6.*c[:, :, 3]*t+12.*c[:, :, 4]*t**2+20.*c[:, :, 5]*t**3
        finished = times >= self.trajectory.duration
        p[finished], v[finished], a[finished] = (self.trajectory.final_position,
            self.trajectory.final_velocity, self.trajectory.final_acceleration)
        return p, v, a, self.headings[indices]

    def window(self, elapsed, *, dt=.1, horizon=21):
        if not np.isfinite(elapsed) or elapsed < 0. or not np.isfinite(dt) or dt <= 0. or horizon < 1:
            raise ValueError("elapsed must be non-negative and dt/horizon positive")
        times = float(elapsed)+np.arange(horizon)*float(dt)
        # Include the original 1ms forward samples for planner body-rate feed-forward.
        p, v, a, heading = self._evaluate(np.r_[times, times+self.config.body_rate_difference_dt])
        q = _attitudes(a, heading, self.config.gravity)
        rates = _body_rates(q[:horizon], q[horizon:], self.config.body_rate_difference_dt) * VECTOR_FLIP
        orientation = _normalize(q[:horizon] * QUATERNION_FLIP)
        return [MPCReference(p[i], orientation[i], v[i], a[i], rates[i]) for i in range(horizon)]


def encode(state, references):
    """Direct equivalent of project(build_features(...)), preserving sign rules.

    Layout: current q/v (7), 21 relative p/q/relative v (210), 20 thrust (20).
    Unused current position and planner body rates are not materialized in 300D.
    The measured acceleration was already absent from the existing contract.
    """
    values = list(references)
    if not isinstance(state, RawDroneState) or not values or any(not isinstance(r, MPCReference) for r in values):
        raise TypeError("expected RawDroneState and nonempty MPCReference sequence")
    values = values[:21]
    values.extend([values[-1]] * (21-len(values)))
    q = _normalize(np.stack([r.orientation for r in values]))
    if q[0, 0] < 0.:
        q[0] *= -1.
    for i in range(1, len(q)):
        if np.dot(q[i], q[i-1]) < 0.:
            q[i] *= -1.
    current = _normalize(state.quaternion)
    if np.dot(current, q[0]) < 0.:
        current = -current
    position = (np.stack([r.position for r in values])-state.position)*VECTOR_FLIP
    velocity = (np.stack([r.velocity for r in values])-state.velocity)*VECTOR_FLIP
    orientation = _normalize(q*QUATERNION_FLIP)
    reference_states = np.concatenate((position, orientation, velocity), axis=1)
    acceleration = np.stack([r.acceleration for r in values[:20]])*VECTOR_FLIP
    # Match the original FEATURE gravity (not QPConfig.gravity=9.80665).
    thrust = np.linalg.norm(acceleration+[0., 0., 9.8066], axis=1)
    feature = np.r_[_normalize(current*QUATERNION_FLIP), state.velocity*VECTOR_FLIP,
                    reference_states.ravel(), thrust].astype(np.float32)
    if feature.shape != (237,) or not np.isfinite(feature).all():
        raise ValueError("expected one finite physical 237D input")
    return feature
