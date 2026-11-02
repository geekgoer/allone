"""Control-aware ZOH safe periods, conditional on an explicit quadrotor tube.

Yang-style construction: lower-bound a barrier over the ENTIRE hold interval.
This is not a claim that the assumed error tube has been certified in AirSim.
"""

from dataclasses import dataclass, asdict
import math

import numpy as np

from gate_visualization import _gate_parts
from gate_route import _rotate_vector


@dataclass(frozen=True)
class HoldBounds:
    acceleration_error_mps2: float
    position_error_m: float
    velocity_error_mps: float
    maximum_hold_s: float
    barrier_rate: float
    evidence: str
    plant_bounds_verified: bool = False
    gravity: float = 9.80665

    def __post_init__(self):
        values = (self.acceleration_error_mps2, self.position_error_m,
                  self.velocity_error_mps, self.maximum_hold_s, self.barrier_rate, self.gravity)
        if (not all(math.isfinite(x) for x in values) or min(values[:3]) < 0.
                or min(values[3:]) <= 0. or not self.evidence):
            raise ValueError('finite explicit state/dynamics error bounds and evidence required')


def nominal_acceleration(quaternion, control, gravity=9.80665):
    """Physical NED command [thrust acceleration, right-handed body rates].

    Pass the effective command AFTER throttle clipping/calibration, not raw DNN
    output. Pitch/yaw AirSim API sign conversion belongs in control_adapter.
    """
    q, u = np.asarray(quaternion, dtype=float), np.asarray(control, dtype=float)
    if (q.shape != (4,) or u.shape != (4,) or not np.isfinite(np.r_[q, u]).all()
            or np.linalg.norm(q) < 1e-12 or u[0] < 0.):
        raise ValueError('finite nonzero quaternion and nonnegative physical thrust required')
    return np.array([0., 0., gravity]) - u[0] * _rotate_vector(q, [0., 0., 1.])


def _barrier_coefficients(distance, speed, acceleration, jerk, bounds):
    d, v, a, j = np.broadcast_arrays(distance, speed, acceleration, jerk)
    if not np.isfinite(np.r_[d.ravel(), v.ravel(), a.ravel(), j.ravel()]).all() or (j < 0.).any():
        raise ValueError('finite barrier data and nonnegative jerk bound required')
    k = bounds.barrier_rate
    h, w, z = d-bounds.position_error_m, v-bounds.velocity_error_mps, a-bounds.acceleration_error_mps2
    coefficients = np.stack((z+2*k*w+k*k*h, -j+2*k*z+k*k*w,
                             -.5*2*k*j+.5*k*k*z, -k*k*j/6.), axis=-1)
    # Reserve a small explicit numerical margin; never round the root upward.
    coefficients[..., 0] -= 1e-10
    coefficients[..., 1:] = np.minimum(coefficients[..., 1:], 0.)
    valid = (h > 0.) & (w+k*h >= 0.) & (coefficients[..., 0] > 0.)
    return coefficients, valid


def barrier_periods(distance, speed, acceleration, jerk, bounds):
    """Vectorized sufficient HOCBF bound for h = n*p-b, unit n.

    h'' >= a0-eps-J*t; h' and h have its integrated lower bounds.
    psi2 = h''+2*k*h'+k²*h >= q(t). Require h(0)>=0 and
    psi1(0)=h'(0)+k*h(0)>=0. Negative polynomial coefficients give a
    monotone minorant of q, hence bisection certifies the complete prefix,
    without an unsafe sampled-time or endpoint-only positivity check.
    """
    coefficients, valid = _barrier_coefficients(distance, speed, acceleration, jerk, bounds)
    h = coefficients[..., 0]
    low, high = np.zeros_like(h), np.full_like(h, bounds.maximum_hold_s)
    c0, c1, c2, c3 = np.moveaxis(coefficients, -1, 0)
    for _ in range(24):
        t = (low+high)/2
        good = ((c3*t+c2)*t+c1)*t+c0 >= 0.
        low, high = np.where(good, t, low), np.where(good, high, t)
    return np.where(valid, low, 0.)


class GateHoldAnalyzer:
    """Inflated finite frame boxes; no infinite aperture walls or 2 s preview.

    Each box is avoided by at least one exterior separating plane throughout
    the hold: max over its six plane periods, then min over all boxes. The
    max is an OR for one obstacle, not a relaxation of simultaneous barriers.
    Geometry is exactly the same _gate_parts used by the visual frame helper.
    """
    def __init__(self, gates, net_width, bounds, *, net_height=4., thickness=.15,
                 body_radius=.50, clearance=.025):
        if (not isinstance(bounds, HoldBounds) or min(net_width, net_height, thickness) <= 0.
                or body_radius < 0. or clearance < 0.):
            raise ValueError('positive geometry and explicit HoldBounds required')
        normals, offsets, labels = [], [], []
        for gate in gates:
            axes = np.array([_rotate_vector(gate.orientation, axis) for axis in np.eye(3)])
            for part, center, dimensions in _gate_parts(gate, net_width+thickness, net_height+thickness, thickness):
                ns = np.concatenate((-axes, axes))
                half = np.asarray(dimensions)/2 + body_radius + clearance
                normals.append(ns)
                offsets.append(ns @ center + np.r_[half, half])
                labels.append(f'{gate.index}:{part}')
        if not labels:
            raise ValueError('at least one finite obstacle required')
        self.normals, self.offsets = np.array(normals), np.array(offsets)
        self.labels, self.bounds = labels, bounds

    def analyze(self, state, control, *, details=False):
        p, v = np.asarray(state.position), np.asarray(state.velocity)
        if p.shape != (3,) or v.shape != (3,) or not np.isfinite(np.r_[p, v]).all():
            raise ValueError('finite physical position and velocity required')
        a = nominal_acceleration(state.quaternion, control, self.bounds.gravity)
        # For qdot = .5*q*[0,omega] and fixed F, |a_nom_dot| <= F*|omega|.
        jerk = float(control[0] * np.linalg.norm(control[1:]))
        distances = self.normals @ p - self.offsets
        speeds, accelerations = self.normals @ v, self.normals @ a
        c, valid = _barrier_coefficients(distances, speeds, accelerations, jerk, self.bounds)
        cap = self.bounds.maximum_hold_s
        full_faces = valid & ((((c[..., 3]*cap+c[..., 2])*cap+c[..., 1])*cap+c[..., 0]) >= 0.)
        # This endpoint test is valid ONLY because q's minorant is monotone.
        # One full-period separating face suffices for each finite obstacle.
        full_boxes = full_faces.any(axis=1)
        per_box = np.full(len(self.labels), cap)
        face = full_faces.argmax(axis=1)
        if not full_boxes.all():
            mask = ~full_boxes
            periods = barrier_periods(distances[mask], speeds[mask], accelerations[mask], jerk, self.bounds)
            per_box[mask], face[mask] = periods.max(axis=1), periods.argmax(axis=1)
        binding = int(np.argmin(per_box))
        horizon = float(per_box[binding])
        result = dict(budget_s=horizon, valid=horizon > 0., binding_obstacle=self.labels[binding],
                      uses_control_in_bound=True, conditional_barrier_certificate=horizon > 0.,
                      plant_bounds_verified=self.bounds.plant_bounds_verified,
                      is_safety_certificate=bool(horizon > 0. and self.bounds.plant_bounds_verified),
                      method='held-control robust HOCBF polynomial minorant v1',
                      zero_budget_means='no certificate under these assumptions; not proof of collision')
        if details:
            result.update(assumptions=asdict(self.bounds), per_obstacle_period_s=per_box.tolist(),
                          separating_face=face.tolist(), nominal_acceleration=a.tolist())
        return result


class GateHoldBatch:
    """Same per-obstacle proof, vectorized across independent control loops."""
    def __init__(self,analyzers):
        self.names=list(analyzers);self.analyzers=analyzers
        self.bounds=next(iter(analyzers.values())).bounds
        if any(a.bounds!=self.bounds for a in analyzers.values()):raise ValueError('identical explicit bounds required')
        width=max(len(a.labels) for a in analyzers.values())
        self.normals=np.zeros((len(analyzers),width,6,3))
        # Padded obstacles are always separated and never constrain a real box.
        self.offsets=np.full((len(analyzers),width,6),-1e12)
        for i,a in enumerate(analyzers.values()):
            self.normals[i,:len(a.labels)]=a.normals;self.offsets[i,:len(a.labels)]=a.offsets

    def analyze(self,states,controls):
        names=[v for v in self.names if v in states]
        if not names:return {}
        ids=[self.names.index(v) for v in names]
        p=np.array([states[v].position for v in names]);v=np.array([states[n].velocity for n in names])
        q=np.array([states[n].quaternion for n in names]);u=np.array([controls[n] for n in names])
        length=np.linalg.norm(q,axis=1)
        if not np.isfinite(np.r_[p.ravel(),v.ravel(),q.ravel(),u.ravel()]).all() or (length<1e-12).any() or (u[:,0]<0).any():
            raise ValueError('finite paired states and effective nonnegative thrust required')
        w,x,y,z=(q/length[:,None]).T
        body_z=np.c_[2*(x*z+w*y),2*(y*z-w*x),1-2*(x*x+y*y)]
        acceleration=np.array([0.,0.,self.bounds.gravity])-u[:,:1]*body_z
        jerk=(u[:,0]*np.linalg.norm(u[:,1:],axis=1))[:,None,None]
        normals=self.normals[ids]
        d=np.einsum('nijk,nk->nij',normals,p)-self.offsets[ids]
        speed=np.einsum('nijk,nk->nij',normals,v)
        a=np.einsum('nijk,nk->nij',normals,acceleration)
        c,valid=_barrier_coefficients(d,speed,a,jerk,self.bounds);cap=self.bounds.maximum_hold_s
        full=valid & ((((c[...,3]*cap+c[...,2])*cap+c[...,1])*cap+c[...,0])>=0.)
        periods=np.full(d.shape[:2],cap);mask=~full.any(axis=2)
        if mask.any():
            j=np.broadcast_to(jerk,d.shape)[mask]
            periods[mask]=barrier_periods(d[mask],speed[mask],a[mask],j,self.bounds).max(axis=1)
        binding=periods.argmin(axis=1)
        face=d.argmax(axis=2);mesh=d.max(axis=2);nearest=mesh.argmin(axis=1)
        result={}
        for i,name in enumerate(names):
            horizon=float(periods[i,binding[i]])
            result[name]=dict(budget_s=horizon,valid=horizon>0.,
                binding_obstacle=self.analyzers[name].labels[binding[i]],uses_control_in_bound=True,
                conditional_barrier_certificate=horizon>0.,plant_bounds_verified=self.bounds.plant_bounds_verified,
                is_safety_certificate=bool(horizon>0. and self.bounds.plant_bounds_verified),
                method='held-control robust HOCBF polynomial minorant v1',
                zero_budget_means='no certificate under these assumptions; not proof of collision',
                mesh_margin_m=float(mesh[i,nearest[i]]),mesh_rate_mps=float(speed[i,nearest[i],face[i,nearest[i]]]))
        return result


def safe_deadline(sample_s, now_s, actuation_s, analysis, maximum_update_s, actuation_time_error_s=0.):
    """All durations share the state observation's clock; no R counted twice.

    A fresh sample AFTER actuation starts a remaining-hold certificate. It
    cannot extend the separate maximum actuation-to-actuation quality limit.
    Subtract a known actuation timestamp uncertainty; an RPC ack is not exact.
    """
    values = (sample_s, now_s, actuation_s, maximum_update_s, actuation_time_error_s)
    if (not all(math.isfinite(x) for x in values) or actuation_s > sample_s or sample_s > now_s
            or maximum_update_s <= 0. or actuation_time_error_s < 0.):
        raise ValueError('ordered state/actuation/analysis clocks and explicit uncertainty required')
    deadline = min(sample_s+analysis['budget_s'], actuation_s+maximum_update_s-actuation_time_error_s)
    return dict(deadline_s=deadline, remaining_s=max(0., deadline-now_s), feasible=deadline > now_s,
                physical_deadline_s=sample_s+analysis['budget_s'], state_age_s=now_s-sample_s,
                is_safety_certificate=analysis['is_safety_certificate'])
