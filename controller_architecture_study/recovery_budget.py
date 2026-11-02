"""Local gate-approach recovery budgets under explicit, unverified assumptions."""

from dataclasses import dataclass
import math

import numpy as np

from gate_route import _gate_local
from .safety_comparison_core import BODY_RADIUS, CLEARANCE


@dataclass(frozen=True)
class RecoveryAssumptions:
    braking_deceleration_mps2: tuple
    outward_acceleration_bound_mps2: tuple
    response_lag_s: float
    forward_acceleration_bound_mps2: float
    maximum_horizon_s: float
    evidence: str

    def __post_init__(self):
        if (len(self.braking_deceleration_mps2)!=2 or len(self.outward_acceleration_bound_mps2)!=2
                or not all(math.isfinite(v) and v>0. for v in self.braking_deceleration_mps2)
                or not all(math.isfinite(v) and v>=0. for v in self.outward_acceleration_bound_mps2)
                or not math.isfinite(self.response_lag_s) or self.response_lag_s<0.
                or not math.isfinite(self.forward_acceleration_bound_mps2) or self.forward_acceleration_bound_mps2<0.
                or not math.isfinite(self.maximum_horizon_s) or not 0.<self.maximum_horizon_s<=.150
                or not self.evidence):
            raise ValueError("explicit finite local recovery assumptions and evidence required")


def recovery_wait(distance, closing_speed, braking, outward_acceleration, response_lag):
    """Solve v*t+a*t²/2+(v+a*t)²/(2*b)<=d, with t=wait+lag.

    Ignoring an initially inward velocity is conservative. The lag is physical
    response AFTER command delivery, not the response time already charged in R.
    A negative margin means outside THIS local recovery set, not inevitable crash.
    """
    values=(distance,closing_speed,braking,outward_acceleration,response_lag)
    if (not all(math.isfinite(x) for x in values) or braking<=0.
            or outward_acceleration<0. or response_lag<0.):
        raise ValueError("finite recovery state, positive braking and nonnegative bounds required")
    v=max(0.,closing_speed)
    stopping=v*v/(2*braking)
    margin=distance-stopping
    if margin<0.:
        return 0.,margin,"outside_local_recovery_set"
    linear=v*(1+outward_acceleration/braking)
    quadratic=.5*outward_acceleration*(1+outward_acceleration/braking)
    if quadratic>0.:
        total=2*margin/(linear+math.sqrt(linear*linear+4*quadratic*margin)) if margin>0. else 0.
    elif linear>0.:
        total=margin/linear
    else:
        total=math.inf
    return max(0.,total-response_lag),margin,"response_lag_exhausted" if total<response_lag else "within_assumed_model"


def recovery_horizon(position, velocity, gates, net_width, assumptions):
    """Four signed faces of local approach corridors, not infinite gate walls.

    Activate a gate when its earliest longitudinal entry is within the next
    budget plus the assumed response/braking time. This replaces a fixed metre
    lookahead with a dynamics-dependent one. A corridor is a conservative domain:
    being outside it before the gate is NOT itself a physical collision.
    """
    if not isinstance(assumptions,RecoveryAssumptions):
        raise TypeError("validated RecoveryAssumptions required")
    p0,v0=np.asarray(position,dtype=float),np.asarray(velocity,dtype=float)
    if p0.shape!=(3,) or v0.shape!=(3,) or not np.isfinite([p0,v0]).all() or not math.isfinite(net_width):
        raise ValueError("finite three-dimensional position, velocity and aperture required")
    radii=(net_width/2-BODY_RADIUS-CLEARANCE,2.-BODY_RADIUS-CLEARANCE)
    if min(radii)<=0.:
        raise ValueError("positive clear aperture after body envelope required")
    cap=assumptions.maximum_horizon_s
    slab=BODY_RADIUS+CLEARANCE+.15/2
    details=[]
    for gate in gates:
        p=_gate_local(p0,gate)
        if p[0]>slab:
            continue
        v=_gate_local(gate.center+v0,gate)
        ahead=max(0.,-p[0]-slab)
        forward=max(0.,float(v[0]))
        accel=assumptions.forward_acceleration_bound_mps2
        denominator=forward+math.sqrt(forward*forward+2*accel*ahead)
        entry=2*ahead/denominator if ahead and denominator else (math.inf if ahead else 0.)
        for axis,(radius,b,a) in enumerate(zip(radii,assumptions.braking_deceleration_mps2,
                assumptions.outward_acceleration_bound_mps2),1):
            for side in (-1,1):
                closing=side*float(v[axis])
                delay=cap+assumptions.response_lag_s
                activation=delay+(max(0.,closing)+a*delay)/b
                if entry>activation:
                    continue
                distance=radius-side*float(p[axis])
                wait,margin,status=recovery_wait(distance,closing,b,a,assumptions.response_lag_s)
                applicable=not (ahead>0. and distance<0.)
                if not applicable:
                    # A turning approach can start outside a future aperture.
                    # This local half-space certificate then says nothing; it
                    # must not turn a legitimate approach into an instant D=0.
                    wait,status=cap,"outside_approach_domain"
                details.append(dict(gate=gate.index,axis="lateral" if axis==1 else "vertical",side=side,
                    distance_m=distance,closing_speed_mps=closing,recovery_margin_m=margin,
                    stopping_distance_m=max(0.,closing)**2/(2*b),
                    longitudinal_entry_lower_s=entry,activation_horizon_s=activation,
                    horizon_s=min(cap,wait),status=status,applicable=applicable))
    return min((r["horizon_s"] for r in details),default=cap),details
