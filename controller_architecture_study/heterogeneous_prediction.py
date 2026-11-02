"""Anticipate saved-route bends and retain recovery until tracking settles."""
import numpy as np


def anticipatory_mode(window,elapsed,observation,net_width,memory):
    from .safety_comparison_core import BODY_RADIUS,CLEARANCE
    offsets=np.arange(21)*.1
    p,v,_,_=window._evaluate(float(elapsed)+offsets)
    speed=np.linalg.norm(v[0,:2])
    normal=np.r_[-v[0,1]/speed,v[0,0]/speed,0.] if speed>=.5 else np.array([0.,1.,0.])
    departure=p-p[0]-offsets[:,None]*v[0]
    lateral=float(np.max(abs(departure@normal)))
    vertical=float(np.max(abs(departure[:,2])))
    radii=(net_width/2-BODY_RADIUS-CLEARANCE,2.-BODY_RADIUS-CLEARANCE)
    risk=observation['risk'];radius=radii[0 if risk['boundary']=='lateral' else 1]
    upcoming=lateral>.3*radii[0] or vertical>.3*radii[1]
    urgent=risk['h']<.3*radius or risk['h']+.5*min(0.,risk['hdot'])<.1*radius
    settled=(risk['h']>=.5*radius and abs(risk['hdot'])<=.2*radius
        and observation['position_error_m']<=min(.3,.5*radius) and observation['velocity_error_mps']<=.6)
    if upcoming or urgent:
        memory.update(active=True,quiet_since=None)
    elif memory.get('active',False):
        if not settled:memory['quiet_since']=None
        elif memory.get('quiet_since') is None:memory['quiet_since']=float(elapsed)
        elif elapsed-memory['quiet_since']>=.5:memory['active']=False
    return memory.get('active',False),dict(horizon_s=2.,future_lateral_departure_m=lateral,
        future_vertical_departure_m=vertical,upcoming=upcoming,urgent=urgent,settled=settled,
        quiet_since=memory.get('quiet_since'),recovery_active=memory.get('active',False))
