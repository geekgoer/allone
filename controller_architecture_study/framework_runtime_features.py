"""Same frozen237D contract without unused body rates or21 reference objects."""

from types import SimpleNamespace

import numpy as np

from .fast_convex_preprocess import _attitudes, _normalize, VECTOR_FLIP, QUATERNION_FLIP


def reference_arrays(window, elapsed, horizon):
    times = float(elapsed)+np.arange(horizon)*.1
    p,v,a,heading = window._evaluate(times)
    q = _normalize(_attitudes(a,heading,window.config.gravity)*QUATERNION_FLIP)
    # MPCReference normalizes once more when the original window builds it.
    q = _normalize(q)
    return p,v,a,q


def reference_point(window, elapsed):
    p,v,a,q = reference_arrays(window,elapsed,1)
    return SimpleNamespace(position=p[0],velocity=v[0],acceleration=a[0],orientation=q[0])


def risk_reference(window, elapsed):
    p,v,a,_=window._evaluate(np.array([float(elapsed)]))
    if np.linalg.norm(v[0,:2])<.5:
        return reference_point(window,elapsed)
    return SimpleNamespace(position=p[0],velocity=v[0],acceleration=a[0])


def encode_direct(window, state, elapsed):
    p,v,a,q = reference_arrays(window,elapsed,21)
    q = _normalize(q)  # the original encoder normalizes the reference stack
    if q[0,0] < 0.:
        q[0] *= -1.
    for i in range(1,len(q)):
        if q[i]@q[i-1] < 0.:
            q[i] *= -1.
    current = _normalize(state.quaternion)
    if current@q[0] < 0.:
        current = -current
    ref = np.concatenate(((p-state.position)*VECTOR_FLIP,_normalize(q*QUATERNION_FLIP),
        (v-state.velocity)*VECTOR_FLIP),axis=1)
    thrust = np.linalg.norm(a[:20]*VECTOR_FLIP+[0.,0.,9.8066],axis=1)
    result = np.r_[_normalize(current*QUATERNION_FLIP),state.velocity*VECTOR_FLIP,ref.ravel(),thrust].astype(np.float32)
    if result.shape != (237,) or not np.isfinite(result).all():
        raise ValueError("one finite physical237D input required")
    return result
