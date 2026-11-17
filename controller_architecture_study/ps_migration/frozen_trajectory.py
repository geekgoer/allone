"""Load already-frozen polynomial coefficients; no path fitting or MPC."""
from types import SimpleNamespace
import numpy as np
from .features import FastReferenceWindow


def from_layout(layout):
    segments = []
    expected_start = 0.
    for item in layout['segments']:
        c = np.asarray(item['coefficients'], dtype=float)
        duration = float(item['duration'])
        start = float(item['start_time'])
        heading = np.asarray(item['heading_orientation'], dtype=float)
        if (c.shape != (3, 6) or heading.shape != (4,) or not np.isfinite(c).all()
                or not np.isfinite(heading).all() or np.linalg.norm(heading) < 1e-12
                or not np.isfinite(duration) or duration <= 0. or not np.isfinite(start)
                or not np.isclose(start, expected_start, rtol=0., atol=1e-10)):
            raise ValueError('Invalid frozen polynomial segment')
        expected_start = start+duration
        segments.append(SimpleNamespace(start_time=start, end_time=expected_start,
            heading_orientation=heading, primitive=SimpleNamespace(coefficients=c, duration=duration)))
    if not segments or not np.isclose(expected_start, layout['duration'], rtol=0., atol=1e-9):
        raise ValueError('Invalid frozen trajectory duration')
    c, t = segments[-1].primitive.coefficients, segments[-1].primitive.duration
    trajectory = SimpleNamespace(segments=segments, duration=expected_start,
        final_position=c@np.array([1., t, t*t, t**3, t**4, t**5]),
        final_velocity=c[:, 1:]@np.array([1., 2*t, 3*t*t, 4*t**3, 5*t**4]),
        final_acceleration=c[:, 2:]@np.array([2., 6*t, 12*t*t, 20*t**3]))
    return FastReferenceWindow(trajectory, SimpleNamespace(gravity=9.8066, body_rate_difference_dt=.001))
