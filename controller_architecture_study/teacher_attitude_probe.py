"""Isolated attitude diagnostics; the frozen label teacher is never edited."""
import argparse
from dataclasses import replace
from pathlib import Path
import time

import numpy as np

from airsim_interface import RawDroneState
from falcon_paper_fidelity.motion_primitive import _rotation_matrix, _zup_flu_attitude
from state_adapter import normalize_quaternion, quaternion_inverse, quaternion_multiply, quaternion_to_rotation_vector
from .convex_teacher import ConvexTeacher, QPConfig
from .decision_dataset import atomic_json, sha256
from .dynamic_probe_common import describe, read_json
from .fast_convex_preprocess import FastReferenceWindow
from .fly_decision_dnn import StoredRoutePlanner


VARIANTS = {'baseline': (3.5, False), 'gain5': (5., False), 'gain7': (7., False),
            'qp_ff': (3.5, True), 'qp_ff_gain7': (7., True)}


class _CaptureSolution:
    """Observe the same public solve result without a second QP solve."""
    def __init__(self, solver):
        self.inner, self.result = solver, None

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def solve(self, **kwargs):
        self.result = self.inner.solve(**kwargs)
        return self.result


def desired_attitude(acceleration, heading, gravity):
    return normalize_quaternion(_zup_flu_attitude(acceleration, heading, gravity)*[1., 1., -1., -1.])


def qp_feedforward(current, desired, following, dt):
    """Predicted desired-body rate, transported into the current body frame."""
    desired_rate = quaternion_to_rotation_vector(
        quaternion_multiply(quaternion_inverse(desired), following))/dt
    error = quaternion_multiply(quaternion_inverse(current), desired)
    return _rotation_matrix(error)@desired_rate


class DiagnosticTeacher(ConvexTeacher):
    def __init__(self, variant='baseline', *, attitude_gain=None):
        gain, self.use_feedforward = VARIANTS[variant]
        if attitude_gain is not None:
            gain = float(attitude_gain)
            if not np.isfinite(gain) or gain <= 0.:
                raise ValueError('diagnostic attitude gain must be positive and finite')
        self.variant = variant
        super().__init__(replace(QPConfig(), attitude_gain=gain))
        self.solver = _CaptureSolution(self.solver)

    def signature(self):
        return {**super().signature(), 'diagnostic_variant': self.variant,
                'feedforward': 'optimized_q0_to_q1_current_body_frame' if self.use_feedforward else 'none',
                'qp_solves_per_query': 1, 'production_teacher_replaced': False,
                'source_sha256': {name: sha256(Path(__file__).with_name(name))
                                  for name in ('convex_teacher.py', 'teacher_attitude_probe.py')}}

    def query(self, state, references):
        start = time.perf_counter()
        action, info = super().query(state, references)
        base_ms = info['solve_ms']
        acceleration = self.solver.result.x[:6].reshape(2, 3)
        q0 = desired_attitude(acceleration[0], references[0].orientation, self.settings.gravity)
        q1 = desired_attitude(acceleration[1], references[1].orientation, self.settings.gravity)
        error = quaternion_to_rotation_vector(quaternion_multiply(quaternion_inverse(state.quaternion), q0))
        feedback = self.settings.attitude_gain*error
        ff = qp_feedforward(state.quaternion, q0, q1, self.settings.prediction_dt) if self.use_feedforward else np.zeros(3)
        rates = feedback+ff
        clipped = np.clip(rates, [-3., -3., -2.], [3., 3., 2.])
        if self.use_feedforward:
            action = np.r_[action[0], clipped].astype(np.float32)
        info.update(diagnostic_variant=self.variant, next_acceleration=acceleration[1].copy(),
                    desired_quaternion=q0, next_desired_quaternion=q1,
                    attitude_error_rad=error, feedback_rates=feedback, feedforward_rates=ff,
                    preclip_rates=rates, rate_clipped=bool(np.any(clipped != rates)),
                    base_query_ms=base_ms, solve_ms=(time.perf_counter()-start)*1000)
        return action, info


def integrate_rotation(q, rates, dt):
    speed = np.linalg.norm(rates)
    delta = np.r_[np.cos(speed*dt/2), rates*np.sin(speed*dt/2)/speed] if speed > 1e-12 else np.array([1., 0., 0., 0.])
    return normalize_quaternion(quaternion_multiply(q, delta))


def simulate(case, variant, *, period=.03, delay=.007, step=.005, attitude_gain=None):
    """Perfect body-rate response; no motor/drag/quantization/jitter model."""
    planner = StoredRoutePlanner(case['layouts']['Drone1'])
    window = FastReferenceWindow(planner.trajectory, planner.config)
    teacher = DiagnosticTeacher(variant, attitude_gain=attitude_gain)
    first = window.window(0.)[0]
    p, v, q = first.position.copy(), first.velocity.copy(), first.orientation.copy()
    held = np.array([teacher.settings.gravity, 0., 0., 0.])
    duration = case['duration_s']
    events = [(float(t), 2, 0) for t in np.arange(0., duration+step/2, step)]
    for i, t in enumerate(np.arange(0., duration-delay, period)):
        events.extend(((float(t), 0, i), (float(t+delay), 1, i)))
    previous, pending, trace, diagnostics = 0., {}, [], []
    for t, kind, i in sorted(events):
        dt = t-previous
        midpoint = integrate_rotation(q, held[1:], dt/2)
        acceleration = np.array([0., 0., teacher.settings.gravity])-held[0]*_rotation_matrix(midpoint)[:, 2]
        p, v = p+v*dt+.5*acceleration*dt**2, v+acceleration*dt
        q, previous = integrate_rotation(q, held[1:], dt), t
        if kind == 0:
            state = RawDroneState(p, q, v, acceleration, held[1:])
            pending[i], info = teacher.query(state, window.window(t))
            diagnostics.append(info)
        elif kind == 1:
            held = pending.pop(i)
        else:
            trace.append(dict(t=t, position=p.tolist(), velocity=v.tolist(), quaternion=q.tolist(),
                              error_m=float(np.linalg.norm(p-window.window(t)[0].position))))
    positions = np.array([r['position'] for r in trace])
    crossings = []
    for gate in case['layouts']['Drone1']['gates']:
        local = (positions-gate['center'])@_rotation_matrix(gate['orientation'])
        indices = np.flatnonzero((local[:-1, 0]<0)&(local[1:, 0]>=0))
        row = dict(gate=gate['index'], crossed=bool(len(indices)))
        if len(indices):
            k = int(indices[0])
            ratio = -local[k, 0]/(local[k+1, 0]-local[k, 0])
            crossing = local[k]*(1.-ratio)+local[k+1]*ratio
            row.update(lateral_offset_m=float(crossing[1]), vertical_offset_m=float(crossing[2]),
                       body_margin_m=float(min(case['net_width_m']/2-.525-abs(crossing[1]),
                                               1.4-abs(crossing[2]))))
        crossings.append(row)
    report = dict(variant=variant, signature=teacher.signature(), period_ms=period*1000,
        imposed_delay_ms=delay*1000, integration_step_s=step,
        scope='offline ideal body-rate model; no AirSim, board or safety certificate',
        crossings=crossings, position_rmse_m=float(np.sqrt(np.mean([r['error_m']**2 for r in trace]))),
        query_ms=describe([r['solve_ms'] for r in diagnostics]),
        rate_clipped_fraction=float(np.mean([r['rate_clipped'] for r in diagnostics])))
    return report, trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    case = read_json(args.case)
    report = dict(case_sha256=sha256(args.case), variants=[])
    for variant in VARIANTS:
        item, trace = simulate(case, variant)
        report['variants'].append(item)
        atomic_json(args.output/f'{variant}_trace.json', trace)
        atomic_json(args.output/'report.json', report)
        print(variant, 'gate4', item['crossings'][3], 'query_ms', item['query_ms'], flush=True)
    print(f'Offline report: {args.output / "report.json"}')


if __name__ == '__main__':
    main()
