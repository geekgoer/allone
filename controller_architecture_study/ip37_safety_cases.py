"""Prepare natural mid-turn gate scenarios offline; no AirSim or board calls."""
import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np

from .decision_dataset import atomic_json, sha256
from .probe_airsim_three import load_layout


def evaluate(segment, t, derivative=0):
    c = np.array(segment['coefficients'], dtype=float)
    for _ in range(derivative):
        c = c[:, 1:]*np.arange(1, c.shape[1])
    return c @ np.power(t, np.arange(c.shape[1]))


def time_gates(layout):
    """Attach reference crossing times to the saved endpoint gates."""
    for gate in layout['gates']:
        matches = [s['start_time']+s['duration'] for s in layout['segments']
                   if np.linalg.norm(evaluate(s, s['duration'])-gate['center']) < 1e-6]
        if len(matches) != 1:
            raise ValueError('source gate must match exactly one reference endpoint')
        gate.update(reference_time_s=matches[0], kind='endpoint')


def insert_straight(layout, seconds, boundary):
    """Insert steady flight at a C2 boundary; translate the entire suffix."""
    if seconds == 0.:
        return
    segment = layout['segments'][boundary]
    p, v, a = (evaluate(segment, 0., d) for d in range(3))
    if seconds < 0 or np.linalg.norm(a) > 1e-7 or np.linalg.norm(v) < 1.:
        raise ValueError('positive extension at a zero-acceleration moving boundary required')
    start = segment['start_time']
    previous = layout['segments'][boundary-1]
    if any(np.linalg.norm(evaluate(previous, previous['duration'], d)-value) > 1e-6
           for d, value in enumerate((p, v, a))):
        raise ValueError('source boundary is not C2')
    shift = v*seconds
    for old in layout['segments'][boundary:]:
        c = np.array(old['coefficients'])
        c[:, 0] += shift
        old.update(start_time=old['start_time']+seconds, coefficients=c.tolist())
    c = np.zeros((3, 6))
    c[:, 0], c[:, 1] = p, v
    layout['segments'].insert(boundary, dict(start_time=start, duration=seconds,
        coefficients=c.tolist(), heading_orientation=segment['heading_orientation']))
    for gate in layout['gates']:
        if gate['reference_time_s'] > start+1e-8:
            gate['center'] = (np.array(gate['center'])+shift).tolist()
            gate['reference_time_s'] += seconds
    yaw = np.arctan2(v[1], v[0])
    q = [float(np.cos(yaw/2)), 0., 0., float(np.sin(yaw/2))]
    for t in np.arange(2., seconds+1e-8, 2.):
        layout['gates'].append(dict(center=(p+v*t).tolist(), orientation=q,
                                   reference_time_s=start+float(t), kind='straight'))
    layout['duration'] += seconds


def add_turn_gates(layout):
    """Use trajectory curvature, never a static/dynamic controller's failure trace."""
    candidates = []
    for segment in layout['segments'][1:-1]:
        points = []
        for t in np.linspace(.2, .8, 41)*segment['duration']:
            v, a = evaluate(segment, t, 1), evaluate(segment, t, 2)
            curvature = float(np.linalg.norm(np.cross(v, a))/max(np.linalg.norm(v)**3, 1e-9))
            points.append((curvature, float(t), v, a))
        curvature, t, v, a = max(points, key=lambda x: x[0])
        phase = segment['start_time']+t
        if curvature < .02 or min(abs(phase-g['reference_time_s']) for g in layout['gates']) < .3:
            continue
        candidates.append((curvature, phase, evaluate(segment, t), v, a))
    if len(candidates) < 2:
        raise ValueError('two distinct curved segments are required')
    for curvature, phase, p, v, a in sorted(candidates, key=lambda x: -x[0])[:2]:
        yaw = float(np.arctan2(v[1], v[0]))
        layout['gates'].append(dict(center=p.tolist(), orientation=[float(np.cos(yaw/2)), 0., 0., float(np.sin(yaw/2))],
            reference_time_s=phase, kind='turn', curvature_per_m=curvature,
            reference_velocity=v.tolist(), reference_acceleration=a.tolist()))


def make_case(seed=0, net_width=2.1, *, vehicles=6):
    if seed not in (0, 1, 2) or net_width < 1.5 or not np.isfinite(net_width) or not 1 <= vehicles <= 12:
        raise ValueError('seed 0..2 and finite net width >=1.5m required')
    layouts, sources, roles = {}, {}, {}
    for i in range(vehicles):
        vehicle = f'Drone{i+1}'
        # Keep the old six-loop case byte-compatible in meaning. New groups
        # mix early/middle/late turns with smooth loops, including at N=4.
        hard = i < 3 if vehicles <= 6 else i % 6 < 3
        source = (49, 2) if hard else (50, 4)
        original, digest = load_layout(*source)
        layout = deepcopy(original)
        time_gates(layout)
        prefix = 6.*((i+seed) % 3) if hard else 6.
        insert_straight(layout, prefix, 1)
        insert_straight(layout, 12.-prefix, len(layout['segments'])-1)
        if hard:
            add_turn_gates(layout)
        layout['gates'].sort(key=lambda g: g['reference_time_s'])
        for number, gate in enumerate(layout['gates'], 1):
            gate['index'] = number
        layouts[vehicle] = layout
        sources[vehicle] = dict(environment=source[0], route=source[1], source_sha256=digest)
        roles[vehicle] = dict(role=('early', 'middle', 'late')[int(prefix/6)] if hard else 'smooth',
                              turn_gate_times_s=[g['reference_time_s'] for g in layout['gates'] if g['kind'] == 'turn'])
    return dict(version='ip37_natural_turn_gates_v1', seed=seed, layouts=layouts, sources=sources,
                roles=roles, net_width_m=net_width, gate_center_width_m=net_width+.15,
                gate_center_height_m=4., duration_s=float(np.ceil(max(l['duration'] for l in layouts.values())+2.)),
                external_disturbance=False, state_writes_after_start=False,
                live_flight_verified=False, generator_sha256=sha256(Path(__file__)))


def audit(case):
    """Finite reference probes, not a continuous safety or controller certificate."""
    from environment import Gate
    from .held_control_safety import GateHoldAnalyzer, HoldBounds

    result = {}
    bounds = HoldBounds(2., .01, .05, .15, 10., 'offline geometry only')
    for name, layout in case['layouts'].items():
        segments = layout['segments']
        continuity = [float(np.linalg.norm(evaluate(a, a['duration'], d)-evaluate(b, 0., d)))
                      for a, b in zip(segments, segments[1:]) for d in range(3)]
        clock_error = max(abs(a['start_time']+a['duration']-b['start_time'])
                          for a, b in zip(segments, segments[1:]))
        gates = [Gate(g['index'], g['center'], g['orientation']) for g in layout['gates']]
        analyzer = GateHoldAnalyzer(gates, case['net_width_m'], bounds,
            net_height=case['gate_center_height_m']-.15, body_radius=.5, clearance=.025)
        margin, speed, acceleration = [], [], []
        for s in segments:
            for t in np.linspace(0., s['duration'], int(np.ceil(s['duration']/.01))+1):
                p, v, a = (evaluate(s, t, d) for d in range(3))
                margin.append(float((analyzer.normals@p-analyzer.offsets).max(axis=1).min()))
                speed.append(float(np.linalg.norm(v)))
                acceleration.append(float(np.linalg.norm(a)))
        result[name] = dict(max_C2_error=max(continuity), max_clock_gap_s=clock_error,
            min_sampled_reference_margin_m=min(margin), max_sampled_speed_mps=max(speed),
            max_sampled_acceleration_mps2=max(acceleration), gates=len(gates),
            role=case['roles'][name])
    return dict(passed=all(r['max_C2_error'] < 1e-6 and r['max_clock_gap_s'] < 1e-8
                          and r['min_sampled_reference_margin_m'] > 0. for r in result.values()),
                scope='reference sampled at <=10ms, inflated finite frame boxes; not flight validation', vehicles=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=(0, 1, 2), default=0)
    parser.add_argument('--net-width', type=float, default=2.1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    case = make_case(args.seed, args.net_width)
    report = audit(case)
    atomic_json(args.output/'case.json', case)
    atomic_json(args.output/'geometry_audit.json', report)
    print(f"geometry_passed={report['passed']}; real flight NOT started; {args.output}")
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
