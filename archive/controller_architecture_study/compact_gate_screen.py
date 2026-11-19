"""Inspect timing traces and add one intermediate gate to a frozen route."""
import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from .decision_dataset import atomic_json
from .ip37_safety_cases import audit, evaluate


def main():
    root = Path('controller_architecture_study/results')
    out = root/'compact_gate_screen_20261108_v1'
    out.mkdir(parents=True, exist_ok=False)
    source = root/'lateral_screen_live_20261108_v1/case_0.7.json'
    case = json.loads(source.read_text(encoding='utf-8'))
    metrics = []
    for p in sorted((root/'timing_screen_20261108_v1').glob('*/result.json')):
        trace = json.loads((p.parent/'evaluation_Drone1.json').read_text(encoding='utf-8'))
        jobs = json.loads((p.parent/'requests.json').read_text(encoding='utf-8'))
        errors = [np.linalg.norm(np.asarray(r['position'])-r['reference_position']) for r in trace]
        acks = sorted(r['command_return_s'] for r in jobs if 'command_return_s' in r)
        metrics.append(dict(case=p.parent.name, max_tracking_error_m=max(errors),
            max_error_time_s=trace[int(np.argmax(errors))]['t'],
            mean_command_interval_ms=float(np.mean(np.diff(acks))*1000),
            max_command_interval_ms=float(np.max(np.diff(acks))*1000)))
    atomic_json(out/'transient_metrics.json', metrics)
    dense = deepcopy(case)
    layout = dense['layouts']['Drone1']
    gates = sorted(layout['gates'], key=lambda g: g['reference_time_s'])
    pairs = [(b['reference_time_s']-a['reference_time_s'], a, b) for a, b in zip(gates, gates[1:])
             if 3. <= a['reference_time_s'] and b['reference_time_s'] <= 12.]
    _, a, b = max(pairs, key=lambda x: x[0])
    t = (a['reference_time_s']+b['reference_time_s'])/2
    seg = next(s for s in layout['segments'] if s['start_time'] <= t <= s['start_time']+s['duration'])
    local = t-seg['start_time']
    pos, vel = evaluate(seg, local), evaluate(seg, local, 1)
    yaw = float(np.arctan2(vel[1], vel[0]))
    layout['gates'].append(dict(center=pos.tolist(), orientation=[float(np.cos(yaw/2)), 0., 0., float(np.sin(yaw/2))], reference_time_s=t, kind='extra_midpoint'))
    layout['gates'].sort(key=lambda g: g['reference_time_s'])
    for i, gate in enumerate(layout['gates'], 1):
        gate['index'] = i
    check = audit(dense)
    atomic_json(out/'case.json', dense)
    atomic_json(out/'audit.json', check)
    print('metrics:', metrics, flush=True)
    print('added gate time:', t, 'neighbor times:', a['reference_time_s'], b['reference_time_s'], 'audit:', check['passed'], flush=True)
    if not check['passed']:
        raise RuntimeError('reference geometry audit failed')


if __name__ == '__main__':
    main()
