"""Time-compress a frozen route, preserving geometry and reporting dynamics."""
from pathlib import Path
import json

from .decision_dataset import atomic_json
from .ip37_safety_cases import audit


def main():
    root = Path('controller_architecture_study/results')
    source = root/'lateral_screen_live_20261108_v1/case_0.7.json'
    out = root/'fast_route_screen_20261108_v2'
    out.mkdir(parents=True, exist_ok=False)
    case = json.loads(source.read_text(encoding='utf-8'))
    scale = .85
    baseline = audit(case)
    for layout in case['layouts'].values():
        for segment in layout['segments']:
            segment['start_time'] *= scale
            segment['duration'] *= scale
            segment['coefficients'] = [[v/scale**k for k, v in enumerate(row)] for row in segment['coefficients']]
        layout['duration'] *= scale
        for gate in layout['gates']:
            gate['reference_time_s'] *= scale
            for key, power in (('reference_velocity', 1), ('reference_acceleration', 2)):
                if key in gate:
                    gate[key] = [v/scale**power for v in gate[key]]
    for role in case['roles'].values():
        role['turn_gate_times_s'] = [t*scale for t in role['turn_gate_times_s']]
    case['duration_s'] = max(layout['duration'] for layout in case['layouts'].values())+2.
    case['time_scale'] = scale
    check = audit(case)
    atomic_json(out/'case.json', case)
    atomic_json(out/'audit.json', check)
    atomic_json(out/'baseline_audit.json', baseline)
    print('reference audit:', check['passed'], 'Drone1:', check['vehicles']['Drone1'], flush=True)
    if not check['passed']:
        raise RuntimeError('reference geometry audit failed')


if __name__ == '__main__':
    main()
