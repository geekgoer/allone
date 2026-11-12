"""Keep the urgent route fast and restore the calm route's original timing."""
import json
from pathlib import Path
from .decision_dataset import atomic_json
from .ip37_safety_cases import audit

root = Path('controller_architecture_study/results')
fast = json.loads((root/'fast_route_screen_20261108_v2/case.json').read_text(encoding='utf-8'))
slow = json.loads((root/'lateral_screen_live_20261108_v1/case_0.7.json').read_text(encoding='utf-8'))
fast['layouts']['Drone4'] = slow['layouts']['Drone4']
fast['roles']['Drone4'] = slow['roles']['Drone4']
fast['duration_s'] = max(x['duration'] for x in fast['layouts'].values())+2.
fast['mixed_speed_note'] = 'Drone1 fast 0.85 timing; Drone4 original 0.70 lateral route timing'
out = root/'two_loop_mixed_speed_20261108_v1'
out.mkdir(parents=True, exist_ok=False)
check = audit(fast)
atomic_json(out/'case.json', fast)
atomic_json(out/'audit.json', check)
if not check['passed']:
    raise RuntimeError('reference geometry audit failed')
print('reference audit passed')
