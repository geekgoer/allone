"""Read-only decomposition of logged delays and online rejection snapshots."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
from .decision_dataset import atomic_json


def summary(values):
    return {'n': len(values), 'median': float(np.median(values)), 'p99': float(np.percentile(values, 99)),
            'max': float(max(values))} if values else {'n': 0}


def main():
    root = Path('controller_architecture_study/results')
    out = root/'online_rejection_audit_20261108_v1'
    out.mkdir(exist_ok=False)
    all_runs = []
    pairs = {'sample_to_state': ('sample_start_s', 'state_ready_s'),
             'state_to_collision': ('state_ready_s', 'collision_ready_s'),
             'collision_to_feature': ('collision_ready_s', 'feature_ready_s'),
             'command_rpc': ('command_start_s', 'command_return_s'),
             'paired_state_rpc': ('paired_sample_start_s', 'paired_state_ready_s'),
             'paired_state_to_finish': ('paired_state_ready_s', 'finished_s')}
    for name in ('r0', 'r1'):
        p = root/'two_loop_adaptive_20261108_v1'/name
        rows = json.loads((p/'requests.json').read_text(encoding='utf-8'))
        decisions = json.loads((p/'decisions.json').read_text(encoding='utf-8'))
        rejected = [x for x in decisions if x['status'] == 'no_feasible_repair']
        decomposition = {}
        for label, (a, b) in pairs.items():
            values = [1000*(r[b]-r[a]) for r in rows if a in r and b in r]
            decomposition[label] = summary(values)
        extremes = {}
        for label, a, b in [('prep', 'sample_start_s', 'feature_ready_s'),
                            ('post', 'command_start_s', 'finished_s')]:
            available = [r for r in rows if a in r and b in r]
            x = max(available, key=lambda r: r[b]-r[a])
            extremes[label] = {'vehicle': x['vehicle'], 'cycle': x['cycle'], 'status': x['status'],
                'elapsed_ms': 1000*(x[b]-x[a]), 'prep_cpu_ms': x.get('prep_thread_cpu_ms'),
                'command_cpu_ms': x.get('command_thread_cpu_ms'), 'paired_cpu_ms': x.get('paired_thread_cpu_ms'),
                'parts_ms': {key: 1000*(x[end]-x[start]) for key, (start, end) in pairs.items() if start in x and end in x}}
        item = {'run': name, 'stops': dict(Counter(x['stop_reason'] for x in rejected)),
                'initial_type': dict(Counter(x['initial_type'] for x in rejected)),
                'state_age_ms': summary([max(x['state_ages_ms'].values()) for x in rejected if x['state_ages_ms']]),
                'negative_deadline_context': sum(any(loop['deadline'] <= 0 for loop in x['loop_context']) for x in rejected),
                'decomposition_ms': decomposition, 'extremes': extremes}
        all_runs.append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)
    atomic_json(out/'report.json', all_runs)


if __name__ == '__main__':
    main()
