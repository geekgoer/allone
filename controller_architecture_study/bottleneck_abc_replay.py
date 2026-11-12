"""Paired A/B/C comparison on archived-generator seeds and fresh synthetic cases."""
import argparse
from dataclasses import replace
from hashlib import sha256
import json
from math import ceil
from pathlib import Path
from time import perf_counter_ns

from .bottleneck_abc import abc_repair
from .bottleneck_experiment import random_case
from .bottleneck_joint import heuristic, simulate


REGIMES = ('balanced', 'ps_pressure', 'pl_pressure', 'switch_pressure')


def run(output, seed_start, seeds):
    output.mkdir(parents=True, exist_ok=False)
    protocol = {
        'seed_start': seed_start, 'seeds': seeds, 'loops': [4, 6, 12],
        'regimes': REGIMES, 'engine': {'whole': [4], 'segmented': [1, 2, 4]},
        'initial_policy': ['edf', 4], 'max_checks': 5, 'max_edits': 8,
        'charged_delay': 'max(0.5 ms, method wall time), post hoc full replay',
        'scope': 'Synthetic finite-window timing; same generator as v5, no new oracle.',
        'source_sha256': {
            name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('bottleneck_abc.py', 'bottleneck_joint.py',
                         'bottleneck_experiment.py', 'bottleneck_abc_replay.py')
        },
    }
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    rows = []
    methods = {
        'v5': heuristic,
        'mode_only': lambda scenario: heuristic(scenario, allowed={'mode'}),
        'abc': abc_repair,
    }
    for n in protocol['loops']:
        for engine, groups in protocol['engine'].items():
            for regime in REGIMES:
                for seed in range(seed_start, seed_start + seeds):
                    scenario = replace(
                        random_case(seed, n, regime, full_grid=True),
                        legal_groups=tuple(groups), initial_policy=('edf', 4),
                    )
                    row = {'seed': seed, 'loops': n, 'engine': engine, 'regime': regime, 'methods': {}}
                    row['fixed'] = simulate(scenario, (0,) * n, scenario.initial_policy)['feasible']
                    # Rotate call order so one method does not always warm the replay code.
                    names = tuple(methods)
                    shift = len(rows) % len(names)
                    for name in names[shift:] + names[:shift]:
                        start = perf_counter_ns()
                        result = methods[name](scenario)
                        wall = (perf_counter_ns() - start) / 1e6
                        if result['accepted']:
                            assert simulate(scenario, result['choice'], result['policy'])['feasible']
                        charged = replace(scenario, platform=replace(
                            scenario.platform, decision=max(scenario.platform.decision, wall)))
                        row['methods'][name] = {
                            key: val for key, val in result.items() if key != 'forecast'
                        }
                        row['methods'][name].update(
                            wall_ms=wall,
                            accepted_charged=result['accepted'] and simulate(
                                charged, result['choice'], result['policy'])['feasible'],
                        )
                    rows.append(row)
            print(f'Completed {n} loops, {engine}: {len(rows)} cases', flush=True)

    def stats(cases, name):
        outcomes = [row['methods'][name] for row in cases]
        timings = sorted(row['methods'][name]['wall_ms'] for row in cases if not row['fixed'])
        return {
            'accepted': sum(r['accepted'] for r in outcomes),
            'accepted_charged': sum(r['accepted_charged'] for r in outcomes),
            'adjustment_mean_ms': sum(timings) / len(timings) if timings else None,
            'adjustment_p99_ms': timings[ceil(.99 * len(timings)) - 1] if timings else None,
        }

    summary = {}
    for engine in protocol['engine']:
        for n in protocol['loops']:
            cases = [row for row in rows if row['engine'] == engine and row['loops'] == n]
            failed = [row for row in cases if not row['fixed']]
            summary[f'{engine}_{n}'] = {
                'cases': len(cases), 'fixed': len(cases) - len(failed),
                'by_type': {
                    kind: {
                        'cases': len(group),
                        **{method: stats(group, method) for method in methods},
                    }
                    for kind in ('A', 'B', 'C')
                    if (group := [r for r in failed if r['methods']['abc']['initial_type'] == kind])
                },
                **{method: stats(cases, method) for method in methods},
            }
    (output / 'cases.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed-start', type=int, required=True)
    parser.add_argument('--seeds', type=int, default=20)
    args = parser.parse_args()
    run(args.output, args.seed_start, args.seeds)
