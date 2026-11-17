"""Offline control-obligation replay driven by the actual requant18 delivery."""
import argparse
from dataclasses import asdict, replace
from itertools import product
import json
from pathlib import Path
from random import Random
from statistics import mean
from time import perf_counter_ns

from .obligation_scheduler import Calendar, ControlJob, SkipPermission, plan
from .requant18_profile import pareto, profile_dict, read_profile


METHODS = {
    'current_edf': dict(enable_skip=False, enable_retime=False, enable_reorder=False, enable_modes=False),
    'budget_modes': dict(enable_skip=False, enable_retime=False, enable_reorder=False),
    'lookahead_modes': dict(enable_skip=False, enable_retime=False),
    'joint_no_skip': dict(enable_skip=False),
    'joint_recovery': {},
}


def case(seed, count, modes):
    """State qualifications, hold and freshness are supplied synthetic contracts."""
    rng = Random(seed)
    jobs, permissions = [], []
    for i in range(count):
        vehicle = f'Drone{i+1}'
        critical = i < (count+1)//2
        cap = .0042 if critical else .008
        allowed = tuple(m.numeric_id for m in modes if m.selection_mse <= cap)
        feasible = [m for m in modes if m.numeric_id in allowed]
        fast = min(feasible, key=lambda m: m.service_ms).numeric_id
        current = rng.choice(feasible).numeric_id
        release = rng.uniform(2., 10.) + i*rng.uniform(0., 1.)
        gap = rng.uniform(2., 7.)
        successor_release = release+30.
        hold = successor_release+rng.uniform(2., 8.)
        prep = rng.uniform(.1, .5)
        first = ControlJob(f'{vehicle}:0', vehicle, release, release+.2, prep, .15, .3,
                           release+gap, hold, release+gap+.5, allowed, current,
                           ((max(.6, release-2.), release+gap-2.),
                            (release+1., release+gap+1.)))
        second = replace(first, name=f'{vehicle}:1', release=successor_release,
                         prep_ready=successor_release+.2, slot_deadline=successor_release+10.,
                         fresh_deadline=successor_release+8., release_options=(),
                         current_mode=fast)
        jobs.extend((first, second))
        permissions.append(SkipPermission(first.name, second.name, hold, release-.1,
                                          not critical and rng.random() < .75,
                                          'synthetic phase qualification'))
    return Calendar(tuple(jobs)), tuple(permissions)


def result_row(calendar, permissions, modes, method, decision_ms):
    started = perf_counter_ns()
    result = plan(calendar, modes, now=0., decision_ms=decision_ms,
                  permissions=permissions, **METHODS[method])
    elapsed = (perf_counter_ns()-started)/1e6
    row = dict(method=method, accepted=result['accepted'], action=result['action'],
               checks=result['checks'], candidates=result['candidates'],
               diagnosis=result['diagnosis'], elapsed_windows_ms=elapsed,
               trace=result['trace'])
    if result['accepted']:
        installed = calendar.commit(result, decision_ms)
        before = {job.name: job for job in calendar.jobs}
        row.update(skipped=result['skipped'], order=result['order'], choice=result['choice'],
                   mode_edits=sum(job.current_mode != before[job.name].current_mode for job in installed.jobs),
                   release_edits=sum(job.release != before[job.name].release for job in installed.jobs),
                   min_slack_ms=min(result['forecast']['slack'].values()),
                   retained_selection_mse=mean(modes[job.current_mode].selection_mse for job in installed.jobs),
                   retained_check_mse=mean(modes[job.current_mode].check_mse for job in installed.jobs))
        # Exercise committed recovery lifecycle in simulated actuation order.
        for name, end in sorted(result['forecast']['command_finish'].items(), key=lambda x: x[1]):
            installed = installed.delivered(name, end)
        assert not installed.pending and not installed.jobs
    return row


def mechanism_examples(models):
    """Constructed service-only example, not a flight recovery qualification."""
    modes = {m.numeric_id: m for m in models}
    def job(name, vehicle, release, deadline, model):
        return ControlJob(name, vehicle, release, release, 0., 0., 0.,
                          deadline, deadline, deadline, (model,), model)
    calendar = Calendar((
        job('protected', 'A', 0., 4., 8),
        job('donor', 'B', 2., 6., 6),
        job('recover', 'B', 30., 35., 10)))
    permission = SkipPermission('donor', 'recover', 35., 1.5, True, 'synthetic constructed hold')
    early = replace(permission, eligible=False, evidence='synthetic protected phase')
    short = replace(permission, hold_until=30.01)
    results = {}
    for label, permit in (('eligible_late', permission), ('protected_early', early),
                          ('recovery_too_tight', short)):
        result = plan(calendar, modes, now=0., decision_ms=.5, permissions=(permit,))
        results[label] = dict(accepted=result['accepted'], action=result['action'],
                              checks=result['checks'], trace=result['trace'])
    assert results['eligible_late']['accepted']
    assert not results['protected_early']['accepted']
    assert not results['recovery_too_tight']['accepted']
    return results


def run(delivery, output, seeds, decision_grid):
    if output.exists():
        raise ValueError('Use a new results directory')
    models, identity = read_profile(delivery)
    rows = []
    for factor, count, decision_ms, seed in product((1., 1.5), (4, 8), decision_grid, range(seeds)):
        scaled = tuple(replace(m, service_ms=m.service_ms*factor) for m in models)
        calendar, permissions = case(seed, count, scaled)
        modes = {m.numeric_id: m for m in scaled}
        for method in METHODS:
            row = result_row(calendar, permissions, modes, method, decision_ms)
            rows.append(dict(seed=seed, vehicles=count, service_factor=factor,
                             decision_reserve_ms=decision_ms, **row))
    summary = []
    for decision_ms, method in product(decision_grid, METHODS):
        group = [row for row in rows if row['method'] == method and row['decision_reserve_ms'] == decision_ms]
        passed = [row for row in group if row['accepted']]
        times = sorted(row['elapsed_windows_ms'] for row in group)
        summary.append(dict(method=method, decision_reserve_ms=decision_ms, count=len(group),
                            accepted=len(passed), planned_skips=sum(row['skipped'] is not None for row in passed),
                            mean_checks=mean(row['checks'] for row in group),
                            windows_p99_ms=times[int(.99*(len(times)-1))],
                            windows_max_ms=max(times)))
    output.mkdir(parents=True)
    data = dict(scope='synthetic finite PS/PL replay with real new-board profiles',
                identity=identity, models=profile_dict(models),
                selection_frontier=[m.numeric_id for m in pareto(models)], summary=summary,
                mechanism_examples=mechanism_examples(models))
    (output/'summary.json').write_text(json.dumps(data, indent=2)+'\n', encoding='utf-8')
    (output/'runs.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows), encoding='utf-8')
    # Keep paired scenario inputs alongside decisions for exact offline reruns.
    sample, permissions = case(0, 4, models)
    (output/'case_seed0.json').write_text(json.dumps(
        dict(calendar=asdict(sample), permissions=[asdict(p) for p in permissions]), indent=2)+'\n', encoding='utf-8')
    for item in summary:
        print(item)
    print('Output', output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--delivery', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, default=40)
    parser.add_argument('--decision-ms', nargs='+', type=float, default=[.5, 3., 8.])
    args = parser.parse_args()
    if args.seeds < 1 or any(x < 0 or x != x or x == float('inf') for x in args.decision_ms):
        parser.error('Use positive seeds and finite nonnegative decision reserves')
    run(args.delivery, args.output, args.seeds, args.decision_ms)
