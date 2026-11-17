"""Finite scheduling experiment with assumed Main18 PTQ quality and phase contracts."""
import argparse
from dataclasses import dataclass, replace
from itertools import product
import json
from pathlib import Path
from random import Random
from statistics import mean
from time import perf_counter_ns

from .bottleneck_abc import abc_repair
from .bottleneck_joint import Job, Loop, Mode, Option, Platform, Scenario, simulate
from .replay_remaining_chain import pipeline_passed


# Quoted Ubuntu selection-set results. Only the six disclosed per-model rows.
ERROR = {'5': .007157, '10': .005027, '11': .003289, '16': .003303,
         '2': .003876, '6': .005173}
METHODS = ('fixed', 'abc', 'abc_drop_blind', 'abc_drop_phase', 'abc_drop_phase_cost_only')
CHECK_BUDGET = 5
EPS = 1e-8


@dataclass(frozen=True)
class Obligation:
    phase: str
    hold_until: float
    # Estimated phase of the whole interval from the previous command to recovery.
    clear_to_hold: bool
    skipped_before: bool
    error_limit: float


def scenario_for(seed, modes, n, mix, decision_ms, quality=True):
    rng = Random(seed)
    platform = Platform(prep=rng.uniform(.15, .6), dispatch=.08, post=.2,
                        ingress=.2, tail=.35, decision=decision_ms, switch=.011,
                        initial_pl_block=rng.uniform(0., 3.), rearm=.25)
    loops, recovery, obligations = [], [], {}
    for i in range(n):
        name = str(i)
        phase = 'early' if i < (n+1)//2 else 'late'
        release = rng.uniform(1., 7.) + (rng.uniform(0., 8.) * i/n if mix == 'staggered' else 0.)
        period = 50.
        slack = rng.uniform(8., 19.) if phase == 'early' else rng.uniform(11., 26.)
        deadline = release + slack
        recover_release = release + period
        hold_until = recover_release + rng.uniform(9., 22.)
        if rng.random() < .12:
            hold_until = recover_release + rng.uniform(0., 3.)
        cap = .0055 if phase == 'early' else .008
        obligations[name] = Obligation(phase, hold_until, rng.random() >= .25,
                                       rng.random() < .10, cap)
        current = modes[rng.choice(('11', '16', '2', '6'))]
        options = [Option(period, current)]
        release_locked = i % 3 == 0 or release <= decision_ms
        legal_modes = sorted(modes.values(), key=lambda m: (ERROR[m.name] if quality else 0.,
                                                           m.segments[0], m.name))
        for candidate_period, model in product((50., 40., 60.), legal_modes):
            if candidate_period == period and model == current:
                continue
            if release <= decision_ms:
                continue
            if release_locked and candidate_period != period:
                continue
            if quality and ERROR[model.name] > cap:
                continue
            # The next nominal slot stays fixed in this finite experiment.
            options.append(Option(candidate_period, model))
        loops.append(Loop(name, release-period, release, deadline, 125., tuple(options),
                          release_locked=release_locked))
        successor = modes['10']
        recovery.append(Job(f'{name}:recovery', name, recover_release,
                            min(recover_release + period, hold_until), successor.segments,
                            successor.name))
    # A zero generation horizon emits one editable release per loop. All next
    # slots are explicit mandatory jobs and remain present after an omission.
    return Scenario(tuple(loops), platform, horizon=0., carry=tuple(recovery),
                    initial_policy=('edf', 1), legal_groups=(1,)), obligations


def drop_eligible(loop, contract, decision_ms, phase_aware):
    if loop.release_locked or loop.next_release <= decision_ms or contract.skipped_before:
        return False
    return not phase_aware or (contract.phase == 'late' and contract.clear_to_hold)


def recovery_feasible(forecast, omitted, contracts):
    if not pipeline_passed({'accepted': forecast['feasible'], 'forecast': forecast}):
        return False
    if omitted is None:
        return True
    job = f'{omitted}:recovery'
    return (job in forecast['command_finish'] and
            forecast['command_finish'][job] <= contracts[omitted].hold_until + EPS)


def run_policy(scenario, contracts, method):
    t0 = perf_counter_ns()
    allow_drop = 'drop' in method
    phase_aware = 'phase' in method
    if method == 'fixed':
        choice, order = (0,) * len(scenario.loops), scenario.initial_policy
        forecast = simulate(scenario, choice, order)
        diagnosis, checks = 'fixed', 1
    else:
        result = abc_repair(scenario, max_checks=CHECK_BUDGET, allow_guarded_edf=True)
        choice, order, forecast = result['choice'], result['policy'], result['forecast']
        diagnosis, checks = result['initial_type'], result['checks']
    ok = recovery_feasible(forecast, None, contracts)
    omitted = None
    active = scenario.loops
    if allow_drop and not ok:
        misses = set(forecast['misses'])
        # Simulate removal from the same proposed plan. Try a missed job first,
        # then the largest saving so a drop can also relieve another loop.
        ranked = sorted(enumerate(active), key=lambda item: (
            f'{item[1].name}:0' not in misses,
            -item[1].options[choice[item[0]]].mode.segments[0], item[1].name))
        for i, loop in ranked:
            if checks >= CHECK_BUDGET:
                break
            name = loop.name
            if not drop_eligible(loop, contracts[name], scenario.platform.decision, phase_aware):
                continue
            kept = active[:i] + active[i+1:]
            indices = choice[:i] + choice[i+1:]
            probe = simulate(replace(scenario, loops=kept), indices, order)
            checks += 1
            if recovery_feasible(probe, name, contracts):
                omitted, active, choice, forecast, ok = name, kept, indices, probe, True
                break
    if ok and method != 'fixed' and not method.endswith('cost_only') and checks < CHECK_BUDGET:
        upgrades = []
        for i, loop in enumerate(active):
            current = loop.options[choice[i]]
            slack = forecast['slack'][f'{loop.name}:0']
            for index, option in enumerate(loop.options):
                if option.period != current.period:
                    continue
                gain = ERROR[current.mode.name] - ERROR[option.mode.name]
                extra = option.mode.segments[0] - current.mode.segments[0]
                if gain > 0 and extra <= slack + EPS:
                    # With no added service, quality gains take precedence.
                    upgrades.append((extra > 0, -gain/max(extra, 1e-6), i, index))
        if upgrades:
            _, _, i, index = min(upgrades)
            proposed = choice[:i] + (index,) + choice[i+1:]
            probe = simulate(replace(scenario, loops=active), proposed, order)
            checks += 1
            if recovery_feasible(probe, omitted, contracts):
                choice, forecast = proposed, probe
    protected = omitted is not None and not drop_eligible(
        next(loop for loop in scenario.loops if loop.name == omitted), contracts[omitted],
        scenario.platform.decision, True)
    fit = [ERROR[loop.options[index].mode.name] for loop, index in zip(active, choice)]
    quality_ok = all(error <= contracts[loop.name].error_limit
                     for loop, error in zip(active, fit))
    return dict(timing_ok=ok, contract_ok=ok and not protected and quality_ok,
                dropped=omitted, protected_drop=protected, quality_ok=quality_ok,
                mean_kept_error=mean(fit), checks=checks, diagnosis=diagnosis,
                actions={loop.name: {'period_ms': loop.options[index].period,
                                     'model_id': loop.options[index].mode.name}
                         for loop, index in zip(active, choice)},
                order=order[0],
                min_slack_ms=min(forecast['slack'].values()),
                runtime_ms=(perf_counter_ns()-t0)/1e6)


def paired_phase_example(modes):
    platform = Platform(decision=.5)
    loops = (Loop('protected', -48., 2., 8., 100., (Option(50., modes['11']),), True),
             Loop('donor', -48., 2., 8., 100., (Option(50., modes['6']),)))
    recovery = Job('donor:recovery', 'donor', 52., 62., modes['10'].segments, '10')
    scenario = Scenario(loops, platform, horizon=0., carry=(recovery,),
                        initial_policy=('edf', 1), legal_groups=(1,))
    contracts = {'protected': Obligation('early', 61., False, False, .008),
                 'donor': Obligation('late', 62., True, False, .008)}
    late = run_policy(scenario, contracts, 'abc_drop_phase')
    early_contract = {**contracts, 'donor': replace(contracts['donor'], phase='early')}
    early = run_policy(scenario, early_contract, 'abc_drop_phase')
    blind = run_policy(scenario, early_contract, 'abc_drop_blind')
    tight = replace(scenario, carry=(replace(recovery, deadline=52.1),))
    tight_contract = {**contracts, 'donor': replace(contracts['donor'], hold_until=52.1)}
    expired = run_policy(tight, tight_contract, 'abc_drop_phase')
    assert late['contract_ok'] and late['dropped'] == 'donor'
    assert not early['contract_ok'] and early['dropped'] is None
    assert blind['timing_ok'] and blind['protected_drop'] and not blind['contract_ok']
    assert not expired['contract_ok'] and expired['dropped'] is None
    return dict(late=late, early=early, early_phase_blind=blind, recovery_too_late=expired)


def replay(profile, output, seeds, decision_ms):
    rows = json.loads(profile.read_text(encoding='utf-8'))['models']
    costs = {str(item['id']): item['service_p99_ms'] for item in rows
             if str(item['id']) in ERROR}
    if costs.keys() != ERROR.keys() or any(value <= 0 for value in costs.values()):
        raise ValueError('Expected all six positive old-board service P99 values')
    runs = []
    for factor, n, mix in product((1., 1.25, 1.5), (4, 8), ('staggered', 'synchronous')):
        modes = {key: Mode(key, (cost * factor,)) for key, cost in costs.items()}
        for seed in range(seeds):
            scenario, contracts = scenario_for(seed, modes, n, mix, decision_ms)
            for method in METHODS:
                if method.endswith('cost_only'):
                    scenario, contracts = scenario_for(seed, modes, n, mix, decision_ms, quality=False)
                runs.append(dict(factor=factor, n=n, mix=mix, seed=seed, method=method,
                                 **run_policy(scenario, contracts, method)))
    summary = {}
    for method in METHODS:
        group = [row for row in runs if row['method'] == method]
        times = sorted(row['runtime_ms'] for row in group)
        summary[method] = dict(
            cases=len(group), timing=sum(row['timing_ok'] for row in group),
            contract=sum(row['contract_ok'] for row in group),
            drops=sum(row['dropped'] is not None for row in group),
            protected_drops=sum(row['protected_drop'] for row in group),
            retained_error_on_contract_passes=(
                mean(row['mean_kept_error'] for row in group if row['contract_ok'])
                if any(row['contract_ok'] for row in group) else None),
            mean_checks=mean(row['checks'] for row in group),
            p99_windows_ms=times[int(.99*(len(times)-1))])
    assert all(row['checks'] <= CHECK_BUDGET for row in runs)
    assert not summary['abc_drop_phase']['protected_drops']
    output.mkdir(parents=True, exist_ok=False)
    example = paired_phase_example({key: Mode(key, (cost,)) for key, cost in costs.items()})
    frontier = sorted(
        (key for key in costs if not any(other != key and costs[other] <= costs[key]
                                         and ERROR[other] <= ERROR[key]
                                         and (costs[other], ERROR[other]) != (costs[key], ERROR[key])
                                         for other in costs)),
        key=costs.get)
    data = dict(scope='synthetic finite scheduling replay, no flight or new-board measurement',
                assumed_decision_reserve_ms=decision_ms,
                new_error_assumed=ERROR, old_board_p99_ms=costs, point_estimate_frontier=frontier,
                summary=summary,
                paired_example_at_nominal_cost=example)
    (output/'summary.json').write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    (output/'runs.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in runs), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, default=40)
    parser.add_argument('--decision-ms', type=float, default=.5)
    args = parser.parse_args()
    if args.seeds < 1 or not 0 <= args.decision_ms < float('inf'):
        parser.error('--seeds must be positive and --decision-ms finite and nonnegative')
    replay(args.profile, args.output, args.seeds, args.decision_ms)
