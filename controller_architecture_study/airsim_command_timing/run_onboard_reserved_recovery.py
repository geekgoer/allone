"""Find a sufficient correction length and compare its scheduling across tight windows."""

import argparse
from pathlib import Path
from statistics import mean

from .run_onboard_deadline_route import AMPLITUDES, DEADLINE_MS, GATES, PORT, gate3, run_case, scenario
from .run_onboard_single_pulse import matched
from .run_onboard_stage_recovery import RESERVE_MS, TARGET
from .run_onboard_value import save
from .run_ps_onboard_pair import BOARD_PACKAGE


CALLS = 17
REPETITIONS = 3
HEALTHY_TOLERANCE_M = .025
POLICIES = ('queue', 'risk', 'reserve')


def valid_steps(*runs):
    return all(abs(r['velocity_step']['requested_delta_mps'] - r['velocity_step']['observed_delta_mps']) < .05
               for r in runs)


def state_match(a, b):
    return all(matched(a['states'][v], b['states'][v])['matched'] for v in AMPLITUDES)


def check_doses(runs):
    pair_results = {}
    for rep in range(REPETITIONS):
        baseline = runs[f'dose_r{rep+1}_1']
        for n in (1, 2, 3):
            name = f'dose_r{rep+1}_{n}'
            d = runs[name]
            valid = (d['timely'] == 1 and len(d['adopted']) == n
                     and all(r['vehicle'] == TARGET and 0 <= r['age_ms'] <= DEADLINE_MS
                             for r in d['adopted'].values())
                     and valid_steps(baseline, d) and state_match(baseline, d))
            pair_results[name] = dict(valid=valid, clear=not d['contacts'][TARGET], pass_gates=d['pass_gates'],
                                      gate3_m=gate3(d)[TARGET])
    feasible = [n for n in (1, 2, 3) if all(pair_results[f'dose_r{r+1}_{n}']['valid'] and
                   pair_results[f'dose_r{r+1}_{n}']['pass_gates'] for r in range(REPETITIONS))]
    return dict(pairs=pair_results, sufficient=min(feasible, default=None))


def result_for(r, windows):
    timely = [a for a in r['adopted'].values() if a['on_time']]
    interval_ms = sum(b-a for a,b in r['background'])/1000
    control_ms = sum(t.get('dpu_run_us', 0) for t in r['requests'])/1000
    completion = {v:sum(x['vehicle'] == v for x in timely) for v in AMPLITUDES}
    return dict(timely_by_vehicle=completion,
                recovery_complete={x['window'] for x in timely if x['vehicle'] == TARGET} == set(range(windows)),
                gate3_m=gate3(r), contact=r['contacts'], background_jobs=r['background_calls'],
                background_call_ms=interval_ms, foreground_call_ms=control_ms, pass_gates=r['pass_gates'],
                released=[w['window'] for w in r['windows'] if w['released']],
                first_candidates=[w['first_candidate'] for w in r['windows']])


def schedule_audit(runs, windows):
    audit = {name:result_for(r, windows) for name,r in runs.items()}
    for rep in range(1, REPETITIONS+1):
        baseline = runs[f'tight_r{rep}_queue']
        for policy in POLICIES:
            name = f'tight_r{rep}_{policy}'
            audit[name]['state_matched'] = state_match(baseline, runs[name]) and valid_steps(baseline, runs[name])
    return audit


def run(args):
    with scenario(PORT, args.output, amplitudes=AMPLITUDES, distances=GATES) as (client, gates):
        def fly(name, **kw):
            return run_case(args, client, gates, name, reserve=RESERVE_MS, **kw)
        rotations = [(1,2,3), (2,3,1), (3,1,2)]
        save(args.output/'design.json', dict(doses=rotations, stage2_background_calls_per_window=CALLS,
             stage2_policies=POLICIES, healthy_tolerance_m=HEALTHY_TOLERANCE_M, window_spacing_ms=75,
             adoption_target_ms=DEADLINE_MS, reserve_ms=RESERVE_MS))
        doses = {}
        for rep, order in enumerate(rotations, 1):
            for n in order:
                name = f'dose_r{rep}_{n}'
                doses[name] = fly(name, policy='recovery', calls=6, target=TARGET, fixed=TARGET,
                                  recovery_commands=n)
                save(args.output/'doses.json', doses)
        dose_audit = check_doses(doses)
        save(args.output/'dose_audit.json', dose_audit)
        windows = dose_audit['sufficient']
        if windows is None or windows == 1:
            print(f'No multi-window comparison for this dose result ({windows=}).', flush=True)
            return

        def scheduled(name, policy, *, target=None, healthy=()):
            return fly(name, policy=policy, calls=CALLS, target=target,
                       recovery_schedule=dict(windows=windows, healthy_error_m=healthy))

        warm = [scheduled(f'healthy_calibration_{i+1}', 'observe') for i in range(2)]
        healthy = [{v:max(r['windows'][w]['risk'][v] for r in warm)+HEALTHY_TOLERANCE_M for v in AMPLITUDES}
                   for w in range(windows)]
        plan = [(rep+1, p) for rep in range(REPETITIONS) for p in POLICIES[rep:]+POLICIES[:rep]]
        save(args.output/'schedule_plan.json', dict(windows=windows, calls_per_window=CALLS,
             healthy_error_m=healthy, plan=plan, background_priority='complete_batch_before_foreground'))
        runs = {}
        for rep, p in plan:
            name = f'tight_r{rep}_{p}'
            runs[name] = scheduled(name, p, target=TARGET, healthy=healthy)
            save(args.output/'schedule_runs.json', runs)
        runs['healthy_release'] = scheduled('healthy_release', 'reserve', healthy=healthy)
        save(args.output/'schedule_runs.json', runs)
        audit = schedule_audit(runs, windows)
        save(args.output/'schedule_audit.json', audit)
        for policy in POLICIES:
            rows = [r for name, r in audit.items() if name.endswith('_'+policy)]
            print(policy, 'complete', sum(r['recovery_complete'] for r in rows),
                  'contacts', sum(bool(r['contact'][TARGET]) for r in rows),
                  'mean peer updates', mean(r['timely_by_vehicle']['Drone1'] for r in rows), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--binary', required=True)
    parser.add_argument('--package', default=BOARD_PACKAGE)
    parser.add_argument('--windows-ip', default='192.168.77.1')
    parser.add_argument('--board', default='axu3egb')
    parser.add_argument('--ssh-hop', default='vitis-vm')
    run(parser.parse_args())
