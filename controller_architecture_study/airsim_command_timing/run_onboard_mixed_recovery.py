"""Compare one single-update need and two paired needs on three real DPU slots."""

import argparse
from pathlib import Path
from statistics import median

from .run_onboard_deadline_route import DEADLINE_MS, GATES, PORT, STEP_MPS, run_case, scenario
from .run_onboard_dual_chain import A, B
from .run_onboard_reserved_recovery import CALLS
from .run_onboard_single_pulse import matched
from .run_onboard_stage_recovery import RESERVE_MS
from .run_onboard_value import read, save
from .run_ps_onboard_pair import BOARD_PACKAGE


C = 'Drone3'
AMPLITUDES = {A: .57, B: -.57, C: .535}
WINDOWS = (0, 125, 250)
OPTIONS = {A: ((0, 1), (0, 2)), B: ((0, 2), (1, 2)), C: ((1,), (2,))}
CALIBRATION = {'no_single': (A, A, ''), 'single_first': (A, C, A),
               'single_second': (A, A, C), 'second_pair': ('', B, B)}
CALIBRATION_STEPS = (.231, .234, .228)
POLICIES = ('edf', 'finish_first', 'complete')


def audit(reports):
    base = next(iter(reports.values()))
    rows = {}
    for name, report in reports.items():
        states_ok = all(matched(base['states'][v], report['states'][v])['matched'] for v in AMPLITUDES)
        pulse_ok = all(abs(report['velocity_step'][v]['x_m']-base['velocity_step'][v]['x_m']) <= .022 and
                       abs(report['velocity_step'][v]['requested_delta_mps']-
                           report['velocity_step'][v]['observed_delta_mps']) < .05 for v in AMPLITUDES)
        windows = report['windows']
        origin = windows[0]['start_ns']
        planned = {i: w['first_candidate'] for i, w in enumerate(windows) if w['first_candidate']}
        adopted = {r['window']: r['vehicle'] for r in report['adopted'].values() if r['on_time']}
        timing_ok = (len(windows) == len(WINDOWS) and adopted == planned and
                     len(adopted) == len(report['adopted']) and
                     all(abs((w['start_ns']-origin)/1e6-t) <= 6. for w,t in zip(windows,WINDOWS)))
        qualified = {v: tuple(i for i in sorted(adopted) if adopted[i] == v) in OPTIONS[v] for v in AMPLITUDES}
        rows[name] = dict(comparable=states_ok and pulse_ok and timing_ok, states_ok=states_ok, pulses_ok=pulse_ok,
                          timing_ok=timing_ok, planned=planned, adopted=adopted, qualified=qualified,
                          passes=report['passes'], contacts=report['contacts'],
                          gate3_mm={v: None if g['gate3'] is None else 1000*g['gate3']
                                    for v,g in report['by_gate_m'].items()},
                          choose_us=[w['pair_choose_us'] for w in windows if 'pair_choose_us' in w])
    return rows


def summary(groups):
    paired = [g for g in groups.values() if all(r['comparable'] and len(r['adopted']) == len(WINDOWS)
                                                for r in g.values())]
    rows = {}
    for policy in POLICIES:
        values = [r for g in groups.values() for r in g[policy]['choose_us']]
        arms = [g[policy] for g in paired]
        rows[policy] = dict(paired=len(paired), timely=sum(len(r['adopted']) for r in arms),
                            qualified=sum(sum(r['qualified'].values()) for r in arms),
                            passed=sum(sum(r['passes'].values()) for r in arms),
                            contacts={v: sum(bool(r['contacts'][v]) for r in arms) for v in AMPLITUDES},
                            median_choose_us=median(values), max_choose_us=max(values))
    return rows


def run(args):
    with scenario(PORT, args.output, amplitudes=AMPLITUDES, distances=GATES) as (client, gates):
        save(args.output/'design.json', dict(windows_ms=WINDOWS, amplitudes=AMPLITUDES, options=OPTIONS,
             calibration=CALIBRATION, policies=POLICIES, background_calls=CALLS, model_id=11,
             pair_step_mps=STEP_MPS, singleton_screen_mps=CALIBRATION_STEPS, adoption_ms=DEADLINE_MS))

        def fly(name, step, fixed=(), policy=''):
            result = run_case(args, client, gates, name, policy='fixed', calls=CALLS, reserve=RESERVE_MS,
                paired_steps=(AMPLITUDES, {A: STEP_MPS, B: STEP_MPS, C: step}),
                recovery_schedule=dict(windows=len(WINDOWS), spacing_ms=125, healthy_error_m=[],
                                       mixed_recovery=True, fixed_recipients=fixed, pair_choice=policy))
            result['passes'] = {v: r['pass_gates'] for v,r in read(args.output/name/'report.json').items()}
            return result

        screens = {}
        for index, step in enumerate(CALIBRATION_STEPS, 1):
            raw = {arm: fly(f'cal{index}_{arm}', step, fixed) for arm, fixed in CALIBRATION.items()}
            rows = audit(raw)
            qualified = (all(r['comparable'] for r in rows.values()) and
                         rows['no_single']['contacts'][C] and rows['single_first']['passes'][C] and
                         rows['single_second']['passes'][C] and rows['second_pair']['passes'][B])
            screens[index] = dict(step_mps=step, qualified=bool(qualified), arms=rows)
            save(args.output/'calibration.json', screens)
            if qualified:
                break
        else:
            print('No calibrated single-update case in the fixed screen', flush=True)
            return
        save(args.output/'frozen.json', dict(step_mps=step, calibration_round=index))
        groups = {}
        for rep in range(args.repeats):
            shift = rep % len(POLICIES)
            order = POLICIES[shift:]+POLICIES[:shift]
            groups[rep+1] = audit({p: fly(f'r{rep+1}_{p}', step, policy=p) for p in order})
            save(args.output/'groups.json', groups)
            result = summary(groups)
            save(args.output/'summary.json', result)
            print('MIXED_COMPARISON', result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--binary', required=True)
    parser.add_argument('--package', default=BOARD_PACKAGE)
    parser.add_argument('--windows-ip', default='192.168.77.1')
    parser.add_argument('--board', default='axu3egb')
    parser.add_argument('--ssh-hop', default='vitis-vm')
    parser.add_argument('--repeats', type=int, default=2)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    run(args)
