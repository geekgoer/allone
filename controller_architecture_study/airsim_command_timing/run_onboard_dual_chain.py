"""Isolate the physical effect of two competing correction pairs on the real DPU."""

import argparse
from pathlib import Path

from .run_onboard_deadline_route import GATES, PORT, STEP_MPS, gate3, run_case, scenario
from .run_onboard_reserved_recovery import CALLS
from .run_onboard_single_pulse import matched
from .run_onboard_stage_recovery import RESERVE_MS
from .run_onboard_value import save
from .run_ps_onboard_pair import BOARD_PACKAGE


A, B = 'Drone1', 'Drone2'
AMPLITUDES = {A: .57, B: -.57}
ARMS = {
    'interleaved': (A, B, A, B),
    'early_only': (A, B, '', ''),
    'late_only': ('', '', A, B),
    'consecutive': (A, A, B, B),
    'reversed': (B, A, B, A),
}


def audit(reports, windows_ms):
    baseline = reports['interleaved']
    arms = {}
    for arm, report in reports.items():
        planned = ARMS[arm]
        adopted = {r['window']: r['vehicle'] for r in report['adopted'].values() if r['on_time']}
        times = [w['start_ns'] for w in report['windows']]
        time_ok = (len(times) == len(planned) and len(report['adopted']) == len(adopted) and
                   adopted == {i: v for i, v in enumerate(planned) if v} and
                   all(abs((stamp-times[0])/1e6-ms) <= 6. for stamp, ms in zip(times, windows_ms)))
        state_ok = all(matched(baseline['states'][v], report['states'][v])['matched'] for v in AMPLITUDES)
        steps_ok = all(abs(report['velocity_step'][v]['x_m'] - baseline['velocity_step'][v]['x_m']) <= .022
                       and abs(report['velocity_step'][v]['observed_delta_mps']
                               - report['velocity_step'][v]['requested_delta_mps']) < .05
                       for v in AMPLITUDES)
        arms[arm] = dict(comparable=time_ok and state_ok and steps_ok, timing_ok=time_ok,
                         state_matched=state_ok, steps_matched=steps_ok,
                         timely=len(adopted), adopted=adopted, gate3_m=gate3(report), contacts=report['contacts'],
                         pass_gates=report['pass_gates'],
                         per_vehicle_pass={v: not report['contacts'][v] and
                                           all(m is not None and m > 0 for m in report['by_gate_m'][v].values())
                                           for v in AMPLITUDES})
    qualified = {v: all(arms[arm]['comparable'] for arm in ('interleaved', 'early_only', 'late_only')) and
                    arms['interleaved']['per_vehicle_pass'][v] and
                    bool(arms['early_only']['contacts'][v]) and bool(arms['late_only']['contacts'][v])
                 for v in AMPLITUDES}
    return dict(arms=arms, both_single_miss_but_pair_pass=qualified)


def run(args):
    windows_ms = tuple(w*args.window_spacing_ms for w in range(4))
    with scenario(PORT, args.output, amplitudes=AMPLITUDES, distances=GATES) as (client, gates):
        save(args.output/'design.json', dict(arms=ARMS, amplitudes=AMPLITUDES, windows_ms=windows_ms,
             background_calls=CALLS, model_id=11, step_mps=STEP_MPS, step_offset_m=.26))
        groups = {}
        for rep in range(args.repeats):
            order = tuple(ARMS)
            order = order[rep % len(order):]+order[:rep % len(order)]
            reports = {arm: run_case(args, client, gates, f'r{rep+1}_{arm}', policy='fixed', calls=CALLS,
                reserve=RESERVE_MS,
                paired_steps=(AMPLITUDES, {v: STEP_MPS for v in AMPLITUDES}),
                recovery_schedule=dict(windows=len(windows_ms), spacing_ms=args.window_spacing_ms,
                                       healthy_error_m=[], fixed_recipients=ARMS[arm]))
                for arm in order}
            groups[rep+1] = audit(reports, windows_ms)
            save(args.output/'groups.json', groups)
            print('PAIRED_QUALIFICATION', groups[rep+1]['both_single_miss_but_pair_pass'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--binary', required=True)
    parser.add_argument('--package', default=BOARD_PACKAGE)
    parser.add_argument('--windows-ip', default='192.168.77.1')
    parser.add_argument('--board', default='axu3egb')
    parser.add_argument('--ssh-hop', default='vitis-vm')
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--window-spacing-ms', type=int, choices=(75, 100, 125), default=75)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    run(args)
