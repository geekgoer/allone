"""Measure PS phase overlap and independently compare single and repeated correction."""

import argparse
from pathlib import Path
from statistics import median

from .run_onboard_deadline_route import AMPLITUDES, DEADLINE_MS, GATES, PORT, gate3, run_case, scenario
from .run_onboard_single_pulse import matched
from .run_onboard_value import save
from .run_ps_onboard_pair import BOARD_PACKAGE


RESERVE_MS = 4.5
RECOVERY = ('local', 'one', 'three')
TARGET = 'Drone2'


def timings(report):
    requests = [r for r in report['requests'] if 'return_ms' in r]
    dpu_ms = sum(b-a for a, b in report['background'])/1000
    return dict(sample_ms=report['sample_done_ms'], background_ms=report['background_end_ms']-
                report['background_start_ms'], background_dpu_ms=dpu_ms, ready_ms=report['background_done_ms'],
                encode_ms=[r['encode_ms'] for r in requests],
                rpc_ms=[r['return_ms']-r['submit_ms'] for r in requests if 'submit_ms' in r],
                timely=report['timely'], adopted=report['adopted'],
                statuses=[r['status'] for r in report['requests']])


def recovery_pair(local, arm, expected):
    state_match = {v: matched(local['states'][v], arm['states'][v]) for v in AMPLITUDES}
    target_steps = [r['velocity_step'] for r in (local, arm)]
    timing_pass = (arm['timely'] == 1 and len(arm['adopted']) == expected
                   and all(a['vehicle'] == TARGET and 0 <= a['age_ms'] <= DEADLINE_MS
                           for a in arm['adopted'].values()))
    valid = (all(m['matched'] for m in state_match.values()) and timing_pass and not local['adopted']
             and all(s is not None and abs(s['requested_delta_mps']-s['observed_delta_mps']) < .05
                     for s in target_steps))
    a, b = gate3(local)[TARGET], gate3(arm)[TARGET]
    return dict(valid=valid, timing_pass=timing_pass, state_match=state_match, local_margin_m=a,
                control_margin_m=b, gain_m=b-a if a is not None and b is not None else None,
                local_contact=local['contacts'][TARGET], control_contact=arm['contacts'][TARGET],
                adoption=arm['adopted'])


def run(args):
    timing_plan = [(r, mode) for r in range(1, 5) for mode in
                   (('serial14', 'overlap14') if r % 2 else ('overlap14', 'serial14'))]
    tight_plan = [(r, calls) for r in range(1, 5) for calls in ((18, 19) if r % 2 else (19, 18))]
    recovery_plan = [(r, a) for r in range(1, 4) for a in RECOVERY[r-1:]+RECOVERY[:r-1]]
    with scenario(PORT, args.output, amplitudes=AMPLITUDES, distances=GATES) as (client, gates):
        def fly(name, **kw):
            return run_case(args, client, gates, name, reserve=RESERVE_MS, **kw)

        if args.recheck_one:
            save(args.output/'design.json', dict(adoption_target_ms=DEADLINE_MS, reserve_ms=RESERVE_MS,
                 background_calls=17, overlap_sample=True, repetitions=4))
            results = {}
            for rep in range(1, 5):
                name = f'overlap17_r{rep}'
                results[name] = timings(fly(name, policy='static', calls=17, overlap=True))
                save(args.output/'stages.json', results)
            return

        save(args.output/'design.json', dict(adoption_target_ms=DEADLINE_MS, reserve_ms=RESERVE_MS,
            timing=timing_plan, tight=tight_plan, recovery=recovery_plan, recovery_target=TARGET,
            recovery_background_calls=6, restart_after_ms=[75, 150]))
        timing = {}
        for rep, mode in timing_plan:
            name = f'{mode}_r{rep}'
            timing[name] = timings(fly(name, policy='static', calls=14, overlap=mode == 'overlap14'))
            save(args.output/'stages.json', timing)
        for rep, calls in tight_plan:
            name = f'overlap{calls}_r{rep}'
            timing[name] = timings(fly(name, policy='static', calls=calls, overlap=True))
            save(args.output/'stages.json', timing)
        runs, pairs = {}, {}
        for rep, arm in recovery_plan:
            name = f'recovery_r{rep}_{arm}'
            runs[name] = fly(name, policy='local' if arm == 'local' else 'recovery', calls=6,
                             target=TARGET, fixed=TARGET, recovery_commands=3 if arm == 'three' else 1)
            save(args.output/'recovery.json', runs)
        for rep in range(1, 4):
            for arm, count in (('one', 1), ('three', 3)):
                rows = [runs[f'recovery_r{rep}_{p}'] for p in ('local', arm)]
                pairs[f'r{rep}_{arm}'] = recovery_pair(*rows, count)
        save(args.output/'recovery_pairs.json', pairs)
        for mode in ('serial14','overlap14','overlap18','overlap19'):
            rows = [r for name, r in timing.items() if name.startswith(mode)]
            print(mode, 'ready_median_ms', round(median(r['ready_ms'] for r in rows), 3),
                  'timely', [r['timely'] for r in rows], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--binary', required=True)
    parser.add_argument('--package', default=BOARD_PACKAGE)
    parser.add_argument('--windows-ip', default='192.168.77.1')
    parser.add_argument('--board', default='axu3egb')
    parser.add_argument('--ssh-hop', default='vitis-vm')
    parser.add_argument('--recheck-one', action='store_true', help='Repeat the 17-background one-command check four times')
    run(parser.parse_args())
