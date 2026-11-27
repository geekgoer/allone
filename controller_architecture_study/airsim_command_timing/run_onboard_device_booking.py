"""Measure a protected inference opportunity against job-boundary priority."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import median
import subprocess

from .run_onboard_deadline_route import AMPLITUDES, GATES, PORT, gate3, run_case, scenario
from .run_onboard_reserved_recovery import CALLS, state_match
from .run_onboard_stage_recovery import RESERVE_MS, TARGET
from .run_onboard_value import read, save
from .run_ps_onboard_pair import BOARD_PACKAGE, ssh_command


ARMS = ('boundary', 'protected')
BOARD_ARMS = (*ARMS, 'latest')
FLIGHTS = (6, 11, 6, 11, 6, 6)


def measure_board(args):
    args.output.mkdir(parents=True)
    with (args.output/'board.log').open('w', encoding='utf-8') as log:
        proc = subprocess.run(ssh_command(args), input=b'{"policy":"booking_bench"}\n',
                              stdout=subprocess.PIPE, stderr=log, check=True, timeout=90)
    report = json.loads(proc.stdout.splitlines()[-1])
    save(args.output/'board.json', report)
    pairs = defaultdict(dict)
    for row in report['trials']:
        origin, release = row['origin_us'], row['release_us']
        cb, ce, crb, cre = row['foreground']
        bb, be, brb, bre = row['background']
        assert cb >= origin + report['ready_us'] and bb >= origin + release
        assert cb <= crb < cre <= ce and bb <= brb < bre <= be
        assert (be <= cb if row['ran_before'] else ce <= bb)
        arm = row['arm']
        forecast = arm != 'boundary' and row['ran_before']
        data = dict(delivery_proxy_ok=ce-origin+report['post_budget_us'] <= report['deadline_us'],
                    ready_wait_ms=(cb-origin-report['ready_us'])/1000, bg_window_end_ms=(be-origin)/1000,
                    overran_guard=forecast and be > row['protection_limit_us'])
        key = row['model'], release, row['rep']
        if arm in pairs[key]:
            raise ValueError('Duplicate board trial')
        pairs[key][arm] = data
    groups = {}
    for model, release in sorted({key[:2] for key in pairs}):
        selected = [v for k, v in pairs.items() if k[:2] == (model, release)]
        assert len(selected) == 20 and all(set(pair) == set(BOARD_ARMS) for pair in selected)
        arms = {arm: [p[arm] for p in selected] for arm in BOARD_ARMS}
        added_bg = {arm: median(p[arm]['bg_window_end_ms']-p['boundary']['bg_window_end_ms'] for p in selected)
                    for arm in ('protected', 'latest')}
        groups[f'id{model}_at{release}'] = dict(model=model, release_us=release, pairs=len(selected),
            median_added_background_ms=added_bg,
            **{arm: dict(timely=sum(x['delivery_proxy_ok'] for x in rows),
                         median_ready_wait_ms=median(x['ready_wait_ms'] for x in rows),
                         background_miss30=sum(x['bg_window_end_ms'] > 30 for x in rows),
                         background_miss40=sum(x['bg_window_end_ms'] > 40 for x in rows),
                         overran_guard=sum(x['overran_guard'] for x in rows)) for arm, rows in arms.items()})
    save(args.output/'board_summary.json', groups)


def check_pair(reports, model):
    baseline, protected = (reports[k] for k in ARMS)
    steps = [r['velocity_step'] for r in (baseline, protected)]
    rows = {}
    for arm in ARMS:
        report = reports[arm]
        booked = report['windows'][1]['device_booking']
        adopted = {r['window']: r for r in report['adopted'].values() if r['on_time']}
        first_ack = any(r['window'] == 0 and r['accepted'] for r in report['requests'])
        if booked['background_model'] != model or booked['claimed'] != (arm == 'protected' and first_ack):
            raise ValueError('Wrong booking status')
        before, finished, run_begin, run_end = booked['background']
        assert booked['release_us'] <= before <= run_begin < run_end <= finished
        deadline = booked['ready_target_us']-22000+30000
        rows[arm] = dict(first_on_time=0 in adopted, followup_on_time=1 in adopted,
                         followup_ms=adopted[1]['window_ms'] if 1 in adopted else None,
                         pass_gates=report['pass_gates'], contacts=report['contacts'], gate3_m=gate3(report),
                         ran_before=booked['ran_before'], background_response_ms=(finished-booked['release_us'])/1000,
                         background_miss30=finished > deadline, background_miss40=finished > deadline+10000)
    comparable = (state_match(baseline, protected) and abs(steps[0]['x_m']-steps[1]['x_m']) <= .022 and
                  all(abs(s['requested_delta_mps']-s['observed_delta_mps']) < .05 for s in steps) and
                  all(r['first_on_time'] for r in rows.values()))
    return dict(model=model, comparable=comparable, arms=rows,
                added_background_ms=rows['protected']['background_response_ms']-rows['boundary']['background_response_ms'])


def fly(args):
    board = read(args.board_report)
    budgets = {model: board['costs'][str(model)]['guard_us'] for model in set(FLIGHTS)}
    pairs = {}
    cases = ((6, 12000), (6, 20000)) if args.latest_compare else tuple((model, 20000) for model in FLIGHTS)
    policies = BOARD_ARMS if args.latest_compare else ARMS
    with scenario(PORT, args.output, amplitudes=AMPLITUDES, distances=GATES) as (client, gates):
        save(args.output/'design.json', dict(budgets_us=budgets, cases=cases, policies=policies, readiness_ms=22,
             deadline_ms=30, windows_ms=(0,75)))
        for rep, (model, release) in enumerate(cases, 1):
            order = policies if rep % 2 else policies[::-1]
            reports = {arm: run_case(args, client, gates, f'r{rep}_{arm}', policy='fixed', calls=CALLS,
                reserve=RESERVE_MS, target=TARGET,
                recovery_schedule=dict(windows=2, healthy_error_m=[], fixed_recipients=(TARGET,TARGET),
                    device_booking=dict(model=model, guard_us=budgets[model], protect=arm != 'boundary',
                                        release_us=release, latest_start=arm == 'latest')))
                       for arm in order}
            pair = check_pair(reports, model)
            if args.latest_compare:
                latest = check_pair(dict(boundary=reports['boundary'], protected=reports['latest']), model)
                pair['arms']['latest'] = latest['arms']['protected']
                pair['latest_added_background_ms'] = latest['added_background_ms']
                pair['comparable'] &= latest['comparable']
                pair['release_us'] = release
            pairs[rep] = pair
            save(args.output/'pairs.json', pairs)
    if args.latest_compare:
        print(pairs, flush=True)
        return
    summary = {}
    for model in set(FLIGHTS):
        selected = [p for p in pairs.values() if p['comparable'] and p['model'] == model]
        summary[model] = dict(comparable=len(selected), **{arm: {
            field: sum(p['arms'][arm][field] for p in selected)
            for field in ('followup_on_time','pass_gates','background_miss30','background_miss40')} for arm in ARMS},
            median_added_background_ms=median(p['added_background_ms'] for p in selected) if selected else None)
    save(args.output/'summary.json', summary)
    print(summary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--binary', required=True)
    parser.add_argument('--package', default=BOARD_PACKAGE)
    parser.add_argument('--windows-ip', default='192.168.77.1')
    parser.add_argument('--board', default='axu3egb')
    parser.add_argument('--ssh-hop', default='vitis-vm')
    stage = parser.add_mutually_exclusive_group(required=True)
    stage.add_argument('--board-only', action='store_true')
    stage.add_argument('--board-report', type=Path)
    parser.add_argument('--latest-compare', action='store_true', help='Probe latest-start protection in two flight conditions')
    args = parser.parse_args()
    if args.latest_compare and not args.board_report:
        parser.error('--latest-compare requires --board-report')
    measure_board(args) if args.board_only else fly(args)
