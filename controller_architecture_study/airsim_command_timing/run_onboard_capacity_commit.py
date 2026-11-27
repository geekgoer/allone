"""Measure queue visibility and the cost of securing a follow-up DPU opportunity."""

import argparse
from bisect import bisect_left, bisect_right
import csv
from hashlib import sha256
import json
from math import ceil
from pathlib import Path
from statistics import median

from .run_onboard_deadline_route import AMPLITUDES, GATES, PORT, STEP_MPS, gate3, run_case, scenario
from .run_onboard_reserved_recovery import state_match
from .run_onboard_stage_recovery import RESERVE_MS, TARGET
from .run_onboard_value import save
from .run_ps_onboard_pair import BOARD_PACKAGE


COUNTS = (17, 19)


def queue_visibility(source):
    api, requests = source/'api.csv', source/'trial'/'requests.jsonl'
    with api.open(newline='', encoding='utf-8') as f:
        calls = sorted((int(r['begin_ns']), int(r['end_ns'])) for r in csv.DictReader(f) if int(r['status']) == 0)
    starts = [a for a, _ in calls]
    lead, matched_invocations = [], set()
    with requests.open(encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            enqueue, submit, returned = (int(row.get(k, 0)) for k in
                                         ('dpu_enqueue_us', 'dpu_submit_us', 'dpu_local_return_us'))
            if not 0 < enqueue <= submit <= returned:
                continue
            first = bisect_left(starts, submit*1000)
            if bisect_right(starts, returned*1000) == first+1 and calls[first][1] <= (returned+1)*1000:
                if first in matched_invocations:
                    raise ValueError('DNNDK invocation attributed to more than one request')
                matched_invocations.add(first)
                lead.append((calls[first][0]-enqueue*1000)/1e6)
    lead.sort()
    if not lead:
        raise ValueError('No requests matched a DNNDK interval')
    return dict(source=str(source.resolve()), hashes={p.name: sha256(p.read_bytes()).hexdigest() for p in (api, requests)},
                invocations=len(calls), matched=len(lead), lead_ms={f'p{p}': lead[ceil(p*len(lead)/100)-1]
                                                                 for p in (50, 95, 99, 100)},
                at_least_ms={ms: sum(x >= ms for x in lead) for ms in (10, 30, 75, 300)})


def audit_row(report):
    first, second = report['windows']
    bg = second['background']
    anchor = bg[0][0]
    assert len(first['background']) == COUNTS[0] and len(bg) == COUNTS[1]
    return dict(on_time=[r['window'] for r in sorted(report['adopted'].values(), key=lambda r: r['window']) if r['on_time']],
                result=report['pass_gates'], contacts=report['contacts'], gate3_m=gate3(report),
                deferred=second['deferred_background_calls'], status=[r['status'] for r in report['requests']],
                tail_finish_ms=[(b-anchor)/1000 for _, b in bg[-2:]],
                background_jobs=report['background_calls'])


def run(args):
    with scenario(PORT, args.output, amplitudes=AMPLITUDES, distances=GATES) as (client, gates):
        save(args.output/'design.json', dict(windows_ms=(0, 75), background_counts=COUNTS, perturbation_mps=STEP_MPS,
                                            target=TARGET, repetitions=4, reserve_ms=RESERVE_MS))
        pairs = {}
        for rep in range(1, 5):
            order = ('fifo', 'protected') if rep % 2 else ('protected', 'fifo')
            reports = {arm: run_case(args, client, gates, f'r{rep}_{arm}', policy='fixed', calls=COUNTS[0],
                reserve=RESERVE_MS, target=TARGET, recovery_schedule=dict(windows=2, healthy_error_m=[],
                    fixed_recipients=(TARGET, TARGET), second_background_calls=COUNTS[1],
                    protect_followup=arm == 'protected')) for arm in order}
            rows = {k: audit_row(r) for k, r in reports.items()}
            steps = [r['velocity_step'] for r in reports.values()]
            comparable = (state_match(reports['fifo'], reports['protected'])
                          and abs(steps[0]['x_m']-steps[1]['x_m']) <= .022
                          and all(abs(s['requested_delta_mps']-s['observed_delta_mps']) < .05 for s in steps)
                          and all(0 in r['on_time'] for r in rows.values()))
            pairs[rep] = dict(comparable=comparable, arms=rows,
                added_tail_ms=[after-before for after, before in zip(rows['protected']['tail_finish_ms'],
                                                                     rows['fifo']['tail_finish_ms'])])
            save(args.output/'pairs.json', pairs)
        valid = [p for p in pairs.values() if p['comparable']]
        save(args.output/'summary.json', dict(comparable=len(valid),
            results={k: dict(followup_adopted=sum(1 in p['arms'][k]['on_time'] for p in valid),
                             team_passes=sum(p['arms'][k]['result'] for p in valid)) for k in order},
            median_added_tail_ms=median(d for p in valid for d in p['added_tail_ms']) if valid else None))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    arm = parser.add_mutually_exclusive_group(required=True)
    arm.add_argument('--binary')
    arm.add_argument('--queue-profile', type=Path, nargs='+')
    parser.add_argument('--package', default=BOARD_PACKAGE)
    parser.add_argument('--windows-ip', default='192.168.77.1')
    parser.add_argument('--board', default='axu3egb')
    parser.add_argument('--ssh-hop', default='vitis-vm')
    args = parser.parse_args()
    if args.queue_profile:
        args.output.mkdir(parents=True)
        save(args.output/'queue_visibility.json', [queue_visibility(p) for p in args.queue_profile])
    else:
        run(args)
