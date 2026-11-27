"""Check when one already-running DPU job exhausts a ready control job's slack."""

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
import csv
from hashlib import sha256
import json
from pathlib import Path


def stats(values):
    ordered = sorted(values)
    return {k: ordered[round(p * (len(ordered) - 1))] for k, p in
            (('median', .5), ('p99', .99), ('max', 1.))} if ordered else {}


def live_blocker(starts, jobs, ready):
    i = bisect_left(starts, ready) - 1
    if i >= 0 and ready < jobs[i][1]:
        return jobs[i]
    return None


def effect(ready, end, deadline, cost):
    if ready + cost > deadline:
        return 'already_too_late'
    return 'boundary_too_late' if end + cost > deadline else 'fits'


def correction_windows(study):
    groups = json.loads((study/'groups.json').read_text(encoding='utf-8'))
    design = json.loads((study/'design.json').read_text(encoding='utf-8'))
    duration = round(design['deadline_ms'] * 1000)
    reserve = round(design['reserve_ms'] * 1000)
    rows = []
    for name, group in groups.items():
        if not group['comparable']:
            continue
        for policy, arm in group['arms'].items():
            if not arm['critical']:
                continue
            path = study/f'{name}_{policy}'/'native_ps.json'
            phase = json.loads(path.read_text(encoding='utf-8'))['windows'][1]
            deadline = phase['background_deadline_us']
            # The poll has returned both states. Encoding will still take time.
            observed = deadline - duration + round(phase['sample_ms'] * 1000)
            calls = phase['background']
            active = live_blocker([a for a, _ in calls], calls, observed)
            end = active[1] if active else observed
            rows.append(dict(group=name, policy=policy, actual_timely=arm['followup_on_time'],
                             observed_to_deadline_us=deadline-observed, residual_us=end-observed,
                             reserve_us=reserve, with_reserve=effect(observed, end, deadline, reserve),
                             source_sha256=sha256(path.read_bytes()).hexdigest()))
    return rows


def call_sample(donor, ready, calls, starts):
    submit = donor.get('dpu_submit_us')
    readable = donor.get('dpu_result_readable_us')
    if submit is None or readable is None or not submit <= readable:
        return {}
    first = bisect_left(starts, submit * 1000)
    if bisect_left(starts, (readable + 1) * 1000) != first + 1:
        return {}
    call = calls[first]
    begin, end = call['begin_ns'] / 1000, call['end_ns'] / 1000
    if not begin < end < readable + 1 or call['status'] != 0:
        return {}
    wall = end - begin
    runqueue = call['runqueue_ns'] / 1000
    # The counters bracket the inner call. Remove time outside that call and
    # then any runnable waiting that could have occurred before this arrival.
    outside = max(0., readable - submit - wall) + 1.
    remaining_wait = (max(0., runqueue - outside - max(0., ready + 1. - begin))
                      if 0 <= runqueue <= wall else None)
    return dict(call_wall_us=wall, reported_driver_profile_us=call['driver_profile_us'],
                post_ready_cpu_wait_lower_us=remaining_wait)


def fleet_trace(source):
    requests = source/'trial'/'requests.jsonl'
    api = source/'api.csv'
    identity_path = source/'trial'/'execution_identity.json'
    identity = json.loads(identity_path.read_text(encoding='utf-8'))
    command_reserve = round(identity['live_config']['command_reserve_ms'] * 1000)
    digest = sha256()
    admitted, occupied = [], []
    releases = 0
    with requests.open('rb') as stream:
        for line in stream:
            digest.update(line)
            row = json.loads(line)
            releases += 1
            queued = row.get('calendar_queued') or {}
            if queued.get('phase') == 'queued' and not queued.get('cancelled', False):
                admitted.append(row)
            if row.get('inference_dispatched'):
                start, end = row['dpu_dequeue_us'], row['dpu_local_return_us']
                if end <= start:
                    raise ValueError('Invalid executor interval')
                occupied.append((start, end, row))
    occupied.sort(key=lambda job: job[0])
    if any(a[1] > b[0] for a, b in zip(occupied, occupied[1:])):
        raise ValueError('Overlapping executor intervals on a single DPU')
    starts = [a for a, _, _ in occupied]
    with api.open(newline='', encoding='utf-8') as stream:
        calls = sorted(({k: int(v) for k, v in row.items()} for row in csv.DictReader(stream)),
                       key=lambda call: call['begin_ns'])
    api_starts = [row['begin_ns'] for row in calls]

    fixed, observed, initial_reserve_only = Counter(), Counter(), Counter()
    fixed_witnesses, observed_witnesses, lower_witnesses = [], [], []
    residuals = []
    for row in admitted:
        q, d = row['dpu_enqueue_us'], row['actuation_deadline_us']
        active = live_blocker(starts, occupied, q)
        end = active[1] if active else q
        residuals.append(end - q)
        ack = row.get('command_return_us')
        fixed_effect = effect(q, end, d, 4500)
        fixed[fixed_effect] += 1
        initial_reserve_only[effect(q, end, d, command_reserve)] += 1
        own_start, own_end = row.get('dpu_dequeue_us'), row.get('dpu_local_return_us')
        service = None
        if row.get('inference_dispatched') and ack and q <= own_start < own_end < ack:
            if end > own_start:
                raise ValueError('Matched job starts before blocking job ends')
            service = ack - own_start
            measured_effect = effect(q, end, d, service)
            observed[measured_effect] += 1
        else:
            measured_effect = None
        fixed_hit = fixed_effect == 'boundary_too_late'
        measured_hit = measured_effect == 'boundary_too_late'
        if not (fixed_hit or measured_hit or (active and q <= d < end)):
            continue
        start, end, donor = active
        witness = dict(vehicle=row['vehicle'], cycle=row['cycle'], status=row['status'],
                       blocker_vehicle=donor['vehicle'], blocker_cycle=donor['cycle'],
                       blocker_model=donor['model_id'], blocker_service_us=end-start,
                       budget_us=d-q, residual_us=end-q, own_to_ack_us=service,
                       deadline_sources=[k for k in ('period', 'fresh', 'hold')
                                         if row.get(k+'_deadline_us') == d],
                       dispatch_rejection=(row.get('calendar_dispatch') or {}).get('dispatch_rejection'),
                       **call_sample(donor, q, calls, api_starts))
        if fixed_hit:
            fixed_witnesses.append(witness)
        if measured_hit:
            observed_witnesses.append(witness)
        if q <= d < end:
            lower_witnesses.append(witness)
    return dict(source=str(source.resolve()), requests_sha256=digest.hexdigest(),
                api_sha256=sha256(api.read_bytes()).hexdigest(),
                identity_sha256=sha256(identity_path.read_bytes()).hexdigest(),
                releases=releases, admitted=len(admitted),
                executed=len(occupied), busy_at_ready=sum(r > 0 for r in residuals),
                positive_residual_us=stats([r for r in residuals if r > 0]),
                fixed_budget_us=4500, fixed=fixed, own_to_ack_observed=observed,
                initial_command_reserve_us=command_reserve, initial_reserve_only=initial_reserve_only,
                fixed_witnesses=fixed_witnesses, observed_witnesses=observed_witnesses,
                deadline_before_executor_release=lower_witnesses)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--correction-study', type=Path, required=True)
    parser.add_argument('--fleet-traces', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = dict(correction_windows=correction_windows(args.correction_study),
                  fleet_traces=[fleet_trace(source) for source in args.fleet_traces])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as file:
        json.dump(result, file, indent=2)


if __name__ == '__main__':
    main()
