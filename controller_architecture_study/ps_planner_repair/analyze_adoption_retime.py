"""Audit why genuine late-adoption events failed to retime the next PS job."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

from .critical_update_screen import load, rows


def period_candidates(anchor, old, hold, fresh, stop, periods, budget, release_unchanged=False):
    candidates = []
    for period in sorted(p for p in periods if p > old):
        release = anchor + (old if release_unchanged else period)
        window = min(period, hold - release, fresh, stop - release)
        candidates.append(dict(period_ms=period, release_ms=release, window_ms=window,
                               slack_ms=window - budget, fits=budget < window))
    return candidates


def budgets(log, vehicle_rows, attempt, config, final):
    now = attempt['slot_return_us']
    # A completed row has already called Costs::observe after preparing its input.
    measured = [r for r in log if r['slot_return_us'] <= now and 'feature_ready_us' in r]
    peak = max(measured, key=lambda r: r['feature_ready_us'] - r['sample_start_us'])
    prep = max(config['prep_ms'], (peak['feature_ready_us'] - peak['sample_start_us']) / 1000)
    post, reserve = config['post_ms'], config['command_reserve_ms']
    by_cycle = {r['cycle']: r for r in vehicle_rows}
    for r in vehicle_rows:
        d = r.get('prior_adoption_decision')
        if d is None or r['cycle'] > attempt['cycle']:
            continue
        prior = by_cycle[d['cycle']]
        used = d['used_reserve_ms']
        reserve = max(reserve, used)
        post = max(post, used + (prior['control_converted_us'] - prior['output_ready_us']) / 1000)
    model_id = config['model_id']
    initial = next(m['service_ms'] for m in config['modes'] if m['numeric_id'] == model_id)
    service = initial
    for r in log:
        done = r.get('calendar_finished', {})
        if r['model_id'] == model_id and done.get('board_done_ms', float('inf')) * 1000 <= now:
            service = max(service, done['board_done_ms'] - done['board_start_ms'])
    loop = final['loops'][attempt['vehicle']]
    caps = dict(prep_ms=final['config']['prep_ms'], service_ms=final['modes'][str(model_id)]['service_ms'],
                post_ms=max(loop.get('post_ms', config['post_ms']), loop.get('command_reserve_ms', reserve),
                            final['config']['post_ms'], final['config']['command_reserve_ms']))
    active = dict(prep_ms=prep, service_ms=service, post_ms=max(post, reserve))
    for stage, cost in active.items():
        if cost > caps[stage] + 0.002:
            raise ValueError(f'Observed cost exceeds final running maximum for {stage}')
    return active, caps, dict(vehicle=peak['vehicle'], cycle=peak['cycle'],
                              state_rpc_ms=(peak['state_ready_us'] - peak['sample_start_us']) / 1000,
                              collision_rpc_ms=(peak['collision_ready_us'] - peak['state_ready_us']) / 1000,
                              after_collision_ms=(peak['feature_ready_us'] - peak['collision_ready_us']) / 1000)


def peer_slot_floor(groups, retirements, attempt, config, costs, candidate, stop):
    now = attempt['slot_return_us'] / 1000
    due = candidate['release_ms'] + candidate['window_ms']
    # Use only published periods and the initial shared post budget. This can
    # show a peer already cannot fit alone, without guessing its live hold.
    floor = costs['prep_ms'] + costs['service_ms'] + max(config['post_ms'], config['command_reserve_ms'])
    blocked = []
    for name, sequence in groups.items():
        if name == attempt['vehicle'] or retirements.get(name, float('inf')) <= now:
            continue
        published = [row['calendar_release'] for row in sequence
                     if row['calendar_release']['published_ms'] <= now]
        if not published:
            continue
        last = max(published, key=lambda row: row['cycle'])
        period = last['period_ms']
        release = last['release_ms'] + period
        for row in sequence:
            edit = row.get('command_retime')
            if edit and edit['cycle'] == last['cycle'] + 1 and edit['commit_ms'] <= now:
                period, release = edit['period_ms'], edit['next_release_ms']
        if now < release < due and floor > min(period, config['fresh_ms'], stop - release):
            blocked.append(dict(vehicle=name, next_release_ms=release, period_ms=period))
    return dict(minimum_empirical_path_ms=floor, peer_slot_conflicts=blocked)


def analyze(trial, final_path, include_peers=False):
    config, final = load(trial / 'config_source.json'), load(final_path)
    if not config['use_adoption_event'] or not config['enable_command_retime']:
        raise ValueError('Use the adoption-enabled retime trial')
    stop = final['config']['measurement_stop_ms']
    log = list(rows(trial))
    groups = defaultdict(list)
    for row in log:
        groups[row['vehicle']].append(row)
    retirements = {row['vehicle']: row['detected_ms'] for row in load(trial / 'hold_expiries.json')} if include_peers else {}
    results = []
    for vehicle, sequence in groups.items():
        activations = {d['cycle'] for row in sequence if (d := row.get('prior_adoption_decision'))
                       and d['evidence'] == 'adoption_late' and d['new_trigger'] and not d['old_trigger']}
        by_cycle = {row['cycle']: row for row in sequence}
        for row in sequence:
            attempt = row.get('command_retime_attempt')
            if not attempt or attempt['late_ack_cycle'] not in activations:
                continue
            if attempt['accepted'] or not row.get('observe_accepted'):
                raise ValueError('This audit expects uninstalled attempts with a matching hold observation')
            costs, caps, spike = budgets(log, sequence, row, config, final)
            now = row['slot_return_us'] / 1000
            hold = row['hold_deadline_us'] / 1000
            old = row['period_ms']
            previous = row['calendar_release']['release_ms']
            successor = by_cycle.get(row['cycle'] + 1)
            published = successor is not None and successor['calendar_release']['published_ms'] <= now
            dispatch_rejected = 'dispatch_rejection_evidence' in row.get('calendar_dispatch', {})
            actual = period_candidates(previous, old, hold, config['fresh_ms'], stop, config['periods_ms'], sum(costs.values()))
            no_model = period_candidates(previous, old, hold, config['fresh_ms'], stop, config['periods_ms'],
                                         costs['prep_ms'] + costs['post_ms'])
            pinned = period_candidates(previous, old, hold, config['fresh_ms'], stop, config['periods_ms'],
                                       sum(caps.values()), release_unchanged=True)
            results.append(dict(vehicle=vehicle, cycle=row['cycle'], trigger=attempt['late_ack_cycle'],
                                dispatch_rejection_recorded=dispatch_rejected, successor_already_published=published,
                                hold_left_at_worker_return_ms=hold - now, hold_left_from_anchor_ms=hold - previous,
                                wall_us=attempt['wall_us'], costs_lower_ms=costs, costs_final_cap_ms=caps, prep_peak=spike,
                                current_rule=actual, current_rule_zero_dpu=no_model,
                                keep_successor_release=pinned))
            if include_peers and dispatch_rejected and not published:
                candidate = next((p for p in pinned if p['fits']), None)
                if candidate is not None:
                    results[-1]['peer_slot_screen'] = peer_slot_floor(groups, retirements, row, config, costs, candidate, stop)
    return dict(scope='Offline local screen; lower bounds from prior completed observations; no full group admission',
                attempts=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial', type=Path, required=True)
    parser.add_argument('--calendar', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--include-peers', action='store_true')
    args = parser.parse_args()
    report = analyze(args.trial, args.calendar, args.include_peers)
    with args.output.open('x', encoding='utf-8') as out:
        json.dump(report, out, indent=2, allow_nan=False)
    print(json.dumps(dict(attempts=len(report['attempts']),
                          missing_dispatch=sum(not a['dispatch_rejection_recorded'] for a in report['attempts']),
                          impossible_even_without_dpu=sum(not any(p['fits'] for p in a['current_rule_zero_dpu'])
                                                           for a in report['attempts']))))


if __name__ == '__main__':
    main()
