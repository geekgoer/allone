"""Reprice the first peer that blocked a release-preserving retime."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path

from ..airsim_command_timing.audit_ps_trace import bounds, relation
from .critical_update_screen import load, rows


def prep_stages(row):
    keys = ('sample_start_us', 'state_ready_us', 'collision_ready_us',
            'ps_lock_requested_us', 'ps_lock_acquired_us', 'feature_ready_us')
    stamps = [row[key] for key in keys]
    elapsed = [(end - start) / 1000 for start, end in zip(stamps, stamps[1:])]
    if min(elapsed) < 0:
        raise ValueError(f'Out-of-order preparation timestamps for {row["vehicle"]}:{row["cycle"]}')
    return dict(zip(('state_path', 'collision_path', 'before_lock', 'lock_wait', 'lock_held'), elapsed))


def adoption_evidence(command, decision_row, events, clocks):
    decision = decision_row['prior_adoption_decision']
    result = dict(cycle=command['cycle'], trace_id=decision['id'], online=decision['evidence'],
                  used_reserve_ms=decision['used_reserve_ms'], ack_reserve_ms=decision['old_reserve_ms'],
                  ping_wall_ms=(command['command_return_us'] - command['command_ping_start_us']) / 1000)
    event = events.get(decision['id'])
    if event is not None:
        start, end, received = event
        converted = command['control_converted_us'] * 1000
        low, high = bounds(clocks, converted, 200)
        cutoff = math.ceil(command['command_admission']['deadline_us'] * 1000)
        result.update(adoption_from_conversion_ms=[(start - high - converted) / 1e6,
                                                   (end - low - converted) / 1e6],
                      receipt_from_next_sample_ms=(received - decision_row['sample_start_us'] * 1000) / 1e6,
                      receipt_ns=received, versus_deadline=relation(start, end, cutoff, clocks, 200))
    return result


def audit(trial):
    config = load(trial / 'config_source.json')
    if config['adaptive'] or not config['retime_pin_release'] or not config['isolate_command_costs']:
        raise ValueError('Expected the fixed-model, isolated-cost pin-release experiment')
    log = list(rows(trial))
    commands = {r['command_trace_id']: r for r in log if 'command_trace_id' in r}
    origin = load(trial / 'execution_identity.json')['origin_us']
    clocks = load(trial / 'trace_clock_before.json') + load(trial / 'trace_clock_after.json')
    captured = load(trial / 'trace_live_events.json')['events']
    events = {id_: (start, end, received) for id_, start, end, received in captured}
    preparation = [(r, prep_stages(r)) for r in log if 'dpu_enqueue_us' in r and 'feature_ready_us' in r]
    service = [(r, r['calendar_finished']) for r in log if r['model_id'] == config['model_id']
               and 'board_done_ms' in r.get('calendar_finished', {})]
    service_floor = next(m['service_ms'] for m in config['modes'] if m['numeric_id'] == config['model_id'])
    retirements = {r['vehicle']: r['detected_ms'] * 1000 for r in load(trial / 'hold_expiries.json')}
    if any('command_retime' in r for r in log):
        raise ValueError('Fleet count assumes each live loop kept its initial published period')
    periods = {v: config.get('initial_periods', {}).get(v, config['period_ms']) for v in config['names']}
    post_floor = max(config['post_ms'], config['command_reserve_ms'])
    delivery = []
    for r in log:
        if 'prior_adoption_decision' not in r or 'sample_start_us' not in r:
            continue
        decision = r['prior_adoption_decision']
        prior = commands[decision['id']]
        if (r['vehicle'], decision['cycle']) != (prior['vehicle'], prior['cycle']):
            raise ValueError('Adoption decision refers to the wrong vehicle or cycle')
        cost = decision['used_reserve_ms'] + (prior['control_converted_us'] - prior['output_ready_us']) / 1000
        delivery.append((r, cost, prior))

    results = []
    for attempt in sorted(log, key=lambda r: r.get('slot_return_us', 0)):
        guard = attempt.get('command_retime_attempt', {}).get('guard', {})
        if guard.get('status') != 'peer_path_exceeds_slot':
            continue
        peer, now = guard['peer'], attempt['slot_return_us']
        # At enqueue the preparatory observation has already updated Costs.
        known = [(r, stages) for r, stages in preparation if r['dpu_enqueue_us'] <= now]
        peak, peak_stages = max(known, key=lambda item: sum(item[1].values()))
        local = [stages for r, stages in known if r['vehicle'] == peer]
        global_prep = max(config['prep_ms'], sum(peak_stages.values()))
        private_prep = max(config['prep_ms'], max((sum(x.values()) for x in local), default=0))
        private_path = max((x['state_path'] + x['collision_path'] + x['before_lock'] for x in local), default=0)
        shared_critical = max(x['lock_held'] for _, x in known)
        split_prep = max(config['prep_ms'], private_path + shared_critical)
        dpu = max(service_floor, max((s['board_done_ms'] - s['board_start_ms'] for _, s in service
                                     if s['board_done_ms'] * 1000 <= now), default=0))
        by_vehicle = {v: (None, post_floor, None) for v in periods if retirements.get(v, float('inf')) > now}
        for r, cost, command in delivery:
            v = r['vehicle']
            if v in by_vehicle and r['sample_start_us'] <= now and cost > by_vehicle[v][1]:
                by_vehicle[v] = (r, cost, command)
        evidence_row, post, prior = by_vehicle[peer]
        original = global_prep + dpu + post
        # The recorded guard value checks that intervening updates between our
        # earliest cutoff and the actual lock acquisition did not change the sum.
        if not math.isclose(original, guard['peer_path_ms'], rel_tol=0, abs_tol=0.001):
            raise ValueError(f'Cannot reproduce peer cost for {attempt["vehicle"]}:{attempt["cycle"]}')
        window = min(guard['peer_period_ms'], config['fresh_ms'])  # No extra hold/stop clipping.
        alternatives = dict(global_prep=original, per_vehicle_prep=private_prep + dpu + post,
                            split_no_queue=split_prep + dpu + post, no_prep=dpu + post, post_only=post)
        results.append(dict(target=attempt['vehicle'], target_cycle=attempt['cycle'],
                            target_time_ms=(now - origin) / 1000, peer=peer, window_ms=window,
                            original_recorded_ms=guard['peer_path_ms'], costs_ms=alternatives,
                            still_blocks={name: cost > window + 1e-8 for name, cost in alternatives.items()},
                            prep_ms=dict(global_max=global_prep, per_vehicle_max=private_prep, split=split_prep,
                                         peer_private_path=private_path, shared_critical=shared_critical),
                            prep_peak=dict(vehicle=peak['vehicle'], cycle=peak['cycle'], stages_ms=peak_stages),
                            dpu_ms=dpu, post_ms=post,
                            fleet_post_over_period={v: dict(post_ms=cost, period_ms=periods[v],
                                source=adoption_evidence(command, r, events, clocks) if r is not None else None)
                                for v, (r, cost, command) in by_vehicle.items() if cost > periods[v]},
                            post_source=adoption_evidence(prior, evidence_row, events, clocks)
                            if evidence_row is not None else None))
    if not results:
        raise ValueError('No recorded peer-slot rejections to audit')
    counts = Counter(name for r in results for name, blocked in r['still_blocks'].items() if blocked)
    return dict(scope='Offline first-blocking-peer optimistic slot screen; no shared-queue or flight forecast',
                attempts_reconstructed=len(results), still_blocks={name: counts[name] for name in results[0]['costs_ms']},
                by_peer=Counter(r['peer'] for r in results),
                fleet_post_over_period_counts=Counter(len(r['fleet_post_over_period']) for r in results), attempts=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.trial)
    with args.output.open('x', encoding='utf-8') as out:
        json.dump(result, out, indent=2, allow_nan=False)
    print(json.dumps({key: value for key, value in result.items() if key != 'attempts'}, indent=2))


if __name__ == '__main__':
    main()
