"""Compare the first retime trigger on one frozen PS/Blocks timeline."""

import argparse
from collections import Counter
import json
from pathlib import Path

from .audit_ps_trace import bounds, relation


def possible_trigger(start, end, deadline, reserve_limit, samples, drift):
    """Bound the recorded trigger: completion >= deadline and completion > reserve_limit."""
    deadline_low, deadline_high = bounds(samples, deadline, drift)
    reserve_low, reserve_high = bounds(samples, reserve_limit, drift)
    if end < deadline + deadline_low or end <= reserve_limit + reserve_low:
        return 'no'
    if start >= deadline + deadline_high and start > reserve_limit + reserve_high:
        return 'yes'
    return 'uncertain'


def compare(rows, captured, samples, enabled):
    drift = captured['drift_ppm_assumed']
    records = {trace['command_trace_id']: trace for trace in captured['measurements']}
    events = {}
    attempts = []
    counts = Counter()
    untagged_status = Counter()
    slow = []
    dispositions = Counter()
    for row in rows:
        key = row['vehicle'], row['cycle']
        if 'command_trace_id' not in row:
            untagged_status[row['status']] += 1
            continue
        measurement = records[row['command_trace_id']]
        detail = dict(vehicle=key[0], cycle=key[1], status=row['status'],
                      ping_ms=measurement.get('ping_ms'), ping_trigger=None, adoption_trigger='unknown',
                      adoption_by_deadline=measurement.get('versus_deadline', 'unknown'))
        events[key] = detail
        stamp = measurement.get('server_ns', {})
        if 'new_held_version' in row and row.get('command_admission', {}).get('accepted'):
            deadline = round(row['command_admission']['deadline_us'] * 1000)
            reserve_limit = row['control_converted_us'] * 1000 + round(row['period_ms'] * 1e6)
            ack = row['command_return_us'] * 1000
            detail['ping_trigger'] = ack >= deadline and ack > reserve_limit
            if stamp.get('tick_end'):
                detail['adoption_trigger'] = possible_trigger(stamp['tick_start'], stamp['tick_end'], deadline,
                                                              reserve_limit, samples, drift)
                detail['submit_to_tick_interval_ms'] = measurement['submit_to_tick_interval_ms']
            counts[f"{detail['ping_trigger']}/{detail['adoption_trigger']}"] += 1
            if detail['ping_trigger'] or detail['adoption_by_deadline'] == 'after_cutoff':
                if stamp.get('received'):
                    detail['rpc_vs_deadline'] = relation(stamp['received'], stamp['received'], deadline, samples, drift)
                if stamp.get('goal'):
                    detail['receive_to_goal_ms'] = (stamp['goal'] - stamp['received']) / 1e6
                if stamp.get('tick_end'):
                    detail['goal_to_tick_end_ms'] = (stamp['tick_end'] - stamp['goal']) / 1e6
        else:
            counts['missing_board_admission'] += 1
        if detail['ping_ms'] is not None and detail['ping_ms'] >= 20:
            slow.append(detail)
    for row in rows:
        attempt = row.get('command_retime_attempt')
        if attempt is None:
            continue
        key = row['vehicle'], attempt['late_ack_cycle']
        trigger = events[key]
        if not enabled or not trigger['ping_trigger'] or row['status'] != 'dropped_inference':
            raise ValueError(f'Observed attempt does not match the recorded trigger: {key}')
        dispositions[trigger['adoption_trigger']] += 1
        attempts.append(dict(vehicle=key[0], cycle=row['cycle'], first_trigger_cycle=key[1],
                             adoption_trigger=trigger['adoption_trigger'], installed=attempt['accepted']))
    return dict(enabled=enabled, drift_ppm_assumed=drift, tagged=len(events),
                total_rows=len(rows), before_command_status=untagged_status, trigger_comparison=counts,
                slow_ping_count=len(slow), slow_ping=sorted(slow, key=lambda d: -d['ping_ms']),
                trigger_or_late_commands=[d for d in events.values()
                                          if d['ping_trigger'] or d['adoption_by_deadline'] == 'after_cutoff'],
                observed_attempts_by_adoption_trigger=dispositions, observed_attempts=attempts,
                installed=sum(attempt['installed'] for attempt in attempts),
                scope='Single trigger substituted on a frozen timeline. Calendar budgets and subsequent actions are not replayed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with (args.trial / 'requests.jsonl').open(encoding='utf-8') as stream:
        rows = list(map(json.loads, stream))
    captured = json.loads(args.audit.read_text(encoding='utf-8'))
    config = json.loads((args.trial / 'config_source.json').read_text(encoding='utf-8'))
    samples = []
    for label in ('before', 'after'):
        samples += json.loads((args.trial / f'trace_clock_{label}.json').read_text(encoding='utf-8'))
    report = compare(rows, captured, samples, config['enable_command_retime'])
    with args.output.open('x', encoding='utf-8') as out:
        json.dump(report, out, indent=2)
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ('slow_ping', 'observed_attempts', 'trigger_or_late_commands')}, indent=2))


if __name__ == '__main__':
    main()
