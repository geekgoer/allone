"""Summarize the live receipt switch and retain each changed trigger's evidence."""

import argparse
from collections import Counter
import json
from pathlib import Path


def summarize(trial):
    def load(name):
        return json.loads((trial / name).read_text(encoding='utf-8'))

    enabled = load('config_source.json')['use_adoption_receipt']
    origin = load('execution_identity.json')['origin_us']
    runtime = load('summary.json')
    outcomes, evidence, trigger_counts = Counter(), Counter(), Counter()
    edits, changes, slow = [], [], []
    commands = 0
    with (trial / 'requests.jsonl').open(encoding='utf-8') as stream:
        for row in map(json.loads, stream):
            outcomes[row['status']] += 1
            if row.get('command_retime_attempt'):
                edits.append(dict(vehicle=row['vehicle'], cycle=row['cycle'], **row['command_retime_attempt']))
            proof = row.get('receipt_policy')
            if proof is None:
                continue
            if proof['enabled'] != enabled:
                raise ValueError('Receipt switch changed within the run')
            commands += 1
            ack = row['command_return_us']
            converted = row['control_converted_us']
            point = row['command_admission']['deadline_us']
            legacy = ack >= point and (ack - converted) > row['period_ms'] * 1000
            evidence[proof['evidence']] += 1
            trigger_counts[f'{legacy}/{proof["trigger"]}'] += 1
            if legacy and not proof['trigger']:
                changes.append(dict(vehicle=row['vehicle'], cycle=row['cycle'],
                                    prior_cost_ms=(ack - converted) / 1000,
                                    new_cost_ms=(proof['budget_end_ns'] - 1000 * converted) / 1e6))
            if ack - row['command_ping_start_us'] >= 20000:
                slow.append(dict(vehicle=row['vehicle'], cycle=row['cycle'], since_origin_ms=(ack - origin) / 1000,
                                 rpc_reply_ms=(ack - row['command_ping_start_us']) / 1000,
                                 evidence=proof['evidence'], old_trigger=legacy, live_trigger=proof['trigger']))
    return dict(receipt_enabled=enabled, outcomes=outcomes, commands=commands, evidence=evidence, triggers=trigger_counts,
                slow_rpcs=slow, retime_attempts=edits, installed=sum(edit['accepted'] for edit in edits),
                trigger_suppressions=changes, measured_s=(runtime['pause_start_us'] - origin) / 1e6,
                retired={v['name']: v['reason'] for v in runtime['vehicles'] if v['retired']})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trial', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    data = summarize(args.trial)
    with args.output.open('x', encoding='utf-8') as output:
        json.dump(data, output, indent=2)
    print(json.dumps({k: v for k, v in data.items() if k not in ('retime_attempts', 'slow_rpcs')}, indent=2))


if __name__ == '__main__':
    main()
