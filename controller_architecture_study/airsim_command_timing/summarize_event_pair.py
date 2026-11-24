"""Compare independent server adoption with online, deadline-preserving decisions."""

import argparse
from collections import Counter
import json
from pathlib import Path


def summarize(trial):
    def load(name):
        return json.loads((trial / name).read_text(encoding='utf-8'))

    enabled = load('config_source.json')['use_adoption_event']
    stream = load('trace_live_events.json')
    app = load('summary.json')
    origin = load('execution_identity.json')['origin_us']
    received = {event[0]: event for event in stream['events']}
    sent, decision_ids = {}, set()
    outcomes, evidence, effects, attempts = Counter(), Counter(), Counter(), Counter()
    guards = Counter()
    guard_wall_max_us = 0
    changed = []
    with (trial / 'requests.jsonl').open(encoding='utf-8') as source:
        for row in map(json.loads, source):
            outcomes[row['status']] += 1
            if 'command_trace_id' in row:
                sent[row['command_trace_id']] = row
            if 'command_retime_attempt' in row:
                attempt = row['command_retime_attempt']
                attempts[str(attempt['accepted'])] += 1
                if 'guard' in attempt:
                    guards[attempt['guard']['status']] += 1
                    guard_wall_max_us = max(guard_wall_max_us, attempt['wall_us'])
            decision = row.get('prior_adoption_decision')
            if decision is None:
                continue
            prior = sent[decision['id']]
            if row['vehicle'] != prior['vehicle'] or decision['cycle'] != prior['cycle'] or decision['id'] in decision_ids:
                raise ValueError('Adoption decision has the wrong command version')
            decision_ids.add(decision['id'])
            evidence[decision['evidence']] += 1
            effects[f"{decision['old_trigger']}/{decision['new_trigger']}"] += 1
            if decision['old_trigger'] != decision['new_trigger']:
                changed.append(dict(vehicle=row['vehicle'], next_cycle=row['cycle'], **decision))
    before_reply = sum(event[3] <= sent[id]['command_return_us'] * 1000
                       for id, event in received.items() if 'command_return_us' in sent.get(id, {}))
    return dict(enabled=enabled, stream_failed=stream['failed'], outcomes=outcomes,
                tagged=len(sent), received=len(received.keys() & sent.keys()), received_by_rpc_reply=before_reply,
                next_cycle_decisions=len(decision_ids), evidence=evidence, trigger_changes=effects, changed=changed,
                retime_attempts=attempts, retime_guard_counts=guards, retime_guard_wall_max_us=guard_wall_max_us,
                measured_s=(app['pause_start_us'] - origin) / 1e6,
                retired={v['name']: v['reason'] for v in app['vehicles'] if v['retired']})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trial', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    report = summarize(args.trial)
    with args.output.open('x', encoding='utf-8') as out:
        json.dump(report, out, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
