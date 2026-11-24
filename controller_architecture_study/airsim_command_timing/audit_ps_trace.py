"""Join PS command logs to Blocks' firmware events by command ID after the run."""

import argparse
from collections import Counter
import json
from pathlib import Path

from .probe_blocks import LocalRpc


def bounds(samples, point, drift_ppm):
    # Each clock RPC occurred somewhere in the PS send/receive interval.
    # Widen that observation by the user's relative clock-rate assumption.
    low, high = None, None
    for send, server, receive in samples:
        if send > receive:
            raise ValueError('Invalid clock sample')
        delta = max(abs(point - send), abs(point - receive))
        allowance = (delta * drift_ppm + 999_999) // 1_000_000
        left, right = server - receive - allowance, server - send + allowance
        low = left if low is None else max(low, left)
        high = right if high is None else min(high, right)
    if low is None or low > high:
        raise ValueError('Inconsistent clock samples under the specified drift assumption')
    return low, high


def relation(start, finish, cutoff, samples, drift):
    low, high = bounds(samples, cutoff, drift)
    if finish <= cutoff + low:
        return 'by_cutoff'
    if start > cutoff + high:
        return 'after_cutoff'
    return 'uncertain'


def audit(rows, records, samples, drift):
    traces = {r[0]: r[1:] for r in records}
    ids = {r['command_trace_id'] for r in rows}
    if len(traces) != len(records) or len(ids) != len(rows) or not traces.keys() <= ids:
        raise ValueError('Unexpected or duplicate command ID')
    counts = Counter()
    details = []
    for row in rows:
        trace_id = row['command_trace_id']
        record = traces.get(trace_id)
        detail = {k: row[k] for k in ('vehicle', 'cycle', 'status', 'command_trace_id')}
        details.append(detail)
        if not record:
            counts['no_server_trace'] += 1
            continue
        received, goal, start, finish = record
        if not received or (goal and goal < received) or ((start or finish) and not 0 < received <= goal <= start <= finish):
            raise ValueError(f'Invalid server event order for {trace_id}')
        detail['server_ns'] = dict(received=received, goal=goal, tick_start=start, tick_end=finish)
        if not finish:
            counts['goal_only' if goal else 'rpc_only'] += 1
            continue
        if 'command_ping_start_us' not in row or 'command_return_us' not in row:
            counts['adopted_no_ping_record'] += 1
            continue
        actual_ping = row.get('probe_real_ping_return_us', row['command_return_us'])
        submit = row['command_submit_us'] * 1000
        low, high = bounds(samples, submit, drift)
        detail['submit_to_tick_interval_ms'] = [(start - submit - high) / 1e6, (finish - submit - low) / 1e6]
        detail['ping_ms'] = (actual_ping - row['command_ping_start_us']) / 1000
        detail['recorded_ack_delay_after_ping_ms'] = (row['command_return_us'] - actual_ping) / 1000
        detail['versus_ping'] = relation(start, finish, actual_ping * 1000, samples, drift)
        cutoff = row.get('command_admission', {}).get('deadline_us', row.get('actuation_deadline_us'))
        if cutoff is not None:
            detail['versus_deadline'] = relation(start, finish, int(cutoff * 1000), samples, drift)
        counts['adopted_' + detail['versus_ping'] + '_ping'] += 1
    long_ping = [d for d in details if d.get('ping_ms', 0) >= 20]
    by_deadline = Counter(d['versus_deadline'] for d in details if 'versus_deadline' in d)
    disagreements = Counter((d['status'], d['versus_deadline']) for d in details if 'versus_deadline' in d)
    return dict(attempted=len(rows), outcomes=counts, actual_by_deadline=by_deadline,
                ping_status_vs_deadline={f'{status}/{verdict}': n for (status, verdict), n in disagreements.items()},
                slow_ping_ms=[d for d in long_ping], measurements=details)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, default=41591)
    parser.add_argument('--sim-address', default='192.168.77.1')
    parser.add_argument('--drift-ppm', type=int, default=200)
    args = parser.parse_args()
    if args.port == 41451 or args.drift_ppm < 0 or args.output.exists():
        parser.error('Use a diagnostic port, nonnegative drift, and new output path')
    with (args.trial / 'requests.jsonl').open(encoding='utf-8') as stream:
        rows = [r for r in map(json.loads, stream) if 'command_trace_id' in r]
    if not rows:
        raise ValueError('No tagged PS requests')
    samples = []
    for label in ('before', 'after'):
        samples += json.loads((args.trial / f'trace_clock_{label}.json').read_text(encoding='utf-8'))
    records = []
    with LocalRpc(args.port, args.sim_address) as rpc:
        for start in range(0, len(rows), 512):
            records += rpc.call('falconTraceRead', [r['command_trace_id'] for r in rows[start:start + 512]])
    result = audit(rows, records, samples, args.drift_ppm)
    with args.output.open('x', encoding='utf-8') as out:
        json.dump(dict(drift_ppm_assumed=args.drift_ppm, server_records=records, **result), out, indent=2)
    print(json.dumps(dict(attempted=result['attempted'], outcomes=result['outcomes'],
                          slow_pings=len(result['slow_ping_ms'])), indent=2))


if __name__ == '__main__':
    main()
