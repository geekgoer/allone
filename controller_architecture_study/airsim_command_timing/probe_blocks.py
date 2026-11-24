"""Measure tagged command adoption in the isolated local Blocks diagnostic."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import secrets
import socket
from statistics import median
from time import perf_counter_ns, sleep

import msgpack


class LocalRpc:
    def __init__(self, port):
        self.socket = socket.create_connection(('127.0.0.1', port), timeout=5)
        self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.reader = msgpack.Unpacker(raw=False, max_buffer_size=2**24)
        self.next_id = 0
        self.responses = {}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.socket.close()

    def send(self, method, *args):
        self.next_id += 1
        self.socket.sendall(msgpack.packb([0, self.next_id, method, args], use_bin_type=True))
        return self.next_id

    def result(self, call_id):
        while call_id not in self.responses:
            chunk = self.socket.recv(65536)
            if not chunk:
                raise ConnectionError('Blocks closed the RPC stream')
            self.reader.feed(chunk)
            for kind, request_id, error, result in self.reader:
                if kind != 1:
                    raise ValueError(f'Unexpected RPC message type {kind}')
                self.responses[request_id] = (error, result)
        error, result = self.responses.pop(call_id)
        if error is not None:
            raise RuntimeError(f'RPC {call_id} failed: {error}')
        return result

    def call(self, method, *args):
        return self.result(self.send(method, *args))


def clock_bounds(rpc):
    windows = []
    for _ in range(24):
        start = perf_counter_ns()
        server = rpc.call('falconTraceClock')
        end = perf_counter_ns()
        windows.append((server - end, server - start))
    return max(a for a, _ in windows), min(b for _, b in windows)


def collect_vehicle(port, name, index, fleet, base, origin, count, period_ns, paired):
    with LocalRpc(port) as rpc:
        rows = []
        for cycle in range(count):
            release = origin + cycle * period_ns + index * period_ns // fleet
            wait = release - perf_counter_ns()
            if wait > 0:
                sleep(wait / 1e9)
            if paired:
                scenario = ('normal', 'before_goal', 'after_ping')[(cycle + cycle // 3) % 3]
            else:
                scenario = 'natural'
            trace_id = base + cycle * fleet + index
            sample = perf_counter_ns()
            row = dict(id=trace_id, vehicle=name, cycle=cycle, scenario=scenario, release_ns=release, sample_ns=sample)
            rows.append(row)
            if not paired and sample >= release + period_ns:
                row['source_status'] = 'skipped_stale_release'
                continue
            rpc.call('getMultirotorState', name)
            state = perf_counter_ns()
            rpc.call('simGetCollisionInfo', name)
            submit = perf_counter_ns()
            row['state_ns'] = state
            if not paired and submit >= release + period_ns:
                row['source_status'] = 'skipped_stale_preparation'
                continue
            delay = 40 if scenario == 'before_goal' else 0
            move = rpc.send('falconTraceMove', trace_id, 0.0, 0.0, 0.0, 0.05, 0.012, delay, name)
            ping_ok = rpc.call('ping')
            ping = perf_counter_ns()
            if not ping_ok:
                raise RuntimeError('Blocks ping failed')
            if scenario == 'after_ping':
                sleep(0.04)
            observed = perf_counter_ns()
            row.update(submit_ns=submit, ping_ns=ping, observed_ns=observed,
                       move_request_id=move, source_status='submitted')
        # Movement-call acknowledgments can mean duration elapsed or superseded.
        # They are audited after the measured enqueue+ping loop.
        for row in rows:
            if 'move_request_id' in row:
                row['move_completed'] = rpc.result(row.pop('move_request_id'))
        return rows


def join_traces(rows, records, offset):
    low, high = offset
    traces = {record[0]: record[1:] for record in records}
    if len(traces) != len(records) or low > high:
        raise ValueError('Duplicate trace ID or inconsistent clock calibration')

    def relation(start, end, cutoff):
        if end <= cutoff + low:
            return 'on_time'
        return 'late' if start > cutoff + high else 'straddles'

    for row in rows:
        stamps = traces.get(row['id'])
        if 'submit_ns' not in row:
            if stamps is not None:
                raise ValueError(f'Unsent command has a trace for {row["id"]}')
            row['adoption'] = row['adoption_from_release'] = row['source_status']
            continue
        if not stamps:
            row['adoption'] = 'missing_rpc_trace'
            row['adoption_from_release'] = 'missing_rpc_trace'
            continue
        received, goal, start, end = stamps
        if not ((goal == 0 or received <= goal) and ((start == end == 0) or received <= goal <= start <= end)
                and received >= row['submit_ns'] + low):
            raise ValueError(f'Impossible server trace for command {row["id"]}')
        row.update(server_received_ns=received, goal_ns=goal, motor_start_ns=start, motor_end_ns=end)
        if not end:
            row['adoption'] = 'goal_only' if goal else 'rpc_only'
            row['adoption_from_release'] = row['adoption']
        else:
            row['adoption'] = relation(start, end, row['submit_ns'] + 20_000_000)
            row['adoption_from_release'] = relation(start, end, row['release_ns'] + 20_000_000)
            row['adoption_vs_ping'] = relation(start, end, row['ping_ns'])
    return rows


def summarize(rows):
    output = {}
    for scenario in sorted({r['scenario'] for r in rows}):
        group = [r for r in rows if r['scenario'] == scenario]
        sent = [r for r in group if 'ping_ns' in r]
        adopted = [r for r in group if r.get('motor_end_ns')]
        pings = [(r['ping_ns'] - r['submit_ns']) / 1e6 for r in sent]
        state = [(r['state_ns'] - r['sample_ns']) / 1e6 for r in group if 'state_ns' in r]
        slow_ping = [r for r, ms in zip(sent, pings) if ms > 20]
        output[scenario] = dict(count=len(group), submitted=len(sent), adoption=Counter(r['adoption'] for r in group),
                                adoption_from_release=Counter(r['adoption_from_release'] for r in group),
                                ping_ms_median=median(pings) if pings else None,
                                ping_ms_max=max(pings, default=None), state_ms_max=max(state, default=None),
                                ping_over_20=len(slow_ping),
                                adoption_on_time_despite_slow_ping=sum(r['adoption'] == 'on_time' for r in slow_ping),
                                adoption_late_despite_fast_ping=sum(r['adoption'] == 'late' for r, ms in zip(sent, pings) if ms <= 20),
                                observed_over_20=sum(r['observed_ns'] - r['submit_ns'] > 20e6 for r in sent),
                                tick_ms_after_server_receive_median=median(
                                    (r['motor_end_ns'] - r['server_received_ns']) / 1e6 for r in adopted) if adopted else None)
    return output


def run(port, fleet, seconds, paired):
    period_ns = 200_000_000 if paired else 20_000_000
    count = 36 if paired else int(seconds * 50)
    if port == 41451 or not 1 <= fleet <= 12 or not 1 <= count * fleet <= 30000 or (paired and fleet != 1):
        raise ValueError('Use a diagnostic port, one vehicle for paired mode, and at most 30000 commands')
    vehicles = [f'Drone{i + 1}' for i in range(fleet)]
    base = (secrets.randbits(32) + 1) * 32768
    with LocalRpc(port) as rpc:
        available = rpc.call('listVehicles')
        if any(name not in available for name in vehicles):
            raise ValueError(f'Missing SimpleFlight vehicles, found {available}')
        first = clock_bounds(rpc)
        for name in vehicles:
            rpc.call('enableApiControl', True, name)
        try:
            origin = perf_counter_ns() + 700_000_000
            with ThreadPoolExecutor(max_workers=fleet) as pool:
                jobs = [pool.submit(collect_vehicle, port, name, i, fleet, base, origin, count, period_ns, paired)
                        for i, name in enumerate(vehicles)]
                rows = [row for job in jobs for row in job.result()]
            second = clock_bounds(rpc)
            offset = (max(first[0], second[0]), min(first[1], second[1]))
            records = []
            for start in range(0, count * fleet, 512):
                records.extend(rpc.call('falconTraceRead', list(range(base + start, base + min(start + 512, count * fleet)))))
        finally:
            for name in vehicles:
                rpc.call('enableApiControl', False, name)
    return dict(port=port, stale_policy='skip_before_state_and_submit' if not paired else 'paired',
                offset_server_minus_client_ns=offset, summary=summarize(join_traces(rows, records, offset)), rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=41591)
    parser.add_argument('--vehicles', type=int, default=1)
    parser.add_argument('--seconds', type=float, default=20)
    parser.add_argument('--paired', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output path')
    data = run(args.port, args.vehicles, args.seconds, args.paired)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
    print(json.dumps(data['summary'], indent=2))


if __name__ == '__main__':
    main()
