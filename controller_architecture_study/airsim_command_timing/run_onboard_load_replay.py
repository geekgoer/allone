"""Replay historical DPU slot availability during paired AirSim flights."""

import argparse
from bisect import bisect_right
import csv
from functools import partial
from hashlib import sha256
from itertools import cycle
import json
from pathlib import Path

from config import load_calibration
from ..ip37_shadow import make_client
from .run_onboard_dpu import verify_model
from .run_onboard_fleet_value import gate_choice, run_trial, trial_margins
from .run_onboard_pilot import scenario
from .run_onboard_stagger_validation import LEADING, ROUTES, frozen_selector
from .run_onboard_value import PORT, SLOT_S, SLOT_START_S, SLOTS, allocate, read, save


STAGGER = .90
WINDOW_STARTS_S = (1, 17)
POLICIES = ('round_robin', 'fixed_table', 'predicted_position', 'live_position')


def slot_calls(intervals, origin_ns):
    starts = [start for start, _ in intervals]
    slots = []
    for seq in range(1, SLOTS+1):
        stamp = origin_ns + round((SLOT_START_S+(seq-1)*SLOT_S)*1e9)
        index = bisect_right(starts, stamp)-1
        call = intervals[index] if index >= 0 and stamp < intervals[index][1] else None
        slots.append(dict(slot=seq, source_probe_ns=stamp, covering_call_ns=call))
    return slots


def load_windows(source):
    calls = source / 'api.csv'
    requests = source / 'trial' / 'requests.jsonl'
    with calls.open(newline='', encoding='utf-8') as f:
        intervals = sorted((int(r['begin_ns']), int(r['end_ns'])) for r in csv.DictReader(f))
    with requests.open(encoding='utf-8') as f:
        events = (json.loads(line) for line in f)
        first_release = min(r['calendar_release']['release_ms'] for r in events if r.get('calendar_release'))
    if (any(a >= b for a, b in intervals) or
            any(end > start for (_, end), (start, _) in zip(intervals, intervals[1:]))):
        raise ValueError('Expected disjoint, nonempty DNNDK invocation intervals')
    origin = round(first_release*1e6)
    windows = {}
    for second in WINDOW_STARTS_S:
        slots = slot_calls(intervals, origin+second*1_000_000_000)
        if not intervals[0][0] <= slots[0]['source_probe_ns'] <= slots[-1]['source_probe_ns'] < intervals[-1][1]:
            raise ValueError('Replay slot outside the recorded invocation timespan')
        windows[f'w{second:02}'] = slots
    return dict(source=str(source.resolve()), first_release_ns=origin, invocation_count=len(intervals),
                source_sha256={p.name: sha256(p.read_bytes()).hexdigest() for p in (calls, requests)},
                windows=windows)


def predicted_selector(plan, choose, seq, positions, last):
    predicted = plan[seq-1]['nominal_x_m']
    return choose(seq, {v: predicted[v] for v in positions}, last)


def run(args):
    args.output.mkdir(parents=True)
    load = load_windows(args.trace_source)
    save(args.output / 'load_masks.json', load)
    frozen = {name: read(args.plans_from / name / 'stagger_plans.json') for name in ROUTES}
    save(args.output / 'departure_plans.json', frozen)
    board, manifest = make_client(args.bundle, args.board, args.board_port, 35.)
    summary = {}
    try:
        identity = verify_model(board, manifest, args.bundle)
        save(args.output / 'deployment.json', identity)
        calibration = load_calibration()
        for route_index, (route, (amplitudes, distances)) in enumerate(ROUTES.items()):
            prior = read(args.plans_from / route / 'deployment.json')
            plan_data = frozen[route]
            if (prior != identity or plan_data['leading'] != LEADING or plan_data['amplitudes'] != amplitudes or
                    plan_data['gate_x_m'] != list(distances)):
                raise ValueError('Frozen plans must match the model and route')
            choose = partial(gate_choice, gains=plan_data['gains_m'], baseline=plan_data['baseline_m'],
                             distances=distances, phases=plan_data['probe_x_m'])
            plan = plan_data['schedules'][str(STAGGER)]
            with scenario(PORT, args.output / route, amplitudes=amplitudes, distances=distances) as (client, gates):
                for window_index, (window, slots) in enumerate(load['windows'].items()):
                    occupied = {r['slot'] for r in slots if r['covering_call_ns']}
                    available = [r['slot'] for r in slots if r['slot'] not in occupied]
                    shift = 2*route_index+window_index
                    order = POLICIES[shift:]+POLICIES[:shift]
                    for rep, policies in enumerate((order, order[::-1]), 1):
                        for policy in policies:
                            turns = cycle(amplitudes)
                            selectors = {'round_robin': lambda seq, pos, last: (next(turns), dict.fromkeys(pos, 0.)),
                                         'fixed_table': partial(frozen_selector, plan),
                                         'predicted_position': partial(predicted_selector, plan, choose),
                                         'live_position': choose}
                            path = args.output / route / f'{window}_r{rep}_{policy}'
                            feed = partial(allocate, board, calibration, None, selector=selectors[policy],
                                           amplitudes=amplitudes, occupied=occupied)
                            report = run_trial(client, gates, path, feed, LEADING, STAGGER, amplitudes=amplitudes)
                            requests = read(path / 'board_requests.json')
                            traces = [read(path / f'{v}.json') for v in (LEADING, 'Drone1')]
                            summary[f'{route}/{path.name}'] = dict(
                                route=route, window=window, repetition=rep, policy=policy,
                                observed_stagger_s=(traces[1][0][0]-traces[0][0][0])/1e9,
                                board=report['board'], vehicles=report['vehicles'],
                                slots=[r['slot'] for r in requests], recipients=[r['vehicle'] for r in requests],
                                bad_requests=[r for r in requests if r['status'] != 'ready'],
                                **trial_margins(path, amplitudes=amplitudes, distances=distances))
                            save(args.output / 'summary.json', summary)
                            if [r['slot'] for r in requests] != available:
                                raise RuntimeError('Request slots differ from the frozen availability')
    finally:
        board.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace-source', type=Path, required=True, help='Historical board run with api.csv and trial/')
    parser.add_argument('--plans-from', type=Path, required=True, help='Prior route-specific stagger experiments')
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--board', default='192.168.77.2')
    parser.add_argument('--board-port', type=int, default=5240)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
