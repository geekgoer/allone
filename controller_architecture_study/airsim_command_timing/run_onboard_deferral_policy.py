"""Compare a single optional DPU exchange with two full-flight baselines."""

import argparse
from functools import partial
from math import sin
from pathlib import Path
from statistics import median

import numpy as np

from config import load_calibration
from ..ip37_shadow import make_client
from .run_onboard_dpu import verify_model
from .run_onboard_fleet_value import gate_choice, gate_minimum, run_trial, trial_margins
from .run_onboard_load_replay import STAGGER, load_windows
from .run_onboard_pilot import BODY, Z, scenario
from .run_onboard_stagger_validation import LEADING, ROUTES, frozen_selector
from .run_onboard_value import BAND_M, PORT, SLOTS, SLOT_S, SLOT_START_S, allocate, read, save


WINDOW_STARTS = (1, 9)
POLICIES = ('fixed_table', 'original', 'allow_once')


def gate_profiles(calibration):
    shadows = sorted(Path(calibration['calibration_source']).glob('cal_*_shadow'))
    profiles = {}
    for v, amplitude in calibration['amplitudes'].items():
        traces = [read(path / f'{v}.json') for path in shadows]
        profiles[v] = {}
        for index, x in enumerate(calibration['gate_x_m'][1:], 2):
            gate = f'gate{index}'
            critical = median(gate_minimum(trace, (x, amplitude*sin(.6*x), Z))[1] for trace in traces)
            reserve = max(abs(value) for phase, samples in calibration['matched_deltas_m'].items()
                          if phase.startswith(f'g{index}_') for value in samples)
            profiles[v][gate] = dict(critical_x_m=critical, baseline_m=calibration['baseline_m'][v][gate],
                                     cost_reserve_m=reserve)
    return profiles


def make_policy(calibration, profiles, baseline):
    exchange = {}

    def select(seq, positions, last):
        donor, values = baseline(seq, positions, last)
        if exchange and 'repay_slot' not in exchange:
            return exchange['donor'], values
        if exchange or not last[donor] or seq-last[donor] < 4 or seq >= SLOTS or values[donor] > 0:
            return donor, values
        if positions[donor] < calibration['gate_x_m'][0]+BODY+.075:
            return donor, values
        donor_limits = [gate['baseline_m']-gate['cost_reserve_m'] for gate in profiles[donor].values()
                        if positions[donor] <= gate['critical_x_m']+BAND_M]
        if not donor_limits:
            return donor, values
        floor = min(donor_limits)
        offers = []
        for receiver, progress in positions.items():
            if receiver == donor:
                continue
            phase = next((p for p, x in calibration['probe_x_m'].items() if abs(x-progress) <= BAND_M), None)
            if phase is None:
                continue
            benefit = min(calibration['matched_deltas_m'][phase])
            target = profiles[receiver][f'gate{phase[1]}']
            if benefit > 0 and progress+BAND_M < target['critical_x_m'] and floor > target['baseline_m']+benefit:
                offers.append((target['baseline_m'], -benefit, receiver, phase))
        if not offers:
            return donor, values
        _, negative_benefit, receiver, phase = min(offers)
        exchange.update(slot=seq, donor=donor, receiver=receiver, phase=phase,
                        donor_floor_m=floor, receiver_gain_m=-negative_benefit, positions_x_m=positions)
        return receiver, values

    def completed(row):
        if (exchange and 'repay_slot' not in exchange and row['slot'] > exchange['slot'] and
                row['vehicle'] == exchange['donor'] and row['accepted']):
            exchange['repay_slot'] = row['slot']

    return select, completed, exchange


def fleet_plan(template, count):
    types = {f'Drone{i}': f'Drone{1+(i-1)%2}' for i in range(1, count+1)}
    order = [f'Drone{i}' for pair in range(1, count, 2) for i in (pair+1, pair)]
    departures = {v: i*STAGGER/(count-1) for i, v in enumerate(order)}
    expanded = dict(template, amplitudes={v: template['amplitudes'][kind] for v, kind in types.items()},
                    baseline_m={v: template['baseline_m'][kind] for v, kind in types.items()})
    return expanded, types, departures


def preplanned_table(calibration, types, departures, available, choose):
    shadows = sorted(Path(calibration['calibration_source']).glob('cal_*_shadow'))
    series = {v: [np.asarray(read(path / f'{v}.json'))[:, :2] for path in shadows] for v in set(types.values())}
    last = dict.fromkeys(types, 0)
    table = {}
    for seq in available:
        slot_age = SLOT_START_S+(seq-1)*SLOT_S+max(departures.values())
        positions = {v: float(median(np.interp(slot_age-departures[v], (trace[:, 0]-trace[0, 0])/1e9,
                                             trace[:, 1]) for trace in series[kind]))
                     for v, kind in types.items()}
        recipient, values = choose(seq, positions, last)
        table[seq] = dict(vehicle=recipient, values_m=values)
        last[recipient] = seq
    return table


def run(args):
    prior = args.replay_from
    calibration = read(prior / 'departure_plans.json')
    trace_source = Path(read(prior / 'load_masks.json')['source'])
    load = load_windows(trace_source, window_starts=WINDOW_STARTS)
    profiles = {route: gate_profiles(plan) for route, plan in calibration.items()}
    configurations = ROUTES
    fleet = {}
    if args.fleet > 2:
        template = calibration['independent']
        plan, types, departures = fleet_plan(template, args.fleet)
        configurations = {'independent': (plan['amplitudes'], plan['gate_x_m'])}
        calibration = {'independent': plan}
        profiles = {'independent': {v: profiles['independent'][kind] for v, kind in types.items()}}
        fleet = dict(types=types, departures_s=departures, static_masks_known=True)
    args.output.mkdir(parents=True)
    save(args.output / 'load_masks.json', load)
    save(args.output / 'gate_profiles.json', profiles)
    save(args.output / 'departure_plans.json', calibration)
    if fleet:
        save(args.output / 'fleet.json', fleet)
    board, manifest = make_client(args.bundle, args.board, args.board_port, 35.)
    summary = {}
    try:
        identity = verify_model(board, manifest, args.bundle)
        if identity != read(prior / 'deployment.json'):
            raise ValueError('Expected the source replay board deployment')
        save(args.output / 'deployment.json', identity)
        calib = load_calibration()
        for route, (amplitudes, distances) in configurations.items():
            plan = calibration[route]
            baseline = partial(gate_choice, gains=plan['gains_m'], baseline=plan['baseline_m'],
                               distances=distances, phases=plan['probe_x_m'])
            fixed = partial(frozen_selector, plan['schedules'][str(STAGGER)])
            with scenario(PORT, args.output / route, amplitudes=amplitudes, distances=distances) as (client, gates):
                for window, slots in load['windows'].items():
                    occupied = {r['slot'] for r in slots if r['covering_call_ns']}
                    available = [r['slot'] for r in slots if r['slot'] not in occupied]
                    if fleet:
                        table = preplanned_table(plan, types, departures, available, baseline)
                        save(args.output / f'static_plan_{window}.json', table)

                        def fixed(seq, positions, last):
                            return table[seq]['vehicle'], table[seq]['values_m']

                    for rep, order in enumerate((POLICIES, POLICIES[::-1]), 1):
                        for policy in order:
                            select, completed, exchange = make_policy(plan, profiles[route], baseline)
                            path = args.output / route / f'{window}_r{rep}_{policy}'
                            feed = partial(allocate, board, calib, None,
                                           selector=dict(fixed_table=fixed, original=baseline, allow_once=select)[policy],
                                           on_result=completed if policy == 'allow_once' else None,
                                           amplitudes=amplitudes, occupied=occupied)
                            report = run_trial(client, gates, path, feed, LEADING, STAGGER, amplitudes=amplitudes,
                                               departures_s=fleet.get('departures_s'))
                            requests = read(path / 'board_requests.json')
                            if [r['slot'] for r in requests] != available:
                                raise RuntimeError('Request slots differ from the frozen availability')
                            if exchange:
                                used = {(v, row[5]): row[0] for v in amplitudes
                                        for row in read(path / f'{v}.json') if row[4] == 3}
                                exchange['exchange_applied'] = (exchange['receiver'], exchange['slot']) in used
                                exchange['repay_applied'] = (exchange['donor'], exchange.get('repay_slot')) in used
                            summary[f'{route}/{path.name}'] = dict(
                                route=route, window=window, repetition=rep, policy=policy,
                                exchange=exchange, board=report['board'], vehicles=report['vehicles'],
                                recipients=[r['vehicle'] for r in requests],
                                **trial_margins(path, amplitudes=amplitudes, distances=distances))
                            save(args.output / 'summary.json', summary)
    finally:
        board.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay-from', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fleet', type=int, choices=(2, 4, 6), default=2)
    parser.add_argument('--board', default='192.168.77.2')
    parser.add_argument('--board-port', type=int, default=5240)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
