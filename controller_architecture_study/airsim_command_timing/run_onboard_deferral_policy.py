"""Compare a single optional DPU exchange with two full-flight baselines."""

import argparse
from functools import partial
from math import sin
from pathlib import Path
from statistics import median

from config import load_calibration
from ..ip37_shadow import make_client
from .run_onboard_dpu import verify_model
from .run_onboard_fleet_value import gate_choice, gate_minimum, run_trial, trial_margins
from .run_onboard_load_replay import STAGGER, load_windows
from .run_onboard_pilot import BODY, Z, scenario
from .run_onboard_stagger_validation import LEADING, ROUTES, frozen_selector
from .run_onboard_value import BAND_M, PORT, SLOTS, allocate, read, save


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
        receiver = next(v for v in positions if v != donor)
        if positions[donor] < calibration['gate_x_m'][0]+BODY+.075:
            return donor, values
        phase = next((p for p, x in calibration['probe_x_m'].items()
                      if abs(x-positions[receiver]) <= BAND_M), None)
        if phase is None:
            return donor, values
        benefit = min(calibration['matched_deltas_m'][phase])
        target = profiles[receiver][f'gate{phase[1]}']
        donor_limits = [gate['baseline_m']-gate['cost_reserve_m'] for gate in profiles[donor].values()
                        if positions[donor] <= gate['critical_x_m']+BAND_M]
        if (benefit > 0 and positions[receiver]+BAND_M < target['critical_x_m'] and donor_limits and
                min(donor_limits) > target['baseline_m']+benefit):
            exchange.update(slot=seq, donor=donor, receiver=receiver, phase=phase,
                            donor_floor_m=min(donor_limits), receiver_gain_m=benefit, positions_x_m=positions)
            return receiver, values
        return donor, values

    def completed(row):
        if (exchange and 'repay_slot' not in exchange and row['slot'] > exchange['slot'] and
                row['vehicle'] == exchange['donor'] and row['accepted']):
            exchange['repay_slot'] = row['slot']

    return select, completed, exchange


def run(args):
    prior = args.replay_from
    calibration = read(prior / 'departure_plans.json')
    trace_source = Path(read(prior / 'load_masks.json')['source'])
    load = load_windows(trace_source, window_starts=WINDOW_STARTS)
    profiles = {route: gate_profiles(plan) for route, plan in calibration.items()}
    args.output.mkdir(parents=True)
    save(args.output / 'load_masks.json', load)
    save(args.output / 'gate_profiles.json', profiles)
    save(args.output / 'departure_plans.json', calibration)
    board, manifest = make_client(args.bundle, args.board, args.board_port, 35.)
    summary = {}
    try:
        identity = verify_model(board, manifest, args.bundle)
        if identity != read(prior / 'deployment.json'):
            raise ValueError('Expected the source replay board deployment')
        save(args.output / 'deployment.json', identity)
        calib = load_calibration()
        for route, (amplitudes, distances) in ROUTES.items():
            plan = calibration[route]
            baseline = partial(gate_choice, gains=plan['gains_m'], baseline=plan['baseline_m'],
                               distances=distances, phases=plan['probe_x_m'])
            fixed = partial(frozen_selector, plan['schedules'][str(STAGGER)])
            with scenario(PORT, args.output / route, amplitudes=amplitudes, distances=distances) as (client, gates):
                for window, slots in load['windows'].items():
                    occupied = {r['slot'] for r in slots if r['covering_call_ns']}
                    available = [r['slot'] for r in slots if r['slot'] not in occupied]
                    for rep, order in enumerate((POLICIES, POLICIES[::-1]), 1):
                        for policy in order:
                            select, completed, exchange = make_policy(plan, profiles[route], baseline)
                            path = args.output / route / f'{window}_r{rep}_{policy}'
                            feed = partial(allocate, board, calib, None,
                                           selector=dict(fixed_table=fixed, original=baseline, allow_once=select)[policy],
                                           on_result=completed if policy == 'allow_once' else None,
                                           amplitudes=amplitudes, occupied=occupied)
                            report = run_trial(client, gates, path, feed, LEADING, STAGGER, amplitudes=amplitudes)
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
    parser.add_argument('--board', default='192.168.77.2')
    parser.add_argument('--board-port', type=int, default=5240)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
