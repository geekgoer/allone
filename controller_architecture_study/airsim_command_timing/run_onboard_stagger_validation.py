"""Compare online DPU selection against departure-aware static calendars."""

import argparse
from functools import partial
from pathlib import Path
from statistics import median

from config import load_calibration
from ..ip37_shadow import make_client
from .run_onboard_dpu import verify_model
from .run_onboard_fleet_value import (calibrate, gate_choice, nominal_schedule, probe_phases, run_trial,
                                      trial_margins)
from .run_onboard_pilot import DISTANCES, VEHICLES, scenario
from .run_onboard_value import PORT, SLOTS, STAGGER_S, allocate, read, save


ROUTES = {'original': (VEHICLES, DISTANCES),
          'independent': ({'Drone1': .52, 'Drone2': -.58}, (3.30, 6.35, 9.35))}
LEADING = 'Drone2'
ARMS = ('round_robin', 'static_nominal', 'online', 'static_actual')
ORDERS = {'original': ((0, 1, 2, 3), (1, 0, 3, 2)),
          'independent': ((2, 3, 0, 1), (3, 2, 1, 0))}
STAGGERS = (.90, 1.50)


def freeze(source, amplitudes, distances):
    phases = probe_phases(distances)
    pairs = read(source / 'calibration.json')['pairs']
    margins = {path.name: trial_margins(path, amplitudes=amplitudes, distances=distances)['by_gate_m']
               for path in source.glob('cal_*') if (path / 'value_report.json').is_file()}
    collected = {phase: [] for phase in phases}
    for pair in pairs:
        v, phase, rep = (pair[k] for k in ('vehicle', 'phase', 'repetition'))
        gate = f'gate{phase[1]}'
        before = margins[f'cal_{rep}_shadow'][v][gate]
        after = margins[f'cal_{rep}_{phase}'][v][gate]
        if pair['matched'] and before is not None and after is not None:
            collected[phase].append(after-before)
    if any(len(rows) < 2 for rows in collected.values()):
        raise RuntimeError('Insufficient matched calibration for the interpolated margin')
    gains = {phase: median(rows) for phase, rows in collected.items()}
    shadows = sorted(source.glob('cal_*_shadow'))
    baseline = {v: {g: median(margins[p.name][v][g] for p in shadows) for g in ('gate2', 'gate3')}
                for v in amplitudes}
    schedules = {str(s): nominal_schedule(shadows, LEADING, gains, baseline, stagger=s,
                                          distances=distances, phases=phases)
                 for s in (*STAGGERS, STAGGER_S)}
    return dict(calibration_source=str(source.resolve()), gains_m=gains, matched_deltas_m=collected,
                baseline_m=baseline, nominal_stagger_s=STAGGER_S, schedules=schedules,
                leading=LEADING, amplitudes=amplitudes, gate_x_m=distances, probe_x_m=phases)


def frozen_selector(plan, seq, positions, last):
    slot = plan[seq-1]
    return slot['vehicle'], slot['values_m']


def run(args):
    amplitudes, distances = ROUTES[args.route]
    board, manifest = make_client(args.bundle, args.board, args.board_port, 35.)
    try:
        identity = verify_model(board, manifest, args.bundle)
        calibration = load_calibration()
        with scenario(PORT, args.output, amplitudes=amplitudes, distances=distances) as (client, gates):
            save(args.output / 'deployment.json', identity)
            source = args.calibration_from or args.output
            if args.calibration_from:
                if read(source / 'deployment.json') != identity:
                    raise ValueError('Calibration and evaluation board deployments differ')
            else:
                calibrate(client, gates, source, board, calibration, amplitudes=amplitudes, distances=distances)
            frozen = freeze(source, amplitudes, distances)
            save(args.output / 'stagger_plans.json', frozen)
            gains = frozen['gains_m']
            plans = frozen['schedules']
            online = partial(gate_choice, gains=gains, baseline=frozen['baseline_m'],
                             distances=distances, phases=frozen['probe_x_m'])
            summary = {}
            for stagger, order in zip(STAGGERS, ORDERS[args.route]):
                selectors = {'round_robin': None, 'online': online,
                             'static_nominal': partial(frozen_selector, plans[str(STAGGER_S)]),
                             'static_actual': partial(frozen_selector, plans[str(stagger)])}
                for i in order:
                    name = ARMS[i]
                    path = args.output / f'eval_{stagger:.2f}_{name}'
                    feed = partial(allocate, board, calibration, None, selector=selectors[name], amplitudes=amplitudes)
                    result = run_trial(client, gates, path, feed, LEADING, stagger, amplitudes=amplitudes)
                    requests = read(path / 'board_requests.json')
                    traces = [read(path / f'{v}.json') for v in (LEADING, 'Drone1')]
                    observed = (traces[1][0][0]-traces[0][0][0])/1e9
                    summary[path.name] = dict(policy=name, programmed_stagger_s=stagger, observed_stagger_s=observed,
                                              board=result['board'], vehicles=result['vehicles'],
                                              recipients=[r['vehicle'] for r in requests],
                                              bad_requests=[r for r in requests if r['status'] != 'ready'],
                                              **trial_margins(path, amplitudes=amplitudes, distances=distances))
                    save(args.output / 'summary.json', summary)
                    if result['board']['requests'] != SLOTS:
                        raise RuntimeError('Request count differs from the common budget')
    finally:
        board.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--route', choices=ROUTES, required=True)
    parser.add_argument('--calibration-from', type=Path, help='Existing calibration for the original route')
    parser.add_argument('--board', default='192.168.77.2')
    parser.add_argument('--board-port', type=int, default=5240)
    args = parser.parse_args()
    if args.calibration_from and args.route != 'original':
        parser.error('External calibration is only defined for the original route')
    run(args)


if __name__ == '__main__':
    main()
