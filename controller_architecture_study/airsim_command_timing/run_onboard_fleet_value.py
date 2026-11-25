"""Compare gate-aware online allocation with a frozen route-aware schedule."""

import argparse
from functools import partial
from pathlib import Path
from statistics import median

import numpy as np

from config import load_calibration
from ..ip37_shadow import make_client
from .run_onboard_dpu import verify_model
from .run_onboard_pilot import APERTURE, BODY, DISTANCES, VEHICLES, Z, metrics, scenario
from .run_onboard_value import (BAND_M, PORT, SLOTS, SLOT_S, SLOT_START_S, STAGGER_S, allocate, paired,
                                probe, read, report_trial, save)


def probe_phases(distances):
    return {f'g{g}_{phase}': round(x-offset, 2) for g, x in enumerate(distances[1:], 2)
            for phase, offset in (('early', 1.75), ('near', .80))}


PHASES = probe_phases(DISTANCES)


def gate_choice(seq, positions, last, gains, baseline, *, distances=DISTANCES, phases=PHASES):
    names = tuple(positions)
    fixed = names[(seq - 1) % len(names)]
    active = {}
    values = dict.fromkeys(names, 0.)
    for v, x in positions.items():
        gate = next((g for g, center in enumerate(distances[1:], 2) if center-2 <= x <= center+.6), None)
        if gate is not None:
            active[v] = baseline[v][f'gate{gate}']
        values[v] = next((gains[key] for key, center in phases.items() if abs(x-center) <= BAND_M), 0.)
    overdue = [v for v in names if seq-last[v] >= 4]
    if overdue:
        return min(overdue, key=lambda v: (last[v], v != fixed)), values
    if not active:
        return fixed, values

    def score(v):
        # Can the next command improve the currently tightest active gate?
        floor = min(margin + (max(0., values[v]) if peer == v else 0.) for peer, margin in active.items())
        return floor, v == fixed

    return max(names, key=score), values


def nominal_schedule(paths, leading, gains, baseline, *, stagger=STAGGER_S, distances=DISTANCES, phases=PHASES):
    series = {v: [np.asarray(read(path / f'{v}.json'))[:, :2] for path in paths] for v in VEHICLES}
    last = dict.fromkeys(VEHICLES, 0)
    schedule = []
    for seq in range(1, SLOTS+1):
        positions = {}
        for v, traces in series.items():
            age = SLOT_START_S + (seq-1)*SLOT_S + (stagger if v == leading else 0.)
            positions[v] = float(median(np.interp(age, (trace[:, 0]-trace[0, 0])/1e9, trace[:, 1])
                                        for trace in traces))
        vehicle, values = gate_choice(seq, positions, last, gains, baseline, distances=distances, phases=phases)
        schedule.append(dict(slot=seq, vehicle=vehicle, nominal_x_m=positions, values_m=values))
        last[vehicle] = seq
    return schedule


def run_trial(client, gates, directory, feed, leading, stagger=STAGGER_S, *, amplitudes=VEHICLES):
    report = report_trial(client, gates, directory, feed, leading, stagger_s=stagger, amplitudes=amplitudes)
    report['gate2'] = {}
    for v, result in report['vehicles'].items():
        trace = read(directory / f'{v}.json')
        approach = metrics(trace, [gates[v][1]], result['contact'], None)['minimum_approach_clearance_m']
        report['gate2'][v] = dict(margin_m=approach)
    margins = [r['minimum_approach_clearance_m'] for r in report['vehicles'].values()]
    report['fleet_min_m'] = min(margins) if None not in margins else None
    report['stagger_s'] = stagger
    save(directory / 'value_report.json', report)
    return report


def interpolated_margin(trace, center):
    """Include both boundaries of the gate slab in the sampled polyline."""
    half_depth = BODY + .075
    residual = []
    for a, b in zip(trace, trace[1:]):
        x0, x1 = a[1], b[1]
        if x0 == x1:
            if abs(x0-center[0]) > half_depth:
                continue
            lo, hi = 0., 1.
        else:
            low, high = sorted(((center[0]-half_depth-x0)/(x1-x0), (center[0]+half_depth-x0)/(x1-x0)))
            lo, hi = max(0., low), min(1., high)
            if lo > hi:
                continue
        for t in (lo, hi):
            deviation = max(abs(a[j]+t*(b[j]-a[j])-center[j-1]) for j in (2, 3))
            residual.append(APERTURE/2-BODY-deviation)
    return min(residual, default=None)


def trial_margins(path, *, amplitudes=VEHICLES, distances=DISTANCES):
    margins = {}
    for v, amplitude in amplitudes.items():
        trace = read(path / f'{v}.json')
        margins[v] = {f'gate{i}': interpolated_margin(trace, (x, amplitude*np.sin(.6*x), Z))
                      for i, x in enumerate(distances, 1)}
    complete = [margin for gates in margins.values() for margin in gates.values()]
    return dict(by_gate_m=margins, fleet_min_m=min(complete) if None not in complete else None)


def audit_margins(output):
    report = {path.name: trial_margins(path) for path in sorted(output.iterdir())
              if (path / 'value_report.json').is_file()}
    save(output / 'interpolated_margin_audit.json', report)
    return report


def calibrate(client, gates, output, board, calibration, *, amplitudes=VEHICLES, distances=DISTANCES):
    comparisons, shadows = [], []
    phases = probe_phases(distances)
    arms = ('shadow', *phases)
    for rep, order in ((1, arms), (2, tuple(reversed(arms)))):
        directories = {name: output / f'cal_{rep}_{name}' for name in order}
        for name in order:
            feed = partial(probe, board, calibration, name, phases=phases, amplitudes=amplitudes)
            result = run_trial(client, gates, directories[name], feed, 'Drone1', amplitudes=amplitudes)
            expected = 0 if name == 'shadow' else 2
            if (result['board']['ready'] != 8 or sum(result['board']['adopted'].values()) != expected or
                    not all(r['pass_gates'] for r in result['vehicles'].values())):
                raise RuntimeError('Incomplete calibration flight or model adoption')
        shadows.append(directories['shadow'])
        comparisons.extend(dict(repetition=rep, **paired(directories['shadow'], directories[phase], v, phase,
                                                         report_key=f'gate{phase[1]}'))
                           for v in VEHICLES for phase in phases)
    groups = {phase: [r['delta_m'] for r in comparisons if r['phase'] == phase and r['matched']] for phase in phases}
    save(output / 'calibration.json', dict(pairs=comparisons, samples=groups))
    if any(len(items) < 2 for items in groups.values()):
        raise RuntimeError('Insufficient matched calibration samples')
    gain = {phase: median(items) for phase, items in groups.items()}
    baseline = {v: {f'gate{g}': median(r['baseline_margin_m'] for r in comparisons if
                                      r['vehicle'] == v and r['phase'].startswith(f'g{g}'))
                    for g in (2, 3)} for v in VEHICLES}
    schedules = {v: nominal_schedule(shadows, v, gain, baseline, distances=distances, phases=phases)
                 for v in VEHICLES}
    save(output / 'frozen_plans.json', dict(gains_m=gain, baseline_m=baseline, nominal_stagger_s=STAGGER_S,
                                           planned_slots=schedules))
    return gain, baseline, schedules


def run(args):
    board, manifest = make_client(args.bundle, args.board, args.board_port, 35.)
    try:
        identity = verify_model(board, manifest, args.bundle)
        calibration = load_calibration()
        with scenario(PORT, args.output) as (client, gates):
            save(args.output / 'deployment.json', identity)
            if args.frozen_from:
                if read(args.frozen_from / 'deployment.json')['wire_tag'] != identity['wire_tag']:
                    raise ValueError('Frozen scores must use the same board deployment')
                frozen_data = read(args.frozen_from / 'frozen_plans.json')
                gains, baseline, plans = (frozen_data[k] for k in ('gains_m', 'baseline_m', 'planned_slots'))
                save(args.output / 'frozen_plans.json', frozen_data)
                save(args.output / 'calibration_source.json', dict(directory=str(args.frozen_from.resolve())))
            else:
                gains, baseline, plans = calibrate(client, gates, args.output, board, calibration)
            summary = {}
            trials = ((1, 'Drone1', STAGGER_S, ('round_robin', 'static', 'online')),
                      (2, 'Drone2', 1.50, ('online', 'static', 'round_robin')))
            for rep, leading, stagger, order in trials:
                def frozen(seq, positions, last):
                    plan = plans[leading][seq-1]
                    return plan['vehicle'], plan['values_m']

                for name in order:
                    selector = {'round_robin': None, 'static': frozen,
                                'online': partial(gate_choice, gains=gains, baseline=baseline)}[name]
                    feed = partial(allocate, board, calibration, None, selector=selector)
                    path = args.output / f'eval_{rep}_{name}'
                    result = run_trial(client, gates, path, feed, leading, stagger)
                    summary[path.name] = result
                    save(args.output / 'summary.json', summary)
                    if result['board']['requests'] != SLOTS:
                        raise RuntimeError('Request count differs from the common budget')
            audit_margins(args.output)
    finally:
        board.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--board', default='192.168.77.2')
    parser.add_argument('--board-port', type=int, default=5240)
    parser.add_argument('--frozen-from', type=Path, help='Repeat evaluation with an existing calibration directory')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
