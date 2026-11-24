"""Measure one DPU update at two route phases, then test same-budget allocation."""

import argparse
from functools import partial
import json
import math
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

from config import load_calibration
from falcon_paper_fidelity.motion_primitive import QuinticMotionPrimitivePlanner
from ..ip37_shadow import make_client
from .probe_blocks import LocalRpc
from .run_onboard_dpu import Route, infer_command, sample, assess, verify_model
from .run_onboard_pilot import VEHICLES, metrics, scenario, trial


PORT = 41592
STAGGER_S = 1.2
PHASES = {'early': 7.25, 'near': 8.20}  # Focus on gate 3 at x=9m.
BAND_M = .25
SLOT_S = .18
SLOTS = 46


def probe(board, calibration, treatment, stop):
    planner = QuinticMotionPrimitivePlanner()
    routes = {v: Route(a) for v, a in VEHICLES.items()}
    remaining = {(v, phase) for v in routes for phase in PHASES}
    records = []
    with LocalRpc(PORT) as rpc:
        while remaining and not stop.wait(.010):
            for vehicle in routes:
                stamp, state = sample(rpc, vehicle)
                for phase, threshold in PHASES.items():
                    if (vehicle, phase) not in remaining or state.position[0] < threshold:
                        continue
                    remaining.remove((vehicle, phase))
                    command, response = infer_command(board, calibration, planner, routes[vehicle], state)
                    slot = len(records) + 1
                    accepted = (phase == treatment and rpc.call('falconPilotRates', vehicle, slot, stamp, *command))
                    records.append(dict(slot=slot, phase=phase, vehicle=vehicle, sample_ns=stamp,
                                        position=state.position.tolist(), velocity=state.velocity.tolist(),
                                        quaternion=state.quaternion.tolist(), accepted=accepted, **response))
    return {}, records


def choose(seq, positions, last, gains):
    names = tuple(positions)
    fixed = names[(seq - 1) % len(names)]
    values = {v: next((gains[phase] for phase, x in PHASES.items() if abs(p-x) <= BAND_M), 0.)
              for v, p in positions.items()} if gains is not None else dict.fromkeys(names, 0.)
    if gains is None:
        return fixed, values
    vehicle = max(names, key=lambda v: (seq-last[v] >= 4, values[v], v == fixed))
    return vehicle, values


def allocate(board, calibration, gains, stop):
    planner = QuinticMotionPrimitivePlanner()
    routes = {v: Route(a) for v, a in VEHICLES.items()}
    last, records = dict.fromkeys(routes, 0), []
    with LocalRpc(PORT) as rpc:
        origin = perf_counter() + .45
        for seq in range(1, SLOTS + 1):
            if stop.wait(max(0., origin+(seq-1)*SLOT_S-perf_counter())):
                break
            states = {v: sample(rpc, v) for v in routes}
            positions = {v: float(state.position[0]) for v, (_, state) in states.items()}
            vehicle, values = choose(seq, positions, last, gains)
            last[vehicle] = seq
            stamp, state = states[vehicle]
            row = dict(slot=seq, vehicle=vehicle, sample_ns=stamp, positions_x_m=positions, values_m=values)
            records.append(row)
            try:
                command, response = infer_command(board, calibration, planner, routes[vehicle], state)
                row.update(response, accepted=rpc.call('falconPilotRates', vehicle, seq, stamp, *command))
            except (OSError, ValueError) as exc:
                row.update(status='error', error=str(exc), accepted=False)
    return {}, records


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def report_trial(client, gates, directory, feed, leading):
    result = trial(client, PORT, directory, gates, 3, False, feed, stagger_s=STAGGER_S, leading=leading)
    board = assess(directory, result, require_gap=False)
    gate3 = {}
    for vehicle, vreport in result.items():
        trace = read(directory / f'{vehicle}.json')
        crossing = next(((a[0]+(9.-a[1])*(b[0]-a[0])/(b[1]-a[1])) for a, b in zip(trace, trace[1:])
                         if a[1] <= 9. < b[1]), None)
        margin = metrics(trace, [gates[vehicle][-1]], vreport['contact'], None)['minimum_approach_clearance_m']
        gate3[vehicle] = dict(crossing_ns=crossing, margin_m=margin)
    crossings = [v['crossing_ns'] for v in gate3.values()]
    separation = (max(crossings)-min(crossings))/1e9 if all(c is not None for c in crossings) else None
    report = dict(vehicles=result, board=board, gate3=gate3, crossing_separation_s=separation, leading=leading)
    save(directory / 'value_report.json', report)
    print(directory.name, f"DPU={board['requests']}/{sum(board['adopted'].values())}",
          [(v, x['pass_gates'], gate3[v]['margin_m']) for v, x in result.items()], flush=True)
    return report


def paired(shadow, tested, vehicle, phase):
    def request(directory):
        return next(r for r in read(directory / 'board_requests.json')
                    if (r['vehicle'], r['phase']) == (vehicle, phase))

    a, b = request(shadow), request(tested)
    distance = float(np.linalg.norm(np.subtract(a['position'], b['position'])))
    speed = float(np.linalg.norm(np.subtract(a['velocity'], b['velocity'])))
    cosine = min(1., abs(float(np.dot(a['quaternion'], b['quaternion']))))
    degrees = math.degrees(2 * math.acos(cosine))
    before, after = (read(path / 'value_report.json')['gate3'][vehicle]['margin_m'] for path in (shadow, tested))
    return dict(vehicle=vehicle, phase=phase, position_difference_m=distance, velocity_difference_mps=speed,
                attitude_difference_deg=degrees, matched=distance <= .04 and speed <= .06 and degrees <= 2.,
                delta_m=after-before)


def run(args):
    board, manifest = make_client(args.bundle, args.board, args.board_port, 35.)
    try:
        model = verify_model(board, manifest, args.bundle)
        calibration = load_calibration()
        with scenario(PORT, args.output) as (client, gates):
            save(args.output / 'deployment.json', model)
            comparisons = []
            for rep, order in ((1, ('shadow', 'early', 'near')), (2, ('near', 'early', 'shadow'))):
                dirs = {name: args.output / f'cal_{rep}_{name}' for name in order}
                for name in order:
                    result = report_trial(client, gates, dirs[name], partial(probe, board, calibration, name), 'Drone1')
                    expected = 0 if name == 'shadow' else 2
                    if (result['board']['ready'] != 4 or sum(result['board']['adopted'].values()) != expected or
                            not all(r['pass_gates'] for r in result['vehicles'].values())):
                        raise RuntimeError('Incomplete single-update calibration trial')
                comparisons.extend(dict(repetition=rep, **paired(dirs['shadow'], dirs[phase], v, phase))
                                   for v in VEHICLES for phase in PHASES)
            groups = {phase: [r['delta_m'] for r in comparisons if r['phase'] == phase and r['matched']] for phase in PHASES}
            save(args.output / 'calibration.json', dict(pairs=comparisons, samples=groups))
            if any(len(items) < 2 for items in groups.values()):
                raise RuntimeError('Too few matched pulse pairs for a meaningful policy')
            gains = {phase: median(items) for phase, items in groups.items()}
            save(args.output / 'frozen_scores.json', dict(gains_m=gains, band_m=BAND_M))
            results = {}
            for rep, leading, order in ((1, 'Drone1', ('fixed', 'value')), (2, 'Drone2', ('value', 'fixed'))):
                for name in order:
                    directory = args.output / f'eval_{rep}_{name}'
                    result = report_trial(client, gates, directory,
                                          partial(allocate, board, calibration, gains if name == 'value' else None), leading)
                    results[directory.name] = result
                    save(args.output / 'summary.json', results)
                    if result['board']['requests'] != SLOTS or result['board']['ready'] != SLOTS:
                        raise RuntimeError('Request budgets differed or board failed an inference')
    finally:
        board.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--board', default='192.168.77.2')
    parser.add_argument('--board-port', type=int, default=5240)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
