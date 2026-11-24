"""Keep the local pilot running while one genuine DPU serves two AirSim vehicles."""

import argparse
from collections import Counter
from functools import partial
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np

from airsim_interface import RawDroneState
from config import load_calibration
from control_adapter import thrust_to_throttle
from falcon_paper_fidelity.motion_primitive import QuinticMotionPrimitivePlanner
from ..convex_features import encode
from ..decision_dataset import sha256
from ..ip37_shadow import make_client, predict
from .probe_blocks import LocalRpc
from .run_onboard_pilot import GAP_S, VEHICLES, Z, scenario, trial


MODEL = 11
SPEED = 1.65
SLOT_S = .080
SLOTS = 96
PAUSE_SLOT = 45


class Route:
    def __init__(self, amplitude):
        self.amplitude = amplitude

    def evaluate(self, t):
        x = min(12., t * SPEED)
        angle = .6 * x
        v = SPEED if x < 12. else 0.
        return (np.array([x, self.amplitude * math.sin(angle), Z]),
                np.array([v, .6 * v * self.amplitude * math.cos(angle), 0.]),
                np.array([0., -(.6 * v)**2 * self.amplitude * math.sin(angle), 0.]),
                np.array([1., 0., 0., 0.]))


def sample(rpc, vehicle):
    started = rpc.call('falconTraceClock')
    state = rpc.call('getMultirotorState', vehicle)
    kin = state['kinematics_estimated']

    def vector(field, axes='xyz'):
        return np.array([kin[field][axis + '_val'] for axis in axes])

    return started, RawDroneState(vector('position'), vector('orientation', 'wxyz'),
                                  vector('linear_velocity'), vector('linear_acceleration'),
                                  vector('angular_velocity'), state['timestamp'])


def infer_command(board, calibration, planner, route, state):
    window = planner.reference_window_for_trajectory(route, elapsed=max(0., state.position[0]) / SPEED,
                                                      dt=.1, horizon=21)
    output, timing = predict(board, MODEL, encode(state, window))
    thrust, roll, pitch, yaw = np.clip(output, [2., -3., -3., -2.], [20., 3., 3., 2.])
    command = [float(roll), float(pitch), float(yaw), thrust_to_throttle(thrust, calibration)]
    return command, dict(status='ready', output=output.tolist(), board=timing)


def feed(port, board, calibration, policy, stop):
    planner = QuinticMotionPrimitivePlanner()
    routes = {v: Route(amplitude) for v, amplitude in VEHICLES.items()}
    last = dict.fromkeys(VEHICLES, 0)
    records, incident = [], None
    with LocalRpc(port) as rpc:
        origin = perf_counter() + .45
        for seq in range(1, SLOTS + 1):
            if seq in (PAUSE_SLOT + 1, PAUSE_SLOT + 2):
                continue
            release = origin + (seq - 1) * SLOT_S
            if stop.wait(max(0., release - perf_counter())):
                break
            samples = {v: sample(rpc, v) for v in routes}
            errors = {v: float(abs(state.position[1] - routes[v].evaluate(max(0., state.position[0]) / SPEED)[0][1]))
                      for v, (_, state) in samples.items()}
            if policy == 'fixed':
                vehicle = tuple(routes)[(seq - 1) % len(routes)]
            else:
                vehicle = max(routes, key=lambda v: (seq - last[v] >= 4, errors[v], seq - last[v]))
            last[vehicle] = seq
            stamp, state = samples[vehicle]
            row = dict(slot=seq, vehicle=vehicle, sample_ns=stamp, lateral_error_m=errors)
            records.append(row)
            try:
                command, response = infer_command(board, calibration, planner, routes[vehicle], state)
                row.update(response)
            except (OSError, ValueError) as exc:
                row.update(status='error', error=str(exc))
                if seq == PAUSE_SLOT:
                    raise RuntimeError('No DPU result available for the late-command test') from exc
                continue
            if seq == PAUSE_SLOT:
                start = rpc.call('falconTraceClock')
                if stop.wait(GAP_S):
                    break
                end = rpc.call('falconTraceClock')
                late = rpc.call('falconPilotRates', vehicle, seq, stamp, *command)
                row['accepted'] = late
                incident = dict(start_ns=start, end_ns=end, late_accepted=late, delayed_vehicle=vehicle)
            else:
                row['accepted'] = rpc.call('falconPilotRates', vehicle, seq, stamp, *command)
    return {v: incident for v in routes}, records


def verify_model(board, manifest, bundle):
    entry = next(m for m in manifest['models'] if m['numeric_id'] == MODEL)
    golden = bundle / entry['golden_file']
    payload = json.loads((bundle / 'payload_manifest.json').read_text(encoding='utf-8'))
    if manifest['wire_tag_hex'] != '9253dd1fc87f8d7f' or sha256(golden) != payload['files'][entry['golden_file']]:
        raise ValueError('Expected the frozen release001 identity and model golden file')
    data = np.loadtxt(golden, max_rows=3)
    errors = []
    for row in data:
        output, _ = predict(board, MODEL, row[:237].astype(np.float32))
        errors.append(float(np.max(np.abs(output-row[237:241]))))
    if max(errors) > 1e-4:
        raise ValueError(f'Frozen INT8 golden output mismatch {errors}')
    return dict(model=MODEL, wire_tag=manifest['wire_tag_hex'], golden_max_abs=max(errors))


def assess(directory, result, require_gap=True):
    requests = json.loads((directory / 'board_requests.json').read_text(encoding='utf-8'))
    accepted = {(r['vehicle'], r['slot']): r for r in requests if r.get('accepted')}
    adopted = Counter()
    for vehicle, metrics in result.items():
        trace = json.loads((directory / f'{vehicle}.json').read_text(encoding='utf-8'))
        incident = metrics['incident']
        central = [r for r in trace if r[4] == 3]
        for row in central:
            request = accepted[vehicle, int(row[5])]
            if row[6] != request['sample_ns'] or not 0 <= row[0] - row[6] < 55_000_000:
                raise ValueError('DPU adoption did not match a fresh accepted request')
        adopted[vehicle] = len(central)
        if require_gap and not (incident and not incident['late_accepted'] and metrics['local_updates_during_gap'] and
                any(r[0] < incident['start_ns'] for r in central) and any(r[0] > incident['end_ns'] for r in central)):
            raise ValueError(f'DPU handoff check failed for {vehicle}')
    return dict(requests=len(requests), ready=sum(r['status'] == 'ready' for r in requests),
                accepted=len(accepted), adopted=adopted, by_vehicle=Counter(r['vehicle'] for r in requests))


def run(args):
    board, manifest = make_client(args.bundle, args.board, args.board_port, 35.)
    reports = {}
    try:
        identity = verify_model(board, manifest, args.bundle)
        calibration = load_calibration()
        with scenario(41592, args.output) as (client, gates):
            (args.output / 'deployment.json').write_text(json.dumps(identity, indent=2), encoding='utf-8')
            order = ('adaptive', 'fixed') if args.adaptive_first else ('fixed', 'adaptive')
            for arm in ('local', *order):
                directory = args.output / arm
                shared = None if arm == 'local' else partial(feed, 41592, board, calibration, arm)
                result = trial(client, 41592, directory, gates, 1 if shared is None else 3, False, shared)
                reports[arm] = dict(vehicles=result, board=assess(directory, result) if shared else None)
                print(arm, {v: r['pass_gates'] for v, r in result.items()}, reports[arm]['board'], flush=True)
                (args.output / 'summary.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
                if not all(r['pass_gates'] for r in result.values()):
                    raise RuntimeError(f'{arm} did not pass its gate check')
        if {reports[arm]['board']['requests'] for arm in ('fixed', 'adaptive')} != {SLOTS-2}:
            raise RuntimeError('The two policies did not run with the full common request budget')
    finally:
        board.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--board', default='192.168.77.2')
    parser.add_argument('--board-port', type=int, default=5240)
    parser.add_argument('--adaptive-first', action='store_true', help='Reverse the two resource policies for the repeat')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
