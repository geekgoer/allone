"""Change one recipient in the frozen two-UAV DPU replay schedule."""

import argparse
from functools import partial
from hashlib import sha256
from pathlib import Path

import numpy as np

from config import load_calibration
from ..ip37_shadow import make_client
from .run_onboard_dpu import verify_model
from .run_onboard_fleet_value import run_trial, trial_margins
from .run_onboard_load_replay import STAGGER
from .run_onboard_pilot import scenario
from .run_onboard_stagger_validation import LEADING, ROUTES
from .run_onboard_value import PORT, allocate, read, save


TARGET_SLOT = 25
CONTROL = 'Drone1'
SWAP = 'Drone2'
ORDERS = ((CONTROL, SWAP), (SWAP, CONTROL), (SWAP, CONTROL), (CONTROL, SWAP))


def freeze(source):
    window = read(source / 'load_masks.json')['windows']['w01']
    occupied = [row['slot'] for row in window if row['covering_call_ns']]
    paths = [source / 'independent' / f'w01_r{rep}_live_position' / 'board_requests.json' for rep in (1, 2)]
    schedules = [{row['slot']: row['vehicle'] for row in read(path)} for path in paths]
    if schedules[0] != schedules[1] or schedules[0].get(TARGET_SLOT) != CONTROL:
        raise ValueError('Prior live-position replicates must agree on the schedule and recipient')
    if list(schedules[0]) != [row['slot'] for row in window if row['slot'] not in occupied]:
        raise ValueError('Prior requests must exactly match the available replay slots')
    return dict(occupied=occupied, schedule=schedules[0], target_slot=TARGET_SLOT, route='independent',
                stagger_s=STAGGER, leading=LEADING,
                source_sha256={path.relative_to(source).as_posix(): sha256(path.read_bytes()).hexdigest()
                               for path in [source / 'load_masks.json', *paths]})


def state_difference(first, second):
    position = float(np.linalg.norm(np.subtract(first['position'], second['position'])))
    velocity = float(np.linalg.norm(np.subtract(first['velocity'], second['velocity'])))
    a, b = (np.asarray(s['quaternion']) for s in (first, second))
    cosine = abs(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))
    attitude = float(np.degrees(2*np.arccos(min(1., cosine))))
    return dict(position_m=position, velocity_mps=velocity, attitude_deg=attitude,
                matched=position <= .04 and velocity <= .06 and attitude <= 2.)


def audit_trial(path, plan, report, *, target_slot=TARGET_SLOT):
    requests = read(path / 'board_requests.json')
    target = next(r for r in requests if r['slot'] == target_slot)
    applied = {(v, r[5], r[6]): r[0] for v in report['vehicles'] for r in read(path / f'{v}.json') if r[4] == 3}
    expected = {(r['vehicle'], r['slot'], r['sample_ns']) for r in requests}
    if ([(r['slot'], r['vehicle']) for r in requests] != list(plan.items()) or set(applied) != expected or
            any(r['status'] != 'ready' or r['board']['actual_model_id'] != 11 or
                r['board']['saturated_input_channels'] != 0 for r in requests)):
        raise RuntimeError('Frozen DPU schedule or request adoption differs from the experiment plan')
    stamp = target['sample_ns']
    target_apply = applied[target['vehicle'], target_slot, stamp]
    other = SWAP if target['vehicle'] == CONTROL else CONTROL
    before = max(t for (v, s, _), t in applied.items() if v == other and s < target_slot)
    after = min(t for (v, s, _), t in applied.items() if v == other and s > target_slot)
    local = sum(row[4] == 1 and before < row[0] < after for row in read(path / f'{other}.json'))
    return dict(target_states=target['states'], target_applied_ms=(target_apply-stamp)/1e6,
                omitted_vehicle=other, omitted_vehicle_dpu_gap_ms=(after-before)/1e6,
                omitted_vehicle_local_updates=local)


def run(args):
    frozen = freeze(args.replay_from)
    amplitudes, distances = ROUTES[frozen['route']]
    board, manifest = make_client(args.bundle, args.board, args.board_port, 35.)
    try:
        identity = verify_model(board, manifest, args.bundle)
        if identity != read(args.replay_from / 'deployment.json'):
            raise ValueError('Expected the same board deployment as the source replay')
        calibration = load_calibration()
        with scenario(PORT, args.output, amplitudes=amplitudes, distances=distances) as (client, gates):
            save(args.output / 'frozen_experiment.json', dict(frozen, orders=ORDERS, deployment=identity))
            summary = {}
            for rep, order in enumerate(ORDERS, 1):
                trials = {}
                for recipient in order:
                    plan = dict(frozen['schedule'])
                    plan[TARGET_SLOT] = recipient
                    path = args.output / f'r{rep}_{recipient}'
                    def selector(seq, positions, last):
                        return plan[seq], dict.fromkeys(positions, 0.)
                    feed = partial(allocate, board, calibration, None, selector=selector, amplitudes=amplitudes,
                                   occupied=frozen['occupied'], capture_state_slot=TARGET_SLOT)
                    result = run_trial(client, gates, path, feed, LEADING, STAGGER, amplitudes=amplitudes)
                    trials[recipient] = dict(
                        board=result['board'], vehicles=result['vehicles'],
                        **trial_margins(path, amplitudes=amplitudes, distances=distances),
                        **audit_trial(path, plan, result))
                control, swap = (trials[v] for v in (CONTROL, SWAP))
                initial = {v: state_difference(control['target_states'][v], swap['target_states'][v])
                           for v in amplitudes}
                summary[f'r{rep}'] = dict(trials=trials, pre_slot_difference=initial,
                                          matched=all(row['matched'] for row in initial.values()))
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
