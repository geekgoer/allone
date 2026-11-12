"""Single-drone AirSim flight controlled by the IP37 board model."""
import argparse
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from .control_value_probe import flight
from .decision_dataset import atomic_json, sha256
from .fly_convex_policy import load_airsim_client
from .fly_decision_dnn import StoredRoutePlanner
from .ip37_shadow import make_client, predict
from .probe_airsim_three import load_layout


class BoardPolicy:
    def __init__(self, client, model_id):
        self.client, self.model_id = client, model_id
        self.records = []

    def predict(self, model, feature):
        row = {'request_start_s': perf_counter(), 'raw237': feature.tolist()}
        self.records.append(row)
        try:
            action, timing = predict(self.client, self.model_id, feature)
            row.update(timing, output=action.tolist(), status='ready')
            return action, timing
        except (OSError, ValueError) as exc:
            row.update(status='error', error=f'{type(exc).__name__}: {exc}')
            raise


def run(args):
    if not np.isfinite(args.period_ms) or args.period_ms <= 0:
        raise ValueError('period-ms must be positive and finite')
    args.output.mkdir(parents=True, exist_ok=False)
    board = None
    policy = None
    result = {'flight_passed': False, 'error': None}
    try:
        board, manifest = make_client(args.bundle, args.board, args.board_port, args.period_ms)
        entry = next(m for m in manifest['models'] if m['numeric_id'] == args.model_id)
        golden = args.bundle / entry.get('golden_file', f"inputs/{args.model_id}.txt")
        if 'golden_file_sha256' in entry and sha256(golden) != entry['golden_file_sha256']:
            raise ValueError('Candidate golden file hash mismatch')
        sample = np.loadtxt(golden, max_rows=1)
        action, _ = predict(board, args.model_id, sample[:237].astype(np.float32))
        if np.max(abs(action - sample[237:241])) > 1e-4:
            raise ValueError('Board golden check failed before flight')

        layout, route_hash = load_layout(args.environment_id, args.route_index)
        planner = StoredRoutePlanner(layout)
        first = planner.reference_window_for_trajectory(
            planner.trajectory, elapsed=0., dt=.1, horizon=21)[0]
        start = {'reference_phase_s': 0., 'position': first.position.tolist(),
                 'quaternion': first.orientation.tolist(), 'velocity': [0., 0., 0.]}
        options = SimpleNamespace(vehicle=args.vehicle, duration=max(20., planner.trajectory.duration + 2.))
        trial = {'model': f'IP37_model{args.model_id}', 'period_ms': args.period_ms,
                 'extra_delay_ms': 0., 'repeat': 0, 'scenario': 'startup', 'id': 'ip37_actual_control'}
        atomic_json(args.output / 'deployment.json', {
            'outputs_applied': True, 'controller': 'IP37 board DNN', 'mpc_fallbacks': 0,
            'endpoint': board.peer, 'wire_tag_hex': manifest['wire_tag_hex'], 'model': entry,
            'route_sha256': route_hash, 'environment_id': args.environment_id, 'route_index': args.route_index,
            'period_ms': args.period_ms, 'duration_s': options.duration,
            'gate_width_m': 4., 'gate_height_m': 4.,
        })
        load_airsim_client()
        from airsim_interface import require_airsim
        client = require_airsim().MultirotorClient(ip=args.airsim_host, port=41451, timeout_value=3.)
        client.confirmConnection()
        if args.vehicle not in client.listVehicles():
            raise ValueError(f'AirSim vehicle not present: {args.vehicle}')
        policy = BoardPolicy(board, args.model_id)
        print(f'Board DNN controls {args.vehicle}, T={args.period_ms:g}ms, '
              f'env={args.environment_id}, route={args.route_index}.', flush=True)
        result = flight(client, policy, layout, start, trial, options, args.output)
        crossings = result['crossings']
        result.update(
            outputs_applied=True, controller='IP37 board DNN', mpc_fallbacks=0,
            timing_and_observation_passed=result['comparable'],
            flight_passed=result['error'] is None and result['window_covered'] and
                          len(crossings) == 5 and all(c['valid'] for c in crossings),
        )
    except (Exception, KeyboardInterrupt) as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        if board is not None:
            board.close()
        if policy is not None:
            atomic_json(args.output / 'board_requests.json', policy.records)
        atomic_json(args.output / 'result.json', result)
        print(f"flight_passed={result['flight_passed']} error={result['error']}", flush=True)
        if 'crossings' in result:
            print(f"gates={sum(c['valid'] for c in result['crossings'])}/5 "
                  f"tracking_RMSE_m={result.get('tracking_rmse_m')} "
                  f"max_error_m={result.get('max_tracking_error_m')} "
                  f"skipped_releases={result['skipped_releases']}", flush=True)
        print(f'Report: {args.output / "result.json"}', flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--board', default='192.168.77.2')
    parser.add_argument('--board-port', type=int, default=5238)
    parser.add_argument('--model-id', type=int, choices=(11, 6), default=11)
    parser.add_argument('--environment-id', type=int, choices=range(48, 54), default=48)
    parser.add_argument('--route-index', type=int, choices=range(20), default=0)
    parser.add_argument('--vehicle', default='Drone1')
    parser.add_argument('--airsim-host', default='127.0.0.1')
    parser.add_argument('--period-ms', type=float, default=50.)
    result = run(parser.parse_args())
    raise SystemExit(0 if result['flight_passed'] else 1)
