"""Bounded offline gain/T/R screening; does not select or replace a live teacher."""
import argparse
from pathlib import Path

import numpy as np

from environment import Gate
from .decision_dataset import atomic_json, sha256
from .dynamic_probe_common import read_json
from .held_control_safety import GateHoldAnalyzer, HoldBounds
from .teacher_attitude_probe import simulate


GAINS = (4.5, 5., 7., 10., 14.)
PROFILES_MS = ((30., 7.), (120., 7.), (120., 52.), (120., 100.))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    case = read_json(args.case)
    atomic_json(args.output/'case_frozen.json', case)
    gates = [Gate(g['index'], g['center'], g['orientation']) for g in case['layouts']['Drone1']['gates']]
    geometry = GateHoldAnalyzer(gates, case['net_width_m'],
        HoldBounds(2., .01, .05, .15, 10., 'offline geometry only'),
        net_height=3.85, body_radius=.5, clearance=.025)
    report = dict(scope='ideal body-rate model; no AirSim, DNN, board or safety certificate',
        case_sha256=sha256(args.case), source_sha256=sha256(Path(__file__)),
        gains=GAINS, profiles_ms=PROFILES_MS, integration_step_s=.005,
        criteria='report all results; no gain selected by static-failure contrast', rows=[])
    for gain in GAINS:
        for period_ms, delay_ms in PROFILES_MS:
            row = dict(attitude_gain=gain, period_ms=period_ms, imposed_response_ms=delay_ms)
            try:
                result, trace = simulate(case, 'baseline', period=period_ms/1000,
                    delay=delay_ms/1000, attitude_gain=gain)
                margins = np.array([float((geometry.normals@np.asarray(r['position'])-geometry.offsets)
                                         .max(axis=1).min()) for r in trace])
                first_negative = np.flatnonzero(margins < 0.)
                crossed = [r for r in result['crossings'] if r['crossed']]
                row.update(status='completed_model', position_rmse_m=result['position_rmse_m'],
                    plane_crossings=len(crossed), expected_gates=len(gates),
                    min_sampled_frame_margin_m=float(margins.min()),
                    first_negative_margin_t_s=trace[int(first_negative[0])]['t'] if len(first_negative) else None,
                    rate_clipped_fraction=result['rate_clipped_fraction'], gate4=result['crossings'][3],
                    signature=result['signature'])
                atomic_json(args.output/f'K{gain:g}_T{period_ms:g}_R{delay_ms:g}_trace.json', trace)
                atomic_json(args.output/f'K{gain:g}_T{period_ms:g}_R{delay_ms:g}_detail.json', result)
            except Exception as exc:
                row.update(status='model_or_solver_inconclusive', error=f'{type(exc).__name__}: {exc}')
            report['rows'].append(row)
            atomic_json(args.output/'report.json', report)
            print({k: v for k, v in row.items() if k not in ('signature', 'gate4')}, flush=True)
    print(f'Offline report: {args.output / "report.json"}', flush=True)


if __name__ == '__main__':
    main()
