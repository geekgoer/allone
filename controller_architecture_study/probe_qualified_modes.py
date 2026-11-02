"""Offline cost-ordered command qualification on existing real-flight states.

No counterfactual flight/recovery claim. Every attempted model's inference
cost is measured, including rejected candidates. Current runtime is untouched.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from config import load_calibration
from control_adapter import control_to_airsim_command, throttle_to_thrust
from environment import Gate
from .decision_dataset import atomic_json
from .fair_static_dynamic import case_for, MODEL_BANK
from .fast_convex_preprocess import FastReferenceWindow
from .fly_decision_dnn import StoredRoutePlanner
from .framework_runtime import ControllerPool
from .framework_runtime_features import encode_direct, reference_point
from .held_control_safety import GateHoldAnalyzer, HoldBounds, nominal_acceleration
from .probe_airsim_three import translate_layout
from .audit_held_control import distribution


def qualify_output(state, control, reference, analyzer, hold_s, *, position_limit, velocity_limit):
    """Sufficient one-hold geometry + endpoint tracking bounds, NOT recovery.

    Uses the same Taylor remainder as the barrier. Endpoint tracking limits
    explicitly encode mission progress relative to the saved reference. This
    is a local contract only, not recursive feasibility or global stability.
    """
    safety = analyzer.analyze(state, control)
    a = nominal_acceleration(state.quaternion, control, analyzer.bounds.gravity)
    j = float(control[0]*np.linalg.norm(control[1:]))
    b = analyzer.bounds
    ep = np.linalg.norm(state.position+hold_s*state.velocity+.5*hold_s**2*a-reference.position)
    ep += b.position_error_m+b.velocity_error_mps*hold_s+.5*b.acceleration_error_mps2*hold_s**2+j*hold_s**3/6
    ev = np.linalg.norm(state.velocity+hold_s*a-reference.velocity)
    ev += b.velocity_error_mps+b.acceleration_error_mps2*hold_s+.5*j*hold_s**2
    good = safety['budget_s'] >= hold_s and ep <= position_limit and ev <= velocity_limit
    return dict(status='qualified' if good else 'unqualified', budget_s=safety['budget_s'],
                endpoint_position_bound_m=float(ep), endpoint_velocity_bound_mps=float(ev),
                scope='conditional one-hold contract only; not an offline recovery label')


def probe(directory, seed, output, samples=36, max_checks=2):
    if samples < 1 or max_checks < 1:
        raise ValueError('positive snapshot count and finite candidate count required')
    if output.exists():
        raise FileExistsError('existing evidence is immutable')
    traces = json.loads((directory/'control_traces.json').read_text())
    initial = json.loads((directory/'initial_state.json').read_text())
    report = json.loads((directory/'report.json').read_text())
    case = case_for(seed)
    pool = ControllerPool(model_names=MODEL_BANK)
    calibration = load_calibration()
    costs = {m:sum(report['worker_profile'][m]['segment_estimates_s']) for m in MODEL_BANK}
    names = sorted(costs, key=lambda m:(costs[m], m))
    records = []
    # Evenly spaced physical snapshots across all loops, NOT selected by outcome.
    locations = [(v, i) for v, rows in traces.items() for i in range(0, len(rows), 8) if not rows[i]['collision']]
    chosen = np.linspace(0, len(locations)-1, min(samples, len(locations)), dtype=int)
    for index in chosen:
        v, i = locations[index]
        row = traces[v][i]
        shift = np.array(initial[v]['position'])-np.array([0., 0., -4.])
        layout = translate_layout(case['layouts'][v], shift)
        planner = StoredRoutePlanner(layout)
        window = FastReferenceWindow(planner.trajectory, planner.config)
        gates = [Gate(g['index'], g['center'], g['orientation']) for g in layout['gates']]
        bounds = HoldBounds(2., .01, .05, .15, 10., 'Same unverified development sensitivity tube as replay audit')
        analyzer = GateHoldAnalyzer(gates, case['net_width_m'], bounds)
        state = SimpleNamespace(**{k:np.array(row[k]) for k in ('position','velocity','quaternion')})
        feature = encode_direct(window, state, row['t'])
        ref = reference_point(window, row['t']+.050)
        attempts, selected = [], None
        began = time.perf_counter()
        for name in names[:max_checks]:
            infer = time.perf_counter()
            raw = pool.predict(name, feature)
            inference_s = time.perf_counter()-infer
            command = control_to_airsim_command(raw, calibration)
            physical = [throttle_to_thrust(command.throttle, calibration), command.roll_rate,
                        -command.pitch_rate, -command.yaw_rate]
            check_start = time.perf_counter()
            evidence = qualify_output(state, physical, ref, analyzer, .050,
                position_limit=max(.30, .995*row['position_error_m']),
                velocity_limit=max(1., .995*row['velocity_error_mps']))
            attempts.append(dict(model=name, inference_s=inference_s,
                                 qualification_s=time.perf_counter()-check_start, **evidence))
            if evidence['status'] == 'qualified':
                selected = name
                break
        records.append(dict(vehicle=v, t=row['t'], selected=selected, attempts=attempts,
                            total_selection_s=time.perf_counter()-began))
    result = dict(scope='offline local command contracts; no deployed mode switching or recovery proof',
        source=str(directory), seed=seed, models_order=names, snapshots=len(records),
        max_checks=max_checks, minimum_over_unchecked_modes_claimed=False,
        selected_counts=dict(Counter(r['selected'] or 'none_qualified' for r in records)),
        overhead_ms=distribution([1000*r['total_selection_s'] for r in records]),
        attempted_inference_including_rejections_ms=distribution([1000*sum(a['inference_s'] for a in r['attempts']) for r in records]),
        positive_delay_certified=False, recursive_recovery_certified=False, deployed=False,
        limitation='Stale-state/selection latency must be propagated through the old held-control tube before deployment; this zero-delay probe cannot authorize a future mode.',
        rows=records)
    atomic_json(output, result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--samples', type=int, default=36)
    p.add_argument('--max-checks', type=int, default=2)
    args = p.parse_args()
    probe(args.directory, args.seed, args.output, args.samples, args.max_checks)


if __name__ == '__main__':
    main()
