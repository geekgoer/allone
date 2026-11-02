"""Replay control-aware budgets on immutable flight logs; never controls AirSim."""

import argparse
from bisect import bisect_right
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import psutil

from environment import Gate
from .decision_dataset import atomic_json, sha256
from .fair_static_dynamic import case_for
from .held_control_safety import GateHoldAnalyzer, HoldBounds, nominal_acceleration


def distribution(values):
    a = np.array(values, dtype=float)
    return None if not len(a) else dict(count=len(a), mean=float(a.mean()),
        p95=float(np.percentile(a, 95)), p99=float(np.percentile(a, 99)), max=float(a.max()))


def audit(directory, seed, output, stride=8):
    if output.exists():
        raise FileExistsError('choose a new report path; existing evidence is immutable')
    rows = json.loads((directory/'requests.json').read_text())
    traces = json.loads((directory/'control_traces.json').read_text())
    initial = json.loads((directory/'initial_state.json').read_text())
    case = case_for(seed)
    # A SENSITIVITY assumption, not a fitted or certified AirSim disturbance.
    bounds = HoldBounds(2., .01, .05, .15, 10.,
        'Development sensitivity only: acceleration tube 2 m/s², state errors 1 cm / .05 m/s. NOT plant-certified.')
    records, durations, residuals = [], [], []
    for v, trace in traces.items():
        shift = np.array(initial[v]['position'])-np.array([0., 0., -4.])
        gates = [Gate(g['index'], np.array(g['center'])+shift, g['orientation']) for g in case['layouts'][v]['gates']]
        analyzer = GateHoldAnalyzer(gates, case['net_width_m'], bounds)
        commands = sorted((r for r in rows if r['vehicle'] == v and 'rpc_return_s' in r), key=lambda r:r['rpc_return_s'])
        times = [r['rpc_return_s'] for r in commands]
        for row in trace[::stride]:
            index = bisect_right(times, row['sample_s'])-1
            u = commands[index]['mapped_control'] if index >= 0 else initial[v]['held_control']
            state = SimpleNamespace(**{k:np.array(row[k]) for k in ('position', 'velocity', 'quaternion')})
            began = time.perf_counter()
            result = analyzer.analyze(state, u)
            durations.append(1000*(time.perf_counter()-began))
            residual = float(np.linalg.norm(np.array(row['acceleration'])-nominal_acceleration(state.quaternion, u)))
            residuals.append(residual)
            records.append(dict(vehicle=v, t=row['t'], **result, instantaneous_acceleration_residual_mps2=residual))
    report = dict(scope='offline sensitivity replay, not new flight or certification', seed=seed,
        source=str(directory), source_hashes={f:sha256(directory/f) for f in ('requests.json','control_traces.json','initial_state.json')},
        method='Yang-style entire-hold HOCBF lower bound for actual mapped controls',
        cpu_affinity=psutil.Process().cpu_affinity(),
        overhead_ms=distribution(durations), positive_budget_fraction=sum(r['valid'] for r in records)/len(records),
        below20ms=sum(r['budget_s'] < .020 for r in records), records=len(records),
        instantaneous_acceleration_residual_mps2=distribution(residuals),
        residual_exceeds_assumed_2mps2=sum(r > 2. for r in residuals),
        timestamp_alignment='post-command state and RPC acknowledgement proxy; not exact plant actuation time',
        residual_scope='instantaneous sanity check only; does not validate a complete held-interval tube',
        plant_bounds_verified=False, physical_safety_certified=False, rows=records)
    atomic_json(output, report)
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    audit(args.directory, args.seed, args.output)


if __name__ == '__main__':
    main()
