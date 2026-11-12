"""Compare board dispatch and ACK-follow-up placement with one fixed flight calendar."""
import argparse
from collections import Counter
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from .audit_ip37_frontend_admission import inspect
from .decision_dataset import atomic_json, sha256
from .dynamic_probe_common import read_json
from .fly_ip37_joint import load_flight_layouts


VEHICLES = tuple(f'Drone{i}' for i in range(1, 7))
PERIOD_MS = 80
ARMS = (
    ('fifo_immediate', 'fifo', ()),
    ('edf_immediate', 'edf', ()),
    ('edf_slack', 'edf', ('--refresh-when-slack',)),
    ('edf_deferred', 'edf', ('--refresh-at-release',)),
)


def run(bundle, case, output, plan_only):
    _, _, frozen = load_flight_layouts(SimpleNamespace(case=case, gate_width=None, duration=None, vehicles=VEHICLES))
    target_gate = next(g for g in frozen['layouts']['Drone1']['gates'] if g['index'] == 4)
    if abs(target_gate['reference_time_s']-5.19) > .005:
        raise ValueError('case does not match the confirmed early Drone1 window')
    common = [sys.executable, '-u', '-m', 'controller_architecture_study.fly_ip37_joint',
              '--bundle', str(bundle.resolve()), '--case', str(case.resolve()),
              '--policy', 'fixed', '--period-ms', str(PERIOD_MS), '--native-sampling',
              '--evaluation-ms', '50', '--vehicles', *VEHICLES]
    plan = []
    for repeat, arms in enumerate((ARMS, ARMS[::-1]), 1):
        for name, order, flags in arms:
            directory = output/f'r{repeat}_{name}'
            plan.append(dict(repeat=repeat, arm=name, directory=str(directory.resolve()),
                             argv=common+['--output', str(directory.resolve()), '--order', order, *flags]))
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output/'plan.json', dict(case_sha256=sha256(case), vehicles=VEHICLES,
        target_gate_reference_s=target_gate['reference_time_s'], commands=plan))
    print(f'{len(plan)} trials, same 6-vehicle calendar at {PERIOD_MS} ms. Plan {output/"plan.json"}', flush=True)
    if plan_only:
        return
    summary = []
    for entry in plan:
        directory = Path(entry['directory'])
        code = subprocess.run(entry['argv'], check=False).returncode
        if not (directory/'setup.json').is_file() or not (directory/'requests.json').is_file():
            raise RuntimeError(f'flight preflight failed, inspect {directory}')
        setup = read_json(directory/'setup.json')
        rows = read_json(directory/'requests.json')
        result = read_json(directory/'result.json')
        if setup.get('origin_s') is None or not rows:
            raise RuntimeError(f'no flight timeline, inspect {directory}')
        report = inspect(rows, setup, True)
        cross = next((g for g in result['vehicles']['Drone1']['crossings'] if g['gate'] == 4), None)
        row = dict(arm=entry['arm'], repeat=entry['repeat'], exit_code=code,
                   flight_passed=result['flight_passed'], sim_clock_passed=result['sim_clock_passed'],
                   critical=report['critical'], critical_records_missing=2-len(report['critical']),
                   busy_by_stage=report['busy_by_stage'],
                   outcomes=dict(Counter(r['status'] for r in rows)), gate4=cross,
                   contacts={name: info['contact'] for name, info in result['vehicles'].items()})
        summary.append(row)
        atomic_json(output/'summary.json', dict(complete=len(summary)==len(plan), trials=summary))
        print(entry['arm'], 'critical', [x['status'] for x in row['critical']], 'gate4', cross, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    run(args.bundle, args.case, args.output, args.plan_only)
