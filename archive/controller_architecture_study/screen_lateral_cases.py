"""Make three paired natural-route cases and run a small real-board screening."""
import argparse
from copy import deepcopy
from pathlib import Path
import json
import subprocess
import sys

import numpy as np

from .decision_dataset import atomic_json, sha256
from .ip37_safety_cases import audit, evaluate


def scale_yaw(q, factor):
    yaw = 2*np.arctan2(q[3], q[0])
    yaw = np.arctan2(factor*np.sin(yaw), np.cos(yaw))
    return [float(np.cos(yaw/2)), 0., 0., float(np.sin(yaw/2))]


def transform(source, factor):
    if not np.isfinite(factor) or not 0. < factor <= 1.:
        raise ValueError('lateral scale must be finite and in (0,1]')
    case = deepcopy(source)
    case['net_width_m'], case['gate_center_width_m'] = 3., 3.15
    case['lateral_scale'] = factor
    if factor == 1.:
        return case
    for layout in case['layouts'].values():
        origin = layout['segments'][0]['coefficients'][1][0]
        for segment in layout['segments']:
            row = segment['coefficients'][1]
            row[0] = origin+factor*(row[0]-origin)
            row[1:] = [factor*x for x in row[1:]]
            segment['heading_orientation'] = scale_yaw(segment['heading_orientation'], factor)
        for gate in layout['gates']:
            gate['center'][1] = origin+factor*(gate['center'][1]-origin)
            gate['orientation'] = scale_yaw(gate['orientation'], factor)
            for key in ('reference_velocity', 'reference_acceleration'):
                if key in gate:
                    gate[key][1] *= factor
            if 'curvature_per_m' in gate:
                v = np.asarray(gate['reference_velocity'])
                a = np.asarray(gate['reference_acceleration'])
                gate['curvature_per_m'] = float(np.linalg.norm(np.cross(v, a))/max(np.linalg.norm(v)**3, 1e-9))
    return case


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--prepare-only', action='store_true')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = json.loads(args.source.read_text(encoding='utf-8'))
    rows = []
    for factor in (1., .7, .4):
        case = transform(source, factor)
        case['screen_source_sha256'] = sha256(args.source)
        path = args.output/f'case_{factor:g}.json'
        check = audit(case)
        atomic_json(args.output/f'audit_{factor:g}.json', check)
        atomic_json(path, case)
        if not check['passed']:
            raise RuntimeError(f'geometry audit failed: {path}')
        for repeat in range(2):
            rows.append(dict(scale=factor, repeat=repeat, case=str(path)))
    atomic_json(args.output/'plan.json', rows)
    if args.prepare_only:
        print('Prepared six flights; no flight started.', flush=True)
        return
    for row in rows:
        directory = args.output/f"scale_{row['scale']:g}_repeat_{row['repeat']}"
        print(f"SCREEN scale={row['scale']} repeat={row['repeat']}", flush=True)
        cmd = [sys.executable, '-u', '-m', 'controller_architecture_study.fly_ip37_joint',
            '--bundle', str(args.bundle), '--board', '192.168.77.2', '--board-port', '5238',
            '--case', row['case'], '--policy', 'fixed', '--period-ms', '50',
            '--vehicles', 'Drone1', '--evaluation-ms', '50', '--output', str(directory)]
        subprocess.run(cmd, check=False)
        report = directory/'result.json'
        if not report.exists():
            raise RuntimeError(f'missing report: {directory}')
        result = json.loads(report.read_text(encoding='utf-8'))
        vehicle = result['vehicles'].get('Drone1', {})
        margins = [g['body_margin_m'] for g in vehicle.get('crossings', []) if g.get('body_margin_m') is not None]
        row.update(directory=str(directory), flight=result['flight_passed'],
                   body=result['body_clearance_passed'], timing=result['timing_before_contact_passed'],
                   clock=result['sim_clock_passed'], outcomes=result['outcomes'], errors=result['errors'],
                   contact=vehicle.get('contact'), min_observed_margin_m=min(margins) if margins else None,
                   tracking_rmse_m=vehicle.get('tracking_rmse_m'), response_ms=result['response_ms'])
        atomic_json(args.output/'summary.json', rows)
        fatal = [e for e in result['errors'] if not (e.startswith('Drone1:') and
                 ('IP37 status=3' in e or 'flight guard exceeded' in e))]
        if fatal or not result['sim_clock_passed']:
            raise RuntimeError(f'infrastructure or input error; inspect {report}')
    print(f"Summary: {args.output/'summary.json'}", flush=True)


if __name__ == '__main__':
    main()
