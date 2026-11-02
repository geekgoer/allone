"""Audit completed before/after runs without changing any flight records."""
import argparse
from pathlib import Path

import numpy as np

from .decision_dataset import atomic_json
from .dynamic_probe_common import read_json


def stats(values):
    a=np.asarray(values,dtype=float)
    return dict(count=len(a),mean=float(a.mean()),p99=float(np.percentile(a,99)),max=float(a.max())) if len(a) else None


def analyze(root):
    summary=read_json(root/'summary.json');rows={}
    for name in summary['trials']:
        directory=root/name;report=read_json(directory/'report.json');requests=read_json(directory/'requests.json')
        for r in requests:
            for field,committed in [('model','committed_model'),('deadline_s','committed_deadline_s'),
                    ('planned_release_s','committed_release_s')]:
                if r[field]!=r[committed]:raise RuntimeError('released commitment changed: '+r['sample_id'])
        active=[r for r in requests if 'completion_frontend_start_s' in r and 'io_response_received_s' in r
            and r['completion_frontend_start_s']<=report['retired_vehicles'].get(r['vehicle'],{}).get('stopped_wall_s',float('inf'))]
        gaps={}
        for v in report['vehicles']:
            start=report['wall_origin_s'];end=min(start+report['duration_s'],
                report['retired_vehicles'].get(v,{}).get('stopped_wall_s',float('inf')))
            times=[start]+sorted(r['rpc_return_s'] for r in requests if r['vehicle']==v
                and start<=r.get('rpc_return_s',float('inf'))<=end)+[end]
            gaps[v]=stats(1000*np.diff(times))
        vehicle_reports=report['vehicles']
        rows[name]=dict(collisions=report['collisions'],
            physical_loops_passed=sum(v['physical_task_passed'] for v in vehicle_reports.values()),
            valid_gates=sum(len({g['gate'] for g in v['crossings'] if g['valid']}) for v in vehicle_reports.values()),
            expected_gates=sum(len(v['expected_gate_ids']) for v in vehicle_reports.values()),
            response_ms=report['response_ms'],input_age_at_command_ms=report['capture_to_command_ms'],
            published=report['requests'],applied=report['applied'],superseded=report['superseded_before_dispatch'],
            nominal_late=len(report['nominal_late_jobs']),nominal_undelivered=len(report['nominal_undelivered_deadlines']),
            physical_deadline_late=len(report['late_jobs']),strict_timing_passed=report['observed_strict_timing_passed'],
            active_cleanup_wait_ms=stats([1000*(r['completion_frontend_start_s']-r['io_response_received_s']) for r in active]),
            io_return_to_slot_release_ms=report['io_return_to_slot_release_ms'],
            command_gaps_ms=gaps,admission_decisions=report['global_admission_decisions'],
            total_J=report['total_J'],error=report['error'],cleanup_errors=report['cleanup_errors'])
    payload=dict(source=str(root),complete=summary['complete'],trials=rows,
        all_released_commitments_preserved=True,hard_realtime_certified=False,
        note='Reported R is for applied jobs; superseded and undelivered requests remain separate failures/skips. Cleanup delays after retirement are excluded.')
    atomic_json(root/'repair_audit.json',payload)
    for name,r in rows.items():
        print(name,'collision',r['collisions'],'loops',r['physical_loops_passed'],
            'gates',f"{r['valid_gates']}/{r['expected_gates']}",'R',r['response_ms'],
            'cleanup',r['active_cleanup_wait_ms'],'skips',r['superseded'],flush=True)
    return payload


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    analyze(parser.parse_args().root)
