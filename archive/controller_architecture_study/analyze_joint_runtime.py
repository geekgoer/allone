"""Audit actual released configurations, front-end waits and paired flight outcomes."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from .decision_dataset import atomic_json
from .dynamic_probe_common import read_json


def stats(values):
    a=np.asarray(values,dtype=float)
    return dict(count=len(a),mean=float(a.mean()),p99=float(np.percentile(a,99)),max=float(a.max())) if len(a) else None


def audit(root):
    summary=read_json(root/'summary.json')
    if not summary['complete']:raise RuntimeError('paired trials have not completed')
    results={}
    for name in summary['trials']:
        directory=root/name;report=read_json(directory/'report.json')
        rows=read_json(directory/'requests.json');admissions=read_json(directory/'admissions.json')
        assert all(r['model']==r['committed_model'] and r['deadline_s']==r['committed_deadline_s']
            and r['planned_release_s']==r['committed_release_s'] for r in rows)
        modes={};counts={};switches={};gaps={}
        for v in report['vehicles']:
            sequence=sorted([r for r in rows if r['vehicle']==v],key=lambda r:r['planned_release_s'])
            counts[v]=dict(Counter(str(round(r['requested_period_s']*1000,3)) for r in sequence))
            changes=[];previous=None
            for r in sequence:
                mode=(r['model'],r['requested_period_s'])
                if mode!=previous:
                    changes.append(dict(t=r['job_released_s']-report['wall_origin_s'],model=mode[0],period_ms=1000*mode[1]))
                    previous=mode
            switches[v]=len(changes)-1;modes[v]=changes
            end=report['retired_vehicles'].get(v,{}).get('stopped_wall_s',report['wall_origin_s']+report['duration_s'])
            acks=sorted(r['rpc_return_s'] for r in rows if r['vehicle']==v and r.get('rpc_return_s',float('inf'))<=end)
            # Include the initial hold and the final open interval up to the
            # vehicle's observation boundary; an applied-only gap hides these.
            gaps[v]=stats(1000*np.diff([report['wall_origin_s'],*acks,end]))
        decisions=report['global_admission_decisions']
        initial=next((a for a in admissions if a['trigger']=='initial_sustained_demand'),None)
        results[name]=dict(collisions=report['collisions'],error=report['error'],
            physical_loops_passed=sum(v['physical_task_passed'] for v in report['vehicles'].values()),
            valid_crossings=sum(sum(c['valid'] for c in v['crossings']) for v in report['vehicles'].values()),
            response_ms=report['response_ms'],frontend_input_wait_ms=report['frontend_input_wait_ms'],
            command_gaps_ms=gaps,actual_mode_switches=switches,actual_configurations=modes,requested_period_counts=counts,
            global_admission_decisions=decisions,initial_sustained_demand=initial,
            admission_wall_ms=sum(a.get('planning_ms',0.) for a in admissions if not a.get('outside_physics')),
            published=len(rows),applied=report['applied'],superseded=report['superseded_before_dispatch'],
            nominal_late=len(report['nominal_late_jobs']),nominal_undelivered=len(report['nominal_undelivered_deadlines']),
            strict_timing_passed=report['observed_strict_timing_passed'],
            all_released_commitments_unchanged=True)
        print(name,'collisions',results[name]['collisions'],'loops',results[name]['physical_loops_passed'],
            'input_wait_ms',results[name]['frontend_input_wait_ms'],'switches',sum(switches.values()),
            'decisions',decisions,flush=True)
    for seed in summary['plan']['seeds']:
        a=read_json(root/f'seed{seed}_reference/initial_state.json');b=read_json(root/f'seed{seed}_joint/initial_state.json')
        assert a.keys()==b.keys() and all(a[v][k]==b[v][k] for v in a for k in ('position','velocity','quaternion','held_control'))
    atomic_json(root/'comparison_audit.json',dict(complete=True,paired_initials_match=True,trials=results,
        hard_realtime_certified=False,all_released_commitments_unchanged=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('source',type=Path)
    audit(parser.parse_args().source)
