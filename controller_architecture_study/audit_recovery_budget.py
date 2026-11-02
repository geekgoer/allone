"""Evaluate proposed B on saved states; no AirSim, model loading or new flights."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from environment import Gate
from .decision_dataset import ROOT, atomic_json, sha256
from .dynamic_probe_common import read_json
from .fair_static_dynamic import case_for
from .framework_runtime import analyze_budget
from .recovery_budget import RecoveryAssumptions


PROFILE=Path(__file__).with_name('configs')/'recovery_budget_prototype_v1.json'
SOURCE=ROOT/'controller_architecture_study/results/global_admission_20261029_v4'


def audit_trial(directory,case,assumptions):
    traces=read_json(directory/'control_traces.json')
    analyses=read_json(directory/'analyses.json')
    report=read_json(directory/'report.json')
    initial=read_json(directory/'initial_state.json')
    lookup={v:{r['sample_s']:r for r in rows} for v,rows in traces.items()}
    gates={}
    for v,layout in case['layouts'].items():
        shift=np.array(initial[v]['position'])-np.array(layout['segments'][0]['coefficients'])[:,0]
        expected=report['vehicles'][v]['expected_gate_ids']
        gates[v]=[Gate(g['index'],np.asarray(g['center'])+shift,g['orientation'])
            for g in layout['gates'] if g['index'] in expected]
    rows=[];unmatched=[];old_ms=[];new_ms=[];old_cpu_ms=[];new_cpu_ms=[]
    for analysis in analyses:
        v=analysis['vehicle'];trace=lookup[v].get(analysis['analysis_s'])
        if trace is None:
            unmatched.append(analysis['sample_id'])
            continue
        state=SimpleNamespace(position=np.array(trace['position']),velocity=np.array(trace['velocity']))
        args=(state,analysis['held_control'],analysis['analysis_s'],analysis['actuation_proxy_s'],gates[v],case['net_width_m'])
        start=time.perf_counter();cpu=time.thread_time()
        old=analyze_budget(*args)
        old_cpu_ms.append(1000*(time.thread_time()-cpu))
        old_ms.append(1000*(time.perf_counter()-start))
        if abs(old['budget_s']-analysis['budget_s'])>1e-8:
            raise RuntimeError('saved geometric budget could not be reproduced: '+analysis['sample_id'])
        start=time.perf_counter();cpu=time.thread_time()
        new=analyze_budget(*args,recovery_assumptions=assumptions)
        new_cpu_ms.append(1000*(time.thread_time()-cpu))
        end=time.perf_counter()
        new_ms.append(1000*(end-start))
        rows.append(dict(vehicle=v,sample_id=analysis['sample_id'],t=trace['t'],
            original_budget_s=old['budget_s'],recovery_budget_s=new['budget_s'],
            original_deadline_s=old['deadline_s'],recovery_deadline_s=new['deadline_s'],
            deadline_tightened=new['deadline_s']<old['deadline_s']-1e-10,
            valid=new['valid'],recovery=new['recovery'],analysis_wall_ms=new_ms[-1],
            analysis_started_s=start,analysis_finished_s=end))
    def first(items):
        return dict(t=items[0]['t'],sample_id=items[0]['sample_id']) if items else None
    vehicles={}
    for v in case['layouts']:
        items=[r for r in rows if r['vehicle']==v]
        tightened=[r for r in items if r['deadline_tightened']]
        invalid=[r for r in items if not r['valid']]
        failure=report['vehicles'][v]['failure']
        vehicles[v]=dict(matched_states=len(items),tightened_deadlines=len(tightened),
            first_tightened=first(tightened),no_positive_budget=len(invalid),first_no_positive_budget=first(invalid),
            outside_approach_domain=sum(any(not c['applicable'] for c in r['recovery']['constraints']) for r in items),
            observed_source_failure=failure,
            landmarks=[min(items,key=lambda r:abs(r['t']-t)) for t in (4.5,5.2,5.6,9.3)
                if items and t<=items[-1]['t']])
    def timing(a):
        return dict(count=len(a),mean=float(np.mean(a)),p99=float(np.percentile(a,99)),maximum=float(max(a)))
    summary=dict(source=str(directory),matched=len(rows),unmatched=len(unmatched),unmatched_ids=unmatched,
        original_geometry_reproduced=True,vehicles=vehicles,
        offline_analysis_ms=dict(original=timing(old_ms),recovery=timing(new_ms)),
        offline_thread_cpu_total_ms=dict(original=sum(old_cpu_ms),recovery=sum(new_cpu_ms)),
        cpu_timing_note='Windows thread_time is quantized; totals only, no per-call CPU percentiles',
        timing_scope='sequential offline calls on saved states, not shared-runtime WCET; wall and thread CPU reported separately')
    return summary,rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=SOURCE)
    parser.add_argument('--assumptions',type=Path,default=PROFILE)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    assumptions=RecoveryAssumptions(**read_json(args.assumptions))
    output=args.output or ROOT/'controller_architecture_study/results'/('recovery_budget_audit_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    output.mkdir(parents=True,exist_ok=False)
    trials={};collections=[];collection_start={}
    def record_collection(phase,info):
        generation=info['generation']
        if phase=='start':collection_start[generation]=time.perf_counter()
        else:collections.append((collection_start.pop(generation,time.perf_counter()),time.perf_counter()))
    gc.callbacks.append(record_collection)
    try:
        for seed in (4,5):
            label=f'seed{seed}_protected';directory=args.source/label
            summary,rows=audit_trial(directory,case_for(seed),assumptions)
            summary['slow_calls']=[dict(sample_id=r['sample_id'],wall_ms=r['analysis_wall_ms'],
                gc_overlap_ms=1000*sum(max(0.,min(end,r['analysis_finished_s'])-max(start,r['analysis_started_s']))
                    for start,end in collections)) for r in rows if r['analysis_wall_ms']>=5.]
            atomic_json(output/(label+'_states.json'),rows)
            trials[label]=summary
            print(label,'matched',summary['matched'],'unmatched',summary['unmatched'],flush=True)
            for v,row in summary['vehicles'].items():
                print(v,'tighter',row['tightened_deadlines'],'no_positive_B',row['no_positive_budget'],
                    'first_tighter',row['first_tightened'],flush=True)
    finally:
        gc.callbacks.remove(record_collection)
    payload=dict(assumptions=asdict(assumptions),assumptions_sha256=sha256(args.assumptions),
        source_sha256={name:sha256(Path(__file__).with_name(name)) for name in
            ('recovery_budget.py','framework_runtime.py','heterogeneous_motivation.py','fair_static_dynamic.py',Path(__file__).name)},
        input_sha256={label:{name:sha256(args.source/label/name) for name in
            ('analyses.json','control_traces.json','initial_state.json','report.json')} for label in trials},
        trials=trials,new_flight_run=False,hard_safety_certified=False,
        scope='saved-state budget/overhead audit; earlier warning is not proof that online flight is rescued')
    atomic_json(output/'summary.json',payload)
    print('Offline audit:',output/'summary.json',flush=True)


if __name__=='__main__':main()
