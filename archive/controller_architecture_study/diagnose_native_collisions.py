"""Read recorded release/I/O/command timelines; optionally isolate failing routes."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from .decision_dataset import ROOT,atomic_json
from .dynamic_probe_common import read_json


SOURCE=ROOT/'controller_architecture_study/results/native_runtime_flight_20261029_v3'


def distribution(values):
    a=np.asarray(values,dtype=float)
    return dict(count=len(a),mean=float(a.mean()),p99=float(np.percentile(a,99)),
        max=float(a.max()),over10ms=int((a>10.).sum())) if len(a) else None


def audit(source,output):
    result={}
    for folder in ('seed4_python','seed4_cpp','seed5_cpp','seed5_python'):
        root=source/folder;report=read_json(root/'report.json');origin=report['wall_origin_s']
        requests=read_json(root/'requests.json');analyses=read_json(root/'analyses.json')
        admissions=read_json(root/'admissions.json');segments=read_json(root/'segments.json')
        fields=dict(submit_to_io_owner_ms=('sample_submitted_s','sample_helper_received_s'),
            snapshot_ms=('sample_io_started_s','sample_io_finished_s'),
            input_return_ms=('sample_io_finished_s','sample_response_received_s'),
            command_ack_to_poststate_ms=('rpc_return_s','io_finished_s'),
            poststate_return_to_cleanup_ms=('io_response_received_s','completion_frontend_start_s'))
        row={name:distribution([1000*(r[b]-r[a]) for r in requests if a in r and b in r])
            for name,(a,b) in fields.items()}
        active_cleanup=[r for r in requests if 'completion_frontend_start_s' in r and 'io_response_received_s' in r
            and r['completion_frontend_start_s']<=report['retired_vehicles'].get(r['vehicle'],{}).get('stopped_wall_s',float('inf'))]
        row['active_poststate_return_to_cleanup_ms']=distribution([1000*(r['completion_frontend_start_s']-r['io_response_received_s'])
            for r in active_cleanup])
        if active_cleanup:
            worst=max(active_cleanup,key=lambda r:r['completion_frontend_start_s']-r['io_response_received_s'])
            row['worst_active_cleanup']=dict(sample_id=worst['sample_id'],
                ack_since_origin_s=worst['rpc_return_s']-origin,
                io_received_after_ack_ms=1000*(worst['io_response_received_s']-worst['rpc_return_s']),
                cleanup_after_ack_ms=1000*(worst['completion_frontend_start_s']-worst['rpc_return_s']))
        changes=Counter((round(a.get('proposed_period_s',a['requested_period_s'])*1000,3),
            round(a['requested_period_s']*1000,3)) for a in analyses)
        row.update(admission_decisions=report['global_admission_decisions'],
            admission_reasons=dict(Counter(a.get('reason','initial_or_accepted') for a in admissions)),
            proposed_vs_retained_period_ms=[dict(proposed=p,retained=r,count=n) for (p,r),n in changes.items()],
            collisions=[])
        for vehicle,event in report['retired_vehicles'].items():
            end=event['stopped_wall_s'] # Detection/retirement boundary; not an exact physical actuation clock.
            applied=sorted((r for r in requests if r['vehicle']==vehicle and r.get('rpc_return_s',float('inf'))<=end),
                key=lambda r:r['rpc_return_s'])
            if not applied:continue
            intervals=[(b['rpc_return_s']-a['rpc_return_s'],a,b) for a,b in zip(applied,applied[1:])
                if b['rpc_return_s']>=end-1.]
            intervals.append((end-applied[-1]['rpc_return_s'],applied[-1],None))
            gap,previous,following=max(intervals,key=lambda item:item[0])
            begin=previous['rpc_return_s'];stop=following['rpc_return_s'] if following else end
            def relative(r,keys):return {k:1000*(r[k]-begin) for k in keys if k in r}
            record=dict(vehicle=vehicle,collision=event,gap_ms=1000*gap,
                gap_start_since_origin_s=begin-origin,gap_end_since_origin_s=stop-origin,
                previous_job=previous['sample_id'],following_job=following['sample_id'] if following else None,
                previous_after_ack_ms=relative(previous,('io_finished_s','io_response_received_s','next_plan_started_s',
                    'next_plan_done_s','completion_frontend_start_s','completion_frontend_done_s')),
                backend_segment_wall_ms=1000*sum(max(0.,min(stop,s['end_s'])-max(begin,s['start_s'])) for s in segments),
                note='Largest command gap intersecting final 1s before collision detection; final open gap is censored at detection.')
            if following:
                record.update(following_after_previous_ack_ms=relative(following,('job_released_s','dispatch_s','sample_s',
                    'feature_start_s','feature_done_s','first_service_s','ready_s','rpc_return_s')),
                    following_compute_ms=following.get('compute_ms'),following_model=following['model'])
            row['collisions'].append(record)
        result[folder]=row
    atomic_json(output/'recorded_causes.json',dict(source=str(source),trials=result,
        queue_wait_is_not_necessarily_state_age=True,exact_OS_cause_of_native_wakeup_tail_unproved=True))
    return result


def isolated(output):
    """Same route/model, one loop, measured small R; no learned policy changes."""
    from .fair_static_dynamic import MODEL_BANK,case_for
    from .framework_cpu_affinity import CpuPlacement
    from .framework_runtime import ControllerPool
    from .framework_runtime_world import FrameworkWorld
    from .probe_collision_root import single
    from .convex_teacher import ConvexTeacher
    from .fly_convex_policy import load_airsim_client
    from .probe_airsim_three import translate_layout
    load_airsim_client()
    original=read_json(SOURCE/'plan.json');configuration=original['configuration']
    cases=[(4,'Drone1'),(4,'Drone2'),(5,'Drone9')]
    results={}
    with CpuPlacement() as placement:
        pool=ControllerPool(model_names=MODEL_BANK);teacher=ConvexTeacher()
        if pool.provenance()!=original['model_provenance']:raise RuntimeError('model artifacts changed')
        for seed,vehicle in cases:
            case=case_for(seed);layout=case['layouts'][vehicle];model=configuration['models'][vehicle]
            original_initial=read_json(SOURCE/f'seed{seed}_cpp'/'initial_state.json')[vehicle]
            # SafetyWorld centres N lanes at 12m spacing. Reusing Drone1 for
            # one source loop must preserve that loop's LOCAL reference/state.
            layout=translate_layout(layout,[0.,-(case['count']-1)*6.,0.])
            # Test 30ms first. Only if it fails, try 20ms, then teacher20.
            for kind,period in (('dnn',.030),('dnn',.020),('teacher',.020)):
                label=f'seed{seed}_{vehicle}_{kind}{int(period*1000)}';directory=output/label
                if (directory/'report.json').exists():payload=read_json(directory/'report.json')
                else:
                    world=FrameworkWorld(layout,0.,1,case['net_width_m'],task_duration=case['duration_s'],layouts={'Drone1':layout})
                    try:
                        payload=single(world,pool,teacher,model,kind,period,directory,
                            dict(id=label,critical_index=-1),case['duration_s'],expected_initial=original_initial)
                        actual=read_json(directory/'initial_state.json')['Drone1']
                        matched=all(np.allclose(actual[k],original_initial[k],rtol=0.,atol=1e-6)
                            for k in ('position','velocity','quaternion','held_control'))
                        if not matched:raise RuntimeError('isolated local initial state differs')
                        atomic_json(directory/'source.json',dict(source_vehicle=vehicle,seed=seed,model=model,
                            local_initial_matched=matched,source_trial=str(SOURCE/f'seed{seed}_cpp'),
                            single_loop_direct_io=True,not_a_multi_loop_capacity_claim=True))
                    finally:atomic_json(output/(label+'_cleanup.json'),world.close())
                results[label]=dict(model=model,physical_task_passed=payload['result']['physical_task_passed'],
                    failure=payload['result']['failure'],input_to_command_ms=payload['input_to_command_ms'],
                    command_gap_ms=payload['command_gap_ms'])
                atomic_json(output/'isolated_summary.json',dict(complete=False,cases=results))
                if payload['result']['physical_task_passed']:break
        atomic_json(output/'isolated_summary.json',dict(complete=True,cases=results,
            meaning='Successful isolated flight demonstrates capability for that route/initial state at the measured timing only.'))
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'controller_architecture_study/results/native_collision_cause_20261029_v1')
    parser.add_argument('--isolated',action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    audit(SOURCE,args.output)
    if args.isolated:print(isolated(args.output),flush=True)
    print('Diagnosis:',args.output,flush=True)


if __name__=='__main__':main()
