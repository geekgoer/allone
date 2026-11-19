"""Read the repaired trials and isolate the remaining Drone10 route failure."""
import argparse
from collections import Counter
from pathlib import Path

from .decision_dataset import ROOT,atomic_json
from .dynamic_probe_common import read_json
from .diagnose_native_collisions import distribution


SOURCE=ROOT/'controller_architecture_study/results/runtime_backlog_repair_20261029_v5'


def audit(output):
    result={}
    for seed in (4,5):
        root=SOURCE/f'seed{seed}_after';report=read_json(root/'report.json')
        rows=read_json(root/'requests.json');analyses=read_json(root/'analyses.json')
        admissions=read_json(root/'admissions.json');origin=report['wall_origin_s']
        end=report['retired_vehicles']['Drone10']['stopped_wall_s']
        applied=sorted((r for r in rows if r['vehicle']=='Drone10' and r.get('rpc_return_s',float('inf'))<=end),
            key=lambda r:r['rpc_return_s'])
        intervals=[(b['rpc_return_s']-a['rpc_return_s'],a,b) for a,b in zip(applied,applied[1:])
            if b['rpc_return_s']>=end-1.]
        gap,previous,following=max(intervals,key=lambda r:r[0]);begin=previous['rpc_return_s']
        stages={};cursor=begin
        for label,key in [('release_wait','job_released_s'),('released_to_dispatch','dispatch_s'),
                ('io_owner_wait','sample_io_started_s'),('state_collision_rpc','sample_io_finished_s'),
                ('sample_return','sample_response_received_s'),('frontend_input_wait','frontend_state_start_s'),
                ('input_observation','feature_start_s'),
                ('feature_build','feature_done_s'),('backend_wait','first_service_s'),
                ('backend_wall','ready_s'),('output_wait','command_send_s'),('command_rpc','rpc_return_s')]:
            value=max(cursor,following.get(key,cursor));stages[label]=1000*(value-cursor);cursor=value
        d10=[a for a in analyses if a['vehicle']=='Drone10']
        costs=report['effective_pipeline_costs']
        window_start=following['sample_response_received_s'];window_end=following['frontend_state_start_s']
        competing=[]
        for row in rows:
            for name,start,stop in [('observe','frontend_state_start_s','feature_start_s'),
                    ('encode','feature_start_s','feature_done_s'),
                    ('cleanup','completion_frontend_start_s','completion_frontend_done_s'),
                    ('plan','next_plan_started_s','next_plan_done_s')]:
                if start in row and stop in row and row[stop]>window_start and row[start]<window_end:
                    competing.append(dict(job=row['sample_id'],stage=name,
                        start_ms=1000*(max(window_start,row[start])-window_start),
                        duration_ms=1000*(min(window_end,row[stop])-max(window_start,row[start]))))
        switches=[];last=None
        for r in sorted((r for r in rows if r['vehicle']=='Drone10'),key=lambda r:r['job_released_s']):
            mode=(r['model'],r['requested_period_s'])
            if mode!=last:switches.append(dict(t=r['job_released_s']-origin,model=mode[0],T_ms=1000*mode[1]));last=mode
        failures=[]
        for a in admissions:
            if not a['trigger'].startswith('Drone10:'):continue
            for branch in a.get('attempts',[]):
                for trial in branch.get('attempts',[]):
                    # The simulator stops at its first violated deadline. Missing
                    # finish entries are not independent proofs of other misses.
                    failures.extend(k for k in trial['unmet_deadline_ids'] if k in trial['finish'])
        phases={}
        for label,lower,upper in [('easy',0.,5.),('bend',8.,end-origin)]:
            subset=[a for a in d10 if lower<=a['analysis_s']-origin<upper]
            phases[label]=dict(count=len(subset),requested_20=sum(a.get('proposed_period_s',a['requested_period_s'])<=.020001 for a in subset),
                retained_20=sum(a['requested_period_s']<=.020001 for a in subset),
                B_zero=sum(a['budget_s']<=0 for a in subset),
                upcoming=sum(bool((a.get('anticipation') or {}).get('upcoming')) for a in subset),
                urgent=sum(bool((a.get('anticipation') or {}).get('urgent')) for a in subset))
        result[f'seed{seed}']=dict(collision=report['retired_vehicles']['Drone10'],switches=switches,
            phases=phases,completed_forecast_miss_vehicles=dict(Counter(k.split(':')[0] for k in failures)),
            frozen_frontend_ms={k:1000*costs[k] for k in ('input_s','output_s','planner_reserve_s')},
            nominal_T30_frontend_demand=10*sum(costs[k] for k in ('input_s','output_s','planner_reserve_s'))/.030,
            competing_frontend_work=competing,
            per_vehicle_danger_counts={v:dict(analyses=len(subset:=[a for a in analyses if a['vehicle']==v]),
                B_zero=sum(a['budget_s']<=0 for a in subset),
                urgent=sum(bool((a.get('anticipation') or {}).get('urgent')) for a in subset),
                upcoming=sum(bool((a.get('anticipation') or {}).get('upcoming')) for a in subset)) for v in report['vehicles']},
            final_1s_worst_closed_command_gap=dict(ms=1000*gap,begin_t=begin-origin,end_t=following['rpc_return_s']-origin,
                previous=previous['sample_id'],following=following['sample_id'],stages_ms=stages,
                compute_ms=following['compute_ms'],input_source=following['input_source']),
            last_commands=[dict(t=r['rpc_return_s']-origin,T_ms=1000*r['requested_period_s'],
                R_ms=1000*(r['rpc_return_s']-r['job_released_s']),age_ms=1000*(r['rpc_return_s']-r['sample_s']),
                compute_ms=r['compute_ms'],job=r['sample_id'],control=r['mapped_control']) for r in applied[-15:]],
            sample_owner_wait_ms=distribution([1000*(r['sample_io_started_s']-r['sample_submitted_s']) for r in rows if 'sample_io_started_s' in r]),
            sample_rpc_ms=distribution([1000*(r['sample_io_finished_s']-r['sample_io_started_s']) for r in rows if 'sample_io_finished_s' in r]))
    atomic_json(output/'recorded_causes.json',result)
    return result


def isolated(output):
    from .fair_static_dynamic import MODEL_BANK,case_for
    from .framework_cpu_affinity import CpuPlacement
    from .framework_runtime import ControllerPool
    from .framework_runtime_world import FrameworkWorld
    from .probe_collision_root import single
    from .convex_teacher import ConvexTeacher
    from .fly_convex_policy import load_airsim_client
    from .probe_airsim_three import translate_layout
    load_airsim_client();plan=read_json(SOURCE/'plan.json');results={}
    with CpuPlacement():
        pool=ControllerPool(model_names=MODEL_BANK);teacher=ConvexTeacher()
        if pool.provenance()!=plan['model_provenance']:raise RuntimeError('source models changed')
        for seed in (4,5):
            case=case_for(seed);expected=read_json(SOURCE/f'seed{seed}_after/initial_state.json')['Drone10']
            layout=translate_layout(case['layouts']['Drone10'],[0.,-(case['count']-1)*6.,0.])
            model=plan['configuration']['models']['Drone10']
            for kind,period in [('dnn',.030),('dnn',.020),('teacher',.020)]:
                label=f'seed{seed}_Drone10_{kind}{int(period*1000)}';directory=output/label
                if (directory/'report.json').exists():payload=read_json(directory/'report.json')
                else:
                    world=FrameworkWorld(layout,0.,1,case['net_width_m'],task_duration=case['duration_s'],layouts={'Drone1':layout})
                    try:
                        payload=single(world,pool,teacher,model,kind,period,directory,
                            dict(id=label,critical_index=-1),case['duration_s'],expected_initial=expected)
                        atomic_json(directory/'source.json',dict(source_trial=str(SOURCE/f'seed{seed}_after'),
                            source_vehicle='Drone10',local_initial_matched=True,model=model))
                    finally:atomic_json(output/(label+'_cleanup.json'),world.close())
                results[label]={k:payload[k] for k in ('result','input_to_command_ms','command_gap_ms')}
                atomic_json(output/'isolated_summary.json',dict(complete=False,cases=results))
                if payload['result']['physical_task_passed']:break
        atomic_json(output/'isolated_summary.json',dict(complete=True,cases=results))
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'controller_architecture_study/results/remaining_native_cause_20261029_v1')
    parser.add_argument('--isolated',action='store_true');args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    result=audit(args.output)
    for name,r in result.items():print(name,r['phases'],r['final_1s_worst_closed_command_gap'],flush=True)
    if args.isolated:isolated(args.output)
    print('Diagnosis:',args.output,flush=True)


if __name__=='__main__':main()
