"""Frozen-model diagnosis: request adaptation versus shared EDF scheduling."""
import argparse
from pathlib import Path
import shutil
import sys

import numpy as np

from .decision_dataset import ROOT,atomic_json,sha256
from .dynamic_probe_common import read_json
from .fair_static_dynamic import MODEL_BANK,case_for,sources
from .framework_cpu_affinity import CpuPlacement
from .framework_runtime import ControllerPool
from .framework_runtime_editor import read_background_throttle
from .heterogeneous_motivation import create_world,execute
from .probe_seven_continuous_timing import matched_initials


SOURCE=ROOT/'controller_architecture_study/results/fair_pool_search_20261028_v1'
OUTPUT=ROOT/'controller_architecture_study/results/dynamic_regression_20261029_v1'
ARMS={
    'fixed':dict(dynamic=False,scheduler='static',frontend_order='category'),
    'fixed_edf':dict(dynamic=False,scheduler='edf',frontend_order='deadline'),
    'adaptive_fixed_priority':dict(dynamic=True,scheduler='static',frontend_order='category'),
    'adaptive_edf':dict(dynamic=True,scheduler='edf',frontend_order='deadline'),
}
ORDERS={4:['fixed','adaptive_edf','adaptive_fixed_priority','fixed_edf'],
        5:['fixed_edf','adaptive_fixed_priority','adaptive_edf','fixed']}


def stats(values):
    a=np.asarray(values,dtype=float)
    return dict(count=len(a),mean=float(a.mean()),p99=float(np.percentile(a,99)),max=float(a.max())) if len(a) else None


def summarize(directory,window=(6.,9.5)):
    report=read_json(directory/'report.json');origin=report['wall_origin_s']
    requests=read_json(directory/'requests.json');analyses=read_json(directory/'analyses.json')
    by_id={r['sample_id']:r for r in analyses};loops={}
    start,stop=window
    for v in report['vehicles']:
        rows=sorted((r for r in requests if r['vehicle']==v and r.get('status')=='applied'
            and start<=r['rpc_return_s']-origin<=stop),key=lambda r:r['rpc_return_s'])
        fields={
            'dispatch_wait_ms':('job_released_s','dispatch_s'),
            'sample_to_output_ms':('sample_s','rpc_return_s'),
            'backend_wait_and_compute_ms':('enqueue_s','ready_s'),
            'ready_to_send_ms':('ready_s','io_submitted_s'),
            'response_ms':('job_released_s','rpc_return_s'),
            'next_plan_ms':('next_plan_started_s','next_plan_done_s'),
        }
        loops[v]={name:stats([1000*(r[end]-r[start]) for r in rows if start in r and end in r])
                  for name,(start,end) in fields.items()}
        approved=[r for r in rows if (by_id.get(r['sample_id'],{}).get('resource_admission') or {}).get('feasible')]
        loops[v].update(command_gap_ms=stats(np.diff([r['rpc_return_s'] for r in rows])*1000),
            target_periods_ms=sorted({round(r['requested_period_s']*1000,6) for r in rows}),
            applied_models=sorted({r['model'] for r in rows}),
            compute_guard_approved=len(approved),
            approved_but_late=sum(r['rpc_return_s']>r['deadline_s'] for r in approved))
    segments=read_json(directory/'segments.json');jobs={r['sample_id']:r for r in requests}
    preemptions=[]
    for left,right in zip(segments,segments[1:]):
        if left['sample_id']==right['sample_id'] or not start<=right['start_s']-origin<=stop:continue
        old,new=jobs[left['sample_id']],jobs[right['sample_id']]
        if old.get('ready_s',0.)<=right['start_s']:continue
        preemptions.append(dict(t=right['start_s']-origin,victim=old['vehicle'],selected=new['vehicle'],
            victim_remaining_deadline_ms=1000*(old['deadline_s']-right['start_s']),
            selected_remaining_deadline_ms=1000*(new['deadline_s']-right['start_s']),
            victim_B_s=by_id.get(old['sample_id'],{}).get('budget_s'),
            selected_B_s=by_id.get(new['sample_id'],{}).get('budget_s')))
    busy=sum(max(0.,min(s['end_s'],origin+stop)-max(s['start_s'],origin+start)) for s in segments)
    return dict(collisions=report['collisions'],retired=report['retired_vehicles'],success=report['success'],
        scheduler=report['scheduler'],frontend_order=report['frontend_order'],
        deadline_policy=report['deadline_policy'],admission_checks=report['admission_checks'],
        B_binding_count=report['B_binding_count'],analyses=len(analyses),
        window_s=list(window),loops=loops,preemptions=preemptions,
        window_includes_postfailure_load=any(r['observed_t_s']<=stop for r in report['retired_vehicles'].values()),
        measured_segment_window_coverage=busy/(stop-start),
        coverage_is_OS_CPU_utilization=False,hard_safety_certified=False)


def audit(root):
    result={}
    for seed in (4,5):
        for arm in ('static','dynamic'):
            result[f'seed{seed}_{arm}']=summarize(SOURCE/'strengthened'/f'physical_seed{seed}_{arm}')
    atomic_json(root/'previous_flight_audit.json',result)
    print('Previous flight audit:',root/'previous_flight_audit.json',flush=True)


def run(root):
    source_plan=read_json(SOURCE/'plan.json');selection=read_json(SOURCE/'strengthened/selection.json')
    sys.setswitchinterval(.0005)
    with CpuPlacement() as placement:
        if placement.layout!=source_plan['cpu_layout']:raise RuntimeError('CPU placement differs from frozen comparison')
        if read_background_throttle()['background_throttle_enabled']:raise RuntimeError('disable Unreal background throttling')
        pool=ControllerPool(model_names=MODEL_BANK)
        if pool.provenance()!=source_plan['provenance']:raise RuntimeError('model/normalization source differs')
        plan=dict(source_selection=str(SOURCE/'strengthened/selection.json'),
            source_selection_sha256=sha256(SOURCE/'strengthened/selection.json'),
            configuration=selection['configuration'],arms=ARMS,orders=ORDERS,
            cpu_layout=placement.layout,provenance=pool.provenance(),
            compute_cost_s=source_plan['compute_cost_s'],recovery_models=source_plan['recovery_models'],
            deadline_policy='observe',no_parameter_tuning_during_experiment=True,
            interpretation='2x2 mechanism ablation: dynamic planning includes its own sensing/analysis overhead; fixed EDF uses periodic deadlines, not state-derived deadlines',
            source_sha256={n:sha256(Path(__file__).with_name(n)) for n in (*sources(),Path(__file__).name)})
        if (root/'plan.json').exists():
            if read_json(root/'plan.json')!=read_json_value(plan):raise RuntimeError('resume protocol changed')
        else:
            frozen=root/'frozen_source';frozen.mkdir()
            for n in plan['source_sha256']:shutil.copy2(Path(__file__).with_name(n),frozen/n)
            atomic_json(root/'plan.json',plan)
        rows={}
        for seed,order in ORDERS.items():
            case=case_for(seed);world=create_world(case,True)
            try:
                for arm in order:
                    label=f'seed{seed}_{arm}';path=root/label
                    if (path/'fast_score.json').exists():score=read_json(path/'fast_score.json')['result']
                    else:
                        score=execute(world,pool,placement,case,selection['configuration'],root,label,
                            physical=True,anticipatory=True,recovery_models=source_plan['recovery_models'],
                            guard_costs=source_plan['compute_cost_s'],**ARMS[arm])
                    rows[label]=dict(score=score,diagnosis=summarize(path))
                    baseline=root/f'seed{seed}_{order[0]}'/'initial_state.json'
                    if not matched_initials(read_json(baseline),read_json(path/'initial_state.json')):
                        raise RuntimeError('ablation initial states differ')
                    atomic_json(root/'progress.json',rows)
            finally:atomic_json(root/f'seed{seed}_cleanup.json',world.close())
        atomic_json(root/'summary.json',dict(complete=True,plan=plan,arms=rows,
            same_initials=True,FPGA_verified=False,hard_safety_certified=False))
        print('Ablations complete:',root/'summary.json',flush=True)


def read_json_value(value):
    import json
    return json.loads(json.dumps(value))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--fly',action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    audit(args.output)
    if args.fly:run(args.output)


if __name__=='__main__':main()
