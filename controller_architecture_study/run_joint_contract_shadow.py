"""Paired real-flight capture of event-merged joint decisions, without takeover."""

import argparse
from pathlib import Path
import shutil
import sys

import psutil

from .decision_dataset import ROOT,atomic_json,sha256
from .dynamic_probe_common import read_json
from .fair_static_dynamic import MODEL_BANK,case_for
from .fly_convex_policy import load_airsim_client
from .framework_cpu_affinity import CpuPlacement
from .framework_runtime import ControllerPool
from .framework_runtime_editor import read_background_throttle
from .heterogeneous_motivation import create_world,execute
from .joint_contract_shadow import JointDecisionShadow
from .probe_seven_continuous_timing import matched_initials
from .run_native_runtime import timing


BASE=ROOT/'controller_architecture_study/results'


def worker_core(placement):
    topology=placement.report['topology'];layout=placement.layout
    available=set(placement.report['parent']['previous'])-set(layout['reserved_physical_siblings'])
    rows=[r for r in topology if r['logical'] in available and r['core']!=0
          and r['efficiency_class']==layout['efficiency_class']]
    if not rows:raise RuntimeError('no separate physical core for the shadow worker')
    chosen=min(rows,key=lambda r:r['core'])
    siblings={r['logical'] for r in topology if r['core']==chosen['core']}
    for editor in placement.editors:
        p=psutil.Process(editor['pid'])
        if p.create_time()!=editor['created_s']:raise RuntimeError('editor PID changed')
        mask=sorted(set(p.cpu_affinity())-siblings)
        if not mask:raise RuntimeError('no remaining editor CPUs')
        p.cpu_affinity(mask);editor['shadow_actual']=p.cpu_affinity()
    # CpuPlacement retains each editor's original full mask and restores it.
    placement.report['shadow_worker_core']=dict(core=chosen['core'],logical=chosen['logical'],
        excluded_editor_siblings=sorted(siblings),exclusive_os_reservation=False)
    return chosen['logical']


def run(output,seed,shadow_only=False,recover_after_miss=False,effect_path=None,fresh_review=False):
    output.mkdir(parents=True,exist_ok=False);load_airsim_client();sys.setswitchinterval(.0005)
    source=BASE/'fair_pool_search_20261028_v1'
    selection=read_json(source/'strengthened/selection.json');old=read_json(source/'plan.json')
    priors=source/'strengthened/search_progress.json'
    timing_source=BASE/'joint_runtime_repair_20261029_v3'
    costs=read_json(timing_source/'pipeline_costs.json')['costs']
    profile=read_json(timing_source/'seed4_joint/report.json')['worker_profile']
    recovery=read_json(Path(__file__).with_name('configs')/'recovery_budget_prototype_v1.json')
    frozen=output/'source';frozen.mkdir()
    files=[Path(__file__).with_name(n) for n in ('joint_contract_shadow.py','joint_contract_optimizer.py',
        'joint_contract_probe.py','joint_contract_recovery.py','probe_seven_continuous_timing.py',
        'framework_releases.py','held_control_safety.py','heterogeneous_motivation.py',
        'framework_native_io.py','controller_effect.py','joint_commit_guard.py',Path(__file__).name)]
    for p in files:shutil.copy2(p,frozen/p.name)
    rows={};first=None
    shadow_profile_path=BASE/'joint_contract_shadow_20261030_v3/baseline/report.json'
    shadow_profile=read_json(shadow_profile_path)['worker_profile']
    placement=CpuPlacement()
    try:
        with placement:
            cpu=worker_core(placement)
            if read_background_throttle()['background_throttle_enabled']:raise RuntimeError('disable editor background throttling')
            pool=ControllerPool(model_names=MODEL_BANK)
            if pool.provenance()!=old['provenance']:raise RuntimeError('controller source changed')
            plan=dict(seed=seed,arms=['shadow'] if shadow_only else ['baseline','shadow'],configuration=selection['configuration'],
                source_sha256={str(p):sha256(p) for p in files},priors_sha256=sha256(priors),
                costs=costs,profile=profile,provenance=pool.provenance(),worker_cpu=cpu,
                shadow_profile=shadow_profile,shadow_profile_source=str(shadow_profile_path),
                shadow_profile_sha256=sha256(shadow_profile_path),
                recommendations_applied=False,disturbance_injected=False,test_read=False,
                recovery_enabled=recover_after_miss,
                fresh_review=fresh_review,
                effect_ranker=str(effect_path) if effect_path else None,
                effect_ranker_sha256=sha256(effect_path) if effect_path else None,
                comparison_scope='functional paired pilot; one pair is NOT a statistical overhead/benefit claim')
            atomic_json(output/'plan.json',plan)
            case=case_for(seed);world=create_world(case,True)
            geometry={v:dict(net_width=world.net_width,gates=[dict(index=g.index,center=g.center.tolist(),
                orientation=g.orientation.tolist()) for g in world.gates[v]]) for v in world.vehicles}
            if effect_path:
                from .controller_effect import reference_spec
                for v in geometry:geometry[v]['reference']=reference_spec(world.windows[v],world.phases[v])
            try:
                for arm in plan['arms']:
                    observer=(JointDecisionShadow(shadow_profile,costs,priors,cpu,output/'shadow_worker',
                        geometry=geometry,recover_after_miss=recover_after_miss,effect_path=effect_path,
                        fresh_review=fresh_review) if arm=='shadow' else None)
                    try:
                        execute(world,pool,placement,case,selection['configuration'],output,arm,
                            dynamic=True,physical=True,anticipatory=True,recovery_models=old['recovery_models'],
                            guard_costs=old['compute_cost_s'],pipeline_costs=costs,global_admission=True,
                            frontend_protection=True,recovery_budget=recovery,planning_profile=profile,
                            io_backend='cpp',runtime_repair=True,joint_adaptation=True,joint_shadow=observer)
                    finally:
                        if observer is not None:observer.close()
                    directory=output/arm;initial=read_json(directory/'initial_state.json')
                    if first is not None and not matched_initials(first,initial):raise RuntimeError('paired initial states differ')
                    first=initial;rows[arm]=timing(directory)
                    if observer is not None:
                        shadow=read_json(output/'shadow_worker/report.json')
                        flight=read_json(directory/'report.json');duration=flight['duration_s']
                        shadow.update(frontend_synchronous_occupied_fraction=shadow['frontend_total_ms']/1000/duration,
                            worker_core_fraction=(shadow['worker_cpu_total_ms']/1000/duration
                                if shadow['worker_cpu_total_ms'] is not None else None),
                            global_decisions_per_s=shadow['processed']/duration,
                            per_job_decision_hypothesis=flight['analyses']/duration)
                        atomic_json(output/'shadow_worker/report.json',shadow)
                    atomic_json(output/'progress.json',rows)
                    print(arm,rows[arm],flush=True)
            finally:atomic_json(output/'cleanup.json',world.close())
        result=dict(complete=True,same_initials=len(rows)>1,trials=rows,recommendations_applied=False,
            shadow=read_json(output/'shadow_worker/report.json'),hard_realtime_certified=False,
            scope='same live control policy, configuration and resources; only asynchronous observer differs')
        atomic_json(output/'summary.json',result)
        print('Summary: '+str(output/'summary.json'),flush=True)
    finally:
        atomic_json(output/'cpu_placement.json',placement.report)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=4)
    parser.add_argument('--shadow-only',action='store_true')
    parser.add_argument('--recover-after-miss',action='store_true')
    parser.add_argument('--effect-ranker',type=Path)
    parser.add_argument('--fresh-review',action='store_true')
    args=parser.parse_args()
    if args.fresh_review and not args.effect_ranker:parser.error('--fresh-review requires --effect-ranker')
    run(args.output,args.seed,args.shadow_only,args.recover_after_miss,args.effect_ranker,args.fresh_review)


if __name__=='__main__':main()
