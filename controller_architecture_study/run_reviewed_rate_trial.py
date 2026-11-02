"""Paired real application of reviewed rates, with fixed models and EDF.

Static selected configuration, legacy dynamic control, and legacy + guarded
joint-rate assistance. This is NOT an exact FALCON/SVM reproduction claim.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys

from .decision_dataset import ROOT,atomic_json,sha256
from .dynamic_probe_common import read_json
from .fair_static_dynamic import MODEL_BANK,case_for
from .fly_convex_policy import load_airsim_client
from .framework_cpu_affinity import CpuPlacement
from .framework_runtime import ControllerPool
from .framework_runtime_editor import read_background_throttle
from .heterogeneous_motivation import create_world,execute
from .joint_contract_shadow import JointDecisionShadow
from .controller_effect import reference_spec
from .probe_seven_continuous_timing import matched_initials
from .run_joint_contract_shadow import worker_core
from .run_native_runtime import timing


BASE=ROOT/'controller_architecture_study/results'


def run(output,boundary_report,seed=6,ranker_path=None):
    boundary=read_json(boundary_report)
    if not boundary['complete']:raise ValueError('finish and audit the boundary collection before application')
    output.mkdir(parents=True,exist_ok=False);load_airsim_client();sys.setswitchinterval(.0005)
    origin=BASE/'fair_pool_search_20261028_v1'
    config=read_json(origin/'strengthened/selection.json')['configuration'];source=read_json(origin/'plan.json')
    prior=origin/'strengthened/search_progress.json'
    profile_path=BASE/'joint_contract_shadow_20261030_v3/baseline/report.json'
    profile=read_json(profile_path)['worker_profile']
    costs=read_json(BASE/'joint_runtime_repair_20261029_v3/pipeline_costs.json')['costs']
    compute={m:sum(r['segment_estimates_s']) for m,r in profile.items()}
    ranker=ranker_path or BASE/'controller_effect_repair_20261031_v2/ranker/ranker.npz'
    recovery=read_json(Path(__file__).with_name('configs')/'recovery_budget_prototype_v1.json')
    order=['static_selected','legacy_dynamic','reviewed_rates','reviewed_rates','legacy_dynamic','static_selected']
    files=[Path(__file__).with_name(n) for n in ('run_reviewed_rate_trial.py','joint_contract_shadow.py',
        'joint_commit_guard.py','joint_contract_optimizer.py','joint_contract_recovery.py','held_control_safety.py',
        'framework_native_io.py','framework_releases.py','framework_runtime_worker.py','controller_effect.py',
        'probe_seven_continuous_timing.py','heterogeneous_motivation.py')]
    frozen=output/'source';frozen.mkdir()
    for p in files:shutil.copy2(p,frozen/p.name)
    plan=dict(order=order,seed=seed,configuration=deepcopy(config),models_fixed_across_arms=True,
        scheduler='two-FC-segment EDF in all arms; deadlines can differ',
        same_costs_in_all_arms=True,profile=profile,costs=costs,compute_costs=compute,
        profile_sha256=sha256(profile_path),ranker_sha256=sha256(ranker),boundary_sha256=sha256(boundary_report),
        boundary_counts=boundary['counts_by_split'],source_sha256={str(p):sha256(p) for p in files},
        controller_provenance=source['provenance'],external_impulse=False,initial_tracking_offset=False,
        virtual_prefix_disabled=True,hard_realtime_certified=False,
        scope='functional repeated paired pilot; reviewed rates supplement the existing dynamic policy; not exact FALCON')
    atomic_json(output/'plan.json',plan)
    placement=CpuPlacement();rows=[];first=None
    try:
        with placement:
            cpu=worker_core(placement)
            if read_background_throttle()['background_throttle_enabled']:raise RuntimeError('disable editor background throttle')
            pool=ControllerPool(model_names=MODEL_BANK)
            if pool.provenance()!=source['provenance']:raise RuntimeError('existing DNN source changed')
            case=case_for(seed);case['virtual_extra_s']=0.
            world=create_world(case,True)
            geometry={v:dict(net_width=world.net_width,gates=[dict(index=g.index,center=g.center.tolist(),
                orientation=g.orientation.tolist()) for g in world.gates[v]],
                reference=reference_spec(world.windows[v],world.phases[v])) for v in world.vehicles}
            try:
                for index,arm in enumerate(order):
                    name=f'{index+1:02d}_{arm}';observer=None
                    if arm=='reviewed_rates':
                        observer=JointDecisionShadow(profile,costs,prior,cpu,output/(name+'_planner'),geometry=geometry,
                            recover_after_miss=True,effect_path=ranker,fresh_review=True,apply_rates=True)
                    try:
                        result=execute(world,pool,placement,case,config,output,name,dynamic=arm!='static_selected',
                            physical=True,anticipatory=arm!='static_selected',recovery_models=dict(config['models']),
                            guard_costs=compute,pipeline_costs=costs,global_admission=arm!='static_selected',
                            frontend_protection=True,recovery_budget=recovery,planning_profile=profile,io_backend='cpp',
                            runtime_repair=True,joint_adaptation=arm!='static_selected',scheduler='edf',joint_shadow=observer)
                    finally:
                        if observer is not None:observer.close()
                    directory=output/name;initial=read_json(directory/'initial_state.json')
                    if first is not None and not matched_initials(first,initial):raise RuntimeError('paired initial states differ')
                    first=initial;requests=read_json(directory/'requests.json');flight=read_json(directory/'report.json')
                    tagged=[r for r in requests if 'joint_commit_generation' in r]
                    delivered=[r for r in tagged if r.get('status')=='applied' and 'rpc_return_s' in r]
                    row=dict(arm=arm,name=name,physical_result=result,timing=timing(directory),
                        tagged_requests=len(tagged),tagged_controls_delivered=len(delivered),
                        changed_rate_controls_delivered=sum(r.get('joint_rate_changed',False) for r in delivered),
                        changed_period_controls_delivered=sum(r.get('joint_period_changed',False) for r in delivered),
                        changed_release_controls_delivered=sum(r.get('joint_release_changed',False) for r in delivered),
                        changed_deadline_controls_delivered=sum(r.get('joint_deadline_changed',False) for r in delivered),
                        models_match_frozen=all(r['model']==config['models'][r['vehicle']] for r in requests),
                        conditional_budget_late_count=len(flight['late_jobs']),
                        control_window_covered=flight['full_flight_completed'],sensor_window_covered=flight['sensor_window_covered'],
                        planner=read_json(output/(name+'_planner')/'report.json') if observer else None)
                    if not row['models_match_frozen']:raise RuntimeError('a supposedly rate-only arm changed models')
                    rows.append(row);atomic_json(output/'progress.json',rows)
                    print('APPLICATION',name,'collisions',flight['collisions'],'tagged_delivered',len(delivered),
                        'J_measured',result['short_measured_J'],flush=True)
            finally:atomic_json(output/'cleanup.json',world.close())
        summary=dict(complete=len(rows)==len(order),same_initials=True,rows=rows,
            total_installed_groups=sum((r['planner'] or {}).get('installed_groups',0) for r in rows),
            tagged_controls_delivered=sum(r['tagged_controls_delivered'] for r in rows),
            changed_rate_controls_delivered=sum(r['changed_rate_controls_delivered'] for r in rows),
            changed_period_controls_delivered=sum(r['changed_period_controls_delivered'] for r in rows),
            changed_release_controls_delivered=sum(r['changed_release_controls_delivered'] for r in rows),
            recommendations_actually_executed=any(r['tagged_controls_delivered'] for r in rows),
            models_fixed=True,scope=plan['scope'],hard_realtime_certified=False)
        atomic_json(output/'summary.json',summary)
        print('Application summary:',output/'summary.json','installed',summary['total_installed_groups'],
            'delivered',summary['tagged_controls_delivered'],flush=True)
    finally:atomic_json(output/'cpu_placement.json',placement.report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--boundary-report',type=Path,required=True)
    p.add_argument('--seed',type=int,default=6);p.add_argument('--ranker',type=Path)
    a=p.parse_args();run(a.output,a.boundary_report,a.seed,a.ranker)
