"""Validate global next-job admission and frontend protection on the frozen ten-loop case."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import shutil
import sys

from .decision_dataset import ROOT,atomic_json,sha256
from .dynamic_probe_common import read_json
from .diagnose_dynamic_regression import summarize,read_json_value
from .fair_static_dynamic import MODEL_BANK,case_for,sources
from .framework_cpu_affinity import CpuPlacement
from .framework_runtime import ControllerPool
from .framework_runtime_editor import read_background_throttle
from .heterogeneous_deadlines import freeze_costs
from .heterogeneous_motivation import create_world,execute
from .probe_seven_continuous_timing import matched_initials


SOURCE=ROOT/'controller_architecture_study/results/fair_pool_search_20261028_v1'
TIMING_SOURCE=ROOT/'controller_architecture_study/results/dynamic_regression_20261029_v1/seed4_fixed'
OUTPUT=ROOT/'controller_architecture_study/results/global_admission_checks'
ARMS={
    'original':dict(global_admission=False,frontend_protection=False),
    'frontend_only':dict(global_admission=False,frontend_protection=True),
    'protected':dict(global_admission=True,frontend_protection=True),
}
ORDERS={4:['original','frontend_only','protected'],5:['protected','frontend_only','original']}


def run(root,protected_only=False,recovery_budget=None):
    root.mkdir(parents=True,exist_ok=True)
    cost_path=root/'pipeline_costs.json'
    if not cost_path.exists():freeze_costs(TIMING_SOURCE,cost_path,statistic='p95')
    cost_record=read_json(cost_path)
    if cost_record['source_sha256']!=sha256(TIMING_SOURCE/'requests.json') or cost_record['statistic']!='p95':
        raise RuntimeError('frozen timing profile changed')
    old=read_json(SOURCE/'plan.json');selection=read_json(SOURCE/'strengthened/selection.json')
    orders={seed:[a for a in order if not protected_only or a=='protected'] for seed,order in ORDERS.items()}
    assumptions=read_json(recovery_budget) if recovery_budget else None
    planning_profile=None
    if assumptions is not None:
        from .recovery_budget import RecoveryAssumptions
        RecoveryAssumptions(**assumptions)
        orders={4:['protected','recoverability'],5:['recoverability','protected']}
        planning_profile=read_json(TIMING_SOURCE/'report.json')['worker_profile']
    sys.setswitchinterval(.0005)
    with CpuPlacement() as placement:
        if placement.layout!=old['cpu_layout']:raise RuntimeError('CPU placement differs')
        if read_background_throttle()['background_throttle_enabled']:raise RuntimeError('disable Unreal background throttling')
        pool=ControllerPool(model_names=MODEL_BANK)
        if pool.provenance()!=old['provenance']:raise RuntimeError('model artifacts differ')
        names=(*sources(),Path(__file__).name,'diagnose_dynamic_regression.py')
        plan=dict(selection_sha256=sha256(SOURCE/'strengthened/selection.json'),configuration=selection['configuration'],
            arms=({a:ARMS['protected'] for a in ('protected','recoverability')} if assumptions is not None else
                {a:s for a,s in ARMS.items() if not protected_only or a=='protected'}),orders=orders,cpu_layout=placement.layout,provenance=pool.provenance(),
            pipeline_cost_sha256=sha256(cost_path),timing_statistic='p95; pending tails retain measured maxima',
            soft_prediction_only=True,maximum_change_candidates=2,deadline_policy='observe',
            protected_initialization='retain frozen static c/T before the first feedback decision',
            source_sha256={name:sha256(Path(__file__).with_name(name)) for name in names})
        if assumptions is not None:
            plan.update(recovery_budget=assumptions,recovery_budget_sha256=sha256(recovery_budget),
                planning_profile=planning_profile,planning_profile_source=str(TIMING_SOURCE/'report.json'),
                planning_profile_source_sha256=sha256(TIMING_SOURCE/'report.json'),
                comparison='same dynamic policy, models and resources; change only geometric vs recovery budget')
        if (root/'plan.json').exists():
            if read_json(root/'plan.json')!=read_json_value(plan):raise RuntimeError('resume protocol changed')
        else:
            frozen=root/'frozen_source';frozen.mkdir()
            for name in names:shutil.copy2(Path(__file__).with_name(name),frozen/name)
            atomic_json(root/'plan.json',plan)
        rows={}
        for seed,order in orders.items():
            case=case_for(seed);world=create_world(case,True)
            try:
                for arm in order:
                    label=f'seed{seed}_{arm}';directory=root/label
                    if (directory/'fast_score.json').exists():result=read_json(directory/'fast_score.json')['result']
                    else:
                        result=execute(world,pool,placement,case,selection['configuration'],root,label,
                            dynamic=True,physical=True,anticipatory=True,recovery_models=old['recovery_models'],
                            guard_costs=old['compute_cost_s'],pipeline_costs=cost_record['costs'],
                            recovery_budget=assumptions if arm=='recoverability' else None,
                            planning_profile=planning_profile,**plan['arms'][arm])
                    report=read_json(directory/'report.json')
                    baseline=root/f'seed{seed}_{order[0]}'/'initial_state.json'
                    if not matched_initials(read_json(baseline),read_json(directory/'initial_state.json')):
                        raise RuntimeError('initial states differ')
                    rows[label]=dict(result=result,global_admission_enabled=report['global_admission_enabled'],
                        frontend_protection_enabled=report['frontend_protection_enabled'],
                        initial_taskset_forecast_passed=report['initial_taskset_forecast_passed'],
                        decisions=report['global_admission_decisions'],admission_ms=report['admission_ms'],
                        all_reconfigurations_admitted=report['all_reconfigurations_admitted'])
                    atomic_json(root/'progress.json',rows)
                    print(label,'decisions',rows[label]['decisions'],'admission_ms',rows[label]['admission_ms'],flush=True)
            finally:atomic_json(root/f'seed{seed}_cleanup.json',world.close())
        common={}
        for seed,order in orders.items():
            failures=[r['observed_t_s'] for arm in order
                for r in read_json(root/f'seed{seed}_{arm}'/'report.json')['retired_vehicles'].values()]
            window=(6.,min([9.5]+[t-.05 for t in failures]))
            if window[1]>window[0]:
                for arm in order:common[f'seed{seed}_{arm}']=summarize(root/f'seed{seed}_{arm}',window)
        atomic_json(root/'summary.json',dict(complete=True,plan=plan,arms=rows,common_precontact=common,
            same_initials=True,hard_safety_certified=False,FPGA_verified=False))
        print('Protected study complete:',root/'summary.json',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--protected-only',action='store_true')
    parser.add_argument('--recovery-budget',type=Path,help='compare original B against explicit recovery assumptions')
    args=parser.parse_args()
    output=args.output or OUTPUT/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    run(output,args.protected_only,args.recovery_budget)


if __name__=='__main__':main()
