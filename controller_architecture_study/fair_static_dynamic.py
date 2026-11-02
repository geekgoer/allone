"""One frozen model/resource pool: twenty static candidates, then paired flights."""
import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import sys

import numpy as np

from .decision_dataset import ROOT,atomic_json,sha256
from .dynamic_probe_common import read_json
from .framework_cpu_affinity import CpuPlacement
from .framework_runtime import ControllerPool,MODELS
from .framework_runtime_editor import read_background_throttle
from .heterogeneous_motivation import (make_case,create_world,strong_starts,configuration_key,
    propose,execute)
from .probe_seven_continuous_timing import matched_initials


MODEL_BANK=(*MODELS,'plain_mlp_h2_w128_seed0','plain_mlp_h3_w256_seed0')
PROTOCOL=dict(count=10,speed=5.,difficulty=.75,width=1.7,max_inflight=10,
    calibration_seed=0,validation_seeds=[4,5],budget=20,rng_seed=123,
    release_contract='committed_calendar_v1',deadline_policy='observe')


def search_starts(vehicles,model_names,recovery_models):
    starts=strong_starts(vehicles)
    for config in starts:config['max_inflight']=PROTOCOL['max_inflight']
    # Every dynamic-accessible model is explicitly tried by the static search.
    for model in model_names:
        candidate=deepcopy(starts[0]);candidate['models'][vehicles[0]]=model
        starts.append(candidate)
    for model in ('plain_mlp_h2_w128_seed0','plain_mlp_h3_w256_seed0'):
        candidate=deepcopy(starts[1]);candidate['models']={v:model for v in vehicles}
        starts.append(candidate)
    # Include the previously successful controller combination as a STATIC seed.
    hybrid=deepcopy(starts[1]);hybrid['models']=dict(recovery_models)
    starts.append(hybrid)
    alternate=deepcopy(hybrid)
    alternate['models']={v:'plain_mlp_h3_w256_seed0' if m=='plain_mlp_h2_w128_seed0' else m
                         for v,m in hybrid['models'].items()}
    starts.append(alternate)
    unique={configuration_key(config):config for config in starts}
    return list(unique.values())


def candidate_score(row):
    result=row.get('result')
    value=result.get('J_hat') if result and result.get('complete') else None
    return float(value) if value is not None and np.isfinite(value) else None


def best_candidate(rows):
    eligible=[r for r in rows if candidate_score(r) is not None]
    if not eligible:raise RuntimeError('no complete, finite nominal performance estimate')
    return max(eligible,key=candidate_score)


def case_for(seed):
    case=make_case(seed,PROTOCOL['difficulty'],PROTOCOL['speed'],PROTOCOL['count'])
    case['net_width_m']=PROTOCOL['width']
    return case


def sources():
    return ('fair_static_dynamic.py','heterogeneous_motivation.py','heterogeneous_load_guard.py',
        'heterogeneous_prediction.py','heterogeneous_deadlines.py','recovery_budget.py','framework_releases.py',
        'probe_seven_continuous_timing.py','framework_runtime.py','framework_runtime_worker.py',
        'framework_runtime_io.py','framework_runtime_wire.py','framework_runtime_io_wire.py',
        'framework_runtime_wait.py','framework_runtime_features.py','framework_runtime_world.py',
        'framework_cpu_affinity.py','safety_comparison_core.py','safety_comparison_flight.py')


def run(root,resume):
    root.mkdir(parents=True,exist_ok=True)
    if (root/'plan.json').exists() and not resume:raise RuntimeError('use --resume; existing results are immutable')
    source=ROOT/'controller_architecture_study/results/post_release_collision_cause_20261028'
    recovery={v:r['recovery'] for v,r in read_json(source/'candidate_config.json')['configuration']['mode_policy'].items()}
    if any(m not in MODEL_BANK for m in recovery.values()):raise ValueError('unequal controller pools')
    sys.setswitchinterval(.0005)
    with CpuPlacement() as placement:
        pool=ControllerPool(model_names=MODEL_BANK)
        if read_background_throttle()['background_throttle_enabled']:raise RuntimeError('editor background throttling must be disabled')
        calibration=case_for(PROTOCOL['calibration_seed']);vehicles=list(calibration['layouts'])
        if (root/'plan.json').exists():
            plan=read_json(root/'plan.json')
            if plan['protocol']!=PROTOCOL or plan['provenance']!=pool.provenance() or plan['recovery_models']!=recovery:
                raise RuntimeError('resume protocol or controller artifacts differ')
            if plan['cpu_layout']!=placement.layout:raise RuntimeError('resume resource placement changed')
            if any(sha256(Path(__file__).with_name(n))!=h for n,h in plan['source_sha256'].items()):
                raise RuntimeError('source changed since the experiment was frozen')
        else:
            frozen=root/'frozen_source';frozen.mkdir()
            for name in sources():shutil.copy2(Path(__file__).with_name(name),frozen/name)
            plan=dict(protocol=PROTOCOL,calibration_case=calibration,provenance=pool.provenance(),
                cpu_layout=placement.layout,recovery_models=recovery,
                recovery_source=str(source/'candidate_config.json'),recovery_source_sha256=sha256(source/'candidate_config.json'),
                compute_cost_s={m:sum(p['segment_estimates_s']) for m,p in pool.profile.items()},
                source_sha256={n:sha256(frozen/n) for n in sources()},
                starts=search_starts(vehicles,MODEL_BANK,recovery),
                selection_rule='highest finite J_hat among complete nominal profiles; no collision/deadline outcome filter',
                calibration_gate_meshes=False,validation_gate_meshes=True,
                physical_validation_order=[['seed4_static','seed4_dynamic'],['seed5_dynamic','seed5_static']],
                validation_used_for_static_selection=False,exact_global_optimum=False,full_FALCON_SVM=False,
                compute_screen='frozen measured single-worker demand <=0.95; not full response-time acceptance',
                score_scope='same measured-window cost aggregation and representative-prefix estimate for both methods')
            atomic_json(root/'plan.json',plan)
        costs=plan['compute_cost_s'];rows=[];seen=set();rng=np.random.default_rng(PROTOCOL['rng_seed'])
        previous=read_json(root/'search_progress.json')['rows'] if (root/'search_progress.json').exists() else []
        world=create_world(calibration,False)
        try:
            for step in range(PROTOCOL['budget']):
                config=deepcopy(plan['starts'][step]) if step<len(plan['starts']) else propose(
                    best_candidate(rows)['configuration'],step,rng,seen,model_names=MODEL_BANK)
                key=configuration_key(config)
                if key in seen:raise RuntimeError('duplicate static candidate')
                seen.add(key)
                if step<len(previous):
                    row=previous[step]
                    if configuration_key(row['configuration'])!=key:raise RuntimeError('resume candidate stream changed')
                else:
                    load=sum(costs[m]*1000/config['periods_ms'][v] for v,m in config['models'].items())
                    capacity=dict(admitted=load<=.95,compute_load=load,is_WCET=False)
                    print(f'SEARCH {step+1}/20 load={load:.3f}',flush=True)
                    result=execute(world,pool,placement,calibration,config,root,f'candidate_{step:02d}',physical=False) if capacity['admitted'] else None
                    row=dict(index=step,configuration=config,capacity=capacity,result=result)
                rows.append(row)
                best=best_candidate(rows) if any(candidate_score(r) is not None for r in rows) else None
                atomic_json(root/'search_progress.json',dict(rows=rows,evaluations=len(rows),
                    best_index=best['index'] if best else None,best_J_hat=candidate_score(best) if best else None))
            best=best_candidate(rows)
            selection=dict(configuration=best['configuration'],J_hat=candidate_score(best),index=best['index'],
                evaluated_candidates=len(rows),scored_candidates=sum(candidate_score(r) is not None for r in rows),
                model_bank=list(MODEL_BANK),resource=dict(max_inflight=10,compute_workers=1),
                frozen_before_validation=True,collision_filter_used=False,exact_global_optimum=False,
                calibration_case=calibration,provenance=pool.provenance())
            if (root/'selection.json').exists():
                if read_json(root/'selection.json')!=selection:raise RuntimeError('frozen static selection changed')
            else:atomic_json(root/'selection.json',selection)
        finally:atomic_json(root/'calibration_cleanup.json',world.close())

        selection_hash=sha256(root/'selection.json');config=selection['configuration'];pairs={}
        # One common nominal validation pair supplies comparable complete J_hat.
        stages=[('nominal_seed4',case_for(4),False,('static','dynamic')),
                ('physical_seed4',case_for(4),True,('static','dynamic')),
                ('physical_seed5',case_for(5),True,('dynamic','static'))]
        for stage,case,physical,order in stages:
            world=create_world(case,physical);arms={}
            try:
                for arm in order:
                    label=stage+'_'+arm;directory=root/label
                    if (directory/'fast_score.json').exists():result=read_json(directory/'fast_score.json')['result']
                    else:result=execute(world,pool,placement,case,config,root,label,dynamic=arm=='dynamic',physical=physical,
                        anticipatory=True,recovery_models=recovery,guard_costs=costs)
                    arms[arm]=result
                matched=matched_initials(read_json(root/(stage+'_static/initial_state.json')),
                    read_json(root/(stage+'_dynamic/initial_state.json')))
                if not matched:raise RuntimeError('paired initial states differ')
                pairs[stage]=dict(case=case,physical_gates=physical,arms=arms,initial_states_match=matched)
                atomic_json(root/'validation_progress.json',pairs)
            finally:atomic_json(root/(stage+'_cleanup.json'),world.close())
        if sha256(root/'selection.json')!=selection_hash:raise RuntimeError('validation changed the static selection')
        atomic_json(root/'summary.json',dict(selection=selection,selection_sha256=selection_hash,
            same_model_bank=True,same_resource_limits=True,pairs=pairs,complete=True,
            strict_deadlines_required_for_this_stage=False,hard_realtime_certified=False,
            full_FALCON=False,FPGA_verified=False))
        atomic_json(root/'cpu_placement.json',placement.report)
        print('FAIR STUDY COMPLETE:',root/'summary.json',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'controller_architecture_study/results/fair_pool_search_20261028_v1')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();run(args.output,args.resume)


if __name__=='__main__':main()
