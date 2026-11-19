"""Fresh, predeclared scene holdout AFTER freezing both rankers; no fitting."""
import argparse
import json
from pathlib import Path

import numpy as np

from airsim_interface import RawDroneState
from state_adapter import quaternion_multiply
from .controller_selector.collect import run_trial,physical_control,source_hashes
from .controller_selector.data import PROTOCOL
from .controller_effect import EffectRanker,evaluation,outcome_utility
from .decision_dataset import ROOT,atomic_json,sha256
from .fast_convex_preprocess import FastReferenceWindow
from .fly_convex_policy import load_airsim_client
from .fly_decision_dnn import StoredRoutePlanner
from .framework_campaign import layout_for
from .framework_cpu_affinity import CpuPlacement
from .framework_runtime import ControllerPool
from .framework_runtime_features import reference_point,encode_direct
from .framework_runtime_world import FrameworkWorld


def make_case(seed,pool,calibration):
    rng=np.random.default_rng(seed);family=('smooth','continuous','late')[seed%3]
    layout=layout_for(0,3,seed,family)
    gate=min(8,layout['hard_onset_gate']) if family=='late' else int(rng.integers(2,8))
    segment=layout['segments'][gate];phase=segment['start_time']+segment['duration']-.7
    planner=StoredRoutePlanner(layout);window=FastReferenceWindow(planner.trajectory,planner.config)
    ref=reference_point(window,phase)
    raw=RawDroneState(ref.position,ref.orientation,ref.velocity,ref.acceleration,np.zeros(3))
    held=physical_control(pool.predict('plain_mlp_h9_w1024_seed0',encode_direct(window,raw,phase)),calibration)
    rotation=rng.uniform(-.07,.07,3);angle=np.linalg.norm(rotation)
    dq=np.r_[np.cos(angle/2),rotation*np.sin(angle/2)/angle]
    return dict(id=f'fresh_s{seed}',split='holdout',route_seed=seed,family=family,phase_s=phase,layout=layout,
        relative_initial=dict(position_error=[float(rng.uniform(-.25,.25)),float(rng.uniform(-.25,.25)),float(rng.uniform(-.12,.12))],
            velocity_error=[float(rng.uniform(-.6,.6)),float(rng.uniform(-1.,1.)),float(rng.uniform(-.4,.4))],
            quaternion=quaternion_multiply(ref.orientation,dq).tolist(),held_control=held.tolist()))


def run(old,new,output):
    from config import load_calibration
    source=ROOT/'controller_architecture_study/results/controller_selector_237d_v2/data/plan.json'
    original=json.loads(source.read_text());load_airsim_client()
    output.mkdir(parents=True,exist_ok=True)
    models=['plain_mlp_h2_w128_seed0','plain_mlp_h3_w256_seed0','plain_mlp_h9_w1024_seed0']
    placement=CpuPlacement();rows=[];evaluation_rows=[]
    try:
        with placement:
            pool=ControllerPool(model_names=original['models'])
            if pool.provenance()!=original['provenance']:raise RuntimeError('controller provenance changed')
            cases=[make_case(s,pool,load_calibration()) for s in range(3200,3208)]
            plan=dict(cases=cases,models=models,cadences_s=[.020,.080],delays_s=[.020,.050],
                old_ranker_sha256=sha256(old),new_ranker_sha256=sha256(new),test_evaluated=False,
                no_fitting=True,source_sha256=source_hashes(),collector_sha256=sha256(Path(__file__)))
            if (output/'plan.json').exists():
                if json.loads((output/'plan.json').read_text())!=plan:raise RuntimeError('holdout source changed')
            else:atomic_json(output/'plan.json',plan)
            for case in cases:
                world=FrameworkWorld(case['layout'],case['phase_s'],1,PROTOCOL['net_width_m'],task_duration=2.)
                try:
                    tasks=[(m,t,d) for m in models for t in plan['cadences_s'] for d in plan['delays_s']]
                    np.random.default_rng(case['route_seed']+7919).shuffle(tasks)
                    for m,t,d in tasks:
                        directory=output/'trials'/f"{case['id']}_{m}_T{t*1000:.0f}_d{d*1000:.0f}"
                        if (directory/'report.json').exists():r=json.loads((directory/'report.json').read_text())
                        else:r=run_trial(world,pool,m,d,case,directory,
                            protocol={**PROTOCOL,'cadence_s':t,'update_bound_s':t+.010})
                        if not r['label']['valid']:raise RuntimeError(f"invalid holdout trial retained: {directory}")
                        trace=json.loads((directory/'trace.json').read_text())
                        evaluation_rows.append({**r,'utility':outcome_utility(r,trace),
                            'initial_observation':{k:trace[0][k] for k in ('risk','box_margin_m')}})
                        rows.append(r);atomic_json(output/'rows.json',rows)
                        print(f"holdout {len(rows)}/96 {case['id']} {m} T={t*1000:.0f} qualified={r['label']['qualified']}",flush=True)
                finally:atomic_json(output/f"cleanup_{case['id']}.json",world.close())
        old_model,new_model=EffectRanker(old),EffectRanker(new)
        results=dict(old=evaluation(evaluation_rows,old_model,original['model_cost_s']),
            guarded=evaluation(evaluation_rows,new_model,original['model_cost_s']))
        new_model.override_margin=None
        results['new_raw_diagnostic']=evaluation(evaluation_rows,new_model,original['model_cost_s'])
        report=dict(complete=True,results=results,no_hyperparameter_retuning=True,
            test_evaluated=False,model_sources=plan,scope='eight fresh paired scene groups; finite candidates, not safety certification')
        atomic_json(output/'report.json',report)
        print({k:{a:b for a,b in v.items() if a!='selections'} for k,v in results.items()},flush=True)
    finally:atomic_json(output/'cpu_placement.json',placement.report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('old','new','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();run(a.old,a.new,a.output)
