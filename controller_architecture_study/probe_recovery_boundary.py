"""Paired capability-boundary sampling; bounded discovery, untouched confirmation.

Initial-state stress only. No mid-flight impulse, controller retraining or
result-dependent geometry changes. Every queried outcome is retained.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil

import numpy as np

from .collect_effect_holdout import make_case
from .controller_selector.collect import run_trial,source_hashes
from .controller_selector.data import PROTOCOL
from .decision_dataset import ROOT,atomic_json,sha256
from .fast_convex_preprocess import FastReferenceWindow
from .fly_convex_policy import load_airsim_client
from .fly_decision_dnn import StoredRoutePlanner
from .framework_cpu_affinity import CpuPlacement
from .framework_runtime import ControllerPool
from .framework_runtime_features import reference_point
from .framework_runtime_world import FrameworkWorld
from .state_mode_probe import normal_frame


SMALL='plain_mlp_h2_w128_seed0'
ACTIONS=((SMALL,.020),(SMALL,.080),('plain_mlp_h3_w256_seed0',.020),('plain_mlp_h9_w1024_seed0',.020))


def attempt_reports(directory):
    return ([directory/'report.json'] if (directory/'report.json').exists() else [])+sorted(directory.glob('attempt_*/report.json'))


def selected_reports(output):
    """Reuse valid outcomes, including failures; retry only missing labels."""
    selected=[];invalid=[]
    for directory in sorted((output/'trials').glob('*')):
        valid=[]
        for path in attempt_reports(directory):
            row=json.loads(path.read_text())
            (valid if row['label']['valid'] else invalid).append(path)
        if len(valid)>1:raise RuntimeError(f'multiple valid attempts for one frozen action: {directory}')
        selected.extend(valid)
    return selected,invalid


def next_attempt(directory):
    if not directory.exists():return directory
    for index in range(1,3):
        candidate=directory/f'attempt_{index:03}'
        if not candidate.exists():return candidate
    raise RuntimeError(f'two infrastructure retries exhausted; inspect retained attempts: {directory}')


def retain_identity(output,identity):
    path=output/'plan.json'
    if path.exists():
        before=json.loads(path.read_text());comparable=dict(identity)
        old_hash=before.pop('collector_sha256');new_hash=comparable.pop('collector_sha256')
        if before!=comparable:raise RuntimeError('boundary collection physical protocol or controller identity changed')
        revision=output/'collector_revisions'/f'{new_hash}.json'
        if old_hash!=new_hash and not revision.exists():
            atomic_json(revision,dict(original_sha256=old_hash,current_sha256=new_hash,
                reason='Resume infrastructure-invalid attempts without overwriting them; physical protocol unchanged'))
            shutil.copy2(__file__,revision.with_suffix('.py'))
    else:atomic_json(path,identity)


def next_hardness(observed):
    """Refine an observed transition; never assume global monotonic recovery."""
    levels=sorted(observed)
    intervals=list(zip(levels,levels[1:]))
    transition=[(a,b) for a,b in intervals if observed[a]!=observed[b]]
    a,b=max(transition or intervals,key=lambda x:(x[1]-x[0],-x[0]))
    return round((a+b)/2,4)


def boundary_case(base,hardness):
    case=deepcopy(base)
    planner=StoredRoutePlanner(case['layout']);window=FastReferenceWindow(planner.trajectory,planner.config)
    ref=reference_point(window,case['phase_s']);normal,_=normal_frame(ref)
    # Constant geometry/trajectory. Stress changes only the independently reset
    # initial lateral displacement and outward speed, in physical units.
    case['id']=f"boundary_{case['split']}_s{case['route_seed']}_a{hardness:.4f}"
    case['hardness']=hardness
    case['relative_initial'].update(position_error=(.25*hardness*normal).tolist(),
        velocity_error=(1.2*hardness*normal).tolist(),quaternion=ref.orientation.tolist())
    return case


def summarize(output):
    plan=json.loads((output/'plan.json').read_text());rows=[];groups={};invalid=[]
    paths,invalid=selected_reports(output)
    for path in paths:
        row=json.loads(path.read_text());rows.append(row)
        groups.setdefault(row['group'],[]).append(row)
    evidence=[];incomplete=[]
    expected={(m,t) for m,t in ACTIONS}
    for group,block in groups.items():
        if {(r['model'],r['cadence_s']) for r in block}!=expected:
            incomplete.append(group);continue
        if not all(np.allclose(block[0]['state_feature'],r['state_feature'],atol=2e-4,rtol=0) for r in block):
            raise RuntimeError(f'paired input mismatch: {group}')
        indexed={(r['model'],r['cadence_s']):r for r in block}
        fast,slow=indexed[(SMALL,.020)],indexed[(SMALL,.080)]
        recovery={m:indexed[(m,.020)]['label']['qualified'] for m in (SMALL,ACTIONS[2][0],ACTIONS[3][0])}
        collisions={f"{r['model']}/T{r['cadence_s']}":r['label']['collision'] for r in block}
        evidence.append(dict(group=group,split=block[0]['split'],route_seed=block[0]['route_seed'],
            hardness=block[0]['hardness'],fast_qualified=fast['label']['qualified'],slow_qualified=slow['label']['qualified'],
            period_recovery_differs=fast['label']['qualified']!=slow['label']['qualified'],
            fast_collision=fast['label']['collision'],slow_collision=slow['label']['collision'],
            model_recovery=recovery,model_recovery_differs=len(set(recovery.values()))>1,
            collision_outcomes=collisions))
    report=dict(queried_trials=len(rows)+len(invalid),valid_trials=len(rows),
        complete=len(rows)==120 and len(evidence)==30 and not incomplete,
        invalid=[str(p) for p in invalid],selected_reports=[str(p) for p in paths],incomplete_groups=incomplete,
        groups=evidence,discovery_adaptive=True,confirmation_uniform_predeclared=True,
        changed_controller_weights=False,midflight_impulse=False,test_used_for_selection=False,
        scope='stepped recovery at sampled initial states and discrete time points; not natural multi-loop deployment')
    report['counts_by_split']={s:dict(groups=sum(r['split']==s for r in evidence),
        period_disagreements=sum(r['split']==s and r['period_recovery_differs'] for r in evidence),
        model_disagreements=sum(r['split']==s and r['model_recovery_differs'] for r in evidence),
        slow_collides_fast_does_not=sum(r['split']==s and r['slow_collision'] and not r['fast_collision'] for r in evidence))
        for s in ('discovery','confirmation')}
    atomic_json(output/'report.json',report)
    print('Boundary summary:',report['queried_trials'],report['counts_by_split'],flush=True)
    return report


def run(output):
    from config import load_calibration
    source=ROOT/'controller_architecture_study/results/controller_selector_237d_v2/data/plan.json'
    original=json.loads(source.read_text());load_airsim_client()
    output.mkdir(parents=True,exist_ok=True);placement=CpuPlacement()
    try:
        with placement:
            pool=ControllerPool(model_names=original['models']);calibration=load_calibration()
            if pool.provenance()!=original['provenance']:raise RuntimeError('controller pool changed')
            cases=[]
            for split,seeds in (('discovery',range(3400,3404)),('confirmation',range(3500,3502))):
                for seed in seeds:
                    case=make_case(seed,pool,calibration);case['split']=split;cases.append(case)
            identity=dict(cases=cases,actions=[list(a) for a in ACTIONS],delay_s=.010,
                protocol=PROTOCOL,initial_scaling=dict(lateral_position_m=.25,outward_speed_mps=1.2),
                discovery_endpoints=[0.,1.6],refinement_queries=3,confirmation_hardness=[0.,.4,.8,1.2,1.6],
                max_trials=120,source_sha256=source_hashes(),collector_sha256=sha256(Path(__file__)),
                provenance=pool.provenance(),source_plan_sha256=sha256(source),outcome_dependent_geometry=False)
            retain_identity(output,identity)
            for base in cases:
                world=FrameworkWorld(base['layout'],base['phase_s'],1,PROTOCOL['net_width_m'],task_duration=2.)
                observed={};used=[]
                try:
                    for step in range(5):
                        if base['split']=='confirmation':hardness=identity['confirmation_hardness'][step]
                        elif step<2:hardness=identity['discovery_endpoints'][step]
                        else:hardness=next_hardness(observed)
                        case=boundary_case(base,hardness);block=[]
                        actions=list(ACTIONS);np.random.default_rng(base['route_seed']+step*7919).shuffle(actions)
                        for model,period in actions:
                            path=output/'trials'/f"{case['id']}_{model}_T{period*1000:.0f}"
                            existing=[json.loads(p.read_text()) for p in attempt_reports(path)]
                            result=next((r for r in existing if r['label']['valid']),None)
                            if result is None:
                                path=next_attempt(path)
                                result=run_trial(world,pool,model,.010,case,path,
                                    protocol={**PROTOCOL,'cadence_s':period,'update_bound_s':period+.010})
                                result.update(route_seed=base['route_seed'],hardness=hardness,
                                    experiment_kind='sampled_initial_recovery_boundary',midflight_impulse=False)
                                atomic_json(path/'report.json',result)
                            if not result['label']['valid']:
                                raise RuntimeError(f'invalid boundary trial retained, not a negative recovery label: {path}')
                            block.append(result)
                            print(f"{case['id']} {model} T={period*1000:.0f} recovery={result['label']['qualified']} collision={result['label']['collision']}",flush=True)
                        used.append(hardness)
                        observed[hardness]=tuple(next(r for r in block if (r['model'],r['cadence_s'])==a)['label']['qualified'] for a in ACTIONS[:2])
                        atomic_json(output/f"progress_{base['id']}.json",dict(hardness=used,anchor_outcomes=observed))
                finally:atomic_json(output/f"cleanup_{base['id']}.json",world.close())
            summarize(output)
    finally:atomic_json(output/'cpu_placement.json',placement.report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--summarize',action='store_true');args=p.parse_args()
    summarize(args.output) if args.summarize else run(args.output)
