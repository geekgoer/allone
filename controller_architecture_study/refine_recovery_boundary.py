"""64 bounded paired trials: discovery refinement and fixed fresh-route confirmation."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from .collect_effect_holdout import make_case
from .controller_selector.collect import source_hashes
from .controller_selector.data import PROTOCOL
from .decision_dataset import ROOT,atomic_json,sha256
from .fly_convex_policy import load_airsim_client
from .framework_cpu_affinity import CpuPlacement
from .framework_runtime import ControllerPool
from .framework_runtime_world import FrameworkWorld
from .probe_recovery_boundary import ACTIONS,boundary_case,next_hardness,paired_block,selected_reports,summarize


DISCOVERY=(3400,3402)
CONFIRMATION=(3600,3601)
CONFIRMATION_LEVELS=(.425,.475,.525,.575)


def anchors(source,seed):
    paths,_=selected_reports(source)
    rows=[json.loads(p.read_text()) for p in paths]
    result={}
    for level in (.4,.6):
        block=[r for r in rows if r['route_seed']==seed and r['hardness']==level and r['split']=='discovery']
        index={(r['model'],r['cadence_s']):r for r in block}
        if set(index)!=set(ACTIONS):raise ValueError(f'missing frozen discovery anchors: {seed}/{level}')
        result[level]=tuple(index[a]['label']['qualified'] for a in ACTIONS)
    return result


def run(source,output):
    from config import load_calibration
    old=json.loads((source/'plan.json').read_text())
    if not json.loads((source/'report.json').read_text())['complete']:raise ValueError('complete original boundary data required')
    load_airsim_client();output.mkdir(parents=True,exist_ok=True);placement=CpuPlacement()
    try:
        with placement:
            pool=ControllerPool(model_names=old['provenance']['models']);calibration=load_calibration()
            if pool.provenance()!=old['provenance']:raise ValueError('controller provenance changed')
            cases=[deepcopy(next(c for c in old['cases'] if c['route_seed']==s and c['split']=='discovery')) for s in DISCOVERY]
            for seed in CONFIRMATION:
                case=make_case(seed,pool,calibration);case['split']='confirmation';cases.append(case)
            initial={str(seed):anchors(source,seed) for seed in DISCOVERY}
            plan=dict(cases=cases,actions=[list(a) for a in ACTIONS],delay_s=.010,protocol=PROTOCOL,
                discovery_anchors=initial,refinement_queries=4,confirmation_hardness=list(CONFIRMATION_LEVELS),max_trials=64,
                source_boundary_sha256=sha256(source/'report.json'),source_plan_sha256=sha256(source/'plan.json'),
                source_sha256=source_hashes(),collector_sha256=sha256(Path(__file__)),
                shared_collector_sha256=sha256(Path(__file__).with_name('probe_recovery_boundary.py')),
                provenance=pool.provenance(),calibration=vars(calibration),outcome_dependent_geometry=False,test_read=False,
                no_fitting=True,confirmation_not_used_for_adaptive_queries=True)
            if (output/'plan.json').exists():
                if json.loads((output/'plan.json').read_text())!=json.loads(json.dumps(plan)):
                    raise ValueError('refinement source/plan changed; keep the old directory and choose a new output')
            else:atomic_json(output/'plan.json',plan)
            for base in cases:
                world=FrameworkWorld(base['layout'],base['phase_s'],1,PROTOCOL['net_width_m'],task_duration=2.)
                observed=anchors(source,base['route_seed']) if base['split']=='discovery' else {}
                try:
                    if vars(world.calibration)!=vars(calibration):raise ValueError('actuator calibration changed during collection')
                    for step in range(4):
                        level=next_hardness(observed) if base['split']=='discovery' else CONFIRMATION_LEVELS[step]
                        case=boundary_case(base,level);block=paired_block(world,pool,case,output,step)
                        index={(r['model'],r['cadence_s']):r for r in block}
                        observed[level]=tuple(index[a]['label']['qualified'] for a in ACTIONS)
                        atomic_json(output/f"progress_{base['route_seed']}.json",dict(observed=observed))
                finally:atomic_json(output/f"cleanup_{base['route_seed']}.json",world.close())
            report=summarize(output)
            report.update(source_boundary=str(source),anchors_reused_not_recounted=True,no_fitting=True,test_read=False)
            atomic_json(output/'report.json',report)
    finally:atomic_json(output/'cpu_placement.json',placement.report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.source,a.output)
