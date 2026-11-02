"""Small matched cadence extension; no control-DNN training or test-set use."""

import argparse
from pathlib import Path

import numpy as np

from .controller_selector.collect import run_trial,source_hashes
from .controller_selector.data import PROTOCOL
from .decision_dataset import ROOT,atomic_json,sha256
from .dynamic_probe_common import read_json
from .framework_cpu_affinity import CpuPlacement
from .framework_runtime import ControllerPool
from .framework_runtime_world import FrameworkWorld
from .fly_convex_policy import load_airsim_client


MODELS=('plain_mlp_h2_w128_seed0','plain_mlp_h3_w256_seed0')


def run(source,output):
    output.mkdir(parents=True,exist_ok=True)
    original=read_json(source/'plan.json')
    cases=[]
    for split,count in (('train',8),('validation',4)):
        available=[c for c in original['cases'] if c['split']==split]
        cases.extend(available[i] for i in np.linspace(0,len(available)-1,count,dtype=int))
    identity=dict(source=str(source),source_plan_sha256=sha256(source/'plan.json'),
        cases=cases,models=list(MODELS),cadences_s=[.020,.080],delays_s=[.010,.050],
        base_protocol=PROTOCOL,model_provenance=original['provenance'],
        source_sha256=source_hashes(),collector_sha256=sha256(Path(__file__)),
        test_read=False,validation_groups_never_fit=True,
        recurring_input_age_s=.010,scope='paired stepped-physics outcome points; no real-time or interval safety claim')
    if (output/'plan.json').exists():
        if read_json(output/'plan.json')!=identity:raise RuntimeError('cadence collector identity changed')
    else:atomic_json(output/'plan.json',identity)
    load_airsim_client()
    rows=[]
    with CpuPlacement() as placement:
        pool=ControllerPool(model_names=tuple(original['models']))
        if pool.provenance()!=original['provenance']:raise RuntimeError('frozen controller pool changed')
        for ci,case in enumerate(cases):
            world=FrameworkWorld(case['layout'],case['phase_s'],1,PROTOCOL['net_width_m'],
                task_duration=PROTOCOL['duration_s'])
            try:
                tasks=[(m,t,d) for m in MODELS for t in identity['cadences_s'] for d in identity['delays_s']]
                np.random.default_rng(case['route_seed']+7919).shuffle(tasks)
                for model,cadence,delay in tasks:
                    parent=output/'trials'/f"{case['id']}_T{cadence*1000:.0f}_d{delay*1000:.0f}_{model}"
                    attempts=sorted(parent.glob('attempt_*/report.json'))
                    valid=next((read_json(p) for p in attempts if read_json(p)['label']['valid']),None)
                    if valid is None:
                        protocol={**PROTOCOL,'cadence_s':cadence,'update_bound_s':cadence+.010}
                        for attempt in range(len(attempts)+1,3):
                            directory=parent/f'attempt_{attempt:03d}'
                            valid=run_trial(world,pool,model,delay,case,directory,protocol=protocol)
                            print(f"[{ci+1}/{len(cases)}] {case['id']} {model} T={cadence*1000:.0f} d={delay*1000:.0f} "
                                  f"valid={valid['label']['valid']} qualified={valid['label'].get('qualified')} error={valid['error']}",flush=True)
                            if valid['error_kind'] in ('infrastructure','interrupted'):raise RuntimeError(valid['error'])
                            if valid['label']['valid']:break
                    rows.append(valid)
                    atomic_json(output/'rows.json',rows)
                    atomic_json(output/'progress.json',dict(expected=96,completed=len(rows),
                        valid=sum(r['label']['valid'] for r in rows),complete=len(rows)==96))
            finally:atomic_json(output/f"cleanup_{case['id']}.json",world.close())
    atomic_json(output/'cpu_placement.json',placement.report)
    print('Cadence extension complete:',output,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'controller_architecture_study/results/controller_selector_237d_v2/data')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.source,args.output)


if __name__=='__main__':main()
