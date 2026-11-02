"""One-loop low-latency feasibility, with an actual per-vehicle collision stop."""

import argparse
from pathlib import Path
import time

import numpy as np

from control_adapter import send_control
from .convex_teacher import ConvexTeacher
from .decision_dataset import atomic_json
from .dynamic_probe_common import read_json
from .framework_cpu_affinity import CpuPlacement
from .framework_runtime import ControllerPool,MODELS
from .dynamic_probe_common import PLAIN
from .framework_runtime_features import encode_direct
from .framework_runtime_wait import DeadlineWaiter
from .framework_runtime_world import FrameworkWorld


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',type=Path,required=True)
    parser.add_argument('--vehicle',default='Drone2')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--models',nargs='+',choices=sorted(p.name for p in PLAIN.iterdir() if p.name.startswith('plain_mlp_') and (p/'best_model.pt').exists()))
    parser.add_argument('--period-ms',type=float,default=20.)
    args=parser.parse_args()
    document=read_json(args.case)
    case=document.get('validation_case',document.get('case',document));layout=case['layouts'][args.vehicle]
    root=args.output;root.mkdir(parents=True,exist_ok=False)
    if not np.isfinite(args.period_ms) or args.period_ms<10.:raise ValueError('period must be at least10ms')
    trials=([('teacher10',.010,None)]+[(m,args.period_ms/1000,m) for m in args.models] if args.models else
        [('teacher10',.010,None),('dnn20',.020,'plain_mlp_h9_w1024_seed0'),('dnn10',.010,'plain_mlp_h9_w1024_seed0')])
    placement=CpuPlacement()
    with placement:
        pool=ControllerPool(model_names=args.models);teacher=ConvexTeacher()
        atomic_json(root/'plan.json',dict(case=case,vehicle=args.vehicle,teacher=teacher.signature(),provenance=pool.provenance(),
            trials=trials,injected_latency=False,controller_clock='saved global polynomial time; no receding restart'))
        world=FrameworkWorld(layout,0.,1,case['net_width_m'],task_duration=case['duration_s'])
        rows=[]
        try:
            for name,period,model in trials:
                directory=root/name;directory.mkdir()
                states=world.initialize(dict(critical_index=-1),directory)
                initial_hit=type('Hit',(),dict(time_stamp=world.initial['Drone1']['collision_stamp'],object_name='',has_collided=False))()
                teacher.reset();commands=[];contact=None;error=None
                origin=time.perf_counter()
                world.observe('Drone1',states['Drone1'],initial_hit,origin)
                world.client.simPause(False)
                release=origin
                try:
                    with DeadlineWaiter() as timer:
                        while time.perf_counter()-origin<case['duration_s']+1.:
                            timer.wait([],release)
                            raw,hit,sampled=world.snapshot('Drone1')
                            obs=world.observe('Drone1',raw,hit,sampled)
                            if obs['collision']:
                                contact=dict(t=obs['t'],object=hit.object_name)
                                world.park_vehicle('Drone1',raw);break
                            if obs['t']>=case['duration_s']:break
                            if model is None:action,_=teacher.query(raw,world.windows['Drone1'].window(obs['t']))
                            else:action=pool.predict(model,encode_direct(world.windows['Drone1'],raw,obs['t']))
                            send_control(world.drones['Drone1'],action,world.calibration,duration=1.)
                            world.client.ping();returned=time.perf_counter()
                            commands.append(dict(t=obs['t'],response_ms=1000*(returned-sampled),
                                planned_release_s=release,sample_s=sampled,command_return_s=returned))
                            release=origin+(int((returned-origin)/period)+1)*period
                except Exception as exc:error=f'{type(exc).__name__}: {exc}'
                finally:world.hover_all()
                physical=world.physical_report(case['duration_s'])['Drone1']
                result=dict(name=name,model=model,period_ms=1000*period,contact=contact,physical=physical,error=error,
                    mean_R_ms=float(np.mean([r['response_ms'] for r in commands])) if commands else None,
                    p99_R_ms=float(np.percentile([r['response_ms'] for r in commands],99)) if commands else None,
                    max_R_ms=max((r['response_ms'] for r in commands),default=None),
                    p99_update_gap_ms=float(np.percentile(np.diff([r['command_return_s'] for r in commands])*1000,99)) if len(commands)>1 else None,
                    max_update_gap_ms=float(np.max(np.diff([r['command_return_s'] for r in commands]))*1000) if len(commands)>1 else None,
                    success=physical['recovery_passed'] and physical['clock_valid'] and not error)
                atomic_json(directory/'traces.json',world.traces);atomic_json(directory/'commands.json',commands)
                rows.append(result);atomic_json(root/'summary.json',rows)
                print('CAPABILITY',result,flush=True)
        finally:atomic_json(root/'cleanup_report.json',world.close())


if __name__=='__main__':main()
