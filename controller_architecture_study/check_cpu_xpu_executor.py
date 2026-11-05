"""Offline real-process CPU/XPU consistency check; never connects to AirSim."""
import argparse
from pathlib import Path
import time

import numpy as np
import psutil

from .decision_dataset import atomic_json
from .framework_cpu_affinity import choose_layout,cpu_topology
from .framework_runtime import ControllerPool
from .framework_runtime_worker import RuntimeServer


def run(output,models):
    output.mkdir(parents=True,exist_ok=False)
    layout=choose_layout(cpu_topology(),psutil.Process().cpu_affinity())
    pool=ControllerPool(model_names=models)
    executor=RuntimeServer('edf',layout['inference_cpus'],scheduler_cpus=layout['frontend_cpus'],model_names=models)
    expected={};rows={};results=[]
    try:
        if executor.provenance!=pool.provenance():raise RuntimeError('executor/controller provenance mismatch')
        # Paired mean +/- one training std. No protected evaluation data read.
        for j,model in enumerate(models):
            for sign in (-1.,0.,1.):
                ident=f'{model}:{sign}';feature=np.asarray(pool.norm['x_mean']+sign*pool.norm['x_std'],dtype=np.float32)
                expected[ident]=pool.predict(model,feature)
                now=time.perf_counter();row=dict(sample_id=ident,vehicle=ident,model=model,priority=j,
                    deadline_s=now+10.,planned_release_s=now)
                rows[ident]=dict(row);executor.submit(feature,row)
        until=time.perf_counter()+20.
        while len(results)<len(expected):
            results.extend(executor.drain())
            if time.perf_counter()>until:raise TimeoutError('CPU/XPU offline dispatch did not complete')
            if len(results)<len(expected):time.sleep(.001)
        errors=[]
        for action,row in results:
            if action is None:raise RuntimeError('offline consistency job expired')
            original=rows[row['sample_id']]
            if any(row[k]!=original[k] for k in ('model','deadline_s')):raise RuntimeError('job context changed')
            errors.append(float(np.max(np.abs(action-expected[row['sample_id']]))))
        if max(errors)>1e-6:raise RuntimeError(f'CPU/XPU output mismatch: {max(errors)}')
    finally:executor.close()
    if len(executor.cpu_dispatches)!=len(executor.segments):raise RuntimeError('CPU dispatch/XPU segment count mismatch')
    if set(executor.placement['actual'])&set(executor.placement['host_scheduler']['actual']):
        raise RuntimeError('CPU scheduler and simulated XPU were not placed separately')
    report=dict(status='OFFLINE_CPU_XPU_PASS',jobs=len(results),max_abs_error=max(errors),
        dispatches=len(executor.cpu_dispatches),placement=executor.placement,profile=executor.profile,
        provenance=executor.provenance,AirSim_connected=False,FPGA_verified=False,is_WCET=False)
    atomic_json(output/'report.json',report)
    atomic_json(output/'cpu_dispatches.json',executor.cpu_dispatches)
    print('Offline CPU/XPU:',report['status'],'jobs',report['jobs'],'error',report['max_abs_error'])
    print('Report:',output/'report.json')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--models',default='plain_mlp_h2_w128_seed0')
    a=p.parse_args();run(a.output,tuple(a.models.split(',')))
