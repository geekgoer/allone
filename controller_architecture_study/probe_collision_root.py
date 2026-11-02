"""Causal probes: freeze chosen requests, remove/replay waits, and restore pre-crash states."""
import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

import numpy as np

from control_adapter import send_control
from .decision_dataset import ROOT,atomic_json,sha256
from .dynamic_probe_common import read_json
from .fair_static_dynamic import MODEL_BANK,case_for,sources
from .framework_cpu_affinity import CpuPlacement
from .framework_runtime import ControllerPool
from .framework_runtime_editor import read_background_throttle
from .framework_runtime_features import encode_direct
from .framework_runtime_wait import DeadlineWaiter
from .framework_runtime_world import FrameworkWorld
from .heterogeneous_motivation import create_world
from .inflight_sensing import InflightSensing
from .probe_seven_continuous_timing import flight,matched_initials
from .convex_teacher import ConvexTeacher


SOURCE=ROOT/'controller_architecture_study/results/global_admission_20261029_v4'
OUTPUT=ROOT/'controller_architecture_study/results/collision_root_20261029_v1'
DURATION=6.3


def tape_for(directory,wait):
    report=read_json(directory/'report.json');origin=report['wall_origin_s']
    requests=read_json(directory/'requests.json');tapes={}
    for v in report['vehicles']:
        rows=sorted((r for r in requests if r['vehicle']==v and r['planned_release_s']-origin<DURATION),
                    key=lambda r:r['planned_release_s'])
        result=[]
        for r in rows:
            row={k:r[k] for k in ('vehicle','sample_id','model','priority','requested_period_s','slot')}
            row.update(planned_release_s=r['planned_release_s']-origin,
                minimum_release_s=r['planned_release_s']-origin,deadline_s=r['deadline_s']-origin,
                nominal_deadline_s=r['nominal_deadline_s']-origin,skipped_releases=0,serial_vehicle=True)
            if wait:
                row.update(dispatch_not_before_s=r.get('dispatch_s',r['planned_release_s'])-origin,
                           delivery_not_before_s=r.get('io_started_s',r['planned_release_s'])-origin)
            result.append(row)
        gap=max(.020,float(np.median(np.diff([r['planned_release_s'] for r in result[-11:]]))))
        # Source retired collided loops. Extend only AFTER their recorded arrivals;
        # primary comparison is the first failing gate, not later fleet performance.
        while result[-1]['planned_release_s']+gap<DURATION:
            old=result[-1];r=old['planned_release_s']+gap
            row={**old,'sample_id':v+':'+str(int(old['sample_id'].split(':')[1])+1),
                'planned_release_s':r,'minimum_release_s':r,'deadline_s':old['deadline_s']+gap,
                'nominal_deadline_s':old['nominal_deadline_s']+gap,'slot':old['slot']+1,'extension_after_source_retirement':True}
            for key in ('dispatch_not_before_s','delivery_not_before_s'):
                if key in row:row[key]=old[key]+gap
            result.append(row)
        tapes[v]=result
    return tapes


def restore_scenario(source,phase):
    trace=read_json(source/'control_traces.json')['Drone2']
    row=min(trace,key=lambda r:abs(r['t']-phase))
    requests=read_json(source/'requests.json')
    previous=max((r for r in requests if r['vehicle']=='Drone2' and r.get('status')=='applied'
                  and r['rpc_return_s']<=row['sample_s']),key=lambda r:r['rpc_return_s'])
    override=dict(position_error=(np.array(row['position'])-row['reference_position']).tolist(),
        velocity_error=(np.array(row['velocity'])-row['reference_velocity']).tolist(),
        quaternion=row['quaternion'],acceleration=row['acceleration'],angular_velocity=row['angular_velocity'],
        held_control=previous['mapped_control'])
    return row['t'],dict(id='restored_before_contact',critical_index=-1,relative_initial=override),dict(
        source_t=row['t'],source_sample_s=row['sample_s'],held_command_id=previous['sample_id'],source_state=row)


def single(world,pool,teacher,model,kind,period,directory,scenario,duration,expected_initial=None):
    directory.mkdir();states=world.initialize(scenario,directory);raw=states['Drone1'];w=world.windows['Drone1']
    if expected_initial is not None and not all(np.allclose(world.initial['Drone1'][k],expected_initial[k],rtol=0.,atol=1e-6)
            for k in ('position','velocity','quaternion','held_control')):
        raise RuntimeError('isolated local initial state differs; physics remains paused')
    pool.predict(model,encode_direct(w,raw,world.phases['Drone1']))
    teacher.query(raw,w.window(world.phases['Drone1'],horizon=21))
    rows=[];origin=time.perf_counter();world.client.simPause(False)
    with DeadlineWaiter() as timer:
        # Cover the simulation-time endpoint despite a small wall/sim drift.
        # The existing obs['t'] condition stops the run; do not invent coverage.
        for k in range(int(np.ceil((duration+.150)/period))+1):
            release=origin+k*period;timer.wait([],release)
            raw,hit,captured=world.snapshot('Drone1');obs=world.observe('Drone1',raw,hit,captured)
            if obs['collision'] or obs['t']>=duration:break
            feature_start=time.perf_counter()
            if kind=='teacher':u,diagnostic=teacher.query(raw,w.window(world.phases['Drone1']+obs['t'],horizon=21))
            else:u=pool.predict(model,encode_direct(w,raw,world.phases['Drone1']+obs['t']))
            send_control(world.drones['Drone1'],u,world.calibration,duration=1.);world.client.ping()
            rows.append(dict(t=obs['t'],sample_s=captured,release_s=release,command_ack_s=time.perf_counter(),
                solve_ms=1000*(time.perf_counter()-feature_start),control=np.asarray(u).tolist()))
        raw,hit,captured=world.snapshot('Drone1');world.observe('Drone1',raw,hit,captured)
    result=world.physical_report(duration)['Drone1']
    a=np.asarray([r['command_ack_s']-r['sample_s'] for r in rows])*1000
    gaps=np.diff([r['command_ack_s'] for r in rows])*1000
    payload=dict(kind=kind,period_ms=period*1000,phase=world.phases['Drone1'],result=result,
        input_to_command_ms=dict(mean=float(a.mean()),max=float(a.max())),
        command_gap_ms=dict(mean=float(gaps.mean()),max=float(gaps.max())),rows=rows,
        local_single_loop_test=True,not_a_multi_loop_capacity_claim=True)
    atomic_json(directory/'report.json',payload);atomic_json(directory/'trace.json',world.traces)
    print(directory.name,'passed',result['physical_task_passed'],'failure',result['failure'],
        'age',payload['input_to_command_ms'],'gap',payload['command_gap_ms'],flush=True)
    return payload


def run(root):
    root.mkdir(parents=True,exist_ok=True)
    sys.setswitchinterval(.0005)
    reference=read_json(SOURCE/'plan.json');configuration=reference['configuration']
    tapes={f'seed{s}_{arm}':tape_for(SOURCE/f'seed{s}_protected',arm=='wait')
           for s in (4,5) for arm in ('fast','wait')}
    recovery_cases=[('start_dnn20',0.,'dnn',.020),('start_dnn40',0.,'dnn',.040),('start_teacher20',0.,'teacher',.020),
        ('state52_dnn20',5.2,'dnn',.020),('state52_teacher20',5.2,'teacher',.020),
        ('state56_dnn20',5.6,'dnn',.020),('state56_teacher20',5.6,'teacher',.020)]
    with CpuPlacement() as placement:
        if placement.layout!=reference['cpu_layout']:raise RuntimeError('CPU resources changed')
        if read_background_throttle()['background_throttle_enabled']:raise RuntimeError('disable editor background throttling')
        pool=ControllerPool(model_names=MODEL_BANK);teacher=ConvexTeacher()
        if pool.provenance()!=reference['provenance']:raise RuntimeError('controller sources changed')
        names=(*sources(),Path(__file__).name,'framework_runtime_wait.py','convex_teacher.py')
        names=tuple(dict.fromkeys(names))
        protocol=dict(source=str(SOURCE),source_plan_sha256=sha256(SOURCE/'plan.json'),duration_s=DURATION,
            configuration=configuration,tape_orders={4:['fast','wait'],5:['wait','fast']},single_cases=recovery_cases,
            tapes=tapes,source_sha256={n:sha256(Path(__file__).with_name(n)) for n in names},
            scope='timing mechanism diagnosis; waits reproduce recorded service instants; not physical wind/position disturbance or final controller',
            primary_gate=4,primary_vehicle='Drone2',new_model_training=False)
        if (root/'plan.json').exists():
            import json
            if read_json(root/'plan.json')!=json.loads(json.dumps(protocol)):raise RuntimeError('resume changed protocol')
        else:
            frozen=root/'frozen_source';frozen.mkdir()
            for n in names:shutil.copy2(Path(__file__).with_name(n),frozen/n)
            atomic_json(root/'plan.json',protocol)
        results={}
        for seed,order in ((4,('fast','wait')),(5,('wait','fast'))):
            case=case_for(seed);case['duration_s']=DURATION;world=create_world(case,True)
            try:
                for arm in order:
                    name=f'seed{seed}_{arm}';directory=root/name
                    if not (directory/'report.json').exists():
                        settings=deepcopy(configuration)
                        settings.update(release_tape=tapes[name],deadline_policy='observe',release_contract='committed_calendar_v1')
                        sensing=InflightSensing(world.vehicles,world.calibration,placement.layout['io_cpus'][1],None)
                        flight(world,pool,30.,dict(id=name,critical_index=-1),'static',directory,DURATION,placement.layout,
                            sensing=sensing,resident_worker=True,scheduler='edf',frontend_order='deadline',
                            loop_config=settings,model=MODEL_BANK[0],stop_on_collision=True,collision_policy='vehicle')
                    report=read_json(directory/'report.json');initial=read_json(directory/'initial_state.json')
                    if not matched_initials(initial,read_json(SOURCE/f'seed{seed}_protected/initial_state.json')):
                        raise RuntimeError('joint replay initial state differs')
                    results[name]=dict(collisions=report['collisions'],primary=report['vehicles']['Drone2'],
                        full_report=str(directory/'report.json'),error=report['error'])
                    atomic_json(root/'progress.json',results)
                    print(name,'collisions',report['collisions'],'primary_failure',report['vehicles']['Drone2']['failure'],flush=True)
            finally:atomic_json(root/f'seed{seed}_cleanup.json',world.close())
        source=SOURCE/'seed4_protected';model=configuration['models']['Drone2']
        for name,phase,kind,period in recovery_cases:
            directory=root/name
            if (directory/'report.json').exists():result=read_json(directory/'report.json')
            else:
                scenario=dict(id=name,critical_index=-1);metadata={}
                if phase:phase,scenario,metadata=restore_scenario(source,phase)
                layout=case_for(4)['layouts']['Drone2'];duration=DURATION-phase
                world=FrameworkWorld(layout,phase,1,1.7,task_duration=duration,layouts={'Drone1':layout})
                try:
                    result=single(world,pool,teacher,model,kind,period,directory,scenario,duration)
                    atomic_json(directory/'source_state.json',metadata)
                finally:atomic_json(root/(name+'_cleanup.json'),world.close())
            results[name]=result;atomic_json(root/'progress.json',results)
        atomic_json(root/'summary.json',dict(complete=True,results=results,protocol=protocol))
        print('Cause probes complete:',root/'summary.json',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args();run(args.output)


if __name__=='__main__':main()
