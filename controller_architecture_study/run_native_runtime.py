"""Matched native/Python I/O trials on the existing ten-loop scenario."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import shutil
import sys
import time

import numpy as np

from config import load_calibration
from .decision_dataset import ROOT,atomic_json,sha256
from .dynamic_probe_common import read_json
from .fair_static_dynamic import MODEL_BANK,case_for,sources
from .fly_convex_policy import load_airsim_client
from .framework_cpu_affinity import CpuPlacement
from .framework_native_io import LIBRARY,NativeCommandIO
from .framework_runtime import ControllerPool
from .framework_runtime_io import ProcessCommandIO
from .framework_runtime_editor import read_background_throttle
from .heterogeneous_deadlines import freeze_costs
from .heterogeneous_motivation import create_world,execute
from .probe_seven_continuous_timing import matched_initials
from .run_protected_joint import SOURCE,TIMING_SOURCE


def distribution(values):
    a=np.asarray(values,dtype=float)
    if not len(a):return None
    return dict(count=len(a),median=float(np.median(a)),mean=float(a.mean()),
        p99=float(np.percentile(a,99)),p999=float(np.percentile(a,99.9)),max=float(a.max()),
        over10ms=int((a>10.).sum()))


def timing(directory):
    report=read_json(directory/'report.json');rows=read_json(directory/'requests.json')
    fields=dict(io_dispatch_ms=('sample_submitted_s','sample_io_started_s'),
        frontend_input_wait_ms=('sample_response_received_s','frontend_state_start_s'),
        release_to_dispatch_ms=('job_released_s','dispatch_s'),
        received_to_slot_release_ms=('io_response_received_s','pipeline_released_s'),
        received_to_cleanup_ms=('io_response_received_s','completion_frontend_start_s'),
        state_and_collision_ms=('sample_io_started_s','sample_io_finished_s'),
        input_return_ms=('sample_io_finished_s','sample_response_received_s'),
        response_ms=('job_released_s','rpc_return_s'),
        release_parent_delay_ms=('job_released_s','release_parent_received_s'))
    out={name:distribution([1000*(r[b]-r[a]) for r in rows if a in r and b in r])
        for name,(a,b) in fields.items()}
    out['command_gap_ms']={v:distribution(1000*np.diff(sorted(r['rpc_return_s'] for r in rows
        if r['vehicle']==v and 'rpc_return_s' in r))) for v in report['vehicles']}
    out.update(collisions=report['collisions'],physical_task_passed=report['physical_task_passed'],
        full_flight_completed=report['full_flight_completed'],total_J=report['total_J'],
        requests=report['requests'],nominal_late=len(report['nominal_late_jobs']),
        nominal_undelivered=len(report['nominal_undelivered_deadlines']),error=report['error'],
        superseded=report.get('superseded_before_dispatch',0),admission_decisions=report['global_admission_decisions'])
    return out


def io_probe(root,placement,count):
    """Read only. Both backends use identical ten-vehicle fan-out and CPU mask."""
    calibration=load_calibration();vehicles=[f'Drone{i}' for i in range(1,11)]
    results={}
    for backend in ('python','cpp'):
        cls=ProcessCommandIO if backend=='python' else NativeCommandIO
        io=cls(len(vehicles) if backend=='python' else vehicles,calibration,.003,
            placement.layout['io_cpus'][:1],allow_late=True)
        rows=[];sequence=0;packets={}
        try:
            for _ in range(count):
                for v in vehicles:
                    sequence+=1
                    io.sample(dict(vehicle=v,sample_id=f'probe:{sequence}',model='read_only',
                        deadline_s=time.perf_counter()+3.,status='sampling'))
                end=time.perf_counter()+5.
                remaining=len(vehicles)
                while remaining:
                    ready=io.drain_samples()
                    for packet,row in ready:
                        rows.append(row);remaining-=1
                        packets[row['vehicle']]=dict(position=packet[0].position.tolist(),sim_ns=packet[0].timestamp_ns)
                    if time.perf_counter()>end:raise RuntimeError('read-only I/O probe timed out')
                    if remaining:time.sleep(.0005)
        finally:io.close()
        atomic_json(root/f'{backend}_io_requests.json',rows)
        fields=dict(dispatch_ms=('sample_submitted_s','sample_io_started_s'),
            rpc_and_decode_ms=('sample_io_started_s','sample_io_finished_s'),
            return_ms=('sample_io_finished_s','sample_response_received_s'))
        results[backend]={name:distribution([1000*(r[b]-r[a]) for r in rows]) for name,(a,b) in fields.items()}
        results[backend].update(last_state=packets,placement=io.placement)
        print(backend,results[backend],flush=True)
    atomic_json(root/'io_probe.json',results)


def run(root,seeds,backend,io_only=False,count=100,compare_repair=False,compare_joint=False):
    root.mkdir(parents=True,exist_ok=False)
    if not LIBRARY.is_file():raise RuntimeError('run build_native_runtime.ps1 first')
    load_airsim_client();sys.setswitchinterval(.0005)
    with CpuPlacement() as placement:
        if io_only:
            io_probe(root,placement,count);return
        if read_background_throttle()['background_throttle_enabled']:raise RuntimeError('disable editor background throttling')
        old=read_json(SOURCE/'plan.json');selection=read_json(SOURCE/'strengthened/selection.json')
        pool=ControllerPool(model_names=MODEL_BANK)
        if pool.provenance()!=old['provenance']:raise RuntimeError('controller artifacts changed')
        if placement.layout!=old['cpu_layout']:raise RuntimeError('CPU placement changed')
        calibration=(ROOT/'controller_architecture_study/results/runtime_backlog_repair_20261029_v5/seed4_after'
            if compare_joint else TIMING_SOURCE)
        costs=freeze_costs(calibration,root/'pipeline_costs.json',statistic='p95',native=compare_joint)['costs']
        profile=read_json(TIMING_SOURCE/'report.json')['worker_profile']
        assumptions=read_json(Path(__file__).with_name('configs')/'recovery_budget_prototype_v1.json')
        chosen=('reference','joint') if compare_joint else ('before','after') if compare_repair else ('python','cpp') if backend=='both' else (backend,)
        plan=dict(backends=['cpp'] if compare_repair or compare_joint else list(chosen),arms=list(chosen),
            runtime_repair_comparison=compare_repair,joint_adaptation_comparison=compare_joint,seeds=seeds,configuration=selection['configuration'],
            cpu_layout=placement.layout,model_provenance=pool.provenance(),native_dll_sha256=sha256(LIBRARY),
            release_contract='independent calendar for BOTH backends; feedback changes unreleased jobs only',
            native_scope='RPC I/O + absolute release thread; Python B/admission/segmented inference retained',
            timing_profile='same frozen native calibration for both arms' if compare_joint else 'same frozen pre-migration profile, empirical only',
            recovery_budget=assumptions,disturbance_injection=False,hard_realtime_certified=False)
        atomic_json(root/'plan.json',plan)
        frozen=root/'frozen_source';frozen.mkdir()
        names=(*sources(),'framework_native_io.py',Path(__file__).name)
        for name in names:shutil.copy2(Path(__file__).with_name(name),frozen/name)
        shutil.copytree(Path(__file__).with_name('native_runtime'),frozen/'native_runtime',
            ignore=shutil.ignore_patterns('build'))
        summaries={}
        for index,seed in enumerate(seeds):
            order=chosen if index%2==0 else tuple(reversed(chosen))
            case=case_for(seed);world=create_world(case,True)
            first=None
            try:
                for arm in order:
                    label=f'seed{seed}_{arm}'
                    execute(world,pool,placement,case,selection['configuration'],root,label,
                        dynamic=True,physical=True,anticipatory=True,recovery_models=old['recovery_models'],
                        guard_costs=old['compute_cost_s'],pipeline_costs=costs,
                        global_admission=True,frontend_protection=True,recovery_budget=assumptions,
                        planning_profile=profile,io_backend='cpp' if compare_repair or compare_joint else arm,
                        runtime_repair=True if compare_joint else (arm=='after') if compare_repair else None,
                        joint_adaptation=compare_joint and arm=='joint')
                    directory=root/label
                    initial=read_json(directory/'initial_state.json')
                    if first is not None and not matched_initials(first,initial):raise RuntimeError('paired initial states differ')
                    first=initial;summaries[label]=timing(directory)
                    atomic_json(root/'summary.json',dict(complete=False,plan=plan,trials=summaries))
                    item=summaries[label]
                    print(label,'collisions',item['collisions'],'response_ms',item['response_ms'],
                        'superseded',item['superseded'],'changes',item['admission_decisions'],flush=True)
            finally:atomic_json(root/f'seed{seed}_cleanup.json',world.close())
        atomic_json(root/'summary.json',dict(complete=True,plan=plan,trials=summaries,
            runtime_integration_passed=all(r['error'] is None for r in summaries.values()),
            observed_flight_and_nominal_timing_passed=all(r['physical_task_passed'] and not r['nominal_late']
                and not r['nominal_undelivered'] for r in summaries.values()),
            hard_realtime_certified=False))
        print('Native comparison:',root/'summary.json',flush=True)
        print('Trial completion does not imply flight/timing acceptance; inspect collisions and undelivered jobs.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--seeds',default='4,5')
    parser.add_argument('--backend',choices=('both','python','cpp'),default='both')
    parser.add_argument('--io-only',action='store_true',help='read-only ten-vehicle RPC comparison; no flight/reset')
    parser.add_argument('--count',type=int,default=100,help='ten-vehicle fan-out rounds for --io-only')
    parser.add_argument('--compare-repair',action='store_true',help='same C++ backend: old FIFO/completion vs repaired runtime')
    parser.add_argument('--compare-joint',action='store_true',help='same C++ backend and native cost profile: per-loop vs joint adaptation')
    args=parser.parse_args()
    seeds=[int(s) for s in args.seeds.split(',')]
    if not seeds or args.count<1:parser.error('nonempty seeds and positive count required')
    if (args.compare_repair or args.compare_joint) and args.io_only:parser.error('runtime comparison requires flight')
    if args.compare_repair and args.compare_joint:parser.error('choose one comparison')
    root=args.output or ROOT/'controller_architecture_study/results/native_runtime'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    run(root,seeds,args.backend,args.io_only,args.count,args.compare_repair,args.compare_joint)


if __name__=='__main__':main()
