"""CPU-owned segment selection and an explicitly commanded XPU analogue.

The coordinator runs on a host CPU core. Only the executor loads networks and
keeps activations; it never chooses a job. Timings are CPU/IPC measurements,
not FPGA/DPU timing equivalence or WCET.
"""
from copy import deepcopy
import multiprocessing as mp
from multiprocessing.connection import wait
import time

import numpy as np

from .framework_cpu_affinity import set_process_cpus
from .framework_runtime_wire import RuntimeConnection


def serve_segments(connection,pool,allow_late=False):
    jobs={};segments=[]
    while True:
        message=connection.recv()
        if message[0]=='stop':return segments
        if message[0]=='submit':
            _,feature,row=message;ident=row['sample_id']
            if ident in jobs:raise RuntimeError('duplicate XPU context')
            jobs[ident]=(pool.start(row['model'],feature),dict(row))
            continue
        if message[0]!='segment_run':raise ValueError('XPU accepts only context or segment commands')
        ident=message[1];job,row=jobs[ident];start=time.perf_counter()
        if start>=row['deadline_s'] and not allow_late:
            del jobs[ident];row.update(status='expired_before_segment',ready_s=start)
            connection.send(('segment_result',None,None,row,True));continue
        row.setdefault('first_service_s',start)
        before=job.cursor;job.step();end=time.perf_counter()
        event=dict(vehicle=row['vehicle'],sample_id=ident,start_s=start,end_s=end,
            layer_start=before,layer_end=job.cursor)
        segments.append(event);row['compute_ms']=row.get('compute_ms',0.)+1000*(end-start)
        if job.done:
            del jobs[ident];row.update(status='ready',ready_s=end)
        connection.send(('segment_result',event,job.output if job.done else None,row,job.done))


def calibrate(connection,profile,repeats=3):
    """Charge measured host/XPU round trips to every predicted segment."""
    result=deepcopy(profile)
    for model,entry in result.items():
        observations=[[] for _ in entry['segment_estimates_s']]
        for repeat in range(repeats):
            now=time.perf_counter();ident=f'calibration:{model}:{repeat}'
            row=dict(sample_id=ident,vehicle='calibration',model=model,deadline_s=now+30.,priority=0,
                enqueue_s=now,worker_enqueue_s=now)
            connection.send(('submit',np.zeros(237,dtype=np.float32),row))
            cursor=0
            while True:
                began=time.perf_counter();connection.send(('segment_run',ident))
                if not connection.poll(10.):raise TimeoutError('XPU segment calibration timed out')
                message=connection.recv()
                if message[0]!='segment_result':raise RuntimeError(f'XPU calibration failed: {message[0]}')
                _,event,_,returned,done=message
                if event is None or returned.get('status')=='expired_before_segment':
                    raise RuntimeError('XPU calibration command expired')
                observations[cursor].append(time.perf_counter()-began);cursor+=1
                if done:break
            if cursor!=len(observations):raise RuntimeError('segment profile/context mismatch')
        entry.update(segment_estimates_s=[max(old,1.25*max(values)+.0001)
            for old,values in zip(entry['segment_estimates_s'],observations)],
            cpu_xpu_roundtrip_samples_s=observations,host_dispatch_included=True,is_WCET=False)
    return result


def coordinate(parent,executor,policy):
    """Select on CPU; one segment command outstanding, no mid-segment switch."""
    queued={};active=None;dispatches=[]
    while True:
        while parent.poll():
            message=parent.recv()
            if message[0]=='stop':return dispatches
            if message[0]!='submit':raise ValueError('unknown CPU scheduler message')
            _,feature,row=message;ident=row['sample_id']
            if ident in queued:raise RuntimeError('duplicate CPU scheduled job')
            row=dict(row);row['worker_enqueue_s']=time.perf_counter();queued[ident]=row
            executor.send(('submit',feature,row))
            parent.send(('accepted',ident,row['model'],row['deadline_s'],row['worker_enqueue_s']))
        while executor.poll():
            message=executor.recv()
            if message[0]!='segment_result':raise RuntimeError(f'XPU failed: {message[0]}')
            _,event,action,row,done=message
            if row['sample_id']!=active:raise RuntimeError('XPU completed a segment not commanded by CPU')
            dispatched=dispatches[-1];dispatched['result_received_s']=time.perf_counter()
            dispatched['executor_end_s']=event['end_s'] if event else row['ready_s']
            active=None
            if done:
                del queued[row['sample_id']];parent.send(('done',action,row))
        if active is None and queued:
            began=time.perf_counter()
            row=min(queued.values(),key=lambda r:((r['priority'] if policy=='static' else r['deadline_s']),r['enqueue_s']))
            active=row['sample_id']
            dispatches.append(dict(sample_id=active,model=row['model'],deadline_s=row['deadline_s'],
                dispatch_s=time.perf_counter(),selection_ms=1000*(time.perf_counter()-began)))
            executor.send(('segment_run',active))
        wait([parent.connection,executor.connection])


def coordinator_worker(connection,policy,cpus,scheduler_cpus,allow_late,resident_worker,model_names):
    from .framework_runtime_worker import worker
    parent=RuntimeConnection(connection);executor=None;process=None
    try:
        placement=set_process_cpus(scheduler_cpus)
        context=mp.get_context('spawn');link,child=context.Pipe();executor=RuntimeConnection(link)
        process=context.Process(target=worker,args=(child,policy,cpus,allow_late,resident_worker,model_names,True),daemon=True)
        process.start();child.close()
        if not executor.poll(30.):raise TimeoutError('XPU initialization timed out')
        message=executor.recv()
        if message[0]!='ready':raise RuntimeError(f'XPU initialization failed: {message[0]}')
        profile,provenance,xpu_placement=message[1:]
        profile=calibrate(executor,profile)
        placement=dict(xpu_placement,host_scheduler=placement,dispatch_owner='CPU',execution_owner='XPU analogue',
            host_dispatch_calibrated=True,FPGA_timing_equivalent=False)
        parent.send(('ready',profile,provenance,placement))
        dispatches=coordinate(parent,executor,policy)
        executor.send(('stop',))
        if not executor.poll(10.):raise TimeoutError('XPU shutdown timed out')
        closed=executor.recv()
        if closed[0]!='closed':raise RuntimeError(f'XPU shutdown failed: {closed[0]}')
        mission_segments=[r for r in closed[1] if not r['sample_id'].startswith('calibration:')]
        parent.send(('closed',mission_segments,parent.stalls+executor.stalls+closed[2],dispatches))
    except BaseException as exc:
        parent.send(('error',f'{type(exc).__name__}: {exc}'))
    finally:
        if process is not None:
            process.join(2.)
            if process.is_alive():process.terminate();process.join(2.)
        if executor is not None:executor.close()
        parent.close()
