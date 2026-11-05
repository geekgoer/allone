from copy import deepcopy
import multiprocessing as mp
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock,patch

import numpy as np
import pytest

from controller_architecture_study.framework_cpu_xpu import coordinate,serve_segments,calibrate
from controller_architecture_study.framework_runtime_wire import RuntimeConnection,encode,decode
from controller_architecture_study.joint_contract_shadow import JointDecisionShadow,Trigger
from controller_architecture_study.joint_commit_guard import refresh_epoch_metadata
from controller_architecture_study.framework_runtime import simulate_pipeline
from controller_architecture_study.heterogeneous_load_guard import resource_loads


class ToyPool:
    def __init__(self,entered=None,gate=None):
        self.steps=[];self.entered=entered;self.gate=gate

    def start(self,model,feature):
        pool=self
        class Job:
            cursor=0;done=False;output=None
            def step(self):
                if pool.entered is not None and not pool.steps:
                    pool.entered.set()
                    if not pool.gate.wait(3.):raise TimeoutError('test segment gate timed out')
                pool.steps.append(model);self.cursor+=1
                self.done=self.cursor==(3 if model=='large' else 1)
                if self.done:self.output=np.full(4,float(feature[0]))
        return Job()


def job(ident,model,deadline,priority=0):
    return dict(sample_id=ident,vehicle=ident,model=model,deadline_s=deadline,priority=priority,enqueue_s=time.perf_counter())


@pytest.mark.parametrize('policy,expected',[
    ('edf',['large','small','large','large']),('static',['large','large','large','small'])])
def test_cpu_selects_at_boundaries_and_executor_never_runs_uncommanded(policy,expected):
    client,host=mp.Pipe();scheduler,xpu=mp.Pipe();entered=threading.Event();gate=threading.Event()
    pool=ToyPool(entered,gate);result={};errors=[]
    def execute():
        try:result['segments']=serve_segments(RuntimeConnection(xpu),pool)
        except BaseException as exc:errors.append(exc)
    def schedule():
        try:result['dispatches']=coordinate(RuntimeConnection(host),RuntimeConnection(scheduler),policy)
        except BaseException as exc:errors.append(exc)
    executor=threading.Thread(target=execute,daemon=True);owner=threading.Thread(target=schedule,daemon=True)
    executor.start();owner.start();wire=RuntimeConnection(client);now=time.perf_counter()
    low=job('a','large',now+20.,0);urgent=job('b','small',now+10.,1)
    frozen=deepcopy((low,urgent))
    try:
        wire.send(('submit',np.ones(237,dtype=np.float32)*7,low))
        assert entered.wait(3.) and not pool.steps
        wire.send(('submit',np.ones(237,dtype=np.float32)*3,urgent));gate.set()
        done={}
        while len(done)<2:
            assert wire.poll(5.),errors
            message=wire.recv()
            if message[0]=='done':done[message[2]['sample_id']]=message
        wire.send(('stop',));owner.join(3.);assert not owner.is_alive(),errors
        RuntimeConnection(scheduler).send(('stop',));executor.join(3.);assert not executor.is_alive(),errors
        assert not errors and pool.steps==expected
        assert [r['model'] for r in result['dispatches']]==expected
        assert (low,urgent)==frozen
        for ident,value,source in [('a',7,low),('b',3,urgent)]:
            np.testing.assert_array_equal(done[ident][1],np.full(4,value))
            assert done[ident][2]['deadline_s']==source['deadline_s']
            assert done[ident][2]['model']==source['model']
    finally:
        gate.set()
        for link in (client,host,scheduler,xpu):link.close()


def test_registering_contexts_does_not_execute_any_segment():
    pool=ToyPool();row=job('a','large',time.perf_counter()+10.)
    messages=iter([('submit',np.zeros(237),row),('stop',)])
    pipe=SimpleNamespace(recv=lambda:next(messages),send=Mock())
    assert serve_segments(pipe,pool)==[] and pool.steps==[]
    pipe.send.assert_not_called()


def test_expired_xpu_command_preserves_deadline_and_does_no_compute():
    pool=ToyPool();row=job('a','small',0.)
    messages=iter([('submit',np.zeros(237),row),('segment_run','a'),('stop',)])
    pipe=SimpleNamespace(recv=lambda:next(messages),send=Mock())
    assert serve_segments(pipe,pool)==[] and not pool.steps
    returned=pipe.send.call_args.args[0]
    assert returned[0]=='segment_result' and returned[-1] is True and returned[2] is None
    assert returned[3]['deadline_s']==0. and returned[3]['status']=='expired_before_segment'


def test_new_segment_wire_preserves_context_without_pickle():
    row=job('a','small',10.);row.update(ready_s=2.,compute_ms=.2,status='ready')
    event=dict(sample_id='a',start_s=1.,end_s=2.,layer_start=0,layer_end=2)
    message=('segment_result',event,np.arange(4,dtype=float),row,True)
    encoded=encode(message);assert encoded[:1]==b'G'
    actual=decode(encoded)
    assert actual[0:2]==message[0:2] and actual[3:]==message[3:]
    np.testing.assert_array_equal(actual[2],message[2])
    assert decode(encode(('segment_run','a')))==('segment_run','a')
    with pytest.raises(ValueError):decode(encoded[:-1])


def test_profile_includes_real_segment_roundtrip_not_just_tensor_compute():
    parent,child=mp.Pipe();pool=ToyPool();errors=[]
    def run():
        try:serve_segments(RuntimeConnection(child),pool)
        except BaseException as exc:errors.append(exc)
    thread=threading.Thread(target=run,daemon=True);thread.start();wire=RuntimeConnection(parent)
    original={'large':dict(segment_estimates_s=[1e-8]*3)}
    try:
        profile=calibrate(wire,original,repeats=2)
        assert profile['large']['host_dispatch_included'] and not profile['large']['is_WCET']
        for cost,observed in zip(profile['large']['segment_estimates_s'],profile['large']['cpu_xpu_roundtrip_samples_s']):
            assert len(observed)==2 and cost>=max(observed)
        assert original['large']['segment_estimates_s']==[1e-8]*3
    finally:
        wire.send(('stop',));thread.join(3.);parent.close();child.close()
    assert not thread.is_alive() and not errors


def service_stub():
    obj=JointDecisionShadow.__new__(JointDecisionShadow)
    obj.started=True;obj.apply_rates=False;obj.next_service_s=-float('inf');obj.service_period_s=.010;obj.service_budget_s=.003
    obj.service_rows=[];obj.reply=None;obj.trigger=Trigger();obj.capture=Mock();obj.review_latest=Mock()
    return obj


def test_management_service_is_not_dependent_on_foreground_queue_becoming_empty():
    obj=service_stub();clock=[0.]
    with patch('controller_architecture_study.joint_contract_shadow.time.perf_counter',side_effect=lambda:clock[0]):
        # Every frontend iteration has another input ready. The top-of-loop
        # service still runs; an idle-only integration would never call it.
        for k in range(101):
            clock[0]=k*.001;obj.trigger.notice('a',clock[0],-.001)
            obj.tick({},None,None,{}, {},1)
    assert len(obj.service_rows)>=9 and obj.capture.call_count==len(obj.service_rows)
    starts=[r['start_s'] for r in obj.service_rows]
    assert max(np.diff(starts))<=.011000001


def test_management_overrun_is_reported_and_review_gets_a_turn():
    obj=service_stub();obj.reply=(None,None,SimpleNamespace(value=2),None);obj.reply_seen=1
    clock=[1.];obj.review_latest.side_effect=lambda *a,**k:clock.__setitem__(0,1.006)
    with patch('controller_architecture_study.joint_contract_shadow.time.perf_counter',side_effect=lambda:clock[0]):
        obj.tick({},None,None,{}, {},1)
    assert obj.service_rows[0]['budget_exceeded'] and obj.service_rows[0]['operation']=='review'
    assert obj.review_latest.call_args.kwargs['budget_s']==.003
    obj.capture.assert_not_called()


def test_factual_refresh_commits_near_future_without_reconfiguration_or_erasing_pending_miss():
    row=dict(vehicle='a',sample_id='a:8',model='frozen',priority=0,planned_release_s=.4583,
        requested_period_s=.030,nominal_deadline_s=.4883,deadline_s=.4479,obligation_actuation_s=.3679)
    plans={'a':row};pending=[dict(row,stage='compute',sample_id='a:7')];original=deepcopy(pending)
    certs={'a':dict(available=True,sample_s=.439802,actuation_s=.4398,deadline_s=.5198)}
    io=SimpleNamespace(calendar_basis=Mock(return_value={'a':dict(version=4)}),
        guarded_batch=Mock(return_value=dict(passed=True,installed=True,reason='context_current')))
    with patch('time.perf_counter',return_value=.4545):
        records=refresh_epoch_metadata(plans,pending,certs,{'a':8},io,.4545,end_s=1.,expires_s=.4575)
    assert plans['a']['deadline_s']==.5198 and pending==original
    for key in ('model','planned_release_s','requested_period_s','nominal_deadline_s'):
        assert plans['a'][key]==row[key]
    assert records[0]['installed'] and not records[0]['configuration_changed']
    assert io.guarded_batch.call_args.kwargs=={'commit':True}


def test_metadata_refresh_does_not_change_frontend_when_native_publication_or_control_races():
    row=dict(vehicle='a',sample_id='a:8',model='frozen',planned_release_s=.4583,requested_period_s=.030,
        deadline_s=.4479,obligation_actuation_s=.3679)
    plans={'a':row};original=deepcopy(plans)
    certs={'a':dict(available=True,sample_s=.439802,actuation_s=.4398,deadline_s=.5198)}
    io=SimpleNamespace(calendar_basis=lambda p:{},guarded_batch=Mock(return_value=dict(installed=False,reason='calendar_changed')))
    with patch('time.perf_counter',return_value=.4545):
        records=refresh_epoch_metadata(plans,[],certs,{'a':8},io,.4545,end_s=1.,expires_s=.4575)
    assert plans==original and not records[0]['installed']


def test_periodic_cpu_management_is_charged_without_stopping_xpu():
    future=[dict(vehicle='a',sample_id='a:1',model='m',priority=0,planned_release_s=.010,deadline_s=.015)]
    profile={'m':dict(segment_estimates_s=[.001])};base=dict(input_s=.001,output_s=.001)
    original=deepcopy(future)
    feasible,finish,_=simulate_pipeline(0.,future,[],profile,base)
    charged={**base,'management_period_s':.010,'management_budget_s':.003}
    blocked,_,_=simulate_pipeline(0.,future,[],profile,charged)
    assert feasible and finish['a:1']<.015 and not blocked
    assert future==original
    loads=resource_loads({'a':.020},{'a':'m'},{'m':.001},charged)
    assert np.isclose(loads['frontend'],.4) and np.isclose(loads['backend'],.05)


@pytest.mark.parametrize('installed',[True,False])
def test_cpu_service_refresh_uses_real_safety_return_and_keeps_guard_semantics(installed):
    from environment import Gate
    from controller_architecture_study.held_control_safety import GateHoldAnalyzer,GateHoldBatch,HoldBounds
    obj=service_stub();obj.apply_rates=True;obj.end_s=1.;obj.refresh_rows=[]
    raw=SimpleNamespace(position=np.zeros(3),velocity=np.zeros(3),quaternion=np.array([1.,0.,0.,0.]))
    obj.states={'a':dict(raw=raw,held_control=[9.80665,0.,0.,0.],actuation_s=.4398,
        observation=dict(sample_s=.439802),control_token=8)}
    bounds=HoldBounds(2.,.01,.05,.15,10.,'test assumptions; not plant certified')
    obj.review_batch=GateHoldBatch({'a':GateHoldAnalyzer([Gate(1,np.array([100.,0.,0.]),
        np.array([1.,0.,0.,0.]))],2.,bounds)})
    plans={'a':dict(vehicle='a',sample_id='a:8',model='frozen',priority=0,
        planned_release_s=.4583,requested_period_s=.030,nominal_deadline_s=.4883,
        deadline_s=.4479,obligation_actuation_s=.3679)}
    before=deepcopy(plans)
    io=SimpleNamespace(calendar_basis=lambda p:{'a':dict(version=4)},
        guarded_batch=Mock(return_value=dict(passed=installed,installed=installed,
            reason='context_current' if installed else 'calendar_changed')))
    # Use the actual analyzer AND safe_deadline schema through tick, rather
    # than testing the downstream helper with an abbreviated certificate.
    with patch('time.perf_counter',return_value=.4545):
        obj.tick(plans,lambda now:[],io,{'a':.4283},{'a':.4398},1)
    assert obj.refresh_rows[0]['installed'] is installed
    assert np.isclose(plans['a']['deadline_s'],.5198 if installed else .4479)
    for key in ('model','sample_id','planned_release_s','requested_period_s','nominal_deadline_s'):
        assert plans['a'][key]==before['a'][key]
