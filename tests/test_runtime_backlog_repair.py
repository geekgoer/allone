from copy import deepcopy

import pytest

from controller_architecture_study.framework_releases import ReleaseQueue,release_completed_io,acknowledged_sources
from controller_architecture_study.probe_seven_continuous_timing import next_frontend_work


def job(v='a',i=0):
    return dict(vehicle=v,sample_id=f'{v}:{i}',model='m',priority=0,planned_release_s=i*.020,
        deadline_s=.080,requested_period_s=.020,status='planned')


def test_latest_queue_records_every_arrival_but_keeps_only_unstarted_newest():
    queue=ReleaseQueue(['a','b'],'latest');rows=[]
    for i in range(4):
        row=job(i=i);queue.publish(row,i*.020);rows.append(row)
    assert [r['sample_id'] for r in queue.rows()]==['a:3']
    assert len(queue.superseded)==3 and all('dispatch_s' not in r for r in queue.superseded)
    assert [r['superseded_by'] for r in rows[:3]]==['a:1','a:2','a:3']
    assert all(r['deadline_s']==.080 and r['committed_deadline_s']==.080 for r in rows)


def test_new_arrival_never_discards_or_changes_an_inflight_job():
    queue=ReleaseQueue(['a'],'latest');old=job();queue.publish(old,0.);queue.take('a',.001)
    before=deepcopy(old);pending={'a':old}
    for i in (1,2,3):queue.publish(job(i=i),i*.020)
    assert old==before and queue.dispatchable(pending,10)=={}
    old['io_response_received_s']=.065
    release_completed_io(pending,old,.066)
    assert not pending and queue.dispatchable(pending,10)['a']['sample_id']=='a:3'
    assert old['pipeline_released_s']==.066


def test_completion_of_old_job_cannot_release_a_new_job_slot():
    old=job();old['io_response_received_s']=.050
    with pytest.raises(RuntimeError):release_completed_io({'a':job(i=1)},old,.051)
    with pytest.raises(ValueError):release_completed_io({'a':job()},job(),.051)


def test_feedback_planning_uses_latest_acknowledged_control_not_new_unfinished_job():
    old=job();old.update(status='applied',rpc_return_s=.030)
    feedback={'a':(None,old)};pending={'a':job(i=2)}
    assert acknowledged_sources(pending,feedback)['a'] is old
    new=pending['a'];new.update(status='applied',rpc_return_s=.050)
    assert acknowledged_sources(pending,feedback)['a'] is new


def test_eager_cleanup_cannot_starve_behind_repeated_old_release_deadlines():
    completed={'a':(None,{**job(),'rpc_return_s':.010})}
    inputs={'b':(None,job('b'))}
    key=lambda row:(row['deadline_s'],row['planned_release_s'])
    assert next_frontend_work(completed,inputs,{},.1,key,protect=True,eager_completion=True)[:2]==('completed','a')


def test_recovery_keeps_late_work_cost_and_cannot_ignore_a_future_deadline():
    from controller_architecture_study.framework_runtime import simulate_pipeline
    profile={'m':dict(segment_estimates_s=[.008]),'fast':dict(segment_estimates_s=[.001])}
    costs=dict(input_s=.001,output_s=.001)
    late={**job('old'),'model':'m','deadline_s':-.001,'stage':'compute'}
    healthy={**job('healthy'),'model':'fast','deadline_s':.005}
    assert simulate_pipeline(0.,[healthy],[],profile,costs)[0]
    assert not simulate_pipeline(0.,[healthy],[late],profile,costs,missed_deadline_ids={'old:0'})[0]
    with pytest.raises(ValueError):
        simulate_pipeline(0.,[healthy],[late],profile,costs,missed_deadline_ids={'healthy:0'})


def test_recovery_can_change_future_configuration_but_never_reports_old_miss_repaired():
    from unittest.mock import patch
    from controller_architecture_study.heterogeneous_deadlines import admit_change
    profile={'m':dict(segment_estimates_s=[.001])};costs=dict(input_s=.001,output_s=.001)
    candidate={**job('new',1),'planned_release_s':0.,'minimum_release_s':0.,'deadline_s':.020}
    fallback={**candidate,'requested_period_s':.040}
    late={**job('old'),'deadline_s':-.001,'stage':'compute'}
    before=deepcopy((candidate,fallback,late))
    with patch('controller_architecture_study.heterogeneous_deadlines.time.perf_counter',return_value=0.):
        row,info=admit_change(0.,{},[late],candidate,fallback,profile,costs,recover_after_miss=True)
    assert row==candidate and info['decision']=='recovery_accepted'
    assert not info['feasible'] and info['future_feasible'] and not info['historical_deadlines_repaired']
    assert info['already_expired_ids']==['old:0'] and info['protected_future_ids']==['new:1']
    assert (candidate,fallback,late)==before


def test_recovery_still_rejects_model_growth_that_breaks_another_healthy_loop():
    from unittest.mock import patch
    from controller_architecture_study.heterogeneous_deadlines import admit_change
    profile={'small':dict(segment_estimates_s=[.001,.001]),'large':dict(segment_estimates_s=[.003]*3),
        'other':dict(segment_estimates_s=[.002]*3),'late':dict(segment_estimates_s=[.0001])}
    costs=dict(input_s=.001,output_s=.001)
    candidate={**job('a',1),'planned_release_s':0.,'model':'large','minimum_release_s':0.,'deadline_s':.014}
    fallback={**candidate,'model':'small'}
    pending=[{**job('b'),'model':'other','deadline_s':.017,'stage':'compute','ingress_remaining_s':0.},
        {**job('c'),'model':'late','deadline_s':-.001,'stage':'compute','ingress_remaining_s':0.}]
    with patch('controller_architecture_study.heterogeneous_deadlines.time.perf_counter',return_value=0.):
        row,info=admit_change(0.,{},pending,candidate,fallback,profile,costs,recover_after_miss=True)
    assert row['model']=='small' and info['decision']=='recovery_retained_previous'
    assert 'b:0' in info['protected_future_ids'] and not info['feasible']


def test_deferred_planning_is_charged_but_does_not_hold_a_vehicle_pipeline_slot():
    from controller_architecture_study.framework_runtime import simulate_pipeline
    profile={'m':dict(segment_estimates_s=[.001])}
    costs=dict(input_s=.001,output_s=.001,io_s=.001,command_s=.0005,sample_io_s=.001,frontend_protection=True)
    new={**job('a',1),'planned_release_s':0.,'deadline_s':.009,'serial_vehicle':True}
    old={**job('a'),'planned_release_s':-.020,'stage':'analysis','command_completed_s':-.001,
        'analysis_remaining_s':.006,'nonblocking_maintenance':True,'requested_period_s':.1}
    passed,finish,_=simulate_pipeline(0.,[new],[old],profile,costs,inflight_limit=1)
    assert passed and finish['a:1']<=.009
    assert not simulate_pipeline(0.,[new],[{**old,'nonblocking_maintenance':False}],profile,costs,inflight_limit=1)[0]


def test_older_input_keeps_its_real_time_without_rewinding_newer_observation_trace():
    from types import SimpleNamespace
    from unittest.mock import patch
    import numpy as np
    from controller_architecture_study.framework_runtime_world import FrameworkWorld
    world=FrameworkWorld.__new__(FrameworkWorld)
    world.initial={'a':dict(sim_ns=100,collision_stamp=0)};world.phases={'a':0.}
    world.windows={'a':None};world.offsets={'a':np.zeros(3)};world.net_width=1.7
    world.traces={'a':[dict(t=2.,sim_ns=2000000100)]};world.failures={'a':None}
    raw=SimpleNamespace(timestamp_ns=1000000100,position=np.array([0.,0.,-4.]),velocity=np.zeros(3),
        quaternion=np.array([1.,0.,0.,0.]),acceleration=np.zeros(3),angular_velocity=np.zeros(3))
    hit=SimpleNamespace(time_stamp=0,object_name='',has_collided=False)
    reference=SimpleNamespace(position=raw.position,velocity=raw.velocity,orientation=raw.quaternion)
    before=deepcopy(world.traces)
    with patch('controller_architecture_study.framework_runtime_world.reference_point',return_value=reference), \
         patch('controller_architecture_study.framework_runtime_world.route_risk',return_value={'h':.3}):
        row=world.observe('a',raw,hit,12.,record=False)
        assert row['t']==1. and row['sample_s']==12. and world.traces==before
        with pytest.raises(RuntimeError,match='backwards'):world.observe('a',raw,hit,12.)
        raw.timestamp_ns=99
        with pytest.raises(RuntimeError,match='origin'):world.observe('a',raw,hit,12.,record=False)


def test_forecast_releases_normal_completed_io_even_while_its_planning_is_deferred():
    from controller_architecture_study.framework_runtime import simulate_pipeline
    profile={'m':dict(segment_estimates_s=[.001])}
    costs=dict(input_s=.001,output_s=.006,io_s=.001,command_s=.0005,sample_io_s=.001,frontend_protection=True)
    new={**job('a',1),'planned_release_s':0.,'deadline_s':.009,'serial_vehicle':True}
    old={**job('a'),'planned_release_s':-.020,'stage':'io','command_completed_s':-.001,
        'io_remaining_s':.0001,'requested_period_s':.1}
    assert simulate_pipeline(0.,[new],[old],profile,{**costs,'eager_completion':True},inflight_limit=1)[0]
    assert not simulate_pipeline(0.,[new],[old],profile,costs,inflight_limit=1)[0]


def test_forecast_charges_cleanup_even_when_next_plan_already_exists():
    from types import SimpleNamespace
    from controller_architecture_study.heterogeneous_deadlines import pending_specs
    row={**job(),'rpc_return_s':.010,'next_plan_created':True}
    costs=dict(output_s=.001,planner_reserve_s=.002)
    result=pending_specs(.012,{}, {},{}, {'a':(None,row)},SimpleNamespace(),costs,feedback={})
    assert len(result)==1 and result[0]['nonblocking_maintenance']
    assert result[0]['analysis_remaining_s']==.001
    row['completion_frontend_done_s']=.013
    assert pending_specs(.014,{}, {},{}, {},SimpleNamespace(),costs,feedback={'a':(None,row)})==[]
