from copy import deepcopy
from unittest.mock import patch

import pytest

from controller_architecture_study.framework_releases import retained_successor
from controller_architecture_study.framework_runtime import simulate_pipeline
from controller_architecture_study.heterogeneous_deadlines import admit_change
from controller_architecture_study.probe_seven_continuous_timing import next_frontend_work,input_blocks_command,planning_obligation


def context():
    profile={'small':{'segment_estimates_s':[.001,.001]},
             'large':{'segment_estimates_s':[.003,.003,.003]},
             'other':{'segment_estimates_s':[.002,.002,.002]}}
    costs=dict(input_s=.001,output_s=.001,completion_guard_s=0.)
    candidate=dict(vehicle='a',sample_id='a:1',model='large',priority=0,planned_release_s=0.,
                   minimum_release_s=0.,requested_period_s=.020,deadline_s=.014)
    fallback={**candidate,'model':'small','requested_period_s':.030}
    pending=[dict(vehicle='b',sample_id='b:0',model='other',priority=1,planned_release_s=0.,
                  deadline_s=.017,stage='compute',ingress_remaining_s=0.)]
    return candidate,fallback,pending,profile,costs


def test_change_is_rolled_back_if_another_loops_deadline_would_be_broken():
    candidate,fallback,pending,profile,costs=context()
    before=deepcopy((candidate,fallback,pending))
    with patch('controller_architecture_study.heterogeneous_deadlines.time.perf_counter',return_value=0.):
        row,info=admit_change(0.,{},pending,candidate,fallback,profile,costs)
    assert info['feasible'] and info['decision']=='retained_previous' and info['checked']==2
    assert row['model']=='small' and row['deadline_s']==candidate['deadline_s']
    assert set(info['covered_ids'])=={'a:1','b:0'}
    assert (candidate,fallback,pending)==before


def test_not_yet_released_other_plan_is_included_and_never_rescheduled():
    candidate,fallback,_,profile,costs=context()
    other=dict(vehicle='c',sample_id='c:1',model='other',priority=1,planned_release_s=.001,
               minimum_release_s=.001,deadline_s=.015)
    plans={'c':other};before=deepcopy(plans)
    with patch('controller_architecture_study.heterogeneous_deadlines.time.perf_counter',return_value=0.):
        row,info=admit_change(0.,plans,[],candidate,fallback,profile,costs)
    assert 'c:1' in info['covered_ids'] and plans==before
    assert info['feasible'] and simulate_pipeline(0.,[row,other],[],profile,costs)[0]


def test_neither_feasible_retains_configuration_but_never_claims_admission():
    candidate,fallback,pending,profile,costs=context()
    candidate['deadline_s']=fallback['deadline_s']=.001
    with patch('controller_architecture_study.heterogeneous_deadlines.time.perf_counter',return_value=0.):
        row,info=admit_change(0.,{},pending,candidate,fallback,profile,costs)
    assert not info['feasible'] and info['decision']=='infeasible_retained'
    assert row==fallback and row['deadline_s']==.001


def test_fallback_cannot_hide_failure_by_extending_its_deadline():
    candidate,fallback,pending,profile,costs=context()
    fallback['deadline_s']+=.1
    with pytest.raises(ValueError,match='same physical obligation'):
        admit_change(0.,{},pending,candidate,fallback,profile,costs)


def test_an_already_expired_other_job_is_reported_without_spending_two_simulations():
    candidate,fallback,pending,profile,costs=context()
    pending[0]['deadline_s']=-.001
    with patch('controller_architecture_study.heterogeneous_deadlines.simulate_pipeline') as simulate:
        row,info=admit_change(0.,{},pending,candidate,fallback,profile,costs)
    simulate.assert_not_called()
    assert not info['feasible'] and info['checked']==0 and info['already_expired_ids']==['b:0']
    assert row==fallback


def test_old_late_ack_is_not_treated_as_a_new_unsatisfied_obligation():
    candidate,fallback,_,profile,costs=context()
    candidate['model']='small'
    acknowledged=dict(vehicle='b',sample_id='b:0',model='other',priority=1,planned_release_s=-.01,
        deadline_s=-.005,stage='analysis',command_completed_s=-.001)
    with patch('controller_architecture_study.heterogeneous_deadlines.time.perf_counter',return_value=0.):
        row,info=admit_change(0.,{},[acknowledged],candidate,fallback,profile,costs)
    assert info['feasible'] and info['decision']=='accepted'
    assert acknowledged['command_completed_s']>acknowledged['deadline_s']


def test_rollback_keeps_old_configuration_but_the_fresh_absolute_deadline():
    candidate,fallback,_,_,_=context()
    previous=dict(model='small',requested_period_s=.030,planned_release_s=.010,deadline_s=.020)
    row=retained_successor(candidate,previous,.050)
    assert row['model']=='small' and row['requested_period_s']==.030
    assert row['planned_release_s']==.050 and row['deadline_s']==candidate['deadline_s']
    assert candidate['model']=='large'


def test_a_completed_jobs_expired_deadline_cannot_outrank_ready_input():
    old=dict(vehicle='old',sample_id='old:0',priority=0,deadline_s=.001,planned_release_s=0.,rpc_return_s=.002)
    new=dict(vehicle='new',sample_id='new:0',priority=1,deadline_s=.080,planned_release_s=.020)
    key=lambda r:(r['deadline_s'],r['planned_release_s'])
    assert next_frontend_work({'old':(None,old)},{'new':(None,new)},{},.030,key,'deadline')[0]=='completed'
    assert next_frontend_work({'old':(None,old)},{'new':(None,new)},{},.030,key,'deadline',protect=True)[:2]==('input','new')
    assert next_frontend_work({'old':(None,old)},{},{},.030,key,'deadline',protect=True)[0]=='completed'


def test_input_guard_waits_for_live_urgent_output_but_does_not_deadlock_on_expired_work():
    other=dict(vehicle='b',enqueue_s=0.,deadline_s=.005)
    costs=dict(input_s=.003,command_s=.002,handoff_s=.001)
    assert input_blocks_command(0.,{'vehicle':'a'},{'b':other},costs)
    assert not input_blocks_command(.006,{'vehicle':'a'},{'b':other},costs)
    assert not input_blocks_command(0.,{'vehicle':'b'},{'b':other},costs)
    assert not input_blocks_command(0.,{'vehicle':'a'},{'b':{**other,'rpc_return_s':.001}},costs)


def test_next_release_planning_cannot_starve_behind_less_urgent_inputs():
    old=dict(vehicle='a',sample_id='a:0',priority=1,deadline_s=.005,
             planned_release_s=.010,requested_period_s=.020,rpc_return_s=.015)
    new=dict(vehicle='b',sample_id='b:0',priority=0,deadline_s=.080,planned_release_s=.020)
    work={'a':planning_obligation(old)}
    key=lambda r:(r['deadline_s'],r['planned_release_s'])
    kind,v,_=next_frontend_work({'a':(None,old)},{'b':(None,new)},{},.021,key,'deadline',protect=True,planning=work)
    assert (kind,v)==('planning','a') and work['a']['deadline_s']==.030
    assert old['deadline_s']==.005


def test_cleanup_priority_uses_its_upcoming_release_not_its_expired_deadline():
    old=dict(vehicle='a',sample_id='a:0',priority=0,deadline_s=.001,
             planned_release_s=0.,requested_period_s=.020,rpc_return_s=.002)
    new=dict(vehicle='b',sample_id='b:0',priority=1,deadline_s=.080,planned_release_s=.020)
    key=lambda r:(r['deadline_s'],r['planned_release_s'])
    work={'a':planning_obligation(old,next_release=.100)}
    assert next_frontend_work({'a':(None,old)},{'b':(None,new)},{},.030,key,'deadline',protect=True,planning=work)[:2]==('input','b')


def test_global_forecast_charges_other_loops_next_job_planning_work():
    profile={'m':{'segment_estimates_s':[.001]}}
    old=dict(vehicle='a',sample_id='a:0',priority=0,model='m',planned_release_s=0.,requested_period_s=.003,
        deadline_s=.010,stage='analysis',command_completed_s=0.)
    new=dict(vehicle='b',sample_id='b:0',priority=1,model='m',planned_release_s=0.,deadline_s=.0045)
    costs=dict(input_s=.001,output_s=.001,io_s=.001,command_s=.0005,
               sample_io_s=.001,frontend_protection=True,planner_reserve_s=0.)
    assert simulate_pipeline(0.,[new],[old],profile,costs)[0]
    passed,_,_=simulate_pipeline(0.,[new],[old],profile,{**costs,'planner_reserve_s':.003})
    assert not passed
