from copy import deepcopy
from unittest.mock import patch

import pytest


def job(v,period,release):
    return dict(vehicle=v,sample_id=v+':0',model='m',priority=0,requested_period_s=period,
        planned_release_s=release,minimum_release_s=.020,deadline_s=.090,nominal_deadline_s=release+period)


def test_demand_guard_accounts_for_serial_frontend_and_preserves_urgency_order():
    from controller_architecture_study.heterogeneous_load_guard import allocate,resource_loads
    models={v:'m' for v in 'abc'}
    costs=dict(input_s=.004,output_s=.002,planner_reserve_s=.001)
    selected,info=allocate(dict(a=.020,b=.020,c=.020),dict(a=10.,b=0.,c=0.),models,{'m':.0001},pipeline_costs=costs)
    assert info['feasible'] and selected['a']==.020
    assert selected['b']>.020 or selected['c']>.020
    assert max(resource_loads(selected,models,{'m':.0001},costs).values())<=.95


def test_next_batch_can_pass_while_its_repeating_frontend_load_is_infeasible():
    from controller_architecture_study.heterogeneous_load_guard import resource_loads
    from controller_architecture_study.framework_runtime import simulate_pipeline
    plans=[job('a',.030,0.),job('b',.030,0.)]
    costs=dict(input_s=.009,output_s=.009)
    profile={'m':dict(segment_estimates_s=[.0001])}
    assert simulate_pipeline(0.,plans,[],profile,costs)[0]
    assert resource_loads({r['vehicle']:r['requested_period_s'] for r in plans},
        dict(a='m',b='m'),{'m':.0001},costs)['frontend']>1.


def test_joint_slowdown_and_speedup_can_pass_when_speedup_alone_cannot():
    from controller_architecture_study.heterogeneous_deadlines import admit_group
    base={'a':job('a',.040,.040),'b':job('b',.020,.020)}
    group={'a':job('a',.020,.020),'b':job('b',.060,.060)}
    profile={'m':dict(segment_estimates_s=[.001])}
    costs=dict(input_s=.006,output_s=.005,planner_reserve_s=.003,completion_guard_s=0.)
    before=deepcopy(base)
    with patch('controller_architecture_study.heterogeneous_deadlines.time.perf_counter',return_value=0.):
        single,info=admit_group(0.,base,[],{**base,'a':group['a']},profile,costs,{'m':.001})
        assert single==base and not info['future_feasible']
        selected,info=admit_group(0.,base,[],group,profile,costs,{'m':.001})
    assert info['future_feasible'] and selected==group and base==before
    assert all(selected[v]['deadline_s']==base[v]['deadline_s'] for v in base)


def test_group_cannot_relax_physical_deadlines_or_change_a_released_job():
    from controller_architecture_study.heterogeneous_deadlines import admit_group
    base={'a':job('a',.030,.030)};bad={'a':{**base['a'],'deadline_s':.2}}
    with pytest.raises(ValueError,match='commitment'):
        admit_group(0.,base,[],bad,{},dict(),{})
    bad={'a':{**base['a'],'sample_id':'a:1'}}
    with pytest.raises(ValueError,match='commitment'):
        admit_group(0.,base,[],bad,{},dict(),{})


def test_group_rejects_a_rate_feasible_change_that_breaks_another_loops_deadline():
    from controller_architecture_study.heterogeneous_deadlines import admit_group
    base={'a':job('a',.060,.020),'b':job('b',.060,.020)}
    base['b']['deadline_s']=.025
    candidate=deepcopy(base);candidate['a']['requested_period_s']=.020
    profile={'m':dict(segment_estimates_s=[.006])}
    costs=dict(input_s=.001,output_s=.001,planner_reserve_s=0.,completion_guard_s=0.)
    with patch('controller_architecture_study.heterogeneous_deadlines.time.perf_counter',return_value=0.):
        selected,info=admit_group(0.,base,[],candidate,profile,costs,{'m':.006})
    assert selected==base and not info['future_feasible']


def test_aged_input_cannot_be_starved_by_expired_planning_or_other_inputs():
    from controller_architecture_study.probe_seven_continuous_timing import next_frontend_work
    old=job('old',.020,0.);old['sample_response_received_s']=.001
    new=job('new',.020,.020);new['deadline_s']=.021;new['sample_response_received_s']=.019
    maintenance=job('maintenance',.020,0.);maintenance['deadline_s']=.005
    kind,v,_=next_frontend_work({},dict(old=(None,old),new=(None,new)),{},.022,
        lambda r:(r['deadline_s'],r['planned_release_s']),protect=True,
        planning={'maintenance':maintenance},input_wait_limit_s=.010)
    assert (kind,v)==('input','old')


def test_input_wait_rule_does_not_override_a_live_command_delivery_guard():
    from controller_architecture_study.probe_seven_continuous_timing import input_blocks_command
    assert input_blocks_command(.020,{'vehicle':'a'},
        {'b':dict(vehicle='b',enqueue_s=.019,deadline_s=.022)},dict(input_s=.002,command_s=.001))


def test_resource_relief_cannot_slow_recovery_beyond_its_original_period():
    from controller_architecture_study.heterogeneous_load_guard import allocate
    selected,info=allocate(dict(a=.020,b=.020,c=.020),dict(a=1.,b=2.,c=3.),
        dict(a='m',b='m',c='m'),{'m':.001},pipeline_costs=dict(input_s=.007),
        max_periods=dict(a=.030,b=.030,c=.030))
    assert info['feasible'] and max(selected.values())<=.030
    _,info=allocate(dict(a=.020,b=.020),{},dict(a='m',b='m'),{'m':.001},
        pipeline_costs=dict(input_s=.020),max_periods=dict(a=.030,b=.030))
    assert not info['feasible']
