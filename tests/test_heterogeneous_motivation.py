from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from controller_architecture_study.heterogeneous_motivation import (
    make_case,initial_configuration,cached_score,score_windows,propose,configuration_key)
from controller_architecture_study.fly_decision_dnn import StoredRoutePlanner
from controller_architecture_study.framework_runtime_world import FrameworkWorld


def test_heterogeneous_routes_start_on_the_same_cross_section_at_the_same_speed():
    case=make_case()
    for layout in case['layouts'].values():
        c=np.asarray(layout['segments'][0]['coefficients'])
        np.testing.assert_allclose(c[:,0],[0.,0.,-4.])
        np.testing.assert_allclose(c[:,1],[5.,0.,0.])
        np.testing.assert_allclose(layout['gates'][0]['center'],[10.5,0.,-4.])
    assert case['phase_s']==case['phase_span_s']==0.
    assert {case['layouts'][f'Drone{i}']['family'] for i in (1,3)}=={'continuous'}
    assert {case['layouts'][f'Drone{i}']['family'] for i in range(4,9)}=={'smooth'}
    assert {case['layouts'][f'Drone{i}']['family'] for i in (9,10)}=={'late'}


def test_reference_is_C2_and_fixed_durations_are_not_silently_stretched():
    case=make_case(seed=2,difficulty=1.1)
    for layout in case['layouts'].values():
        trajectory=StoredRoutePlanner(layout).trajectory
        assert np.isclose(trajectory.duration,13.02)
        for left,right in zip(trajectory.segments[:-1],trajectory.segments[1:]):
            for a,b in zip(left.primitive.evaluate(left.primitive.duration),right.primitive.evaluate(0.)):
                np.testing.assert_allclose(a,b,atol=1e-9)


def test_cached_J_combines_costs_and_window_counts_not_scalar_J_values():
    case=make_case();trace=[dict(t=float(t),state_cost=2.) for t in np.arange(0.,14.72,.02)]
    score=cached_score(trace,case)
    assert np.isclose(score['J_hat'],1/42.)
    assert np.isclose(score['short_J'],1/42.)
    windows=score_windows(trace,case['duration_s'])
    assert len(windows)==35 and score['virtual_windows']==200
    assert not score['full_mission_safety_inferred']


def test_partial_collision_trace_does_not_get_a_full_window_score():
    assert cached_score([dict(t=0.,state_cost=1.),dict(t=2.,state_cost=1.)],make_case()) is None


def test_global_configuration_is_heterogeneous_and_proposals_keep_all_vehicles():
    vehicles=[f'Drone{i}' for i in range(1,11)]
    base=initial_configuration(vehicles)
    assert len(set(base['models'].values()))>1
    assert len(set(base['periods_ms'].values()))>1
    rng=np.random.default_rng(123);seen={configuration_key(base)}
    for i in range(1,20):
        c=propose(base,i,rng,seen)
        assert set(c['models'])==set(c['periods_ms'])==set(c['priority_order'])==set(vehicles)
        assert configuration_key(c) not in seen
        seen.add(configuration_key(c))


def test_failure_stop_only_addresses_the_failed_vehicle_without_global_pause():
    calls=[]
    def xyz(*a):return a
    class Client:
        def cancelLastTask(self,**k):calls.append(('cancel',k))
        def simSetKinematics(self,kin,**k):calls.append(('kin',k));assert kin.linear_velocity==(0.,0.,0.)
        def hoverAsync(self,**k):calls.append(('hover',k))
        def ping(self):calls.append(('ping',{}))
    world=FrameworkWorld.__new__(FrameworkWorld);world.client=Client()
    api=SimpleNamespace(KinematicsState=SimpleNamespace,Vector3r=xyz,Quaternionr=xyz)
    with patch('controller_architecture_study.framework_runtime_world.require_airsim',return_value=api):
        world.park_vehicle('Drone1',SimpleNamespace(position=np.array([10.,0.,-4.]),quaternion=np.array([1.,0.,0.,0.])))
    assert [c[0] for c in calls]==['cancel','kin','hover','ping']
    assert all(k.get('vehicle_name','Drone1')=='Drone1' for _,k in calls)


def test_cached_prefix_ends_before_late_turn_enters_the_two_second_preview():
    case=make_case()
    for v in ('Drone9','Drone10'):
        segments=case['layouts'][v]['segments']
        start=next(s['start_time'] for s in segments if abs(np.asarray(s['coefficients'])[1,3:]).max()>1e-9)
        assert case['cache_interval_s'][1]+2.<start


def test_shared_guard_relaxes_low_risk_loops_to_keep_the_same_compute_budget():
    from controller_architecture_study.heterogeneous_load_guard import allocate,load
    vehicles=[f'Drone{i}' for i in range(10)]
    models={v:'big' for v in vehicles};costs={'big':.0027}
    desired={v:.020 for v in vehicles}
    urgency={v:10. if i<3 else 0. for i,v in enumerate(vehicles)}
    result,info=allocate(desired,urgency,models,costs)
    assert info['feasible'] and load(result,models,costs)<=.95
    assert all(result[v]==.020 for v in vehicles[:3])
    assert any(result[v]>.020 for v in vehicles[3:])


def test_shared_guard_reports_infeasibility_without_claiming_a_response_bound():
    from controller_architecture_study.heterogeneous_load_guard import allocate
    _,info=allocate({'a':.020},{'a':1.},{'a':'too_slow'},{'too_slow':.1})
    assert not info['feasible']


def test_frozen_compute_costs_can_be_reconstructed_from_logged_capacity_sums():
    from controller_architecture_study.heterogeneous_load_guard import recover_profile_costs
    rows=[dict(configuration=dict(models={'a':'m1','b':'m2'},periods_ms={'a':20.,'b':40.}),capacity=dict(compute_load=.15)),
        dict(configuration=dict(models={'a':'m1','b':'m2'},periods_ms={'a':40.,'b':20.}),capacity=dict(compute_load=.225))]
    costs,info=recover_profile_costs(rows)
    assert np.isclose(costs['m1'],.001) and np.isclose(costs['m2'],.004)
    assert info['max_residual']<1e-12


def test_speed_changes_time_and_dynamics_without_moving_the_gates_farther_apart():
    slow=make_case(speed=5.);fast=make_case(speed=6.)
    for name,a in slow['layouts'].items():
        b=fast['layouts'][name]
        np.testing.assert_allclose([g['center'] for g in a['gates']],[g['center'] for g in b['gates']])
        for left,right in zip(a['segments'],b['segments']):
            assert np.isclose(right['duration'],left['duration']/1.2)
            np.testing.assert_allclose(right['coefficients'],np.asarray(left['coefficients'])*1.2**np.arange(6),atol=1e-8)
    assert fast['duration_s']<slow['duration_s']


def test_twelve_vehicle_plan_does_not_leave_the_last_late_route_straight():
    case=make_case(count=12)
    for v in ('Drone9','Drone10','Drone11','Drone12'):
        assert any(abs(g['center'][1])>.1 for g in case['layouts'][v]['gates'])
