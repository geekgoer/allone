from dataclasses import replace
import math
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from environment import Gate
from controller_architecture_study.framework_runtime import analyze_budget, fixed_planning_profile
from controller_architecture_study.recovery_budget import RecoveryAssumptions, recovery_horizon, recovery_wait


ASSUMPTIONS=RecoveryAssumptions((2.,2.),(0.,0.),0.,0.,.150,"test ideal acceleration model only")


def test_closed_form_reserves_stopping_distance():
    wait,margin,status=recovery_wait(.30,1.,2.,0.,0.)
    assert wait==pytest.approx(.050)
    assert margin==pytest.approx(.050)
    assert status=="within_assumed_model"


def test_accelerating_hold_root_protects_every_prefix_then_expires():
    d,v,b,a,lag=.8,.6,2.,3.,.04
    wait,_,_=recovery_wait(d,v,b,a,lag)
    def margin(s):
        t=s+lag
        return d-v*t-.5*a*t*t-(v+a*t)**2/(2*b)
    assert wait>0.
    assert min(margin(s) for s in np.linspace(0.,wait,101))>=-1e-12
    assert margin(wait+.001)<0.


def test_physical_lag_is_subtracted_once_and_changes_no_clock():
    instant=recovery_wait(.30,1.,2.,0.,0.)[0]
    delayed=recovery_wait(.30,1.,2.,0.,.020)[0]
    assert instant-delayed==pytest.approx(.020)
    assert recovery_wait(.30,1.,2.,0.,.080)[2]=="response_lag_exhausted"


def test_outside_recovery_set_does_not_invent_positive_twenty_ms_budget():
    wait,margin,status=recovery_wait(.10,1.,2.,0.,0.)
    assert wait==0. and margin<0.
    assert status=="outside_local_recovery_set"


def test_inward_motion_and_stationary_boundary_are_not_closing_motion():
    assert math.isinf(recovery_wait(.30,-1.,2.,0.,0.)[0])
    assert math.isinf(recovery_wait(0.,0.,2.,0.,0.)[0])
    assert recovery_wait(.30,-1.,2.,3.,0.)[0]>0.


@pytest.mark.parametrize("value",[0.,-1.,float("nan"),float("inf")])
def test_invalid_braking_bound_is_rejected(value):
    with pytest.raises(ValueError):
        replace(ASSUMPTIONS,braking_deceleration_mps2=(value,2.))


def test_far_gates_are_not_infinite_corridor_walls():
    gate=Gate(1,[30.,0.,-4.],[1.,0.,0.,0.])
    budget,details=recovery_horizon([0.,2.,-4.],[5.,1.,0.],[gate],1.7,ASSUMPTIONS)
    assert budget==.150 and not details


def test_future_aperture_misalignment_is_not_a_collision_or_zero_budget():
    gate=Gate(1,[10.,0.,-4.],[1.,0.,0.,0.])
    budget,details=recovery_horizon([9.,.7,-4.],[5.,-1.,0.],[gate],1.7,ASSUMPTIONS)
    outside=[r for r in details if not r['applicable']]
    assert outside and outside[0]['status']=='outside_approach_domain'
    assert budget>0.


def test_longitudinal_lookahead_follows_stopping_time_beyond_old_three_metres():
    gate=Gate(1,[10.,0.,-4.],[1.,0.,0.,0.])
    assumed=replace(ASSUMPTIONS,response_lag_s=.08)
    budget,details=recovery_horizon([6.,.025,-4.],[5.,1.,0.],[gate],1.7,assumed)
    assert details and any(r['longitudinal_entry_lower_s']>.6 for r in details)
    assert budget==0.


def test_both_sides_are_checked_even_when_moving_away_from_nearest_side():
    gate=Gate(1,[10.,0.,-4.],[1.,0.,0.,0.])
    budget,details=recovery_horizon([9.9,.15,-4.],[5.,-2.,0.],[gate],1.7,ASSUMPTIONS)
    assert budget==0.
    assert any(r['axis']=='lateral' and r['side']==-1 and r['recovery_margin_m']<0. for r in details)


def test_already_passed_gate_is_ignored():
    gate=Gate(1,[10.,0.,-4.],[1.,0.,0.,0.])
    assert recovery_horizon([11.,1.,-4.],[5.,2.,0.],[gate],1.7,ASSUMPTIONS)==(.150,[])


def test_rotated_gate_has_same_local_budget():
    rotation=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    center=np.array([10.,2.,-4.]);p=np.array([-.1,.025,0.]);v=np.array([5.,1.,0.])
    plain=Gate(1,center,[1.,0.,0.,0.])
    rotated=Gate(1,center,[math.sqrt(.5),0.,0.,math.sqrt(.5)])
    original=recovery_horizon(center+p,v,[plain],1.7,ASSUMPTIONS)[0]
    transformed=recovery_horizon(center+rotation@p,rotation@v,[rotated],1.7,ASSUMPTIONS)[0]
    assert original==pytest.approx(transformed)


def test_budget_integration_only_tightens_original_and_preserves_absolute_time():
    state=SimpleNamespace(position=np.array([9.9,.025,-4.]),velocity=np.array([5.,1.,0.]))
    gate=Gate(1,[10.,0.,-4.],[1.,0.,0.,0.])
    with patch('controller_architecture_study.framework_runtime.safe_horizon',return_value=(.100,[])):
        old=analyze_budget(state,[9.8,0.,0.,0.],10.020,10.,[gate],1.7)
        new=analyze_budget(state,[9.8,0.,0.,0.],10.020,10.,[gate],1.7,recovery_assumptions=ASSUMPTIONS)
    assert old['deadline_s']==pytest.approx(10.080)
    assert new['budget_s']==pytest.approx(.050)
    assert new['deadline_s']==pytest.approx(10.070)
    assert new['recovery']['binding']
    assert not new['uses_control_in_bound'] and not new['is_safety_certificate']
    assert new['held_control']==old['held_control']


def test_no_certificate_returns_a_flag_without_aborting_observation():
    state=SimpleNamespace(position=np.array([9.9,.20,-4.]),velocity=np.array([5.,1.,0.]))
    gate=Gate(1,[10.,0.,-4.],[1.,0.,0.,0.])
    budget=analyze_budget(state,[9.8,0.,0.,0.],10.,10.,[gate],1.7,recovery_assumptions=ASSUMPTIONS)
    assert not budget['valid'] and budget['deadline_s']==10.
    assert budget['recovery']['outside_local_recovery_set']


def test_new_recovery_bound_never_relaxes_an_earlier_geometric_deadline():
    state=SimpleNamespace(position=np.array([9.9,.025,-4.]),velocity=np.array([5.,1.,0.]))
    gate=Gate(1,[10.,0.,-4.],[1.,0.,0.,0.])
    with patch('controller_architecture_study.framework_runtime.safe_horizon',return_value=(.025,[])):
        budget=analyze_budget(state,[9.8,0.,0.,0.],10.,10.,[gate],1.7,recovery_assumptions=ASSUMPTIONS)
    assert budget['budget_s']==.025 and budget['deadline_s']==10.025
    assert not budget['recovery']['binding']


def test_paired_planning_uses_an_independent_copy_of_identical_segment_costs():
    frozen={'m':{'segment_estimates_s':[.001,.002],'is_WCET':False}}
    live={'m':{'segment_estimates_s':[.010,.020]}}
    selected=fixed_planning_profile(live,frozen)
    assert selected==frozen and selected!=live
    frozen['m']['segment_estimates_s'][0]=.009
    assert selected['m']['segment_estimates_s'][0]==.001
    assert not selected['m']['is_WCET']


@pytest.mark.parametrize('values',[[.001],[],[.001,-.002],[.001,float('nan')]])
def test_wrong_segment_structure_is_rejected_before_starting_a_flight(values):
    with pytest.raises(ValueError):
        fixed_planning_profile({'m':{'segment_estimates_s':[.001,.002]}},
            {'m':{'segment_estimates_s':values}})
