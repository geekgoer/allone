import numpy as np

from controller_architecture_study.heterogeneous_motivation import make_case
from controller_architecture_study.fly_decision_dnn import StoredRoutePlanner
from controller_architecture_study.fast_convex_preprocess import FastReferenceWindow
from controller_architecture_study.heterogeneous_prediction import anticipatory_mode


def window(name):
    planner=StoredRoutePlanner(make_case(difficulty=.75)['layouts'][name])
    return FastReferenceWindow(planner.trajectory,planner.config)


def observation():
    return dict(risk=dict(h=.325,hdot=0.,boundary='lateral'),position_error_m=0.,velocity_error_mps=0.)


def test_future_reference_triggers_before_the_first_bend_starts():
    w=window('Drone9');memory={};first=None
    for t in np.arange(0.,7.14,.02):
        active,info=anticipatory_mode(w,t,observation(),1.7,memory)
        if active:first=t;break
    assert first is not None and first<6.14
    assert info['upcoming'] and info['horizon_s']==2.


def test_continuous_bends_do_not_revert_to_cruise_between_acceleration_peaks():
    w=window('Drone9');memory={}
    active=[anticipatory_mode(w,t,observation(),1.7,memory)[0] for t in np.arange(5.8,10.91,.02)]
    assert all(active)


def test_release_needs_half_a_second_of_settled_tracking_and_no_future_bend():
    w=window('Drone4');memory={'active':True}
    assert anticipatory_mode(w,1.,observation(),1.7,memory)[0]
    assert anticipatory_mode(w,1.49,observation(),1.7,memory)[0]
    assert not anticipatory_mode(w,1.51,observation(),1.7,memory)[0]
    memory={'active':True};bad=observation();bad['velocity_error_mps']=1.
    assert anticipatory_mode(w,1.,bad,1.7,memory)[0]
    assert anticipatory_mode(w,2.,bad,1.7,memory)[0]
    assert memory['quiet_since'] is None


def test_straight_tracking_does_not_require_recovery():
    assert not anticipatory_mode(window('Drone4'),4.,observation(),1.7,{})[0]
