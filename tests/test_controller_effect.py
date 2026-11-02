from types import SimpleNamespace
import numpy as np
import pytest

from controller_architecture_study.controller_effect import features241,live_features,reference_from_spec,reference_spec,outcome_utility,action_features,physical_context,contextual_live_features,EffectRanker
from controller_architecture_study.framework_runtime_features import encode_direct
from controller_architecture_study.fast_convex_preprocess import FastReferenceWindow
from controller_architecture_study.fly_decision_dnn import StoredRoutePlanner
from controller_architecture_study.framework_campaign import layout_for


def test_reduced_live_features_match_frozen_241d_contract_without_full_encoding():
    planner=StoredRoutePlanner(layout_for(0,3,4,'continuous'))
    window=FastReferenceWindow(planner.trajectory,planner.config)
    restored=reference_from_spec(reference_spec(window,.4))
    state=SimpleNamespace(position=np.array([2.,.3,-2.]),velocity=np.array([3.,.2,-.1]),
        quaternion=np.array([.98,.1,.02,-.03]))
    state.quaternion/=np.linalg.norm(state.quaternion)
    held=np.array([10.,.2,-.3,.1])
    for t in (.4,2.,window.trajectory.duration+.1):
        full=features241(np.r_[encode_direct(window,state,t),held])
        direct=live_features(restored,t,dict(position=state.position,velocity=state.velocity,
            quaternion=state.quaternion,held_control=held))
        assert np.allclose(full,direct,atol=2e-6,rtol=1e-6)


def test_collision_does_not_gain_from_an_early_truncated_trace():
    row=dict(label=dict(valid=True,collision=True,qualified=False),error_kind='collision')
    assert outcome_utility(row,[])==0.
    row=dict(label=dict(valid=True,collision=False,qualified=True),error_kind=None)
    with pytest.raises(ValueError):outcome_utility(row,[dict(t=.2)])
    trace=[dict(t=0.,state_cost=1.),dict(t=2.,state_cost=1.)]
    assert outcome_utility(row,trace)==1.25


def test_model_specific_time_response_is_representable():
    a=action_features(['a','b'],['a','b'],[.02,.02],[.05,.05])
    assert a.shape==(2,9)
    assert a[0,2]!=a[1,2] and a[0,3]!=a[1,3]


def test_live_geometry_context_matches_observable_training_features():
    from controller_architecture_study.framework_runtime import route_risk
    planner=StoredRoutePlanner(layout_for(0,3,4,'continuous'))
    window=FastReferenceWindow(planner.trajectory,planner.config)
    ref=window.window(2.,horizon=1)[0]
    raw=SimpleNamespace(position=ref.position+[.1,.2,.02],velocity=ref.velocity+[.1,-.2,.1],quaternion=ref.orientation)
    held=np.array([9.8,.1,.2,.3]);risk=route_risk(raw,ref,1.7)
    expected=np.r_[features241(np.r_[encode_direct(window,raw,2.),held]),
        physical_context(dict(risk=risk,box_margin_m=.3))]
    actual=contextual_live_features(window,2.,dict(position=raw.position,velocity=raw.velocity,
        quaternion=raw.quaternion,held_control=held),1.7,.3)
    assert np.allclose(expected,actual,atol=2e-6)


def test_advantage_guard_keeps_baseline_until_evidence_exceeds_error_margin():
    ranker=EffectRanker.__new__(EffectRanker)
    ranker.raw_scores=lambda *a:np.array([.1,.3])
    ranker.costs={'a':1.,'b':2.};ranker.override_margin=.25
    score=ranker.scores(None,['a','b'],[.05,.05],[.02,.02])
    assert score.argmax()==0
    ranker.raw_scores=lambda *a:np.array([.1,.5])
    assert ranker.scores(None,['a','b'],[.05,.05],[.02,.02]).argmax()==1
