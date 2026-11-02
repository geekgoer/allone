from unittest.mock import patch
from controller_architecture_study.heterogeneous_deadlines import admit,delivery_fits
from controller_architecture_study.probe_seven_continuous_timing import control_deadline


def plans():
    return {'a':dict(vehicle='a',sample_id='a:0',model='m',priority=0,deadline_s=.1,planned_release_s=.04,minimum_release_s=.02)}


def test_admission_keeps_model_deadline_and_charges_its_own_time():
    with patch('controller_architecture_study.heterogeneous_deadlines.simulate_pipeline',return_value=(True,{'a:0':.08},2)), \
         patch('controller_architecture_study.heterogeneous_deadlines.time.perf_counter',side_effect=[1.,1.01]):
        proposal,info=admit(0.,plans(),[],{},dict(completion_guard_s=.001))
    assert info['feasible'] and abs(info['planning_ms']-10.)<1e-8
    assert proposal['a']['deadline_s']==.1 and proposal['a']['model']=='m'
    assert abs(info['attempts'][0]['finish']['a:0']-.09)<1e-8


def test_feasible_forecast_becomes_rejected_if_planning_itself_spends_the_slack():
    with patch('controller_architecture_study.heterogeneous_deadlines.simulate_pipeline',return_value=(True,{'a:0':.095},2)), \
         patch('controller_architecture_study.heterogeneous_deadlines.time.perf_counter',side_effect=[1.,1.01,1.02,1.02]):
        proposal,info=admit(0.,plans(),[],{},dict(completion_guard_s=0.))
    assert proposal is None and not info['feasible'] and info['checked']==2


def test_one_earlier_release_candidate_does_not_extend_the_deadline():
    with patch('controller_architecture_study.heterogeneous_deadlines.simulate_pipeline',side_effect=[(False,{},1),(True,{'a:0':.05},2)]), \
         patch('controller_architecture_study.heterogeneous_deadlines.time.perf_counter',side_effect=[1.,1.001,1.002]):
        proposal,info=admit(.01,plans(),[],{},dict(completion_guard_s=0.))
    assert info['checked']==2 and proposal['a']['planned_release_s']==.02
    assert proposal['a']['deadline_s']==.1


def test_expired_delivery_is_not_admitted_as_a_success():
    assert not delivery_fits(.099,plans()['a'],dict(actuation_s=.003))
    assert delivery_fits(.095,plans()['a'],dict(actuation_s=.003))


def test_exhausted_pending_envelope_is_explicitly_rejected_without_resetting_time():
    row=dict(sample_id='b:1',sample_io_remaining_s=-.001)
    proposal,info=admit(0.,plans(),[row],{},dict())
    assert proposal is None and info['checked']==0 and info['sample_id']=='b:1'


def test_static_admission_does_not_move_its_frozen_periodic_release():
    with patch('controller_architecture_study.heterogeneous_deadlines.simulate_pipeline',return_value=(False,{},1)) as simulate:
        proposal,info=admit(0.,plans(),[],{},dict(),policy='static',allow_earlier=False)
    assert proposal is None and info['checked']==1
    assert simulate.call_args.kwargs['policy']=='static'


def test_static_release_jitter_is_not_hidden_by_rewriting_its_calendar():
    with patch('controller_architecture_study.heterogeneous_deadlines.simulate_pipeline',return_value=(True,{'a:0':.07},2)), \
         patch('controller_architecture_study.heterogeneous_deadlines.time.perf_counter',side_effect=[1.,1.01]):
        proposal,info=admit(.05,plans(),[],{},dict(),policy='static',allow_earlier=False)
    assert info['feasible'] and proposal['a']['planned_release_s']==.04


def test_physical_deadline_is_not_silently_replaced_by_nominal_period():
    budget={'deadline_s':.070}
    assert control_deadline(.020,budget,0.)==.020
    assert control_deadline(.020,budget,0.,state_only=True)==.070
    assert control_deadline(.020,budget,0.,False,state_only=True)==.080
