from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock,patch

from controller_architecture_study.joint_commit_guard import review,rebase_slots,repair_current_rates


def setup():
    row=dict(vehicle='a',sample_id='a:1',model='m',priority=0,planned_release_s=1.04,
        requested_period_s=.05,deadline_s=1.09)
    plans={'a':row}
    recommendation=dict(source_future=deepcopy(plans),proposal=deepcopy(plans),
        native_basis={'a':dict(sample_id='a:1',version=2)},control_tokens={'a':8},
        certificates={'a':dict(available=True,deadline_s=1.08)})
    io=SimpleNamespace(guarded_batch=Mock(return_value=dict(passed=True,installed=False,reason='context_current')))
    return plans,recommendation,io,{'m':dict(segment_estimates_s=[.001])}


def test_fresh_review_checks_all_work_and_native_group_without_installing():
    plans,rec,io,profile=setup()
    expired=dict(vehicle='b',sample_id='b:0',model='m',priority=1,stage='input',
        planned_release_s=.9,deadline_s=.99,requested_period_s=.05)
    with patch('time.perf_counter',return_value=1.):
        result=review(rec,plans,[expired],io,profile,dict(input_s=.0001,output_s=.0001),end_s=2.,inflight_limit=2)
    assert result['passed'] and not result['installed']
    assert result['historical_misses']==['b:0'] and not result['historical_deadlines_repaired']
    assert io.guarded_batch.call_args.kwargs=={'commit':False}
    assert io.guarded_batch.call_args.args[1]['a']['deadline_s']==1.08


def test_unchanged_participant_mutation_invalidates_the_whole_review():
    plans,rec,io,profile=setup()
    plans['a']['model']='different'
    with patch('time.perf_counter',return_value=1.):
        result=review(rec,plans,[],io,profile,{},end_s=2.,inflight_limit=2)
    assert result['reason']=='frontend_plan_changed'
    io.guarded_batch.assert_not_called()


def test_new_queue_work_cannot_be_hidden_by_an_old_feasible_forecast():
    plans,rec,io,profile=setup()
    blocked=dict(vehicle='b',sample_id='b:0',model='slow',priority=0,stage='input',
        planned_release_s=.9,deadline_s=1.01,requested_period_s=.05)
    profile['slow']=dict(segment_estimates_s=[.2])
    with patch('time.perf_counter',return_value=1.):
        result=review(rec,plans,[blocked],io,profile,dict(input_s=.0001,output_s=.0001),end_s=2.,inflight_limit=1)
    assert not result['passed']
    io.guarded_batch.assert_not_called()


def test_rebase_reuses_configuration_but_never_old_ids_or_certificates():
    plans,rec,io,profile=setup()
    rec['proposal']['a'].update(model='fast',requested_period_s=.020)
    # The old snapshot is from a different command/slot epoch.
    rec['certificates']['a']['deadline_s']=.8
    plans['a']=dict(plans['a'],sample_id='a:4',planned_release_s=1.06,minimum_release_s=1.06,deadline_s=1.09)
    certs={'a':dict(available=True,sample_s=.999,actuation_s=.99,deadline_s=1.08)}
    rebuilt,info=rebase_slots(rec,plans,[],certs,{'a':.99},1.)
    assert rebuilt['proposal']['a']['sample_id']=='a:4'
    assert rebuilt['proposal']['a']['planned_release_s']==1.01
    assert rebuilt['proposal']['a']['deadline_s']==1.08
    assert rebuilt['certificates'] is certs
    assert not info['old_slot_ids_reused'] and not info['old_certificates_reused']
    assert rec['certificates']['a']['deadline_s']==.8


def test_rebase_does_not_move_a_past_desired_release_to_now():
    plans,rec,io,profile=setup()
    rec['proposal']['a']['requested_period_s']=.020
    rebuilt,info=rebase_slots(rec,plans,[],rec['certificates'],{'a':.9},1.)
    assert rebuilt['proposal']['a']['planned_release_s']==plans['a']['planned_release_s']
    assert info['advice_already_passed_deferred']==['a']


def test_fresh_rate_repair_uses_current_models_and_protects_every_loop():
    row=dict(vehicle='a',sample_id='a:3',model='m',priority=0,planned_release_s=1.07,
        requested_period_s=.080,deadline_s=1.035)
    plans={'a':row,'b':{**row,'vehicle':'b','sample_id':'b:7'}}
    certs={v:dict(available=True,deadline_s=1.035) for v in plans}
    with patch('time.perf_counter',return_value=1.):
        proposal,info=repair_current_rates(plans,[],certs,{'a':.99,'b':.99},1.,
            {'m':dict(segment_estimates_s=[.001])},dict(input_s=.0001,output_s=.0001),inflight_limit=2,budget_s=.001)
    assert info['accepted'] and set(proposal)=={'a','b'}
    for v,r in proposal.items():
        assert r['sample_id']==plans[v]['sample_id'] and r['model']=='m'
        assert r['planned_release_s']<1.035 and r['deadline_s']==1.035


def test_one_feasible_job_does_not_admit_an_overloaded_periodic_configuration():
    plans,rec,io,profile=setup()
    rec['proposal']['a']['requested_period_s']=.020
    profile['m']['segment_estimates_s']=[.030]  # 150% sustained backend load.
    with patch('time.perf_counter',return_value=1.):
        result=review(rec,plans,[],io,profile,dict(input_s=.0001,output_s=.0001),end_s=2.,inflight_limit=2)
    assert result['reason']=='fresh_periodic_load_exceeds_capacity'
    io.guarded_batch.assert_not_called()
