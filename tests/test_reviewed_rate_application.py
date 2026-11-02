from copy import deepcopy
import ctypes
import json
from threading import Lock
from types import SimpleNamespace
from unittest.mock import Mock,patch

from controller_architecture_study.joint_commit_guard import review
from controller_architecture_study.joint_contract_shadow import JointDecisionShadow,tag_rate_change


def fixture():
    row=dict(vehicle='a',sample_id='a:1',model='m',priority=0,planned_release_s=1.04,
        requested_period_s=.05,nominal_deadline_s=9.,deadline_s=1.09)
    plans={'a':row}
    rec=dict(source_future=deepcopy(plans),proposal=deepcopy(plans),native_basis={'a':{}},
        control_tokens={'a':4},certificates={'a':dict(available=True,deadline_s=1.08)})
    io=SimpleNamespace(guarded_batch=Mock(return_value=dict(passed=True,installed=True,reason='context_current')))
    return plans,rec,io


def test_real_commit_returns_the_exact_checked_schedule_with_correct_nominal_deadline():
    plans,rec,io=fixture()
    with patch('time.perf_counter',return_value=1.):
        result=review(rec,plans,[],io,{'m':dict(segment_estimates_s=[.001])},
            dict(input_s=.0001,output_s=.0001),end_s=2.,inflight_limit=1,commit=True,rates_only=True)
    assert result['installed'] and not result['dry_run']
    installed=result['installed_proposal']['a']
    assert installed['deadline_s']==1.08 and abs(installed['nominal_deadline_s']-1.09)<1e-12
    assert plans['a']['nominal_deadline_s']==9.  # caller updates plans only after atomic success
    assert io.guarded_batch.call_args.kwargs=={'commit':True}


def test_rates_only_application_cannot_install_a_different_network():
    plans,rec,io=fixture();rec['proposal']['a']['model']='new'
    with patch('time.perf_counter',return_value=1.):
        result=review(rec,plans,[],io,{'m':dict(segment_estimates_s=[.001]),'new':dict(segment_estimates_s=[.001])},
            dict(input_s=.0001,output_s=.0001),end_s=2.,inflight_limit=1,commit=True,rates_only=True)
    assert result['reason']=='rates_only_rejects_model_change'
    io.guarded_batch.assert_not_called()


def test_legal_twenty_ms_interval_is_normalized_before_native_installation():
    plans,rec,io=fixture();rec['proposal']['a']['requested_period_s']=.020-2e-13
    with patch('time.perf_counter',return_value=1.):
        result=review(rec,plans,[],io,{'m':dict(segment_estimates_s=[.001])},
            dict(input_s=.0001,output_s=.0001),end_s=2.,inflight_limit=1,commit=True,rates_only=True)
    assert result['installed_proposal']['a']['requested_period_s']==.020
    assert io.guarded_batch.call_args.args[1]['a']['requested_period_s']==.020


def test_commit_notifications_are_delivered_to_the_single_frontend_owner_once():
    observer=JointDecisionShadow.__new__(JointDecisionShadow)
    observer.committed_updates=[{'a':{'requested_period_s':.03}}]
    assert observer.drain_committed()==[{'a':{'requested_period_s':.03}}]
    assert observer.drain_committed()==[]


def test_change_evidence_ignores_clock_rounding_but_counts_real_period_and_phase_changes():
    previous=dict(requested_period_s=.020,planned_release_s=10000.,deadline_s=10000.02)
    row={k:v+2e-12 for k,v in previous.items()}
    tag_rate_change(row,previous,7)
    assert not row['joint_rate_changed'] and not row['joint_deadline_changed']
    row['planned_release_s']+=.005
    tag_rate_change(row,previous,8)
    assert row['joint_release_changed'] and not row['joint_period_changed']
    row['requested_period_s']=.030
    tag_rate_change(row,previous,9)
    assert row['joint_period_changed'] and row['joint_commit_generation']==9


def test_application_updates_python_calendar_only_after_native_success():
    plans,rec,io=fixture()
    rec['proposal']['a']['requested_period_s']=.030
    encoded=json.dumps(rec).encode();memory=ctypes.create_string_buffer(encoded)
    observer=JointDecisionShadow.__new__(JointDecisionShadow)
    observer.reply=(memory,SimpleNamespace(value=len(encoded)),SimpleNamespace(value=1),Lock())
    observer.reply_seen=0;observer.fresh_review=True;observer.apply_rates=True
    observer.review_ms=[];observer.review_rows=[];observer.installed_groups=[];observer.committed_updates=[]
    observer.states={'a':dict(raw=SimpleNamespace(),observation={'sample_s':.999},held_control=[9.8,0.,0.,0.],
        actuation_s=.998,control_token=4)}
    observer.review_batch=SimpleNamespace(analyze=lambda *a:{'a':dict(budget_s=.15,is_safety_certificate=False)})
    observer.profile={'m':dict(segment_estimates_s=[.001])}
    observer.costs=dict(input_s=.0001,output_s=.0001);observer.end_s=2.
    io.calendar_basis=lambda plans:{'a':dict(sample_id=plans['a']['sample_id'],version=1)}
    with patch('time.perf_counter',return_value=1.):
        observer.review_latest(plans,lambda now:[],io,1,{'a':.99},{'a':.998})
    assert plans['a']['requested_period_s']==.030
    assert plans['a']['planned_release_s']==1.02
    assert plans['a']['deadline_s']==1.078
    assert len(observer.installed_groups)==1 and observer.review_rows[0]['installed']
    assert observer.drain_committed()[0]['a']['joint_commit_generation']==1
