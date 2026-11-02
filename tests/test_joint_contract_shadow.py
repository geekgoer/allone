import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import time

import numpy as np
import psutil

from controller_architecture_study.joint_contract_shadow import Trigger,JointDecisionShadow,options_for,freeze_near_releases,decide_snapshot,effect_options_for,timing_options_for


def test_normal_changes_merge_but_an_exhausted_margin_edge_does_not_wait():
    trigger=Trigger(.050)
    trigger.notice('a',0.,.030)
    assert trigger.due(0.)
    trigger.consumed(0.)
    trigger.notice('a',.005,.020);trigger.notice('b',.010,.030)
    assert not trigger.due(.020)
    trigger.notice('b',.021,-.001)
    assert trigger.due(.021)
    batch=trigger.consumed(.021)
    assert batch['reason']=='response_margin_edge' and batch['dirty']==['a','b']
    trigger.notice('b',.022,-.002)
    assert not trigger.due(.022)  # Persistent level does not create per-job edge storms.
    assert trigger.due(.072)


def test_ack_state_update_does_not_toggle_the_fresh_budget_urgency_signal():
    trigger=Trigger(.050)
    trigger.notice('a',0.,.020);trigger.consumed(0.)
    # Many post-command observations carry no NEW safety-budget conclusion.
    for k in range(1,20):
        trigger.notice('a',k*.001)
        assert not trigger.due(k*.001)
    assert trigger.levels['a'] is False
    trigger.notice('a',.021,-.001)
    assert trigger.due(.021)


def test_candidate_prior_does_not_extrapolate_or_change_absolute_deadline():
    row=dict(previous_release_s=1.,planned_release_s=1.030,deadline_s=1.2)
    snapshot=dict(future={'v':row})
    options=options_for(snapshot,{'v':{'m':[(.020,1.),(.040,2.)]}},
        {'m':{'segment_estimates_s':[.001]}},{})
    assert sorted(a['release_s'] for a in options['v'])==[1.020,1.030,1.040]
    assert row['deadline_s']==1.2
    assert all('NOT' in a['quality_scope'] for a in options['v'])
    early=options_for(snapshot,{'v':{'m':[(.020,1.),(.040,2.)]}},
        {'m':{'segment_estimates_s':[.001]}},{},planning_ready_s=1.025)
    assert any(abs(a['release_s']-1.025)<1e-10 for a in early['v'])


def test_near_release_is_immutable_and_its_load_is_not_given_to_other_loops():
    fixed=dict(vehicle='a',sample_id='a:1',model='m',planned_release_s=1.002,
               requested_period_s=.020,deadline_s=1.030)
    free={**fixed,'vehicle':'b','sample_id':'b:1','planned_release_s':1.040}
    snapshot=dict(future={'a':fixed,'b':free},pending=[])
    future,pending,caps,locked=freeze_near_releases(snapshot,1.,{'m':{'segment_estimates_s':[.004]}},
        dict(input_s=.002,output_s=.001))
    assert list(future)==['b'] and pending[0]['sample_id']=='a:1'
    assert pending[0]['deadline_s']==1.030 and fixed.get('stage') is None
    assert np.allclose(caps,[.75,.80])


def test_missing_J_workpoint_is_not_proof_of_timing_infeasibility():
    row=dict(vehicle='v',sample_id='v:1',model='m',priority=0,previous_release_s=0.,
        minimum_release_s=.020,planned_release_s=.040,requested_period_s=.040,deadline_s=.080)
    snapshot=dict(future={'v':row},pending=[],inflight_limit=1)
    profile={'m':{'segment_estimates_s':[.001]}};costs=dict(input_s=.0001,output_s=.0001)
    priors={'v':{'m':[(.080,1.)]}}
    with patch('time.perf_counter',return_value=0.):
        supported,info,*_=decide_snapshot(snapshot,{},0.,profile,costs,priors)
        probe,probe_info,*_=decide_snapshot(snapshot,{},0.,profile,costs,priors,feasibility_only=True)
        restored,restored_info,*_=decide_snapshot(snapshot,{},0.,profile,costs,priors,
            recover_after_miss=True,restore_without_J=True)
    assert supported is None and info['reason']=='no_individually_time_feasible_contract'
    assert probe is not None and probe_info['accepted']
    assert not probe_info['quality_ranked'] and probe_info['selection_objective']=='timing_feasibility_only'
    assert probe['v']['deadline_s']==.080 and row['deadline_s']==.080
    assert restored is not None and restored_info['accepted'] and not restored_info['quality_ranked']
    assert restored_info['ranking_rejection']=='no_individually_time_feasible_contract'


def test_effect_ranker_charges_wait_from_physical_observation_to_first_action():
    ranker=SimpleNamespace(models=['m'],
        supported_many=lambda m,t,r:(t>=.02)&(t<=.08)&(r<=.08),
        scores=lambda state,names,periods,delays:-np.array(delays))
    row=dict(previous_release_s=1.,planned_release_s=1.05,minimum_release_s=1.02,deadline_s=1.15)
    snapshot=dict(future={'v':row},pending=[],states={'v':dict(sample_s=1.01)})
    options=effect_options_for(snapshot,{'m':dict(segment_estimates_s=[.002])},
        dict(sample_io_s=.001,command_s=.001),1.02,ranker,{'v':np.zeros(13)})
    action=next(a for a in options['v'] if abs(a['release_s']-1.05)<1e-9)
    assert abs(action['initial_hold_delay_estimate_s']-.044)<1e-10
    assert 'ADVISORY' in action['quality_scope']
    assert row['deadline_s']==1.15


def test_rate_only_application_does_not_search_for_undeployable_network_changes():
    ranker=SimpleNamespace(models=['fixed','cheaper'],supported_many=lambda m,t,r:np.ones(len(t),dtype=bool),
        scores=lambda state,names,periods,delays:np.zeros(len(names)))
    row=dict(model='fixed',previous_release_s=1.,planned_release_s=1.05,minimum_release_s=1.02,deadline_s=1.15)
    snapshot=dict(rates_only=True,future={'v':row},pending=[],states={'v':dict(sample_s=1.01)})
    profile={m:dict(segment_estimates_s=[.002]) for m in ranker.models}
    prior={'v':{m:[(.020,1.),(.080,2.)] for m in ranker.models}}
    for options in (effect_options_for(snapshot,profile,{},1.02,ranker,{'v':np.zeros(13)}),
            timing_options_for(snapshot,profile,{},1.02),options_for(snapshot,prior,profile,{})):
        assert options['v'] and {r['model'] for r in options['v']}=={'fixed'}


def test_effect_decision_does_not_build_and_throw_away_nominal_prior_candidates():
    row=dict(vehicle='v',sample_id='v:1',model='m',priority=0,previous_release_s=0.,
        minimum_release_s=.020,planned_release_s=.040,requested_period_s=.040,deadline_s=.080)
    snapshot=dict(future={'v':row},pending=[],states={'v':dict(sample_s=0.)},inflight_limit=1)
    ranker=SimpleNamespace(models=['m'],supported_many=lambda m,t,r:np.ones(len(t),dtype=bool),
        scores=lambda state,names,periods,delays:np.zeros(len(names)))
    with patch('time.perf_counter',return_value=0.),patch('controller_architecture_study.joint_contract_shadow.options_for',
            side_effect=AssertionError('unnecessary prior construction')):
        decide_snapshot(snapshot,{},0.,{'m':dict(segment_estimates_s=[.001])},
            dict(input_s=.0001,output_s=.0001),{},effect=(ranker,{'v':np.zeros(13)}))


def test_real_process_mailbox_records_without_any_actuation_interface(tmp_path):
    prior=tmp_path/'priors.json'
    prior.write_text(json.dumps(dict(rows=[dict(configuration=dict(models={'v':'m'},periods_ms={'v':t}),
        result=dict(complete=True,per_vehicle={'v':dict(short_J=1.)})) for t in (20.,40.,60.)])),encoding='utf-8')
    shadow=JointDecisionShadow({'m':{'segment_estimates_s':[.001]}},dict(input_s=.0001,output_s=.0001),
        prior,psutil.Process().cpu_affinity()[0],tmp_path/'shadow',geometry={'v':dict(net_width=2.,
            gates=[dict(index=1,center=[100.,0.,0.],orientation=[1.,0.,0.,0.])])})
    now=time.perf_counter()
    row=dict(vehicle='v',sample_id='v:1',model='m',priority=0,previous_release_s=now-.020,
        planned_release_s=now+.040,requested_period_s=.060,minimum_release_s=now,deadline_s=now+.090)
    original=dict(row)
    raw=SimpleNamespace(position=np.zeros(3),velocity=np.zeros(3),quaternion=np.array([1.,0.,0.,0.]))
    # This object intentionally exposes no schedule/send/actuation method.
    io=SimpleNamespace(calendar_matches=lambda plans:True)
    try:
        shadow.start()
        shadow.note('v',raw,dict(sample_s=now-.001),[9.80665,0.,0.,0.],now-.002,
            dict(deadline_s=now+.090),row)
        shadow.observe_ack((raw,None,now-.0001),
            dict(vehicle='v',rpc_return_s=now-.0002,mapped_control=[9.8,0.,0.,0.]),row)
        # Delayed legacy planning must not replace the newer paired command.
        shadow.note('v',raw,dict(sample_s=now-.001),[9.80665,0.,0.,0.],now-.002,
            dict(deadline_s=now+.090),row)
        assert shadow.states['v']['actuation_s']==now-.0002
        shadow.tick({'v':row},lambda t:[],io,{'v':now-.020},{'v':now-.0002},1)
    finally:summary=shadow.close()
    assert not summary['worker_error'] and summary['processed']==1
    assert summary['recommendations_applied'] is False and row==original
    record=json.loads((tmp_path/'shadow/decisions.jsonl').read_text().strip())
    assert record['barriers']['v']['available']
    assert record['original_future']['v']['deadline_s']==original['deadline_s']
    if record['proposal']:
        assert record['proposal']['v']['deadline_s']<=original['deadline_s']
        assert record['proposal']['v']['sample_id']=='v:1'
