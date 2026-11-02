from copy import deepcopy
from unittest.mock import patch

from controller_architecture_study.joint_contract_optimizer import optimize
from controller_architecture_study.joint_contract_recovery import prepare_recovery


def row(v,deadline=.080):
    return dict(vehicle=v,sample_id=v+':1',model='m',priority=0,previous_release_s=0.,
        minimum_release_s=.020,planned_release_s=.040,requested_period_s=.040,deadline_s=deadline)


def test_recovery_keeps_expired_work_and_reports_past_miss():
    future={'new':row('new')}
    pending=[{**row('old',-.001),'stage':'compute','planned_release_s':-.010}]
    frozen=deepcopy((future,pending))
    options={'new':[dict(id='a',model='m',release_s=.040,value=1.)]}
    profile={'m':{'segment_estimates_s':[.002]}};costs=dict(input_s=.001,output_s=.001)
    strict,info=optimize(0.,future,pending,options,profile,costs,inflight_limit=2)
    assert strict is None and not info['future_feasible']
    with patch('controller_architecture_study.joint_contract_optimizer.time.perf_counter',return_value=0.):
        proposal,info=optimize(0.,future,pending,options,profile,costs,inflight_limit=2,recover_after_miss=True)
    assert info['accepted'] and info['future_feasible'] and not info['strict_timing_passed']
    assert info['missed_work_still_charged']==['old:1']
    assert 'old:1' in info['checks'][0]['finish']
    assert info['checks'][0]['finish']['old:1']>=.004  # blocking + FULL old compute
    assert (future,pending)==frozen and not info['historical_deadlines_repaired']


def test_large_expired_job_still_blocks_new_deadline_in_recovery():
    future={'new':row('new',.034)}
    pending=[{**row('old',-.001),'model':'blocker','stage':'compute','planned_release_s':-.010}]
    options={'new':[dict(id='a',model='m',release_s=.020,value=1.)]}
    profile={'m':{'segment_estimates_s':[.008]},'blocker':{'segment_estimates_s':[.015,.015]}}
    with patch('controller_architecture_study.joint_contract_optimizer.time.perf_counter',return_value=0.):
        proposal,info=optimize(0.,future,pending,options,profile,dict(input_s=.001,output_s=.001),
            inflight_limit=2,recover_after_miss=True)
    assert proposal is None and not info['checks'][0]['passed']


def test_recovery_never_ignores_a_deadline_that_has_not_expired():
    future={'new':row('new')}
    pending=[{**row('old',.010),'stage':'compute','planned_release_s':-.010}]
    options={'new':[dict(id='a',model='m',release_s=.040,value=1.)]}
    with patch('controller_architecture_study.joint_contract_optimizer.time.perf_counter',return_value=0.):
        proposal,info=optimize(0.,future,pending,options,{'m':{'segment_estimates_s':[.008]}},
            dict(input_s=.001,output_s=.001),inflight_limit=2,recover_after_miss=True)
    assert proposal is None and not info['accepted']


def test_expired_future_cannot_be_repaired_by_optimizer_flag_alone():
    future={'v':row('v',-.001)}
    proposal,info=optimize(0.,future,[],{'v':[]},{},{},inflight_limit=1,recover_after_miss=True)
    assert proposal is None and info['expired_ids']==['v:1']


def test_new_observation_creates_a_separate_unpublished_obligation():
    future={'v':{**row('v',.110),'planned_release_s':.160}}
    frozen=deepcopy(future)
    certs={'v':dict(available=True,sample_s=.119,deadline_s=.190,is_safety_certificate=False)}
    proposal,info=prepare_recovery(.120,future,[],certs)
    assert proposal['v']['sample_id']==future['v']['sample_id']
    assert proposal['v']['deadline_s']==.190 and proposal['v']['superseded_deadline_s']==.110
    assert 'obligation_version' in proposal['v'] and future==frozen
    assert info['previous_misses'][0]['deadline_s']==.110
    assert not info['historical_deadlines_repaired'] and not info['installed']


def test_unpublished_renewal_requires_post_expiry_paired_state_and_no_unknown_command():
    future={'v':{**row('v',.110),'planned_release_s':.160}}
    good=dict(available=True,sample_s=.119,deadline_s=.190)
    variants=[([],{'v':dict(available=False)}),([],{'v':{**good,'sample_s':.109}}),
        ([],{'v':{**good,'sample_s':.121}}),([],{'v':{**good,'deadline_s':.125}}),
        ([{**row('v',.110),'stage':'io'}],{'v':good})]
    for pending,certs in variants:
        proposal,info=prepare_recovery(.120,future,pending,certs)
        assert proposal['v']['deadline_s']==.110 and info['deferred'] and not info['renewals']


def test_renewal_does_not_touch_near_publication_or_relax_unexpired_commitment():
    certs={'v':dict(available=True,sample_s=.119,deadline_s=.190)}
    near={'v':{**row('v',.110),'planned_release_s':.125}}
    proposal,info=prepare_recovery(.120,near,[],certs)
    assert proposal['v']['deadline_s']==.110 and info['deferred']
    unexpired={'v':{**row('v',.150),'planned_release_s':.160}}
    proposal,info=prepare_recovery(.120,unexpired,[],certs)
    assert proposal['v']['deadline_s']==.150 and not info['renewals']


def test_new_actual_control_ends_old_hold_then_gets_its_own_obligation():
    future={'v':{**row('v',.150),'planned_release_s':.160,'obligation_actuation_s':.080}}
    certs={'v':dict(available=True,sample_s=.119,actuation_s=.118,deadline_s=.190)}
    proposal,info=prepare_recovery(.120,future,[],certs)
    assert proposal['v']['deadline_s']==.190 and proposal['v']['obligation_actuation_s']==.118
    assert info['renewals'][0]['new_control_epoch'] and info['renewals'][0]['previous_obligation_fulfilled']
    assert not info['previous_misses']
    # Same held control cannot postpone the unexpired obligation.
    certs['v']['actuation_s']=.080
    same,info=prepare_recovery(.120,future,[],certs)
    assert same['v']['deadline_s']==.150 and not info['renewals']


def test_late_new_control_does_not_erase_the_prior_hold_violation():
    future={'v':{**row('v',.110),'planned_release_s':.160,'obligation_actuation_s':.080}}
    certs={'v':dict(available=True,sample_s=.119,actuation_s=.118,deadline_s=.190)}
    proposal,info=prepare_recovery(.120,future,[],certs)
    assert proposal['v']['deadline_s']==.190 and info['previous_misses'][0]['deadline_s']==.110
    assert not info['renewals'][0]['previous_obligation_fulfilled']
