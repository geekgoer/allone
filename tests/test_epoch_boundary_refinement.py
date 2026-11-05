import json

from controller_architecture_study.probe_recovery_boundary import ACTIONS,next_hardness
from controller_architecture_study.refine_recovery_boundary import anchors,DISCOVERY,CONFIRMATION,CONFIRMATION_LEVELS
from controller_architecture_study.replay_obligation_epochs import replay


def test_refinement_cannot_adapt_to_confirmation_labels(tmp_path):
    for split in ('discovery','confirmation'):
        for level in (.4,.6):
            for k,(model,period) in enumerate(ACTIONS):
                path=tmp_path/'trials'/f'{split}_{level}_{k}';path.mkdir(parents=True)
                row=dict(route_seed=3400,hardness=level,split=split,model=model,cadence_s=period,
                    label=dict(valid=True,qualified=level==.4 if split=='discovery' else False))
                (path/'report.json').write_text(json.dumps(row))
    observed=anchors(tmp_path,3400)
    assert observed=={.4:(True,)*4,.6:(False,)*4}
    queried=[]
    for _ in range(4):
        x=next_hardness(observed);assert .4<x<.6 and x not in queried
        queried.append(x);observed[x]=(x<.49,)*4
    assert not set(DISCOVERY)&set(CONFIRMATION)
    assert not set(CONFIRMATION)&{3500,3501}  # Original confirmation stays untouched.
    assert (len(DISCOVERY)*4+len(CONFIRMATION)*len(CONFIRMATION_LEVELS))*len(ACTIONS)==64


def test_offline_epoch_replay_records_history_without_mutating_published_contracts(tmp_path):
    source=tmp_path/'source';(source/'source').mkdir(parents=True)
    (source/'source/joint_contract_recovery.py').write_text(
        'def prepare_recovery(now,future,pending,certificates):\n    return future, {"renewals": []}\n')
    row=dict(vehicle='v',sample_id='v:2',model='m',planned_release_s=.160,deadline_s=.110,
        nominal_deadline_s=.180,obligation_actuation_s=.080)
    d=dict(original_future={'v':row},pending=[{**row,'sample_id':'v:1','stage':'compute'}],
        worker_begin_s=.120,barriers={'v':dict(available=True,actuation_s=.118,sample_s=.119,deadline_s=.190)},generation=1)
    log=source/'a_planner/decisions.jsonl';log.parent.mkdir();log.write_text(json.dumps(d)+'\n')
    replay(source,tmp_path/'out')
    report=json.loads((tmp_path/'out/report.json').read_text())
    assert report['counts']['old_expired_future']==1 and report['counts']['new_expired_future']==0
    assert report['rows'][0]['current']['previous_misses']
    assert report['actual_controls_applied']==0 and json.loads(log.read_text())==d
