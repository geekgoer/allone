from copy import deepcopy

from controller_architecture_study.framework_releases import taped_release,ReleaseQueue


def test_replay_offsets_all_clocks_once_without_replaying_control_outputs():
    row=dict(vehicle='a',sample_id='a:0',model='m',priority=0,planned_release_s=.020,
             minimum_release_s=.020,deadline_s=.080,nominal_deadline_s=.080,
             dispatch_not_before_s=.040,delivery_not_before_s=.060)
    tape={'a':[row]};before=deepcopy(tape)
    actual=taped_release(tape,'a',0,100.)
    assert actual['planned_release_s']==100.020 and actual['deadline_s']==100.080
    assert actual['dispatch_not_before_s']==100.040 and actual['delivery_not_before_s']==100.060
    assert tape==before and 'control' not in actual
    assert taped_release(tape,'a',1,100.) is None


def test_wait_replay_retains_arrival_but_defers_service_only_for_the_target_loop():
    q=ReleaseQueue(['a','b'])
    for v,wait in [('a',.040),('b',.020)]:
        q.publish(dict(vehicle=v,sample_id=v+':0',model='m',planned_release_s=.020,
            deadline_s=.080,dispatch_not_before_s=wait),.020)
    assert set(q.dispatchable({},10,.021))=={'b'}
    assert set(q.dispatchable({},10,.040))=={'a','b'}
    assert q.take('a',.041)['job_released_s']==.020
