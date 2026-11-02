from copy import deepcopy
from types import SimpleNamespace

import pytest

from controller_architecture_study.framework_releases import ReleaseQueue, periodic_successor,post_actuation_observation,commit_dynamic_release
from controller_architecture_study.probe_seven_continuous_timing import next_feedback_release
from controller_architecture_study.framework_runtime import simulate_pipeline


def job(vehicle='a', release=0., number=0):
    return dict(vehicle=vehicle, sample_id=f'{vehicle}:{number}', model='m', priority=0,
        planned_release_s=release, deadline_s=release+.020, nominal_deadline_s=release+.020,
        requested_period_s=.020, slot=number, status='planned', serial_vehicle=True)


def test_calendar_successor_preserves_the_held_control_deadline_epoch():
    from controller_architecture_study.framework_releases import calendar_successor
    previous={**job(release=.040),'obligation_actuation_s':.025,'obligation_sample_s':.030}
    following=calendar_successor(previous,1,dynamic=True)
    assert following['planned_release_s']==pytest.approx(.060)
    assert following['deadline_s']==previous['deadline_s']
    assert following['obligation_actuation_s']==.025 and following['obligation_sample_s']==.030


def test_busy_pipeline_cannot_postpone_or_erase_periodic_arrivals():
    queue=ReleaseQueue(['a','b','c','d']);row=job();arrivals=[]
    # Three occupied pipeline slots never prevent publishing the fourth loop.
    for now in (0.,.020,.040,.060):
        queue.publish(row,now);arrivals.append(row)
        assert queue.dispatchable({'b':{},'c':{},'d':{}},3)=={}
        row=periodic_successor(row,len(arrivals))
    assert [r['job_released_s'] for r in arrivals]==[0.,.020,.040,.060]
    assert len(queue.rows())==4
    assert queue.take('a',.065)['dispatch_s']==.065
    assert arrivals[0]['job_released_s']==0. and arrivals[0]['deadline_s']==.020


def test_late_dispatch_and_late_timer_do_not_reset_future_calendar():
    queue=ReleaseQueue(['a']);row=job(release=.020)
    queue.publish(row,.025)
    dispatched=queue.take('a',.035)
    following=periodic_successor(dispatched,1)
    assert following['planned_release_s']==.040
    assert dispatched['job_released_s']==.025
    assert dispatched['planned_release_s']==.020


def test_dynamic_feedback_arriving_late_does_not_silently_shift_committed_clock():
    previous=dict(planned_release_s=.020,job_released_s=.031)
    assert next_feedback_release(previous,.038,.020,.080,.010,calendar=True)==.040
    assert next_feedback_release(previous,.050,.020,.080,.010,calendar=True)==.040


def test_released_mode_deadline_and_release_are_frozen():
    for key,value in [('model','other'),('deadline_s',.050),('planned_release_s',.010)]:
        queue=ReleaseQueue(['a']);row=job();queue.publish(row,0.)
        row[key]=value
        with pytest.raises(RuntimeError,match='commitment'):queue.take('a',.020)


def test_same_vehicle_cannot_dispatch_overlapping_io_but_other_vehicle_can():
    queue=ReleaseQueue(['a','b']);queue.publish(job(),0.);queue.publish(job('b'),0.)
    assert set(queue.dispatchable({'a':{}},3))=={'b'}
    queue.cancel('a','collision_cancelled_while_queued')
    assert all(r['vehicle']=='b' for r in queue.rows())


def test_forecast_charges_queued_same_vehicle_work_without_changing_its_release():
    profile={'m':dict(segment_estimates_s=[.005])}
    costs=dict(input_s=.001,output_s=.001,io_s=.004,command_s=.002,
        sample_io_s=.001,handoff_s=0.,io_slots=3)
    a=job();b=job(release=.001,number=1)
    a['deadline_s']=b['deadline_s']=.1
    before=deepcopy([a,b])
    good,finish,_=simulate_pipeline(0.,[a,b],[],profile,costs,inflight_limit=3)
    assert good and finish['a:1']-finish['a:0']>=.011-1e-9
    assert [a,b]==before


def test_ack_does_not_pair_new_control_with_a_pre_actuation_state():
    old=(SimpleNamespace(timestamp_ns=10),None,.010)
    sensor=(SimpleNamespace(timestamp_ns=30),None,.030)
    feedback=(SimpleNamespace(timestamp_ns=25),None,.025)
    assert post_actuation_observation([old],.020) is None
    assert post_actuation_observation([old,sensor,feedback],.020) is sensor


def test_dynamic_release_missed_during_planning_is_explicit_not_backdated():
    previous=job(release=.020);row=job(release=.040,number=1)
    deadline=row['deadline_s'];mode=row['model']
    commit_dynamic_release(row,previous,.050)
    assert row['target_release_s']==.040 and row['planned_release_s']==.050
    assert abs(row['planning_lateness_s']-.010)<1e-9
    assert abs(row['selected_release_interval_s']-.030)<1e-9
    assert row['deadline_s']==deadline and row['model']==mode
    queue=ReleaseQueue(['a']);queue.publish(row,.051)
    assert queue.take('a',.070)['planned_release_s']==.050


def test_dynamic_ack_stall_keeps_all_arrivals_and_does_not_renew_held_control_budget():
    from controller_architecture_study.framework_releases import calendar_successor
    queue=ReleaseQueue(['a']);row=job();row['deadline_s']=.070
    for i in range(4):
        queue.publish(row,i*.020)
        row=calendar_successor(row,i+1,dynamic=True)
    assert [r['planned_release_s'] for r in queue.rows()]==[0.,.020,.040,.060]
    assert all(r['deadline_s']==.070 for r in queue.rows())
    assert row['planned_release_s']==.080 and row['feedback_pending']


def test_feedback_changes_only_next_unreleased_job_after_multiple_outstanding_arrivals():
    from controller_architecture_study.framework_releases import replan_unreleased
    queue=ReleaseQueue(['a'])
    for i in range(4):queue.publish(job(release=i*.020,number=i),i*.020)
    before=deepcopy(queue.rows())
    candidate=job(release=.020,number=1);candidate['deadline_s']=.150
    next_row=replan_unreleased(candidate,.060,4,.075,.010)
    assert next_row['planned_release_s']==.080 and next_row['sample_id']=='a:4'
    assert next_row['deadline_s']==.150 and queue.rows()==before


def test_retained_future_configuration_uses_same_new_physical_obligation():
    from controller_architecture_study.framework_releases import replan_unreleased,retained_unreleased
    old=job();old.update(model='old',deadline_s=.070,requested_period_s=.040)
    candidate=job();candidate.update(model='new',deadline_s=.150)
    candidate=replan_unreleased(candidate,.060,4,.075,.010)
    fallback=retained_unreleased(candidate,old,.060,4,.075,.010)
    assert (fallback['sample_id'],fallback['deadline_s'])==(candidate['sample_id'],candidate['deadline_s'])
    assert fallback['model']=='old' and fallback['requested_period_s']==.040
    assert old['deadline_s']==.070
