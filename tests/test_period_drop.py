import time

import numpy as np
import pytest

from config import ThrustCalibration
from controller_architecture_study.framework_native_io import LIBRARY, NativeCommandIO
from controller_architecture_study.framework_releases import ReleaseQueue, replan_unreleased
from controller_architecture_study.framework_runtime import simulate_pipeline
from controller_architecture_study.heterogeneous_load_guard import project_frequencies
from controller_architecture_study.run_reviewed_rate_trial import check_static_baseline
from controller_architecture_study.probe_seven_continuous_timing import next_frontend_work, record_dispatch_observation


def job(index, release, period=.030):
    return dict(vehicle='Drone1', sample_id=f'Drone1:{index}', model='m', priority=0,
        planned_release_s=release, requested_period_s=period, deadline_s=release+period,
        fixed_period_slot=True, serial_vehicle=True)


def test_expiring_one_period_leaves_the_next_release_intact():
    queue = ReleaseQueue(['Drone1'])
    first, second = job(0, 1.), job(1, 1.030)
    queue.publish(first, 1.)
    queue.publish(second, 1.030)
    assert list(queue.expire(1.030)) == [first]
    assert first['status'] == 'period_drop_before_inference'
    assert queue.dispatchable({}, 1, 1.030) == {'Drone1': second}
    candidate = replan_unreleased({**second,'requested_period_s':.020}, 1., 1, 1.001)
    assert candidate['planned_release_s'] == 1.030
    assert candidate['nominal_deadline_s'] == 1.050


def test_dispatch_trace_distinguishes_blocked_from_eligible_waiting():
    row=job(0,1.)
    record_dispatch_observation(row,1.003,'same_vehicle_inflight','command','Drone2')
    record_dispatch_observation(row,1.004,'same_vehicle_inflight','command','Drone2')
    record_dispatch_observation(row,1.009,None,'completed','Drone2')
    record_dispatch_observation(row,1.010,None,'release','Drone1')
    assert row['first_dispatch_decision_s']==1.003
    assert row['first_dispatch_eligible_s']==1.009
    assert [x['reason'] for x in row['dispatch_wait_changes']]==[
        'same_vehicle_inflight','eligible_waiting_for_completed','selected']


def test_period_frontend_orders_work_by_due_time_and_limits_consecutive_cleanup():
    old=job(0,1.)
    sampled={**job(1,1.02), 'vehicle':'Drone2'}
    later_input={**job(2,1.03), 'vehicle':'Drone3'}
    completed={'Drone1':(None,old)}
    inputs={'Drone3':(None,later_input)}
    plans={'Drone2':sampled}
    key=lambda row:(row['priority'],row['planned_release_s'])
    def choose(cleanup_due, previous_completion=False, ready_inputs=inputs, ready_plans=plans):
        return next_frontend_work(completed,ready_inputs,ready_plans,1.04,key,'category',eager_completion=True,
            completion_deadlines={'Drone1':cleanup_due},previous_completion=previous_completion)[:2]
    assert choose(1.08)==('release','Drone2')
    assert choose(1.045)==('completed','Drone1')
    assert choose(1.045,True)==('release','Drone2')
    assert choose(1.045,True,{}, {})==('completed','Drone1')


def test_predictor_overlaps_sampling_but_preserves_serial_inference_and_period_deadline():
    costs = dict(input_s=.001, output_s=.001, sample_io_s=.010, io_s=.004,
        command_s=.003, io_slots=1, eager_completion=True)
    profile = {'m': dict(segment_estimates_s=[.003])}
    first = {**job(0,-.030), 'stage':'io', 'io_remaining_s':.018, 'command_completed_s':-.001}
    second = {**job(1,0.), 'deadline_s':.200}
    result, finish, _ = simulate_pipeline(0.,[second],[first],profile,
        {**costs,'periodic_sampling':True},inflight_limit=1)
    assert result and finish[second['sample_id']] == pytest.approx(.026)
    serial_result, _, _ = simulate_pipeline(0.,[second],[first],profile,costs,inflight_limit=1)
    assert not serial_result
    too_short = {**second, 'requested_period_s':.024}
    result, _, _ = simulate_pipeline(0.,[too_short],[first],profile,
        {**costs,'periodic_sampling':True},inflight_limit=1)
    assert not result


def test_frequency_projection_respects_both_resource_constraints():
    rates = project_frequencies([40,40], [1,1], [12.5,12.5], [50,50],
        [[.02,.01],[.01,.02]], [.80,.85])
    assert rates == pytest.approx([25,30], abs=1e-3)
    assert project_frequencies([40,40],[1,1],[12.5,12.5],[50,50],
        [[.02,.01],[.01,.02]],[.3,.85]) is None


def test_static_gate_rejects_late_commands_and_a_hold_gap_before_the_turn():
    report=dict(wall_origin_s=1., initial_taskset_forecast_passed=True, physical_task_passed=True,
        collisions=0, vehicles={'Drone1':{}}, period_deadline_metrics={'first_published_periods_verified':True})
    traces={'Drone1':[dict(t=.1,sample_s=1.1)]}
    jobs=[dict(vehicle='Drone1',sample_id='one',status='applied',rpc_return_s=1.02,nominal_deadline_s=1.04),
        dict(vehicle='Drone1',sample_id='two',status='applied',rpc_return_s=1.06,nominal_deadline_s=1.08)]
    assert check_static_baseline(report,jobs,traces,.1)['passed']
    jobs[1]['rpc_return_s']=1.09
    assert check_static_baseline(report,jobs,traces,.1)['per_vehicle']['Drone1']['missed_ids']==['two']
    jobs[1]['nominal_deadline_s']=1.095
    traces['Drone1'][0]['sample_s']=1.18
    assert not check_static_baseline(report,jobs,traces,.1)['passed']


@pytest.mark.skipif(not LIBRARY.exists(),reason='native I/O library missing')
def test_native_sampling_is_independent_of_command_slot_and_late_commands_are_rejected():
    command = NativeCommandIO(['Drone1'],ThrustCalibration(),.003,mock=True,mock_delay_s=.150)
    sampler = NativeCommandIO(['Drone1'],ThrustCalibration(),0.,mock=True)
    def wait_result(poll):
        end = time.perf_counter()+2
        while time.perf_counter()<end:
            results = poll()
            if results:
                return results[0]
            time.sleep(.001)
        raise AssertionError('mock native operation did not return')
    try:
        now = time.perf_counter()
        action = np.array([9.81,0.,0.,0.])
        command.send(action,job(0,now,1.))
        sampler.sample(job(1,now))
        _, sample = wait_result(sampler.drain_samples)
        _, completed = wait_result(command.drain)
        assert sample['sample_io_started_s'] < completed['io_finished_s']
        late = job(2,time.perf_counter()-.100)
        command.send(action,late)
        _, rejected = wait_result(command.drain)
        assert rejected['status']=='expired_not_applied'
        assert 'rpc_return_s' not in rejected
    finally:
        sampler.close()
        command.close()
