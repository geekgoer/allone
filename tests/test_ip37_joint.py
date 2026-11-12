"""Live-route B semantics without a simulator, model files or board connection."""
from concurrent.futures import Future
from dataclasses import replace
from pathlib import Path
import tempfile
import time
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from controller_architecture_study.ip37_joint_runtime import JointRuntime
from controller_architecture_study.bottleneck_abc import abc_repair
from controller_architecture_study.bottleneck_joint import Job
from controller_architecture_study import fly_ip37_joint as flight


def runtime(names=('a',), *, origin=10., cost=2., adaptive=True):
    return JointRuntime(names, origin, 50., (20., 30., 40., 50., 60., 80.), 11, cost,
                        prep_ms=.1, post_ms=.1, decision_ms=1., adaptive=adaptive)


def paired(rt, name, t, deadline):
    return rt.observe(name, t, deadline, {'budget_s': deadline-t}, rt.loops[name]['control_version'])


class RuntimeTests(unittest.TestCase):
    def test_empirical_period_sets_do_not_assume_monotonic_quality(self):
        rt = JointRuntime(('a', 'b'), 10., 50., (20., 50., 80., 120.), 11, 2.,
                          qualified_periods={'a': [50.], 'b': [50., 120.]})
        paired(rt, 'a', 9.999, 10.1)
        paired(rt, 'b', 9.999, 10.1)
        _, scenario = rt.snapshot(9.999)
        self.assertEqual([o.period for o in scenario.loops[0].options], [50.])
        self.assertEqual([o.period for o in scenario.loops[1].options], [50., 120.])

    def test_empirical_sets_must_include_starting_period(self):
        with self.assertRaises(ValueError):
            JointRuntime(('a',), 10., 50., (50., 120.), 11, 2., qualified_periods={'a': [120.]})

    def test_late_observer_does_not_replace_newer_state(self):
        rt = runtime()
        self.assertTrue(rt.observe('a', 10.02, 10.1, {}, 0))
        self.assertFalse(rt.observe('a', 10.01, 10.1, {}, 0))
        self.assertEqual(rt.loops['a']['observation']['sample_s'], 10.02)

    def test_releases_do_not_wait_for_io_and_busy_releases_are_dropped(self):
        rt = runtime()
        first, = rt.release_due(10.)
        rt.release_due(10.05)
        self.assertEqual(len(rt.rows), 2)
        self.assertEqual(rt.rows[1]['status'], 'dropped_busy_io')
        self.assertAlmostEqual(rt.rows[1]['release_s'], 10.05)
        self.assertTrue(first['expired'])
        rt.finish(first)
        third, = rt.release_due(10.10001)
        self.assertAlmostEqual(third['release_s'], 10.1)
        self.assertEqual(third['cycle'], 2)

    def test_calendar_reports_missed_slots_without_burst_sampling(self):
        rt = runtime()
        jobs = rt.release_due(10.12)
        self.assertEqual(len(jobs), 1)
        self.assertEqual([r['status'] for r in rt.rows[:2]], ['dropped_release_late']*2)
        self.assertAlmostEqual(jobs[0]['release_s'], 10.1)

    def test_end_flush_records_last_inside_release_and_excludes_endpoint(self):
        rt = runtime(('a', 'b', 'c'))
        for row in rt.release_due(10.12, stop_before=10.15):
            rt.finish(row)
        tail, = rt.release_due(10.151, stop_before=10.15)
        self.assertEqual(tail['vehicle'], 'c')
        self.assertAlmostEqual(tail['release_s'], 10.+2*.05/3+.1)
        self.assertEqual(len(rt.rows), 9)
        self.assertTrue(all(r['release_s'] < 10.15-1e-9 for r in rt.rows))

    def test_post_refresh_cost_does_not_change_operational_command_reserve(self):
        rt = runtime()
        reserve = rt.command_reserve_ms
        rt.update_cpu_costs(post_ms=99.)
        self.assertEqual(rt.post_ms, 99.)
        self.assertEqual(rt.command_reserve_ms, reserve)

    def test_edf_cannot_preempt_an_active_board_request(self):
        rt = runtime(('a', 'b', 'c'))
        jobs = rt.release_due(10.04)
        with patch('controller_architecture_study.ip37_joint_runtime.time.perf_counter', return_value=10.04):
            for row in jobs:
                rt.enqueue(row, np.zeros(237))
        # Give c the earliest physical deadline, while a is first in FIFO order.
        jobs[2]['safety_deadline_s'] = 10.041
        first = rt.take(10.04)
        self.assertEqual(first['vehicle'], 'c')
        self.assertIsNone(rt.take(10.0405))
        rt.board_done(first, 10.041, action=np.zeros(4), timing={})
        self.assertEqual(rt.take(10.041)['vehicle'], 'a')

    def test_cancelled_waiting_request_is_never_sent(self):
        rt = runtime()
        row, = rt.release_due(10.)
        with patch('controller_architecture_study.ip37_joint_runtime.time.perf_counter', return_value=10.):
            future = rt.enqueue(row, np.zeros(237))
        future.cancel()
        self.assertIsNone(rt.take(10.001))

    def test_cancelled_active_request_still_blocks_until_board_returns(self):
        rt = runtime()
        row, = rt.release_due(10.)
        with patch('controller_architecture_study.ip37_joint_runtime.time.perf_counter', return_value=10.):
            future = rt.enqueue(row, np.zeros(237))
        rt.take(10.)
        future.cancel()
        self.assertIs(rt.active, row)
        rt.board_done(row, 10.001, action=np.zeros(4))
        self.assertTrue(future.cancelled())
        self.assertIsNone(rt.active)

    def test_snapshot_keeps_old_work_and_reserves_active_remainder_once(self):
        rt = runtime(cost=10.)
        row, = rt.release_due(10.)
        with patch('controller_architecture_study.ip37_joint_runtime.time.perf_counter', return_value=10.):
            rt.enqueue(row, np.zeros(237))
        rt.take(10.)
        paired(rt, 'a', 10., 10.1)
        _, scenario = rt.snapshot(10.003)
        self.assertAlmostEqual(scenario.platform.initial_pl_block, 7.)
        self.assertEqual(len(scenario.carry), 1)
        self.assertEqual(scenario.carry[0].segments, (0.,))
        self.assertAlmostEqual(scenario.carry[0].release, 7.)

    def test_active_execution_overrun_is_not_assumed_finished(self):
        rt = runtime(cost=2.)
        row, = rt.release_due(10.)
        with patch('controller_architecture_study.ip37_joint_runtime.time.perf_counter', return_value=10.):
            rt.enqueue(row, np.zeros(237))
        rt.take(10.)
        paired(rt, 'a', 10., 10.1)
        self.assertIsNone(rt.snapshot(10.003))
        self.assertEqual(rt.skips['active_cost_envelope_exceeded'], 1)

    def test_new_control_invalidates_old_budget_and_rejects_old_version(self):
        rt = runtime()
        row, = rt.release_due(10.)
        rt.finish(row)
        paired(rt, 'a', 10., 10.1)
        rt.command_changed('a')
        self.assertFalse(rt.observe('a', 10.01, 10.1, {}, 0))
        self.assertIsNone(rt.snapshot(10.01))
        paired(rt, 'a', 10.011, 10.1)
        self.assertIsNotNone(rt.snapshot(10.012))

    def test_repair_advances_future_release_without_changing_released_deadline(self):
        rt = runtime()
        old, = rt.release_due(10.)
        old_deadline = old['deadline_s']
        rt.finish(old)
        paired(rt, 'a', 10.001, 10.043)
        revision, scenario = rt.snapshot(10.005)
        result = abc_repair(scenario)
        self.assertTrue(result['accepted'])
        status, updates = rt.install(revision, scenario, result, 10.005, 10.0055)
        self.assertEqual(status, 'installed')
        self.assertEqual(updates[0]['period_ms'], 40.)
        self.assertEqual(old['deadline_s'], old_deadline)
        self.assertAlmostEqual(rt.loops['a']['next'], 10.04)

    def test_install_rejects_a_plan_after_any_new_release(self):
        rt = runtime()
        paired(rt, 'a', 9.999, 10.1)
        revision, scenario = rt.snapshot(9.999)
        result = abc_repair(scenario)
        rt.release_due(10.)
        status, _ = rt.install(revision, scenario, result, 9.999, 9.9995)
        self.assertEqual(status, 'context_changed')

    def test_elapsed_decision_time_is_charged_before_commit(self):
        rt = runtime()
        paired(rt, 'a', 9.99, 10.1)
        revision, scenario = rt.snapshot(9.99)
        result = abc_repair(scenario)
        status, _ = rt.install(revision, scenario, result, 9.99, 9.992)
        self.assertEqual(status, 'decision_budget_exceeded')

    def test_expired_physical_deadline_is_not_reset_at_snapshot(self):
        rt = runtime()
        old, = rt.release_due(10.)
        rt.finish(old)
        paired(rt, 'a', 10., 10.003)
        _, scenario = rt.snapshot(10.004)
        self.assertLess(scenario.loops[0].deadline, 0.)
        self.assertFalse(abc_repair(scenario)['accepted'])

    def test_advancing_release_must_also_fit_the_vehicle_io_pipeline(self):
        rt = runtime()
        old, = rt.release_due(10.)
        paired(rt, 'a', 10., 10.04)
        revision, scenario = rt.snapshot(10.001)
        scenario = replace(scenario, carry=(Job('old', 'a', 0., 49., (35.,), '11'),))
        result = abc_repair(scenario)
        self.assertTrue(result['accepted'])  # shared-resource model alone allows the overlap
        status, _ = rt.install(revision, scenario, result, 10.001, 10.0015)
        self.assertEqual(status, 'per_vehicle_pipeline_overlap')
        self.assertAlmostEqual(old['deadline_s'], 10.05)

    def test_other_loop_committed_work_cannot_be_erased_to_admit_repair(self):
        rt = runtime(('a', 'b'), cost=8.)
        jobs = rt.release_due(10.03)
        for row in jobs:
            row['safety_deadline_s'] = 10.031
        paired(rt, 'a', 10.03, 10.07)
        paired(rt, 'b', 10.03, 10.07)
        revision, scenario = rt.snapshot(10.03)
        result = abc_repair(scenario)
        self.assertFalse(result['accepted'])
        status, _ = rt.install(revision, scenario, result, 10.03, 10.0305)
        self.assertEqual(status, 'no_feasible_repair')
        self.assertEqual(len(scenario.carry), 2)


class PilotTests(unittest.TestCase):
    def test_independent_observer_discards_state_when_command_changes_during_read(self):
        rt = runtime(origin=0.)
        stop = threading.Event()
        stats = dict(accepted=0, stale_rejected=0, pending_skip=0, rpc_ms=[], total_ms=[])
        p = SimpleNamespace(args=SimpleNamespace(airsim_host='127.0.0.1', observer_ms=50., maximum_update_ms=160.),
                            name='a', runtime=rt, origin=0., held=np.array([9.8, 0., 0., 0.]),
                            version=0, actuation_s=0., command_pending=False)
        def read():
            p.version = rt.command_changed('a')
            return object()
        p.analyzer = SimpleNamespace(analyze=lambda raw, held: {})
        fake_client = SimpleNamespace(ping=lambda: None)
        fake_api = SimpleNamespace(MultirotorClient=lambda **kwargs: fake_client)
        drone = SimpleNamespace(get_raw_state=read)
        with patch('airsim_interface.require_airsim', return_value=fake_api), \
             patch('airsim_interface.AirSimDrone', return_value=drone), \
             patch.object(flight, 'safe_deadline', return_value={'deadline_s': 100.}), \
             patch.object(stop, 'wait', side_effect=lambda t: stop.set()):
            flight.observe_independently(p, stop, stats)
        self.assertEqual(stats['stale_rejected'], 1)
        self.assertEqual(stats['accepted'], 0)
        self.assertIsNone(rt.loops['a']['observation'])

    def pilot(self, action):
        p = object.__new__(flight.Pilot)
        p.name, p.args = 'a', SimpleNamespace(post_ms=2.)
        p.drone = SimpleNamespace(get_raw_state=lambda: SimpleNamespace())
        p.client = SimpleNamespace(ping=lambda: None)
        p.held, p.version, p.calibration = np.array([9.8, 0., 0., 0.]), 0, None
        p.error, p.contact = None, None
        p.collision = lambda *a: None
        p.publish = lambda raw, sample: ([], {'budget_s': .1}, {'deadline_s': 11.}, .1)
        future = Future()
        future.set_result(np.asarray(action))
        p.runtime = SimpleNamespace(enqueue=lambda *a: future, command_changed=lambda n: 1,
                                    post_ms=2., command_reserve_ms=2., update_cpu_costs=lambda **kw: None,
                                    finish=lambda r: None, retire=lambda n: None)
        return p

    def test_expired_output_is_dropped_without_a_control_call(self):
        p = self.pilot([9.8, 0., 0., 0.])
        row = dict(deadline_s=9.99)
        with patch.object(flight, 'time', SimpleNamespace(perf_counter=lambda: 10.)), \
             patch('control_adapter.send_control') as send:
            p.run_job(row)
        self.assertEqual(row['status'], 'dropped_before_sample')
        send.assert_not_called()

    def test_real_board_output_reaches_command_and_invalidates_control_version(self):
        action = np.array([9.8, .1, -.1, .02])
        p = self.pilot(action)
        row = dict(deadline_s=10.05)
        with patch.object(flight, 'time', SimpleNamespace(perf_counter=lambda: 10.)), \
             patch.object(flight, 'encode', return_value=np.zeros(237)), \
             patch('control_adapter.send_control') as send:
            p.run_job(row)
        self.assertEqual(row['status'], 'delivered')
        np.testing.assert_allclose(send.call_args.args[1], action)
        self.assertEqual(p.version, 1)
        self.assertEqual(p.actuation_s, 10.)

    def test_board_range_rejection_preserves_the_drone_exit_reason(self):
        p = self.pilot([9.8, 0., 0., 0.])
        failed = Future()
        failed.set_exception(RuntimeError('IP37 status=3, saturated channels=1'))
        p.runtime.enqueue = lambda *a: failed
        row = dict(deadline_s=10.05)
        with patch.object(flight, 'time', SimpleNamespace(perf_counter=lambda: 10.)), \
             patch.object(flight, 'encode', return_value=np.zeros(237)), \
             patch('control_adapter.send_control') as send:
            p.run_job(row)
        self.assertEqual(row['status'], 'error')
        self.assertIn('IP37 status=3', p.error)
        self.assertEqual(row['error'], p.error)
        send.assert_not_called()

    def test_deferred_refresh_releases_worker_and_pairs_at_next_sample(self):
        p = self.pilot([9.8, .1, 0., 0.])
        p.args.refresh_at_release = True
        observed_versions = []
        original_publish = p.publish
        def publish(raw, sample):
            observed_versions.append(p.version)
            return original_publish(raw, sample)
        p.publish = publish
        rows = [dict(deadline_s=10.05), dict(deadline_s=10.1)]
        with patch.object(flight, 'time', SimpleNamespace(perf_counter=lambda: 10.)), \
             patch.object(flight, 'encode', return_value=np.zeros(237)), \
             patch('control_adapter.send_control'), \
             patch.object(p.drone, 'get_raw_state', wraps=p.drone.get_raw_state) as sample:
            for row in rows:
                p.run_job(row)
        self.assertEqual(sample.call_count, 2)
        self.assertEqual(observed_versions, [0, 1])
        self.assertTrue(all(r['status'] == 'delivered' for r in rows))
        self.assertTrue(all(r['paired_refresh'] == 'next_scheduled_sample' for r in rows))
        self.assertTrue(all('paired_sample_start_s' not in r for r in rows))

    def test_slack_guard_does_not_start_a_late_duplicate_read(self):
        p = self.pilot([9.8, 0., 0., 0.])
        p.args.refresh_when_slack = True
        p.args.prep_ms, p.args.decision_ms = 3., 3.
        p.runtime.cv = threading.Condition()
        p.runtime.loops = {'a': {'next': 10.002}}
        row = dict(deadline_s=10.05)
        with patch.object(flight, 'time', SimpleNamespace(perf_counter=lambda: 10.)), \
             patch.object(flight, 'encode', return_value=np.zeros(237)), \
             patch('control_adapter.send_control'):
            p.run_job(row)
        self.assertEqual(row['status'], 'delivered')
        self.assertEqual(row['paired_refresh'], 'next_scheduled_sample')
        self.assertEqual(p.version, 1)

    def test_long_previous_refresh_does_not_block_a_timely_command(self):
        p = self.pilot([9.8, 0., 0., 0.])
        p.runtime.post_ms = 99.
        row = dict(deadline_s=10.03)
        with patch.object(flight, 'time', SimpleNamespace(perf_counter=lambda: 10.)), \
             patch.object(flight, 'encode', return_value=np.zeros(237)), \
             patch('control_adapter.send_control') as send:
            p.run_job(row)
        self.assertEqual(row['status'], 'delivered')
        send.assert_called_once()

    def test_teacher_diagnostic_bypasses_board_and_records_its_own_output(self):
        action = np.array([9.8, .1, -.1, .02])
        p = self.pilot(action)
        p.teacher = SimpleNamespace(query=lambda *a: (action, {'solve_ms': 1.}))
        def forbidden_board(*a):
            raise AssertionError('teacher diagnostic must not submit a board request')
        p.runtime.enqueue = forbidden_board
        row = dict(deadline_s=10.02)
        with patch.object(flight, 'time', SimpleNamespace(perf_counter=lambda: 10.)), \
             patch.object(flight, 'encode', return_value=np.zeros(237)), \
             patch('control_adapter.send_control') as send:
            p.run_job(row)
        self.assertEqual(row['status'], 'delivered')
        self.assertNotIn('board_output', row)
        np.testing.assert_allclose(row['teacher_output'], action)
        np.testing.assert_allclose(send.call_args.args[1], action)

    def test_late_teacher_result_is_not_applied(self):
        p = self.pilot([9.8, 0., 0., 0.])
        clock = [10.]
        def slow_teacher(*a):
            clock[0] = 10.021
            return np.array([9.8, 0., 0., 0.]), {'solve_ms': 21.}
        p.teacher = SimpleNamespace(query=slow_teacher)
        row = dict(deadline_s=10.02)
        with patch.object(flight, 'time', SimpleNamespace(perf_counter=lambda: clock[0])), \
             patch.object(flight, 'encode', return_value=np.zeros(237)), \
             patch('control_adapter.send_control') as send:
            p.run_job(row)
        self.assertEqual(row['status'], 'dropped_before_command')
        send.assert_not_called()


class FrozenCaseTests(unittest.TestCase):
    def test_diagnostic_variant_cannot_silently_change_a_board_run(self):
        args = SimpleNamespace(reference_teacher=False, teacher_variant='qp_ff')
        with patch.object(flight, 'load_flight_layouts', return_value=([], [], None)), \
             patch.object(flight, 'make_client') as connect:
            with self.assertRaisesRegex(ValueError, 'requires reference-teacher'):
                flight.run(args)
        connect.assert_not_called()

    def test_loads_selected_frozen_layouts_and_uses_net_width_contract(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            import json
            path = Path(directory)/'case.json'
            case = dict(version='ip37_natural_turn_gates_v1', duration_s=26.,
                        gate_center_height_m=4., gate_center_width_m=2.25,
                        layouts={'Drone1': {'gates': list(range(13))}},
                        sources={'Drone1': {'environment': 49, 'route': 2}})
            path.write_text(json.dumps(case), encoding='utf-8')
            args = SimpleNamespace(case=path, duration=None, gate_width=None, vehicles=['Drone1'])
            with patch.object(flight, 'load_layout') as legacy:
                layouts, sources, loaded = flight.load_flight_layouts(args)
            legacy.assert_not_called()
            self.assertEqual(args.gate_width, 2.25)
            self.assertEqual(args.duration, 26.)
            self.assertEqual(len(layouts[0][0]['gates']), 13)
            self.assertEqual(sources[0]['route'], 2)
            self.assertEqual(loaded, case)

    def test_frozen_case_rejects_shortened_mission(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            import json
            path = Path(directory)/'case.json'
            path.write_text(json.dumps(dict(version='ip37_natural_turn_gates_v1', duration_s=26.,
                                            gate_center_height_m=4.)), encoding='utf-8')
            args = SimpleNamespace(case=path, duration=10., gate_width=None, vehicles=['Drone1'])
            with self.assertRaisesRegex(ValueError, 'shortened'):
                flight.load_flight_layouts(args)


class GatePreparationTests(unittest.TestCase):
    def test_removes_our_old_bar_and_verifies_disappearance(self):
        from unittest.mock import Mock
        name = 'FalconJointGate_Drone3_1_top'
        client = SimpleNamespace(simListSceneObjects=Mock(side_effect=[[name], []]),
                                 simDestroyObject=Mock(return_value=True))
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            report = flight.prepare_gate_scene(client, Path(directory))
            self.assertTrue((Path(directory)/'gate_preflight.json').exists())
        client.simDestroyObject.assert_called_once_with(name)
        self.assertEqual(report['remaining'], [])

    def test_foreign_gate_blocks_without_deleting_any_object(self):
        from unittest.mock import Mock
        names = ['FalconJointGate_Drone3_1_top', 'FalconValueGate_Drone1_1_left']
        client = SimpleNamespace(simListSceneObjects=Mock(return_value=names), simDestroyObject=Mock())
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            with self.assertRaisesRegex(ValueError, 'FalconValueGate'):
                flight.prepare_gate_scene(client, Path(directory))
        client.simDestroyObject.assert_not_called()

    def test_failed_destruction_cannot_start_a_contaminated_trial(self):
        from unittest.mock import Mock
        name = 'FalconJointGate_Drone3_1_top'
        client = SimpleNamespace(simListSceneObjects=Mock(return_value=[name]),
                                 simDestroyObject=Mock(return_value=False))
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            with patch.object(flight.time, 'sleep'), self.assertRaisesRegex(ValueError, 'cleanup not verified'):
                flight.prepare_gate_scene(client, Path(directory))

    def test_similar_user_name_is_not_treated_as_our_generated_bar(self):
        from unittest.mock import Mock
        client = SimpleNamespace(simListSceneObjects=Mock(return_value=['FalconJointGate_custom_top']),
                                 simDestroyObject=Mock())
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            with self.assertRaisesRegex(ValueError, 'other generated gates'):
                flight.prepare_gate_scene(client, Path(directory))
        client.simDestroyObject.assert_not_called()


class IntegrationTests(unittest.TestCase):
    def test_native_calendar_passes_sample_and_drops_expired_sample_under_original_deadline(self):
        rt = runtime(adaptive=False)
        pilot, executor, io = Mock(), Mock(), Mock()
        pending = {}

        def event(release, cycle):
            return dict(vehicle='a', planned_release_s=release, requested_period_s=.05, slot=cycle,
                        native_release_status=0, native_release_s=release+.001)

        first_sample = dict(vehicle='a', cycle=0, packet=object(), sample_status=0, sample_start_s=10.002)
        io.drain_calendar_samples.return_value = [event(10., 0)], [first_sample]
        flight.consume_native_calendar(io, rt, {'a': pilot}, {'a': executor}, pending)
        self.assertEqual(rt.rows[0]['deadline_s'], 10.05)
        executor.submit.assert_called_once_with(pilot.run_job, rt.rows[0], first_sample)
        rt.finish(rt.rows[0])

        expired = dict(vehicle='a', cycle=1, packet=None, sample_status=1, native_sample_finished_s=10.11)
        io.drain_calendar_samples.return_value = [event(10.05, 1)], [expired]
        flight.consume_native_calendar(io, rt, {'a': pilot}, {'a': executor}, pending)
        self.assertAlmostEqual(rt.rows[1]['deadline_s'], 10.10)
        self.assertEqual(rt.rows[1]['status'], 'dropped_before_sample')
        self.assertIsNone(rt.loops['a']['busy'])
        self.assertEqual(pending, {})
        self.assertEqual(executor.submit.call_count, 1)
        io.complete_sample.assert_called_once_with('a', 1)

    def test_real_threads_calendar_board_and_cleanup_without_hardware(self):
        # Physics/its clock freeze when paused. With no explicit rate/throttle
        # command, resuming physics produces free fall, reproducing the live bug.
        state = {}
        clock = dict(paused=True, last=time.perf_counter(), sim=0.)
        holds, starts, first_live = {}, {}, {}
        primed_while_paused = set()
        def advance():
            now = time.perf_counter()
            dt = 0. if clock['paused'] else now-clock['last']
            clock.update(last=now, sim=clock['sim']+dt)
            for name, kin in state.items():
                acceleration = 0. if name in holds else 9.80665
                kin.position.z_val += kin.linear_velocity.z_val*dt+.5*acceleration*dt*dt
                kin.linear_velocity.z_val += acceleration*dt
        class Vector:
            def __init__(self, x=0., y=0., z=0.):
                self.x_val, self.y_val, self.z_val = x, y, z
        class Quaternion:
            def __init__(self, x, y, z, w):
                self.x_val, self.y_val, self.z_val, self.w_val = x, y, z, w
        class RPC:
            def __init__(self, **kw):
                pass
            def ping(self):
                pass
            confirmConnection = ping
            reset = ping
            def listVehicles(self):
                return ['Drone1', 'Drone2', 'Drone3']
            def simListSceneObjects(self, pattern):
                return []
            def simPause(self, value):
                advance()
                clock['paused'] = value
            def enableApiControl(self, *a, **kw):
                pass
            armDisarm = enableApiControl
            hoverAsync = enableApiControl
            def moveByAngleRatesThrottleAsync(self, *a, **kw):
                advance()
                name = kw['vehicle_name']
                holds[name] = a[3]
                if clock['paused']:
                    primed_while_paused.add(name)
            def simSetKinematics(self, kin, **kw):
                state[kw['vehicle_name']] = kin
                starts[kw['vehicle_name']] = kin.position.z_val
            def getMultirotorState(self, **kw):
                advance()
                name = kw['vehicle_name']
                if not clock['paused'] and name not in first_live:
                    first_live[name] = (state[name].position.z_val-starts[name],
                                        state[name].linear_velocity.z_val)
                return SimpleNamespace(kinematics_estimated=state[name], timestamp=int(clock['sim']*1e9))
            def simGetCollisionInfo(self, **kw):
                return SimpleNamespace(has_collided=False, time_stamp=0)
        api = SimpleNamespace(MultirotorClient=RPC, Vector3r=Vector, Quaternionr=Quaternion,
                              KinematicsState=SimpleNamespace)
        action = np.array([9.80665, 0., 0., 0.])
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            root = Path(directory)
            np.savetxt(root/'golden.txt', np.tile(np.r_[np.zeros(237), action], (2, 1)))
            args = SimpleNamespace(vehicles=['Drone1', 'Drone2', 'Drone3'], period_ms=50.,
                periods_ms=[20., 30., 40., 50., 60., 80.], maximum_update_ms=160.,
                gate_width=4., duration=.5, output=root/'out', bundle=root,
                board='mock', board_port=5238, model_id=11, airsim_host='mock',
                policy='adaptive', order='edf', prep_ms=3., post_ms=2., decision_ms=3.)
            args.evaluation_ms = 50.
            args.trace_thread_cycles = True
            args.initial_periods = root/'initial_periods.json'
            args.initial_periods.write_text('{"Drone1":40,"Drone2":60,"Drone3":80}', encoding='utf-8')
            board = SimpleNamespace(peer=('mock', 5238), close=lambda: None)
            manifest = dict(models=[dict(numeric_id=11, golden_file='golden.txt')],
                            wire_tag_hex='mock', normalization_sha256='mock')
            with patch.object(flight, 'make_client', return_value=(board, manifest)), \
                 patch.object(flight, 'predict', return_value=(action, {})), \
                 patch.object(flight, 'load_airsim_client'), \
                 patch('airsim_interface.require_airsim', return_value=api), \
                 patch('airsim_interface.vehicle_local_to_world_offset', return_value=np.zeros(3)), \
                 patch('gate_visualization.spawn_gate_sequences', return_value=SimpleNamespace(destroy=lambda: None)):
                result = flight.run(args)
            self.assertEqual(result['errors'], [])
            self.assertLess(max(abs(p) for p, _ in first_live.values()), .05,
                            'physics ran before the first scheduled sample')
            self.assertLess(max(abs(v) for _, v in first_live.values()), .5)
            self.assertEqual(primed_while_paused, set(args.vehicles))
            self.assertTrue(result['sim_clock_passed'])
            self.assertEqual(len(result['vehicles']), 3)
            self.assertGreaterEqual(result['requests'], 6)
            self.assertGreater(result['outcomes'].get('delivered', 0), 0)
            self.assertGreater(result['prep_phases_ms']['collision_ready_to_reference_ready']['count'], 0)
            self.assertGreater(result['prep_phases_ms']['safety_deadline_ready_to_observation_published']['count'], 0)
            self.assertGreater(result['prep_phases_ms']['encode_start_to_feature_ready']['count'], 0)
            self.assertTrue((root/'out'/'decisions.json').exists())
            self.assertTrue(result['evaluation']['valid'])
            self.assertTrue((root/'out'/'evaluation_Drone1.json').exists())
            from controller_architecture_study.ip37_campaign import trial_summary
            summary = trial_summary(root/'out', {'periods': {'Drone1': 40., 'Drone2': 60., 'Drone3': 80.},
                                                'order': 'edf'}, 'adaptive')
            self.assertIsNotNone(summary['J_total'])
            self.assertEqual(summary['collisions'], 0)
            self.assertFalse(result['flight_passed'])  # no fabricated gate crossing success
            args.reference_teacher, args.policy = True, 'fixed'
            args.vehicles, args.output = ['Drone1'], root/'teacher'
            args.initial_periods = None
            teacher = SimpleNamespace(query=lambda *a: (action, {'solve_ms': .5}),
                                      signature=lambda: {'teacher': 'mock'})
            with patch.object(flight, 'make_client') as connect, \
                 patch.object(flight, 'predict') as predict_board, \
                 patch.object(flight, 'load_airsim_client'), \
                 patch('controller_architecture_study.convex_teacher.ConvexTeacher', return_value=teacher), \
                 patch('airsim_interface.require_airsim', return_value=api), \
                 patch('airsim_interface.vehicle_local_to_world_offset', return_value=np.zeros(3)), \
                 patch('gate_visualization.spawn_gate_sequences', return_value=SimpleNamespace(destroy=lambda: None)):
                diagnostic = flight.run(args)
                args.teacher_variant, args.output = 'qp_ff', root/'diagnostic'
                with patch('controller_architecture_study.teacher_attitude_probe.DiagnosticTeacher', return_value=teacher) as create:
                    changed = flight.run(args)
                create.assert_called_once_with('qp_ff')
            connect.assert_not_called()
            predict_board.assert_not_called()
            self.assertEqual(diagnostic['errors'], [])
            self.assertFalse(diagnostic['board_used'])
            self.assertIsNone(diagnostic['final_empirical_board_cost_ms'])
            self.assertGreater(diagnostic['outcomes'].get('delivered', 0), 0)
            self.assertGreater(diagnostic['teacher_solve_ms']['count'], 0)
            self.assertFalse(changed['board_used'])
            self.assertEqual(changed['errors'], [])
            self.assertGreater(changed['outcomes'].get('delivered', 0), 0)


if __name__ == '__main__':
    unittest.main()
