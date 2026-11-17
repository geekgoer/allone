from concurrent.futures import Future
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from controller_architecture_study.ip37_obligation_runtime import ObligationRuntime
from controller_architecture_study.obligation_scheduler import ControlJob, Calendar, plan
from controller_architecture_study.requant18_profile import ControllerMode


class LiveObligationTests(unittest.TestCase):
    def runtime(self):
        modes = {1: ControllerMode(1, 'fast', 1., .02, .03),
                 2: ControllerMode(2, 'accurate', 5., .01, .02)}
        rt = ObligationRuntime(['Drone1'], 100., 50., (20., 50., 80.), 2, 6.,
                               backend=SimpleNamespace(modes=modes), candidate_ids=(1, 2),
                               prep_ms=.5, post_ms=.5, decision_ms=3., network_ms=0.)
        rt.loops['Drone1'].update(previous=100., next=100.05, cycle=1)
        rt.observe('Drone1', 100., 100.2, {}, 0)
        return rt

    def test_live_plan_commits_real_future_modes(self):
        rt = self.runtime()
        with patch('controller_architecture_study.ip37_obligation_runtime.time.perf_counter',
                   side_effect=(100.001, 100.002, 100.003)):
            record = rt.plan_once()
        self.assertIn(record['status'], ('installed', 'plan_refreshed'))
        self.assertIn('Drone1:1', rt.future_modes)
        self.assertTrue(rt.dispatch_order)

    def test_planned_skip_preserves_period_and_marks_successor(self):
        rt = self.runtime()
        rt.planned_skip['Drone1'] = 1
        rt.pending_recovery['Drone1'] = dict(cycle=2, deadline_s=100.15)
        self.assertEqual(rt.release_due(100.051), [])
        self.assertEqual(rt.rows[-1]['status'], 'planned_skip')
        self.assertEqual(rt.loops['Drone1']['period_ms'], 50.)
        rows = rt.release_due(100.101)
        self.assertEqual(rows[0]['cycle'], 2)
        self.assertEqual(rows[0]['recovery_due_s'], 100.15)
        rows[0].update(command_return_s=100.13)
        rt.finish(rows[0])
        self.assertNotIn('Drone1', rt.pending_recovery)

    def test_late_recovery_remains_pending(self):
        rt = self.runtime()
        rt.pending_recovery['Drone1'] = dict(cycle=1, deadline_s=100.06)
        row = rt.release_due(100.051)[0]
        row['command_return_s'] = 100.07
        rt.finish(row)
        self.assertFalse(row['recovery_on_time'])
        self.assertIn('Drone1', rt.pending_recovery)

    def test_model_key_reaches_dispatch(self):
        rt = self.runtime()
        row = rt.release_due(100.051)[0]
        row.update(future=Future(), feature=[0.]*237, queue_s=100.052, phase='queued',
                   actuation_deadline_s=100.09)
        rt.queue.append(row)
        rt.future_modes['Drone1:1'] = 1
        taken = rt.take(100.053)
        self.assertEqual(taken['configuration_id'], 1)

    def test_no_affordable_model_is_dropped_before_udp(self):
        rt = self.runtime()
        row = rt.release_due(100.051)[0]
        row.update(future=Future(), queue_s=100.052, phase='queued',
                   actuation_deadline_s=100.0531)
        rt.queue.append(row)
        self.assertIsNone(rt.take(100.053))
        self.assertTrue(row['future'].cancelled())
        self.assertIsNone(rt.active)

    def test_fixed_model_must_fit_even_when_another_model_would_fit(self):
        rt = self.runtime()
        rt.adaptive = False
        row = rt.release_due(100.051)[0]
        row.update(future=Future(), phase='queued', queue_s=100.052,
                   actuation_deadline_s=100.055)
        rt.queue.append(row)
        self.assertIsNone(rt.take(100.053))
        self.assertTrue(row['future'].cancelled())

    def test_published_recovery_is_preserved_in_calendar(self):
        rt = self.runtime()
        rt.pending_recovery['Drone1'] = dict(cycle=1, deadline_s=100.09)
        row = rt.release_due(100.051)[0]
        row.update(future=Future(), phase='queued')
        calendar, _, blocked = rt._calendar(100.052)
        recovery = next(j for j in calendar.jobs if j.name == 'Drone1:1')
        self.assertTrue(recovery.recovery)
        self.assertIn(('Drone1', 'Drone1:1'), calendar.pending)
        decision = plan(calendar, rt.modes, now=0., blocked_until=blocked)
        self.assertTrue(decision['accepted'])

    def test_observation_tightening_invalidates_dispatch_order(self):
        rt = self.runtime()
        rt.dispatch_order = ('Drone1:1',)
        rt.observe('Drone1', 100.01, 100.04, {}, 0)
        self.assertEqual(rt.dispatch_order, ())

    def test_period_edit_retimes_successor_and_preserves_hold(self):
        mode = ControllerMode(1, 'm', 2., .01, .01)
        first = ControlJob('one', 'drone', 5., 5., 0., 0., 0., 6., 6., 20.,
                           (1,), 1, release_options=((2., 5.),), periodic_edit=True)
        second = replace(first, name='two', release=10., prep_ready=10., slot_deadline=15.,
                         hold_deadline=15., fresh_deadline=20., release_options=(), periodic_edit=False)
        result = plan(Calendar((first, second)), {1: mode}, now=0., decision_ms=0.)
        self.assertTrue(result['accepted'])
        revised = {job.name: job for job in result['jobs']}
        self.assertEqual(revised['two'].release, 5.)
        self.assertEqual(revised['one'].hold_deadline, 6.)
        self.assertEqual(revised['one'].fresh_deadline, 17.)
        self.assertEqual(revised['two'].fresh_deadline, 15.)

    def test_unknown_active_service_remainder_blocks_replanning(self):
        rt = self.runtime()
        rt.active = dict(model_id=1, board_start_s=99.)
        self.assertIsNone(rt._calendar(100.))

    def test_runtime_accepts_subnet_configuration_keys(self):
        from controller_architecture_study.controller_backend import Configuration
        mode = ControllerMode(101, 'depth9_width512', 1., .01, .02)
        backend = SimpleNamespace(modes={101: mode},
                                  configurations={101: Configuration(101, 'supernet', mode.name, 101,
                                                                     (('depth', 9), ('width', 512)))})
        rt = ObligationRuntime(['Drone1'], 100., 50., (50.,), 101, 2.,
                               backend=backend, candidate_ids=(101,), prep_ms=.5,
                               post_ms=.5, decision_ms=3., network_ms=0.)
        row = rt.release_due(100.001)[0]
        row.update(future=Future(), phase='queued', queue_s=100.002)
        rt.queue.append(row)
        self.assertEqual(rt.take(100.003)['configuration_id'], 101)

    def test_configuration_interface_imports_without_inference_dependencies(self):
        import subprocess
        import sys
        from pathlib import Path
        code = ("import sys; sys.path.insert(0, sys.argv[1]); "
                "from controller_architecture_study.controller_backend import Configuration; "
                "assert not {'numpy', 'torch', 'controller_architecture_study.ip37_shadow'} & sys.modules.keys()")
        subprocess.run([sys.executable, '-I', '-S', '-c', code,
                        str(Path(__file__).resolve().parents[1])], check=True)


if __name__ == '__main__':
    unittest.main()
