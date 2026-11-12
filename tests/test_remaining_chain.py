import unittest
from controller_architecture_study.bottleneck_joint import Job, Platform, Scenario, simulate, witness
from controller_architecture_study.bottleneck_abc import path_cost
from controller_architecture_study.ip37_joint_runtime import JointRuntime


class RemainingChainTests(unittest.TestCase):
    def test_returned_command_can_meet_deadline_while_worker_misses_next_release(self):
        p = Platform(prep=0., dispatch=0., ingress=0., tail=0., post=61.09,
                     rearm=30.92, decision=0.)
        old = Job('old', 'a', 0., 80., (), 'm')
        forecast = simulate(Scenario((), p, carry=(old,), legal_groups=(1,)), (), ('edf', 1))
        self.assertTrue(forecast['feasible'])
        self.assertAlmostEqual(forecast['command_finish']['old'], 61.09)
        self.assertAlmostEqual(forecast['finish']['old'], 92.01)

    def test_acknowledged_carry_keeps_worker_reservation_after_old_deadline(self):
        rt = JointRuntime(('a',), 10., 50., (50., 80.), 11, 2., prep_ms=1., post_ms=2.)
        row, = rt.release_due(10.)
        row['command_return_s'] = 10.04
        rt.update_cpu_costs(rearm_ms=12.)
        rt.release_due(10.0501)
        self.assertEqual(rt.rows[-1]['status'], 'dropped_busy_io')
        rt.observe('a', 10.051, 10.15, {}, 0)
        _, scenario = rt.snapshot(10.052)
        old, = scenario.carry
        self.assertTrue(old.command_completed)
        self.assertLess(old.deadline, 0.)
        result = simulate(scenario, (0,), ('edf', 1))
        self.assertNotIn(old.name, result['slack'])
        self.assertAlmostEqual(result['finish'][old.name], rt.decision_ms+rt.rearm_ms)

    def test_runtime_post_phase_excludes_completed_preparation(self):
        rt = JointRuntime(('a',), 10., 50., (50., 80.), 11, 2., prep_ms=20., post_ms=3.)
        row, = rt.release_due(10.)
        row['phase'] = 'post'
        rt.observe('a', 10., 10.1, {}, 0)
        _, scenario = rt.snapshot(10.001)
        self.assertEqual(scenario.carry[0].remaining_operations, (('ps', 3.),))
        self.assertAlmostEqual(scenario.carry[0].deadline, 49.)

    def test_completed_preparation_is_not_charged_twice(self):
        p = Platform(prep=20., post=3., decision=1.)
        old = Job('old', 'a', 0., 10., (4.,), 'm', (('pl', 4.), ('ps', 3.)))
        scenario = Scenario((), p, carry=(old,), legal_groups=(1,))
        forecast = simulate(scenario, (), ('edf', 1))
        self.assertTrue(forecast['feasible'])
        self.assertEqual(path_cost(old, p), 7.)
        self.assertLessEqual(witness(scenario, (old,))['deficit'], 0.)

    def test_active_job_keeps_remaining_wait_and_deadline(self):
        p = Platform(prep=20., post=3., decision=1., initial_pl_block=6.)
        old = Job('old', 'a', 6., 8., (0.,), 'm', (('ps', 3.),))
        forecast = simulate(Scenario((), p, carry=(old,), legal_groups=(1,)), (), ('edf', 1))
        self.assertFalse(forecast['feasible'])
        self.assertEqual(forecast['finish']['old'], 9.)


if __name__ == '__main__':
    unittest.main()
