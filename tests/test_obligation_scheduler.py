import unittest
from dataclasses import replace

from controller_architecture_study.obligation_scheduler import (
    Calendar, ControlJob, SkipPermission, dispatch_budget, evaluate, plan,
)
from controller_architecture_study.requant18_profile import ControllerMode, pareto


def task(name, vehicle, ready, deadline, allowed=(1,), current=1, **kwargs):
    return ControlJob(name, vehicle, ready, ready, 0., 0., 0.,
                      deadline, deadline, deadline, allowed, current, **kwargs)


class ObligationSchedulerTests(unittest.TestCase):
    def setUp(self):
        self.modes = {
            1: ControllerMode(1, 'fast', 2., .02, .03),
            2: ControllerMode(2, 'accurate', 3.2, .01, .02),
            3: ControllerMode(3, 'recovery', 1., .04, .05),
        }

    def run_plan(self, calendar, **kwargs):
        return plan(calendar, self.modes, now=0., decision_ms=0., **kwargs)

    def example(self):
        return Calendar((task('a', 'a', 0., 10., (2,), 2),
                         task('b', 'b', 2.5, 5.5),
                         task('c', 'c', 4., 7.),
                         task('c_next', 'c', 7., 8.5, (3,), 3)))

    def permission(self, **kwargs):
        return replace(SkipPermission('c', 'c_next', 8.5, 3.9, True, 'synthetic test'), **kwargs)

    def test_three_deadlines_and_ps_post_are_enforced(self):
        job = replace(task('a', 'a', 0., 10.), fresh_deadline=2.5, post_ms=1.)
        self.assertFalse(self.run_plan(Calendar((job,)))['accepted'])
        job = replace(job, post_ms=.4)
        result = self.run_plan(Calendar((job,)))
        self.assertTrue(result['accepted'])
        self.assertAlmostEqual(result['forecast']['command_finish']['a'], 2.4)

    def test_shared_ps_contention_defeats_isolated_estimate(self):
        jobs = (replace(task('a', 'a', 0., 6.), prep_ms=3.),
                replace(task('b', 'b', 0., 6.), prep_ms=3.))
        self.assertFalse(self.run_plan(Calendar(jobs))['accepted'])

    def test_skip_commits_existing_successor_and_discharges_only_on_delivery(self):
        calendar = self.example()
        result = self.run_plan(calendar, permissions=(self.permission(),))
        self.assertTrue(result['accepted'])
        self.assertEqual(result['skipped'], 'c')
        committed = calendar.commit(result, 0.)
        self.assertEqual(len(committed.jobs), 3)
        self.assertEqual(committed.pending, (('c', 'c_next'),))
        self.assertTrue(next(j for j in committed.jobs if j.name == 'c_next').recovery)
        with self.assertRaises(ValueError):
            committed.delivered('c_next', 9.)
        self.assertEqual(committed.delivered('c_next', 8.).pending, ())

    def test_unknown_phase_and_tight_recovery_are_rejected(self):
        for permission in (self.permission(eligible=False), self.permission(hold_until=7.5),
                           self.permission(valid_until=-1.)):
            self.assertFalse(self.run_plan(self.example(), permissions=(permission,))['accepted'])

    def test_published_job_cannot_be_skipped(self):
        jobs = list(self.example().jobs)
        jobs[2] = replace(jobs[2], release=0., prep_ready=4.)
        self.assertFalse(self.run_plan(Calendar(tuple(jobs)), permissions=(self.permission(),))['accepted'])

    def test_pending_recovery_prevents_second_skip(self):
        calendar = self.example()
        jobs = tuple(replace(j, recovery=True) if j.name == 'c_next' else j for j in calendar.jobs)
        calendar = Calendar(jobs, pending=(('c', 'c_next'),))
        self.assertFalse(self.run_plan(calendar, permissions=(self.permission(),))['accepted'])

    def test_stale_version_or_elapsed_commit_is_rejected(self):
        calendar = Calendar((task('a', 'a', 0., 10.),))
        result = self.run_plan(calendar)
        with self.assertRaises(ValueError):
            replace(calendar, revision=1).commit(result, 0.)
        with self.assertRaises(ValueError):
            calendar.commit(result, .1)
        with self.assertRaises(ValueError):
            calendar.tighten_hold('a', 1.).commit(result, 0.)
        self.assertFalse(self.run_plan(calendar.tighten_hold('a', 1.))['accepted'])

    def test_active_nonpreemptive_work_remains_blocking(self):
        calendar = Calendar((task('a', 'a', 0., 3.),))
        self.assertFalse(self.run_plan(calendar, blocked_until=2.)['accepted'])

    def test_worker_must_finish_before_its_next_release(self):
        jobs = (task('a', 'one', 0., 5.), task('b', 'one', 1., 6.))
        result = self.run_plan(Calendar(jobs))
        self.assertFalse(result['accepted'])
        self.assertTrue(any(row['overlap'] for row in result['trace']))

    def test_head_upgrade_respects_all_tail_obligations(self):
        jobs = (task('a', 'a', 0., 10., (1, 2), 1),
                task('b', 'b', 2.5, 5.5),
                task('c', 'c', 4., 7.))
        choice = {job.name: job.current_mode for job in jobs}
        self.assertEqual(dispatch_budget(jobs, ('a', 'b', 'c'), choice, self.modes, 0.), 3.)
        result = self.run_plan(Calendar(jobs), enable_reorder=False)
        self.assertTrue(result['accepted'])
        self.assertEqual(result['choice']['a'], 1)

    def test_release_edit_cannot_move_hold_deadline(self):
        job = replace(task('a', 'a', 5., 6., release_options=((2., 6.),)), fresh_deadline=10.)
        result = self.run_plan(Calendar((job,)))
        self.assertTrue(result['accepted'])
        self.assertEqual(result['action'], 'release_edit')
        self.assertEqual(result['jobs'][0].release, 2.)
        self.assertEqual(result['jobs'][0].hold_deadline, 6.)

    def test_earlier_sampling_does_not_extend_input_lifetime(self):
        job = task('a', 'a', 5., 6., release_options=((2., 6.),))
        self.assertFalse(self.run_plan(Calendar((job,)))['accepted'])

    def test_explicit_dpu_order_waits_for_unready_task(self):
        jobs = (task('a', 'a', 0., 10.), task('b', 'b', 3., 5.))
        result = evaluate(jobs, ('b', 'a'), {'a': 1, 'b': 1}, self.modes, 0., 0., 0.)
        pl = [op for op in result['forecast']['timeline'] if op['resource'] == 'pl']
        self.assertEqual([(op['job'], op['start']) for op in pl], [('b', 3.), ('a', 5.)])

    def test_bounded_checks_and_no_unchecked_return(self):
        for budget in (1, 2, 3):
            result = self.run_plan(self.example(), permissions=(self.permission(),), max_checks=budget)
            self.assertLessEqual(result['checks'], budget)
            if result['accepted']:
                self.assertTrue(result['forecast']['feasible'])

    def test_frontier_uses_selection_set_not_check_set(self):
        first = ControllerMode(1, 'a', 1., .01, 100.)
        second = ControllerMode(2, 'b', 2., .02, .001)
        self.assertEqual(pareto((first, second)), (first,))

    def test_published_job_cannot_be_retimed_and_locked_model_is_preserved(self):
        job = task('a', 'a', 0., 3., (1, 2), 2, release_options=((1., 4.),),
                   mode_locked=True)
        self.assertFalse(self.run_plan(Calendar((job,)))['accepted'])

    def test_obligations_do_not_hide_missing_successors(self):
        calendar = self.example()
        with self.assertRaises(ValueError):
            self.run_plan(replace(calendar, pending=(('c', 'absent'),)))


if __name__ == '__main__':
    unittest.main()
