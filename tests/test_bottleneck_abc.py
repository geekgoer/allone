import unittest

from controller_architecture_study.bottleneck_abc import abc_repair
from controller_architecture_study.bottleneck_joint import Job, Loop, Mode, Option, Platform, Scenario


def loop(name, release, deadline, segments, alternatives=(), *, locked=True):
    return Loop(name, release - 40., release, deadline, 100.,
                (Option(40., Mode(name, segments)), *alternatives), locked)


class ABCRepairTests(unittest.TestCase):
    def setUp(self):
        self.platform = Platform(prep=0., dispatch=0., post=0., ingress=0.,
                                 tail=0., switch=0., decision=0.)

    def run_case(self, loops, *, groups=(4,), group=4, **kwargs):
        return abc_repair(Scenario(loops, kwargs.pop('platform', self.platform),
                                   horizon=21., initial_policy=('edf', group),
                                   legal_groups=groups, **kwargs))

    def test_a_selects_smallest_advance_that_clears_path(self):
        model = Mode('a', (6.,))
        a = loop('a', 16., 20., model.segments,
                 (Option(37., model), Option(38., model)), locked=False)
        result = self.run_case((a,))
        self.assertTrue(result['accepted'])
        self.assertEqual(result['choice'], (2,))
        self.assertEqual(result['initial_type'], 'A')
        self.assertEqual(result['attempts'][0]['latest_start_ms'], 14.)

    def test_a_locked_release_needs_qualified_faster_mode(self):
        fast = Mode('fast', (3.,))
        a = loop('a', 16., 20., (6.,), (Option(35., Mode('a', (6.,))),
                                        Option(40., Mode('unsafe', (1.,)), False),
                                        Option(40., fast)))
        result = self.run_case((a,))
        self.assertTrue(result['accepted'])
        self.assertEqual(result['choice'], (3,))
        self.assertEqual(result['forecast']['jobs'][0].release, 16.)

    def test_b_repairs_pl_demand_in_overloaded_window(self):
        a = loop('a', 0., 10., (6.,), (Option(40., Mode('small', (3.,))),))
        b = loop('b', 0., 10., (6.,))
        result = self.run_case((a, b))
        self.assertTrue(result['accepted'])
        self.assertEqual(result['initial_type'], 'B')
        self.assertEqual(result['attempts'][0]['kind'], 'pl')
        self.assertEqual(result['attempts'][0]['gain_ms'], 3.)

    def test_b_merges_segments_for_ps_overload(self):
        a = loop('a', 0., 10., (.1,) * 4)
        b = loop('b', 0., 10., (.1,) * 4)
        result = self.run_case((a, b), groups=(1, 4), group=1,
                               platform=Platform(prep=0., dispatch=2., post=0.,
                                                 ingress=0., tail=0., switch=0., decision=0.))
        self.assertTrue(result['accepted'])
        self.assertEqual(result['attempts'][0]['kind'], 'ps')
        self.assertEqual(result['attempts'][0]['action'], 'merge_segments')

    def test_c_only_changes_legal_schedule(self):
        a = loop('a', 0., 20., (2.,) * 4)
        b = loop('b', 1., 5., (2.,))
        result = self.run_case((a, b), groups=(1, 4))
        self.assertTrue(result['accepted'])
        self.assertEqual(result['initial_type'], 'C')
        self.assertEqual(result['choice'], (0, 0))
        self.assertEqual(result['policy'], ('edf', 1))
        self.assertEqual(result['attempts'][0]['action'], 'schedule_only')
        self.assertFalse(self.run_case((a, b))['accepted'])

    def test_committed_carry_work_is_preserved_when_no_repair_exists(self):
        a = loop('a', 0., 10., (6.,))
        result = self.run_case((a,), carry=(Job('old', 'a', 0., 10., (6.,), 'old'),))
        self.assertFalse(result['accepted'])
        self.assertEqual(len(result['forecast']['jobs']), 2)
        self.assertLessEqual(result['checks'], 5)


if __name__ == '__main__':
    unittest.main()
