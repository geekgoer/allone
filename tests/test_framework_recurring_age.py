from types import SimpleNamespace
from unittest.mock import Mock,patch
import unittest

from controller_architecture_study.probe_recurring_input_age import advance_to


class RecurringAgeTests(unittest.TestCase):
    def test_a_sub_tick_without_progress_is_retried_without_inventing_elapsed_time(self):
        raw = SimpleNamespace(timestamp_ns=0)
        following = SimpleNamespace(timestamp_ns=3000000)
        world = SimpleNamespace(initial={"Drone1":{"sim_ns":0}},failures={"Drone1":None},observe=Mock())
        with patch("controller_architecture_study.probe_recurring_input_age.advance",
                side_effect=[(raw,None,0.),(following,None,.003)]) as advance:
            result = advance_to(world,raw,.002)
        self.assertIs(result,following)
        self.assertEqual(advance.call_count,2)
        self.assertTrue(all(c.args[1]>=.003 for c in advance.call_args_list))

    def test_no_progress_cannot_cause_an_unbounded_retry(self):
        raw = SimpleNamespace(timestamp_ns=0)
        world = SimpleNamespace(initial={"Drone1":{"sim_ns":0}},failures={"Drone1":None},observe=Mock())
        with patch("controller_architecture_study.probe_recurring_input_age.advance",return_value=(raw,None,0.)) as advance:
            with self.assertRaisesRegex(RuntimeError,"bounded"):
                advance_to(world,raw,.002)
        self.assertEqual(advance.call_count,8)


if __name__=="__main__":
    unittest.main()
