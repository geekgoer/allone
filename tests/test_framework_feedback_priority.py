import unittest

from controller_architecture_study.framework_runtime_flight import feedback_can_run,commit_plan
from controller_architecture_study.framework_runtime import simulate_pipeline


class FeedbackPriorityTests(unittest.TestCase):
    def test_forecast_charges_feedback_before_the_next_adaptive_sample(self):
        future = [{"sample_id":"next","vehicle":"b","model":"m","priority":1,
            "planned_release_s":0.,"deadline_s":.006}]
        pending = [{"sample_id":"old","vehicle":"a","model":"m","priority":0,
            "planned_release_s":0.,"deadline_s":.1,"stage":"analysis","command_completed_s":0.}]
        profile = {"m":{"segment_estimates_s":[.001]}}
        costs = {"input_s":.001,"sample_io_s":.001,"output_s":.004,"io_s":.001,"command_s":.001}
        self.assertFalse(simulate_pipeline(0.,future,pending,profile,costs)[0])
        self.assertTrue(simulate_pipeline(0.,future,[],profile,costs)[0])

    def test_adaptive_feedback_is_not_blocked_by_an_unreleased_lower_urgency_sample(self):
        costs = {"output_s":.003,"handoff_s":.001,"command_s":.005,"completion_guard_s":.001}
        plans = {"other":{"planned_release_s":.010,"deadline_s":.080}}
        self.assertTrue(feedback_can_run("adaptive",.009,plans,costs,[]))
        self.assertFalse(feedback_can_run("static",.009,plans,costs,[]))

    def test_feedback_must_still_protect_an_inflight_urgent_command(self):
        costs = {"output_s":.003,"handoff_s":.001,"command_s":.005,"completion_guard_s":.001}
        live = [{"status":"submitted","deadline_s":.015}]
        self.assertFalse(feedback_can_run("adaptive",.009,{},costs,live))

    def test_replanning_moves_only_prospective_release_and_keeps_the_control_deadline(self):
        original = {"model":"same","deadline_s":.080,"sample_id":"other:1","planned_release_s":.010}
        plans,waiting = {"other":dict(original)},{}
        proposal = {"other":{**original,"planned_release_s":.020}}
        self.assertTrue(commit_plan(proposal,plans,waiting))
        self.assertEqual(plans["other"]["deadline_s"],.080)
        self.assertEqual(plans["other"]["model"],"same")


if __name__=="__main__":
    unittest.main()
