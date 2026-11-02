import unittest
from unittest.mock import patch
import json
from pathlib import Path

from controller_architecture_study.framework_runtime import bounded_plan


class PlannerBudgetTests(unittest.TestCase):
    def test_real_bootstrap_rejection_has_a_checked_paired_ingress_plan(self):
        data=json.loads((Path(__file__).parent/"fixtures/framework_paired_ingress_bootstrap.json").read_text())
        proposal,info=bounded_plan(0.,data["future"],[],data["profile"],data["costs"],inflight_limit=7)
        self.assertIsNotNone(proposal)
        self.assertEqual(info["release_pattern"],"paired_ingress")
        self.assertLessEqual(info["checked"],5)
        for vehicle,row in proposal.items():
            self.assertEqual(row["deadline_s"],data["future"][vehicle]["deadline_s"])
            self.assertEqual(row["input_age_limit_s"],data["future"][vehicle]["input_age_limit_s"])

    def test_failed_single_shift_retains_feasible_plan_without_bisection(self):
        row={"sample_id":"a:0","vehicle":"a","model":"m","priority":0,
            "deadline_s":.100,"planned_release_s":0.,"stage":"input"}
        profile={"m":{"segment_estimates_s":[.001]}}
        costs={"input_s":.001,"output_s":.001,"dispatch_guard_s":.001}
        with patch("controller_architecture_study.framework_runtime.simulate_pipeline",
                side_effect=[(True,{"a:0":.020},3),(False,{},4)]) as forecast:
            proposal,info=bounded_plan(0.,{"a":row},[],profile,costs)
        self.assertEqual(forecast.call_count,2)
        self.assertEqual(info["checked"],2)
        self.assertTrue(info["feasible"])
        self.assertEqual(proposal["a"]["planned_release_s"],0.)
        self.assertEqual(proposal["a"]["deadline_s"],.100)


if __name__=="__main__":
    unittest.main()
