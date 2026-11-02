import unittest

from controller_architecture_study.framework_runtime import release_deadline,simulate_pipeline,bounded_plan


class InputDeadlineTests(unittest.TestCase):
    def test_release_only_tightens_and_does_not_extend_the_physical_control_deadline(self):
        row = {"deadline_s":10.080,"input_age_limit_s":.030}
        self.assertAlmostEqual(release_deadline(row,10.010),10.040)
        self.assertEqual(release_deadline(row,10.070),10.080)
        self.assertEqual(row["deadline_s"],10.080)

    def test_prediction_rejects_an_old_input_even_when_the_physical_deadline_is_later(self):
        row = {"sample_id":"a","vehicle":"a","model":"m","priority":0,
            "planned_release_s":0.,"deadline_s":1.,"input_age_limit_s":.010}
        profile = {"m":{"segment_estimates_s":[.020]}}
        costs = {"input_s":.001,"output_s":.001}
        self.assertFalse(simulate_pipeline(0.,[row],[],profile,costs)[0])
        del row["input_age_limit_s"]
        self.assertTrue(simulate_pipeline(0.,[row],[],profile,costs)[0])

    def test_freshness_uses_full_frontend_service_for_bounded_staggering(self):
        rows = {v:{"sample_id":v,"vehicle":v,"model":"m","priority":i,
            "planned_release_s":0.,"deadline_s":.100,"input_age_limit_s":.050} for i,v in enumerate(("a","b"))}
        profile = {"m":{"segment_estimates_s":[.001]}}
        costs = {"input_s":.002,"output_s":.003,"handoff_s":.001,"dispatch_guard_s":.005}
        proposal,info = bounded_plan(0.,rows,[],profile,costs)
        self.assertIsNotNone(proposal)
        self.assertEqual(info["release_pattern"],"pipeline_staggered")
        self.assertAlmostEqual(proposal["b"]["planned_release_s"]-proposal["a"]["planned_release_s"],.006)
        self.assertLessEqual(info["checked"],5)

    def test_compact_fallback_fits_when_sparse_and_common_releases_do_not(self):
        profile = {"small":{"segment_estimates_s":[.0002]*5}}
        costs = {"input_s":.0025,"output_s":.0038,"sample_io_s":.0072,"handoff_s":.001,
            "command_s":.0063,"io_s":.0084,"ingress_s":.0024,"egress_s":.0023,
            "completion_guard_s":.001,"dispatch_guard_s":.020,"io_slots":7}
        rows = {v:{"sample_id":v,"vehicle":v,"model":"small","priority":i,
            "planned_release_s":0.,"deadline_s":.030,"input_age_limit_s":.025}
            for i,v in enumerate("ab")}
        pending = []
        for spacing in (0.,.0073):
            proposal = [{**row,"planned_release_s":i*spacing} for i,row in enumerate(rows.values())]
            self.assertFalse(simulate_pipeline(0.,proposal,pending,profile,costs,inflight_limit=7)[0])
        proposal,info = bounded_plan(0.,rows,pending,profile,costs,inflight_limit=7)
        self.assertIsNotNone(proposal)
        self.assertEqual(info["release_pattern"],"staggered")
        self.assertLessEqual(info["checked"],5)
        for v,row in proposal.items():
            self.assertEqual(row["deadline_s"],rows[v]["deadline_s"])
            self.assertLessEqual(info["finish"][v]+costs["completion_guard_s"],release_deadline(row,row["planned_release_s"]))


if __name__=="__main__":
    unittest.main()
