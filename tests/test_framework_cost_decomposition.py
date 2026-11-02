import unittest

from controller_architecture_study.framework_runtime_costs import decomposed_measurements


class CostDecompositionTests(unittest.TestCase):
    def test_only_post_helper_busy_overlap_is_removed_and_nested_work_counts_once(self):
        rows = [{"sample_submitted_s":0.,"sample_io_finished_s":3.,"sample_response_received_s":9.},
            {"frontend_start_s":2.,"feature_done_s":5.},
            {"handled_s":4.,"output_finished_s":7.},
            {"frontend_start_s":9.5,"feature_done_s":10.}]
        decisions = [{"started_s":4.,"finished_s":6.},
            {"started_s":0.,"finished_s":100.,"outside_physics":True}]
        measured,visible,_ = decomposed_measurements(rows,decisions)
        self.assertEqual(visible["sample_io_s"],[9.])
        # Busy union is[2,7]; only[3,7] blocks a completed helper response.
        self.assertEqual(measured["sample_io_s"],[5.])
        self.assertEqual(measured["input_s"],visible["input_s"])
        self.assertEqual(measured["output_s"],visible["output_s"])

    def test_local_actuation_cost_excludes_upstream_queue_and_parent_notification(self):
        rows = [{"io_submitted_s":1.,"io_started_s":1.003,"rpc_return_s":1.004,
            "command_response_received_s":1.008,"output_finished_s":1.010,"handled_s":1.009}]
        measured,visible,_ = decomposed_measurements(rows,[])
        self.assertAlmostEqual(measured["actuation_s"][0],.001)
        self.assertAlmostEqual(visible["command_s"][0],.008)
        self.assertAlmostEqual(measured["command_s"][0],.004)

    def test_unexplained_notification_delay_and_cleanup_exclusion_are_preserved(self):
        rows = [{"sample_submitted_s":0.,"sample_io_finished_s":1.,"sample_response_received_s":4.},
            {"sample_submitted_s":0.,"sample_io_finished_s":1.,"sample_response_received_s":100.},
            {"frontend_start_s":5.,"feature_done_s":6.}]
        measured,visible,cutoff = decomposed_measurements(rows,[])
        self.assertEqual(cutoff,6.)
        self.assertEqual(measured["sample_io_s"],[4.])
        self.assertEqual(visible["sample_io_s"],[4.])


if __name__=="__main__":
    unittest.main()
