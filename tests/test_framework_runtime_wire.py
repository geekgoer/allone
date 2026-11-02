import unittest

import numpy as np

from controller_architecture_study.framework_runtime_wire import decode, encode


class RuntimeWireTests(unittest.TestCase):
    def setUp(self):
        self.row = {"sample_id":"Drone2:19","vehicle":"Drone2","model":"frozen",
            "priority":3,"deadline_s":12345.678901234,"enqueue_s":12345.62}

    def test_input_preserves_context_and_values_but_does_not_send_frontend_history(self):
        feature = np.linspace(-2.,3.,237,dtype=np.float32)
        original = {**self.row,"analysis_risk":{"h":.2},"frozen_deadline_s":self.row["deadline_s"]}
        kind,actual,context = decode(encode(("submit",feature,original)))
        self.assertEqual(kind,"submit")
        np.testing.assert_array_equal(actual,feature)
        self.assertEqual(context,self.row)
        self.assertIn("frozen_deadline_s",original)

    def test_output_preserves_physical_values_status_and_timestamps(self):
        action = np.array([9.80665,-.125,.001,2.],dtype=np.float32)
        result = {**self.row,"status":"ready","ready_s":12345.63,"compute_ms":.17}
        kind,actual,context = decode(encode(("done",action,result)))
        self.assertEqual(kind,"done")
        np.testing.assert_array_equal(actual,action)
        self.assertEqual(context,result)
        self.assertEqual(decode(encode(("done",None,{**self.row,"status":"expired_before_segment"})))[1],None)

    def test_ack_and_control_messages_remain_compatible(self):
        ack = ("accepted","Drone2:19","frozen",self.row["deadline_s"],12345.624)
        self.assertEqual(decode(encode(ack)),ack)
        for message in (("stop",),("ready",{"profile":[.01,.02]}),("closed",[{"sample_id":"x"}])):
            self.assertEqual(decode(encode(message)),message)

    def test_wrong_payload_lengths_are_rejected_before_executor_dispatch(self):
        with self.assertRaises(ValueError):
            decode(encode(("submit",np.zeros(237,dtype=np.float32),self.row))[:-1])
        with self.assertRaises(ValueError):
            decode(b"S\xff\xff\xff\xff")

    def test_next_round_analysis_cannot_block_a_promised_release(self):
        from controller_architecture_study.framework_runtime_flight import analysis_fits_release_gap
        plans = {"Drone6":{"planned_release_s":25934.514081675,"deadline_s":25934.5479997}}
        original = {k:dict(v) for k,v in plans.items()}
        self.assertFalse(analysis_fits_release_gap(25934.5135523,plans,.005093775))
        self.assertTrue(analysis_fits_release_gap(25934.500,plans,.005093775))
        self.assertTrue(analysis_fits_release_gap(25934.5135523,{},.005093775))
        self.assertEqual(plans,original)

    def test_acknowledged_command_does_not_starve_ready_inputs_after_its_deadline(self):
        from controller_architecture_study.framework_runtime_flight import command_would_be_blocked
        costs = {"handoff_s":.001,"command_s":.0075695,"completion_guard_s":.001}
        acknowledged = {"deadline_s":26262.8021285,"status":"applied"}
        computing = {"deadline_s":26262.8175021,"status":"submitted"}
        self.assertFalse(command_would_be_blocked(26262.7928291,.0024555,costs,[acknowledged,computing]))
        self.assertFalse(command_would_be_blocked(26262.900,.0024555,costs,[acknowledged]))
        self.assertTrue(command_would_be_blocked(26262.7928291,.0024555,costs,[{**acknowledged,"status":"submitted"}]))
        self.assertEqual(acknowledged["deadline_s"],26262.8021285)

    def test_failed_admission_keeps_existing_schedule_and_waiting_job(self):
        from controller_architecture_study.framework_runtime_flight import commit_plan
        plans = {"a":{"sample_id":"a:0","model":"frozen","deadline_s":.08,"planned_release_s":.02}}
        waiting = {"b":{"sample_id":"b:0","model":"other","deadline_s":.09,"planned_release_s":0.}}
        original = {**plans,**waiting}
        self.assertFalse(commit_plan(None,plans,waiting))
        self.assertEqual({**plans,**waiting},original)
        proposal = {v:{**r,"planned_release_s":.01} for v,r in original.items()}
        self.assertTrue(commit_plan(proposal,plans,waiting))
        self.assertFalse(waiting)
        self.assertEqual(plans["b"]["deadline_s"],.09)

    def test_admission_never_extends_a_waiting_deadline_or_changes_its_model(self):
        from controller_architecture_study.framework_runtime_flight import commit_plan
        for changed in ({"deadline_s":.1},{"model":"different"}):
            waiting = {"b":{"sample_id":"b:0","model":"frozen","deadline_s":.08}}
            plans = {}
            with self.assertRaises(RuntimeError):
                commit_plan({"b":{**waiting["b"],**changed}},plans,waiting)
            self.assertFalse(plans)
            self.assertEqual(waiting["b"]["deadline_s"],.08)

    def test_admission_retry_requires_progress_not_just_elapsed_time(self):
        from controller_architecture_study.framework_runtime_flight import admission_progress
        pending = {"a":{"sample_id":"a:0","status":"submitted","enqueue_s":1.}}
        token = admission_progress(pending)
        pending["a"]["elapsed_s"] = .01
        self.assertEqual(admission_progress(pending),token)
        pending["a"]["worker_enqueue_s"] = 1.005
        self.assertNotEqual(admission_progress(pending),token)


if __name__ == "__main__":
    unittest.main()
