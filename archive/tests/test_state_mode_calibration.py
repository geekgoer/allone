import unittest

import numpy as np

from airsim_interface import RawDroneState
from controller_architecture_study.state_mode_calibration import moderate_case, nominal_case, tail_metrics
from controller_architecture_study.state_mode_probe import risk_features
from controller_architecture_study.fast_convex_preprocess import FastReferenceWindow
from controller_architecture_study.fly_decision_dnn import StoredRoutePlanner
from controller_architecture_study.state_mode_probe import OUTPUT, read_json
from controller_architecture_study.state_mode_capacity_pilot import build_plan, summarize


class CalibrationTests(unittest.TestCase):
    def test_early_recovery_does_not_hide_late_tracking_failure(self):
        trace = [{"t": t, "position_error_m": .1 if t < 1. else .5,
                  "velocity_error_mps": .1} for t in np.arange(0., 3.01, .02)]
        result = tail_metrics(trace, {"window_covered": True, "control_success": True, "joint_success": True})
        self.assertTrue(result["tail_covered"])
        self.assertFalse(result["tail_tracking_passed"])
        self.assertFalse(result["sustained_joint_passed"])

    def test_tail_does_not_override_earlier_safety_or_timing_failure(self):
        trace = [{"t": t, "position_error_m": .1, "velocity_error_mps": .1}
                 for t in np.arange(0., 3.01, .02)]
        result = tail_metrics(trace, {"window_covered": True, "control_success": False, "joint_success": False})
        self.assertTrue(result["tail_tracking_passed"])
        self.assertFalse(result["sustained_control_passed"])
        result = tail_metrics(trace, {"window_covered": True, "control_success": True, "joint_success": False})
        self.assertTrue(result["sustained_control_passed"])
        self.assertFalse(result["sustained_joint_passed"])

    def test_empty_or_incomplete_window_cannot_pass_tail(self):
        report = {"window_covered": False, "control_success": False, "joint_success": False}
        self.assertFalse(tail_metrics([], report)["tail_tracking_passed"])
        trace = [{"t": 2.6, "position_error_m": 0., "velocity_error_mps": 0.},
                 {"t": 3., "position_error_m": 0., "velocity_error_mps": 0.}]
        self.assertFalse(tail_metrics(trace, {**report, "window_covered": True})["tail_tracking_passed"])

    def test_saved_curved_case_is_matched_without_retiming(self):
        case, layout = moderate_case()
        self.assertEqual(case["route_index"], 1)
        planner = StoredRoutePlanner(layout)
        ref = FastReferenceWindow(planner.trajectory, planner.config).window(case["reference_phase_s"])[0]
        state = RawDroneState(np.asarray(case["position"]), np.asarray(case["quaternion"]),
                              np.asarray(case["velocity"]), np.zeros(3), np.zeros(3))
        risk = risk_features(state, ref)
        self.assertAlmostEqual(risk["h_m"], 1.)
        self.assertAlmostEqual(risk["h_dot_mps"], -.4)
        self.assertGreater(np.linalg.norm(ref.velocity), 4.5)
        self.assertGreater(abs(float(np.cross(ref.velocity[:2], ref.acceleration[:2]))), 1.)
        nominal = nominal_case(case, layout)
        np.testing.assert_array_equal(nominal["position"], ref.position)
        np.testing.assert_array_equal(nominal["velocity"], ref.velocity)
        self.assertNotEqual(nominal["id"], case["id"])
        self.assertAlmostEqual(np.linalg.norm(np.asarray(case["position"])-ref.position), .55)

    def test_capacity_expansion_requires_teacher_and_maximum_calibration(self):
        original = read_json(OUTPUT/"setup.json")["plan"]
        with self.assertRaisesRegex(ValueError, "calibration"):
            build_plan(original, {"trials": []})

    def test_capacity_plan_is_balanced_and_includes_the_excluded_cheap_modes(self):
        original = read_json(OUTPUT/"setup.json")["plan"]
        required = ("teacher_moderate_nominal", "teacher_moderate_perturbed", "supernet_max_moderate_perturbed")
        calibration = {"trials": [{"id": i, "supplementary": {"sustained_joint_passed": True}} for i in required]}
        # The live setup is read-only provenance; no simulator or weights are needed.
        from unittest.mock import patch
        case, _ = moderate_case()
        with patch("controller_architecture_study.state_mode_capacity_pilot.read_json",
                   return_value={"jobs": [{"id": required[1], "case": case}]}):
            plan, _ = build_plan(original, calibration)
        self.assertEqual(len(plan["trials"]), 24)
        self.assertEqual(len({t["id"] for t in plan["trials"]}), 24)
        self.assertEqual([m["effective_macs"] for m in plan["modes"]], [30848, 160000, 553216, 19121152])
        self.assertEqual(len(summarize(plan, [])["rows"]), 12)
        for case in plan["cases"]:
            for mode in plan["modes"]:
                self.assertEqual(sum(t["case"] == case["id"] and t["mode"] == mode["name"]
                                     for t in plan["trials"]), 2)


if __name__ == "__main__":
    unittest.main()
