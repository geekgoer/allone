import unittest
from unittest.mock import patch

import numpy as np

from airsim_interface import RawDroneState
from controller_architecture_study.state_mode_generalization import build_plan, CONTROLLERS, PERIODS_MS
from controller_architecture_study.state_mode_probe import risk_features
from controller_architecture_study.fly_decision_dnn import StoredRoutePlanner
from controller_architecture_study.fast_convex_preprocess import FastReferenceWindow


class GeneralizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan, cls.layouts = build_plan()

    def test_fixed_size_and_balanced_repeats(self):
        self.assertEqual(len(self.plan["trials"]), 111)
        self.assertEqual(len({t["id"] for t in self.plan["trials"]}), 111)
        for case in self.plan["cases"]:
            trials = [t for t in self.plan["trials"] if t["case"] == case["id"]]
            if case["kind"] == "nominal":
                self.assertEqual([t["mode"] for t in trials], ["QP_teacher"])
            else:
                for mode in CONTROLLERS:
                    self.assertEqual(sum(t["mode"] == mode for t in trials), 2)

    def test_period_only_changes_timing_and_pose_pair_preserves_risk_coordinates(self):
        for context in self.plan["contexts"]:
            cases = [c for c in self.plan["cases"] if c["context"] == context["id"] and c["kind"] == "recovery"]
            self.assertEqual({c["period_ms"] for c in cases}, set(PERIODS_MS))
            self.assertNotEqual(context["environment"], 48)
            planner = StoredRoutePlanner(self.layouts[context["id"]])
            ref = FastReferenceWindow(planner.trajectory, planner.config).window(context["phase_s"])[0]
            for case in cases:
                for key in ("position", "velocity", "held_control", "reference_phase_s", "route_sha256"):
                    self.assertEqual(case[key], cases[0][key])
                same_profile = [c for c in cases if c["profile"] == case["profile"]]
                self.assertTrue(all(c["quaternion"] == case["quaternion"] for c in same_profile))
                state = RawDroneState(case["position"], case["quaternion"], case["velocity"], np.zeros(3), np.zeros(3))
                risk = risk_features(state, ref)
                self.assertAlmostEqual(risk["h_m"], 1.)
                self.assertAlmostEqual(risk["h_dot_mps"], -.4)
            self.assertFalse(np.allclose(cases[0]["quaternion"], cases[-1]["quaternion"]))

    def test_test_environment_rejected_before_loading_any_route(self):
        from controller_architecture_study import state_mode_generalization as experiment
        with patch.object(experiment, "CONTEXTS", ((54, 1, .3),)), patch.object(experiment, "load_layout") as loader:
            with self.assertRaisesRegex(ValueError, "validation environments only"):
                experiment.build_plan()
            loader.assert_not_called()

    def test_fast_teacher_diagnostic_changes_period_not_initial_conditions(self):
        from controller_architecture_study.state_mode_fast_teacher import build_plan as fast_plan
        plan, _ = fast_plan()
        self.assertEqual(len(plan["trials"]), 6)
        self.assertEqual({t["mode"] for t in plan["trials"]}, {"QP_teacher"})
        for case in plan["cases"]:
            original = next(c for c in self.plan["cases"] if c["kind"] == "nominal" and c["context"] == case["context"])
            self.assertEqual(case["period_ms"], 25.)
            for key in ("position", "velocity", "quaternion", "held_control", "reference_phase_s", "route_sha256"):
                self.assertEqual(case[key], original[key])


if __name__ == "__main__":
    unittest.main()
