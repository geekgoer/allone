from copy import deepcopy
import unittest

import numpy as np

from airsim_interface import RawDroneState
from state_adapter import quaternion_inverse, quaternion_multiply, quaternion_to_rotation_vector
from controller_architecture_study.state_mode_boundary import CONTROLLERS, PROFILES, BoundaryPool, build_cases, confirmation_cases
from controller_architecture_study.state_mode_probe import risk_features
from controller_architecture_study.fast_convex_preprocess import FastReferenceWindow
from controller_architecture_study.fly_decision_dnn import StoredRoutePlanner
from controller_architecture_study.analyze_state_mode_boundary import trajectory_metrics, paired_outcomes, exact_tail_sensitivity
from controller_architecture_study.state_mode_period_boundary import build_plan as period_plan
from controller_architecture_study.state_mode_period_refinement import select_midpoints


def discovery_fixture():
    groups, trials = {}, []
    for name, *_ in PROFILES:
        groups[name] = {m: {"completed": 1, "control": 1, "sustained": 1, "joint": 1,
            "timing_invalid": 0, "errors": []} for m in CONTROLLERS}
        for m in CONTROLLERS:
            trials.append({"report": {"trial": {"case": name, "mode": m}},
                "supplementary": {"tail_max_position_error_m": .1, "tail_max_velocity_error_mps": .2}})
    return {"trials": trials, "by_case": groups}


class BoundaryTests(unittest.TestCase):
    def test_cases_preserve_matched_physical_margins_and_requested_rotations(self):
        cases, layout = build_cases()
        self.assertEqual(len(cases), 13)
        self.assertEqual(len({c["id"] for c in cases}), 13)
        planner = StoredRoutePlanner(layout)
        ref = FastReferenceWindow(planner.trajectory, planner.config).window(cases[0]["reference_phase_s"])[0]
        for case in cases:
            state = RawDroneState(case["position"], case["quaternion"], case["velocity"], np.zeros(3), np.zeros(3))
            risk = risk_features(state, ref)
            self.assertAlmostEqual(risk["h_m"], case["requested_h_m"])
            self.assertAlmostEqual(risk["h_dot_mps"], case["requested_h_dot_mps"])
            actual = quaternion_to_rotation_vector(quaternion_multiply(quaternion_inverse(ref.orientation), state.quaternion))
            np.testing.assert_allclose(actual, np.radians(case["body_rotation_error_deg"]), atol=1e-12)
            self.assertLess(np.linalg.norm(state.velocity), 8.)

    def test_no_split_confirmation_is_bounded_and_does_not_invent_a_split(self):
        selected = confirmation_cases(discovery_fixture())
        self.assertEqual(selected[0], "baseline")
        self.assertEqual(len(selected), 2)

    def test_both_split_directions_are_retained_and_timing_invalid_is_excluded(self):
        data = discovery_fixture()
        data["by_case"]["roll_plus30"][CONTROLLERS[1]].update(sustained=0, joint=0)
        data["by_case"]["roll_minus30"][CONTROLLERS[2]].update(sustained=0, joint=0)
        data["by_case"]["outward_v2"][CONTROLLERS[1]].update(sustained=0, joint=0, timing_invalid=1)
        selected = confirmation_cases(data)
        self.assertIn("roll_plus30", selected)
        self.assertIn("roll_minus30", selected)
        self.assertNotIn("outward_v2", selected)
        self.assertLessEqual(len(selected), 4)

    def test_teacher_failure_is_not_neural_capacity_evidence(self):
        data = discovery_fixture()
        data["by_case"]["roll_plus30"][CONTROLLERS[0]].update(sustained=0, joint=0)
        data["by_case"]["roll_plus30"][CONTROLLERS[1]].update(sustained=0, joint=0)
        self.assertNotIn("roll_plus30", confirmation_cases(data))
        incomplete = deepcopy(data)
        incomplete["trials"].pop()
        with self.assertRaisesRegex(ValueError, "entire"):
            confirmation_cases(incomplete)

    def test_partial_trace_rmse_is_missing_not_a_short_window_average(self):
        trace = [{"t": float(t), "position_error_m": 2., "velocity_error_mps": 3.}
                 for t in np.linspace(0., 3., 151)]
        metrics = trajectory_metrics(trace)
        self.assertEqual(metrics["position_rmse_m"], 2.)
        self.assertEqual(metrics["velocity_rmse_mps"], 3.)
        self.assertIsNone(trajectory_metrics(trace[:-1])["position_rmse_m"])
        self.assertFalse(trajectory_metrics([])["full_window_rmse_available"])

    def test_pair_does_not_count_a_timing_invalid_split_as_capacity_evidence(self):
        rows = []
        for mode in CONTROLLERS:
            rows.append({"report": {"trial": {"case": "test", "repeat": 0, "mode": mode},
                "timing_comparable": True}, "supplementary": {"sustained_control_passed": mode != CONTROLLERS[1]}})
        self.assertTrue(paired_outcomes({"trials": rows})[0]["large_only_given_teacher_and_timing"])
        rows[1]["report"]["timing_comparable"] = False
        self.assertFalse(paired_outcomes({"trials": rows})[0]["large_only_given_teacher_and_timing"])

    def test_period_grid_changes_timing_not_physical_state_or_reference(self):
        plan, _ = period_plan()
        self.assertEqual(len(plan["trials"]), 54)
        self.assertEqual(len({t["id"] for t in plan["trials"]}), 54)
        self.assertEqual({c["period_ms"] for c in plan["cases"]}, {100., 150., 200.})
        for source in ("baseline", "outward_v2", "roll_minus30"):
            cases = [c for c in plan["cases"] if c["source_case"] == source]
            self.assertEqual(len(cases), 3)
            for key in ("position", "velocity", "quaternion", "reference_phase_s", "held_control", "route_sha256"):
                self.assertTrue(all(c[key] == cases[0][key] for c in cases))
        for c in plan["cases"]:
            for m in CONTROLLERS:
                self.assertEqual(sum(t["case"] == c["id"] and t["mode"] == m for t in plan["trials"]), 2)

    def test_refinement_uses_teacher_bracket_once_and_rejects_ambiguous_cells(self):
        at50 = discovery_fixture()
        coarse = {"by_case": {}}
        for state in ("baseline", "outward_v2", "roll_minus30"):
            for t in (100., 150., 200.):
                coarse["by_case"][f"{state}_T{t:g}"] = {
                    "QP_teacher": {"completed": 2, "sustained": 2 if t == 100. else 0, "timing_invalid": 0}}
        brackets, skipped = select_midpoints(coarse, at50)
        self.assertEqual([b["period_ms"] for b in brackets], [125., 125., 125.])
        self.assertFalse(skipped)
        coarse["by_case"]["baseline_T100"]["QP_teacher"]["timing_invalid"] = 1
        coarse["by_case"]["outward_v2_T150"]["QP_teacher"]["sustained"] = 1
        brackets, skipped = select_midpoints(coarse, at50)
        self.assertEqual(len(brackets), 1)
        self.assertEqual(set(skipped), {"baseline", "outward_v2"})

    def test_tail_sensitivity_clips_observations_beyond_the_fixed_window(self):
        trace = [{"t": t, "position_error_m": p, "velocity_error_mps": .1}
                 for t, p in ((0., .1), (2.49, .1), (2.6, .1), (3., .1), (3.02, .4))]
        self.assertTrue(exact_tail_sensitivity(trace)["exact_tail_passed"])
        self.assertEqual(exact_tail_sensitivity(trace)["exact_tail_max_position_m"], .1)
        self.assertFalse(exact_tail_sensitivity(trace[:-2])["exact_tail_available"])

    def test_neural_control_does_not_query_the_teacher_or_fallback(self):
        from unittest.mock import Mock
        pool = object.__new__(BoundaryPool)
        pool.network, pool.teacher = Mock(), Mock()
        pool.network.predict.return_value = ([1., 2., 3., 4.], {"inference_ms": 1.})
        pool.predict(CONTROLLERS[1], [0.]*237)
        pool.network.predict.assert_called_once()
        pool.teacher.predict.assert_not_called()
        pool.predict(CONTROLLERS[0], [0.]*237)
        pool.teacher.predict.assert_called_once()


if __name__ == "__main__":
    unittest.main()
