"""PPT budget units, next-response causality and coordinate-frame regression."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from environment import Gate
from controller_architecture_study.state_budget_shadow import (
    DEFAULT_CONFIG, analyze, gate_budget, load_config, save_analysis,
)


GEOMETRY = {"width": 4., "height": 4., "thickness": .15, "clearance": .25}


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(DEFAULT_CONFIG)
        self.gate = Gate(1, [0., 0., 0.], [1., 0., 0., 0.])

    def fixture(self, periods=(50., 50., 50.), responses=(.01, .08, .01)):
        samples = [0., .05, .1]
        traces = {"Drone1": [dict(sample_s=t, position=[-.5, 1.45, 0.], velocity=[5., 0., 0.]) for t in samples]}
        rows = [dict(vehicle="Drone1", cycle=k, sample_start_s=t, model="small", period_ms=periods[k],
                     dnn_dispatch_s=t+responses[k]) for k, t in enumerate(samples)]
        gates = {"Drone1": [dict(index=1, center=[0., 0., 0.], orientation=[1., 0., 0., 0.])]}
        return traces, rows, gates

    def test_opening_and_rate_are_in_gate_axes_not_forward_speed(self):
        value = gate_budget([-.5, 1.45, 0.], [13.5, 0., 0.], [self.gate], GEOMETRY, self.config)
        self.assertAlmostEqual(value["validity_budget_ms"], 75.)
        self.assertFalse(value["observed_rate_bound_exceeded"])
        self.assertEqual(value["constraints"][0]["limiting_axis"], "lateral")
        slow = gate_budget([-.5, 1.45, 0.], [2., 0., 0.], [self.gate], GEOMETRY, self.config)
        self.assertEqual(value["validity_budget_ms"], slow["validity_budget_ms"])

    def test_rotated_gate_and_translation_leave_budget_unchanged(self):
        # Gate-local x maps to world y and local y to world -x.
        q = [np.sqrt(.5), 0., 0., np.sqrt(.5)]
        rotated = Gate(1, [10., 20., -4.], q)
        value = gate_budget([8.55, 19.5, -4.], [0., 5., 0.], [rotated], GEOMETRY, self.config)
        self.assertAlmostEqual(value["validity_budget_ms"], 75.)
        self.assertFalse(value["observed_rate_bound_exceeded"])

    def test_far_gate_is_not_applicable_not_a_safety_pass(self):
        value = gate_budget([-16., 3., 0.], [5., 0., 0.], [self.gate], GEOMETRY, self.config)
        self.assertFalse(value["applicable"])
        self.assertIsNone(value["validity_budget_ms"])

    def test_negative_budget_is_not_clamped_and_rate_exceedance_is_visible(self):
        traces, rows, gates = self.fixture()
        traces["Drone1"][0].update(position=[0., 2., 0.], velocity=[5., 3.1, 0.])
        summary, states = analyze(traces, rows, gates, GEOMETRY, self.config)
        self.assertLess(states[0]["T_max_estimate_ms"], 0.)
        self.assertEqual(states[0]["feasible_periods_same_response_ms"], [])
        self.assertTrue(states[0]["outside_inner_opening"])
        self.assertEqual(summary["observed_rate_bound_exceedances"], 1)
        self.assertFalse(summary["is_safety_certificate"])

    def test_next_response_not_current_response_and_no_future_leakage(self):
        traces, rows, gates = self.fixture()
        summary, states = analyze(traces, rows, gates, GEOMETRY, self.config)
        first = states[0]
        self.assertEqual(first["response_history_count"], 0)
        self.assertEqual(first["next_response_estimate_ms"], 30.)
        self.assertAlmostEqual(first["T_max_estimate_ms"], 45.)
        self.assertAlmostEqual(first["actual_age_at_next_dispatch_ms"], 130.)
        self.assertTrue(first["posthoc_budget_exceeded"])
        self.assertTrue(first["next_response_exceeded_estimate"])
        self.assertEqual(states[2]["response_history_count"], 1)  # .13s dispatch is still in the future at .1s.
        changed = deepcopy(rows)
        changed[1]["dnn_dispatch_s"] = .09
        _, alternative = analyze(traces, changed, gates, GEOMETRY, self.config)
        self.assertEqual(first["next_response_estimate_ms"], alternative[0]["next_response_estimate_ms"])
        self.assertEqual(summary["posthoc_comparisons"], 2)
        self.assertIsNone(states[-1]["posthoc_budget_exceeded"])

    def test_actual_sample_interval_is_used_even_after_period_switch_or_skip(self):
        traces, rows, gates = self.fixture(periods=(50., 100., 75.))
        rows[1]["cycle"] = 2
        rows[2]["cycle"] = 3
        _, states = analyze(traces, rows, gates, GEOMETRY, self.config)
        self.assertAlmostEqual(states[0]["actual_next_sample_interval_ms"], 50.)
        self.assertAlmostEqual(states[1]["actual_next_sample_interval_ms"], 50.)
        self.assertFalse(states[0]["next_cycle_contiguous"])

    def test_history_includes_other_clients_but_only_prior_dispatches(self):
        traces, rows, gates = self.fixture()
        rows.append(dict(vehicle="Drone2", cycle=0, sample_start_s=-.05,
                         dnn_dispatch_s=-.01, period_ms=50., model="other"))
        _, states = analyze(traces, rows, gates, GEOMETRY, self.config)
        self.assertAlmostEqual(states[0]["next_response_estimate_ms"], 45.)
        self.assertEqual(states[0]["response_history_count"], 1)

    def test_falsified_rate_assumption_does_not_offer_feasible_periods(self):
        traces, rows, gates = self.fixture()
        traces["Drone1"][0].update(position=[0., 0., 0.], velocity=[0., 3.1, 0.])
        _, states = analyze(traces, rows, gates, GEOMETRY, self.config)
        self.assertGreater(states[0]["T_max_estimate_ms"], 100.)
        self.assertFalse(states[0]["estimate_usable_at_sample"])
        self.assertIsNone(states[0]["nominal_period_within_estimated_budget"])
        self.assertEqual(states[0]["feasible_periods_same_response_ms"], [])

    def test_missing_next_dispatch_and_no_gate_samples_are_not_passes(self):
        traces, rows, gates = self.fixture()
        del rows[1]["dnn_dispatch_s"]
        _, states = analyze(traces, rows, gates, GEOMETRY, self.config)
        self.assertIsNone(states[0]["posthoc_budget_exceeded"])
        for sample in traces["Drone1"]:
            sample["position"][0] = -16.
        summary, states = analyze(traces, rows, gates, GEOMETRY, self.config)
        self.assertEqual(summary["applicable_samples"], 0)
        self.assertIsNone(summary["T_max_estimate_ms"])
        self.assertTrue(all(r["nominal_period_within_estimated_budget"] is None for r in states))

    def test_json_outputs_preserve_source_objects_and_config_checks(self):
        traces, rows, gates = self.fixture()
        original = deepcopy((traces, rows, gates))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            save_analysis(path, traces, rows, gates, GEOMETRY, self.config)
            report = json.loads((path/"state_budget_report.json").read_text())
            self.assertFalse(report["control_changed"])
            for field, value in (("lateral_error_rate_bound_mps", 0.), ("response_margin_ms", -1.),
                                 ("response_history_size", 0), ("candidate_periods_ms", [])):
                config = {**self.config, field: value}
                (path/"config.json").write_text(json.dumps(config))
                with self.assertRaises(ValueError):
                    load_config(path/"config.json")
        self.assertEqual((traces, rows, gates), original)
        with self.assertRaises(ValueError):
            analyze(traces, rows+[dict(rows[0])], gates, GEOMETRY, self.config)


if __name__ == "__main__":
    unittest.main()
