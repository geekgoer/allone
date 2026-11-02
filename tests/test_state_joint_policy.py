"""CPU-only joint feasibility and causal shadow recommendations."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from controller_architecture_study.state_budget_shadow import DEFAULT_CONFIG, load_config
from controller_architecture_study.state_joint_policy import StateResourceSelector
from controller_architecture_study.joint_selection_shadow import replay


def profile():
    return {"runtime": {"dtype": "float32"}, "provenance": {}, "summary_sha256": "pilot",
        "quality_scope": "synthetic frozen route cost", "initial_producer_p95_ms": 3.,
        "initial_dispatch_p95_ms": 1., "initial_service_p95_ms": {"small": 1., "large": 8.},
        "initial_release_jitter_p95_ms": 0.,
        "candidates": [{"case": f"{m}_T{t}", "model": m, "period_ms": float(t), "tracking_mse_m2": e}
                       for m, t, e in [("small", 50, .2), ("small", 75, .3), ("large", 50, .1), ("large", 75, .12)]]}


def config():
    return {**load_config(DEFAULT_CONFIG), "response_floor_ms": 1., "response_margin_ms": 0.}


def state(budget=100., applicable=True, invalid=False):
    return {"applicable": applicable, "validity_budget_ms": budget if applicable else None,
            "observed_rate_bound_exceeded": invalid}


class JointSelectorTests(unittest.TestCase):
    def test_different_models_have_different_end_to_end_budget(self):
        value = StateResourceSelector(profile(), config()).select(3, 0., [], state(60.))
        self.assertEqual(value["status"], "AVAILABLE")
        self.assertEqual(value["recommendation"]["case"], "small_T50")
        choices = {c["case"]: c for c in value["candidate_options"]}
        self.assertEqual(choices["small_T50"]["estimated_sample_response_ms"], 5.)
        self.assertEqual(choices["large_T50"]["estimated_sample_response_ms"], 12.)
        self.assertFalse(choices["large_T50"]["state_feasible"])
        self.assertTrue(choices["large_T50"]["resource_feasible"])
        self.assertFalse(value["is_safety_certificate"])

    def test_dwell_never_keeps_an_infeasible_current_candidate(self):
        policy = StateResourceSelector(profile(), config(), dwell_seconds=100.)
        self.assertEqual(policy.select(3, 0., [], state())["recommendation"]["model"], "large")
        tightened = policy.select(3, .1, [], state(60.))
        self.assertEqual(tightened["recommendation"]["model"], "small")
        self.assertTrue(tightened["changed"])
        self.assertEqual(policy.select(3, .2, [], state())["reason"], "minimum_dwell")

    def test_exhaustion_invalid_assumption_and_not_applicable_are_explicit(self):
        for input_state, clients, expected in [(state(-10.), 3, "BUDGET_EXHAUSTED"),
                (state(), 100, "RESOURCE_EXHAUSTED"), (state(invalid=True), 3, "ASSUMPTION_INVALID")]:
            result = StateResourceSelector(profile(), config()).select(clients, 0., [], input_state)
            self.assertEqual(result["status"], expected)
            self.assertIsNone(result["recommendation"])
        result = StateResourceSelector(profile(), config()).select(3, 0., [], state(applicable=False))
        self.assertEqual(result["status"], "STATE_NOT_APPLICABLE")
        self.assertIsNotNone(result["recommendation"])
        self.assertTrue(all(c["state_feasible"] is None for c in result["candidate_options"]))

    def test_fixed_and_period_only_do_not_silently_switch_model(self):
        fixed = StateResourceSelector(profile(), config()).select(3, 0., [], state(60.), mode="fixed")
        self.assertIsNone(fixed["recommendation"])
        period = StateResourceSelector(profile(), config()).select(3, 0., [], state(), mode="period_only")
        self.assertEqual(period["recommendation"]["model"], "large")

    def test_pending_work_and_future_records_do_not_use_future_labels(self):
        policy = StateResourceSelector(profile(), config())
        future = [dict(model="large", status="ready", service_ms=100., ready_s=10.,
                       sample_start_s=9., scheduled_s=9.) for _ in range(8)]
        history = [dict(sample_start_s=8., enqueue_s=8.1, ready_s=8.2, dnn_dispatch_s=10.)]
        value = policy.select(3, 1., future, state(), dispatch_history=history)
        self.assertEqual(value["completed_response_reserve_ms"], 1.)
        self.assertEqual(value["service_p95_ms"]["large"], 8.)
        overloaded = policy.select(3, 2., {"recent": [], "pending_model_counts": {"large": 10}}, state())
        self.assertEqual(overloaded["status"], "RESOURCE_EXHAUSTED")
        self.assertIsNone(overloaded["recommendation"])

    def test_next_release_must_not_have_elapsed_and_jitter_consumes_budget(self):
        value = profile()
        value["initial_release_jitter_p95_ms"] = 3.
        result = StateResourceSelector(value, config()).select(3, 0., [], state(), release_elapsed_ms=60.)
        choices = {c["case"]: c for c in result["candidate_options"]}
        self.assertFalse(choices["small_T50"]["next_release_not_elapsed"])
        self.assertEqual(result["recommendation"]["period_ms"], 75.)
        self.assertEqual(choices["small_T50"]["estimated_nominal_T_max_ms"], 92.)

    def test_selection_charge_preserves_fixed_and_period_only_candidate_pools(self):
        for mode in ("fixed", "period_only", "adaptive"):
            policy = StateResourceSelector(profile(), config())
            base = policy.select(3, 0., [], state(applicable=False), mode=mode,
                                 committed_period_ms=50., commit=False)
            charged = policy.charge_selection(base, .044, selection_tail_ms=44.,
                release_elapsed_ms=44., committed_period_ms=50.)
            self.assertEqual([c["case"] for c in base["candidate_options"]],
                             [c["case"] for c in charged["candidate_options"]])
            self.assertIsNone(policy.current)
            if mode == "fixed":
                self.assertIsNone(charged["recommendation"])
            else:
                self.assertEqual(charged["recommendation"]["case"], "large_T75")
            # Charging the original observation again must not mutate or double-charge it.
            repeat = policy.charge_selection(base, .044, selection_tail_ms=44.,
                release_elapsed_ms=44., committed_period_ms=50.)
            self.assertEqual(charged, repeat)
            for original, adjusted in zip(base["candidate_options"], charged["candidate_options"]):
                self.assertEqual(adjusted["estimated_release_response_ms"],
                                 original["estimated_release_response_ms"]+44.)


class ShadowReplayTests(unittest.TestCase):
    def fixture(self):
        source = {"mode": "dnn", "runtime": profile()["runtime"], "provenance": {}}
        traces = {"Drone1": [dict(sample_s=t, position=[0., 0., 0.], velocity=[5., 0., 0.]) for t in (0., .05)]}
        requests = [dict(vehicle="Drone1", cycle=i, model="small", period_ms=50.,
            scheduled_s=t, sample_start_s=t, feature_ready_s=t+.001, enqueue_s=t+.002,
            start_s=t+.003, ready_s=t+.007, dnn_dispatch_s=t+.008, status="ready", service_ms=4.,
            output=[999., 999., 999., 999.]) for i, t in enumerate((0., .05))]
        gates = {"Drone1": [dict(index=1, center=[0., 0., 0.], orientation=[1., 0., 0., 0.])]}
        geometry = {"width": 4., "height": 4., "thickness": .15, "clearance": .25}
        return source, requests, traces, gates, geometry

    def test_completed_events_never_replace_the_current_sample_request(self):
        source, requests, traces, gates, geometry = self.fixture()
        other = dict(requests[0], vehicle="Drone2", model="large", period_ms=75.,
                     scheduled_s=.02, sample_start_s=.02, feature_ready_s=.021,
                     enqueue_s=.022, start_s=.023, ready_s=.027, dnn_dispatch_s=.028)
        requests.insert(1, other)
        traces["Drone2"] = [dict(sample_s=.02, position=[0., 0., 0.], velocity=[5., 0., 0.])]
        gates["Drone2"] = deepcopy(gates["Drone1"])
        original = deepcopy(requests)
        monitor = replay(profile(), config(), source, requests, traces, gates, geometry)
        self.assertEqual([(d["vehicle"], d["cycle"]) for d in monitor.decisions],
                         [("Drone1", 0), ("Drone2", 0), ("Drone1", 1)])
        for decision, row in zip(monitor.decisions, requests):
            self.assertEqual(decision["sample_s"], row["sample_start_s"])
            self.assertEqual(decision["scheduled_s"], row["scheduled_s"])
            self.assertEqual(decision["decision_s"], row["feature_ready_s"])
            self.assertEqual(decision["actual_model"], row["model"])
            self.assertEqual(decision["actual_period_ms"], row["period_ms"])
        self.assertEqual(requests, original)
        self.assertEqual(monitor.alignment, {"status": "PASS", "expected_samples": 3,
            "matched_samples": 3, "unique_vehicle_cycle_pairs": 3, "mismatches": 0})

    def test_duplicate_vehicle_cycle_is_not_a_valid_one_to_one_replay(self):
        values = self.fixture()
        values[1][1]["cycle"] = 0
        with self.assertRaisesRegex(ValueError, "request alignment"):
            replay(profile(), config(), *values)

    def test_future_completion_and_control_labels_do_not_change_prior_decisions(self):
        values = self.fixture()
        original = deepcopy(values)
        monitor = replay(profile(), config(), *values)
        values[1][1].update(ready_s=10., dnn_dispatch_s=11., service_ms=1000., output=[-999.]*4)
        alternate = replay(profile(), config(), *values)
        for before, after in zip(monitor.decisions, alternate.decisions):
            self.assertEqual(before["policies"], after["policies"])
        self.assertEqual(values[1][0], original[1][0])
        self.assertTrue(all(not r["recommendations_applied"] for r in monitor.decisions))
        with tempfile.TemporaryDirectory() as directory:
            report = monitor.save(Path(directory), execution_mode="causal_replay", source_status="INCOMPLETE")
            self.assertEqual(report["status"], "SOURCE_INCOMPLETE")
            self.assertFalse(report["control_changed"])
            rows = (Path(directory)/"joint_shadow_decisions.jsonl").read_text().splitlines()
            self.assertEqual(len(rows), 2)
            self.assertEqual(set(json.loads(rows[0])["policies"]), {"fixed", "period_only", "adaptive"})

    def test_changed_runtime_provenance_and_bad_timing_are_rejected(self):
        for field in ("runtime", "provenance"):
            values = self.fixture()
            values[0][field] = {"different": True}
            with self.assertRaises(ValueError):
                replay(profile(), config(), *values)
        values = self.fixture()
        values[1][0]["feature_ready_s"] = .02
        with self.assertRaises(ValueError):
            replay(profile(), config(), *values)

    def test_captured_abort_is_replayed_without_becoming_a_dispatched_control(self):
        source, requests, traces, gates, geometry = self.fixture()
        for row in requests:
            row["joint_shadow_state"] = {"position": [0., 0., 0.], "velocity": [5., 0., 0.]}
            row["joint_shadow_snapshot_s"] = row["sample_start_s"]+.0015
        requests[0].pop("dnn_dispatch_s")
        requests[0].update(ready_s=.1, service_ms=97., control_outcome="deadline_abort")
        monitor = replay(profile(), config(), source, requests, {"Drone1": []}, gates, geometry)
        self.assertEqual(len(monitor.decisions), 2)
        self.assertAlmostEqual(monitor.decisions[1]["decision_s"], .0515)
        policy = monitor.decisions[1]["policies"]["adaptive"]
        self.assertEqual(policy["pending_service_estimate_ms"], 1.)
        self.assertEqual(policy["completed_response_reserve_ms"], 1.)
        self.assertEqual(len(monitor.dispatches), 0)
        self.assertNotIn("dnn_dispatch_s", requests[0])

    def test_capture_time_controls_which_completions_are_known(self):
        source, requests, traces, gates, geometry = self.fixture()
        requests[0].update(ready_s=.052, dnn_dispatch_s=.06, service_ms=49.)
        requests[1].update(joint_shadow_state={"position": [0., 0., 0.], "velocity": [5., 0., 0.]},
                           joint_shadow_snapshot_s=.0525, enqueue_s=.053, start_s=.054)
        monitor = replay(profile(), config(), source, requests, traces, gates, geometry)
        policy = monitor.decisions[1]["policies"]["adaptive"]
        self.assertEqual(policy["pending_service_estimate_ms"], 0.)
        self.assertEqual(policy["completed_response_reserve_ms"], 1.)
        requests[1]["joint_shadow_snapshot_s"] = .054
        with self.assertRaisesRegex(ValueError, "capture timing"):
            replay(profile(), config(), source, requests, traces, gates, geometry)


if __name__ == "__main__":
    unittest.main()
