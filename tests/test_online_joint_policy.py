"""Resource policy and replay accounting tests; no trained model/GPU/AirSim."""

from pathlib import Path
import json
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from controller_architecture_study.online_joint_policy import ResourceSelector
from controller_architecture_study import replay_online_joint as replay
from controller_architecture_study.dynamic_probe_common import InferenceService


def profile():
    return {"device": "cpu", "runtime": {"dtype": "float32"}, "summary_sha256": "pilot",
            "quality_scope": "synthetic route prior",
            "initial_service_p95_ms": {"small": 1., "large": 5.},
            "initial_release_jitter_p95_ms": 0.,
            "candidates": [{"case": f"{name}_T{t}", "model": name, "period_ms": float(t),
                            "tracking_mse_m2": error} for name, t, error in
                           [("small", 50, .2), ("small", 100, .3), ("large", 50, .1)]]}


class SelectorTests(unittest.TestCase):
    def test_period_only_never_switches_model(self):
        policy = ResourceSelector(profile())
        self.assertEqual(policy.select(3, 0., [], "period_only")["case"], "large_T50")
        result = policy.select(100, 1., [], "period_only")
        self.assertEqual(result["model"], "large")
        self.assertFalse(result["estimated_feasible"])
        value = profile()
        value["candidates"][0]["tracking_mse_m2"] = .05
        policy = ResourceSelector(value)
        self.assertEqual(policy.select(60, 0., [], "period_only")["case"], "small_T100")
        self.assertEqual(policy.select(3, 3., [], "period_only")["case"], "small_T50")

    def test_stale_model_cost_is_not_silently_treated_as_fresh(self):
        policy = ResourceSelector(profile())
        slow = [{"model": "large", "status": "ready", "service_ms": 20.,
                 "ready_s": 10., "sample_start_s": 9.9, "scheduled_s": 9.9} for _ in range(8)]
        fresh = policy.select(3, 10.1, slow)
        self.assertEqual(fresh["cost_estimate_source"]["large"], "recent_observation")
        stale = policy.select(3, 13., [])
        self.assertEqual(stale["cost_estimate_source"]["large"], "stale_conservative")
        self.assertEqual(stale["service_p95_ms"]["large"], 20.)
        self.assertEqual(stale["cost_estimate_age_seconds"]["large"], 3.)

    def test_causal_resource_feedback_then_recovery(self):
        policy = ResourceSelector(profile())
        self.assertEqual(policy.select(3, 10., [])["model"], "large")
        overloaded = policy.select(16, 10.1, [])
        self.assertEqual(overloaded["model"], "small")
        self.assertTrue(overloaded["estimated_feasible"])
        # Previously feasible small model stays during the dwell interval.
        self.assertEqual(policy.select(3, 10.2, [])["reason"], "minimum_dwell")
        self.assertEqual(policy.select(3, 13., [])["model"], "large")
        slow = [{"model": "large", "status": "ready", "service_ms": 20.,
                 "ready_s": 14., "sample_start_s": 13.9, "scheduled_s": 13.9} for _ in range(8)]
        self.assertEqual(policy.select(3, 13.5, slow)["model"], "large")  # Future records ignored.
        self.assertEqual(policy.select(3, 14.1, slow)["model"], "small")

    def test_small_model_can_remain_best_and_fixed_does_not_adapt(self):
        value = profile()
        value["candidates"][0]["tracking_mse_m2"] = .05
        policy = ResourceSelector(value)
        self.assertEqual(policy.select(3, 0., [])["model"], "small")
        self.assertFalse(policy.select(8, 10., [])["changed"])
        fixed = ResourceSelector(profile()).select(16, 0., [], "fixed")
        self.assertEqual(fixed["model"], "large")
        self.assertFalse(fixed["estimated_feasible"])

    def test_period_changes_and_infeasibility_do_not_create_new_safety_bounds(self):
        policy = ResourceSelector(profile())
        result = policy.select(60, 0., [])
        self.assertEqual(result["period_ms"], 100.)
        self.assertTrue(result["estimated_feasible"])
        result = policy.select(100, 1., [])
        self.assertFalse(result["estimated_feasible"])
        self.assertIsNone(result["state_dependent_T_max"])
        self.assertEqual(result["period_ms"], 100.)
        pending = [{"model": "large"} for _ in range(30)]
        self.assertFalse(policy.select(3, 2., pending)["estimated_feasible"])


class AccountingTests(unittest.TestCase):
    def test_each_request_keeps_its_original_deadline(self):
        rows = [{"status": "ready", "scheduled_s": 0., "sample_start_s": 0.,
                 "enqueue_s": 0., "start_s": .01, "ready_s": .075, "service_ms": 65.,
                 "vehicle": "client0", "model": "small", "period_ms": t, "deadline_ms": t}
                for t in (50., 100.)]
        rows.append({"status": "dropped_busy", "period_ms": 50., "model": "small"})
        result = replay.summarize(rows)
        self.assertEqual(result["released"], 3)
        self.assertEqual(result["timely_outputs"], 1)
        self.assertEqual(result["deadline_misses"], 2)
        self.assertEqual(result["common_response_limit_misses"]["50"], 3)
        self.assertEqual(result["common_response_limit_misses"]["100"], 1)

    def test_age_excludes_late_outputs_and_accounts_initial_unserved_time(self):
        rows = [{"vehicle": "client0", "status": "ready", "sample_start_s": 0.,
                 "scheduled_s": 0., "ready_s": .1, "deadline_ms": 100.},
                {"vehicle": "client0", "status": "ready", "sample_start_s": .2,
                 "scheduled_s": .2, "ready_s": .4, "deadline_ms": 50.}]
        result = replay.output_age(rows, 0., [1], 1., 1.)
        self.assertAlmostEqual(result["unserved_fraction"], .1)
        self.assertAlmostEqual(result["mean_age_when_served_ms"], 550.)
        self.assertAlmostEqual(result["max_age_when_served_ms"], 1000.)
        empty = replay.output_age([], 0., [1], 1., 1.)
        self.assertEqual(empty["unserved_fraction"], 1.)
        self.assertIsNone(empty["mean_age_when_served_ms"])

    def test_departed_client_time_is_not_counted_as_unserved(self):
        intervals = replay.active_intervals(1, [1, 2, 2, 1, 2], 1., 5.)
        self.assertEqual(intervals, [[1., 3.], [4., 5.]])
        result = replay.output_age([], 0., [1, 2, 2, 1, 2], 1., 5.)
        self.assertEqual(result["active_client_seconds"], 8.)

    def test_age_supports_flight_vehicle_names_and_dispatch_timestamps(self):
        rows = [dict(vehicle=name, status="ready", sample_start_s=0., scheduled_s=0.,
                     ready_s=.1, deadline_ms=100.) for name in ("Drone1", "Drone2", "Drone3")]
        result = replay.output_age(rows, 0., [3], 1., 1.,
                                   vehicle_names=("Drone1", "Drone2", "Drone3"))
        self.assertAlmostEqual(result["unserved_fraction"], .1)
        self.assertAlmostEqual(result["mean_age_when_served_ms"], 550.)
        self.assertAlmostEqual(result["max_age_when_served_ms"], 1000.)


class FakePool:
    norm = {"x_mean": np.zeros(237, dtype=np.float32)}
    parameter_bytes = 0

    def __init__(self, *_):
        pass

    def runtime_info(self):
        return {"dtype": "float32"}

    def predict(self, name, feature):
        return np.zeros(4), {"service_ms": .1, "gpu_compute_ms": None}


def args(path):
    return SimpleNamespace(summary=path/"pilot.json", output_dir=path/"suite", device="cpu",
        clients=[1, 2, 1], stage_seconds=.08, rounds=1, decision_seconds=.04,
        utilization_limit=.75, dwell_seconds=.05, resume=False, preflight=False)


class ReplayTests(unittest.TestCase):
    def test_three_arm_order_is_reversed_in_second_round(self):
        self.assertEqual(replay.trial_cases(2), [(1, "fixed"), (1, "period_only"), (1, "adaptive"),
                                               (2, "adaptive"), (2, "period_only"), (2, "fixed")])

    def test_feedback_never_iterates_full_archive_and_remains_bounded(self):
        class Archive(list):
            def __iter__(self):
                raise AssertionError("feedback scanned historical requests")
        service = InferenceService(lambda *_: None, feedback=True)
        try:
            service.rows = Archive([{}]*20000)
            service.recent.extend({"ready_s": 10., "model": "small", "status": "ready",
                "service_ms": 1., "sample_start_s": 9.9, "scheduled_s": 9.9} for _ in range(5000))
            snapshot = service.feedback_snapshot(10.1)
            self.assertEqual(len(snapshot["recent"]), 4096)
            self.assertEqual(snapshot["capacity"], 4096)
            ResourceSelector(profile()).select(3, 10.1, snapshot)
            self.assertEqual(service.feedback_snapshot(13.)["recent"], [])
        finally:
            service.close()

    def test_feedback_tracks_pending_completion_and_inference_error(self):
        release = threading.Event()
        def predictor(name, feature):
            release.wait(2.)
            if name == "bad":
                raise ValueError("test failure")
            return np.zeros(4), {"service_ms": 1.}
        service = InferenceService(predictor, feedback=True)
        try:
            now = time.perf_counter()
            make_row = lambda vehicle: {"vehicle": vehicle, "scheduled_s": now, "sample_start_s": now}
            good = service.submit("small", np.zeros(237), make_row("a"))
            bad = service.submit("bad", np.zeros(237), make_row("b"))
            self.assertEqual(service.feedback_snapshot(now)["pending_model_counts"], {"small": 1, "bad": 1})
            release.set()
            good.result(2.)
            with self.assertRaises(ValueError):
                bad.result(2.)
            snapshot = service.feedback_snapshot(time.perf_counter())
            self.assertEqual(snapshot["pending_model_counts"], {})
            self.assertEqual(len(snapshot["recent"]), 1)
            self.assertIn("test failure", snapshot["inference_error"])
        finally:
            release.set()
            service.close()

    def test_host_stall_does_not_erase_releases_before_client_departure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = args(root)
            config.clients = [2, 1]
            original = ResourceSelector.select
            def delayed(selector, *values, **kwargs):
                if selector.current is None:
                    time.sleep(.12)
                return original(selector, *values, **kwargs)
            with patch.object(ResourceSelector, "select", delayed):
                report = replay.run_arm(config, profile(), FakePool(), np.zeros((20, 237)),
                                        np.arange(20), np.arange(20), "fixed", root/"arm")
            rows = [json.loads(line) for line in (root/"arm"/"requests.jsonl").read_text().splitlines()]
            self.assertGreater(report["timing"]["skipped_release"], 0)
            departed = [r for r in rows if r["vehicle"] == "client1"]
            self.assertEqual(len(departed), 2)  # t=.025 and .075 before departure at .08.
            self.assertTrue(all(r["stage"] == 0 for r in departed))

    def test_actual_queue_replay_resume_and_no_teacher_label_dependency(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Deliberately no U array: the runtime selector must not access labels.
            np.savez(root/"validation.npz", X=np.zeros((20, 237), dtype=np.float32), row_id=np.arange(20))
            config = args(root)
            with patch.object(replay, "DATA", root), patch.object(replay, "load_profile", return_value=profile()), \
                    patch.object(replay, "ModelPool", FakePool):
                self.assertEqual(replay.run(config), 0)
                suite = replay.read_json(config.output_dir/"summary.json")
                self.assertEqual(suite["status"], "COMPLETE")
                self.assertEqual(len(suite["trials"]), 3)
                for trial in suite["trials"]:
                    report = replay.read_json(Path(trial["report"]))
                    self.assertGreater(report["timing"]["released"], 0)
                    self.assertEqual(report["timing"]["inference_errors"], 0)
                    self.assertEqual(report["timing"]["released"], sum(s["released"] for s in report["per_stage"]))
                    self.assertFalse(report["teacher_labels_read"])
                config.resume = True
                with patch.object(replay, "ModelPool") as pool:
                    self.assertEqual(replay.run(config), 0)
                    pool.assert_not_called()
                config.clients = [2, 1]
                with self.assertRaisesRegex(ValueError, "resume inputs/code"):
                    replay.run(config)

    def test_interrupt_report_retained_and_no_extra_arm_started(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez(root/"validation.npz", X=np.zeros((20, 237), dtype=np.float32), row_id=np.arange(20))
            config = args(root)
            original = ResourceSelector.select
            def interrupted(selector, *values, **kwargs):
                if selector.current is not None:
                    raise KeyboardInterrupt()
                return original(selector, *values, **kwargs)
            with patch.object(replay, "DATA", root), patch.object(replay, "load_profile", return_value=profile()), \
                    patch.object(replay, "ModelPool", FakePool):
                with patch.object(ResourceSelector, "select", interrupted):
                    with self.assertRaisesRegex(RuntimeError, "KeyboardInterrupt"):
                        replay.run(config)
                suite = replay.read_json(config.output_dir/"summary.json")
                self.assertEqual(suite["trials"], [])
                first = config.output_dir/"round1_fixed"/"attempt_001"/"report.json"
                self.assertEqual(replay.read_json(first)["status"], "INCOMPLETE")
                config.resume = True
                self.assertEqual(replay.run(config), 0)
                self.assertTrue(first.is_file())
                self.assertTrue((first.parent.parent/"attempt_002"/"report.json").is_file())


if __name__ == "__main__":
    unittest.main()
