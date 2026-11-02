"""CPU-only joint model/period runner tests; no real simulator or GPU."""

from copy import deepcopy
from pathlib import Path
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from controller_architecture_study import joint_controller_sweep as joint


def evidence():
    return {"input_contract": "237D", "normalization_sha256": "norm",
            "validation_sha256": "validation", "models": {
                name: {"checkpoint_sha256": name, "config_sha256": "config"}
                for name in joint.DEFAULT_MODELS}}


def sources():
    return {name: joint.sha256(Path(joint.__file__).with_name(name)) for name in
            ("probe_airsim_three.py", "dynamic_probe_common.py", "fast_convex_preprocess.py")}


def summary():
    return {"status": "INCOMPLETE", "config": {"models": ["plain4x256"],
            "periods_ms": [50., 75.], "device": "cpu", "phase": "staggered",
            "environment_id": 48, "route_index": 0, "host": "localhost", "port": 41451},
            "source_sha256": sources(), "saved_route_sha256": "route", "provenance": evidence(),
            "route_duration_seconds": .2, "performance_proxy": {"unit": "m^2"}, "trials": []}


def write_flight(folder, suite, model="plain4x256", period=50., error=None):
    folder.mkdir(parents=True)
    trace = {v: [{"sample_s": 1.+t, "tracking_error_m": .2} for t in (0., .1, .2, .3)]
             for v in joint.VEHICLES}
    requests = [{"vehicle": v, "cycle": 0, "scheduled_s": 1.+i*period/3000}
                for i, v in enumerate(joint.VEHICLES)]
    report = {"mode": "dnn", "device": "cpu", "phase": "staggered", "environment_id": 48,
              "route_index": 0, "preprocess_policy": "vectorized", "saved_route_sha256": "route",
              "preprocess_source_sha256": suite["source_sha256"]["fast_convex_preprocess.py"],
              "model_selection": model, "period_ms": period, "deadline_ms": period,
              "resident_models": list(joint.DEFAULT_MODELS), "provenance": evidence(),
              "runtime": {"dtype": "float32"}, "acceptance_passed": error is None,
              "flight_passed": error is None, "realtime_passed": error is None,
              "status": "COMPLETE" if error is None else "INCOMPLETE",
              "errors": [] if error is None else [error],
              "flight": {v: {"crossings": 5, "sim_seconds_per_host_second": 1.} for v in joint.VEHICLES},
              "timing": {"release_to_output_ms": {"p99": 10., "max_observed": 12.},
                  "service_ms": {"mean": 2.}, "released": 3, "ready": 3, "dnn_commands_dispatched": 3,
                  "output_deadline_misses": int(error is not None), "control_deadline_aborts": int(error is not None)}}
    joint.atomic_json(folder / "report.json", report)
    joint.atomic_json(folder / "flight_trace.json", trace)
    (folder / "requests.jsonl").write_text("\n".join(json.dumps(r) for r in requests), encoding="utf-8")
    return folder / "report.json"


def arguments(output, **changes):
    return SimpleNamespace(output_dir=output, models=["plain4x256"], periods_ms=[50., 75.],
        device="cpu", phase="staggered", environment_id=48, route_index=0, host="localhost",
        port=41451, resume=changes.get("resume", False), reuse_report=changes.get("reuse_report", []))


class MetricTests(unittest.TestCase):
    def test_time_weighting_not_sample_count_and_hover_tail_excluded(self):
        def trace(times):
            return [{"sample_s": 10.+t, "tracking_error_m": float(np.sqrt(t))} for t in times]
        coarse = joint.time_mean_squared_error(trace([0., 1., 2., 3.]), 10., 2.)
        fine = joint.time_mean_squared_error(trace([0., .1, .7, 1., 2., 3.]), 10., 2.)
        self.assertAlmostEqual(coarse, 1.)
        self.assertAlmostEqual(fine, coarse)
        values = [{"sample_s": t, "tracking_error_m": 0. if t <= 1. else 100.}
                  for t in (0., .5, 1., 2.)]
        self.assertEqual(joint.time_mean_squared_error(values, 0., 1.), 0.)

    def test_aborted_or_nonmonotonic_trace_cannot_fill_unmeasured_window(self):
        trace = [{"sample_s": t, "tracking_error_m": .1} for t in (0., .1)]
        self.assertIsNone(joint.time_mean_squared_error(trace, 0., .2))
        self.assertIsNone(joint.time_mean_squared_error(trace[::-1], 0., .05))

    def test_budget_is_externally_supplied_and_failed_candidates_never_win(self):
        rows = [{"model": "small", "period_ms": 50., "eligible": True,
                 "tracking_mse_m2": .2, "response_p99_ms": 10.},
                {"model": "large", "period_ms": 75., "eligible": True,
                 "tracking_mse_m2": .1, "response_p99_ms": 20.},
                {"model": "failed", "period_ms": 50., "eligible": False,
                 "tracking_mse_m2": .001, "response_p99_ms": 99.}]
        self.assertEqual(joint.choose(rows)["model"], "large")
        self.assertEqual(joint.choose(rows, 50.)["model"], "small")
        self.assertIsNone(joint.choose(rows, 40.))


class ReportTests(unittest.TestCase):
    def test_existing_flight_is_recomputed_from_traces_not_dnn_label_mse(self):
        with tempfile.TemporaryDirectory() as directory:
            suite = summary()
            path = write_flight(Path(directory)/"flight", suite)
            result = joint.compact_result(path, suite)
            self.assertTrue(result["eligible"])
            self.assertAlmostEqual(result["tracking_mse_m2"], .04)
            self.assertAlmostEqual(result["tracking_rmse_m"], .2)
            self.assertEqual(result["crossings"], dict.fromkeys(joint.VEHICLES, 5))

    def test_deadline_only_relaxation_or_changed_weights_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            suite = summary()
            report = joint.read_json(write_flight(Path(directory)/"flight", suite))
            changed = deepcopy(report)
            changed["deadline_ms"] = 100.
            with self.assertRaisesRegex(ValueError, "D=T"):
                joint.check_report(changed, suite)
            changed = deepcopy(report)
            changed["provenance"]["models"]["plain4x256"]["checkpoint_sha256"] = "different"
            with self.assertRaisesRegex(ValueError, "weights/config"):
                joint.check_report(changed, suite)

    def test_wrong_phase_wrong_runtime_and_simulator_slowdown_not_eligible(self):
        with tempfile.TemporaryDirectory() as directory:
            suite = summary()
            path = write_flight(Path(directory)/"flight", suite)
            report = joint.read_json(path)
            joint.check_report(report, suite)
            changed = {**report, "phase": "sync"}
            with self.assertRaisesRegex(ValueError, "configuration"):
                joint.check_report(changed, suite)
            changed = {**report, "runtime": {"dtype": "float16"}}
            with self.assertRaisesRegex(ValueError, "runtime"):
                joint.check_report(changed, suite)
            report["flight"]["Drone1"]["sim_seconds_per_host_second"] = .5
            joint.atomic_json(path, report)
            self.assertFalse(joint.compact_result(path, suite)["eligible"])


class RunnerTests(unittest.TestCase):
    def fake_child(self, command, suite, error=None):
        period = float(command[command.index("--period-ms")+1])
        self.assertEqual(float(command[command.index("--deadline-ms")+1]), period)
        self.assertEqual(command[command.index("--mode")+1], "dnn")
        self.assertEqual(command[command.index("--preprocess-policy")+1], "vectorized")
        write_flight(Path(command[command.index("--output-dir")+1]), suite,
                     command[command.index("--model")+1], period, error)
        return SimpleNamespace(returncode=0 if error is None else 1)

    def patches(self):
        from contextlib import ExitStack
        stack = ExitStack()
        stack.enter_context(patch.object(joint, "load_layout", return_value=({"duration": .2}, "route")))
        stack.enter_context(patch.object(joint, "provenance", return_value=evidence()))
        return stack

    def test_resume_recovers_completed_child_report_and_never_repeats_flights(self):
        with tempfile.TemporaryDirectory() as directory, self.patches():
            args = arguments(Path(directory)/"suite")
            called = []
            def interrupted_child(command, **kwargs):
                called.append(command)
                value = self.fake_child(command, summary())
                if len(called) == 2:
                    raise KeyboardInterrupt()
                return value
            with patch.object(joint.subprocess, "run", side_effect=interrupted_child):
                with self.assertRaises(KeyboardInterrupt):
                    joint.run(args)
            saved = joint.read_json(args.output_dir/"summary.json")
            self.assertEqual(saved["status"], "INCOMPLETE")
            self.assertEqual(len(saved["trials"]), 1)
            args.resume = True
            with patch.object(joint.subprocess, "run") as child:
                self.assertEqual(joint.run(args), 0)
                child.assert_not_called()
            saved = joint.read_json(args.output_dir/"summary.json")
            self.assertEqual(saved["status"], "COMPLETE")
            self.assertEqual(len(saved["trials"]), 2)
            with patch.object(joint.subprocess, "run") as child:
                self.assertEqual(joint.run(args), 0)
                child.assert_not_called()
            args.periods_ms = [50.]
            with self.assertRaisesRegex(ValueError, "resume configuration"):
                joint.run(args)

    def test_expected_deadline_failure_is_retained_and_next_candidate_runs(self):
        with tempfile.TemporaryDirectory() as directory, self.patches():
            args = arguments(Path(directory)/"suite")
            def child(command, **kwargs):
                error = "Drone3: RuntimeError: late DNN output not applied" if "50.0" in command else None
                return self.fake_child(command, summary(), error)
            with patch.object(joint.subprocess, "run", side_effect=child) as invoked:
                self.assertEqual(joint.run(args), 0)
                self.assertEqual(invoked.call_count, 2)
            saved = joint.read_json(args.output_dir/"summary.json")
            self.assertFalse(saved["trials"][0]["eligible"])
            self.assertTrue(saved["trials"][1]["eligible"])

    def test_child_interrupt_is_retained_as_attempt_and_retried_on_resume(self):
        with tempfile.TemporaryDirectory() as directory, self.patches():
            args = arguments(Path(directory)/"suite")
            with patch.object(joint.subprocess, "run", side_effect=lambda command, **kwargs:
                    self.fake_child(command, summary(), "KeyboardInterrupt: user stopped flight")):
                with self.assertRaises(KeyboardInterrupt):
                    joint.run(args)
            saved = joint.read_json(args.output_dir/"summary.json")
            self.assertEqual(saved["trials"], [])
            first = args.output_dir/"trials"/"plain4x256_T50"/"attempt_001"/"report.json"
            self.assertTrue(first.is_file())
            args.resume = True
            with patch.object(joint.subprocess, "run", side_effect=lambda command, **kwargs:
                    self.fake_child(command, summary())) as child:
                self.assertEqual(joint.run(args), 0)
                self.assertEqual(child.call_count, 2)
                self.assertIn("attempt_002", child.call_args_list[0].args[0][-1])
            self.assertEqual(joint.read_json(first)["errors"], ["KeyboardInterrupt: user stopped flight"])

    def test_collision_stops_suite_and_does_not_restart_on_resume(self):
        with tempfile.TemporaryDirectory() as directory, self.patches():
            args = arguments(Path(directory)/"suite")
            with patch.object(joint.subprocess, "run", side_effect=lambda command, **kwargs:
                    self.fake_child(command, summary(), "Drone1: RuntimeError: collision with gate")) as child:
                with self.assertRaisesRegex(RuntimeError, "collision/setup"):
                    joint.run(args)
                self.assertEqual(child.call_count, 1)
            args.resume = True
            with patch.object(joint.subprocess, "run") as child:
                with self.assertRaisesRegex(RuntimeError, "collision/setup"):
                    joint.run(args)
                child.assert_not_called()

    def test_reuse_existing_report_skips_that_candidate(self):
        with tempfile.TemporaryDirectory() as directory, self.patches():
            folder = Path(directory)
            report = write_flight(folder/"existing", summary())
            args = arguments(folder/"suite", reuse_report=[report])
            with patch.object(joint.subprocess, "run", side_effect=lambda command, **kwargs:
                    self.fake_child(command, summary())) as child:
                self.assertEqual(joint.run(args), 0)
                self.assertEqual(child.call_count, 1)
                self.assertIn("75.0", child.call_args.args[0])
            saved = joint.read_json(args.output_dir/"summary.json")
            self.assertTrue(saved["trials"][0]["reused"])
            trace = report.parent/"flight_trace.json"
            trace.write_text("{}", encoding="utf-8")
            args.resume = True
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                joint.run(args)


if __name__ == "__main__":
    unittest.main()
