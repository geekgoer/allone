"""CPU-only tests: no CUDA, real AirSim, original-data writes or training."""

from copy import deepcopy
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from controller_architecture_study import dynamic_probe_common as common
from controller_architecture_study import probe_airsim_three as live
from controller_architecture_study import probe_capacity as capacity
from controller_architecture_study import probe_airsim_chain as chain
from controller_architecture_study.probe_capacity import compact_result, trial_cases
from controller_architecture_study.probe_state_value import cluster_interval, exploratory_frontier, paired_summary


class PairedTests(unittest.TestCase):
    def test_paired_improvement_and_environment_bootstrap(self):
        errors = np.array([[4., 1.], [3., 2.], [2., 3.], [5., 1.]])
        meta = {"environment_id": np.array([48, 48, 49, 49]), "family_id": np.array([0, 0, 7, 7])}
        row = paired_summary(errors, ["small", "large"], meta)[0]
        self.assertAlmostEqual(row["mean_error_reduction"], 1.75)
        self.assertEqual(row["second_wins_fraction"], .75)
        self.assertEqual(row["environment_clusters"], 2)
        self.assertEqual(row["by_family"]["combined_recovery"]["samples"], 2)
        self.assertIsNone(cluster_interval(np.ones(2), np.ones(2)))

    def test_cost_oracle_includes_static_baseline(self):
        errors = np.array([[0., 8.], [8., 0.]])
        curve = exploratory_frontier(errors, ["a", "b"], [1., 2.])
        self.assertEqual(curve[0]["label_aware_oracle_mse"], 4.)
        self.assertEqual(curve[-1]["label_aware_oracle_mse"], 0.)
        for row in curve:
            self.assertLessEqual(row["label_aware_oracle_mse"], row["best_static_mse"])


class QueueTests(unittest.TestCase):
    def test_fcfs_busy_drop_and_errors_are_accounted(self):
        started, release = threading.Event(), threading.Event()
        def predictor(model, feature):
            started.set()
            release.wait(2.)
            if model == "bad":
                raise ValueError("invalid")
            return np.zeros(4), {"service_ms": 1., "gpu_compute_ms": None}
        service = common.InferenceService(predictor)
        now = time.perf_counter()
        def row(vehicle):
            return dict(vehicle=vehicle, cycle=0, scheduled_s=now, sample_start_s=now)
        try:
            first = service.submit("ok", np.zeros(237), row("Drone1"))
            self.assertTrue(started.wait(1.))
            self.assertIsNone(service.submit("ok", np.zeros(237), row("Drone1")))
            second = service.submit("bad", np.zeros(237), row("Drone2"))
            release.set()
            np.testing.assert_array_equal(first.result(2.), np.zeros(4))
            with self.assertRaises(ValueError):
                second.result(2.)
        finally:
            release.set()
            service.close()
        stats = common.summarize_timing(service.rows, 1e6)
        self.assertEqual(stats["released"], 3)
        self.assertEqual(stats["ready"], 1)
        self.assertEqual(stats["dropped_busy"], 1)
        self.assertEqual(stats["inference_errors"], 1)
        self.assertEqual(stats["output_deadline_misses"], 2)
        self.assertIsNone(stats["sample_to_dnn_dispatch_ms"])

    def test_release_overruns_are_not_hidden_by_catchup(self):
        service = common.InferenceService(lambda *_: None)
        try:
            releases = list(common.release_schedule(time.perf_counter()-1., .3, .1,
                            threading.Event(), service, "Drone1"))
        finally:
            service.close()
        self.assertEqual(releases, [])
        self.assertEqual(common.summarize_timing(service.rows, 50)["skipped_release"], 3)

    def test_timing_separates_shadow_and_application(self):
        row = dict(vehicle="Drone1", status="ready", scheduled_s=1., sample_start_s=1.01,
                   enqueue_s=1.02, start_s=1.03, ready_s=1.04, service_ms=10., mpc_dispatch_s=1.035)
        stats = common.summarize_timing([row], 50.)
        self.assertAlmostEqual(stats["queue_ms"]["mean"], 10.)
        self.assertAlmostEqual(stats["sample_to_output_ms"]["mean"], 30.)
        self.assertEqual(stats["mpc_commands_dispatched"], 1)
        self.assertEqual(stats["dnn_commands_dispatched"], 0)

    def test_stages_include_busy_requests_and_distinguish_wait_from_work(self):
        row = dict(vehicle="Drone1", status="dropped_busy", scheduled_s=1., sample_start_s=1.,
                   state_ready_s=1.002, collision_start_s=1.002, collision_ready_s=1.003,
                   preprocess_requested_s=1.003, preprocess_start_s=1.010,
                   reference_ready_s=1.015, feature_ready_s=1.017, enqueue_s=1.018,
                   mpc_start_s=1.019, mpc_ready_s=1.020,
                   command_start_s=1.020, command_ready_s=1.022)
        stats = common.summarize_timing([row], 50.)
        for key, expected in {"state_rpc_ms": 2., "collision_rpc_ms": 1., "preprocess_wait_ms": 7.,
                              "reference_ms": 5., "feature_ms": 2., "sample_to_enqueue_ms": 18.,
                              "mpc_solve_ms": 1., "command_submit_ms": 2.}.items():
            self.assertAlmostEqual(stats[key]["mean"], expected)
        self.assertEqual(stats["output_deadline_misses"], 1)

    def test_flight_pass_does_not_mask_late_or_missing_requests(self):
        row = dict(vehicle="Drone1", status="ready", scheduled_s=1., sample_start_s=1.,
                   enqueue_s=1., start_s=1., ready_s=1.06, service_ms=60., mpc_dispatch_s=1.02)
        meta = dict(status="COMPLETE", planned_releases=1, all_15_crossings_passed=True)
        with tempfile.TemporaryDirectory() as directory:
            report = common.save_timing(Path(directory), [row], 50., meta)
            self.assertTrue(report["flight_passed"])
            self.assertFalse(report["realtime_passed"])
            self.assertFalse(report["acceptance_passed"])
            row["ready_s"], row["service_ms"] = 1.01, 10.
            report = common.save_timing(Path(directory), [row], 50., {**meta, "planned_releases": 2})
            self.assertFalse(report["realtime_passed"])
            report = common.save_timing(Path(directory), [row], 50., meta)
            self.assertTrue(report["acceptance_passed"])
            row["mpc_dispatch_s"] = 1.08
            report = common.save_timing(Path(directory), [row], 50., meta)
            self.assertFalse(report["realtime_passed"])


class CapacityTests(unittest.TestCase):
    def test_interrupted_suite_keeps_completed_trials(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            with patch.object(capacity, "new_output", return_value=folder), \
                 patch.object(capacity, "replay", side_effect=[{}, KeyboardInterrupt()]), \
                 patch.object(capacity, "compact_result", return_value={"round": 1}), \
                 patch("sys.argv", ["probe_capacity", "--clients", "8", "--models", "plain6x1024"]):
                with self.assertRaises(KeyboardInterrupt):
                    capacity.main()
            report = common.read_json(folder / "summary.json")
            self.assertEqual(report["status"], "INCOMPLETE")
            self.assertEqual(report["trials"], [{"round": 1}])
            self.assertEqual(report["current_trial"]["phase"], "staggered")

    def test_scale_sweep_covers_models_phases_and_repeats(self):
        cases = list(trial_cases([1, 3, 6, 7, 8, 12, 16], ["plain6x1024", "residual33x1024"], 2))
        self.assertEqual(len(cases), 56)
        self.assertEqual(len(set(cases)), 56)
        self.assertEqual(set(c[1:] for c in cases[:28]), set(c[1:] for c in cases[28:]))
        self.assertNotEqual(cases[0][1:], cases[28][1:])
        self.assertNotIn("residual33x1024", common.DEFAULT_MODELS)
        self.assertIn("residual33x1024", common.RUNS)

    def test_capacity_summary_uses_release_latency_and_worst_client(self):
        timing = dict(released=10, ready=8, output_deadline_misses=3, output_deadline_miss_fraction=.3,
                      skipped_release=1, dropped_busy=1, service_ms={"mean": 2.},
                      release_to_output_ms={"p99": 80.}, sample_to_output_ms={"p99": 30.})
        report = dict(timing=timing, clients=8, request_selection="residual33x1024", phase="sync",
                      report_path="test.json", status="COMPLETE", realtime_passed=False,
                      per_vehicle={"Drone1": {"output_deadline_miss_fraction": .5},
                                   "Drone2": {"output_deadline_miss_fraction": .1}})
        row = compact_result(report, 1)
        self.assertEqual(row["response_p99_ms"], 80.)
        self.assertEqual(row["worst_client_miss_fraction"], .5)
        self.assertFalse(row["realtime_passed"])


class FlightHarnessTests(unittest.TestCase):
    def test_expired_or_pending_control_is_not_applied(self):
        from concurrent.futures import Future

        pending, row = Future(), {}
        with self.assertRaisesRegex(RuntimeError, "deadline expired"):
            live.await_control(pending, row, time.perf_counter()-1.)
        self.assertEqual(row["control_outcome"], "deadline_abort")
        ready = Future()
        ready.set_result(np.zeros(4))
        with self.assertRaisesRegex(RuntimeError, "late DNN"):
            live.await_control(ready, row, time.perf_counter()-1.)
        self.assertEqual(row["control_outcome"], "late_output_not_applied")
        np.testing.assert_array_equal(live.await_control(ready, {}, time.perf_counter()+1.), np.zeros(4))

    def test_translation_preserves_coefficients_and_original(self):
        layout = {"segments": [{"coefficients": np.arange(18).reshape(3, 6).tolist()}],
                  "gates": [{"center": [10., 0., -4.]}]}
        before = deepcopy(layout)
        result = live.translate_layout(layout, np.array([0., 20., -.1]))
        self.assertEqual(layout, before)
        np.testing.assert_array_equal(np.array(result["segments"][0]["coefficients"])[:, 1:],
                                      np.array(before["segments"][0]["coefficients"])[:, 1:])
        np.testing.assert_allclose(result["gates"][0]["center"], [10., 20., -4.1])

    def test_three_shadow_producers_have_separate_rpc_clients_and_one_reset(self):
        self._check_shadow("serialized")

    def test_vectorized_shadow_preserves_rpc_ownership_timing_and_deadlines(self):
        self._check_shadow("vectorized")

    def _check_shadow(self, policy):
        state = SimpleNamespace(position=np.array([0., 0., -4.]), quaternion=np.array([1., 0., 0., 0.]),
                                velocity=np.zeros(3), acceleration=np.zeros(3), timestamp_ns=1)
        ref = SimpleNamespace(position=state.position, orientation=state.quaternion)
        clients, resets, placements, encoders = [], [], [], []
        def feature(name):
            encoders.append(name)
            return np.zeros(237)
        class Client:
            def __init__(self, **kwargs):
                self.owner = threading.get_ident()
                clients.append(self)
            def confirmConnection(self): pass
            def listVehicles(self): return list(live.VEHICLES)
            def simListSceneObjects(self, regex): return []
            def reset(self): resets.append(1)
            def simPause(self, value): pass
            def simSetKinematics(self, *args, **kwargs): placements.append(kwargs["vehicle_name"])
            def simGetCollisionInfo(self, **kwargs):
                assert self.owner == threading.get_ident()
                return SimpleNamespace(has_collided=False, time_stamp=0)
            def hoverAsync(self, **kwargs): pass
            def ping(self): return True
        class Drone:
            def __init__(self, client, vehicle): self.client = client
            def enable_api_control(self, value): pass
            def arm(self): pass
            def hover(self): pass
            def get_raw_state(self):
                assert self.client.owner == threading.get_ident()
                return state
        class Planner:
            def __init__(self, layout):
                self.trajectory = SimpleNamespace(duration=.1)
                self.config = SimpleNamespace()
            def reference_window_for_trajectory(self, *args, **kwargs): return [ref]*21
        class Pool:
            def __init__(self, *args):
                self.parameter_bytes = 10
                self.norm = {"x_mean": np.zeros(237)}
            def predict(self, *args): return np.array([9.8, 0., 0., 0.]), {"service_ms": .1, "gpu_compute_ms": None}
            def runtime_info(self): return {"mock": True}
        def short_schedule(start, duration, period, stop, service, vehicle, phase=0.):
            yield 0, time.perf_counter()
        args = SimpleNamespace(mode="shadow", model="plain4x256", device="cpu", phase="sync",
            period_ms=50., deadline_ms=50., environment_id=48, route_index=0, host="localhost", port=41451,
            keep_gates=False, preprocess_policy=policy)
        layout = {"segments": [], "gates": [{"index": 1, "center": [10., 0., -4.], "orientation": [1., 0., 0., 0.]}]}
        api = SimpleNamespace(MultirotorClient=Client, KinematicsState=SimpleNamespace,
                              Vector3r=lambda *v: v, Quaternionr=lambda *v: v)
        from contextlib import ExitStack
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            patches = {"provenance": lambda *_: {}, "ModelPool": Pool, "load_layout": lambda *_: (layout, "fake"),
                "StoredRoutePlanner": Planner, "new_output": lambda *_: Path(directory), "load_airsim_client": lambda: None,
                "require_airsim": lambda: api, "AirSimDrone": Drone, "vehicle_local_to_world_offset": lambda *_: np.zeros(3),
                "spawn_gate_sequences": lambda *a, **k: SimpleNamespace(destroy=lambda: None),
                "ConvexTeacher": lambda: SimpleNamespace(solve=lambda *_: np.array([9.8, 0., 0., 0.])),
                "encode": lambda *_: feature("legacy"), "fast_encode": lambda *_: feature("vectorized"),
                "FastReferenceWindow": lambda *_: SimpleNamespace(window=lambda *_: [ref]*21),
                "send_control": lambda *a, **k: None,
                "release_schedule": short_schedule, "_crossing_results": lambda *_: [(i, None, 0., True) for i in range(5)]}
            for name, replacement in patches.items():
                stack.enter_context(patch.object(live, name, replacement))
            # Three mock releases cannot satisfy the complete planned flight schedule.
            with self.assertRaises(SystemExit) as stopped:
                live.run(args)
            self.assertEqual(stopped.exception.code, 2)
            report = common.read_json(Path(directory)/"report.json")
            self.assertEqual(report["timing"]["mpc_commands_dispatched"], 3)
            self.assertEqual(report["timing"]["dnn_commands_dispatched"], 0)
            self.assertEqual(report["timing"]["ready"], 3)
            self.assertTrue(report["flight_passed"])
            self.assertFalse(report["acceptance_passed"])
            self.assertEqual(report["preprocess_policy"], policy)
            for name in ("state_rpc_ms", "collision_rpc_ms", "preprocess_wait_ms", "reference_ms",
                         "feature_ms", "mpc_solve_ms", "command_submit_ms"):
                self.assertEqual(report["timing"][name]["count"], 3)
        self.assertEqual(len(clients), 4)
        self.assertEqual(len(set(c.owner for c in clients)), 4)
        self.assertEqual(resets, [1])
        self.assertEqual(placements, list(live.VEHICLES))
        self.assertEqual(encoders, ["vectorized" if policy == "vectorized" else "legacy"]*3)


class ChainTests(unittest.TestCase):
    def test_timing_failure_continues_comparison_and_stays_visible(self):
        invoked = []
        def fake_flight(command, **kwargs):
            policy = command[command.index("--preprocess-policy")+1]
            invoked.append(policy)
            folder = Path(command[command.index("--output-dir")+1])
            folder.mkdir()
            report = {"preprocess_policy": policy, "report_path": str(folder/"report.json"),
                      "status": "COMPLETE", "errors": [], "planned_releases": 3,
                      "flight_passed": True, "realtime_passed": False, "acceptance_passed": False,
                      "flight": {}, "timing": dict(released=3, ready=3, skipped_release=0, dropped_busy=0,
                                                    output_deadline_misses=3, mpc_dispatch_deadline_misses=3)}
            common.atomic_json(folder/"report.json", report)
            self.assertEqual(command[command.index("--mode")+1], "shadow")
            return SimpleNamespace(returncode=2)
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(chain, "new_output", return_value=Path(directory)), \
             patch.object(chain.subprocess, "run", side_effect=fake_flight):
            self.assertEqual(chain.run("cpu"), 2)
            summary = common.read_json(Path(directory)/"summary.json")
            self.assertEqual(summary["status"], "COMPLETE")
            self.assertFalse(summary["acceptance_passed"])
            self.assertEqual(len(summary["trials"]), 2)
        self.assertEqual(invoked, ["parallel", "serialized"])

    def test_connection_failure_stops_remaining_flight_and_saves_summary(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(chain, "new_output", return_value=Path(directory)), \
             patch.object(chain.subprocess, "run", return_value=SimpleNamespace(returncode=1)) as flight:
            self.assertEqual(chain.run("cpu"), 1)
            summary = common.read_json(Path(directory)/"summary.json")
            self.assertEqual(summary["status"], "INCOMPLETE")
            self.assertEqual(summary["trials"], [])
            self.assertIn("exit 1", summary["error"])
            self.assertEqual(flight.call_count, 1)


if __name__ == "__main__":
    unittest.main()
