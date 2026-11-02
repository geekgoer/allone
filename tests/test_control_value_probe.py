"""Offline scheduling, pairing, metrics and mocked-flight tests; no real simulator/GPU."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from controller_architecture_study import control_value_probe as probe


def result(model, period=50., delay=0., repeat=0, valid=True, cost=1.):
    trial = probe.trial_matrix([model], ["tracking"], [period], [delay], 1, 0)[0]
    trial.update(repeat=repeat, id=f"{model}_{period}_{delay}_{repeat}")
    return dict(trial=trial, comparable=valid, tracking_mse_m2=cost)


class ValueProbeTests(unittest.TestCase):
    def test_default_matrix_and_reproducible_pairing(self):
        args = (probe.MODELS, probe.SCENARIOS, [50., 100.], [0., 20.], 2, 0)
        matrix = probe.trial_matrix(*args)
        self.assertEqual(len(matrix), 192)
        self.assertEqual(len({r["id"] for r in matrix}), 192)
        self.assertEqual(matrix, probe.trial_matrix(*args))
        for start in range(0, len(matrix), 4):
            block = matrix[start:start+4]
            self.assertEqual({r["model"] for r in block}, set(probe.MODELS))
            self.assertEqual(len({(r["scenario"], r["period_ms"], r["extra_delay_ms"], r["repeat"])
                                  for r in block}), 1)

    def test_invalid_timing(self):
        for periods, delays in (([0], [0]), ([50], [50]), ([50], [-1]), ([float("nan")], [0])):
            with self.assertRaises(ValueError):
                probe.trial_matrix(["x"], ["tracking"], periods, delays, 1, 0)

    def test_additive_delay_and_no_catchup(self):
        self.assertAlmostEqual(probe.due_time(2.007, 20), 2.027)
        self.assertEqual(probe.next_slot(10., 10.13, .05), 3)

    def test_time_weighted_cost_and_failed_window(self):
        trace = [{"t": 0., "error_m": 1.}, {"t": .2, "error_m": 1.},
                 {"t": 1.2, "error_m": 1.}]
        metrics = probe.tracking_metrics(trace, 1.)
        self.assertAlmostEqual(metrics["tracking_mse_m2"], 1.)
        self.assertIsNone(probe.tracking_metrics(trace[:2], 1.)["tracking_mse_m2"])

    def test_failures_not_ranked_and_pair_keys_do_not_mix(self):
        rows = [result(probe.MODELS[0]), result("big", cost=.5),
                result("big", period=100., cost=.1),
                result(probe.MODELS[0], repeat=1), result("big", repeat=1, valid=False, cost=.001)]
        summary = probe.paired_summary(rows)
        self.assertEqual(len(summary["pairs"]), 2)
        self.assertEqual(summary["pairs"][0]["delta_tracking_mse_m2"], -.5)
        self.assertIsNone(summary["pairs"][1]["delta_tracking_mse_m2"])
        self.assertEqual(summary["failed_or_timing_invalid_trials"], 1)
        self.assertEqual(summary["timing_effects"][0]["delta_tracking_mse_m2"], -.4)

    def test_resume_refuses_changed_contract(self):
        probe.check_resume({"hash": "a"}, {"hash": "a"}, True)
        for current, resume in (({"hash": "b"}, True), ({"hash": "a"}, False)):
            with self.assertRaises(ValueError):
                probe.check_resume({"hash": "a"}, current, resume)

    def test_mocked_flight_holds_command_while_delay_and_clock_advance(self):
        from airsim_interface import RawDroneState
        from controller_architecture_study.fly_decision_dnn import StoredRoutePlanner
        from controller_architecture_study.fast_convex_preprocess import FastReferenceWindow

        coefficients = np.zeros((3, 6))
        coefficients[0, 1], coefficients[2, 0] = 5., -3.
        layout = {"duration": 8., "segments": [{"coefficients": coefficients.tolist(),
            "duration": 8., "start_time": 0., "heading_orientation": [1., 0., 0., 0.]}],
            "gates": [{"index": 1, "center": [10., 0., -3.], "orientation": [1., 0., 0., 0.]}]}
        planner = StoredRoutePlanner(layout)
        window = FastReferenceWindow(planner.trajectory, planner.config)
        start = probe.make_start(window, 8., "tracking")
        clock = SimpleNamespace(now=10., origin=10., paused=True)
        def tick(dt):
            clock.now += max(dt, .000001)
        def pause(value):
            clock.paused = value
            if not value:
                clock.origin = clock.now
        class Client:
            def reset(self):
                pass
            def simPause(self, value):
                pause(value)
            def simSetKinematics(self, kin, **kwargs):
                self.kin = kin
            def simGetCollisionInfo(self, **kwargs):
                tick(.0001)
                return SimpleNamespace(time_stamp=0, has_collided=False)
            def ping(self):
                tick(.0001)
        client = Client()
        class Drone:
            def enable_api_control(self, value):
                pass
            def arm(self):
                pass
            def hover(self):
                pass
            def get_raw_state(self):
                tick(.0001)
                t = 0. if clock.paused else clock.now-clock.origin
                return RawDroneState(client.kin.position + np.array([5.*t, 0., 0.]),
                    client.kin.orientation, client.kin.linear_velocity, np.zeros(3), np.zeros(3),
                    timestamp_ns=int(1e9 + t*1e9))
        class Pool:
            lag = .002
            def predict(self, name, x):
                tick(self.lag)
                return np.array([9.81, 0., 0., 0.]), {"service_ms": self.lag*1000}
        api = SimpleNamespace(KinematicsState=SimpleNamespace,
            Vector3r=lambda *v: np.array(v), Quaternionr=lambda x,y,z,w: np.array([w,x,y,z]))
        args = SimpleNamespace(vehicle="Drone1", duration=.22)
        trial = result("small", delay=20.)["trial"]
        pauses, leases = [], []
        def record_pause(value):
            pauses.append(value)
            pause(value)
        client.simPause = record_pause
        with tempfile.TemporaryDirectory() as temp, \
             patch("airsim_interface.require_airsim", return_value=api), \
             patch("airsim_interface.AirSimDrone", return_value=Drone()), \
             patch("airsim_interface.vehicle_local_to_world_offset", return_value=np.zeros(3)), \
             patch("gate_visualization.spawn_gate_sequences"), \
             patch("control_adapter.send_control", side_effect=lambda *a, **kw: leases.append(kw["duration"])), \
             patch.object(probe.time, "perf_counter", side_effect=lambda: clock.now), \
             patch.object(probe.time, "sleep", side_effect=tick):
            report = probe.flight(client, Pool(), layout, start, trial, args, Path(temp))
            commands = json.loads((Path(temp)/"commands.json").read_text())
            trace = json.loads((Path(temp)/"trace.json").read_text())
            good_pauses, good_leases = list(pauses), list(leases)
            Pool.lag = .060
            delayed = Path(temp)/"late"
            delayed.mkdir()
            bad = probe.flight(client, Pool(), layout, start, trial, args, delayed)
            self.assertFalse(bad["comparable"])
            self.assertEqual(bad["deadline_misses"], 1)
            self.assertIn("NOT applied", bad["error"])
            self.assertEqual(leases, good_leases)
        self.assertIsNone(report["error"])
        self.assertTrue(report["comparable"])
        self.assertEqual(good_pauses, [True, False, False])  # only placement, start and cleanup
        self.assertTrue(all(lease == 1. for lease in good_leases))
        dispatched = [r for r in commands if "dispatch_s" in r]
        self.assertGreaterEqual(len(dispatched), 4)
        for row in dispatched:
            self.assertGreaterEqual(row["dispatch_s"], row["inference_ready_s"]+.02)
        self.assertGreater(trace[-1]["reference_position"][0], trace[0]["reference_position"][0]+1.)
        for row in trace:
            actual = np.r_[row["position"], row["velocity"], row["quaternion"]]
            reference = np.r_[row["reference_position"], row["reference_velocity"], row["reference_quaternion"]]
            self.assertEqual(actual.shape, (10,))
            self.assertEqual(reference.shape, (10,))
            self.assertTrue(np.isfinite(actual-reference).all())
            np.testing.assert_allclose(row["reference_velocity"], [5., 0., 0.])
            self.assertAlmostEqual(row["reference_phase_s"], start["reference_phase_s"]+row["t"])


if __name__ == "__main__":
    unittest.main()
