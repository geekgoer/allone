"""300D flight adapter tests; no connection to AirSim and no model training."""

import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from controller_architecture_study import decision_dataset_96k as data
from controller_architecture_study import fly_decision_dnn as flight
from falcon_paper_fidelity.motion_primitive import QuinticMotionPrimitivePlanner
from airsim_interface import RawDroneState


class DecisionFlightTests(unittest.TestCase):
    def test_replayed_coefficients_and_windows_match_collection(self):
        planner = QuinticMotionPrimitivePlanner(data.PLANNER_CONFIG)
        original, layout = data.route_for_group(48, 3, 20, planner)
        stored = flight.StoredRoutePlanner(layout)
        self.assertIs(stored.plan_continuous_route(), stored.trajectory)
        for phase in np.r_[0., original.boundary_times, original.duration, original.duration + 1.]:
            left = planner.reference_window_for_trajectory(original, elapsed=float(phase), dt=.1, horizon=21)
            right = stored.reference_window_for_trajectory(stored.trajectory, elapsed=float(phase), dt=.1, horizon=21)
            state = RawDroneState(left[0].position + [0., .4, -.2], left[0].orientation,
                                  left[0].velocity + [.1, .2, -.1], np.zeros(3), np.zeros(3))
            np.testing.assert_array_equal(data.build_features(state, left), data.build_features(state, right))

    def test_raw_inference_normalization_clipping_and_no_teacher(self):
        import torch
        from controller_architecture_study.models.factory import build_model

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / "best_model.pt"
            model = build_model(torch, model_type="plain_mlp", input_dim=300, output_dim=4,
                                hidden_layers=1, hidden_width=8)
            for parameter in model.parameters():
                parameter.data.zero_()
            model[-1].bias.data.copy_(torch.tensor([1., -4., 4., 3.]))
            torch.save({"model_state": model.state_dict(), "input_dim": 300,
                        "hidden_layers": 1, "hidden_width": 8}, checkpoint)
            data.atomic_json(root / "config.json", {"input_dim": 300, "dataset_source_sha256": "fixture"})
            dataset = root / "data"
            data.atomic_json(dataset / "split_manifest.json", {
                "version": data.VERSION, "contract": data.CONTRACT, "source_manifest_sha256": "fixture"})
            data.atomic_json(dataset / "normalization.json", {"input_contract": data.CONTRACT,
                "x_mean": [0.] * 300, "x_std": [1.] * 300,
                "u_mean": [9., 0., 0., 0.], "u_std": [2., 1., 1., 1.]})
            controller = flight.DecisionDNN(checkpoint, dataset, "cpu")
            planner = QuinticMotionPrimitivePlanner(data.PLANNER_CONFIG)
            trajectory, _ = data.route_for_group(48, 0, 20, planner)
            state, refs, _ = data.sample_problem(trajectory, planner, np.random.default_rng(2), 1)
            np.testing.assert_array_equal(controller.predict_raw(data.build_features(state, refs)), [11., -4., 4., 3.])
            with patch.object(data.RefinedTeacher, "query", side_effect=AssertionError("MPC must not run")):
                np.testing.assert_array_equal(controller.solve(state, refs), [11., -3., 3., 2.])
            self.assertTrue(controller.records[0]["clipped"])
            np.testing.assert_array_equal(controller.records[0]["X"], data.build_features(state, refs))
            controller.reset()
            self.assertEqual(controller.records, [])
            data.atomic_json(root / "config.json", {"input_dim": 300, "dataset_source_sha256": "wrong"})
            with self.assertRaisesRegex(ValueError, "provenance"):
                flight.DecisionDNN(checkpoint, dataset, "cpu")

    def test_failure_keeps_partial_trace_and_report_without_live_connection(self):
        planner = QuinticMotionPrimitivePlanner(data.PLANNER_CONFIG)
        _, layout = data.route_for_group(48, 0, 20, planner)
        environment = flight.TrainingEnvironment(48, 0, 0., "fixture",
            tuple(flight.Gate(item["index"], item["center"], item["orientation"]) for item in layout["gates"]))

        class FakeDNN:
            def __init__(self, *args):
                self.records = []

        def interrupted(client, initial, env, **kwargs):
            controller = kwargs["controller"]
            for x in (0., 1.):
                controller.records.append({"X": np.zeros(300), "raw_U": np.array([10., 0., 0., 0.]),
                    "U": np.array([10., 0., 0., 0.]), "position": np.array([x, 0., -4.]),
                    "velocity": np.array([1., 0., 0.]), "reference_position": np.array([x, 0., -4.]),
                    "inference_ms": 1., "clipped": False})
            raise RuntimeError("vehicle collision with fixture gate")

        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()), \
                patch.object(flight, "DecisionDNN", FakeDNN), patch.object(flight, "load_route", return_value=(layout, environment)), \
                patch.object(flight, "_connect", return_value=object()), patch.object(flight, "cleanup_generated_gates"), \
                patch.object(flight, "run_episode", side_effect=interrupted):
            args = argparse.Namespace(checkpoint=Path(temp) / "best_model.pt", device="cpu", cpu_threads=1,
                environment_id=48, route_index=0, duration=20., vehicle_name="Drone1", keep_gates=True)
            report = flight.fly(args)
            self.assertEqual(report["status"], "FAIL")
            self.assertEqual(report["mpc_fallbacks"], 0)
            self.assertIn("collision", report["error"])
            with np.load(report["trace"]) as trace:
                self.assertEqual(trace["X"].shape, (2, 300))
            reports = list((Path(temp) / "flight").glob("*.json"))
            self.assertEqual(len(reports), 1)
            self.assertEqual(json.loads(reports[0].read_text())["samples"], 2)


if __name__ == "__main__":
    unittest.main()
