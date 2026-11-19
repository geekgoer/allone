from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

from controller_architecture_study.convex_recovery_flight import (
    PerturbedClient, RecoveryRecorder, recovery_metrics)
from controller_architecture_study.evaluate_convex_recovery import execute
from controller_architecture_study import fly_convex_policy as flight
from controller_architecture_study.decision_dataset import PLANNER_CONFIG, sha256
from controller_architecture_study.decision_dataset_96k import route_for_group
from falcon_paper_fidelity.motion_primitive import QuinticMotionPrimitivePlanner
from controller_architecture_study.run_convex_recovery import build_jobs, seed_statistics, summarize


class SeedTests(unittest.TestCase):
    def test_multiseed_keeps_seed0_config_and_has_eight_new_jobs(self):
        def verified(data_dir, base):
            return {"counts": {"train": 76800 if data_dir.name == "features_only" else 153600,
                              "validation": 9600, "test": 9600},
                    "source_manifest_sha256": "source", "augmentation_manifest_sha256": "added",
                    "artifact_sha256": {"train.npz": "train", "test.npz": "test"}}
        with patch("controller_architecture_study.run_convex_recovery.verify_arm", side_effect=verified):
            old = build_jobs(Path("base"), Path("output"), 400)
            new = build_jobs(Path("base"), Path("output"), 400, arms=("features_recovery",), seeds=(0, 1, 2))
        self.assertEqual(new[:4], old[4:])
        self.assertEqual(len(new), 12)
        self.assertEqual(sum(config["seed"] != 0 for _, config, _ in new), 8)
        for _, config, _ in new:
            self.assertEqual(config["max_epochs"], 200)
            self.assertEqual(config["maximum_optimizer_updates"], 60000)
            self.assertNotIn("test.npz", config["dataset_file_sha256"])
        with self.assertRaises(ValueError):
            build_jobs(Path("base"), Path("output"), 400, seeds=(1, 1))

    def test_statistics_do_not_choose_best_seed_or_treat_missing_runs_as_zero(self):
        rows = [{"arm": "features_recovery", "width": width, "seed": seed, "status": "complete",
                 "fixed_train_mse": value/2, "fresh_train_mse": value, "validation_mse": value,
                 "gap_absolute": value/2} for width, seed, value in ((128, 0, 2.), (128, 1, 4.), (256, 0, 1.), (256, 1, 5.))]
        aggregate, paired = seed_statistics(rows, ("features_recovery",), (0, 1, 2))
        self.assertEqual(aggregate[0]["validation_mse_mean"], 3.)
        self.assertAlmostEqual(aggregate[0]["validation_mse_std"], np.sqrt(2))
        self.assertFalse(aggregate[0]["panel_complete"])
        self.assertIsNone(aggregate[-1]["validation_mse_mean"])
        self.assertEqual(paired[0]["validation_wide_minus_narrow"], [-1., 1.])
        self.assertEqual(paired[0]["wider_better_count"], 1)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            old = output/"runs/comparison.json"
            old.parent.mkdir()
            old.write_text("old report")
            path = summarize(output, output, arms=("features_recovery",), seeds=(0, 1, 2))
            self.assertNotEqual(path, old)
            self.assertEqual(old.read_text(), "old report")
            self.assertFalse(json.loads(path.read_text())["test_evaluated"])


def vec(x, y, z):
    return NS(x_val=x, y_val=y, z_val=z)


class FakeClient:
    def __init__(self, apply=True):
        self.paused, self.apply, self.calls = False, apply, []
        self.state = NS(kinematics_estimated=NS(position=vec(10., 0., -4.), linear_velocity=vec(5., 0., 0.),
            orientation=NS(w_val=1., x_val=0., y_val=0., z_val=0.)))

    def getMultirotorState(self, vehicle_name=""):
        return deepcopy(self.state)

    def simIsPause(self):
        return self.paused

    def simPause(self, paused):
        self.paused = paused

    def simSetKinematics(self, state, ignore_collision, vehicle_name=""):
        self.calls.append((ignore_collision, vehicle_name))
        if self.apply:
            self.state.kinematics_estimated = deepcopy(state)


class FlightTests(unittest.TestCase):
    def test_flight_integration_preserves_teacher_acceptance_and_records_kick_before_control(self):
        class Controller:
            config = NS(horizon=20, prediction_dt=.1)
            def reset(self):
                self.records = []
            def solve(self, state, refs):
                self.records.append({"position": state.position, "velocity": state.velocity,
                                     "reference_position": refs[0].position, "solve_ms": 1.})
                return np.ones(4)
        teacher, raw = Controller(), FakeClient()
        teacher.reset()
        _, layout = route_for_group(48, 3, 20, QuinticMotionPrimitivePlanner(PLANNER_CONFIG))
        def run(client, initial, environment, *, controller, planner, **kwargs):
            controller.reset()
            for step in range(140):
                refs = planner.reference_window_for_trajectory(planner.trajectory, elapsed=step*.05, dt=.1, horizon=21)
                kin = raw.state.kinematics_estimated
                kin.position = vec(*refs[0].position)
                kin.linear_velocity = vec(*refs[0].velocity)
                state = client.getMultirotorState(vehicle_name="Drone1").kinematics_estimated
                current = NS(position=np.array([state.position.x_val, state.position.y_val, state.position.z_val]),
                             velocity=np.array([state.linear_velocity.x_val, state.linear_velocity.y_val, state.linear_velocity.z_val]))
                controller.solve(current, refs)
            return NS(deadline_miss=np.zeros(140))
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            route = base/"routes/env48_route03.json"
            route.parent.mkdir()
            route.write_text(json.dumps(layout))
            (base/"manifest.json").write_text(json.dumps({"complete": True,
                "chunks": {"env48_route03": {"route_sha256": sha256(route)}}}))
            acceptance = base/"teacher_acceptance.json"
            acceptance.write_text("nominal teacher acceptance")
            args = NS(output=base, checkpoint=None, device="cpu", vehicle_name="Drone1", keep_gates=False,
                      environment_id=48, route_index=3, disturbance="combined_left", kick_time=3.)
            with patch.object(flight, "ConvexTeacher", return_value=teacher), patch.object(flight, "specification", return_value={}), \
                 patch.object(flight, "load_airsim_client"), patch.object(flight, "_connect", return_value=raw), \
                 patch.object(flight, "cleanup_generated_gates"), patch.object(flight, "run_episode", side_effect=run), \
                 patch.object(flight, "_crossing_results", return_value=[(i, None, 5., True) for i in range(1, 6)]):
                report = flight.fly(args)
            self.assertEqual(report["status"], "PASS")
            self.assertTrue(report["recovery"]["event"]["applied"])
            self.assertAlmostEqual(report["recovery"]["confirmation_time_after_kick_seconds"], .3)
            self.assertEqual(acceptance.read_text(), "nominal teacher acceptance")
            self.assertIn("reference_velocity", teacher.records[60])
            np.testing.assert_allclose(teacher.records[60]["position"]-teacher.records[60]["reference_position"],
                report["recovery"]["event"]["actual"]["position"]-np.array(report["recovery"]["event"]["before"]["position"]))

    def test_kick_waits_for_control_step_applies_physical_state_and_is_one_shot(self):
        controller, raw = NS(records=[]), FakeClient()
        client = PerturbedClient(raw, controller, profile="combined_right", heading_yaw=np.pi/2,
                                 dt=.05, vehicle_name="Drone1")
        client.getMultirotorState(vehicle_name="Drone1")
        self.assertEqual(raw.calls, [])
        controller.records = [{}]*60
        state = client.getMultirotorState(vehicle_name="Drone1")
        np.testing.assert_allclose([state.kinematics_estimated.position.x_val,
                                   state.kinematics_estimated.position.y_val], [9.4, 0.], atol=1e-10)
        self.assertAlmostEqual(state.kinematics_estimated.linear_velocity.x_val, 4.2)
        self.assertAlmostEqual(state.kinematics_estimated.linear_velocity.y_val, -.8)
        self.assertTrue(client.event["applied"])
        self.assertEqual(client.event["simulation_time_seconds"], 3.)
        self.assertEqual(raw.calls, [(False, "Drone1")])
        self.assertFalse(raw.paused)
        client.getMultirotorState(vehicle_name="Drone1")
        self.assertEqual(len(raw.calls), 1)

    def test_readback_mismatch_fails_and_restores_pause(self):
        raw = FakeClient(apply=False)
        client = PerturbedClient(raw, NS(records=[{}]*60), profile="combined_left", heading_yaw=0.,
                                 dt=.05, vehicle_name="Drone1")
        with self.assertRaisesRegex(RuntimeError, "readback mismatch"):
            client.getMultirotorState(vehicle_name="Drone1")
        self.assertFalse(client.event["applied"])
        self.assertFalse(raw.paused)

    def test_sustained_recovery_does_not_accept_late_hover_or_a_single_good_sample(self):
        def row(p, v):
            return {"position": np.array([p, 0., 0.]), "reference_position": np.zeros(3),
                    "velocity": np.array([v, 0., 0.]), "reference_velocity": np.zeros(3)}
        event = {"applied": True, "step": 0}
        records = [row(1., 1.)]*20+[row(.1, .1)]*6
        metrics = recovery_metrics(records, event, .05)
        self.assertTrue(metrics["recovered"])
        self.assertAlmostEqual(metrics["confirmation_time_after_kick_seconds"], 1.25)
        self.assertFalse(recovery_metrics([row(1., 1.)]*70+[row(.1, .1)]*6, event, .05)["recovered"])
        self.assertFalse(recovery_metrics([row(.1, .1), row(1., 1.)]*30, event, .05)["recovered"])
        self.assertFalse(recovery_metrics([], None, .05)["recovered"])

    def test_recorder_delegates_control_and_keeps_reference_velocity_after_reset(self):
        class Controller:
            def reset(self):
                self.records = []
            def solve(self, state, refs):
                self.records.append({"position": state})
                return np.ones(4)
        controller = RecoveryRecorder(Controller())
        controller.reset()
        refs = [NS(velocity=np.array([5., 1., 0.]))]
        np.testing.assert_array_equal(controller.solve(np.zeros(3), refs), np.ones(4))
        np.testing.assert_array_equal(controller.records[0]["reference_velocity"], refs[0].velocity)
        controller.reset()
        self.assertEqual(controller.records, [])

    def test_suite_resumes_and_teacher_failed_cases_are_skipped_not_successful(self):
        config = {"base": "base", "models": [{"model": "dnn", "checkpoint": "model.pt"}],
                  "cases": [{"case": "case", "environment_id": 48, "route_index": 3, "disturbance": "combined_left"}],
                  "protocol": {"kick_time_seconds": 3.}, "vehicle_name": "Drone1"}
        calls = []
        def fly(args):
            calls.append(args)
            return {"status": "FAIL", "samples": 100, "error": "collision",
                    "recovery": {"event": {"applied": True}}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"report.json"
            report = execute(config, path, fly_fn=fly)
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["summary"][1]["skipped_teacher_failed"], 1)
            self.assertIsNone(report["summary"][1]["success_rate"])
            execute(config, path, resume=True, fly_fn=fly)
            self.assertEqual(len(calls), 1)

    def test_interruption_or_connection_failure_is_resumable_and_not_model_failure(self):
        config = {"base": "base", "models": [], "cases": [{"case": "case", "environment_id": 48,
                  "route_index": 3, "disturbance": "combined_right"}],
                  "protocol": {"kick_time_seconds": 3.}, "vehicle_name": "Drone1"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"report.json"
            stopped = execute(config, path, fly_fn=lambda _: {"status": "FAIL", "error": "interrupted by user"})
            self.assertEqual(stopped["status"], "interrupted")
            failed = execute(config, path, resume=True, fly_fn=lambda _: {"status": "FAIL", "error": "no RPC"})
            self.assertEqual(failed["status"], "needs_attention")
            self.assertEqual(failed["results"], {})
            done = execute(config, path, resume=True, fly_fn=lambda _: {"status": "PASS", "samples": 400,
                "recovery": {"event": {"applied": True}}})
            self.assertEqual(done["status"], "complete")


if __name__ == "__main__":
    unittest.main()
