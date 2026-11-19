import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from airsim_interface import RawDroneState
from mpc_controller import MPCReference
from controller_architecture_study.convex_teacher import ConvexTeacher
from controller_architecture_study import convex_dataset as dataset
from controller_architecture_study.decision_dataset import sha256
from controller_architecture_study.train_convex_capacity import run_state, build_jobs, architectures, summarize


def stationary_problem(speed=0.):
    state = RawDroneState(np.zeros(3), np.array([1., 0., 0., 0.]), np.zeros(3), np.zeros(3), np.zeros(3))
    refs = [MPCReference(position=np.array([speed*.1*i, 0., 0.]), orientation=state.quaternion,
                         velocity=np.array([speed, 0., 0.]), acceleration=np.zeros(3), body_rates=np.zeros(3))
            for i in range(21)]
    return state, refs


class ConvexTeacherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.teacher = ConvexTeacher()

    def test_prediction_maps_and_strict_convexity(self):
        acceleration = np.tile([1., 2., -1.], 20)
        times = np.arange(1, 21)*.1
        np.testing.assert_allclose((self.teacher.position_map@acceleration).reshape(20, 3),
                                   .5*times[:, None]**2*np.array([1., 2., -1.]), atol=1e-12)
        np.testing.assert_allclose((self.teacher.velocity_map@acceleration).reshape(20, 3),
                                   times[:, None]*np.array([1., 2., -1.]), atol=1e-12)
        self.assertGreaterEqual(np.linalg.eigvalsh(self.teacher.hessian.toarray()).min(), .6-1e-10)

    def test_hover_and_forward_acceleration_sign(self):
        state, refs = stationary_problem()
        control, _ = self.teacher.query(state, refs)
        np.testing.assert_allclose(control, [9.80665, 0., 0., 0.], atol=1e-5)
        state, refs = stationary_problem(5.)
        control, diagnostics = self.teacher.query(state, refs)
        self.assertGreater(diagnostics["first_acceleration"][0], 1.)
        self.assertLess(control[2], -.1)
        self.assertLess(diagnostics["constraint_violation"], 2e-4)

    def test_planner_rate_is_not_mixed_with_optimized_attitude(self):
        state, refs = stationary_problem(5.)
        control, _ = self.teacher.query(state, refs)
        changed = [replace(ref, body_rates=np.array([2., -2., 2.])) for ref in refs]
        repeated, _ = self.teacher.query(state, changed)
        np.testing.assert_array_equal(control, repeated)

    def test_repeat_history_translation_and_quaternion_sign(self):
        state, refs = stationary_problem(5.)
        base, _ = self.teacher.query(state, refs)
        self.teacher.query(replace(state, position=np.array([.01, -.01, .01])), refs)
        repeated, _ = self.teacher.query(state, refs)
        np.testing.assert_array_equal(base, repeated)
        shift = np.array([18., -3., 8.])
        moved, _ = self.teacher.query(replace(state, position=state.position+shift, quaternion=-state.quaternion),
                                     [replace(ref, position=ref.position+shift) for ref in refs])
        np.testing.assert_allclose(base, moved, atol=2e-5)

    def test_soft_corridor_remains_feasible(self):
        state, refs = stationary_problem()
        action, info = self.teacher.query(replace(state, position=np.array([4., -4., 2.])), refs)
        self.assertGreater(info["max_slack"], 0.)
        self.assertTrue(np.isfinite(action).all())
        a = info["first_acceleration"]
        self.assertLessEqual(np.linalg.norm(a[:2]), np.tan(np.deg2rad(35))*(9.80665-a[2])+2e-4)


class ConvexPipelineTests(unittest.TestCase):
    def test_interrupted_collection_reuses_finished_chunk(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original_query, calls = ConvexTeacher.query, []

            def interrupted(teacher, state, references):
                calls.append(1)
                if len(calls) == 13:
                    raise RuntimeError("simulated interruption")
                return original_query(teacher, state, references)

            with patch.object(ConvexTeacher, "query", interrupted):
                with self.assertRaisesRegex(RuntimeError, "interruption"):
                    dataset.generate(root, smoke=True)
            first = root/"chunks/env00_route00.npz"
            digest = sha256(first)
            unfinished = json.loads((root/"manifest.json").read_text())
            self.assertFalse(unfinished["complete"])
            self.assertEqual(len(unfinished["chunks"]), 1)
            dataset.generate(root, smoke=True, resume=True)
            self.assertEqual(sha256(first), digest)
            self.assertTrue(json.loads((root/"manifest.json").read_text())["complete"])

    def test_chunk_resume_splits_and_training_smoke_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset.generate(root, smoke=True)
            before = {p: sha256(p) for p in (root/"chunks").glob("*.npz")}
            dataset.generate(root, smoke=True, resume=True)
            self.assertEqual(before, {p: sha256(p) for p in before})
            dataset.prepare(root)
            manifest = json.loads((root/"dnn_dataset/split_manifest.json").read_text())
            self.assertEqual(manifest["counts"], {"train": 8, "validation": 8, "test": 8})
            for split, env in (("train", 0), ("validation", 48), ("test", 54)):
                with np.load(root/"dnn_dataset"/f"{split}.npz") as archive:
                    self.assertEqual(set(archive["environment_id"]), {env})
                    self.assertEqual(set(archive["family_id"]), set(range(8)))
            with self.assertRaisesRegex(ValueError, "96k"):
                build_jobs(root, [0], 400)
            path = next(iter(before))
            with path.open("ab") as handle:
                handle.write(b"changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                dataset.generate(root, smoke=True, resume=True)

    def test_incomplete_production_requires_verified_flight(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                dataset.generate(Path(directory))

    def test_configuration_change_is_not_silently_resumed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(run_state({"seed": 0}, root), "new")
            (root/"config.json").write_text(json.dumps({"seed": 0}))
            self.assertEqual(run_state({"seed": 0}, root), "resume")
            with self.assertRaises(ValueError):
                run_state({"seed": 1}, root)

    def test_four_capacity_jobs_have_same_budget_and_do_not_evaluate_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"data_audit.json").write_text(json.dumps({"status": "PASS", "samples": 96000}))
            source = {"counts": {"train": 76800, "validation": 9600, "test": 9600},
                      "source_manifest_sha256": "example", "artifact_sha256": {
                          "train.npz": "train", "validation.npz": "validation", "normalization.json": "norm"}}
            with patch("controller_architecture_study.train_convex_capacity.verify_dataset", return_value=source):
                jobs = build_jobs(root, [0, 1, 2], 400)
            self.assertEqual(len(jobs), 12)
            self.assertEqual({config["hidden_width"] for _, config, _ in jobs}, {64, 128, 256, 1024})
            for _, config, _ in jobs:
                self.assertEqual(config["hidden_layers"], 4)
                self.assertEqual(config["max_epochs"], 400)
                self.assertEqual(config["evaluation_splits"], ["train", "validation"])
                self.assertNotIn("test.npz", config["dataset_file_sha256"])

    def test_scenic_falcon_grid_has_fifteen_distinct_architectures(self):
        expected_scenic = {(6, 1024), (5, 1024), (5, 512), (4, 1024), (4, 512), (4, 256),
                           (3, 1024), (3, 256), (3, 128), (2, 1024), (2, 128), (1, 1024)}
        grid = architectures("scenic_falcon15")
        self.assertEqual(len(grid), 15)
        self.assertEqual(set(grid), expected_scenic | {(9, 512), (21, 512), (33, 512)})
        from controller_architecture_study.compute_complexity import complexity
        costs = [complexity(300, 4, *pair)["parameter_count"] for pair in grid]
        self.assertEqual(costs, sorted(costs))

    def test_expanded_jobs_preserve_shared_runs_and_training_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"data_audit.json").write_text(json.dumps({"status": "PASS", "samples": 96000}))
            source = {"counts": {"train": 76800, "validation": 9600, "test": 9600},
                      "source_manifest_sha256": "example", "artifact_sha256": {
                          "train.npz": "train", "validation.npz": "validation", "normalization.json": "norm"}}
            with patch("controller_architecture_study.train_convex_capacity.verify_dataset", return_value=source):
                old = {name: config for name, config, _ in build_jobs(root, [0], 400)}
                jobs = build_jobs(root, [0], 400, "scenic_falcon15")
            self.assertEqual(len(jobs), 15)
            for name, config, _ in jobs:
                if name in old:
                    self.assertEqual(config, old[name])
                self.assertEqual(config["input_dim"], 300)
                self.assertEqual(config["max_epochs"], 400)
                self.assertEqual(config["evaluation_splits"], ["train", "validation"])
                self.assertNotIn("test.npz", config["dataset_file_sha256"])
                self.assertNotIn("init_checkpoint", config)

    def test_expanded_summary_includes_deep_models_and_correct_complexity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for depth, width in architectures("scenic_falcon15") + [(4, 64)]:
                run = root/"runs"/f"plain_mlp_h{depth}_w{width}_seed0"
                run.mkdir(parents=True)
                (run/"config.json").write_text(json.dumps({"hidden_layers": depth, "hidden_width": width, "seed": 0}))
            previous_report = root/"runs/comparison.json"
            previous_report.write_text("previous four-model report")
            path = summarize(root, "scenic_falcon15")
            report = json.loads(path.read_text())
            self.assertEqual(len(report["models"]), 15)
            self.assertEqual(report["architectures_requested"], 15)
            self.assertFalse(report["test_evaluated"])
            self.assertEqual(previous_report.read_text(), "previous four-model report")
            self.assertEqual(sum(row["architecture_group"] == "FALCON" for row in report["models"]), 3)
            deep = next(row for row in report["models"] if row["hidden_layers"] == 33)
            expected_macs = 300*512 + 32*512**2 + 512*4
            self.assertEqual(deep["mac_count"], expected_macs)
            self.assertEqual(deep["parameter_count"], expected_macs + 33*512 + 4)

    def test_summary_does_not_average_different_depths_of_same_width(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for depth, seed, value in ((4, 0, .1), (4, 1, .3), (5, 0, .8)):
                run = root/"runs"/f"plain_mlp_h{depth}_w512_seed{seed}"
                run.mkdir(parents=True)
                (run/"config.json").write_text(json.dumps({"hidden_layers": depth, "hidden_width": 512, "seed": seed}))
                for name in ("best_model.pt", "last_model.pt"):
                    (run/name).touch()
                (run/"completion.json").write_text(json.dumps({"epochs_completed": 400, "best_epoch": 350}))
                metrics = {"normalized_mse": value, "physical_rmse": .01, "per_output_rmse": {},
                           "p99_abs_error": {}, "by_family_id": {str(i): {"normalized_mse": value}
                                                                  for i in range(8)}}
                (run/"offline_metrics.json").write_text(json.dumps({"train": metrics, "validation": metrics}))
            report = json.loads(summarize(root, "scenic_falcon15").read_text())
            aggregates = {(item["hidden_layers"], item["width"]): item for item in report["by_architecture"]}
            self.assertEqual(len(aggregates), 2)
            self.assertAlmostEqual(aggregates[(4, 512)]["validation_mse_mean"], .2)
            self.assertEqual(aggregates[(4, 512)]["seeds_completed"], 2)
            self.assertEqual(aggregates[(5, 512)]["validation_mse_mean"], .8)
            self.assertIsNone(aggregates[(5, 512)]["validation_mse_std"])


if __name__ == "__main__":
    unittest.main()
