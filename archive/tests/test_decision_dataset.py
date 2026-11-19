"""Temporary fixtures only: no AirSim, production data or GPU training."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from controller_architecture_study import decision_dataset as data
from controller_architecture_study.eval_offline import load_normalization, load_split
from falcon_paper_fidelity.motion_primitive import QuinticMotionPrimitivePlanner
from falcon_paper_fidelity.mpc_complete_300d import build_mpc_complete_300d


class DecisionDatasetTests(unittest.TestCase):
    def setUp(self):
        self.planner = QuinticMotionPrimitivePlanner(data.PLANNER_CONFIG)
        self.trajectory, _ = data.route_for_environment(3, self.planner)
        self.rng = np.random.default_rng(7)

    def test_relative_encoding_reconstructs_local_solver_window(self):
        state, refs, _ = data.sample_problem(self.trajectory, self.planner, self.rng, 7)
        feature = data.build_features(state, refs).astype(np.float64)
        feature[10:220].reshape(21, 10)[:, 7:10] += feature[7:10]
        local_state, local_refs = data.local_problem(state, refs)
        expected = build_mpc_complete_300d(local_state, local_refs)
        np.testing.assert_allclose(feature, expected, atol=1e-6)

    def test_translation_and_quaternion_sign_invariance(self):
        state, refs, _ = data.sample_problem(self.trajectory, self.planner, self.rng, 7)
        shift = np.array([80., -70., 20.])
        altered = replace(state, position=state.position + shift, quaternion=-state.quaternion)
        moved = [replace(ref, position=ref.position + shift, orientation=-ref.orientation) for ref in refs]
        np.testing.assert_allclose(data.build_features(altered, moved), data.build_features(state, refs), atol=1e-6)

    def test_perturbation_keeps_committed_phase(self):
        a, refs_a, phase_a = data.sample_problem(self.trajectory, self.planner, np.random.default_rng(3), 1)
        b, refs_b, phase_b = data.sample_problem(self.trajectory, self.planner, np.random.default_rng(3), 4)
        self.assertEqual(phase_a, phase_b)
        for left, right in zip(refs_a, refs_b):
            np.testing.assert_array_equal(left.position, right.position)
        self.assertGreater(np.linalg.norm(b.position - refs_b[0].position), .08)

    def test_routes_are_c2_and_start_from_rest(self):
        for env in range(4):
            trajectory, _ = data.route_for_environment(env, self.planner)
            np.testing.assert_array_equal(trajectory.evaluate(0.)[1], np.zeros(3))
            for left, right in zip(trajectory.segments[:-1], trajectory.segments[1:]):
                for end, start in zip(left.primitive.evaluate(left.primitive.duration), right.primitive.evaluate(0.)):
                    np.testing.assert_allclose(end, start, atol=1e-9)
            for _ in range(8):
                state, _, _ = data.sample_problem(trajectory, self.planner, self.rng, 0)
                self.assertLessEqual(np.linalg.norm(state.velocity), 3.5 + 1e-10)

    def test_interruption_resume_and_disjoint_training_only_normalization(self):
        class FakeTeacher:
            def __init__(self):
                self.controller = type("Controller", (), {"library_path": Path(data.__file__)})()

            def query(self, state, refs):
                return np.array([10., .1, .2, .3], np.float32), {
                    "iterations": 4, "kkt": 1e-5, "action_delta": 1e-5, "solve_ms": 1.}

        original_save = data.atomic_npz
        with tempfile.TemporaryDirectory() as temp, patch.object(data, "RefinedTeacher", FakeTeacher):
            output = Path(temp) / "pilot"

            def interrupted(path, values):
                if path.name == "env01_0000.npz":
                    raise KeyboardInterrupt()
                original_save(path, values)

            with patch.object(data, "atomic_npz", side_effect=interrupted), self.assertRaises(KeyboardInterrupt):
                data.generate(output, smoke=True)
            first = output / "chunks/env00_0000.npz"
            before = first.read_bytes()
            data.generate(output, smoke=True, resume=True)
            self.assertEqual(first.read_bytes(), before)
            data.prepare(output)
            manifest = json.loads((output / "dnn_dataset/split_manifest.json").read_text())
            self.assertEqual(manifest["counts"], {"train": 256, "validation": 32, "test": 32})
            split_ids = [set(manifest["environment_ids"][key]) for key in ("train", "validation", "test")]
            self.assertFalse(split_ids[0] & split_ids[1] or split_ids[0] & split_ids[2] or split_ids[1] & split_ids[2])
            x, u = load_split(output / "dnn_dataset", "train")
            norm = load_normalization(output / "dnn_dataset")
            np.testing.assert_allclose(norm["x_mean"], x.mean(axis=0, dtype=np.float64), atol=1e-6)
            np.testing.assert_allclose(norm["u_mean"], u.mean(axis=0, dtype=np.float64), atol=1e-6)
            data.prepare(output)
            with self.assertRaisesRegex(ValueError, "unchanged"):
                data.generate(output, smoke=False, resume=True)

    def test_300d_checkpoint_resume_without_test_or_optimizer_updates(self):
        import torch
        from controller_architecture_study.train_controller import train
        from controller_architecture_study.eval_offline import load_checkpoint

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = root / "data"
            dataset.mkdir()
            norm = {"x_mean": [0.] * 300, "x_std": [1.] * 300, "u_mean": [0.] * 4, "u_std": [1.] * 4}
            data.atomic_json(dataset / "normalization.json", norm)
            for split in ("train", "validation"):
                data.atomic_npz(dataset / f"{split}.npz", {"X": np.zeros((4, 300), np.float32),
                    "U": np.ones((4, 4), np.float32), "family_id": np.array([0, 0, 1, 1])})
            config = {"input_dim": 300, "output_dim": 4, "model_type": "plain_mlp", "activation": "relu",
                      "residual": False, "hidden_layers": 1, "hidden_width": 8, "data_dir": str(dataset),
                      "seed": 0, "learning_rate": 1e-4, "batch_size": 4, "max_epochs": 1, "min_epochs": 1,
                      "patience": 1, "gradient_clip_norm": 1., "evaluation_splits": ["train", "validation"]}
            with patch.object(torch.cuda, "is_available", return_value=False), patch.object(torch.optim.Adam, "step") as step:
                metrics = train(config, root / "run", device="cpu")
                self.assertNotIn("test", metrics)
                self.assertEqual(set(metrics["validation"]["by_family_id"]), {"0", "1"})
                self.assertEqual(step.call_count, 1)
            model = load_checkpoint(root / "run/best_model.pt", device="cpu")
            self.assertEqual(model[0].in_features, 300)
            with patch.object(torch.cuda, "is_available", return_value=False), patch.object(torch.optim.Adam, "step") as step:
                train(config, root / "run", device="cpu", resume=True)
                step.assert_not_called()


if __name__ == "__main__":
    unittest.main()
