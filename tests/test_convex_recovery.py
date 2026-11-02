import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from controller_architecture_study import convex_dataset, convex_recovery as recovery
from controller_architecture_study.convex_features import KEPT, REMOVED, CONTRACT, project, encode
from controller_architecture_study.convex_teacher import ConvexTeacher
from controller_architecture_study.decision_dataset import build_features, sha256
from controller_architecture_study.fly_convex_policy import ConvexDNN
from controller_architecture_study.run_convex_recovery import build_jobs, WIDTHS, audit_run, summarize
from controller_architecture_study.train_controller import train
from tests.test_convex_teacher import stationary_problem


class FeatureTests(unittest.TestCase):
    def test_projection_indices_and_online_agreement(self):
        self.assertEqual(len(KEPT), 237)
        self.assertEqual(len(set(REMOVED)), 63)
        self.assertEqual(set(KEPT) | set(REMOVED), set(range(300)))
        features = np.arange(600, dtype=np.float32).reshape(2, 300)
        np.testing.assert_array_equal(project(features), features[:, KEPT])
        state, refs = stationary_problem(5.)
        np.testing.assert_array_equal(encode(state, refs), project(build_features(state, refs)))
        changed = [replace(ref, body_rates=np.array([.3, -.1, .2])) for ref in refs]
        np.testing.assert_array_equal(encode(state, refs), encode(state, changed))
        self.assertFalse(np.array_equal(build_features(state, refs), build_features(state, changed)))
        teacher = ConvexTeacher()
        np.testing.assert_array_equal(teacher.query(state, refs)[0], teacher.query(state, changed)[0])
        with self.assertRaises(ValueError):
            project(np.zeros(237))


class RecoveryTests(unittest.TestCase):
    def make_source(self, base):
        convex_dataset.generate(base, smoke=True)
        convex_dataset.prepare(base)
        return {path: sha256(path) for path in base.rglob('*') if path.is_file()}

    def test_interruption_reuses_chunk_and_prepare_preserves_frozen_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            base, output = Path(directory)/"base", Path(directory)/"output"
            before = self.make_source(base)
            query = ConvexTeacher.query
            calls = []

            def interrupted(teacher, state, refs):
                calls.append(1)
                if len(calls) == 8:
                    raise RuntimeError("simulated interruption")
                return query(teacher, state, refs)

            with patch.object(ConvexTeacher, "query", interrupted):
                with self.assertRaisesRegex(RuntimeError, "interruption"):
                    recovery.generate(base, output, smoke=True)
            chunk = output/"chunks/env00_route00_added.npz"
            digest = sha256(chunk)
            recovery.generate(base, output, smoke=True, resume=True)
            self.assertEqual(sha256(chunk), digest)
            manifest_hash = sha256(output/"manifest.json")
            recovery.generate(base, output, smoke=True, resume=True)
            self.assertEqual(sha256(output/"manifest.json"), manifest_hash)
            recovery.prepare(base, output)
            fixed_hashes = []
            for arm in recovery.ARMS:
                data_dir = output/"datasets"/arm
                split = recovery.verify_arm(data_dir, base)
                self.assertEqual(split["counts"], {"train": 8 if arm == "features_only" else 12,
                                                  "validation": 8, "test": 8})
                for name in ("validation", "test"):
                    original = recovery.load_npz(base/"dnn_dataset"/f"{name}.npz")
                    saved = recovery.load_npz(data_dir/f"{name}.npz")
                    for key in original:
                        np.testing.assert_array_equal(saved[key], project(original[key]) if key == "X" else original[key])
                norm = recovery.read_json(data_dir/"normalization.json")
                original_norm = recovery.read_json(base/"dnn_dataset/normalization.json")
                for key in ("x_mean", "x_std"):
                    np.testing.assert_array_equal(norm[key], np.asarray(original_norm[key])[list(KEPT)])
                for key in ("u_mean", "u_std"):
                    self.assertEqual(norm[key], original_norm[key])
                audit = recovery.load_npz(data_dir/"audit/train.npz")
                probe = recovery.load_npz(data_dir/"probe/train.npz")
                training = recovery.load_npz(data_dir/"train.npz")
                self.assertFalse(set(training["row_id"]) & set(probe["row_id"]))
                self.assertEqual(set(training["environment_id"]), {0})
                self.assertEqual(set(probe["family_id"]), set(range(8)))
                self.assertEqual(len(audit["X"]), 8)
                fixed_hashes.append({key: value for key, value in split["artifact_sha256"].items() if key != "train.npz"})
            self.assertEqual(fixed_hashes[0], fixed_hashes[1])
            self.assertEqual(before, {path: sha256(path) for path in before})
            # Preparing again must not invalidate training configs or frozen evaluation hashes.
            old = recovery.read_json(output/"datasets/features_recovery/split_manifest.json")
            recovery.prepare(base, output)
            self.assertEqual(old, recovery.read_json(output/"datasets/features_recovery/split_manifest.json"))
            with chunk.open("ab") as handle:
                handle.write(b"changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                recovery.generate(base, output, smoke=True, resume=True)

    def test_four_widths_two_arms_have_matched_updates_and_no_test_evaluation(self):
        def verified(data_dir, base):
            return {"counts": {"train": 76800 if data_dir.name == "features_only" else 153600,
                              "validation": 9600, "test": 9600},
                    "source_manifest_sha256": "source", "augmentation_manifest_sha256": "added",
                    "artifact_sha256": {name: name for name in ("train.npz", "validation.npz", "test.npz", "normalization.json")}}
        with patch("controller_architecture_study.run_convex_recovery.verify_arm", side_effect=verified):
            jobs = build_jobs(Path("base"), Path("output"), 400)
        self.assertEqual(len(jobs), 8)
        self.assertEqual(WIDTHS, (128, 256, 512, 1024))
        for _, config, _ in jobs:
            self.assertEqual(config["maximum_optimizer_updates"], 60000)
            self.assertEqual(config["min_epochs"], config["max_epochs"])
            self.assertEqual(config["evaluation_splits"], ["train", "validation"])
            self.assertNotIn("test.npz", config["dataset_file_sha256"])
            self.assertNotIn("init_checkpoint", config)

    def test_237d_checkpoint_resume_and_live_projection(self):
        import torch

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            base, output = Path(directory)/"base", Path(directory)/"output"
            self.make_source(base)
            recovery.generate(base, output, smoke=True)
            recovery.prepare(base, output)
            data_dir = output/"datasets/features_recovery"
            manifest = recovery.verify_arm(data_dir, base)
            config = {"input_dim": 237, "output_dim": 4, "input_contract": CONTRACT,
                "model_type": "plain_mlp", "activation": "relu", "residual": False,
                "hidden_layers": 2, "hidden_width": 8, "seed": 0, "batch_size": 4,
                "learning_rate": 1e-4, "max_epochs": 2, "min_epochs": 2, "patience": 2,
                "gradient_clip_norm": 1., "data_dir": str(data_dir), "evaluation_splits": ["train", "validation"],
                "dataset_source_sha256": manifest["source_manifest_sha256"],
                "augmentation_manifest_sha256": manifest["augmentation_manifest_sha256"],
                "dataset_file_sha256": manifest["artifact_sha256"]}
            run = output/"run"
            save = torch.save
            calls = []

            def interrupted(value, target, *args, **kwargs):
                if "last_model" in str(target):
                    calls.append(1)
                    if len(calls) == 3:
                        raise RuntimeError("interrupted epoch 2")
                return save(value, target, *args, **kwargs)

            with patch.object(torch, "save", side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError, "epoch 2"):
                    train(config, run, device="cpu")
            saved_epoch = torch.load(run/"last_model.pt", weights_only=False)["epoch"]
            self.assertEqual(saved_epoch, 1)
            train(config, run, device="cpu", resume=True)
            done = recovery.read_json(run/"completion.json")
            self.assertEqual(done["epochs_completed"], 2)
            self.assertEqual(done["evaluation_splits"], ["train", "validation"])
            controller = ConvexDNN(run/"best_model.pt", base, "cpu")
            state, refs = stationary_problem(5.)
            feature = build_features(state, refs)
            live = controller.predict_raw(feature)
            x = (project(feature)-controller.norm["x_mean"])/controller.norm["x_std"]
            with torch.inference_mode():
                offline = controller.model(torch.from_numpy(x[None])).numpy()[0]
            expected = offline*controller.norm["u_std"]+controller.norm["u_mean"]
            np.testing.assert_array_equal(live, expected.astype(float))
            self.assertEqual(controller.input_dim, 237)
            audit_run(run, base, "cpu")
            fixed = recovery.read_json(run/"fixed_state_audit.json")
            self.assertFalse(fixed["test_evaluated"])
            self.assertEqual(fixed["original_train"]["samples"], 8)
            self.assertEqual(fixed["fresh_train_states"]["samples"], 8)
            with patch("controller_architecture_study.run_convex_recovery.evaluate", side_effect=AssertionError("cache missed")):
                audit_run(run, base, "cpu")

    def test_summary_gap_uses_fixed_train_not_harder_training_mixture(self):
        with tempfile.TemporaryDirectory() as directory:
            output, base = Path(directory)/"new", Path(directory)/"old"
            run = output/"runs/features_recovery/plain_mlp_h4_w128_seed0"
            run.mkdir(parents=True)
            (run/"config.json").write_text(json.dumps({"training_samples": 153600, "maximum_optimizer_updates": 60000}))
            for name in ("best_model.pt", "last_model.pt"):
                (run/name).touch()
            (run/"completion.json").write_text(json.dumps({"epochs_completed": 200, "best_epoch": 190}))
            (run/"offline_metrics.json").write_text(json.dumps({"train": {"normalized_mse": .2},
                                                                "validation": {"normalized_mse": .1}}))
            (run/"fixed_state_audit.json").write_text(json.dumps({"original_train": {"normalized_mse": .01},
                                                                 "fresh_train_states": {"normalized_mse": .08}}))
            report = recovery.read_json(summarize(base, output))
            self.assertEqual(len(report["models"]), 8)
            row = next(row for row in report["models"] if row["status"] == "complete")
            self.assertAlmostEqual(row["gap_absolute"], .09)
            self.assertAlmostEqual(row["validation_train_ratio"], 10.)
            self.assertFalse(report["test_evaluated"])


if __name__ == "__main__":
    unittest.main()
