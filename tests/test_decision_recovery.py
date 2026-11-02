"""Temporary-data regressions; fake MPC, disabled optimizer, no production jobs."""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from controller_architecture_study import decision_dataset_96k as original
from controller_architecture_study import decision_recovery as recovery
from controller_architecture_study import run_recovery_ablation as runner
from tests.test_decision_dataset_96k import FakeTeacher


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base, self.output = Path(self.temp.name) / "base", Path(self.temp.name) / "added"
        with patch.object(original, "RefinedTeacher", FakeTeacher), patch.object(original, "ENVIRONMENTS", 4), \
                patch.object(original, "SPLITS", {"train": [0, 1], "validation": [2], "test": [3]}), \
                contextlib.redirect_stdout(io.StringIO()):
            original.generate(self.base, smoke=True)
            original.prepare(self.base)
        self.before = {str(p.relative_to(self.base)): recovery.sha256(p) for p in self.base.rglob("*") if p.is_file()}

    def generate(self, resume=False):
        with patch.object(recovery, "RefinedTeacher", FakeTeacher), contextlib.redirect_stdout(io.StringIO()):
            recovery.generate(self.base, self.output, resume=resume)

    def prepare(self):
        with contextlib.redirect_stdout(io.StringIO()):
            recovery.prepare(self.base, self.output)

    def source_models(self):
        split = json.loads((self.base / "dnn_dataset/split_manifest.json").read_text())
        for width in runner.WIDTHS:
            run = self.base / "runs" / f"plain_mlp_h4_w{width}_seed0"
            recovery.atomic_json(run / "config.json", {
                "input_dim": 300, "hidden_layers": 4, "hidden_width": width,
                "model_type": "plain_mlp", "activation": "relu", "residual": False,
                "dataset_source_sha256": split["source_manifest_sha256"]})
            (run / "best_model.pt").write_bytes(b"immutable-checkpoint-fixture")

    def test_resume_skip_committed_chunk_and_preserve_every_source_file(self):
        save = recovery.atomic_npz

        def interrupt(path, values):
            if path.name == "env01_route00.npz":
                raise KeyboardInterrupt()
            save(path, values)

        with patch.object(recovery, "atomic_npz", side_effect=interrupt), self.assertRaises(KeyboardInterrupt):
            self.generate()
        first = self.output / "chunks/env00_route00.npz"
        first_hash = recovery.sha256(first)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.prepare()
        self.generate(resume=True)
        self.assertEqual(first_hash, recovery.sha256(first))
        self.prepare()
        self.assertEqual(self.before, {str(p.relative_to(self.base)): recovery.sha256(p)
                                      for p in self.base.rglob("*") if p.is_file()})
        audit = json.loads((self.output / "data_audit.json").read_text())
        self.assertEqual(audit["added_train_samples"], 64)
        self.assertEqual(audit["train_samples"], 80)
        self.assertTrue(audit["validation_byte_identical"] and audit["normalization_byte_identical"])
        self.assertFalse((self.output / "dnn_dataset/test.npz").exists())
        digest = recovery.sha256(self.output / "dnn_dataset/split_manifest.json")
        self.generate(resume=True)
        self.prepare()
        self.assertEqual(digest, recovery.sha256(self.output / "dnn_dataset/split_manifest.json"))
        values = recovery.read_npz(first)
        self.assertEqual(np.bincount(values["augmentation_kind"]).tolist(), [4, 4, 4, 12, 8])
        self.assertEqual(len(np.unique(values["phase_seconds"][:12])), 1)
        self.assertEqual(len(np.unique(values["phase_seconds"][12:24])), 1)
        # Same clock/window, different states and freshly computed labels.
        self.assertGreater(len(np.unique(values["X"][:12], axis=0)), 1)
        np.testing.assert_allclose(values["U"][:, 1:], -values["X"][:, 10:13] * [1., -1., -1.] * .1, atol=1e-6)
        self.assertTrue(np.all(values["actual_speed"] <= 8. + 1e-10))

    def test_neighbor_center_decodes_all_ned_flu_coordinates(self):
        class ZeroNoise:
            def uniform(self, low, high, size=None):
                return np.zeros(size) if size else 0.

        source, _ = recovery.source_info(self.base)
        layout, rows = recovery.source_route(self.base, source, "env00_route00")
        planner = recovery.StoredRoutePlanner(layout)
        index = int(np.flatnonzero(rows["family_id"] == 7)[0])
        refs = planner.reference_window_for_trajectory(planner.trajectory,
            elapsed=float(rows["phase_seconds"][index]), dt=.1, horizon=21)
        state = recovery.sample_state(refs[0], ZeroNoise(), 4, rows["X"][index])
        np.testing.assert_allclose(recovery.build_features(state, refs), rows["X"][index], atol=3e-6)

    def test_heldout_files_never_loaded_by_augmentation(self):
        load = recovery.np.load

        def guarded(path, *args, **kwargs):
            self.assertNotIn(Path(path).name, ("validation.npz", "test.npz"))
            self.assertFalse(Path(path).name.startswith(("env02_", "env03_")))
            return load(path, *args, **kwargs)

        with patch.object(recovery.np, "load", side_effect=guarded):
            self.generate()
            self.prepare()

    def test_changed_source_or_completed_chunk_is_rejected(self):
        self.generate()
        chunk = self.output / "chunks/env00_route00.npz"
        chunk.write_bytes(chunk.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "changed completed"):
            self.generate(resume=True)
        with self.assertRaisesRegex(ValueError, "changed augmentation"):
            self.prepare()
        with self.assertRaisesRegex(ValueError, "separate"):
            with patch.object(recovery, "RefinedTeacher", FakeTeacher):
                recovery.generate(self.base, self.base)

    def test_four_architectures_three_arms_no_test_and_hash_binding(self):
        self.generate()
        self.prepare()
        self.source_models()
        jobs = runner.build_jobs(self.base, self.output, runner.ARMS, 300)
        self.assertEqual(len(jobs), 12)
        self.assertEqual({cfg["hidden_width"] for _, cfg, _ in jobs}, {64, 128, 256, 1024})
        for _, cfg, target in jobs:
            self.assertTrue(cfg["preserve_initial_best"])
            self.assertEqual(cfg["evaluation_splits"], ["train", "validation"])
            self.assertEqual(cfg["hidden_layers"], 4)
            self.assertIn(self.output, target.parents)
            self.assertEqual(cfg["dataset_source_sha256"],
                             json.loads((self.base / "dnn_dataset/split_manifest.json").read_text())["source_manifest_sha256"])
            if cfg["ablation_arm"] == "added_regularized":
                self.assertEqual(cfg["lr_schedule"], "cosine")
                self.assertGreater(cfg["weight_decay"], 0)
        path = self.output / "dnn_dataset/normalization.json"
        path.write_text(path.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "changed augmented"):
            runner.build_jobs(self.base, self.output, runner.ARMS, 300)

    def test_serial_runner_saves_completed_jobs_then_resumes_failure(self):
        self.source_models()
        calls = []

        def child(command, **kwargs):
            calls.append(command)
            target = Path(command[command.index("--output-dir") + 1])
            config = json.loads(Path(command[command.index("--config") + 1]).read_text())
            recovery.atomic_json(target / "config.json", config)
            (target / "last_model.pt").write_bytes(b"fixture")
            if len(calls) == 2:
                raise subprocess.CalledProcessError(1, command)
            (target / "best_model.pt").write_bytes(b"fixture")
            recovery.atomic_json(target / "completion.json", {"best_epoch": 0, "epochs_completed": 100})
            recovery.atomic_json(target / "initial_metrics.json", {"validation": {"normalized_mse": .03}})
            recovery.atomic_json(target / "offline_metrics.json", {"train": {"normalized_mse": .001},
                "validation": {"normalized_mse": .03, "by_family_id": {"7": {"normalized_mse": .1}}}})

        argv = ["runner", "--base", str(self.base), "--output", str(self.output), "--arm", "old_control", "--device", "cpu"]
        with patch("sys.argv", argv), patch.object(runner.subprocess, "run", side_effect=child), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(subprocess.CalledProcessError):
                runner.main()
            runner.main()
        self.assertEqual(len(calls), 5)
        self.assertIn("--resume", calls[2])
        report = json.loads((self.output / "runs/comparison.json").read_text())
        self.assertEqual(sum(row["status"] == "complete" for row in report["models"]), 4)
        self.assertFalse(report["test_evaluated"])


if __name__ == "__main__":
    unittest.main()
