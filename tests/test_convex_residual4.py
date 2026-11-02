from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from controller_architecture_study import run_convex_residual4 as residual
from controller_architecture_study.compute_complexity import complexity
from controller_architecture_study.decision_dataset import atomic_json
from controller_architecture_study.models.factory import build_model
from tests.test_convex_scenic_falcon18 import manifest, completed


class ResidualComparisonTests(unittest.TestCase):
    def prepare(self, root):
        with patch.object(residual.sweep, "verify_arm", return_value=manifest()):
            jobs = residual.sweep.build_jobs(root/"base", root/"data/datasets/features_recovery", root/"plain")
        for _, config, run in jobs:
            if (config["hidden_layers"], config["hidden_width"]) in residual.SHAPES:
                completed(run, config)
        return jobs

    def test_four_matched_jobs_and_no_original_config_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plain_jobs = self.prepare(root)
            before = deepcopy(plain_jobs)
            with patch.object(residual.sweep, "build_jobs", return_value=plain_jobs), patch.object(residual, "OUTPUT", root/"residual"):
                jobs = residual.build_jobs()
                self.assertFalse((root/"residual").exists())
                self.assertEqual(plain_jobs, before)
                self.assertEqual([(c["hidden_layers"], c["hidden_width"]) for _, c, _ in jobs], list(residual.SHAPES))
                for name, config, run in jobs:
                    self.assertEqual(config["model_type"], "residual_mlp")
                    self.assertEqual(config["max_epochs"], 600)
                    self.assertEqual(config["maximum_optimizer_updates"], 180000)
                    self.assertEqual(config["evaluation_splits"], ["train", "validation"])
                    self.assertNotIn("test.npz", config["dataset_file_sha256"])
                    self.assertTrue(name.startswith("residual_mlp_"))
                    self.assertIn("models/residual_mlp.py", config["trainer_source_sha256"])
                    self.assertEqual(run.parent, root/"residual/runs")
                report = json.loads(residual.paired_report(jobs).read_text())
                self.assertEqual(len(report["models"]), 4)
                self.assertNotIn("residual_validation_mse", report["models"][0])
                summary = json.loads((root/"residual/comparison.json").read_text())
                self.assertEqual(summary["architectures_requested"], 4)
                self.assertEqual(summary["architecture_set"], "residual4_matched_plain")
                _, config, run = jobs[0]
                completed(run, config)
                atomic_json(run/"offline_metrics.json", {"train": {"normalized_mse": .05},
                                                         "validation": {"normalized_mse": .15}})
                atomic_json(run/"fixed_state_audit.json", {"original_train": {"normalized_mse": .04},
                                                          "fresh_train_states": {"normalized_mse": .14}})
                report = json.loads(residual.paired_report(jobs).read_text())
                self.assertEqual(report["models"][0]["status"], "complete")
                self.assertAlmostEqual(report["models"][0]["validation_reduction_percent"], 25.)
                self.assertFalse(report["test_evaluated"])

    def test_missing_plain_baseline_and_changed_metrics_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plain_jobs = self.prepare(root)
            with patch.object(residual.sweep, "build_jobs", return_value=plain_jobs), patch.object(residual, "OUTPUT", root/"residual"):
                jobs = residual.build_jobs()
                plain_run = Path(jobs[0][1]["plain_baseline"])
                atomic_json(plain_run/"offline_metrics.json", {"train": {}, "validation": {}})
                with self.assertRaisesRegex(ValueError, "metrics changed"):
                    residual.paired_report(jobs)
                with patch.object(residual.sweep, "run_state", return_value="new"), self.assertRaisesRegex(ValueError, "not complete"):
                    residual.build_jobs()

    def test_residual_forward_backward_and_equal_fc_parameter_count(self):
        torch.set_num_threads(1)
        for hidden in (4, 6, 21, 33):
            # Small width keeps the regression bounded; the full-width count uses the same formula.
            model = build_model(torch, model_type="residual_mlp", input_dim=237, output_dim=4,
                                hidden_layers=hidden, hidden_width=16)
            self.assertEqual(sum(p.numel() for p in model.parameters()), complexity(237, 4, hidden, 16)["parameter_count"])
            output = model(torch.randn(2, 237))
            self.assertEqual(tuple(output.shape), (2, 4))
            output.square().mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
