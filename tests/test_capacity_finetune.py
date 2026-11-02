"""Recovery tests with synthetic data and a disabled optimizer step.

No real datasets, trained models, AirSim connections or GPU training are used.
"""

import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from controller_architecture_study import train_controller as trainer
from controller_architecture_study.models.factory import build_model
from controller_architecture_study.run_capacity_finetune import build_jobs, job_state, summarize


class CapacityFinetuneTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source" / "best_model.pt"
        self.source.parent.mkdir()
        model = build_model(torch, model_type="plain_mlp", input_dim=12, output_dim=4,
                            hidden_layers=1, hidden_width=8)
        for parameter in model.parameters():
            parameter.data.zero_()
        torch.save({"model_state": model.state_dict(), "hidden_layers": 1,
                    "hidden_width": 8, "model_type": "plain_mlp", "epoch": 42}, self.source)
        self.original = self.source.read_bytes()
        self.config = {
            "input_dim": 12, "output_dim": 4, "model_type": "plain_mlp",
            "activation": "relu", "residual": False, "hidden_layers": 1, "hidden_width": 8,
            "data_dir": "unused_synthetic_fixture", "seed": 0, "learning_rate": 1e-5,
            "batch_size": 4, "max_epochs": 2, "min_epochs": 1, "patience": 1,
            "gradient_clip_norm": 1.0, "preserve_initial_best": True,
            "evaluation_splits": ["train", "validation"], "init_checkpoint": str(self.source),
        }
        (self.source.parent / "config.json").write_text(json.dumps(self.config), encoding="utf-8")
        self.target = self.root / "new_run"
        self.splits = []

    def mocked_io(self, initial_loss=0.25, interrupt_final=False):
        stack = ExitStack()
        stack.enter_context(patch.object(torch.cuda, "is_available", return_value=False))
        # Exercise checkpoint/control flow without taking any training updates.
        step = stack.enter_context(patch.object(torch.optim.Adam, "step", autospec=True))
        norm = {"x_mean": np.zeros(12, np.float32), "x_std": np.ones(12, np.float32),
                "u_mean": np.zeros(4, np.float32), "u_std": np.ones(4, np.float32)}
        stack.enter_context(patch.object(trainer, "load_normalization", return_value=norm))

        def load(_directory, split, _limit=None):
            self.splits.append(split)
            return np.zeros((4, 12), np.float32), np.ones((4, 4), np.float32)

        calls = []

        def evaluate(_model, _directory, split, **kwargs):
            self.splits.append(split)
            calls.append(split)
            if interrupt_final and len(calls) == 3:
                raise RuntimeError("simulated interruption during final evaluation")
            return {"samples": 4, "normalized_mse": initial_loss}

        stack.enter_context(patch.object(trainer, "load_split", side_effect=load))
        stack.enter_context(patch.object(trainer, "evaluate", side_effect=evaluate))
        return stack, step

    def test_initial_best_is_retained_and_test_split_is_never_read(self):
        context, step = self.mocked_io()
        with context:
            trainer.train(self.config, self.target, device="cpu")
        selected = torch.load(self.target / "best_model.pt", weights_only=False)
        self.assertEqual(selected["epoch"], 0)
        self.assertEqual(selected["best_validation_normalized_mse"], 0.25)
        self.assertEqual(step.call_count, 1)
        self.assertNotIn("test", self.splits)
        self.assertEqual(self.source.read_bytes(), self.original)
        self.assertEqual(job_state(self.config, self.target), "complete")
        rows = summarize([("small", self.config, self.target)], self.root)
        self.assertEqual(rows[0]["validation_improvement_percent"], 0)
        self.assertEqual(rows[0]["best_epoch"], 0)

    def test_resume_after_early_stop_does_not_train_an_extra_epoch(self):
        context, _ = self.mocked_io(interrupt_final=True)
        with context, self.assertRaisesRegex(RuntimeError, "simulated interruption"):
            trainer.train(self.config, self.target, device="cpu")
        self.assertEqual(job_state(self.config, self.target), "resume")
        context, step = self.mocked_io()
        with context:
            trainer.train(self.config, self.target, resume=True, device="cpu")
        step.assert_not_called()
        summary = json.loads((self.target / "completion.json").read_text())
        self.assertEqual(summary["epochs_completed"], 1)

    def test_interrupted_initialization_can_be_restarted(self):
        context, _ = self.mocked_io()
        with context, patch.object(trainer, "evaluate", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                trainer.train(self.config, self.target, device="cpu")
        self.assertTrue((self.target / "config.json").exists())
        self.assertFalse((self.target / "last_model.pt").exists())
        context, _ = self.mocked_io()
        with context:
            trainer.train(self.config, self.target, resume=True, device="cpu")
        self.assertEqual(job_state(self.config, self.target), "complete")

    def test_resume_repairs_best_if_shutdown_occurs_between_checkpoint_saves(self):
        self.config["max_epochs"] = 1
        original_save = trainer._save_checkpoint

        def interrupted_save(path, torch_module, payload):
            if path.name == "best_model.pt" and payload["epoch"] == 1:
                raise RuntimeError("simulated shutdown before best save")
            original_save(path, torch_module, payload)

        context, _ = self.mocked_io(initial_loss=2.0)
        with context, patch.object(trainer, "_save_checkpoint", side_effect=interrupted_save):
            with self.assertRaisesRegex(RuntimeError, "simulated shutdown"):
                trainer.train(self.config, self.target, device="cpu")
        self.assertEqual(torch.load(self.target / "best_model.pt", weights_only=False)["epoch"], 0)
        context, step = self.mocked_io()
        with context:
            trainer.train(self.config, self.target, resume=True, device="cpu")
        step.assert_not_called()
        self.assertEqual(torch.load(self.target / "best_model.pt", weights_only=False)["epoch"], 1)

    def test_resume_from_epoch_zero_writes_only_one_csv_header(self):
        context, _ = self.mocked_io()
        with context, patch.object(trainer, "load_split", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                trainer.train(self.config, self.target, device="cpu")
        self.assertEqual(torch.load(self.target / "last_model.pt", weights_only=False)["epoch"], 0)
        context, _ = self.mocked_io()
        with context:
            trainer.train(self.config, self.target, resume=True, device="cpu")
        lines = (self.target / "train_log.csv").read_text().splitlines()
        self.assertEqual(sum(line.startswith("epoch,") for line in lines), 1)

    def test_source_directory_cannot_be_used_as_warmstart_output(self):
        with self.assertRaisesRegex(ValueError, "must not overwrite"):
            trainer.train(self.config, self.source.parent, device="cpu")
        self.assertEqual(self.source.read_bytes(), self.original)

    def test_legacy_default_still_evaluates_validation_and_test(self):
        for key in ("preserve_initial_best", "evaluation_splits", "init_checkpoint"):
            self.config.pop(key)
        context, _ = self.mocked_io()
        with context:
            result = trainer.train(self.config, self.target, device="cpu")
        self.assertEqual(set(result), {"validation", "test"})

    def test_cosine_and_l2_resume_without_resetting_the_learning_rate(self):
        self.config.update(lr_schedule="cosine", min_learning_rate=1e-6,
                           weight_decay=1e-5, max_epochs=4, patience=10)
        observed = []

        def disabled_step(optimizer, *args, **kwargs):
            observed.append(optimizer.param_groups[0]["lr"])
            self.assertEqual(optimizer.param_groups[0]["weight_decay"], 1e-5)
            if len(observed) == 2:
                raise KeyboardInterrupt()

        context, step = self.mocked_io()
        step.side_effect = disabled_step
        with context:
            with self.assertRaises(KeyboardInterrupt):
                trainer.train(self.config, self.target, device="cpu")
        context, step = self.mocked_io()
        step.side_effect = disabled_step
        with context:
            trainer.train(self.config, self.target, resume=True, device="cpu")
        expected = [trainer.learning_rate_for_epoch(self.config, epoch) for epoch in (1, 2, 2, 3, 4)]
        np.testing.assert_allclose(observed, expected)
        saved = torch.load(self.target / "last_model.pt", weights_only=False)
        self.assertEqual(saved["epoch"], 4)
        self.assertAlmostEqual(saved["optimizer_state"]["param_groups"][0]["lr"], 1e-6)
        self.assertEqual([row["epoch"] for row in saved["history"]], [1, 2, 3, 4])
        self.assertAlmostEqual(saved["history"][-1]["learning_rate"], 1e-6)
        self.assertEqual(self.source.read_bytes(), self.original)
        self.assertNotIn("test", self.splits)

    def test_changed_source_or_configuration_is_not_silently_skipped(self):
        context, _ = self.mocked_io()
        with context:
            trainer.train(self.config, self.target, device="cpu")
        changed = {**self.config, "learning_rate": 3e-6}
        with self.assertRaisesRegex(ValueError, "choose a new results directory"):
            job_state(changed, self.target)

    def test_job_preparation_is_read_only_and_hashes_source(self):
        config_path = self.root / "suite.json"
        config_path.write_text(json.dumps({"overrides": {"learning_rate": 1e-5}, "jobs": [
            {"name": "small", "checkpoint": str(self.source)}]}), encoding="utf-8")
        results = self.root / "not_created"
        jobs = build_jobs(config_path, results)
        self.assertFalse(results.exists())
        self.assertEqual(len(jobs[0][1]["source_checkpoint_sha256"]), 64)
        self.assertEqual(self.source.read_bytes(), self.original)

    def test_corrected_scenic_layer_counts_do_not_reuse_old_experiments(self):
        path = trainer.ROOT / "controller_architecture_study/configs/scenic_table3_v2_layercount_corrected.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(len(config["architectures"]), 12)
        for item in config["architectures"]:
            self.assertEqual(2 * item["hidden_layers"] + 1, item["paper_L"])
        self.assertEqual(config["already_trained_in_previous_coarse_sweep"], [])


if __name__ == "__main__":
    unittest.main()
