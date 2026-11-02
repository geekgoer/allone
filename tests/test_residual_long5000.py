"""Bounded CPU/control-flow checks; no real model/data updates or GPU training."""

import json
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from controller_architecture_study import train_controller as trainer
from controller_architecture_study.models.factory import build_model


class ResidualLongTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name)/"source.pt"
        self.output = Path(self.temp.name)/"new_run"
        self.config = dict(input_dim=237, output_dim=4, model_type="residual_mlp",
            activation="relu", residual=True, hidden_layers=3, hidden_width=8,
            seed=0, batch_size=4, learning_rate=1e-5, min_learning_rate=1e-6,
            lr_schedule="cosine", weight_decay=1e-5, max_epochs=3, min_epochs=1,
            patience=1, early_stopping=False, checkpoint_every=2,
            gradient_clip_norm=1., data_dir="mock_data_not_opened",
            init_checkpoint=str(self.source), restore_initial_optimizer=True,
            preserve_initial_best=True, evaluation_splits=["train", "validation"])
        model = build_model(torch, model_type="residual_mlp", input_dim=237,
                            output_dim=4, hidden_layers=3, hidden_width=8)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, foreach=False)
        for parameter in model.parameters():
            parameter.data.zero_()
            optimizer.state[parameter] = dict(step=torch.tensor(73.),
                exp_avg=torch.full_like(parameter, .02), exp_avg_sq=torch.full_like(parameter, .01))
        torch.save(dict(model_state=model.state_dict(), optimizer_state=optimizer.state_dict(),
            model_type="residual_mlp", input_dim=237, hidden_layers=3, hidden_width=8, epoch=595), self.source)
        self.source_bytes = self.source.read_bytes()
        self.splits = []

    def mocked_io(self, initial_mse=.25):
        stack = ExitStack()
        stack.enter_context(patch.object(torch.cuda, "is_available", return_value=False))
        step = stack.enter_context(patch.object(torch.optim.Adam, "step", autospec=True))
        stack.enter_context(patch.object(trainer, "load_normalization", return_value=dict(
            x_mean=np.zeros(237, np.float32), x_std=np.ones(237, np.float32),
            u_mean=np.zeros(4, np.float32), u_std=np.ones(4, np.float32))))

        def load(_path, split, _limit=None):
            self.splits.append(split)
            return np.zeros((4, 237), np.float32), np.ones((4, 4), np.float32)

        def evaluate(_model, _path, split, **kwargs):
            self.splits.append(split)
            return dict(samples=4, normalized_mse=initial_mse)

        stack.enter_context(patch.object(trainer, "load_split", side_effect=load))
        stack.enter_context(patch.object(trainer, "evaluate", side_effect=evaluate))
        return stack, step

    def test_no_early_stopping_and_initial_best_retained(self):
        context, step = self.mocked_io()
        with context:
            trainer.train(self.config, self.output, device="cpu")
        self.assertEqual(step.call_count, 3)
        saved = torch.load(self.output/"last_model.pt", weights_only=False)
        self.assertEqual(saved["epoch"], 3)
        self.assertEqual(saved["stale"], 3)
        self.assertEqual(torch.load(self.output/"best_model.pt", weights_only=False)["epoch"], 0)
        self.assertNotIn("test", self.splits)
        self.assertEqual(self.source.read_bytes(), self.source_bytes)

    def test_adam_moments_restored_but_new_schedule_applied(self):
        context, _ = self.mocked_io()
        with context:
            trainer.train(self.config, self.output, device="cpu")
        saved = torch.load(self.output/"last_model.pt", weights_only=False)
        for state in saved["optimizer_state"]["state"].values():
            self.assertEqual(state["step"].item(), 73)
            torch.testing.assert_close(state["exp_avg"], torch.full_like(state["exp_avg"], .02))
        group = saved["optimizer_state"]["param_groups"][0]
        self.assertAlmostEqual(group["lr"], 1e-6)
        self.assertAlmostEqual(group["weight_decay"], 1e-5)

    def test_checkpoint_interval_and_final_epoch_are_saved(self):
        original_save, saved_epochs = trainer._save_checkpoint, []

        def save(path, module, payload):
            if path.name == "last_model.pt":
                saved_epochs.append(payload["epoch"])
            original_save(path, module, payload)

        context, _ = self.mocked_io()
        with context, patch.object(trainer, "_save_checkpoint", side_effect=save):
            trainer.train(self.config, self.output, device="cpu")
        self.assertEqual(saved_epochs, [0, 2, 3])

    def test_improvement_is_saved_before_the_periodic_snapshot(self):
        self.config["checkpoint_every"] = 10
        context, step = self.mocked_io(initial_mse=2.)
        step.side_effect = [None, KeyboardInterrupt()]
        with context, self.assertRaises(KeyboardInterrupt):
            trainer.train(self.config, self.output, device="cpu")
        self.assertEqual(torch.load(self.output/"last_model.pt", weights_only=False)["epoch"], 1)
        self.assertEqual(torch.load(self.output/"best_model.pt", weights_only=False)["epoch"], 1)

    def test_resume_ignores_patience_and_continues_saved_cosine_clock(self):
        context, step = self.mocked_io()
        step.side_effect = [None, None, KeyboardInterrupt()]
        with context, self.assertRaises(KeyboardInterrupt):
            trainer.train(self.config, self.output, device="cpu")
        self.assertEqual(torch.load(self.output/"last_model.pt", weights_only=False)["epoch"], 2)
        context, step = self.mocked_io()
        with context:
            trainer.train(self.config, self.output, device="cpu", resume=True)
        self.assertEqual(step.call_count, 1)
        saved = torch.load(self.output/"last_model.pt", weights_only=False)
        self.assertEqual([row["epoch"] for row in saved["history"]], [1, 2, 3])
        self.assertAlmostEqual(saved["history"][-1]["learning_rate"], 1e-6)
        context, step = self.mocked_io()
        with context:
            trainer.train(self.config, self.output, device="cpu", resume=True)
        step.assert_not_called()

    def test_legacy_early_stop_default_is_unchanged(self):
        self.config.pop("early_stopping")
        context, step = self.mocked_io()
        with context:
            trainer.train(self.config, self.output, device="cpu")
        self.assertEqual(step.call_count, 1)
        self.assertEqual(torch.load(self.output/"last_model.pt", weights_only=False)["epoch"], 1)

    def test_invalid_flags_are_rejected(self):
        self.config["early_stopping"] = "false"
        with self.assertRaisesRegex(ValueError, "must be booleans"):
            trainer.train(self.config, self.output, device="cpu")

    def test_production_config_is_one_frozen_model_and_5000_epochs(self):
        config = json.loads((trainer.ROOT/"controller_architecture_study/configs/residual33x1024_long5000_v1.json").read_text())
        self.assertEqual((config["input_dim"], config["hidden_layers"], config["hidden_width"]), (237, 33, 1024))
        self.assertEqual(config["max_epochs"], 5000)
        self.assertIs(config["early_stopping"], False)
        self.assertEqual(config["evaluation_splits"], ["train", "validation"])
        self.assertAlmostEqual(trainer.learning_rate_for_epoch(config, 1), 1e-5)
        self.assertAlmostEqual(trainer.learning_rate_for_epoch(config, 5000), 1e-6)


if __name__ == "__main__":
    unittest.main()
