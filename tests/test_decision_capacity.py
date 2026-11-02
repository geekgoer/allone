import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from controller_architecture_study import train_decision_pilot as runner


class DecisionCapacityTests(unittest.TestCase):
    def fixture(self, root):
        experiment = root / "experiment"
        runner.atomic_json(experiment / "dnn_dataset/split_manifest.json", {
            "version": runner.VERSION_96K, "contract": runner.CONTRACT,
            "source_manifest_sha256": "fixture",
            "counts": {"train": 76800, "validation": 9600, "test": 9600}})
        runner.atomic_json(experiment / "data_audit.json", {"status": "PASS", "samples": 96000})
        return experiment

    def complete_child(self, command, **kwargs):
        output = Path(command[command.index("--output-dir") + 1])
        config = json.loads(Path(command[command.index("--config") + 1]).read_text())
        self.assertEqual(config["evaluation_splits"], ["train", "validation"])
        runner.atomic_json(output / "config.json", config)
        runner.atomic_json(output / "completion.json", {"epochs_completed": 400, "best_epoch": 300})
        runner.atomic_json(output / "offline_metrics.json", {
            "train": {"normalized_mse": .01}, "validation": {"normalized_mse": .02}})

    def test_capacity_reuses_pilot_preserves_summary_and_resumes_failed_child(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            experiment = self.fixture(root)
            with patch.object(runner, "ROOT", root), patch.object(runner, "EXPERIMENT_96K", experiment), \
                    patch("sys.argv", ["pilot", "--dataset-version", "96k"]), \
                    patch.object(runner.subprocess, "run", side_effect=self.complete_child):
                runner.main()
            pilot_before = {path: path.read_bytes() for path in (experiment / "runs").rglob("*.json")}
            commands = []
            fail_once = [True]

            def child(command, **kwargs):
                commands.append(command)
                output = Path(command[command.index("--output-dir") + 1])
                if output.name == "plain_mlp_h4_w256_seed0" and fail_once[0]:
                    fail_once[0] = False
                    config = json.loads(Path(command[command.index("--config") + 1]).read_text())
                    runner.atomic_json(output / "config.json", config)
                    raise subprocess.CalledProcessError(1, command)
                self.complete_child(command, **kwargs)

            with patch.object(runner, "ROOT", root), patch.object(runner, "EXPERIMENT_96K", experiment), \
                    patch("sys.argv", ["pilot", "--dataset-version", "96k", "--grid", "capacity"]), \
                    patch.object(runner.subprocess, "run", side_effect=child):
                with self.assertRaises(subprocess.CalledProcessError):
                    runner.main()
                partial = json.loads((experiment / "runs/comparison_capacity.json").read_text())
                self.assertEqual(len(partial["models"]), 2)
                runner.main()
                self.assertEqual(len(commands), 7)  # Six new models, plus one failed attempt.
                self.assertIn("--resume", commands[2])
                runner.main()
                self.assertEqual(len(commands), 7)
            for path, saved in pilot_before.items():
                self.assertEqual(path.read_bytes(), saved)
            report = json.loads((experiment / "runs/comparison_capacity.json").read_text())
            self.assertFalse(report["test_split_evaluated"])
            self.assertEqual(len(report["models"]), 9)
            for model in report["models"]:
                self.assertEqual(model["complexity"]["fp32_weight_bytes"],
                                 model["complexity"]["parameter_count"] * 4)

    def test_capacity_dry_run_does_not_create_runs_or_start_children(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            experiment = self.fixture(root)
            with patch.object(runner, "ROOT", root), patch.object(runner, "EXPERIMENT_96K", experiment), \
                    patch("sys.argv", ["pilot", "--dataset-version", "96k", "--grid", "capacity", "--dry-run"]), \
                    patch.object(runner.subprocess, "run") as child:
                runner.main()
                child.assert_not_called()
            self.assertFalse((experiment / "runs").exists())


if __name__ == "__main__":
    unittest.main()
