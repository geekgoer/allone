"""Regression: new seed-zero models belong to their configured results folder."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from controller_architecture_study.summarize_followup import summarize


class FollowupLocationTests(unittest.TestCase):
    def test_new_seed_zero_and_matched_deep_residual_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results"
            (root / "protocol.json").write_text("{}", encoding="utf-8")
            config = root / "experiment.json"
            config.write_text(json.dumps({
                "protocol": "protocol.json",
                "plain_seed_repeats": [{"hidden_layers": 22, "hidden_width": 128, "seeds": [0]}],
                "residual_tests": [{"hidden_layers": 22, "hidden_width": 128, "seeds": [0]}],
            }), encoding="utf-8")
            for model_type, loss in (("plain_mlp", 0.02), ("residual_mlp", 0.01)):
                folder = results / f"{model_type}_h22_w128_seed0"
                folder.mkdir(parents=True)
                metrics = {"normalized_mse": loss, "physical_mse": loss * 2,
                           "p95_error": 0.1, "p99_error": 0.2}
                (folder / "offline_metrics.json").write_text(
                    json.dumps({"validation": metrics, "test": metrics}), encoding="utf-8")
                (folder / "complexity.json").write_text(
                    json.dumps({"parameter_count": 348932, "mac_count": 346112}), encoding="utf-8")
                (folder / "train_log.csv").write_text(
                    "epoch,train_normalized_mse,validation_normalized_mse\n1,0.03,0.02\n",
                    encoding="utf-8")
            with patch("controller_architecture_study.summarize_followup.COARSE", root / "absent_baseline"):
                rows, groups = summarize(config, results)
            self.assertEqual(len(rows), 2)
            self.assertEqual(len(groups), 2)
            self.assertTrue(all(Path(row["checkpoint_path"]).parent.parent == results for row in rows))
            report = (results / "analysis.md").read_text(encoding="utf-8")
            self.assertIn("Matched 22×128 seed comparison", report)
            self.assertIn("0.500", report)


if __name__ == "__main__":
    unittest.main()
