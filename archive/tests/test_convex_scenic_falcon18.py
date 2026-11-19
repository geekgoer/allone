import csv
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from controller_architecture_study import run_convex_scenic_falcon18 as sweep
from controller_architecture_study.decision_dataset import atomic_json


def manifest():
    return {"arm": "features_recovery", "counts": {"train": 153600, "validation": 9600, "test": 9600},
            "source_manifest_sha256": "source", "augmentation_manifest_sha256": "augmentation",
            "artifact_sha256": {"train.npz": "train", "validation.npz": "val",
                                "normalization.json": "norm", "test.npz": "test"}}


def completed(run, config):
    atomic_json(run/"config.json", config)
    for name in ("best_model.pt", "last_model.pt"):
        (run/name).touch()
    metrics = {"train": {"normalized_mse": .1}, "validation": {"normalized_mse": .2}}
    atomic_json(run/"offline_metrics.json", metrics)
    atomic_json(run/"completion.json", {"epochs_completed": config["max_epochs"], "best_epoch": 550})
    with (run/"train_log.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ("epoch", "best_validation_normalized_mse", "mean_grad_norm_before_clip", "clip_fraction")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, 601):
            writer.writerow(dict(zip(fields, (epoch, .25 if epoch <= 500 else .2, .3, .1))))


class SweepTests(unittest.TestCase):
    def build(self, directory, **kwargs):
        root = Path(directory)
        with patch.object(sweep, "verify_arm", return_value=manifest()):
            return sweep.build_jobs(root/"base", root/"recovery/datasets/features_recovery", root/"new", **kwargs)

    def test_catalog_has_correct_twelve_plus_six_sorted_by_237d_parameters(self):
        rows = sweep.architectures()
        self.assertEqual(len(rows), 18)
        self.assertEqual(sum(row["group"] == "SCENIC" for row in rows), 12)
        self.assertEqual(sum(row["group"] == "FALCON" for row in rows), 6)
        self.assertEqual({(row["hidden_layers"], row["width"]) for row in rows if row["group"] == "FALCON"},
                         {(hidden, width) for hidden in (9, 21, 33) for width in (512, 1024)})
        costs = [sweep.complexity(237, 4, row["hidden_layers"], row["width"])["parameter_count"] for row in rows]
        self.assertEqual(costs, sorted(costs))
        for row in rows:
            self.assertEqual(row["paper_layers"], 2*row["hidden_layers"]+1 if row["group"] == "SCENIC"
                             else row["hidden_layers"]+1)

    def test_jobs_do_not_write_or_halve_budget_or_warm_start_or_use_test(self):
        with tempfile.TemporaryDirectory() as directory:
            jobs = self.build(directory)
            self.assertEqual(list(Path(directory).iterdir()), [])
            self.assertEqual(len(jobs), 18)
            for _, config, run in jobs:
                self.assertEqual((config["max_epochs"], config["min_epochs"], config["patience"]), (600, 600, 600))
                self.assertEqual(config["maximum_optimizer_updates"], 180000)
                self.assertEqual(config["evaluation_splits"], ["train", "validation"])
                self.assertNotIn("test.npz", config["dataset_file_sha256"])
                self.assertNotIn("init_checkpoint", config)
                self.assertEqual(config["input_dim"], 237)
                self.assertEqual(run.parent, Path(directory)/"new/runs")

    def test_invalid_seeds_smoke_data_and_overwriting_old_area_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                self.build(directory, seeds=(0, 0))
            with self.assertRaises(ValueError):
                self.build(directory, epochs=0)
            root = Path(directory)
            with self.assertRaises(ValueError):
                sweep.build_jobs(root/"base", root/"recovery/datasets/features_recovery", root/"recovery/runs/new")
            bad = manifest()
            bad["counts"]["train"] = 80
            with patch.object(sweep, "verify_arm", return_value=bad), self.assertRaises(ValueError):
                sweep.build_jobs(root/"base", root/"recovery/datasets/features_recovery", root/"new")

    def test_summary_distinguishes_three_gaps_and_never_fills_missing_with_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            jobs = self.build(directory)
            _, config, run = jobs[0]
            completed(run, config)
            path = sweep.summarize(jobs, Path(directory)/"new")
            report = json.loads(path.read_text())
            self.assertEqual(report["models"][0]["status"], "trained_pending_audit")
            self.assertNotIn("validation_mse", report["models"][1])
            atomic_json(run/"fixed_state_audit.json", {"original_train": {"normalized_mse": .05},
                                                       "fresh_train_states": {"normalized_mse": .19}})
            report = json.loads(sweep.summarize(jobs, Path(directory)/"new").read_text())
            row = report["models"][0]
            self.assertEqual(report["runs_completed"], 1)
            self.assertFalse(report["test_evaluated"])
            self.assertAlmostEqual(row["gap_mixture_absolute"], .1)
            self.assertAlmostEqual(row["gap_absolute"], .15)
            self.assertAlmostEqual(row["validation_fresh_ratio"], .2/.19)
            self.assertAlmostEqual(row["convergence"]["best_validation_improvement_percent"], 20.)

    def test_serial_queue_skips_completed_resumes_partial_and_saves_before_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = self.build(directory)[:3]
            completed(jobs[0][2], jobs[0][1])
            atomic_json(jobs[1][2]/"config.json", jobs[1][1])
            commands = []

            def process(command, **kwargs):
                commands.append(command)
                self.assertTrue(kwargs["check"])
                if len(commands) == 2:
                    raise subprocess.CalledProcessError(1, command)
                completed(jobs[1][2], jobs[1][1])

            def audit(run, base, device):
                self.assertEqual(device, "cpu")
                atomic_json(run/"fixed_state_audit.json", {"original_train": {"normalized_mse": .05},
                                                          "fresh_train_states": {"normalized_mse": .19}})

            with patch.object(sweep.subprocess, "run", side_effect=process), patch.object(sweep, "audit_run", side_effect=audit):
                with self.assertRaises(subprocess.CalledProcessError):
                    sweep.execute(jobs, root/"base", root/"new", "cuda")
            self.assertEqual(len(commands), 2)
            self.assertIn("--resume", commands[0])
            self.assertNotIn("--resume", commands[1])
            self.assertNotIn(str(jobs[0][2]), str(commands))
            report = json.loads((root/"new/comparison.json").read_text())
            self.assertEqual(report["runs_completed"], 2)
            self.assertEqual(report["models"][2]["status"], "not_started")
            changed = dict(jobs[0][1], max_epochs=800)
            with self.assertRaises(ValueError):
                sweep.run_state(changed, jobs[0][2])


if __name__ == "__main__":
    unittest.main()
