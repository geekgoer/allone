import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from mpc_controller import MPCReference


ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = ROOT / "experiments" / "paper_track_p_12d_5ms_v1" / "fpga_shadow_historical5ms"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, MODULE_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CLOSED_LOOP = _load("falcon_paper_closed_loop", "run_fpga_closed_loop.py")
DATASET = _load("falcon_paper_dataset", "build_falcon_svr_dataset.py")
SWEEP = _load("falcon_paper_sweep", "run_paper_style_svr_sweep.py")


class FalconPaperMetricTests(unittest.TestCase):
    def test_quaternion_sign_equivalence_has_zero_orientation_error(self):
        raw = SimpleNamespace(
            position=np.zeros(3),
            velocity=np.zeros(3),
            quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        reference = MPCReference(
            np.zeros(3),
            orientation=np.array([-1.0, 0.0, 0.0, 0.0]),
        )
        actual, target, error_squared = CLOSED_LOOP._paper_state_pair(raw, reference)
        self.assertEqual(actual.shape, (10,))
        self.assertEqual(target.shape, (10,))
        np.testing.assert_allclose(error_squared, np.zeros(10))

    def test_k20_windows_are_averaged_before_reciprocal(self):
        summary = CLOSED_LOOP._paper_metric_summary([2.0] * 43)
        self.assertEqual(summary["paper_horizon_K"], 20)
        self.assertEqual(summary["paper_window_samples"], 21)
        self.assertEqual(summary["paper_complete_windows"], 2)
        self.assertEqual(summary["paper_discarded_samples"], 1)
        np.testing.assert_allclose(summary["paper_window_costs"], [42.0, 42.0])
        self.assertAlmostEqual(summary["paper_route_cost_mean"], 42.0)
        self.assertAlmostEqual(summary["J_paper_style"], 1.0 / 42.0)

    def test_dataset_accepts_only_valid_paper_metric(self):
        models = {
            1: {
                "model_name": "depth10_width512",
                "depth": 10,
                "width": 512,
                "parameter_count": 1,
                "beta_test_physical_mse": 0.2,
                "beta_test_normalized_mse": 0.1,
                "c_inverse_beta": 10.0,
            }
        }
        trial = {
            "model_id": 1,
            "scenario_id": "S01",
            "completed": True,
            "fallback_count": 0,
            "deadline_miss_count": 0,
            "timeout_count": 0,
            "R_max_ms": 5.0,
            "paper_metric_version": CLOSED_LOOP.PAPER_METRIC_VERSION,
            "paper_horizon_K": 20,
            "paper_window_samples": 21,
            "paper_complete_windows": 2,
            "paper_used_samples": 42,
            "paper_discarded_samples": 1,
            "paper_window_costs": [4.0, 4.0],
            "paper_route_cost_mean": 4.0,
            "J_paper_style": 0.25,
        }
        row = DATASET._make_row(
            input_dir=Path("paper_run"),
            run_manifest={"experiment_id": "test", "period_ms": 50},
            trial=trial,
            models=models,
        )
        self.assertTrue(row["valid_for_svr"])
        self.assertEqual(row["J_paper_style"], 0.25)
        trial.pop("J_paper_style")
        invalid = DATASET._make_row(
            input_dir=Path("paper_run"),
            run_manifest={"experiment_id": "test", "period_ms": 50},
            trial=trial,
            models=models,
        )
        self.assertFalse(invalid["valid_for_svr"])

    def test_default_sweep_plan_has_40_unique_runs_and_7200_episodes(self):
        periods = [30, 40, 50, 60, 70, 80, 90, 100]
        delay_fractions = [0.0, 0.1, 0.2, 0.3, 0.4]
        plan = SWEEP._build_plan(periods, delay_fractions)
        self.assertEqual(len(plan), 40)
        self.assertEqual(len({item["run_id"] for item in plan}), 40)
        self.assertEqual(len(plan) * 6 * 6 * 5, 7200)

    def test_sweep_run_status_requires_every_episode_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            summary_path = run_dir / "fpga_closed_loop_summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "paper_metric": {"version": SWEEP.PAPER_METRIC_VERSION},
                        "episodes_requested": 2,
                        "episodes_completed": 1,
                        "trials": [{"completed": True}, {"completed": False}],
                    }
                ),
                encoding="utf-8",
            )
            self.assertFalse(SWEEP._run_status(run_dir, 2)["complete"])
            summary_path.write_text(
                json.dumps(
                    {
                        "paper_metric": {"version": SWEEP.PAPER_METRIC_VERSION},
                        "episodes_requested": 2,
                        "episodes_completed": 2,
                        "trials": [{"completed": True}, {"completed": True}],
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(SWEEP._run_status(run_dir, 2)["complete"])
            summary_path.write_text(
                json.dumps(
                    {
                        "episodes_requested": 2,
                        "episodes_completed": 2,
                        "trials": [{"completed": True}, {"completed": True}],
                    }
                ),
                encoding="utf-8",
            )
            self.assertFalse(SWEEP._run_status(run_dir, 2)["complete"])

    def test_sweep_manifest_rejects_configuration_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = Path(temporary) / "sweep_manifest.json"
            configuration = {"periods_ms": [50], "delay_fractions": [0.0]}
            SWEEP._load_or_create_manifest(manifest_path, configuration)
            with self.assertRaises(ValueError):
                SWEEP._load_or_create_manifest(
                    manifest_path,
                    {"periods_ms": [50], "delay_fractions": [0.1]},
                )


if __name__ == "__main__":
    unittest.main()
