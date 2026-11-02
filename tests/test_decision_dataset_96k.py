"""96k sampling/resume regressions using temporary data and a fake teacher."""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from controller_architecture_study import decision_dataset_96k as data
from controller_architecture_study import train_decision_pilot as runner
from falcon_paper_fidelity.motion_primitive import QuinticMotionPrimitivePlanner


class FakeTeacher:
    def __init__(self):
        self.controller = type("Controller", (), {"library_path": Path(data.__file__)})()

    def query(self, state, refs):
        control = np.r_[10., np.clip((state.position - refs[0].position) * .1, -.5, .5)].astype(np.float32)
        return control, {"iterations": 4, "kkt": 1e-5, "action_delta": 1e-5, "solve_ms": 1.}


class Decision96kTests(unittest.TestCase):
    def test_edge_sampling_is_bounded_reproducible_and_focuses_edges(self):
        radius = np.array([.6, 1., .6])
        a, b = np.random.default_rng(15), np.random.default_rng(15)
        samples = np.array([data.edge_uniform(a, radius) for _ in range(3000)])
        repeated = np.array([data.edge_uniform(b, radius) for _ in range(3000)])
        np.testing.assert_array_equal(samples, repeated)
        self.assertTrue(np.all(np.abs(samples) <= radius))
        # 30% focused samples + 30% of the remaining 70% uniform samples ~= 51%.
        fraction = np.mean(np.abs(samples / radius) >= .7)
        self.assertGreater(fraction, .48)
        self.assertLess(fraction, .54)

    def test_each_full_group_has_four_geometries_and_different_c2_routes(self):
        planner = QuinticMotionPrimitivePlanner(data.PLANNER_CONFIG)
        layouts = []
        for index in range(20):
            trajectory, layout = data.route_for_group(48, index, 20, planner)
            layouts.append(layout)
            self.assertEqual(layout["environment_id"], 48)
            self.assertEqual(layout["route_id"], 960 + index)
            self.assertGreaterEqual(layout["parent_route_generator_id"], 200000)
            np.testing.assert_allclose(trajectory.evaluate(0.)[1], 0., atol=1e-12)
            for left, right in zip(trajectory.segments[:-1], trajectory.segments[1:]):
                for end, start in zip(left.primitive.evaluate(left.primitive.duration), right.primitive.evaluate(0.)):
                    np.testing.assert_allclose(end, start, atol=1e-9)
        self.assertEqual([sum(item["geometry_family"] == kind for item in layouts) for kind in range(4)], [5] * 4)
        self.assertEqual(len({json.dumps(item["gates"], sort_keys=True) for item in layouts}), 20)

    def test_independent_states_do_not_reset_the_reference_clock(self):
        planner = QuinticMotionPrimitivePlanner(data.PLANNER_CONFIG)
        trajectory, _ = data.route_for_group(0, 3, 20, planner)
        _, refs_a, phase_a = data.sample_problem(trajectory, planner, np.random.default_rng(3), 1)
        state_b, refs_b, phase_b = data.sample_problem(trajectory, planner, np.random.default_rng(3), 4)
        self.assertEqual(phase_a, phase_b)
        self.assertGreater(phase_b, 0.)
        for left, right in zip(refs_a, refs_b):
            np.testing.assert_array_equal(left.position, right.position)
        self.assertGreater(np.linalg.norm(state_b.position - refs_b[0].position), .08)
        for family in range(8):
            state, _, phase = data.sample_problem(trajectory, planner, np.random.default_rng(family), family)
            self.assertLessEqual(np.linalg.norm(state.velocity), (3.5 if family == 0 else 8.) + 1e-10)
            self.assertAlmostEqual(np.linalg.norm(state.quaternion), 1.)
            if family == 0 and phase == 0.:
                np.testing.assert_array_equal(state.velocity, np.zeros(3))

    def test_smoke_interrupt_resume_hashes_splits_and_train_only_normalization(self):
        original_save = data.atomic_npz
        with tempfile.TemporaryDirectory() as temp, patch.object(data, "RefinedTeacher", FakeTeacher), contextlib.redirect_stdout(io.StringIO()):
            output = Path(temp) / "pilot"

            def interrupted(path, values):
                if path.name == "env01_route00.npz":
                    raise KeyboardInterrupt()
                original_save(path, values)

            with patch.object(data, "atomic_npz", side_effect=interrupted), self.assertRaises(KeyboardInterrupt):
                data.generate(output, smoke=True)
            first = output / "chunks/env00_route00.npz"
            before = first.read_bytes()
            with self.assertRaisesRegex(ValueError, "incomplete"):
                data.prepare(output)
            data.generate(output, smoke=True, resume=True)
            self.assertEqual(first.read_bytes(), before)
            data.prepare(output)
            split_path = output / "dnn_dataset/split_manifest.json"
            saved = json.loads(split_path.read_text())
            self.assertEqual(saved["counts"], {"train": 384, "validation": 48, "test": 48})
            for field in ("environment_ids", "route_ids"):
                groups = [set(saved[field][split]) for split in ("train", "validation", "test")]
                self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
            norm = json.loads((output / "dnn_dataset/normalization.json").read_text())
            with np.load(output / "dnn_dataset/train.npz") as train:
                np.testing.assert_allclose(norm["x_mean"], train["X"].mean(axis=0, dtype=np.float64))
                np.testing.assert_allclose(norm["u_mean"], train["U"].mean(axis=0, dtype=np.float64))
            audit = json.loads((output / "data_audit.json").read_text())
            self.assertEqual(audit["samples"], 480)
            self.assertEqual(audit["repeat_checked_samples"], 60)
            self.assertEqual([item["samples"] for item in audit["families"].values()], [60] * 8)
            digest = data.sha256(split_path)
            data.generate(output, smoke=True, resume=True)
            data.prepare(output)
            self.assertEqual(digest, data.sha256(split_path))
            with self.assertRaisesRegex(ValueError, "unchanged"):
                data.generate(output, smoke=False, resume=True)
            route = output / "routes/env00_route00.json"
            route.write_text(route.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "changed completed"):
                data.generate(output, smoke=True, resume=True)
            with self.assertRaisesRegex(ValueError, "changed chunk/route"):
                data.prepare(output)

    def test_runner_96k_serial_saves_before_advancing_and_resumes(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            experiment = Path(temp) / "experiment"
            data.atomic_json(experiment / "dnn_dataset/split_manifest.json", {
                "version": data.VERSION, "contract": data.CONTRACT, "source_manifest_sha256": "fixture",
                "counts": {"train": 76800, "validation": 9600, "test": 9600}})
            data.atomic_json(experiment / "data_audit.json", {"status": "PASS", "samples": 96000})
            commands = []

            def fake_child(command, **kwargs):
                commands.append(command)
                output = Path(command[command.index("--output-dir") + 1])
                config_path = Path(command[command.index("--config") + 1])
                config = json.loads(config_path.read_text())
                self.assertEqual(config["evaluation_splits"], ["train", "validation"])
                data.atomic_json(output / "config.json", config)
                if len(commands) == 2:
                    raise subprocess.CalledProcessError(1, command)
                data.atomic_json(output / "completion.json", {"status": "complete"})
                data.atomic_json(output / "offline_metrics.json", {
                    "train": {"normalized_mse": .01}, "validation": {"normalized_mse": .02}})

            with patch.object(runner, "EXPERIMENT_96K", experiment), patch.object(runner, "ROOT", Path(temp)), \
                    patch("sys.argv", ["pilot", "--dataset-version", "96k", "--device", "cpu"]), \
                    patch.object(runner.subprocess, "run", side_effect=fake_child):
                with self.assertRaises(subprocess.CalledProcessError):
                    runner.main()
                comparison = json.loads((experiment / "runs/comparison.json").read_text())
                self.assertEqual(len(comparison["models"]), 1)
                runner.main()
            self.assertEqual(len(commands), 4)
            self.assertIn("--resume", commands[2])
            self.assertNotIn("--resume", commands[3])
            comparison = json.loads((experiment / "runs/comparison.json").read_text())
            self.assertEqual(len(comparison["models"]), 3)
            self.assertFalse(comparison["test_split_evaluated"])


if __name__ == "__main__":
    unittest.main()
