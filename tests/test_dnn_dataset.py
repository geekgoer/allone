import unittest
import tempfile
from pathlib import Path

import numpy as np

from prepare_dnn_dataset import environment_split_masks, normalization_from_train, prepare


class DnnDatasetTests(unittest.TestCase):
    def test_environment_split_is_disjoint_and_80_10_10(self):
        environment_id = np.repeat(np.arange(40), 7)
        masks = environment_split_masks(environment_id)
        membership = sum(mask.astype(np.int8) for mask in masks.values())
        np.testing.assert_array_equal(membership, np.ones_like(membership))
        self.assertEqual(int(masks["train"].sum()), 224)
        self.assertEqual(int(masks["validation"].sum()), 28)
        self.assertEqual(int(masks["test"].sum()), 28)

    def test_normalization_uses_safe_nonzero_scales(self):
        x = np.ones((4, 12))
        u = np.ones((4, 4))
        normalization = normalization_from_train(x, u)
        self.assertTrue(np.all(normalization["x_std"] > 0))
        self.assertTrue(np.all(normalization["u_std"] > 0))

    def test_prepare_supports_path_aware_25_dimensional_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows_per_environment = 5
            environment_id = np.repeat(np.arange(10), rows_per_environment)
            rows = len(environment_id)
            source = root / "teacher.npz"
            np.savez_compressed(
                source,
                X=np.arange(rows * 25, dtype=np.float32).reshape(rows, 25),
                U=np.arange(rows * 4, dtype=np.float32).reshape(rows, 4),
                environment_id=environment_id,
                episode_id=environment_id,
                initial_state_id=np.zeros(rows, dtype=np.int64),
                sample_family=np.arange(rows, dtype=np.int64) % 3,
            )
            output = root / "prepared"
            manifest = prepare(source, output)
            self.assertEqual(manifest["input_dimensions"], 25)
            self.assertEqual(manifest["splits"]["train"]["samples"], 40)
            with np.load(output / "train.npz", allow_pickle=False) as archive:
                self.assertEqual(archive["X"].shape, (40, 25))
                self.assertIn("sample_family", archive.files)

    def test_prepare_propagates_track_p_input_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_root = root / "teacher"
            final_root = dataset_root / "final"
            final_root.mkdir(parents=True)
            rows_per_environment = 3
            environment_id = np.repeat(np.arange(10), rows_per_environment)
            rows = len(environment_id)
            source = final_root / "teacher.npz"
            features = [f"track_p_feature_{index}" for index in range(12)]
            np.savez_compressed(
                source,
                X=np.arange(rows * 12, dtype=np.float32).reshape(rows, 12),
                U=np.arange(rows * 4, dtype=np.float32).reshape(rows, 4),
                environment_id=environment_id,
                episode_id=environment_id,
                initial_state_id=np.zeros(rows, dtype=np.int64),
            )
            (dataset_root / "manifest.json").write_text(
                __import__("json").dumps({
                    "input_dimensions": 12,
                    "input_features": features,
                    "input_contract": "track-p-test-contract",
                    "strategy": "track-p-test-teacher",
                }),
                encoding="utf-8",
            )
            output = root / "prepared"
            manifest = prepare(source, output)
            self.assertEqual(manifest["input_features"], features)
            self.assertEqual(manifest["input_contract"], "track-p-test-contract")
            self.assertEqual(manifest["teacher_policy"], "track-p-test-teacher")


if __name__ == "__main__":
    unittest.main()
