import tempfile
import unittest
from pathlib import Path

import numpy as np

from dataset_store import (
    EpisodeSamples,
    is_valid_episode,
    load_episode,
    merge_episodes,
    save_episode_atomic,
    validate_merged_dataset,
)


def _episode(episode_id: int, steps: int = 4) -> EpisodeSamples:
    return EpisodeSamples(
        X=np.full((steps, 12), episode_id, dtype=np.float64),
        U=np.full((steps, 4), episode_id + 0.5, dtype=np.float64),
        episode_id=episode_id,
        initial_state_id=episode_id // 2,
        environment_id=episode_id % 2,
        step_id=np.arange(steps),
        target_gate_id=np.ones(steps, dtype=np.int64),
        timestamp=np.arange(steps, dtype=np.float64),
        position=np.zeros((steps, 3)),
        quaternion=np.tile([1.0, 0.0, 0.0, 0.0], (steps, 1)),
        velocity=np.zeros((steps, 3)),
        acceleration=np.zeros((steps, 3)),
        reference_position=np.zeros((steps, 3)),
        reference_orientation=np.tile([1.0, 0.0, 0.0, 0.0], (steps, 1)),
        mpc_solve_time=np.zeros(steps),
        actual_loop_period=np.full(steps, 0.05),
        deadline_miss=np.zeros(steps, dtype=np.int8),
    )


class DatasetStoreTests(unittest.TestCase):
    def test_episode_is_atomic_and_validatable(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "episode_0000.npz"
            save_episode_atomic(path, _episode(0))
            self.assertTrue(path.exists())
            self.assertTrue(is_valid_episode(path, expected_steps=4, expected_episode_id=0))
            self.assertEqual(load_episode(path)["X"].shape, (4, 12))

    def test_merge_and_validate_exact_shapes(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            paths = []
            for episode_id in range(4):
                path = directory / f"episode_{episode_id:04d}.npz"
                save_episode_atomic(path, _episode(episode_id))
                paths.append(path)
            final = directory / "final" / "teacher_dataset.npz"
            stats = merge_episodes(
                paths,
                final,
                expected_steps=4,
                expected_episode_count=4,
            )
            self.assertEqual(stats["X_shape"], [16, 12])
            self.assertEqual(stats["U_shape"], [16, 4])
            validated = validate_merged_dataset(
                final,
                expected_samples=16,
                expected_episodes=4,
                expected_steps=4,
                expected_initial_states=2,
                expected_environments=2,
            )
            self.assertEqual(validated["samples"], 16)

    def test_corrupt_or_wrong_length_episode_is_not_resumable(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "episode_0000.npz"
            path.write_bytes(b"not an npz archive")
            self.assertFalse(is_valid_episode(path, expected_steps=4, expected_episode_id=0))

    def test_optional_robust_metadata_round_trips_and_merges(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            paths = []
            for episode_id in range(2):
                episode = _episode(episode_id)
                episode.path_context = np.zeros((4, 4, 3))
                episode.reference_speed = np.full(4, 1.5 + episode_id)
                episode.data_source = np.full(4, episode_id, dtype=np.int8)
                episode.perturbation = np.zeros((4, 3))
                episode.student_control = np.zeros((4, 4))
                episode.shield_active = np.zeros(4, dtype=np.int8)
                path = directory / "episodes" / f"episode_{episode_id:04d}.npz"
                save_episode_atomic(path, episode)
                paths.append(path)
            final = directory / "final" / "teacher_dataset.npz"
            merge_episodes(paths, final, expected_steps=4, expected_episode_count=2)
            arrays = load_episode(final)
            self.assertEqual(arrays["path_context"].shape, (8, 4, 3))
            self.assertEqual(arrays["student_control"].shape, (8, 4))
            self.assertEqual(arrays["data_source"].dtype, np.int8)


if __name__ == "__main__":
    unittest.main()
