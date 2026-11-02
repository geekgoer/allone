import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from collect_teacher_dataset import GateTargetTracker, load_collection_config, _initial_states, _training_environments
from dataset_store import EpisodeSamples, merge_episodes, save_episode_atomic
from environment import generate_gate_sequence
from mpc_controller import MPCController
from validate_teacher_dataset import validate_dataset


def _episode(episode_id: int, steps: int = 4) -> EpisodeSamples:
    return EpisodeSamples(
        X=np.full((steps, 12), episode_id + 1.0),
        U=np.full((steps, 4), episode_id + 0.5),
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


class TeacherCollectionTests(unittest.TestCase):
    def test_default_collection_contract_is_400_by_100(self):
        config, _ = load_collection_config("configs/teacher_collection.json")
        self.assertEqual(config["initial_state"]["count"], 10)
        self.assertEqual(config["environment"]["count"], 40)
        self.assertEqual(config["timing"]["steps_per_episode"], 100)
        self.assertEqual(len(_initial_states(config)) * len(_training_environments(config)), 400)

    def test_full_route_collection_contract_is_35_seconds_and_five_gates(self):
        config, _ = load_collection_config("configs/teacher_collection_full_route.json")
        self.assertEqual(config["timing"]["steps_per_episode"], 700)
        self.assertEqual(config["timing"]["episode_seconds"], 35.0)
        self.assertEqual(config["dataset"]["required_gate_ids"], [1, 2, 3, 4, 5])
        self.assertTrue(config["dataset"]["require_all_gates"])

    def test_gate_tracker_progresses_on_inside_plane_crossings_without_stops(self):
        gates = generate_gate_sequence(0.0, seed=0)
        tracker = GateTargetTracker(
            gates,
            np.array([0.0, 0.0, -2.0]),
            approach_distance=2.0,
            exit_distance=1.0,
            threshold=0.8,
            stabilization_speed=0.25,
            reference_speed=2.0,
            start_time=0.0,
        )
        self.assertEqual(tracker.target_gate_id, 1)
        window = tracker.reference_window(3.9, dt=0.05, horizon=20)
        self.assertGreater(np.linalg.norm(window[0].velocity), 0.0)
        tracker.update(np.array([9.9, 0.0, -2.0]), 4.0)
        tracker.update(np.array([10.1, 0.0, -2.0]), 4.1)
        self.assertEqual(tracker.target_gate_id, 2)
        self.assertEqual(tracker.phase, "route")

    def test_mpc_reset_hook_is_available(self):
        MPCController().reset()

    def test_validator_checks_manifest_episodes_and_final_dataset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            episode_paths = []
            episodes = {}
            for episode_id in range(4):
                path = root / "episodes" / f"episode_{episode_id:04d}.npz"
                save_episode_atomic(path, _episode(episode_id))
                episode_paths.append(path)
                episodes[f"episode_{episode_id:04d}"] = {
                    "status": "completed",
                    "file": str(Path("episodes") / path.name),
                }
            final_path = root / "final" / "teacher_dataset.npz"
            merge_episodes(
                episode_paths,
                final_path,
                expected_steps=4,
                expected_episode_count=4,
            )
            (root / "collection_config.json").write_text(
                json.dumps({"mpc": {"max_thrust_accel": 18.0}}), encoding="utf-8"
            )
            manifest = {
                "status": "completed",
                "episode_count": 4,
                "steps_per_episode": 4,
                "initial_state_count": 2,
                "environment_count": 2,
                "final_dataset": str(Path("final") / final_path.name),
                "episodes": episodes,
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            result = validate_dataset(root)
            self.assertEqual(result["X_shape"], [16, 12])
            self.assertEqual(result["U_shape"], [16, 4])
            self.assertEqual(result["warnings"], [])


if __name__ == "__main__":
    unittest.main()
