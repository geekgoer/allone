import unittest

import numpy as np

from initial_state_generator import InitialStateGenerator
from training_environment_generator import generate_training_environments
from generate_iid_mpc_dataset import _generate_shard
from generate_track_p_12d_dataset import _generate_shard as _generate_track_p_shard
from track_p_12d_contract import build_track_p_teacher_query


class DatasetGeneratorTests(unittest.TestCase):
    def test_initial_states_are_reproducible(self):
        first = InitialStateGenerator(count=10, seed=7).generate()
        second = InitialStateGenerator(count=10, seed=7).generate()
        self.assertEqual(len(first), 10)
        for left, right in zip(first, second):
            np.testing.assert_allclose(left.position, right.position)
            np.testing.assert_allclose(left.orientation, right.orientation)
            np.testing.assert_allclose(left.velocity, right.velocity)
            self.assertEqual(left.to_dict(), right.to_dict())

    def test_training_environments_are_reproducible_and_diverse(self):
        first = generate_training_environments(count=40, seed=0)
        second = generate_training_environments(count=40, seed=0)
        self.assertEqual(len(first), 40)
        self.assertEqual([item.to_dict() for item in first], [item.to_dict() for item in second])
        self.assertEqual({len(item.gates) for item in first}, {5})
        self.assertEqual(
            {tuple(gate.center[0] for gate in item.gates) for item in first},
            {(10.0, 20.0, 30.0, 40.0, 50.0)},
        )
        self.assertGreater(
            len({tuple(tuple(np.round(gate.center, 6)) for gate in item.gates) for item in first}),
            10,
        )
        self.assertTrue(
            all(
                np.allclose(item.adjacent_yz_displacement, item.epsilon)
                for item in first
            )
        )

    def test_cartesian_product_has_400_fixed_episode_ids(self):
        states = InitialStateGenerator(count=10, seed=0).generate()
        environments = generate_training_environments(count=40, seed=0)
        pairs = [
            (state.initial_state_id, environment.environment_id)
            for state in states
            for environment in environments
        ]
        self.assertEqual(len(pairs), 400)
        self.assertEqual(pairs[0], (0, 0))
        self.assertEqual(pairs[-1], (9, 39))
        self.assertEqual(len(set(pairs)), 400)

    def test_environment_records_speed_and_spacing_variants(self):
        environments = generate_training_environments(
            count=6,
            seed=17,
            speed_values=(1.5, 2.0, 3.5),
            x_spacing_values=(8.0, 10.0),
        )
        self.assertEqual(
            [item.reference_speed for item in environments],
            [1.5, 2.0, 3.5, 1.5, 2.0, 3.5],
        )
        self.assertEqual(
            [item.x_spacing for item in environments],
            [8.0, 10.0, 8.0, 10.0, 8.0, 10.0],
        )

    def test_independent_mpc_shard_is_reproducible_and_path_aware(self):
        config = {
            "random_seed": 3,
            "timing": {"dt": 0.05},
            "route": {"start_x": 0.0, "approach_distance": 2.0, "exit_distance": 1.0},
            "sampling": {
                "family_probabilities": [0.25, 0.45, 0.30],
                "position_error_limits": [[0.2] * 3, [0.8] * 3, [2.0] * 3],
                "attitude_error_limits": [[0.05] * 3, [0.15] * 3, [0.35] * 3],
                "velocity_error_limits": [[0.2] * 3, [0.8] * 3, [2.0] * 3],
                "acceleration_limits": [[0.5] * 3, [2.0] * 3, [4.0] * 3],
                "max_total_speed": 7.0,
                "gate_focus_fraction": 0.5,
                "gate_focus_radius": 3.0,
            },
            "input": {"path_context_offsets": [1, 5, 10, 19]},
            "mpc": {"horizon": 20, "reference_speed": 2.0},
        }
        environment = generate_training_environments(count=1, seed=3)[0]
        first = _generate_shard(environment, config=config, samples=32)
        second = _generate_shard(environment, config=config, samples=32)
        self.assertEqual(first["X"].shape, (32, 25))
        self.assertEqual(first["U"].shape, (32, 4))
        np.testing.assert_array_equal(first["X"], second["X"])
        np.testing.assert_array_equal(first["U"], second["U"])
        self.assertEqual(np.unique(first["X"], axis=0).shape[0], 32)

    def test_track_p_query_uses_gate_error_but_teacher_tracks_lane(self):
        query = build_track_p_teacher_query(
            gate_position=[10.0, 2.0, -3.0],
            gate_orientation=[1.0, 0.0, 0.0, 0.0],
            gate_position_error=[5.0, 1.5, -0.5],
            gate_attitude_error=[0.1, -0.2, 0.3],
            velocity=[2.0, -0.5, 0.25],
            acceleration=[0.4, -0.3, 0.2],
            dt=0.05,
            horizon=20,
            reference_speed=2.0,
        )
        self.assertEqual(query.policy_input.shape, (12,))
        np.testing.assert_allclose(query.policy_input[:3], [5.0, 1.5, -0.5])
        np.testing.assert_allclose(query.current_position, [5.0, 0.5, -2.5])
        # One 2 m/s x 50 ms step plus complete lateral/vertical lane recovery.
        np.testing.assert_allclose(query.teacher_state[:3], [0.1, 1.5, -0.5])
        np.testing.assert_allclose(query.reference_window[-1].position, [7.0, 2.0, -3.0])

    def test_track_p_shard_is_reproducible_and_12_dimensional(self):
        families = [
            {
                "name": "nominal",
                "probability": 0.6,
                "gate_position_error": [[0.1, 10.0], [-0.5, 0.5], [-0.4, 0.4]],
                "gate_attitude_error": [[-0.1, 0.1]] * 3,
                "velocity": [[0.0, 3.0], [-0.5, 0.5], [-0.5, 0.5]],
                "acceleration": [[-1.0, 1.0]] * 3,
            },
            {
                "name": "recovery",
                "probability": 0.4,
                "gate_position_error": [[0.1, 14.0], [-2.0, 2.0], [-1.5, 1.5]],
                "gate_attitude_error": [[-0.4, 0.4]] * 3,
                "velocity": [[-2.0, 6.0], [-2.0, 2.0], [-2.0, 2.0]],
                "acceleration": [[-4.0, 4.0]] * 3,
            },
        ]
        config = {
            "random_seed": 7,
            "timing": {"dt": 0.05},
            "sampling": {
                "edge_probability": 0.3,
                "edge_band_fraction": 0.15,
                "max_total_speed": 7.0,
                "max_total_acceleration": 6.0,
                "families": families,
            },
            "mpc": {"horizon": 20, "reference_speed": 2.0},
        }
        environment = generate_training_environments(
            count=1, seed=7, speed_values=(2.0,)
        )[0]
        first = _generate_track_p_shard(environment, config=config, samples=64)
        second = _generate_track_p_shard(environment, config=config, samples=64)
        self.assertEqual(first["X"].shape, (64, 12))
        self.assertEqual(first["U"].shape, (64, 4))
        np.testing.assert_array_equal(first["X"], second["X"])
        np.testing.assert_array_equal(first["U"], second["U"])
        self.assertEqual(np.unique(first["X"], axis=0).shape[0], 64)
        self.assertTrue(np.all(first["X"][:, 0] > 0.0))


if __name__ == "__main__":
    unittest.main()
