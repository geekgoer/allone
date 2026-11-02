import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from falcon_paper_fidelity.contracts import (
    build_gate_input,
    build_paper_gate_euler_input,
    build_trajectory_input,
)
from falcon_paper_fidelity.acado_mpc import (
    AcadoMPCConfig,
    AcadoRtiMPC,
    _ned_frd_quaternion_to_zup_flu,
    _ned_frd_vector_to_zup_flu,
)
from falcon_paper_fidelity.collector import (
    ACADO_POLICY_VERSION,
    ACADO_MARKOV_CONTINUOUS_POLICY_VERSION,
    EpisodeFailure,
    _is_rpc_transport_failure,
    _load_config,
    _policy_version,
    _prepare,
)
from falcon_paper_fidelity.dataset import prepare_episode_split
from falcon_paper_fidelity.motion_primitive import (
    MotionPrimitiveConfig,
    MotionPrimitiveInfeasible,
    QuinticMotionPrimitivePlanner,
    QuinticPolynomial,
)
from falcon_paper_fidelity.recovery_dataset import raw_state_from_input, sample_recovery_input
from airsim_interface import RawDroneState
from mpc_controller import MPCReference
from environment import Gate


ROOT = Path(__file__).resolve().parents[1]


class FalconPaperFidelityTests(unittest.TestCase):
    def test_literal_paper_input_uses_wrapped_euler_difference(self):
        state = RawDroneState(
            position=np.array([1.0, -2.0, -3.0]),
            quaternion=np.array([np.cos(0.25), 0.0, 0.0, np.sin(0.25)]),
            velocity=np.array([4.0, -0.5, 0.25]),
            acceleration=np.array([0.1, 0.2, -0.3]),
            angular_velocity=np.zeros(3),
        )
        value = build_paper_gate_euler_input(
            state,
            np.array([10.0, 1.0, -2.5]),
            np.array([1.0, 0.0, 0.0, 0.0]),
        )
        np.testing.assert_allclose(value[:3], [9.0, 3.0, 0.5])
        np.testing.assert_allclose(value[3:6], [0.0, 0.0, -0.5])
        np.testing.assert_allclose(value[6:9], state.velocity)
        np.testing.assert_allclose(value[9:12], state.acceleration)

    def test_v5_nominal_start_has_a_distinct_policy_identity(self):
        path = (
            ROOT
            / "experiments"
            / "falcon_paper_fidelity_acado_v5_phase_consistent"
            / "paper_core_config.json"
        )
        config = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(
            _policy_version(config), ACADO_MARKOV_CONTINUOUS_POLICY_VERSION
        )

    def test_rpc_transport_failures_are_separate_from_episode_failures(self):
        self.assertTrue(_is_rpc_transport_failure(TimeoutError("Request timed out")))
        self.assertTrue(_is_rpc_transport_failure(ConnectionResetError("connection reset")))
        self.assertFalse(_is_rpc_transport_failure(EpisodeFailure("episode wall-clock timeout")))
        self.assertFalse(_is_rpc_transport_failure(MotionPrimitiveInfeasible("bad primitive")))

    def test_acado_frame_change_is_its_own_inverse(self):
        vector = np.array([1.2, -3.4, 5.6])
        quaternion = np.array([0.9, 0.1, -0.2, 0.3])
        np.testing.assert_allclose(
            _ned_frd_vector_to_zup_flu(_ned_frd_vector_to_zup_flu(vector)), vector
        )
        converted = _ned_frd_quaternion_to_zup_flu(quaternion)
        restored = _ned_frd_quaternion_to_zup_flu(converted)
        np.testing.assert_allclose(restored, quaternion / np.linalg.norm(quaternion))

    def test_acado_solver_hover_and_axis_directions(self):
        library = (
            ROOT
            / "native"
            / "acado_rpg_mpc"
            / "build"
            / "bin"
            / "libfalcon_acado_mpc.dll"
        )
        if not library.is_file():
            self.skipTest("ACADO DLL has not been built")
        controller = AcadoRtiMPC(AcadoMPCConfig(library=str(library)))
        state = RawDroneState(
            position=np.zeros(3),
            quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
            velocity=np.zeros(3),
            acceleration=np.zeros(3),
            angular_velocity=np.zeros(3),
        )
        hold = [MPCReference(np.zeros(3)) for _ in range(21)]
        output = controller.solve(state, hold)
        self.assertAlmostEqual(output[0], 9.8066, places=3)
        np.testing.assert_allclose(output[1:], 0.0, atol=1e-5)
        controller.reset()
        forward = [MPCReference(np.array([2.0, 0.0, 0.0])) for _ in range(21)]
        self.assertLess(controller.solve(state, forward)[2], 0.0)
        controller.reset()
        right = [MPCReference(np.array([0.0, 2.0, 0.0])) for _ in range(21)]
        self.assertGreater(controller.solve(state, right)[1], 0.0)

    def test_quintic_satisfies_all_boundary_conditions(self):
        p0 = np.array([1.0, -2.0, 0.5])
        v0 = np.array([0.4, -0.2, 0.1])
        a0 = np.array([0.1, 0.2, -0.3])
        p1 = np.array([4.0, 1.0, -1.0])
        v1 = np.array([2.0, 0.0, 0.0])
        a1 = np.zeros(3)
        curve = QuinticPolynomial(p0, v0, a0, p1, v1, a1, 2.5)
        start = curve.evaluate(0.0)
        end = curve.evaluate(2.5)
        for actual, expected in zip(start, (p0, v0, a0)):
            np.testing.assert_allclose(actual, expected, atol=1e-10)
        for actual, expected in zip(end, (p1, v1, a1)):
            np.testing.assert_allclose(actual, expected, atol=1e-9)

    def test_reference_window_is_deterministic_and_crosses_gate(self):
        planner = QuinticMotionPrimitivePlanner(
            MotionPrimitiveConfig(nominal_speed=5.0, terminal_speed=5.0)
        )
        kwargs = dict(
            position=[0.0, 1.2, -2.0],
            velocity=[1.0, -0.2, 0.0],
            acceleration=[0.0, 0.0, 0.0],
            gate_position=[10.0, 0.0, -2.5],
            gate_orientation=[1.0, 0.0, 0.0, 0.0],
            dt=0.05,
            horizon=20,
        )
        first = planner.reference_window(**kwargs)
        second = planner.reference_window(**kwargs)
        self.assertEqual(len(first), 20)
        np.testing.assert_allclose(
            np.asarray([item.position for item in first]),
            np.asarray([item.position for item in second]),
        )
        self.assertGreater(first[-1].position[0], first[0].position[0])
        self.assertGreater(
            max(abs(item.orientation[2]) for item in first),
            1e-4,
        )
        self.assertGreater(
            max(np.linalg.norm(item.body_rates) for item in first),
            1e-4,
        )

    def test_infeasible_primitive_is_never_silently_returned(self):
        planner = QuinticMotionPrimitivePlanner(
            MotionPrimitiveConfig(maximum_acceleration=1.0, maximum_duration=2.0)
        )
        with self.assertRaises(MotionPrimitiveInfeasible):
            planner.plan(
                np.zeros(3),
                np.zeros(3),
                np.array([5.0, 0.0, 0.0]),
                np.array([10.0, 0.0, -2.0]),
                np.array([1.0, 0.0, 0.0, 0.0]),
            )

    def test_continuous_route_crosses_boundaries_without_reference_freeze(self):
        planner = QuinticMotionPrimitivePlanner(
            MotionPrimitiveConfig(
                nominal_speed=5.0,
                terminal_speed=5.0,
                maximum_speed=8.0,
                maximum_acceleration=8.0,
            )
        )
        gates = (
            Gate(1, np.array([10.0, 1.6, -2.5]), np.array([1.0, 0.0, 0.0, 0.0])),
            Gate(2, np.array([20.0, 0.0, -4.1]), np.array([1.0, 0.0, 0.0, 0.0])),
        )
        trajectory = planner.plan_continuous_route(
            [0.0, 0.0, -2.0],
            [0.75, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            gates,
            final_exit_distance=5.0,
        )
        self.assertEqual(len(trajectory.segments), 3)
        for gate, boundary in zip(gates, trajectory.boundary_times):
            position, velocity, acceleration, _ = trajectory.evaluate(boundary)
            np.testing.assert_allclose(position, gate.center, atol=1e-8)
            np.testing.assert_allclose(velocity, [5.0, 0.0, 0.0], atol=1e-8)
            np.testing.assert_allclose(acceleration, 0.0, atol=1e-8)
            left = trajectory.evaluate(boundary - 1e-5)
            right = trajectory.evaluate(boundary + 1e-5)
            for left_value, right_value in zip(left[:3], right[:3]):
                np.testing.assert_allclose(left_value, right_value, atol=5e-4)

        crossing_time = trajectory.boundary_times[0]
        references = planner.reference_window_for_trajectory(
            trajectory,
            elapsed=crossing_time - 0.5,
            dt=0.1,
            horizon=21,
        )
        self.assertLess(references[0].position[0], gates[0].center[0])
        self.assertGreater(references[-1].position[0], gates[0].center[0])
        positions = np.asarray([item.position for item in references])
        self.assertTrue(np.all(np.linalg.norm(np.diff(positions, axis=0), axis=1) > 1e-5))
        _, final_velocity, final_acceleration, _ = trajectory.evaluate(
            trajectory.duration + 1.0
        )
        np.testing.assert_allclose(final_velocity, 0.0, atol=1e-8)
        np.testing.assert_allclose(final_acceleration, 0.0, atol=1e-8)

    def test_trajectory_input_contains_reference_errors_not_gate_position(self):
        state = RawDroneState(
            position=np.array([1.0, 2.0, -3.0]),
            quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
            velocity=np.array([2.0, -1.0, 0.5]),
            acceleration=np.array([9.0, 9.0, 9.0]),
            angular_velocity=np.zeros(3),
        )
        reference = MPCReference(
            position=np.array([1.5, 1.0, -2.0]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            velocity=np.array([5.0, 0.0, 0.0]),
            acceleration=np.array([0.2, -0.3, 0.4]),
        )
        value = build_trajectory_input(state, reference)
        np.testing.assert_allclose(value[0:3], [0.5, -1.0, 1.0])
        np.testing.assert_allclose(value[3:6], 0.0)
        np.testing.assert_allclose(value[6:9], [3.0, 1.0, -0.5])
        np.testing.assert_allclose(value[9:12], [0.2, -0.3, 0.4])

    def test_recovery_state_round_trip_preserves_12d_contract(self):
        rng = np.random.default_rng(7)
        gate = np.array([0.0, 0.0, -3.0])
        orientation = np.array([1.0, 0.0, 0.0, 0.0])
        for family in range(6):
            x = sample_recovery_input(rng, family)
            raw = raw_state_from_input(x, gate)
            np.testing.assert_allclose(build_gate_input(raw, gate, orientation), x, atol=1e-10)

    def test_config_matches_published_count_and_timing(self):
        path = ROOT / "experiments" / "falcon_paper_fidelity_v1" / "paper_core_config.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(config["initial_state"]["count"], 10)
        self.assertEqual(config["environment"]["count"], 40)
        self.assertEqual(config["timing"]["steps_per_episode"], 100)
        self.assertEqual(config["mpc"]["horizon"], 20)
        self.assertAlmostEqual(config["timing"]["dt"], 0.05)
        self.assertEqual(10 * 40 * 100, 40000)

    def test_acado_config_is_isolated_and_has_distinct_policy(self):
        path = (
            ROOT
            / "experiments"
            / "falcon_paper_fidelity_acado_v1"
            / "paper_core_config.json"
        )
        config = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(_policy_version(config), ACADO_POLICY_VERSION)
        self.assertEqual(config["mpc"]["horizon"], 20)
        self.assertAlmostEqual(config["timing"]["dt"], 0.05)
        self.assertAlmostEqual(config["mpc"]["prediction_dt"], 0.1)
        self.assertEqual(config["dataset"]["root"], "paper_core_40k_acado")

    def test_smoke_manifest_is_resumable_and_uses_list_artifacts(self):
        config_path = ROOT / "experiments" / "falcon_paper_fidelity_v1" / "paper_core_config.json"
        config, resolved = _load_config(config_path)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "smoke"
            effective, states, environments, manifest_path = _prepare(
                root, config, resolved, resume=False, smoke_test=True
            )
            self.assertEqual(len(states), 2)
            self.assertEqual(len(environments), 2)
            self.assertEqual(effective["timing"]["steps_per_episode"], 100)
            self.assertEqual(len(json.loads((root / "initial_states.json").read_text())), 2)
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["expected_samples"], 400)
            _prepare(root, config, resolved, resume=True, smoke_test=True)

    def test_paper_split_is_exact_and_episode_disjoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_root = root / "paper" / "final"
            source_root.mkdir(parents=True)
            episode_id = np.repeat(np.arange(400), 100)
            rows = len(episode_id)
            source = source_root / "core.npz"
            np.savez_compressed(
                source,
                X=np.zeros((rows, 12), dtype=np.float32),
                U=np.zeros((rows, 4), dtype=np.float32),
                episode_id=episode_id,
                initial_state_id=episode_id // 40,
                environment_id=episode_id % 40,
            )
            output = root / "prepared"
            manifest = prepare_episode_split(source, output, seed=2025)
            self.assertEqual(manifest["splits"]["train"]["samples"], 32000)
            self.assertEqual(manifest["splits"]["validation"]["samples"], 4000)
            self.assertEqual(manifest["splits"]["test"]["samples"], 4000)
            episode_sets = [set(manifest["splits"][name]["episode_ids"]) for name in ("train", "validation", "test")]
            self.assertFalse(episode_sets[0] & episode_sets[1])
            self.assertFalse(episode_sets[0] & episode_sets[2])
            self.assertFalse(episode_sets[1] & episode_sets[2])


if __name__ == "__main__":
    unittest.main()
