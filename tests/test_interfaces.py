import unittest

import numpy as np

from airsim_interface import AirSimDrone, RawDroneState
from config import ThrustCalibration
from control_adapter import control_to_airsim_command, thrust_to_throttle
from environment import (
    generate_gate_sequence,
    generate_gate_sequences,
    gate_manhattan_displacements,
)
from mpc_controller import MPCController, MPCReference
from dnn_model_utils import build_policy_input
from state_adapter import (
    airsim_to_controller_frame,
    build_controller_state,
    controller_to_airsim_frame,
    quaternion_to_rotation_vector,
)
from reference_generator import (
    gate_alignment_waypoint,
    gate_passage_waypoints,
    linear_reference_window_at_time,
    polyline_reference_window_at_time,
)


class _Vector:
    def __init__(self, x, y, z):
        self.x_val, self.y_val, self.z_val = x, y, z


class _Quaternion:
    def __init__(self, w, x, y, z):
        self.w_val, self.x_val, self.y_val, self.z_val = w, x, y, z


class _Kinematics:
    position = _Vector(1.0, 2.0, -3.0)
    orientation = _Quaternion(1.0, 0.0, 0.0, 0.0)
    linear_velocity = _Vector(0.1, 0.2, 0.3)
    linear_acceleration = _Vector(-0.1, -0.2, -0.3)
    angular_velocity = _Vector(0.01, 0.02, 0.03)


class _State:
    kinematics_estimated = _Kinematics()
    timestamp = 123


class _Client:
    def getMultirotorState(self, vehicle_name=""):
        self.vehicle_name = vehicle_name
        return _State()


class InterfaceTests(unittest.TestCase):
    def setUp(self):
        self.state = RawDroneState(
            position=[1.0, 2.0, -3.0],
            quaternion=[1.0, 0.0, 0.0, 0.0],
            velocity=[0.1, 0.2, 0.3],
            acceleration=[-0.1, -0.2, -0.3],
            angular_velocity=[0.01, 0.02, 0.03],
        )

    def test_airsim_interface_flattens_state(self):
        drone = AirSimDrone(_Client(), "Drone1")
        state = drone.get_raw_state()
        np.testing.assert_allclose(state.position, [1.0, 2.0, -3.0])
        np.testing.assert_allclose(state.velocity, [0.1, 0.2, 0.3])
        self.assertEqual(drone.client.vehicle_name, "Drone1")

    def test_controller_state_shape_and_semantics(self):
        state = build_controller_state(self.state, [2.0, 4.0, -5.0], [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(state.shape, (12,))
        np.testing.assert_allclose(state[0:3], [1.0, 2.0, -2.0])
        np.testing.assert_allclose(state[3:6], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(state[6:9], [0.1, 0.2, 0.3])
        np.testing.assert_allclose(state[9:12], [-0.1, -0.2, -0.3])

    def test_attitude_error_uses_relative_rotation(self):
        angle = np.pi / 2.0
        q_target = [np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0)]
        vector = quaternion_to_rotation_vector(q_target)
        np.testing.assert_allclose(vector, [0.0, 0.0, angle], atol=1e-9)

    def test_frame_conversion_is_explicit(self):
        value = [1.0, 2.0, -3.0]
        np.testing.assert_allclose(airsim_to_controller_frame(value), value)
        np.testing.assert_allclose(airsim_to_controller_frame(value, "z_up"), [1.0, 2.0, 3.0])
        np.testing.assert_allclose(controller_to_airsim_frame([1.0, 2.0, 3.0], "z_up"), value)

    def test_control_mapping_and_fixed_shape(self):
        calibration = ThrustCalibration(
            hover_throttle=0.42,
            hover_thrust_accel=9.81,
            thrust_accel_per_throttle=20.0,
            calibration_complete=True,
        )
        self.assertAlmostEqual(thrust_to_throttle(9.81, calibration), 0.42)
        command = control_to_airsim_command([9.81, 0.1, -0.2, 0.3], calibration)
        self.assertEqual(command.throttle, 0.42)
        self.assertEqual(command.roll_rate, 0.1)
        self.assertEqual(command.pitch_rate, 0.2)
        self.assertEqual(command.yaw_rate, -0.3)
        self.assertEqual(command.duration, 0.05)

    def test_gate_environment_coordinates_and_epsilon(self):
        for epsilon in (0.0, 0.8, 1.6):
            gates = generate_gate_sequence(epsilon, seed=7)
            np.testing.assert_allclose([gate.center[0] for gate in gates], [10, 20, 30, 40, 50])
            np.testing.assert_allclose(gate_manhattan_displacements(gates), [epsilon] * 4)

    def test_gate_groups_are_independent_and_configurable(self):
        groups = generate_gate_sequences(
            ["Drone1", "Drone2", "Drone3"],
            epsilon=0.8,
            seed=0,
            group_spacing=6.0,
            gate_count=5,
            x_start=4.0,
            x_spacing=7.0,
        )
        self.assertEqual(set(groups), {"Drone1", "Drone2", "Drone3"})
        self.assertEqual([len(groups[name]) for name in groups], [5, 5, 5])
        np.testing.assert_allclose([gate.center[0] for gate in groups["Drone2"]], [4, 11, 18, 25, 32])
        np.testing.assert_allclose(groups["Drone1"][0].center[1], -12.4)
        np.testing.assert_allclose(groups["Drone2"][0].center[1], 0.0)
        np.testing.assert_allclose(groups["Drone3"][0].center[1], 12.4)
        self.assertFalse(np.shares_memory(groups["Drone1"][0].center, groups["Drone2"][0].center))
        self.assertTrue(
            all(gate.center[2] <= -2.0 + 1e-9 for gates in groups.values() for gate in gates)
        )

    def test_reference_window_uses_gate_position(self):
        groups = generate_gate_sequences(["Drone1"], epsilon=0.8, seed=0)
        gate = groups["Drone1"][0]
        start = gate.center - np.array([0.4, 0.1, 0.0])
        window = linear_reference_window_at_time(
            start, gate.center, elapsed=0.0, dt=0.05, horizon=20, speed=2.0
        )
        self.assertFalse(np.allclose(window[0].position, gate.center))
        np.testing.assert_allclose(window[-1].position, gate.center)
        np.testing.assert_allclose(window[-1].velocity, np.zeros(3))

    def test_gate_passage_waypoints_cross_the_gate_plane(self):
        before, after = gate_passage_waypoints([10.0, 2.0, -2.0])
        np.testing.assert_allclose(before, [8.0, 2.0, -2.0])
        np.testing.assert_allclose(after, [11.0, 2.0, -2.0])

    def test_gate_alignment_keeps_current_forward_coordinate(self):
        staging = gate_alignment_waypoint([3.0, -3.0, -2.0], [10.0, -12.4, -2.8])
        np.testing.assert_allclose(staging, [3.0, -12.4, -2.8])

    def test_mpc_uses_future_reference_samples(self):
        controller = MPCController(dt=0.05, horizon=4)
        raw = RawDroneState(
            position=[0.0, 0.0, -2.0],
            quaternion=[1.0, 0.0, 0.0, 0.0],
            velocity=[0.0, 0.0, 0.0],
            acceleration=[0.0, 0.0, 0.0],
            angular_velocity=[0.0, 0.0, 0.0],
        )
        first = MPCReference([0.0, 0.0, -2.0])
        positive = [first] + [MPCReference([1.0, 0.0, -2.0])] * 3
        negative = [first] + [MPCReference([-1.0, 0.0, -2.0])] * 3
        state = build_controller_state(raw, first.position, first.orientation)
        positive_control = controller.solve(state, positive)
        negative_control = controller.solve(state, negative)
        self.assertLess(positive_control[2], 0.0)
        self.assertGreater(negative_control[2], 0.0)

    def test_path_aware_policy_input_contains_future_reference(self):
        state = np.array(
            [0.2, -0.1, 0.3, 0.01, 0.02, 0.03, 1.0, 0.0, 0.0, 0.1, 0.2, 0.3]
        )
        window = [
            MPCReference([float(index), 0.5 * index, -2.0], velocity=[2.0, 0.0, 0.0])
            for index in range(20)
        ]
        value = build_policy_input(state, window, input_dim=25, reference_speed=2.0)
        self.assertEqual(value.shape, (25,))
        np.testing.assert_allclose(value[:12], state)
        current_position = window[0].position - state[:3]
        expected_context = np.asarray(
            [window[index].position - current_position for index in (1, 5, 10, 19)]
        )
        np.testing.assert_allclose(value[12:24].reshape(4, 3), expected_context)
        self.assertEqual(value[24], 2.0)

    def test_polyline_reference_stops_only_at_final_point(self):
        moving = polyline_reference_window_at_time(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1.0, 0.0]],
            elapsed=0.8,
            dt=0.1,
            horizon=2,
            speed=1.0,
        )
        self.assertGreater(np.linalg.norm(moving[-1].velocity), 0.0)
        stopped = polyline_reference_window_at_time(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            elapsed=2.0,
            dt=0.1,
            horizon=2,
            speed=1.0,
        )
        np.testing.assert_allclose(stopped[-1].velocity, np.zeros(3))


if __name__ == "__main__":
    unittest.main()
