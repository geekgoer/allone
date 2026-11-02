import unittest
from unittest.mock import patch

import numpy as np

from environment import Gate, generate_gate_sequence, generate_gate_sequences
from gate_visualization import (
    _gate_parts,
    cleanup_generated_gates,
    spawn_gate_sequence,
    spawn_gate_sequences,
)


class _Vector3r:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x_val, self.y_val, self.z_val = x, y, z


class _Quaternionr:
    def __init__(self, x_val=0.0, y_val=0.0, z_val=0.0, w_val=1.0):
        self.x_val = x_val
        self.y_val = y_val
        self.z_val = z_val
        self.w_val = w_val


class _Pose:
    def __init__(self, position_val=None, orientation_val=None):
        self.position = position_val
        self.orientation = orientation_val


class _FakeAirSim:
    Vector3r = _Vector3r
    Quaternionr = _Quaternionr
    Pose = _Pose


class _SpawnClient:
    def __init__(self):
        self.spawn_calls = []
        self.destroyed = []

    def simListSceneObjects(self):
        return ["FalconGate_old_left", "UnrelatedObject"]

    def simDestroyObject(self, name):
        self.destroyed.append(name)
        return True

    def simSpawnObject(self, object_name, asset_name, pose, scale, physics_enabled, is_blueprint):
        self.spawn_calls.append(
            (object_name, asset_name, pose, scale, physics_enabled, is_blueprint)
        )
        return object_name


class _GameModeSpawnClient(_SpawnClient):
    def simListAssets(self):
        return ["EditorCube", "EditorCylinder"]


class GateVisualizationTests(unittest.TestCase):
    def test_gate_parts_form_four_bars(self):
        gate = Gate(1, [10.0, 0.0, -2.0], [1.0, 0.0, 0.0, 0.0])
        parts = _gate_parts(gate, width=4.0, height=4.0, thickness=0.15)
        self.assertEqual([part[0] for part in parts], ["left", "right", "top", "bottom"])
        np.testing.assert_allclose(parts[0][1], [10.0, -2.0, -2.0])
        np.testing.assert_allclose(parts[1][1], [10.0, 2.0, -2.0])
        np.testing.assert_allclose(parts[2][1], [10.0, 0.0, -4.0])
        np.testing.assert_allclose(parts[3][1], [10.0, 0.0, 0.0])

    def test_spawn_and_cleanup(self):
        client = _SpawnClient()
        gates = generate_gate_sequence(0.0, seed=0)
        with patch("airsim_interface.airsim", _FakeAirSim):
            scene = spawn_gate_sequence(client, gates)
        self.assertEqual(len(scene.object_names), 20)
        self.assertEqual(len(client.spawn_calls), 20)
        self.assertEqual(client.spawn_calls[0][1], "1M_Cube")
        self.assertFalse(client.spawn_calls[0][4])
        self.assertFalse(client.spawn_calls[0][5])
        self.assertIn("FalconGate_old_left", client.destroyed)
        scene.destroy()
        self.assertEqual(len([name for name in client.destroyed if name.startswith("FalconGate_1_")]), 4)

    def test_game_mode_falls_back_to_size_corrected_editor_cube(self):
        client = _GameModeSpawnClient()
        gates = generate_gate_sequence(0.0, seed=0, gate_count=1)
        with patch("airsim_interface.airsim", _FakeAirSim):
            scene = spawn_gate_sequence(client, gates, asset_name="1M_Cube")
        self.assertEqual(len(scene.object_names), 4)
        self.assertTrue(all(call[1] == "EditorCube" for call in client.spawn_calls))
        left_scale = client.spawn_calls[0][3]
        self.assertAlmostEqual(left_scale.x_val, 0.075)
        self.assertAlmostEqual(left_scale.y_val, 0.075)
        self.assertAlmostEqual(left_scale.z_val, 2.0)

    def test_single_sequence_applies_world_offset(self):
        client = _SpawnClient()
        gates = generate_gate_sequence(0.0, seed=0, gate_count=1)
        with patch("airsim_interface.airsim", _FakeAirSim):
            spawn_gate_sequence(client, gates, world_offset=[0.0, -3.0, 0.0])
        first_pose = client.spawn_calls[0][2]
        self.assertAlmostEqual(first_pose.position.y_val, -5.0)

    def test_cleanup_is_prefix_scoped(self):
        client = _SpawnClient()
        removed = cleanup_generated_gates(client)
        self.assertEqual(removed, ["FalconGate_old_left"])
        self.assertNotIn("UnrelatedObject", client.destroyed)

    def test_spawn_multiple_independent_groups(self):
        client = _SpawnClient()
        groups = generate_gate_sequences(["Drone1", "Drone2", "Drone3"], 0.0, seed=0)
        with patch("airsim_interface.airsim", _FakeAirSim):
            scene = spawn_gate_sequences(client, groups)
        self.assertEqual(len(scene.object_names), 60)
        self.assertTrue(any(name.startswith("FalconGate_Drone1_") for name in scene.object_names))
        self.assertTrue(any(name.startswith("FalconGate_Drone2_") for name in scene.object_names))
        self.assertTrue(any(name.startswith("FalconGate_Drone3_") for name in scene.object_names))

    def test_multiple_groups_apply_vehicle_world_offsets(self):
        client = _SpawnClient()
        groups = generate_gate_sequences(
            ["Drone1", "Drone2", "Drone3"], 0.0, seed=0, group_spacing=6.0
        )
        offsets = {
            "Drone1": [0.0, -3.0, 0.0],
            "Drone2": [0.0, 0.0, 0.0],
            "Drone3": [0.0, 3.0, 0.0],
        }
        with patch("airsim_interface.airsim", _FakeAirSim):
            spawn_gate_sequences(client, groups, vehicle_world_offsets=offsets)
        left_bar_y = {
            call[0].split("_")[1]: call[2].position.y_val
            for call in client.spawn_calls
            if call[0].endswith("_1_left")
        }
        self.assertAlmostEqual(left_bar_y["Drone1"], -11.0)
        self.assertAlmostEqual(left_bar_y["Drone2"], -2.0)
        self.assertAlmostEqual(left_bar_y["Drone3"], 7.0)


if __name__ == "__main__":
    unittest.main()
