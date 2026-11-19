import unittest

import numpy as np

from config import ThrustCalibration
from multi_drone_manager import MultiDroneManager


class _Vector:
    def __init__(self, x, y, z):
        self.x_val, self.y_val, self.z_val = x, y, z


class _Quaternion:
    def __init__(self):
        self.w_val, self.x_val, self.y_val, self.z_val = 1.0, 0.0, 0.0, 0.0


class _Kinematics:
    def __init__(self, x):
        self.position = _Vector(x, 0.0, -2.0)
        self.orientation = _Quaternion()
        self.linear_velocity = _Vector(0.0, 0.0, 0.0)
        self.linear_acceleration = _Vector(0.0, 0.0, 0.0)
        self.angular_velocity = _Vector(0.0, 0.0, 0.0)


class _State:
    def __init__(self, x):
        self.kinematics_estimated = _Kinematics(x)
        self.timestamp = 1


class _Future:
    def __init__(self, label):
        self.label = label
        self.joined = False

    def join(self):
        self.joined = True


class _Client:
    def __init__(self):
        self.calls = []

    def getMultirotorState(self, vehicle_name=""):
        return _State({"Drone1": 0.0, "Drone2": 1.0, "Drone3": 2.0}[vehicle_name])

    def enableApiControl(self, enabled, vehicle_name=""):
        self.calls.append(("api", enabled, vehicle_name))

    def armDisarm(self, arm, vehicle_name=""):
        self.calls.append(("arm", arm, vehicle_name))

    def takeoffAsync(self, vehicle_name=""):
        self.calls.append(("takeoff", vehicle_name))
        return _Future(vehicle_name)

    def hoverAsync(self, vehicle_name=""):
        self.calls.append(("hover", vehicle_name))
        return _Future(vehicle_name)

    def landAsync(self, vehicle_name=""):
        self.calls.append(("land", vehicle_name))
        return _Future(vehicle_name)

    def moveByAngleRatesThrottleAsync(self, *args, vehicle_name=""):
        self.calls.append(("control", vehicle_name, args))
        return _Future(vehicle_name)


class MultiDroneManagerTests(unittest.TestCase):
    def test_reads_named_states(self):
        client = _Client()
        manager = MultiDroneManager(["Drone1", "Drone2", "Drone3"], client=client)
        states = manager.get_all_states()
        self.assertEqual(list(states), ["Drone1", "Drone2", "Drone3"])
        np.testing.assert_allclose(states["Drone3"].position, [2.0, 0.0, -2.0])

    def test_lifecycle_submits_all_before_waiting(self):
        client = _Client()
        manager = MultiDroneManager(["Drone1", "Drone2", "Drone3"], client=client)
        manager.enable_api_control_all()
        manager.arm_all()
        futures = manager.takeoff_all()
        self.assertEqual([call[0] for call in client.calls[-3:]], ["takeoff"] * 3)
        manager.wait_all(futures)
        self.assertTrue(all(future.joined for future in futures.values()))

    def test_control_requires_each_vehicle(self):
        manager = MultiDroneManager(["Drone1", "Drone2", "Drone3"], client=_Client())
        calibration = ThrustCalibration(calibration_complete=True)
        with self.assertRaises(KeyError):
            manager.send_control_all({"Drone1": [9.81, 0.0, 0.0, 0.0]}, calibration)


if __name__ == "__main__":
    unittest.main()
