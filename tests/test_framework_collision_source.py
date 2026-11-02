from types import SimpleNamespace
from unittest.mock import Mock, patch
import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from airsim_interface import RawDroneState
from controller_architecture_study.framework_runtime_world import FrameworkWorld
from controller_architecture_study.framework_runtime import COLLISION_CONTRACT, ModeMap


class CollisionSourceTests(unittest.TestCase):
    def test_old_collision_or_mismatched_safety_maps_are_rejected_before_use(self):
        with TemporaryDirectory(dir=Path.cwd()) as tmp:
            path = Path(tmp)/"map.json"
            for data in ({"collision_observation":"old"},
                    {"collision_observation":COLLISION_CONTRACT,"safety_predicate":"route_tube"}):
                path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    ModeMap(path,{},2.)

    def test_snapshot_uses_dedicated_collision_api_not_empty_state_field(self):
        xyz = SimpleNamespace(x_val=0.,y_val=0.,z_val=-2.)
        kin = SimpleNamespace(position=xyz,linear_velocity=xyz,linear_acceleration=xyz,
            angular_velocity=xyz,orientation=SimpleNamespace(w_val=1.,x_val=0.,y_val=0.,z_val=0.))
        empty = SimpleNamespace(has_collided=False,time_stamp=0,object_name="")
        contact = SimpleNamespace(has_collided=False,time_stamp=12345,object_name="gate")
        client = SimpleNamespace(getMultirotorState=Mock(return_value=SimpleNamespace(
            kinematics_estimated=kin,timestamp=23456,collision=empty)),simGetCollisionInfo=Mock(return_value=contact))
        raw,hit,_ = FrameworkWorld.snapshot_from(client,"Drone7")
        self.assertIs(hit,contact)
        self.assertEqual(raw.timestamp_ns,23456)
        client.simGetCollisionInfo.assert_called_once_with(vehicle_name="Drone7")

    def test_new_stamp_detects_contact_after_transient_flag_cleared_and_ignores_startup_stamp(self):
        ref = SimpleNamespace(position=np.array([0.,0.,-2.]),velocity=np.array([5.,0.,0.]),
            acceleration=np.zeros(3),orientation=np.array([1.,0.,0.,0.]))
        raw = RawDroneState(ref.position,ref.orientation,ref.velocity,np.zeros(3),np.zeros(3),2000)
        for stamp,flag,expected in ((0,False,False),(100,True,False),(101,False,True)):
            with self.subTest(stamp=stamp):
                world = FrameworkWorld.__new__(FrameworkWorld)
                world.initial = {"Drone1":{"sim_ns":1000,"collision_stamp":100}}
                world.phases,world.windows = {"Drone1":0.},{"Drone1":None}
                world.offsets,world.net_width = {"Drone1":np.zeros(3)},2.
                world.traces,world.failures = {"Drone1":[]},{"Drone1":None}
                hit = SimpleNamespace(has_collided=flag,time_stamp=stamp,object_name="gate")
                with patch("controller_architecture_study.framework_runtime_world.reference_point",return_value=ref):
                    row = world.observe("Drone1",raw,hit,1.)
                self.assertEqual(row["collision"],expected)
                self.assertEqual(world.failures["Drone1"] is not None,expected)

    def test_persistent_collision_stamp_keeps_the_first_detection_time(self):
        ref = SimpleNamespace(position=np.array([0.,0.,-2.]),velocity=np.zeros(3),
            acceleration=np.zeros(3),orientation=np.array([1.,0.,0.,0.]))
        world = FrameworkWorld.__new__(FrameworkWorld)
        world.initial = {"Drone1":{"sim_ns":0,"collision_stamp":0}}
        world.phases,world.windows = {"Drone1":0.},{"Drone1":None}
        world.offsets,world.net_width = {"Drone1":np.zeros(3)},2.
        world.traces,world.failures = {"Drone1":[]},{"Drone1":None}
        hit = SimpleNamespace(has_collided=False,time_stamp=101,object_name="gate")
        with patch("controller_architecture_study.framework_runtime_world.reference_point",return_value=ref):
            for stamp in (1_000_000_000,2_000_000_000):
                raw = RawDroneState(ref.position,ref.orientation,ref.velocity,np.zeros(3),np.zeros(3),stamp)
                world.observe("Drone1",raw,hit,float(stamp)/1e9)
        self.assertEqual(world.failures["Drone1"]["t"],1.)


if __name__ == "__main__":
    unittest.main()
