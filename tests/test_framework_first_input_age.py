from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch
import unittest

import numpy as np

from airsim_interface import RawDroneState
from controller_architecture_study.probe_controller_step_recoverability import trial


class FirstInputAgeTests(unittest.TestCase):
    def test_delayed_first_command_uses_the_pre_delay_state_not_a_fresh_resample(self):
        def state(t):
            return RawDroneState(np.array([t,0.,-2.]),np.array([1.,0.,0.,0.]),
                np.zeros(3),np.zeros(3),np.zeros(3),round(t*1e9))
        world = SimpleNamespace(initial={"Drone1":{"sim_ns":0}},traces={"Drone1":[]},
            failures={"Drone1":None},windows={"Drone1":None},phases={"Drone1":0.},
            drones={"Drone1":None},calibration=None,client=SimpleNamespace(ping=Mock()))
        world.initialize = Mock(return_value={"Drone1":state(0.)})
        world.snapshot = Mock(return_value=(state(0.),None,0.))
        world.observe = lambda v,s,c,t:world.traces[v].append({"t":s.timestamp_ns/1e9})
        world.physical_report = Mock(return_value={"Drone1":{"recovery_passed":True}})
        pool = SimpleNamespace(predict=Mock(return_value=np.array([9.8,.1,.2,.3])))
        with TemporaryDirectory(dir=Path.cwd()) as temporary:
            with patch("controller_architecture_study.probe_controller_step_recoverability.advance",
                    side_effect=[(state(.040),None,.040),(state(3.),None,3.)]), patch(
                    "controller_architecture_study.probe_controller_step_recoverability.encode_direct",
                    side_effect=lambda w,s,t:np.array([s.position[0]])) as encoder, patch(
                    "controller_architecture_study.probe_controller_step_recoverability.send_control") as send:
                report = trial(world,pool,"model",{},.040,Path(temporary)/"trial",
                    initial_wait=.040,initial_input_age=.040)
        self.assertIsNone(report["error"])
        self.assertAlmostEqual(report["first_actual_input_age_s"],.040)
        self.assertEqual(encoder.call_count,1)
        self.assertEqual(encoder.call_args.args[2],0.)
        self.assertEqual(pool.predict.call_args.args[1][0],0.)
        self.assertEqual(send.call_count,1)

    def test_response_age_cannot_exceed_the_held_control_wait(self):
        with self.assertRaises(ValueError):
            trial(None,None,None,None,None,None,initial_wait=.020,initial_input_age=.040)


if __name__ == "__main__":
    unittest.main()
