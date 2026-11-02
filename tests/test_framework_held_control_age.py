import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from controller_architecture_study.probe_recurring_input_age import trial


class HeldControlAgeTests(unittest.TestCase):
    def test_reconstruction_does_not_restart_the_old_controls_update_budget(self):
        raw = SimpleNamespace(timestamp_ns=0)
        world = SimpleNamespace(initial={"Drone1":{"sim_ns":0}},failures={"Drone1":None},
            phases={"Drone1":0.},windows={"Drone1":None},drones={"Drone1":None},calibration=None,
            client=SimpleNamespace(ping=Mock()),traces={"Drone1":[]},observe=Mock(),
            initialize=Mock(return_value={"Drone1":raw}),snapshot=Mock(return_value=(raw,None,0.)),
            physical_report=Mock(return_value={"Drone1":{"recovery_passed":True}}))
        def move(_world,_raw,target):
            raw.timestamp_ns=round(target*1e9)
            return raw
        with TemporaryDirectory(dir=Path.cwd()) as tmp, \
                patch("controller_architecture_study.probe_recurring_input_age.advance_to",side_effect=move), \
                patch("controller_architecture_study.probe_recurring_input_age.encode_direct",return_value=np.zeros(237)), \
                patch("controller_architecture_study.probe_recurring_input_age.send_control"):
            directory=Path(tmp)/"trial"
            result=trial(world,SimpleNamespace(predict=Mock(return_value=[9.8,0.,0.,0.])),"m",{},(.030,),
                directory,held_control_age_s=.025)
            first=json.loads((directory/"commands.json").read_text())[0]
            self.assertIsNone(result["error"])
            self.assertAlmostEqual(first["actuation_simulation_s"],.010)
            self.assertAlmostEqual(first["update_gap_s"],.035)
            self.assertLessEqual(first["input_age_s"],.010)


if __name__=="__main__":
    unittest.main()
