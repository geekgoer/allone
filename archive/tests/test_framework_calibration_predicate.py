from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import numpy as np

from controller_architecture_study.framework_runtime_calibrate import rollout


class World:
    def __init__(self, collision=False):
        self.t = 0.
        self.collide = collision
        self.traces = {"Drone1":[]}
        self.failures = {"Drone1":None}
        self.windows = {"Drone1":SimpleNamespace(window=lambda *a:[])}
        self.phases = {"Drone1":0.}
        self.drones,self.calibration = {"Drone1":None},None
        self.client = SimpleNamespace(getMultirotorState=lambda **k:SimpleNamespace(collision=None),
            simPause=lambda *a:None,ping=lambda:None)

    def initialize(self, *args):
        return {"Drone1":None}

    def snapshot(self, *args):
        self.t += .25
        if self.collide:
            self.failures["Drone1"] = {"kind":"collision"}
        return None,None,0.

    def observe(self, *args):
        self.traces["Drone1"].append({"t":self.t,"risk":{"h":-.01,"hdot":0.}})

    def physical_report(self, duration, *, enforce_tube, realtime):
        self.used_tube = enforce_tube
        return {"Drone1":{"clock_valid":True,"recovery_passed":
            self.t>=duration and not self.failures["Drone1"] and not enforce_tube}}


class CalibrationPredicateTests(unittest.TestCase):
    def execute(self, tube, collision=False):
        world = World(collision)
        pool = SimpleNamespace(predict=lambda *a:np.array([9.8,0.,0.,0.]))
        clock = iter(np.arange(0.,20.,.005))
        with TemporaryDirectory(dir=Path.cwd()) as tmp, \
                patch("controller_architecture_study.framework_runtime_calibrate.encode",return_value=np.zeros(237)), \
                patch("controller_architecture_study.framework_runtime_calibrate.send_control"), \
                patch("controller_architecture_study.framework_runtime_calibrate.time.perf_counter",side_effect=lambda:next(clock)), \
                patch("controller_architecture_study.framework_runtime_calibrate.time.sleep"):
            report = rollout(world,pool,"model",(.15,-.5),0.,Path(tmp)/"trial",enforce_tube=tube)
        return world,report

    def test_gate_task_collects_the_complete_window_after_reference_tube_exit(self):
        strict,old = self.execute(True)
        gate,new = self.execute(False)
        self.assertLess(strict.t,3.)
        self.assertFalse(old["recovery_passed"])
        self.assertGreaterEqual(gate.t,3.)
        self.assertTrue(new["recovery_passed"])
        self.assertFalse(gate.used_tube)
        self.assertEqual(new["safety_predicate"],"gate_task")

    def test_gate_task_still_stops_on_an_actual_collision(self):
        world,report = self.execute(False,collision=True)
        self.assertLess(world.t,3.)
        self.assertFalse(report["recovery_passed"])


if __name__ == "__main__":
    unittest.main()
