from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from unittest.mock import patch

import numpy as np

from config import load_calibration
from controller_architecture_study.framework_runtime_io import CommandIO
from controller_architecture_study.framework_runtime_editor import read_background_throttle
from controller_architecture_study.probe_seven_continuous_timing import requested_period, urgency, matched_initials
from controller_architecture_study.probe_timing_full_route import following_slot


def test_continuous_period_contract_and_nonclosing_baseline():
    assert requested_period(.040,{"h":.4,"hdot":.1})==.040
    a=requested_period(.040,{"h":.4,"hdot":-.3})
    b=requested_period(.040,{"h":.401,"hdot":-.3})
    assert .020<a<b<.040
    assert b-a < .0001
    assert requested_period(.040,{"h":-.1,"hdot":-2.})==.020
    assert urgency({"h":.4,"hdot":-.6})>urgency({"h":.4,"hdot":-.3})


def test_initial_comparison_includes_held_control_and_every_vehicle():
    a={"a":dict(position=[0],velocity=[0],quaternion=[1],held_control=[9.8])}
    assert matched_initials(a,a)
    assert not matched_initials(a,{"a":{**a["a"],"held_control":[8.]}})
    assert not matched_initials(a,{**a,"b":a["a"]})


def test_extra_correction_returns_to_original_calendar_without_short_interval():
    phase=.040*6/7
    slot,skipped=following_slot(-1,.020,phase,.040)
    assert (slot,skipped)==(0,0)
    assert np.isclose(phase+slot*.040,.03428571428571429)
    assert following_slot(slot,.050,phase,.040)==(1,0)
    # An original slot less than20ms after the corrective job must be skipped.
    assert following_slot(-1,.020,.010,.040)==(1,1)


def test_late_io_is_opt_in_and_keeps_original_deadline():
    class Client:
        def __init__(self,**kwargs):
            self.client=SimpleNamespace(close=lambda:None)
        def ping(self): return True
        def getMultirotorState(self,**kwargs): pass
        def moveByAngleRatesThrottleAsync(self,*args,**kwargs): pass
    with patch("controller_architecture_study.framework_runtime_io.require_airsim",return_value=SimpleNamespace(MultirotorClient=Client)), \
            patch("controller_architecture_study.framework_runtime_io.FrameworkWorld.snapshot_from",return_value=(None,None,1.)):
        for allow in (False,True):
            channel=CommandIO(1,load_calibration(),.020,allow_late=allow)
            try:
                deadline=time.perf_counter()-1.
                row={"vehicle":"Drone1","deadline_s":deadline}
                channel.send(np.array([9.8,0.,0.,0.]),row)
                channel.futures["Drone1"].result(timeout=2.)
                assert row["status"]==("applied" if allow else "expired_not_applied")
                assert row["deadline_s"]==deadline
                if allow:
                    assert row["rpc_return_s"]>deadline
            finally:
                channel.close()


def test_editor_query_follows_fresh_reply_in_numbered_log():
    with TemporaryDirectory(dir=Path.cwd()) as folder:
        old=Path(folder)/"Blocks.log"
        current=Path(folder)/"Blocks_2.log"
        old.write_text("[001]True\n")
        current.write_text("old unrelated log\n")
        def query(command):
            with current.open("a") as f:
                f.write("[002]False\n")
            return True
        client=SimpleNamespace(simRunConsoleCommand=query,client=SimpleNamespace(close=lambda:None))
        with patch("controller_architecture_study.framework_runtime_editor.LOG",old), \
                patch("controller_architecture_study.fly_convex_policy.load_airsim_client"), \
                patch("airsim_interface.require_airsim",return_value=SimpleNamespace(MultirotorClient=lambda **kw:client)):
            result=read_background_throttle()
        assert result["background_throttle_enabled"] is False
        assert result["log"]==str(current)
