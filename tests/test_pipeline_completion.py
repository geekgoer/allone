from concurrent.futures import Future
from multiprocessing.connection import wait
from types import SimpleNamespace
from unittest.mock import patch
import json

import numpy as np
import pytest

from config import load_calibration
from controller_architecture_study.framework_runtime_io import CommandIO, ProcessCommandIO
from controller_architecture_study.framework_runtime_wait import CompletionEvent
from controller_architecture_study.framework_runtime_features import reference_point, risk_reference
from controller_architecture_study.framework_runtime import route_risk
from controller_architecture_study.probe_seven_continuous_timing import cached_input, calibrated_response_reserve, next_frontend_work
from controller_architecture_study.probe_inflight_events import event_outcomes
from controller_architecture_study.probe_frontend_order import normal_metrics


def test_completion_wakeup_is_auto_reset_and_future_completion_signals_it():
    event=CompletionEvent()
    try:
        assert not wait([event.handle],0.)
        event.set()
        assert wait([event.handle],0.)
        assert not wait([event.handle],0.)
        class Client:
            def __init__(self,**kwargs): self.client=SimpleNamespace(close=lambda:None)
            def ping(self): return True
            def getMultirotorState(self,**kwargs): pass
        with patch("controller_architecture_study.framework_runtime_io.require_airsim",return_value=SimpleNamespace(MultirotorClient=Client)), \
                patch("controller_architecture_study.framework_runtime_io.FrameworkWorld.snapshot_from",return_value=(None,None,1.)):
            io=CommandIO(1,load_calibration(),.020,completed=event.set)
            try:
                io.sample(dict(vehicle="Drone1"))
                assert wait([event.handle],2.)
                io.samples["Drone1"].result(timeout=2.)
                packet,row=io.drain_samples()[0]
                assert row["sample_helper_collected_s"]>=row["sample_io_finished_s"]
            finally:
                io.close()
    finally:
        event.close()


def test_parent_receive_timestamp_preserves_committed_identity_and_deadline():
    io=ProcessCommandIO.__new__(ProcessCommandIO)
    row=dict(sample_id="v:1",vehicle="v",model="fixed",deadline_s=12.)
    io.rows={"v:1":row}; io.samples={"v":Future()}
    result={**row,"sample_message_send_s":10.}
    io._receive(("sample_done",("state","collision",10.),result),10.001)
    assert row["deadline_s"]==12. and row["model"]=="fixed"
    assert row["sample_parent_read_s"]==10.001
    assert io.samples["v"].done()


@pytest.mark.parametrize("speed",[0.,.49,.5,5.])
def test_risk_reference_keeps_geometry_and_hover_orientation(speed):
    class Window:
        config=SimpleNamespace(gravity=9.80665)
        def _evaluate(self,times):
            n=len(times)
            return np.tile([2.,3.,-4.],(n,1)),np.tile([speed,.0,0.],(n,1)),np.tile([.1,.2,0.],(n,1)),np.tile([1.,0.,0.,0.],(n,1))
    w=Window(); raw=SimpleNamespace(position=np.array([2.,3.2,-4.1]),velocity=np.array([speed,1.,.2]))
    assert route_risk(raw,risk_reference(w,1.),2.)==route_risk(raw,reference_point(w,1.),2.)


def test_cache_requires_release_freshness_and_actual_command_time():
    row=dict(planned_release_s=1.)
    record=lambda t:dict(packet=(None,None,t))
    assert cached_input(record(1.001),row,.99,1.005)
    assert not cached_input(record(.999),row,.99,1.003)
    assert not cached_input(record(1.001),row,1.002,1.005)
    assert not cached_input(record(1.001),row,.99,1.022)
    assert not cached_input(None,row,.99,1.005)
    stale={"packet":(SimpleNamespace(timestamp_ns=5),None,1.001)}
    assert not cached_input(stale,row,.99,1.005,6)


def test_release_competes_with_inputs_under_edf_without_changing_static_order():
    old=dict(deadline_s=2.,priority=0,planned_release_s=.9)
    urgent=dict(deadline_s=1.,priority=6,planned_release_s=1.)
    future={**urgent,"planned_release_s":3.}
    edf=lambda r:(r["deadline_s"],r["planned_release_s"])
    static=lambda r:(r["priority"],r["planned_release_s"])
    assert next_frontend_work({}, {"a":(None,old)}, {"b":urgent,"c":future},1.,edf)[:2]==("release","b")
    assert next_frontend_work({}, {"a":(None,old)}, {"b":urgent},1.,static)[:2]==("input","a")


def test_frontend_ablation_changes_only_processing_order():
    waiting=dict(deadline_s=1.,priority=6,planned_release_s=.5)
    newer=dict(deadline_s=2.,priority=0,planned_release_s=.9)
    static=lambda r:(r["priority"],r["planned_release_s"])
    inputs={"low":(None,waiting)}; plans={"high":newer}
    assert next_frontend_work({},inputs,plans,1.,static,"priority")[:2]==("release","high")
    assert next_frontend_work({},inputs,plans,1.,static,"category")[:2]==("input","low")
    assert next_frontend_work({},inputs,plans,1.,static,"deadline")[:2]==("input","low")
    assert waiting==dict(deadline_s=1.,priority=6,planned_release_s=.5)
    assert newer==dict(deadline_s=2.,priority=0,planned_release_s=.9)


def test_later_high_priority_jobs_cannot_overtake_older_deadline_in_deadline_frontend():
    waiting=dict(deadline_s=1.,priority=6,planned_release_s=.5)
    static=lambda r:(r["priority"],r["planned_release_s"])
    for arrival in (.6,.7,.8,.9):
        newer=dict(deadline_s=arrival+1.,priority=0,planned_release_s=arrival)
        assert next_frontend_work({}, {"low":(None,waiting)}, {"high":newer},1.,static,"deadline")[:2]==("input","low")


def test_reserve_charges_cached_input_age_not_only_dispatch_time():
    report={"response_ms":{"p99":2.},"capture_to_command_ms":{"p99":10.}}
    assert calibrated_response_reserve(report)==pytest.approx(.013)


def test_frontend_metrics_charge_missing_slots_and_original_deadlines():
    report=dict(wall_origin_s=0.,duration_s=.12,vehicles={"a":{},"b":{}})
    rows=[dict(vehicle="a",slot=0,status="applied",rpc_return_s=.03,job_released_s=.001),
        dict(vehicle="a",slot=1,status="applied",rpc_return_s=.10,job_released_s=.041),
        dict(vehicle="b",slot=0,status="applied",rpc_return_s=.05,job_released_s=.021),
        dict(vehicle="b",slot=1,status="submitted")]
    result=normal_metrics(report,rows)
    assert result["expected"]==5
    assert result["missing"]==2 and result["late"]==1
    assert result["effective_misses"]==3
    assert result["planned_response_ms"]["max"]==pytest.approx(60.)
    assert result["worst_vehicle_release_p99_ms"]==pytest.approx(1.)


def test_cached_state_captured_before_impulse_is_not_counted_as_fresh(tmp_path):
    requests=[dict(vehicle="Drone1",sample_s=.99,sample_submitted_s=1.01,rpc_return_s=1.02),
        dict(vehicle="Drone1",sample_s=1.03,sample_submitted_s=1.04,rpc_return_s=1.05)]
    (tmp_path/"requests.json").write_text(json.dumps(requests))
    (tmp_path/"sensor_decisions.json").write_text("[]")
    report={"sensor_summary":{"injections":[dict(rpc_return_s=1.,estimated_pose_writeback_bound_m=0.,post_simulation_s=1.)]}}
    outcome=event_outcomes(report,tmp_path,dict(vehicle="Drone1",time_s=1.))
    assert outcome["first_fresh_control_after_impulse_ms"]==pytest.approx(50.)
