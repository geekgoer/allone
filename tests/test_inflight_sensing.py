from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from controller_architecture_study.inflight_sensing import event_edge, inject_velocity, InflightSensing, boundary_urgency
from controller_architecture_study.probe_seven_continuous_timing import sensor_event_plan, two_sided_period, next_feedback_release
from controller_architecture_study.probe_timing_full_route import following_slot
from controller_architecture_study.framework_runtime import route_risk

import numpy as np


def test_state_event_latches_until_margin_recovers():
    bad={"h":.4,"hdot":-1.}
    assert event_edge(True,bad)==(True,False)
    assert event_edge(False,bad)==(False,False)
    assert event_edge(False,{"h":.2,"hdot":0.})==(False,True)
    assert event_edge(True,{"h":.4,"hdot":.5})==(False,True)


def test_two_sided_prediction_detects_crossing_center_toward_opposite_wall():
    risk={"h":.275,"hdot":1.6}
    assert event_edge(True,risk)==(False,True)
    assert event_edge(True,risk,radius=.475)==(True,False)
    assert boundary_urgency(risk,.475)>1.
    assert event_edge(True,{"h":.275,"hdot":.3},radius=.475)==(False,True)


def test_margin_at_exact_center_has_right_directional_derivative():
    ref=SimpleNamespace(position=np.zeros(3),velocity=np.array([5.,0.,0.]),acceleration=np.zeros(3))
    state=SimpleNamespace(position=np.zeros(3),velocity=np.array([5.,1.6,0.]))
    assert route_risk(state,ref,2.)["hdot"]==-1.6


def test_sustained_feedback_keeps_short_period_while_crossing_toward_opposite_wall():
    risk={"h":.37,"hdot":1.65,"boundary":"lateral"}
    assert two_sided_period(.040,risk,2.)==.020
    assert two_sided_period(.040,{**risk,"hdot":0.},2.)==.040
    assert .020<=two_sided_period(.040,{**risk,"hdot":.3},2.)<.040


def test_feedback_release_uses_actual_previous_release_without_renewing_deadline():
    previous={"planned_release_s":1.,"job_released_s":1.030}
    assert next_feedback_release(previous,1.040,.020,1.060,.010)==1.050
    # An already exhausted promise is logged; it cannot create a <20ms burst.
    assert next_feedback_release(previous,1.040,.020,1.010,.010)==1.050
    assert previous=={"planned_release_s":1.,"job_released_s":1.030}


def test_sensor_correction_keeps_identity_deadline_and_periodic_calendar():
    old=dict(planned_release_s=1.040,deadline_s=1.080,nominal_deadline_s=1.080,
        slot=26,model="frozen",sample_id="a:27",vehicle="a")
    before=deepcopy(old)
    event=dict(deadline_s=1.021,capture_s=1.001,received_s=1.003)
    new=sensor_event_plan(old,event,1.004,.980,.040)
    assert old==before
    assert new["model"]==old["model"] and new["sample_id"]==old["sample_id"]
    assert new["deadline_s"]==1.021 and new["slot"]==25 and new["event_inserted"]
    assert following_slot(new["slot"],1.024,0.,.040)==(26,0)
    # A nearby periodic job is advanced/coalesced, not followed by a burst.
    merged=sensor_event_plan(old,event,1.030,1.000,.040)
    assert not merged["event_inserted"] and merged["slot"]==26
    assert merged["deadline_s"]==event["deadline_s"]  # an expired promise is never silently extended


def test_impulse_changes_only_lateral_velocity_and_preserves_collision_checks():
    xyz=lambda x,y,z:SimpleNamespace(x_val=x,y_val=y,z_val=z)
    state=SimpleNamespace(position=xyz(4.,5.,-3.),orientation=SimpleNamespace(w_val=1.,x_val=0.,y_val=0.,z_val=0.),
        linear_velocity=xyz(5.,.1,0.),angular_velocity=xyz(.2,.3,.4),
        linear_acceleration=xyz(0.,1.,0.),angular_acceleration=xyz(0.,0.,.1))
    class Client:
        def simGetGroundTruthKinematics(self,**kwargs): return deepcopy(state)
        def simSetKinematics(self,value,ignore_collision,vehicle_name):
            self.value,self.ignore,self.vehicle=deepcopy(value),ignore_collision,vehicle_name
        def getMultirotorState(self,**kwargs): return SimpleNamespace(timestamp=2_000_000_000)
    client=Client()
    result=inject_velocity(client,dict(vehicle="Drone3",delta_y_mps=.9,time_s=2.),0,0.)
    assert client.ignore is False and client.vehicle=="Drone3"
    assert vars(client.value.position)==vars(state.position)
    assert vars(client.value.orientation)==vars(state.orientation)
    assert vars(client.value.angular_velocity)==vars(state.angular_velocity)
    assert client.value.linear_velocity.y_val==1.
    assert client.value.linear_velocity.x_val==5.
    assert result["requested_velocity"]==[5.,1.,0.]


def test_controller_sensor_channel_rejects_disturbance_notifications():
    service=InflightSensing.__new__(InflightSensing)
    service.connection=SimpleNamespace(poll=lambda:True,recv=lambda:("injection",None,{}))
    with pytest.raises(RuntimeError,match="non-state"):
        service.drain()


def test_sensor_returns_observed_state_without_test_harness_metadata():
    service=InflightSensing.__new__(InflightSensing)
    raw=SimpleNamespace(timestamp_ns=123)
    messages=[("sample_done",(raw,None,2.),{"vehicle":"Drone5"})]
    service.connection=SimpleNamespace(poll=lambda:bool(messages),recv=lambda:messages.pop(0))
    service.control=SimpleNamespace(poll=lambda:False)
    service.records,service.latest_ns=[],{}
    with patch("controller_architecture_study.inflight_sensing.time.perf_counter",return_value=3.):
        result=service.drain()
    assert result[0]["packet"][0] is raw
    assert service.latest_ns=={"Drone5":123}
    assert set(result[0])=={"packet","row","received_s"}
