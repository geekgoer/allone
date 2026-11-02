from controller_architecture_study.probe_timing_full_route import following_slot, timing_win, screening_candidate
from controller_architecture_study.framework_runtime import route_risk
from types import SimpleNamespace

import numpy as np


def test_overrun_discards_old_releases_instead_of_bursting():
    assert following_slot(0,.051,0.,.020)==(3,2)
    assert following_slot(0,.020,0.,.020)==(1,0)
    assert following_slot(3,.067,0.,.020)==(4,0)


def test_timing_win_requires_actual_collision_full_coverage_and_paired_states():
    fast = dict(success=True,initial_state_signature={"position":[0,0,-2]})
    slow = dict(collision_observed=True,full_flight_completed=True,physical={"clock_valid":True},
        initial_state_signature=fast["initial_state_signature"])
    assert timing_win(fast,slow)
    for key in ("collision_observed","full_flight_completed"):
        assert not timing_win(fast,{**slow,key:False})
    assert not timing_win(fast,{**slow,"physical":{"clock_valid":False}})
    assert not timing_win(fast,{**slow,"initial_state_signature":{"position":[1,0,-2]}})
    unsettled = {**fast,"success":False,"full_flight_completed":True,
        "physical":{"physical_task_passed":True,"clock_valid":True}}
    assert screening_candidate(unsettled,slow)
    assert not timing_win(unsettled,slow)


def test_terminal_hover_can_be_observed_without_changing_the_controller():
    ref = SimpleNamespace(position=np.zeros(3),velocity=np.zeros(3),acceleration=np.zeros(3),
        orientation=np.array([2**-.5,0.,0.,2**-.5]))
    raw = SimpleNamespace(position=np.array([.1,0.,0.]),velocity=np.array([.2,0.,0.]))
    risk = route_risk(raw,ref,2.)
    assert np.isclose(risk["h"],.375)
    assert np.isclose(risk["hdot"],-.2)
