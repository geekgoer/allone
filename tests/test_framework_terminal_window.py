import numpy as np

from controller_architecture_study.framework_runtime_world import FrameworkWorld


def world_with_times(times):
    world=FrameworkWorld.__new__(FrameworkWorld)
    world.net_width=2.
    world.gates={"Drone1":[]}
    world.expected={"Drone1":[]}
    world.failures={"Drone1":None}
    world.traces={"Drone1":[dict(t=t,position=[t,0.,-2.],reference_position=[t,0.,-2.],
        sim_ns=int(t*1e9),sample_s=10.+t,position_error_m=0.,velocity_error_mps=0.,
        state_cost=1.,risk={"h":.4}) for t in times]}
    return world


def test_post_task_snapshot_gap_does_not_fail_already_covered_terminal_window():
    world=world_with_times([0.,2.75,2.84,2.93,3.01,3.15])
    p=world.physical_report(3.)["Drone1"]
    assert p["tail_observation_coverage_passed"]
    assert p["recovery_passed"]
    assert np.isclose(p["max_tail_observation_gap_s"],.09)


def test_gap_inside_terminal_window_still_fails():
    world=world_with_times([0.,2.75,2.94,3.01,3.15])
    assert not world.physical_report(3.)["Drone1"]["tail_observation_coverage_passed"]
