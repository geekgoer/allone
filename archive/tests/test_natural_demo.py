import numpy as np

from controller_architecture_study.probe_natural_demo import natural_case


def test_natural_case_starts_at_rest_and_contains_no_initial_offset_or_impulse():
    # The actual saved reference is the task: verify rest and full-route coverage,
    # rather than relabeling a disturbed initial condition as natural flight.
    case=natural_case(15,2.)
    from controller_architecture_study.fly_decision_dnn import StoredRoutePlanner
    planner=StoredRoutePlanner(case["layout"])
    initial=planner.reference_window_for_trajectory(planner.trajectory,elapsed=0.,dt=.1,horizon=1)[0]
    np.testing.assert_array_equal(initial.velocity,np.zeros(3))
    np.testing.assert_array_equal(initial.acceleration,np.zeros(3))
    assert case["external_impulse"] is False
    assert case["initial_tracking_offset"] is False
    assert case["phase_s"]==0.
    assert case["phase_span_s"]==0.
    assert case["duration_s"]>=planner.trajectory.duration+case["phase_span_s"]+2.
