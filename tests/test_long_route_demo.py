import numpy as np

from controller_architecture_study.fly_decision_dnn import StoredRoutePlanner
from controller_architecture_study.long_route_demo import build_case, nominal_timing_passed, MODELS
from controller_architecture_study.probe_seven_continuous_timing import collision_vehicle_count, two_sided_period, mode_is_difficult


def test_long_easy_prefix_joins_saved_difficult_tail_without_state_jumps():
    case=build_case()
    trajectory=StoredRoutePlanner(case["layout"]).trajectory
    assert len(case["layout"]["gates"])==17
    for a,b in zip(trajectory.segments[:-1],trajectory.segments[1:]):
        for left,right in zip(a.primitive.evaluate(a.primitive.duration),b.primitive.evaluate(0.)):
            np.testing.assert_allclose(left,right,atol=1e-9)
    for segment in trajectory.segments[1:13]:
        _,v,a=segment.primitive.evaluate(segment.primitive.duration/2)
        assert np.isclose(np.linalg.norm(v),5.)
        np.testing.assert_allclose(a,np.zeros(3),atol=1e-12)
    assert case["external_impulse"] is False
    assert case["initial_tracking_offset"] is False
    assert case["hard_start_reference_s"]>25.


def test_skipped_control_updates_are_not_timing_successes():
    report=dict(nominal_late_jobs=[],nominal_undelivered_deadlines=[],skipped_releases=2,error=None)
    assert not nominal_timing_passed(report)


def test_contact_captured_before_the_next_sensor_tick_is_not_lost():
    traces={"Drone1":[{"collision":False}],"Drone2":[{"collision":True}]}
    assert collision_vehicle_count(traces,{"vehicle":"Drone1"})==2
    assert collision_vehicle_count(traces,{"vehicle":"Drone2"})==1


def test_crossed_boundary_does_not_relax_period_when_closing_speed_reaches_zero():
    assert two_sided_period(.080,dict(h=-.02,hdot=0.,boundary="lateral"),2.)==.020


def test_midcurve_gates_are_reference_points_with_forward_plane_normals():
    case=build_case(width=1.6,mid_curve_gates=True)
    trajectory=StoredRoutePlanner(case["layout"]).trajectory
    for gate,segment in zip(case["layout"]["gates"][13:],trajectory.segments[13:17]):
        p,v,_=segment.primitive.evaluate(segment.primitive.duration/2)
        np.testing.assert_allclose(gate["center"],p,atol=1e-12)
        w,_,_,z=gate["orientation"]
        normal=np.array([w*w-z*z,2*w*z,0.])
        assert normal@v/np.linalg.norm(v)>.999


def test_normal_narrow_gate_margin_is_not_misclassified_as_recovery():
    risk=dict(h=.275,hdot=0.,boundary="lateral")
    assert not mode_is_difficult(risk,np.zeros(3),1.6,relative=True)
    assert mode_is_difficult({**risk,"h":.05},np.zeros(3),1.6,relative=True)
    assert mode_is_difficult(risk,np.array([0.,3.,0.]),1.6,relative=True)


def test_static_search_does_not_omit_any_of_the_six_falcon_architectures():
    assert set(MODELS)=={f"plain_mlp_h{h}_w{w}_seed0" for h in (9,21,33) for w in (512,1024)}


def test_extended_prefix_keeps_tail_dynamics_and_all_gate_offsets():
    from controller_architecture_study.long_route_prefix_search import extend_prefix
    old=build_case(mid_curve_gates=True)
    new=extend_prefix(old,336.)
    a=StoredRoutePlanner(old["layout"]).trajectory
    b=StoredRoutePlanner(new["layout"]).trajectory
    assert len(new["layout"]["gates"])==len(old["layout"]["gates"])
    assert old["layout"]["segments"][1]["duration"]==2.
    for left,right in zip(b.segments[:-1],b.segments[1:]):
        for x,y in zip(left.primitive.evaluate(left.primitive.duration),right.primitive.evaluate(0.)):
            np.testing.assert_allclose(x,y,atol=1e-9)
    shift=np.array([1680.,0.,0.])
    for left,right in zip(a.segments[13:],b.segments[13:]):
        p,v,acc=left.primitive.evaluate(left.primitive.duration*.75)
        q,w,acc2=right.primitive.evaluate(right.primitive.duration*.75)
        np.testing.assert_allclose(q,p+shift,atol=1e-9)
        np.testing.assert_allclose(w,v,atol=1e-12)
        np.testing.assert_allclose(acc2,acc,atol=1e-12)
    for left,right in zip(old["layout"]["gates"][13:],new["layout"]["gates"][13:]):
        np.testing.assert_allclose(np.asarray(right["center"])-left["center"],shift,atol=1e-9)


def test_fast_plane_filter_preserves_original_crossings_in_rotated_frames():
    from controller_architecture_study.safety_comparison_flight import crossing_candidates
    from environment import Gate
    from gate_route import segment_gate_crossing, _rotate_vector
    rng=np.random.default_rng(123)
    for _ in range(8):
        q=rng.normal(size=4);q/=np.linalg.norm(q)
        gate=Gate(1,rng.normal(size=3)+[1800.,0.,-4.],q)
        local=rng.normal(size=(80,3))
        local[:3,0]=[-1e-10,0.,1e-10]
        points=np.array([gate.center+_rotate_vector(q,p) for p in local])
        expected=[i for i in range(len(points)-1) if segment_gate_crossing(points[i],points[i+1],gate)]
        actual=[i for i in crossing_candidates(points,gate) if segment_gate_crossing(points[i],points[i+1],gate)]
        assert expected==actual
