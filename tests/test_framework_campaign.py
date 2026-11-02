import numpy as np

from controller_architecture_study.framework_campaign import make_scene


def test_small_mixed_scenes_still_include_all_route_types():
    for n in (3, 6, 10):
        case = make_scene(n, 1000, 'mixed')
        assert {v['family'] for v in case['layouts'].values()} == {'continuous', 'smooth', 'late'}
        assert all(len(v['gates']) == 10 for v in case['layouts'].values())
        assert case['virtual_extra_s'] == 0.
        assert not case['external_impulse'] and not case['initial_tracking_offset']
        assert case['duration_s'] > max(v['duration'] for v in case['layouts'].values())


def test_seeds_change_spacing_and_bends_deterministically():
    a, b = make_scene(3, 1000, 'mixed'), make_scene(3, 1001, 'mixed')
    assert a == make_scene(3, 1000, 'mixed')
    ga = np.array([g['center'] for g in a['layouts']['Drone1']['gates']])
    gb = np.array([g['center'] for g in b['layouts']['Drone1']['gates']])
    assert not np.allclose(ga[:, 0], gb[:, 0])
    assert not np.allclose(ga[:, 1:], gb[:, 1:])


def test_barrier_opt_in_does_not_change_the_original_budget_default():
    from types import SimpleNamespace
    from unittest.mock import patch
    from environment import Gate
    from controller_architecture_study.framework_runtime import analyze_budget
    from controller_architecture_study.held_control_safety import HoldBounds, GateHoldAnalyzer
    state = SimpleNamespace(position=np.array([0., 0., -4.]), velocity=np.array([5., 0., 0.]),
                            quaternion=np.array([1., 0., 0., 0.]))
    gates = [Gate(1, [10., 0., -4.], [1., 0., 0., 0.])]
    analyzer = GateHoldAnalyzer(gates, 1.7, HoldBounds(2., .01, .05, .15, 10., 'test'))
    with patch('controller_architecture_study.framework_runtime.safe_horizon', return_value=(.04, [])):
        old = analyze_budget(state, [9.80665, 0., 0., 0.], 10.01, 10., gates, 1.7)
        new = analyze_budget(state, [9.80665, 0., 0., 0.], 10.01, 10., gates, 1.7, held_analyzer=analyzer)
    assert old['budget_s'] == .04 and not old['uses_control_in_bound']
    assert new['budget_s'] > .14 and new['uses_control_in_bound']
    assert not new['is_safety_certificate'] and not new['actuation_timestamp_certified']
