from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from environment import Gate
from controller_architecture_study.held_control_safety import (
    HoldBounds, GateHoldAnalyzer, barrier_periods, nominal_acceleration, safe_deadline)


def bounds(**kwargs):
    return HoldBounds(**dict(acceleration_error_mps2=0., position_error_m=0., velocity_error_mps=0.,
                             maximum_hold_s=.15, barrier_rate=10., evidence='synthetic exact model', **kwargs))


def test_actual_control_direction_changes_the_period_at_identical_position_velocity():
    b = bounds()
    inward = barrier_periods(.20, -1., 8., 0., b)
    outward = barrier_periods(.20, -1., -8., 0., b)
    assert inward > outward
    assert outward == 0.  # failure to certify is not a claim of instant collision


def test_whole_prefix_not_just_the_end_is_bounded():
    rng = np.random.default_rng(13)
    for _ in range(80):
        d, v, a, j = rng.uniform(.02, 2.), rng.uniform(-2., 2.), rng.uniform(-5., 5.), rng.uniform(0., 30.)
        b = bounds()
        period = float(barrier_periods(d, v, a, j, b))
        if not period:
            continue
        t = np.linspace(0., period, 301)
        h = d+v*t+.5*a*t*t-j*t**3/6
        hd = v+a*t-.5*j*t*t
        hdd = a-j*t
        assert np.min(h) > 0.
        assert np.min(hdd+20*hd+100*h) >= -1e-8


def test_more_uncertainty_cannot_increase_the_safe_period():
    tight = bounds()
    loose = HoldBounds(.8, .01, .03, .15, 10., 'larger test tube')
    assert barrier_periods(.2, -.2, .5, 5., loose) <= barrier_periods(.2, -.2, .5, 5., tight)


def test_gate_boxes_allow_approaching_outside_a_future_aperture():
    gate = Gate(1, [10., 0., -4.], [1., 0., 0., 0.])
    analyzer = GateHoldAnalyzer([gate], 1.7, bounds())
    state = SimpleNamespace(position=np.array([0., 2., -4.]), velocity=np.array([5., 0., 0.]), quaternion=np.array([1., 0., 0., 0.]))
    r = analyzer.analyze(state, [9.80665, 0., 0., 0.])
    assert r['budget_s'] > .149
    assert r['conditional_barrier_certificate'] and not r['is_safety_certificate']
    assert r['uses_control_in_bound']


def test_inside_inflated_bar_has_no_certificate():
    gate = Gate(1, [10., 0., -4.], [1., 0., 0., 0.])
    state = SimpleNamespace(position=np.array([10., .925, -4.]), velocity=np.zeros(3), quaternion=np.array([1., 0., 0., 0.]))
    assert GateHoldAnalyzer([gate], 1.7, bounds()).analyze(state, [9.80665, 0., 0., 0.])['budget_s'] == 0.


def test_quaternion_and_body_rate_dynamics_satisfy_the_certified_obstacle_interval():
    gate = Gate(1, [10., 0., -4.], [1., 0., 0., 0.])
    analyzer = GateHoldAnalyzer([gate], 1.7, bounds())
    state = SimpleNamespace(position=np.array([9.7, .1, -4.]), velocity=np.array([5., .1, 0.]), quaternion=np.array([1., 0., 0., 0.]))
    u = np.array([9.80665, .4, -.3, .2])
    period = analyzer.analyze(state, u)['budget_s']
    assert period > .02
    def rhs(t, x):
        w, a, b, c = x[6:]
        wx, wy, wz = u[1:]
        qdot = .5*np.array([-a*wx-b*wy-c*wz, w*wx+b*wz-c*wy,
                            w*wy+c*wx-a*wz, w*wz+a*wy-b*wx])
        return np.r_[x[3:6], nominal_acceleration(x[6:], u), qdot]
    solution = solve_ivp(rhs, [0., period], np.r_[state.position, state.velocity, state.quaternion],
                         rtol=1e-10, atol=1e-11, dense_output=True)
    positions = solution.sol(np.linspace(0., period, 401))[:3].T
    separation = np.einsum('bfi,ti->tbf', analyzer.normals, positions)-analyzer.offsets
    assert np.min(np.max(separation, axis=-1)) > 0.


def test_measurement_age_consumes_budget_and_ack_uncertainty_is_charged():
    a = dict(budget_s=.05, is_safety_certificate=False)
    r = safe_deadline(10.01, 10.04, 10., a, .08, .002)
    assert r['deadline_s'] == pytest.approx(10.06)
    assert r['remaining_s'] == pytest.approx(.02)
    assert safe_deadline(10.01, 10.07, 10., a, .08)['feasible'] is False
    with pytest.raises(ValueError):
        safe_deadline(10., 10.01, 10.02, a, .08)


def test_zero_quaternion_or_negative_thrust_is_rejected():
    with pytest.raises(ValueError):
        nominal_acceleration([0., 0., 0., 0.], [9.8, 0., 0., 0.])
    with pytest.raises(ValueError):
        nominal_acceleration([1., 0., 0., 0.], [-1., 0., 0., 0.])


def test_full_period_fast_path_matches_all_faces_under_rotated_geometry():
    gate = Gate(1, [10., 1., -4.], [np.cos(.3), 0., 0., np.sin(.3)])
    b = bounds()
    analyzer = GateHoldAnalyzer([gate], 1.7, b)
    rng = np.random.default_rng(19)
    for _ in range(24):
        state = SimpleNamespace(position=np.array([rng.uniform(5., 11.), rng.uniform(-1., 2.), -4.]),
                                velocity=np.array([5., rng.uniform(-1., 1.), 0.]), quaternion=np.array([1., 0., 0., 0.]))
        u = np.array([9.80665, .2, -.3, 0.])
        brute = barrier_periods(analyzer.normals@state.position-analyzer.offsets,
                                analyzer.normals@state.velocity,
                                analyzer.normals@nominal_acceleration(state.quaternion, u),
                                u[0]*np.linalg.norm(u[1:]), b).max(axis=1).min()
        assert abs(analyzer.analyze(state, u)['budget_s']-brute) < 1e-8
