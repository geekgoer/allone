from copy import deepcopy
from unittest.mock import patch

from controller_architecture_study.bounded_release_planner import plan_releases


def job(v, deadline=.060):
    return dict(vehicle=v, sample_id=v+':1', model='m', priority=0,
                minimum_release_s=0., previous_release_s=0., planned_release_s=.030,
                deadline_s=deadline, requested_period_s=.030)


def test_joint_check_does_not_independently_choose_conflicting_latest_releases():
    future = {v: job(v) for v in ('a', 'b')}
    frozen = deepcopy(future)
    with patch('controller_architecture_study.bounded_release_planner.time.perf_counter', return_value=0.):
        selected, info = plan_releases(0., future, [], {'m': {'segment_estimates_s': [.006]}},
                                      dict(input_s=.001, output_s=.001), inflight_limit=2)
    assert info['feasible'] and 1 < info['checked'] <= 5
    assert future == frozen
    for v in future:
        assert selected[v]['deadline_s'] == future[v]['deadline_s']
        assert selected[v]['model'] == 'm'
        assert .020 <= selected[v]['requested_period_s'] < .060


def test_expired_obligation_is_not_erased_to_make_a_candidate_feasible():
    pending = [{**job('old'), 'deadline_s': -.001}]
    selected, info = plan_releases(0., {'a': job('a')}, pending, {}, {}, inflight_limit=2)
    assert selected is None and info['checked'] == 0
    assert info['already_expired_ids'] == ['old:1']
    assert not info['historical_deadlines_repaired']


def test_no_isolated_capacity_is_reported_without_relaxing_deadline():
    future = {'a': job('a', .021)}
    selected, info = plan_releases(0., future, [], {'m': {'segment_estimates_s': [.010]}},
                                  dict(input_s=.001, output_s=.001), inflight_limit=1)
    assert selected is None and not info['feasible']
    assert future['a']['deadline_s'] == .021
