from copy import deepcopy
from itertools import product
from unittest.mock import patch

import numpy as np
import pytest

from controller_architecture_study.framework_runtime import simulate_pipeline
from controller_architecture_study.joint_contract_optimizer import optimize, priced_candidates


def job(v, deadline=.080):
    return dict(vehicle=v, sample_id=v+':1', model='large', priority=0,
                previous_release_s=0., minimum_release_s=.020,
                planned_release_s=.040, requested_period_s=.040, deadline_s=deadline)


def action(name, model, period, value):
    return dict(id=name, model=model, release_s=period, value=value)


def test_joint_speedup_requires_other_loop_relief_and_preserves_obligations():
    future = {'urgent':job('urgent', .034), 'calm':job('calm', .090)}
    profile = {'large':{'segment_estimates_s':[.010,.010]}, 'small':{'segment_estimates_s':[.002]}}
    options = {'urgent':[action('fast','small',.020,10.)],
               'calm':[action('keep','large',.040,5.), action('relief','small',.060,4.)]}
    frozen = deepcopy((future, options))
    costs = dict(input_s=.004, output_s=.004)
    # 0.008/0.020 + 0.008/0.040 > the supplied frontend capacity.
    isolated, reason = optimize(0., future, [], {**options, 'calm':options['calm'][:1]}, profile, costs,
        inflight_limit=2, capacities=(.95,.55), planning_budget_s=.010)
    assert isolated is None and reason['reason'] == 'even_resource_lower_bound_exceeds_capacity'
    with patch('controller_architecture_study.joint_contract_optimizer.time.perf_counter', return_value=0.):
        selected, report = optimize(0., future, [], options, profile, costs,
            inflight_limit=2, capacities=(.95,.55), planning_budget_s=.010)
    assert report['accepted'] and selected['calm']['model'] == 'small'
    assert selected['urgent']['requested_period_s'] == .020
    assert selected['calm']['requested_period_s'] == .060
    assert report['value'] == 14. and report['resource_dual_upper'] >= 14.-1e-9
    assert (future, options) == frozen
    for v in future:
        assert selected[v]['deadline_s'] == future[v]['deadline_s']
        assert selected[v]['sample_id'] == future[v]['sample_id']
    assert not report['installed'] and not report['physical_safety_certified']


def test_price_bound_dominates_exhaustive_optimum_without_monotone_model_assumption():
    rng = np.random.default_rng(24)
    for _ in range(24):
        choices = [[dict(id=f'{i}/{j}', value=float(rng.uniform(-2,3)),
                        demand=rng.uniform(.02,.4,size=2)) for j in range(4)] for i in range(3)]
        cap = np.array([.7,.7])
        feasible = [(sum(choices[i][j]['value'] for i,j in enumerate(ix)), ix)
            for ix in product(range(4),repeat=3)
            if np.all(sum((choices[i][j]['demand'] for i,j in enumerate(ix))) <= cap)]
        candidates, info = priced_candidates(choices, cap, 12)
        if feasible:
            optimum = max(v for v,ix in feasible)
            assert info['resource_dual_upper'] >= optimum-1e-9
        for candidate in candidates:
            assert max(np.array(candidate['load'])-cap) <= 1e-9
            assert candidate['value'] <= info['resource_dual_upper']+1e-9


def test_segment_blocking_of_existing_job_cannot_be_ignored():
    future = {'new':job('new', .034)}
    pending = [{**job('running'), 'model':'blocker', 'sample_id':'running:0', 'stage':'compute',
                'planned_release_s':-.001, 'deadline_s':.100}]
    profile = {'large':{'segment_estimates_s':[.008]}, 'blocker':{'segment_estimates_s':[.015,.015]}}
    options = {'new':[action('fast','large',.020,1.)]}
    before = deepcopy(pending)
    with patch('controller_architecture_study.joint_contract_optimizer.time.perf_counter', return_value=0.):
        selected, info = optimize(0., future, pending, options, profile,
            dict(input_s=.001, output_s=.001), inflight_limit=2)
    assert selected is None
    assert len(info['checks']) == 1 and not info['checks'][0]['passed']
    assert pending == before and not info['deadlines_relaxed']


def test_full_edf_forecast_rejects_a_resource_feasible_but_simultaneous_deadline_conflict():
    future = {v:job(v, .034) for v in ('a','b')}
    options = {v:[action('one','large',.020,1.)] for v in future}
    profile = {'large':{'segment_estimates_s':[.010]}}
    with patch('controller_architecture_study.joint_contract_optimizer.time.perf_counter', return_value=0.):
        selected, report = optimize(0., future, [], options, profile,
            dict(input_s=.001, output_s=.001), inflight_limit=2, capacities=(1.1,.95))
    assert selected is None and len(report['checks']) == 1
    assert not report['checks'][0]['passed']


def test_accepted_value_is_actual_finite_table_optimum_when_unconstrained_best_fits():
    future = {v:job(v, .100) for v in ('a','b','c')}
    options = {v:[action('wide','large',.050,1.), action('narrow','small',.040,2.)] for v in future}
    profile = {'large':{'segment_estimates_s':[.002,.002]}, 'small':{'segment_estimates_s':[.001]}}
    with patch('controller_architecture_study.joint_contract_optimizer.time.perf_counter', return_value=0.):
        selected, report = optimize(0., future, [], options, profile,
            dict(input_s=.001, output_s=.001), inflight_limit=3)
    assert report['accepted'] and report['value'] == 6. and report['duality_gap'] == 0.
    ok, finishes, _ = simulate_pipeline(0., list(selected.values()), [], profile,
        dict(input_s=.001,output_s=.001,frontend_reserved_until_s=.006), inflight_limit=3)
    assert ok and len(finishes) == 3


def test_expired_obligation_and_planning_overrun_never_become_safe():
    future = {'a':job('a')}
    options = {'a':[action('one','large',.040,1.)]}
    profile = {'large':{'segment_estimates_s':[.001]}}
    pending = [{**job('old'), 'deadline_s':-.001}]
    selected, info = optimize(0.,future,pending,options,profile,{},inflight_limit=2)
    assert selected is None and info['expired_ids'] == ['old:1']
    ticks = iter([0.,0.,.020])
    with patch('controller_architecture_study.joint_contract_optimizer.time.perf_counter', side_effect=lambda:next(ticks)):
        selected, info = optimize(0.,future,[],options,profile,
            dict(input_s=.001,output_s=.001),inflight_limit=2)
    assert selected is None and info['reason'] == 'planning_budget_exceeded'
    assert info['checks'][0]['passed']


def test_released_job_and_nonfinite_budget_are_rejected():
    row = job('a')
    options = {'a':[action('one','large',.040,1.)]}
    for future, pending, budget in [({'a':row}, [row], .006),
                                  ({'a':{**row,'stage':'compute'}}, [], .006),
                                  ({'a':row}, [], float('nan'))]:
        with pytest.raises(ValueError):
            optimize(0.,future,pending,options,{}, {},inflight_limit=2,planning_budget_s=budget)


def test_input_freshness_deadline_is_not_relaxed_with_physical_deadline():
    future = {'a':{**job('a',.100), 'input_age_limit_s':.010}}
    options = {'a':[action('one','large',.020,1.)]}
    with patch('controller_architecture_study.joint_contract_optimizer.time.perf_counter', return_value=0.):
        selected, report = optimize(0.,future,[],options,{'large':{'segment_estimates_s':[.015]}},
            dict(input_s=.001,output_s=.001),inflight_limit=2)
    assert selected is None and not report['checks'][0]['passed']
    assert future['a']['input_age_limit_s'] == .010


def test_completed_history_is_reported_without_requiring_a_retroactive_guard():
    future = {'a':job('a')}
    options = {'a':[action('one','large',.040,1.)]}
    profile = {'large':{'segment_estimates_s':[.001]}}
    pending = [{**job('old',-.001), 'stage':'analysis', 'command_completed_s':-.0005,
                'analysis_remaining_s':.001, 'nonblocking_maintenance':True}]
    with patch('controller_architecture_study.joint_contract_optimizer.time.perf_counter', return_value=0.):
        selected, report = optimize(0.,future,pending,options,profile,
            dict(input_s=.001,output_s=.001,completion_guard_s=.0005),inflight_limit=2)
    assert selected is not None and report['previous_completed_misses'] == ['old:1']
    assert not report['historical_deadlines_repaired']
