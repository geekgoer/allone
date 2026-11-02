from controller_architecture_study.cost_ordered_modes import select_mode, delayed_recovery_qualification


def test_cheapest_qualified_is_not_assumed_to_be_the_deepest():
    r = select_mode({'small': 1., 'deep': 3., 'wide': 2.},
                    lambda m: {'status': 'qualified' if m == 'wide' else 'unqualified'},
                    max_checks=3, elapsed_budget_s=.1, fallback='deep')
    assert r['model'] == 'wide' and r['minimum_cost_proven']
    assert [x['model'] for x in r['checked']] == ['small', 'wide']


def test_unknown_does_not_abort_or_prove_minimality():
    r = select_mode({'small': 1., 'large': 2.},
                    lambda m: {'status': 'unknown' if m == 'small' else 'qualified'},
                    max_checks=2, elapsed_budget_s=.1, fallback='large')
    assert r['qualified'] and not r['minimum_cost_proven']
    r = select_mode({'small': 1., 'large': 2.}, lambda m: {'status': 'unqualified'},
                    max_checks=1, elapsed_budget_s=.1, fallback='large')
    assert r['fallback_used'] and not r['qualified']


def test_good_recovery_with_wrong_delay_does_not_qualify():
    evidence = dict(status='empirical_admissible', delay_limit_s=.03,
                    future_update_limit_s=.06, input_age_limit_s=.02)
    assert delayed_recovery_qualification(evidence, delay_s=.04, update_s=.05, input_age_s=.01)['status'] == 'unqualified'
    assert delayed_recovery_qualification(evidence, delay_s=.02, update_s=.05, input_age_s=.01)['status'] == 'qualified'
    assert delayed_recovery_qualification({}, delay_s=.02, update_s=.05, input_age_s=.01)['status'] == 'unknown'
