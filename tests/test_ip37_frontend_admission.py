from controller_architecture_study.audit_ip37_frontend_admission import inspect


def test_rearm_guard_uses_only_refreshes_already_observed_before_current_ack():
    def job(cycle, **values):
        return dict(vehicle='Drone1', cycle=cycle, release_s=cycle*.05, period_ms=50, **values)

    rows = [
        job(0, status='delivered', command_return_s=.01, paired_sample_start_s=.011, finished_s=.03),
        job(1, status='delivered', command_return_s=.09, paired_sample_start_s=.091, finished_s=.12),
        job(2, status='dropped_busy_io', published_s=.10),
        job(3, status='delivered', command_return_s=.17, finished_s=.21),
        job(4, status='dropped_busy_io', published_s=.20),
    ]
    report = inspect(rows, dict(origin_s=0., decision_budget_ms=3.), False)
    assert report['busy_by_stage'] == dict(post_ack_optional_refresh=1, active_without_ack_or_refresh_marker=1)
    assert report['observed_refreshes'] == 2 and report['past_max_would_defer'] == 1
    block, = report['refresh_blocks']
    assert block['past_max_defers'] and block['unchanged_next_release']
    assert 9.999 < block['ack_to_publish_ms'] < 10.001


def test_retirement_before_critical_phase_does_not_relabel_an_early_command():
    row = dict(vehicle='Drone1', cycle=0, release_s=0., period_ms=80, status='delivered',
               command_return_s=.010, finished_s=.012)
    assert inspect([row], dict(origin_s=0.), True)['critical'] == []
