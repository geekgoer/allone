from controller_architecture_study.probe_main18_local_delay import ORDERS, pair_rows, protocol


def test_protocol_selects_two_consecutive_pre_gate_releases():
    plan = protocol()
    assert [row['route'] for row in plan['windows']] == ['smooth', 'turns']
    for w in plan['windows']:
        first, last = w['release_indices']
        assert last == first+1
        assert .275 <= w['gate_reference_s']-w['window_start_s'] <= .325
    assert all(sorted(order) == [20, 30, 40] for order in ORDERS)


def test_pair_requires_both_delays_and_matched_physical_state():
    state = dict(sample_s=0., position=[0., 0., 0.], velocity=[0., 0., 0.],
                 quaternion=[1., 0., 0., 0.], held_control=[10., 0., 0., 0.])
    rows = {delay: dict(route='turns', repeat=1, intervention_start=state, mean_age_ms=delay,
                        measured_age_ms=[delay, delay], treatment_delivered=True,
                        crossing_margin_m=.4, min_frame_margin_m=.3, peak_lateral_error_m=.1)
            for delay in (20, 30, 40)}
    rows[30].update(mean_age_ms=30, measured_age_ms=[20., 40.])
    rows[40].update(crossing_margin_m=.35)
    low, high = pair_rows(rows)
    assert not low['applied_contrast_valid'] and low['delta']['crossing_margin_m'] is None
    assert high['applied_contrast_valid'] and high['delta']['crossing_margin_m'] < 0.
    rows[40]['intervention_start'] = {**state, 'position': [.1, 0., 0.]}
    assert not pair_rows(rows)[1]['matched_pre_intervention']
