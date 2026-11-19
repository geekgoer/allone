from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from airsim_interface import RawDroneState
from controller_architecture_study.probe_main18_full_routes import make_plan, window_metrics
from controller_architecture_study.probe_main18_matched_recovery import airsim_trial


def test_plan_densifies_gates_without_retiming_the_turns():
    plan = make_plan()
    smooth, turns, dense = plan['routes']
    assert plan['expected_trials'] == 12
    assert plan['geometry_audit']['passed']
    assert [r['gate_count'] for r in plan['routes']] == [5, 7, 9]
    assert turns['mean_gate_gap_m'] > dense['mean_gate_gap_m']
    assert smooth['moving_heading_span_deg'] < turns['moving_heading_span_deg']
    assert plan['layouts']['turns']['segments'] == plan['layouts']['dense_turns']['segments']


def test_window_metrics_uses_the_common_interval_without_extrapolation():
    trace = [dict(t=0., position_error_m=0.), dict(t=2., position_error_m=2.)]
    metrics = window_metrics(trace, 1.)
    assert np.isclose(metrics['rmse_m'], 1./np.sqrt(3.))
    assert metrics['peak_m'] == 1.
    assert window_metrics(trace, 3.) is None


@pytest.mark.parametrize('delays, steps, applied, dropped, skip, raw_control', [
    (None, [.020, .047, .053], [.020], None, (), False),
    ({0: .040}, [.040, .047, .053], [.040], [], (), False),
    ({0: .040}, [.051], [], [0], (), False),
    ({0: .020}, [.020, .050], [], [], (0,), False),
    ({0: .024}, [.024, .050], [.024], [], (), True),
])
def test_airsim_terminal_observation_and_control_deadline(tmp_path, delays, steps, applied, dropped, skip, raw_control):
    positions = np.zeros(3)
    q = np.array([1., 0., 0., 0.])

    def sample(t):
        return RawDroneState(positions, q, positions, positions, positions, round(t*1e9)), None, 0.

    world = SimpleNamespace(
        initialize=lambda *_: {'Drone1': sample(0.)[0]}, snapshot=lambda _: sample(0.),
        phases={'Drone1': 0.}, windows={'Drone1': Mock()}, net_width=2.4, gates={'Drone1': []},
        initial={'Drone1': {'held_control': [10., 0., 0., 0.]}}, traces={'Drone1': []},
        failures={'Drone1': None}, drones={'Drone1': None}, calibration=None, client=Mock(),
    )

    def observe(_, raw, *_args):
        world.traces['Drone1'].append(dict(t=raw.timestamp_ns/1e9, position_error_m=0., velocity_error_mps=0.))

    def physical_report(duration, **_):
        return {'Drone1': {'window_covered': world.traces['Drone1'][-1]['t'] >= duration}}

    world.observe, world.physical_report = observe, physical_report
    prefix = 'controller_architecture_study.'
    probe = prefix+'probe_main18_matched_recovery.'
    analyzer = Mock()
    analyzer.analyze.return_value = {'budget_s': .1}
    controller = Mock(return_value=np.zeros(4)) if raw_control else None
    with patch(probe+'geometry', return_value=analyzer), patch(probe+'encode') as encode, \
            patch(probe+'send_control') as send, \
            patch(prefix+'probe_controller_step_recoverability.advance',
                  side_effect=[sample(t) for t in steps]) as advance:
        report, commands, trace = airsim_trial(world, lambda *_: np.zeros(4), 'model', {}, tmp_path/'run',
                                                duration=.05, release_delays=delays, skip_releases=skip,
                                                state_controller=controller)
    if raw_control:
        encode.assert_not_called()
        controller.assert_called_once()
        assert isinstance(controller.call_args.args[0], RawDroneState)
        assert controller.call_args.args[1] is world.windows['Drone1'].window.return_value
    assert send.call_count == len(commands) == len(applied)
    assert [c['apply_s'] for c in commands] == applied
    assert all(c['scheduled_sample_s'] == 0. for c in commands)
    assert report.get('dropped_deadline_releases') == dropped
    assert report.get('intentional_skips', []) == list(skip)
    assert advance.call_count == len(steps)
    assert report['physical']['window_covered'] and trace[-1]['t'] == steps[-1]
