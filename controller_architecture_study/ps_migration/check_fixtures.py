"""Numerical reference fixtures for PS ports; these are not flight trials."""
from dataclasses import asdict
import json
from pathlib import Path


def make_features(project):
    import numpy as np
    from airsim_interface import RawDroneState
    from config import load_calibration
    from control_adapter import control_to_airsim_command
    from controller_architecture_study.fast_convex_preprocess import FastReferenceWindow, encode
    from controller_architecture_study.convex_features import encode as scalar_encode
    from controller_architecture_study.fly_decision_dnn import StoredRoutePlanner
    case = json.loads((project/'controller_architecture_study/results/release001_live_case_20261116/case.json').read_text())
    rows = []
    for vehicle, layout in case['layouts'].items():
        planner = StoredRoutePlanner(layout)
        window = FastReferenceWindow(planner.trajectory, planner.config)
        for i, elapsed in enumerate((1.5, 3., 6., 9., 12.)):
            refs = window.window(elapsed, include_body_rates=False)
            state = RawDroneState(refs[0].position+[.1, -.2, .05],
                                  refs[0].orientation*(-1 if i % 2 else 1),
                                  refs[0].velocity+[.2, -.1, .05], np.zeros(3), np.zeros(3))
            feature = encode(state, refs)
            np.testing.assert_allclose(feature, scalar_encode(state, refs), atol=2e-6, rtol=2e-6)
            rows.append(dict(vehicle=vehicle, reference_time_s=elapsed,
                state={k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in asdict(state).items()},
                references=[{k: v.tolist() for k, v in asdict(r).items()} for r in refs],
                expected_raw237=feature.tolist()))
    calibration = load_calibration()
    commands = [dict(physical_control=u, expected_command=asdict(control_to_airsim_command(u, calibration, 1.)))
                for u in ([9.81, 0., 0., 0.], [10.2, .3, -.2, .1], [20., 3., -3., 2.])]
    return dict(scope='constructed numerical parity fixtures, not new flight or recovery evidence',
                feature_atol=2e-6, feature_rtol=2e-6, cases=rows,
                calibration=asdict(calibration), commands=commands)


def check(root):
    import numpy as np
    from airsim_interface import RawDroneState
    from config import ThrustCalibration
    from control_adapter import control_to_airsim_command
    from mpc_controller import MPCReference
    from controller_architecture_study.fast_convex_preprocess import encode
    data = json.loads((root/'fixtures/feature_command_parity.json').read_text())
    for row in data['cases']:
        state = RawDroneState(**row['state'])
        refs = [MPCReference(**r) for r in row['references']]
        np.testing.assert_allclose(encode(state, refs), row['expected_raw237'],
                                   atol=data['feature_atol'], rtol=data['feature_rtol'])
    calibration = ThrustCalibration.from_mapping(data['calibration'])
    for row in data['commands']:
        actual = asdict(control_to_airsim_command(row['physical_control'], calibration, 1.))
        for key, expected in row['expected_command'].items():
            np.testing.assert_allclose(actual[key], expected, atol=1e-12)
    from controller_architecture_study.obligation_scheduler import Calendar, ControlJob, SkipPermission, plan
    from controller_architecture_study.requant18_profile import ControllerMode
    scheduling = json.loads((root/'fixtures/scheduler_parity.json').read_text())
    modes = {m['numeric_id']: ControllerMode(**m) for m in scheduling['modes']}
    for row in scheduling['cases']:
        value = row['calendar']
        calendar = Calendar(tuple(ControlJob(**j) for j in value['jobs']), value['revision'],
                            tuple(tuple(p) for p in value['pending']))
        result = plan(calendar, modes, now=0., decision_ms=scheduling['decision_ms'],
                      permissions=tuple(SkipPermission(**p) for p in row['permissions']))
        actual = {key: result.get(key) for key in row['expected']}
        if json.loads(json.dumps(actual)) != row['expected']:
            raise ValueError(f"Scheduler parity failed at seed {row['seed']}")
    print(f"FIXTURES_PASS features={len(data['cases'])} commands={len(data['commands'])} calendars={len(scheduling['cases'])}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[3])
    check(parser.parse_args().root.resolve())
