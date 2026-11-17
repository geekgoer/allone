"""Check generated PS kernels against frozen numeric fixtures, without legacy imports."""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from ps_core.controls import ThrustCalibration, control_to_airsim_command
from ps_core.features import FastReferenceWindow, encode
from ps_core.hold import GateHoldAnalyzer, HoldBounds
from ps_core.profile import ControllerMode
from ps_core.scheduler import Calendar, ControlJob, SkipPermission, plan
from ps_core.types import RawDroneState, ReferencePoint, Gate
from ps_core.trajectory import from_layout


def reference_window(layout):
    """Use the runtime loader so verification cannot hide a missing adapter."""
    return from_layout(layout)


def check(root):
    root = Path(root)
    data = json.loads((root/'verification/feature_command_parity.json').read_text())
    case = json.loads((root/'config/case.json').read_text())
    for row in data['cases']:
        state = RawDroneState(**row['state'])
        refs = [ReferencePoint(**r) for r in row['references']]
        np.testing.assert_allclose(encode(state, refs), row['expected_raw237'], atol=2e-6, rtol=2e-6)
        calculated = reference_window(case['layouts'][row['vehicle']]).window(row['reference_time_s'], include_body_rates=False)
        for actual, expected in zip(calculated, refs):
            for field in ('position', 'orientation', 'velocity', 'acceleration'):
                np.testing.assert_allclose(getattr(actual, field), getattr(expected, field), atol=2e-6, rtol=2e-6)
    calibration = ThrustCalibration.from_mapping(data['calibration'])
    for row in data['commands']:
        actual = asdict(control_to_airsim_command(row['physical_control'], calibration, 1.))
        for key, expected in row['expected_command'].items():
            np.testing.assert_allclose(actual[key], expected, atol=1e-12)
    scheduling = json.loads((root/'verification/scheduler_parity.json').read_text())
    modes = {m['numeric_id']: ControllerMode(**m) for m in scheduling['modes']}
    for row in scheduling['cases']:
        value = row['calendar']
        calendar = Calendar(tuple(ControlJob(**j) for j in value['jobs']), value['revision'], tuple(tuple(p) for p in value['pending']))
        result = plan(calendar, modes, now=0., decision_ms=scheduling['decision_ms'],
                      permissions=tuple(SkipPermission(**p) for p in row['permissions']))
        actual = {key: result.get(key) for key in row['expected']}
        assert json.loads(json.dumps(actual)) == row['expected'], row['seed']
    print(f"PS_SCOPE_FIXTURES_PASS features_and_references={len(data['cases'])} commands={len(data['commands'])} calendars={len(scheduling['cases'])}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    check(parser.parse_args().root)
