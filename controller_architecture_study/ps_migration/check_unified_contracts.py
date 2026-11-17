"""Check packaged API signs, source identity and relocated Windows binding offline."""
import ast
from dataclasses import asdict
import hashlib
import importlib
import json
from pathlib import Path
import sys


def check(root):
    root = Path(root).resolve()
    sys.path[:0] = [str(root/'source'), str(root/'platforms')]
    from ps_core.controls import ThrustCalibration, control_to_airsim_command, throttle_to_thrust
    native = importlib.import_module('windows.framework_native_io')
    assert native.LIBRARY == root/'platforms/windows/native_runtime/build/Release/falcon_native_io.dll'
    identity = json.loads((root/'contracts/API_SOURCE_IDENTITY.json').read_text())
    python_client = root/'third_party/AirSim/PythonClient/airsim/client.py'
    cpp_client = root/'third_party/AirSim/AirLib/src/vehicles/multirotor/api/MultirotorRpcLibClient.cpp'
    assert hashlib.sha256(python_client.read_bytes()).hexdigest() == identity['python_client']['sha256']
    assert hashlib.sha256(cpp_client.read_bytes()).hexdigest() == identity['cpp_client']['sha256']
    tree = ast.parse(python_client.read_text(encoding='utf-8-sig'))
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'moveByAngleRatesThrottleAsync')
    call = next(n.value for n in ast.walk(fn) if isinstance(n, ast.Return) and isinstance(n.value, ast.Call))
    assert [ast.unparse(a) for a in call.args[1:4]] == ['roll_rate', '-pitch_rate', '-yaw_rate']
    assert 'async_call("moveByAngleRatesThrottle", roll_rate, pitch_rate, yaw_rate,' in cpp_client.read_text()
    cpp = (root/'platforms/windows/native_runtime/native_io.cpp').read_text()
    assert ('moveByAngleRatesThrottleAsync(float(task.control[1]),float(task.control[2]),'
            'float(task.control[3]),') in ''.join(cpp.split())
    calibration = ThrustCalibration.from_mapping(json.loads(
        (root/'source/config/thrust_calibration.json').read_text()))
    cases = json.loads((root/'contracts/wire_command_cases.json').read_text())['cases']
    for case in cases:
        cmd = asdict(control_to_airsim_command(case['controller_output'], calibration, 1.))
        assert cmd == case['python_api_command']
        wire = [cmd['roll_rate'], -cmd['pitch_rate'], -cmd['yaw_rate'], cmd['throttle']]
        assert wire == case['expected_rpc_and_airlib_cpp_arguments']
        actual = [throttle_to_thrust(cmd['throttle'], calibration), *wire[:3]]
        assert max(abs(a-b) for a, b in zip(actual, case['effective_control_after_throttle_clipping'])) < 1e-10
    # Only the two local import statements changed in the pre-existing binding.
    edits = json.loads((root/'PACKAGING_EDITS.json').read_text())
    origins = json.loads((root/'ASSET_ORIGINS.json').read_text())
    for name, replacements in edits.items():
        content = (root/name).read_text()
        for before, after in replacements.items():
            assert content.count(after) == 1
            content = content.replace(after, before)
        # Original source CRLF bytes are preserved by reconstructing that encoding.
        candidates = [content.encode(), content.replace('\n', '\r\n').encode()]
        assert origins[name]['source_sha256'] in [hashlib.sha256(v).hexdigest() for v in candidates]
    forbidden = {'torch', 'scipy', 'osqp', 'controller_architecture_study'}
    assert not any(m.split('.')[0] in forbidden for m in sys.modules)
    return dict(command_cases=len(cases), cpp_and_python_api_hashes=True,
                binding_only_imports_changed=True, legacy_imports_absent=True,
                scope='offline contracts; no Linux build or live flight')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    print(json.dumps(check(parser.parse_args().root), indent=2))
