"""One migration delivery: scoped Python, original C++ IO, DPU, SDK and evidence."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT/'controller_architecture_study'
BASE = STUDY/'results/ps_verified_handoff_20261116_002/route_b_ps_verified_20261116'
CPP = STUDY/'results/cpp_reuse_addendum_20261116_001/route_b_cpp_addendum_20261116'
DELIVERY = STUDY/'docs/main18_requant_release001_delivery_20261115'
SDK = STUDY/'results/ps_migration_handoff_20261116_002/route_b_ps_handoff_20261116/third_party_reference/AirSim'


def build(output):
    if output.exists():
        raise ValueError('Use a new output directory; previous deliveries are immutable')
    package = output/'route_b_unified_20261116'
    package.mkdir(parents=True)
    roles, origins, edits = {}, {}, {}

    def copy(path, relative, role):
        target = package/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.read_bytes() != path.read_bytes():
                raise ValueError(f'Different copies at {relative}')
        else:
            shutil.copyfile(path, target)
        key = Path(relative).as_posix()
        roles[key] = role
        origins[key] = dict(source=(path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)),
                            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        return target

    def tree(folder, relative, role):
        for path in sorted(folder.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and not path.name.endswith('.pyc'):
                copy(path, Path(relative)/path.relative_to(folder), role)

    for name in ('source', 'evidence', 'verification_inputs'):
        tree(BASE/name, name, 'python_reference_or_frozen_evidence')
    # One physical copy of each deployment asset, not a partial reference plus a second full package.
    deploy = DELIVERY/'board_package_requant18_release001'
    tree(deploy, 'reference_assets/deployment', 'frozen_board_deployment_and_numerical_reference')
    for name in ('verify_bundle.py', 'audit_evidence.py'):
        copy(BASE/'tools'/name, 'tools/'+name, 'offline_verification')
    for name in ('check_unified_contracts.py', 'check_unified.py', 'compare_prior_reading.py'):
        copy(STUDY/'ps_migration'/name, 'tools/'+name, 'offline_verification')
    for name in ('native_io.cpp', 'CMakeLists.txt'):
        copy(STUDY/'native_runtime'/name, 'platforms/windows/native_runtime/'+name,
             'original_windows_cpp_not_linux_implementation')
    target = copy(STUDY/'framework_native_io.py', 'platforms/windows/framework_native_io.py',
                  'existing_ctypes_binding_with_import_relocation_only')
    content = target.read_text(encoding='utf-8')
    replacements = {'from airsim_interface import RawDroneState': 'from ps_core.types import RawDroneState',
                    'from control_adapter import control_to_airsim_command, throttle_to_thrust':
                    'from ps_core.controls import control_to_airsim_command, throttle_to_thrust'}
    for before, after in replacements.items():
        assert content.count(before) == 1
        content = content.replace(before, after)
    target.write_text(content, encoding='utf-8')
    edits[target.relative_to(package).as_posix()] = replacements
    for name in ('__init__.py', 'build_native_runtime.ps1', 'build_runner.sh'):
        relative = 'platforms/'+('board/' if name == 'build_runner.sh' else 'windows/')+name
        copy(STUDY/'ps_migration/unified_platforms'/name, relative, 'package_local_build_or_namespace')
    # Retain only native tests independent of old ABC/JointRuntime, not the full old test import closure.
    original = ROOT/'tests/test_framework_native_io.py'
    parsed = ast.parse(original.read_text(encoding='utf-8'))
    functions = [node for node in parsed.body if isinstance(node, ast.FunctionDef)
                 and not ({'JointRuntime', 'abc_repair'} & {x.id for x in ast.walk(node) if isinstance(x, ast.Name)})
                 and not any(isinstance(x, ast.ImportFrom) and (x.module or '').startswith('controller_architecture_study')
                             for x in ast.walk(node))]
    header = ('import time\nimport numpy as np\nimport pytest\n'
              'from ps_core.controls import ThrustCalibration\n'
              'from windows.framework_native_io import LIBRARY, NativeCommandIO\n'
              "pytestmark = pytest.mark.skipif(not LIBRARY.exists(), reason='build packaged native runtime first')\n\n")
    tests = package/'platforms/windows/test_native_io.py'
    command_test = STUDY/'ps_migration/unified_platforms/native_command_test.py'
    tests.write_text(header+'\n\n'.join(ast.unparse(n) for n in functions)+'\n'
                     +command_test.read_text(encoding='utf-8'), encoding='utf-8')
    origins[tests.relative_to(package).as_posix()] = dict(source='tests/test_framework_native_io.py',
        source_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
        retained_symbols=[n.name for n in functions], transform='import relocation; original independent test ASTs',
        additional_command_test_source=command_test.relative_to(ROOT).as_posix(),
        additional_command_test_sha256=hashlib.sha256(command_test.read_bytes()).hexdigest())
    roles[tests.relative_to(package).as_posix()] = 'native_io_mock_tests_not_dpu_or_flight_simulation'
    # Build dependencies for the existing client, without Unreal, CPU DNN or training source.
    for name in ('AirLib/include', 'AirLib/deps/eigen3', 'MavLinkCom/include',
                 'MavLinkCom/common_utils', 'MavLinkCom/mavlink',
                 'external/rpclib/rpclib-2.3.0'):
        tree(SDK/name, 'third_party/AirSim/'+name, 'airsim_cpp_rpc_build_dependency')
    for name in ('AirLib/src/api/RpcLibClientBase.cpp',
                 'AirLib/src/vehicles/multirotor/api/MultirotorRpcLibClient.cpp',
                 'AirLib/src/common/common_utils/FileSystem.cpp', 'LICENSE', 'SNAPSHOT_NOTE.md'):
        copy(SDK/name, 'third_party/AirSim/'+name, 'airsim_cpp_rpc_build_dependency')
    # rpclib configures this documentation template even with examples/tests disabled.
    copy(Path('F:/Airsim/AirSim/external/rpclib/rpclib-2.3.0/doc/pages/versions.md.in'),
         'third_party/AirSim/external/rpclib/rpclib-2.3.0/doc/pages/versions.md.in',
         'required_rpclib_cmake_configure_input')
    # Needed to audit the Python-vs-C++ sign convention, not a second runtime SDK.
    copy(Path('F:/Airsim/AirSim/PythonClient/airsim/client.py'),
         'third_party/AirSim/PythonClient/airsim/client.py', 'api_sign_audit_reference')
    for name in ('wire_command_cases.json', 'API_SOURCE_IDENTITY.json'):
        copy(CPP/name, 'contracts/'+name, 'python_rpc_cpp_sign_contract')
    for name in ('PS_UNIFIED_HANDOFF_20261116.md', 'UBUNTU_READING_CORRECTION_20261116.md',
                 'CPP_REUSE_AND_API_CONTRACT_20261116.md', 'ROUTE_B_LIVE_RELEASE001_20261116.md'):
        copy(STUDY/'docs'/name, ('README_FIRST.md' if name.startswith('PS_UNIFIED') else 'docs/'+name),
             'current_migration_instructions')
    mapping = dict(python_core='source/ps_core',
        unchanged_since_verified002=sorted(p.relative_to(BASE).as_posix() for p in (BASE/'source').rglob('*') if p.is_file()),
        symbol_sources=json.loads((package/'source/SOURCE_MAP.json').read_text())['symbols'],
        reusable_cpp={
            'controller_architecture_study/native_runtime/native_io.cpp': 'platforms/windows/native_runtime/native_io.cpp',
            'controller_architecture_study/framework_native_io.py': 'platforms/windows/framework_native_io.py',
            'board_package_requant18_release001/pool237_runner.cc': 'reference_assets/deployment/pool237_runner.cc',
            'board_package_requant18_release001/pool237_config.h': 'reference_assets/deployment/pool237_config.h'},
        do_not_reread=['training/teacher solvers', 'CPU DNN/CPU segmented accelerator', 'old quantization services',
                      'old ABC integration when implementing the obligation scheduler'],
        boundary='Windows native IO is reusable source; no new obligation-to-native or complete PS event loop is claimed')
    def save(name, value):
        (package/name).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    save('MIGRATION_MAP.json', mapping)
    dependencies = json.loads((BASE/'EXTERNAL_DEPENDENCIES.json').read_text())
    dependencies['deployment']['bundled_location'] = 'reference_assets/deployment'
    dependencies['sdk'] = dict(bundled_location='third_party/AirSim',
        boundary='sources needed for the native client; not Unreal or a compiled ARM library')
    dependencies['external_platform_only'] = ['Linux cross toolchain/sysroot, installed DNNDK and board DPU driver/bitstream',
                                            'Unreal/AirSim simulator; MSVC/CMake for Windows IO rebuild',
                                            'Python >=3.11, NumPy; pytest for native tests']
    save('EXTERNAL_DEPENDENCIES.json', dependencies)
    save('COMPLETENESS.json', dict(
        source_delivery_self_contained=True,
        depends_on_previous_project_or_archive=False,
        complete_ps_application_implemented=False,
        supplied=['12 Python numerical/lifecycle/scheduler reference modules and fixtures',
                  'original Windows C++ release/sample/command code, relocated ctypes binding, local build and tests',
                  'original board C++ runner/header, all18 ELF, binary, normalization, golden and validation',
                  'RPC source build dependencies, raw timing CSV, getter evidence and five flight logs'],
        pending=['port Windows clock/event/affinity/IO implementation to Linux',
                 'connect new scheduler and raw237 to PS release/sample/local-DPU/command event loop',
                 'ARM compile, physical state matching, local DPU golden and end-to-end fixed/dynamic flight'],
        not_proven=['WCET or physical safety certificate', 'skip/recovery eligibility', 'dynamic advantage over best static J']))
    save('FILE_ROLES.json', roles)
    save('ASSET_ORIGINS.json', origins)
    save('PACKAGING_EDITS.json', edits)
    hashes = {p.relative_to(package).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(package.rglob('*')) if p.is_file()}
    save('HANDOFF_MANIFEST.json', dict(format='route_b_unified_v1', date='2026-11-16', sha256=hashes,
        scope='self-contained source and assets; Windows and DNNDK implementations, PS orchestration pending'))
    archive = output/(package.name+'.tar.gz')
    def permissions(info):
        info.uid = info.gid = 0
        info.uname = info.gname = ''
        info.mode = 0o755 if info.isdir() or info.name.endswith(('.sh', '/pool237_runner')) else 0o644
        return info
    with tarfile.open(archive, 'w:gz', compresslevel=6) as tar:
        tar.add(package, arcname=package.name, filter=permissions)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_name(archive.name+'.sha256').write_text(f'{digest}  {archive.name}\n', encoding='ascii')
    print(json.dumps(dict(archive=str(archive), bytes=archive.stat().st_size, sha256=digest, files=len(hashes)+1), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    build(parser.parse_args().output.resolve())
