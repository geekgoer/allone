"""Extract PS-relevant symbols, rather than the old Windows import closure."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def build(output):
    if output.exists():
        raise ValueError('Use a new output directory')
    package = output/'route_b_ps_scope_20261116'
    core = package/'ps_core'
    core.mkdir(parents=True)
    (core/'__init__.py').write_text('"""PS-port numerical reference; hardware adapters are external."""\n')
    origins = []

    def symbols(source, names, methods=None, rename=None):
        source = Path(source)
        text = (ROOT/source).read_text(encoding='utf-8-sig')
        tree = ast.parse(text)
        nodes = {n.name: n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
        pieces = []
        for name in names:
            original = nodes[name]
            node = copy.deepcopy(original)
            if methods and name in methods:
                node.body = [n for n in node.body if not isinstance(n, ast.FunctionDef) or n.name in methods[name]]
                for method in node.body:
                    if isinstance(method, ast.FunctionDef) and method.name == '__init__':
                        method.body = [n for n in method.body if not (
                            isinstance(n, ast.Assign) and any(isinstance(t, ast.Attribute)
                            and t.attr == 'native_io' for t in n.targets))]
            if rename:
                class Rewrite(ast.NodeTransformer):
                    def visit_Name(self, item):
                        item.id = rename.get(item.id, item.id)
                        return item
                    def visit_FunctionDef(self, item):
                        item.name = rename.get(item.name, item.name)
                        return self.generic_visit(item)
                    def visit_ClassDef(self, item):
                        item.name = rename.get(item.name, item.name)
                        return self.generic_visit(item)
                node = Rewrite().visit(node)
            pieces.append(ast.unparse(ast.fix_missing_locations(node)))
            origins.append(dict(source=source.as_posix(), symbol=name,
                start_line=min([original.lineno]+[n.lineno for n in original.decorator_list]),
                end_line=original.end_lineno, retained_methods=methods.get(name) if methods else None,
                source_sha256=hashlib.sha256((ROOT/source).read_bytes()).hexdigest()))
        return '\n\n'.join(pieces)+'\n'

    def module(name, header, body):
        (core/name).write_text('# Generated PS-relevant reference; see SOURCE_MAP.json.\n'
                              +header+'\n\n'+body, encoding='utf-8')

    module('profile.py', 'from dataclasses import dataclass\nimport hashlib\nimport json\nimport math\nfrom pathlib import Path',
           symbols('controller_architecture_study/requant18_profile.py', ('ControllerMode', 'pareto', 'read_profile')))
    module('backend_contract.py', 'from dataclasses import dataclass\nfrom typing import Protocol',
           symbols('controller_architecture_study/controller_backend.py', ('Configuration', 'ControllerBackend')))
    materialized = '''def make_jobs(scenario, choice, group):
    if scenario.loops or choice or group != 1:
        raise ValueError('PS scope accepts materialized whole-job obligations only')
    if any(j.remaining_operations is None or sum(r == 'pl' for r, _ in j.remaining_operations) > 1 for j in scenario.carry):
        raise ValueError('Explicit whole-task paths required')
    return tuple(scenario.carry)

def chain(job, platform):
    if job.remaining_operations is None:
        raise ValueError('Explicit whole-task path required')
    return list(job.remaining_operations)
'''
    module('forecast.py', 'from dataclasses import dataclass\nimport heapq\nimport itertools',
           symbols('controller_architecture_study/bottleneck_joint.py', ('Platform', 'Job', 'Scenario'))
           +materialized+symbols('controller_architecture_study/bottleneck_joint.py', ('isolated_finish', 'simulate', 'witness')))
    scheduler_names = [n.name for n in ast.parse((ROOT/'controller_architecture_study/obligation_scheduler.py').read_text()).body
                       if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
    module('scheduler.py', 'from dataclasses import dataclass, replace\nimport math\n'
           'from .forecast import Job, Platform, Scenario, simulate, witness\nEPS = 1e-8',
           symbols('controller_architecture_study/obligation_scheduler.py', scheduler_names))
    lifecycle_methods = ('__init__', 'release_due', 'observe', 'command_changed', 'update_cpu_costs',
                         'enqueue', 'board_done', 'finish', 'retire', 'public_rows')
    module('lifecycle.py', 'from concurrent.futures import Future\nfrom collections import Counter\nimport threading\nimport time',
           symbols('controller_architecture_study/ip37_joint_runtime.py', ('JointRuntime',),
                   methods={'JointRuntime': lifecycle_methods}))
    module('runtime.py', 'from dataclasses import replace\nimport time\n'
           'from .lifecycle import JointRuntime\nfrom .scheduler import Calendar, ControlJob, SkipPermission, plan',
           symbols('controller_architecture_study/ip37_obligation_runtime.py', ('ObligationRuntime',)))
    module('types.py', 'from __future__ import annotations\nfrom dataclasses import dataclass, field\n'
           'from typing import Any\nimport numpy as np',
           symbols('state_adapter.py', ('_quaternion', 'normalize_quaternion', 'quaternion_multiply'))
           +symbols('airsim_interface.py', ('RawDroneState',))
           +symbols('mpc_controller.py', ('_vec3', 'MPCReference'), rename={'MPCReference': 'ReferencePoint'})
           +symbols('environment.py', ('Gate',)))
    module('geometry.py', 'from typing import Iterable\nimport numpy as np\n'
           'from .types import Gate, normalize_quaternion, quaternion_multiply',
           symbols('gate_visualization.py', ('_normalise_quaternion', '_quaternion_multiply', '_rotate_vector'),
                   rename={'_rotate_vector': '_visual_rotate_vector'})
           +symbols('gate_visualization.py', ('_gate_parts',), rename={'_rotate_vector': '_visual_rotate_vector'})
           +symbols('gate_route.py', ('_rotate_vector',)))
    feature_names = [n.name for n in ast.parse((ROOT/'controller_architecture_study/fast_convex_preprocess.py').read_text()).body
                     if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
    module('features.py', 'import numpy as np\nfrom .types import RawDroneState, ReferencePoint\n'
           'VECTOR_FLIP = np.array([1., -1., -1.])\nQUATERNION_FLIP = np.array([1., 1., -1., -1.])',
           symbols('controller_architecture_study/fast_convex_preprocess.py', feature_names,
                   rename={'MPCReference': 'ReferencePoint'}))
    module('trajectory.py', 'from types import SimpleNamespace\nimport numpy as np\nfrom .features import FastReferenceWindow',
           symbols('controller_architecture_study/ps_migration/frozen_trajectory.py', ('from_layout',)))
    hold_names = ('HoldBounds', 'nominal_acceleration', '_barrier_coefficients', 'barrier_periods', 'GateHoldAnalyzer', 'safe_deadline')
    module('hold.py', 'from dataclasses import dataclass, asdict\nimport math\nimport numpy as np\n'
           'from .geometry import _gate_parts, _rotate_vector',
           symbols('controller_architecture_study/held_control_safety.py', hold_names))
    module('controls.py', 'from __future__ import annotations\nfrom dataclasses import dataclass\n'
           'from typing import Any, Mapping\nfrom pathlib import Path\nimport json\nimport numpy as np\n'
           "DEFAULT_CALIBRATION_PATH = Path(__file__).resolve().parents[1]/'config/thrust_calibration.json'",
           symbols('config.py', ('ThrustCalibration', 'load_calibration'))
           +symbols('control_adapter.py', ('AirSimCommand', 'validate_control', 'thrust_to_throttle',
                                         'throttle_to_thrust', 'control_to_airsim_command')))
    fixture_root = ROOT/'controller_architecture_study/results/ps_migration_handoff_20261116_002/route_b_ps_handoff_20261116/fixtures'
    (package/'verification').mkdir()
    for name in ('feature_command_parity.json', 'scheduler_parity.json'):
        shutil.copyfile(fixture_root/name, package/'verification'/name)
    (package/'tests').mkdir()
    remap = {'controller_architecture_study.ip37_obligation_runtime': 'ps_core.runtime',
             'controller_architecture_study.obligation_scheduler': 'ps_core.scheduler',
             'controller_architecture_study.requant18_profile': 'ps_core.profile',
             'controller_architecture_study.controller_backend': 'ps_core.backend_contract',
             'controller_architecture_study.ip37_shadow': 'ps_core.ip37_shadow'}
    for name in ('test_obligation_scheduler.py', 'test_ip37_obligation_runtime.py'):
        text = (ROOT/'tests'/name).read_text()
        for old, new in remap.items():
            text = text.replace(old, new)
        (package/'tests'/name).write_text(text, encoding='utf-8')
    shutil.copyfile(ROOT/'controller_architecture_study/ps_migration/test_frozen_trajectory.py', package/'tests/test_frozen_trajectory.py')
    (package/'config').mkdir()
    shutil.copyfile(ROOT/'configs/thrust_calibration.json', package/'config/thrust_calibration.json')
    shutil.copyfile(ROOT/'controller_architecture_study/results/release001_live_case_20261116/case.json', package/'config/case.json')
    shutil.copyfile(ROOT/'controller_architecture_study/ps_migration/verify_bundle.py', package/'verification/verify_bundle.py')
    shutil.copyfile(ROOT/'controller_architecture_study/ps_migration/check_ps_scope.py', package/'verification/check_ps_scope.py')
    parity = ROOT/'controller_architecture_study/results/ps_scope_20261116_001/original_parity.json'
    if parity.exists():
        shutil.copyfile(parity, package/'verification/original_parity.json')
    def save(name, value):
        (package/name).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    save('SOURCE_MAP.json', dict(scope='generated symbol-level extraction, not a PS executable', symbols=origins,
         adaptations=['local imports rewired', 'only materialized whole-job forecast supported',
                      'old native/ABC lifecycle methods removed', 'test import names rewired',
                      'configuration file resolved inside this package']))
    previous = ROOT/'controller_architecture_study/results/route_b_source_20261116_001/route_b_source_20261116/FILE_MAP.json'
    audit = []
    for item in json.loads(previous.read_text(encoding='utf-8'))['files']:
        kept = [s['symbol'] for s in origins if s['source'] == item['path']]
        verification_only = item['path'] in ('tests/test_obligation_scheduler.py',
            'tests/test_ip37_obligation_runtime.py', 'controller_architecture_study/ps_migration/verify_bundle.py')
        audit.append(dict(previous_path=item['path'],
                          action='selected_symbols_only' if kept else 'verification_only' if verification_only
                          else 'excluded_as_module', retained_symbols=kept))
    save('OLD_SOURCE_AUDIT.json', audit)
    save('DPU_BACKEND.json', dict(model_family='independent237', protocol_reference='IP37/1', tag='9253dd1fc87f8d7f',
         existing_runner='board_package_requant18_release001/pool237_runner.cc',
         integrate_symbols=['Pool::find(uint32_t)', 'Pool::infer(index, raw237, normalized=false)'],
         execution='actual local DNNDK DPU; whole inference is non-preemptive',
         missing=['local DPU adapter', 'remote AirSim RPC', 'real-time release/IO integration', 'ARM build and flight validation']))
    guide = ROOT/'controller_architecture_study/docs/PS_SCOPE_SOURCE_AUDIT_20261116.md'
    shutil.copyfile(guide, package/'README.md')
    hashes = {p.relative_to(package).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(package.rglob('*')) if p.is_file()}
    save('HANDOFF_MANIFEST.json', dict(scope='PS-relevant reference extraction', sha256=hashes))
    archive = output/(package.name+'.tar.gz')
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(package, arcname=package.name)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_name(archive.name+'.sha256').write_text(f'{digest}  {archive.name}\n')
    print(json.dumps(dict(archive=str(archive), core_modules=len(list(core.glob('*.py')))-1,
                         files=len(hashes)+1, bytes=archive.stat().st_size), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    build(parser.parse_args().output.resolve())
