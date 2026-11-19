"""Export current Route B source and small fixtures without binary/data payloads."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tarfile

from .package_ps_migration import DELIVERY, ROOT, TESTS, local_sources


CORE = ('obligation_scheduler.py', 'bottleneck_joint.py', 'ip37_obligation_runtime.py',
        'ip37_joint_runtime.py', 'controller_backend.py', 'requant18_profile.py', 'bottleneck_abc.py')
SUPPORT = ('controller_architecture_study/fly_ip37_joint.py',
           'controller_architecture_study/fast_convex_preprocess.py',
           'controller_architecture_study/held_control_safety.py',
           'controller_architecture_study/ip37_shadow.py', 'airsim_interface.py',
           'control_adapter.py', 'config.py', 'mpc_controller.py', 'state_adapter.py',
           'environment.py', 'gate_route.py', 'gate_visualization.py')


def build(output):
    if output.exists():
        raise ValueError('Use a new output directory')
    package = output/'route_b_source_20261116'
    project = package/'project'
    seeds = [Path('controller_architecture_study')/name for name in CORE]
    seeds += [Path(name) for name in SUPPORT]
    seeds += [Path('tests')/name for name in TESTS]
    seeds += [Path('controller_architecture_study')/name for name in
              ('replay_obligation_release.py', 'package_route_b_source.py',
               'ps_migration/check_fixtures.py', 'ps_migration/verify_bundle.py')]
    sources = local_sources(seeds)
    assets = [Path(p) for p in ('configs/thrust_calibration.json',
              'controller_architecture_study/configs/airsim_safety_12.json',
              'controller_architecture_study/results/release001_live_case_20261116/case.json',
              'controller_architecture_study/results/release001_framework_audit_20261116/inputs_for_ubuntu.json')]
    assets += [Path('controller_architecture_study/results/decision_convex_96k_v1')/p
               for p in ('manifest.json', 'routes/env48_route00.json',
                         'routes/env49_route02.json', 'routes/env50_route04.json')]
    docs = [Path('controller_architecture_study/docs')/name for name in
            ('Route-B源码指南_20261116.md', 'Route-B的Release001实测_20261116.md',
             'Route-B的Framework概览_20261115.md', 'Ubuntu-PS移植交接_20261116.md')]
    files = sources | set(assets) | set(docs)
    for path in sorted(files):
        target = project/path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/path, target)
    core = {Path('controller_architecture_study')/name for name in CORE}
    records = []
    for path in sorted(sources):
        category = ('core' if path in core else 'flight_support' if path.as_posix() in SUPPORT
                    else 'tests' if path.parts[0] == 'tests'
                    else 'export_and_checks' if path.name.startswith(('package_', 'check_', 'verify_'))
                    or 'ps_migration' in path.parts else 'import_compatibility')
        imports = set()
        package_name = '.'.join(path.parts[:-1])
        for node in ast.walk(ast.parse((ROOT/path).read_text(encoding='utf-8-sig'))):
            if isinstance(node, ast.Import):
                imports.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                base = (importlib.util.resolve_name('.'*node.level+(node.module or ''), package_name)
                        if node.level else node.module or '')
                imports.add(base)
        records.append(dict(path=path.as_posix(), category=category,
                            bytes=(ROOT/path).stat().st_size, imports=sorted(imports)))
    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    save(package/'FILE_MAP.json', dict(scope='static reference import closure, not a dead-code proof', files=records))
    fixture_source = ROOT/'controller_architecture_study/results/ps_migration_handoff_20261116_002/route_b_ps_handoff_20261116/fixtures'
    for name in ('feature_command_parity.json', 'scheduler_parity.json'):
        target = package/'fixtures'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(fixture_source/name, target)
    from .requant18_profile import profile_dict, read_profile
    modes, identity = read_profile(ROOT/DELIVERY)
    save(package/'fixtures/profile_snapshot.json', dict(identity=identity, modes=profile_dict(modes),
         scope='offline reference snapshot; not a substitute for original payload and CSV validation'))
    for name in ('dpu_model_registry.json', 'normalization.json', 'wire_identity.json'):
        destination = package/'asset_references'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/DELIVERY/'board_package_requant18_release001'/name, destination)
    save(package/'EXTERNAL_ASSETS.json', dict(identity=identity,
         normalization_sha256=json.loads((package/'asset_references/dpu_model_registry.json').read_text())['normalization_sha256'],
         original_delivery_relative_path=DELIVERY.as_posix(),
         required_delivery_entries=['board_package_requant18_release001 (complete original payload)',
                                    'release_001/freeze.json', 'results/release001_board480/summary.json',
                                    'results/release001_board480/requests.csv'],
         omitted=['DPU runner and ELF', 'full golden and validation data', 'original timing CSV',
                  'historical flight requests', 'AirSim C++ SDK', 'training datasets and weights'],
         acquisition='reuse the verified Ubuntu release001 delivery or the previous complete PS handoff'))
    readme = (ROOT/docs[0]).read_text(encoding='utf-8')
    readme = readme.replace('](Ubuntu-PS移植交接_20261116.md)',
                            '](project/controller_architecture_study/docs/Ubuntu-PS移植交接_20261116.md)')
    (package/'README.md').write_text(readme, encoding='utf-8')
    (package/'DEPENDENCIES.md').write_text(
        '# Dependency layers\n\n'
        '* Core scheduling and lifecycle tests: Python >=3.11 standard library only.\n'
        '* Feature/command/calendar fixtures: NumPy; no board, AirSim or Torch required.\n'
        '* Full legacy reference integration imports: NumPy, SciPy, Torch; teacher comparison uses OSQP 1.0.5.\n'
        '* Live reference flight: compatible AirSim RPC client, complete release001 delivery and actual DPU service.\n'
        '* PS C++ migration: existing DNNDK/sysroot/toolchain and matching AirSim RPC sources from the complete handoff.\n\n'
        'The core interface imports its IP37 inference backend only when infer() is called. '
        'This does not remove all legacy dependencies from the flight entry. '
        'FILE_MAP.json labels retained import compatibility files; they are not all active algorithms.\n', encoding='utf-8')
    hashes = {p.relative_to(package).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(package.rglob('*')) if p.is_file()}
    save(package/'HANDOFF_MANIFEST.json', dict(format='route_b_source_only_v1', date='2026-11-16',
         scope='source plus small fixtures; external assets required for flight', sha256=hashes))
    archive = output/(package.name+'.tar.gz')
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(package, arcname=package.name)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_name(archive.name+'.sha256').write_text(f'{digest}  {archive.name}\n', encoding='ascii')
    print(json.dumps(dict(archive=str(archive), bytes=archive.stat().st_size,
                         files=len(hashes), source_files=len(sources), core_files=len(core), sha256=digest), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    build(parser.parse_args().output.resolve())
