"""Build a self-contained reference handoff; does not implement the PS port."""
import argparse
import ast
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tarfile


ROOT = Path(__file__).resolve().parents[1]
DELIVERY = Path('controller_architecture_study/docs/main18_requant_release001_delivery_20261115')
TESTS = ('test_ip37_obligation_runtime.py', 'test_ip37_joint.py', 'test_obligation_scheduler.py',
         'test_bottleneck_joint.py', 'test_remaining_chain.py', 'test_fast_convex_preprocess.py',
         'test_held_control_safety.py', 'test_held_control_batch.py', 'test_framework_held_control_age.py')


def local_sources(seeds):
    pending, seen = list(seeds), set()
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        package = '.'.join(path.parts[:-1])
        for node in ast.walk(ast.parse((ROOT/path).read_text(encoding='utf-8-sig'))):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = (importlib.util.resolve_name('.'*node.level+(node.module or ''), package)
                        if node.level else node.module or '')
                names = [base]+[base+'.'+alias.name for alias in node.names]
            for name in names:
                base = Path(*name.split('.'))
                candidate = next((p for p in (base.with_suffix('.py'), base/'__init__.py')
                                  if (ROOT/p).is_file()), None)
                if candidate is not None:
                    pending.append(candidate)
        for parent in path.parents:
            init = parent/'__init__.py'
            if (ROOT/init).is_file():
                pending.append(init)
    return seen


def build(output, airsim_source):
    if output.exists():
        raise ValueError('Use a new output directory')
    output.mkdir(parents=True)
    package = output/'route_b_ps_handoff_20261116'
    project = package/'project'
    seeds = [Path('controller_architecture_study')/name for name in
             ('fly_ip37_joint.py', 'replay_obligation_release.py', 'controller_backend.py',
              'package_ps_migration.py', 'ps_migration/verify_bundle.py', 'ps_migration/check_fixtures.py')]
    seeds += [Path('tests')/name for name in TESTS]
    files = local_sources(seeds)
    for name in ('AGENTS.md', 'README.md', 'docs/工程Framework.md', 'docs/协作进度.md',
                 'configs/thrust_calibration.json',
                 'controller_architecture_study/configs/airsim_safety_12.json',
                 'controller_architecture_study/run_release001_framework.ps1',
                 'controller_architecture_study/docs/研究文档索引.md',
                 'controller_architecture_study/docs/Route-B代码导览_20261109.md',
                 'controller_architecture_study/docs/Route-B的Framework概览_20261115.md',
                 'controller_architecture_study/docs/Route-B论文草稿_20261115.md',
                 'controller_architecture_study/docs/Route-B的Release001实现_20261116.md',
                 'controller_architecture_study/docs/Route-B的Release001实测_20261116.md',
                 'controller_architecture_study/docs/Ubuntu-PS移植交接_20261116.md'):
        files.add(Path(name))
    # Copy the deployment untouched, including ELF and every payload-manifest file.
    files.update(p.relative_to(ROOT) for p in (ROOT/DELIVERY/'board_package_requant18_release001').rglob('*') if p.is_file())
    files.update(p.relative_to(ROOT) for p in (ROOT/DELIVERY/'release_001').rglob('*') if p.is_file())
    for name in ('results/release001_board480/summary.json', 'results/release001_board480/requests.csv',
                 'FINAL_REPORT_AND_WINDOWS_HANDOFF_20261115.md', 'MAIN18_RELEASE_RUNBOOK_20261115.md'):
        files.add(DELIVERY/name)
    evidence = ('release001_live_case_20261116', 'release001_framework_audit_20261116',
                'release001_obligation_switch1_20261116_02', 'release001_obligation_switch2_20261116_01',
                'release001_obligation_samecase_fixed2_20261116_01')
    for name in evidence:
        folder = ROOT/'controller_architecture_study/results'/name
        files.update(p.relative_to(ROOT) for p in folder.glob('*.json'))
    route_base = Path('controller_architecture_study/results/decision_convex_96k_v1')
    files.add(route_base/'manifest.json')
    files.update(route_base/'routes'/name for name in
                 ('env48_route00.json', 'env49_route02.json', 'env50_route04.json'))
    for path in sorted(files):
        target = project/path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/path, target)
    shutil.copyfile(project/'controller_architecture_study/docs/Ubuntu-PS移植交接_20261116.md',
                    package/'UBUNTU_AGENT_TASK.md')
    (package/'README_FIRST.md').write_text(
        '# Ubuntu / Zynq PS handoff\n\nRead UBUNTU_AGENT_TASK.md first. '
        'This is reference source, deployment and evidence, NOT an implemented PS controller.\n\n'
        'Verify: `python3 project/controller_architecture_study/ps_migration/verify_bundle.py`\n', encoding='utf-8')
    sdk = package/'third_party_reference/AirSim'
    source_dirs = ('AirLib/include', 'AirLib/src', 'AirLib/deps/eigen3',
                   'AirLib/deps/rpclib/include', 'MavLinkCom/include', 'MavLinkCom/src',
                   'MavLinkCom/common_utils', 'MavLinkCom/mavlink',
                   'external/rpclib/rpclib-2.3.0/include',
                   'external/rpclib/rpclib-2.3.0/lib',
                   'external/rpclib/rpclib-2.3.0/dependencies',
                   'external/rpclib/rpclib-2.3.0/cmake',
                   'cmake/AirLib', 'cmake/MavLinkCom', 'cmake/cmake-modules')
    sdk_files = [p for name in source_dirs for p in (airsim_source/name).rglob('*')
                 if p.is_file() and not set(p.parts) & {'__pycache__', 'build', 'temp'}
                 and p.suffix.lower() not in ('.lib', '.a', '.so', '.dll', '.exe', '.obj', '.pdb', '.pyc')]
    sdk_files += [airsim_source/name for name in
                  ('LICENSE', 'external/rpclib/rpclib-2.3.0/CMakeLists.txt',
                   'external/rpclib/rpclib-2.3.0/LICENSE.md',
                   'external/rpclib/rpclib-2.3.0/rpclib.pc.in')]
    for path in sdk_files:
        target = sdk/path.relative_to(airsim_source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    head = (airsim_source/'.git/HEAD').read_text().strip()
    if head.startswith('ref: '):
        ref = airsim_source/'.git'/head.removeprefix('ref: ')
        head = ref.read_text().strip() if ref.is_file() else head
    (sdk/'SNAPSHOT_NOTE.md').write_text(
        '# AirSim client build reference\n\n'
        'Selected source files, not a full Unreal checkout or a verified ARM build. '
        'Use the board toolchain/sysroot to build the required RPC client.\n'
        f'Local tree HEAD reference: {head}\n'
        'Working tree cleanliness was not established; per-file hashes are authoritative.\n', encoding='utf-8')
    from .ps_migration.check_fixtures import make_features
    from .requant18_profile import read_profile
    from .replay_obligation_release import case
    from .obligation_scheduler import plan
    fixtures = package/'fixtures'
    fixtures.mkdir()
    def save(name, value):
        (fixtures/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    save('feature_command_parity.json', make_features(project))
    modes, identity = read_profile(project/DELIVERY)
    calendars = []
    for seed in range(6):
        calendar, permissions = case(seed, 4, modes)
        result = plan(calendar, {m.numeric_id: m for m in modes}, now=0., decision_ms=3., permissions=permissions)
        expected = {key: result.get(key) for key in ('accepted', 'action', 'order', 'choice', 'skipped', 'checks')}
        calendars.append(dict(seed=seed, calendar=asdict(calendar), permissions=[asdict(p) for p in permissions], expected=expected))
    save('scheduler_parity.json', dict(scope='synthetic finite scenarios using frozen empirical costs',
         decision_ms=3., identity=identity, modes=[asdict(m) for m in modes], cases=calendars))
    git_head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    hashes = {p.relative_to(package).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(package.rglob('*')) if p.is_file()}
    manifest = dict(format='route_b_ps_reference_handoff_v1', date='2026-11-16',
                    git_head_for_context=git_head, working_tree_snapshot=True,
                    scope='reference only; PS port not implemented', source_files=len(files),
                    sha256=hashes)
    (package/'HANDOFF_MANIFEST.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
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
    print(json.dumps(dict(archive=str(archive), bytes=archive.stat().st_size,
                         sha256=digest, files=len(hashes)), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--airsim-source', type=Path, default=Path('F:/Airsim/AirSim'))
    args = parser.parse_args()
    build(args.output.resolve(), args.airsim_source.resolve())
