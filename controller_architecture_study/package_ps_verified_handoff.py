"""Build a layered migration delivery: clean kernels, references and evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

from .package_ps_scope import build as build_scope

ROOT = Path(__file__).resolve().parents[1]
DELIVERY = ROOT/'controller_architecture_study/docs/main18_requant_release001_delivery_20261115'
FLIGHTS = ('release001_obligation_fixed2_20261116_01',
           'release001_obligation_flying_fixed2_20261116_01',
           'release001_obligation_switch1_20261116_02',
           'release001_obligation_switch2_20261116_01',
           'release001_obligation_samecase_fixed2_20261116_01')


def build(output):
    if output.exists():
        raise ValueError('Use a new output directory')
    output.mkdir(parents=True)
    build_scope(output/'kernel_build')
    package = output/'route_b_ps_verified_20261116'
    shutil.copytree(output/'kernel_build/route_b_ps_scope_20261116', package/'source')
    package.mkdir(exist_ok=True)
    roles, origins = {}, {}
    def copy(path, relative, role):
        target = package/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        roles[str(Path(relative).as_posix())] = role
        origins[str(Path(relative).as_posix())] = str(path.relative_to(ROOT).as_posix())
    def tree(path, relative, role):
        for p in path.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:
                copy(p, Path(relative)/p.relative_to(path), role)
    deploy = DELIVERY/'board_package_requant18_release001'
    for name in ('dpu_model_registry.json', 'normalization.json', 'payload_manifest.json',
                 'SHA256SUMS.txt', 'wire_identity.json'):
        copy(deploy/name, 'reference_assets/deployment/'+name, 'reference_identity')
    for name in ('golden', 'validation'):
        tree(deploy/name, 'reference_assets/deployment/'+name, 'offline_numerical_reference')
    for name in ('release_001',):
        tree(DELIVERY/name, 'evidence/release001/'+name, 'release_freeze_provenance')
    for name in ('release001_board480', 'release001_board_goldens', 'release001_board_service'):
        tree(DELIVERY/'results'/name, 'evidence/release001/results/'+name, 'original_board_evidence')
    for name in ('environment_before.json', 'host_execution.json', 'board_all18_goldens.log',
                 'board_target_check.log', 'cross_compile_runner.log', 'board_udp_all18_480.log'):
        copy(DELIVERY/'execution_20261115_run001'/name, 'evidence/release001/execution/'+name, 'original_environment_evidence')
    for name in FLIGHTS:
        tree(ROOT/'controller_architecture_study/results'/name,
             'evidence/windows_reference/'+name, 'original_windows_baseline_not_runtime')
    copy(ROOT/'controller_architecture_study/results/release001_framework_audit_20261116/inputs_for_ubuntu.json',
         'verification_inputs/range_rejections.json', 'failure_reproduction_not_validation')
    for name in ('FINAL_REPORT_AND_WINDOWS_HANDOFF_20261115.md',):
        copy(DELIVERY/name, 'docs/'+name, 'historical_measurement_report')
    for name in ('PS_VERIFIED_HANDOFF_20261116.md', 'PS_SCOPE_SOURCE_AUDIT_20261116.md', 'UBUNTU_PS_MIGRATION_HANDOFF_20261116.md',
                 'ROUTE_B_LIVE_RELEASE001_20261116.md'):
        copy(ROOT/'controller_architecture_study/docs'/name, 'docs/'+name, 'migration_documentation')
    copy(ROOT/'controller_architecture_study/docs/PS_VERIFIED_HANDOFF_20261116.md', 'README_FIRST.md', 'authoritative_entry')
    copy(ROOT/'controller_architecture_study/ps_migration/audit_handoff_evidence.py',
         'tools/audit_evidence.py', 'offline_audit_tool_not_runtime')
    copy(ROOT/'controller_architecture_study/ps_migration/verify_bundle.py',
         'tools/verify_bundle.py', 'offline_audit_tool_not_runtime')
    registry = json.loads((deploy/'dpu_model_registry.json').read_text())
    finalized = {name.removeprefix('./'): digest for digest, name in
                 (line.split(maxsplit=1) for line in (deploy/'SHA256SUMS.txt').read_text().splitlines() if line.strip())}
    dependencies = dict(deployment=dict(name='board_package_requant18_release001',
        wire_tag=registry['wire_tag_hex'], normalization_sha256=registry['normalization_sha256'],
        expected_board_location='/home/root/board_package_requant18_release001',
        files=finalized, needed_for='local DPU adapter, golden replay and ARM linking'),
        sdk=dict(needed_for='C++ remote AirSim RPC client',
            existing_source='previous complete handoff: third_party_reference/AirSim',
            note='use verified existing sources/toolchain; no Windows binaries in controller'),
        runtime=dict(reference_python='>=3.11', numeric_dependency='NumPy',
            production_port='C++/DNNDK plus RPC and board event loops, still to implement'))
    (package/'EXTERNAL_DEPENDENCIES.json').write_text(json.dumps(dependencies, indent=2)+'\n', encoding='utf-8')
    checklist = dict(scope='completeness audit of reference kernels and migration materials',
        ready=['numeric state/reference/control contracts', 'frozen trajectory loader', 'held-control budget',
               'finite whole-task scheduler and lifecycle', 'verified deployment/profile loader',
               'golden/reference data and original cost CSV', 'representative success/failure raw logs'],
        not_implemented=['PS local DPU adapter', 'ARM AirSim RPC adapter',
                         'PS release/IO/actuation/event orchestration', 'board build and end-to-end validation'],
        not_proven=['strict WCET', 'state-dependent controller recovery qualification',
                    'new-INT8 skip/recovery physical qualification', 'dynamic advantage over best static J baseline'])
    (package/'COMPLETENESS.json').write_text(json.dumps(checklist, indent=2)+'\n', encoding='utf-8')
    for p in (package/'source').rglob('*'):
        if p.is_file():
            roles[p.relative_to(package).as_posix()] = 'ps_reference_runtime' if 'ps_core' in p.parts else 'source_configuration_or_verification'
    (package/'FILE_ROLES.json').write_text(json.dumps(roles, indent=2)+'\n', encoding='utf-8')
    (package/'ASSET_ORIGINS.json').write_text(json.dumps(origins, indent=2)+'\n', encoding='utf-8')
    hashes = {p.relative_to(package).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(package.rglob('*')) if p.is_file()}
    (package/'HANDOFF_MANIFEST.json').write_text(json.dumps(dict(format='verified_layered_ps_handoff_v1',
         scope='reference and evidence complete as enumerated; PS application still pending', sha256=hashes), indent=2)+'\n', encoding='utf-8')
    archive = output/(package.name+'.tar.gz')
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(package, arcname=package.name)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_name(archive.name+'.sha256').write_text(f'{digest}  {archive.name}\n', encoding='ascii')
    print(json.dumps(dict(archive=str(archive), bytes=archive.stat().st_size, files=len(hashes)+1, sha256=digest), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    build(parser.parse_args().output.resolve())
