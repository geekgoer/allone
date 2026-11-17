#!/usr/bin/env python3
"""Checksum immutable handoffs; read no training code semantically and execute none."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def check(root):
    manifest = root/'HANDOFF_MANIFEST.json'
    listing = json.loads(manifest.read_text(encoding='utf-8'))['sha256']
    failures = []
    for name, expected in listing.items():
        path = (root/name).resolve()
        if root.resolve() not in path.parents or not path.is_file() or sha(path) != expected:
            failures.append(name)
    return dict(root=str(root.resolve()), files=len(listing), failures=failures,
                status='PASS' if not failures else 'FAIL', manifest_sha256=sha(manifest))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--unified', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    pairs = [
        ('project/controller_architecture_study/docs/main18_requant_release001_delivery_20261115/board_package_requant18_release001/pool237_runner.cc',
         'reference_assets/deployment/pool237_runner.cc'),
        ('project/controller_architecture_study/results/release001_live_case_20261116/case.json', 'source/config/case.json'),
        ('project/configs/thrust_calibration.json', 'source/config/thrust_calibration.json'),
        ('fixtures/feature_command_parity.json', 'source/verification/feature_command_parity.json'),
        ('fixtures/scheduler_parity.json', 'source/verification/scheduler_parity.json')]
    checks = [check(args.original), check(args.unified)]
    matches = [dict(original=a, unified=b, sha256=sha(args.original/a),
                    identical=sha(args.original/a)==sha(args.unified/b)) for a,b in pairs]
    result = dict(schema='ps-migration-source-proof/v1', handoffs=checks,
                  identical_selected_sources=matches,
                  original_archive_present=args.original.with_suffix('.tar.gz').is_file(),
                  status='PASS' if all(c['status']=='PASS' for c in checks) and all(c['identical'] for c in matches) else 'FAIL')
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print('SOURCE_PROOF {} files={}+{} paired_identical={}'.format(result['status'],checks[0]['files'],checks[1]['files'],len(matches)))
    raise SystemExit(0 if result['status']=='PASS' else 1)


if __name__ == '__main__':
    main()
