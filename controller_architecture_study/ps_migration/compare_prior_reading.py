"""Identify previously read original files that still match this delivery."""
import argparse
import hashlib
import json
from pathlib import Path


def compare(root, old):
    origins = json.loads((root/'ASSET_ORIGINS.json').read_text())
    records = []
    seen = set()
    for row in origins.values():
        relative = row['source']
        # The old package contains project/<repository-relative-path>.
        if relative.startswith('controller_architecture_study/results/') or ':' in relative or relative in seen:
            continue
        seen.add(relative)
        paths = [old/relative, old/'project'/relative]
        found = next((p for p in paths if p.is_file()), None)
        records.append(dict(source=relative, status='NOT_IN_OLD' if found is None else
            'UNCHANGED_REUSE_READING' if hashlib.sha256(found.read_bytes()).hexdigest() == row['source_sha256']
            else 'CHANGED_REVIEW_DIFF'))
    # Symbol extraction maps preserve the original functions and hashes even after relocating modules.
    mapping = json.loads((root/'MIGRATION_MAP.json').read_text())
    for row in mapping['symbol_sources']:
        if row['source'] in seen:
            continue
        seen.add(row['source'])
        found = next((p for p in (old/row['source'], old/'project'/row['source']) if p.is_file()), None)
        records.append(dict(source=row['source'], status='NOT_IN_OLD' if found is None else
            'UNCHANGED_REUSE_READING' if hashlib.sha256(found.read_bytes()).hexdigest() == row['source_sha256']
            else 'CHANGED_REVIEW_DIFF'))
    print(json.dumps(dict(scope='file identity only; new integration still needs review', files=records),
                     indent=2, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--old-root', type=Path, required=True)
    args = parser.parse_args()
    compare(args.root.resolve(), args.old_root.resolve())
