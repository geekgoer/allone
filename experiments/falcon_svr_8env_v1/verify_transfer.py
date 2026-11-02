"""Check the transferred files, optionally checking offline Python imports too."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', action='store_true', help='also check Python and imports; does not contact hardware')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'transfer_manifest.json').read_text(encoding='utf-8'))
    bad = []
    for name, digest in manifest['sha256'].items():
        path = ROOT / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            bad.append(name)
    if bad:
        raise SystemExit('Missing/changed files:\n' + '\n'.join(bad))
    print(f"PASS: {len(manifest['sha256'])} transferred files match SHA-256.")
    if args.runtime:
        if sys.version_info[:2] != (3, 11):
            raise SystemExit('Use CPython 3.11 x64 for this frozen dependency set.')
        sys.path.insert(0, str(ROOT / 'vendor'))
        import airsim
        import msgpackrpc
        import numpy as np
        import tornado
        print(dict(python=sys.version, numpy=np.__version__, airsim=airsim.__version__,
                   airsim_source=airsim.__file__, tornado=tornado.version))
        from collect import exp
        config, _, _ = exp.load_experiment(exp.HERE / 'collection_config.json')
        print(f"PASS: offline plan has {len(exp.make_plan(config))} trials; no board/AirSim connection was opened.")


if __name__ == '__main__':
    main()
