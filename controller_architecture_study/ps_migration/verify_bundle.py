"""Verify a PS migration handoff without network, AirSim or third-party modules."""
import hashlib
import json
from pathlib import Path


def verify(root):
    root = Path(root).resolve()
    manifest = json.loads((root/'HANDOFF_MANIFEST.json').read_text(encoding='utf-8'))
    for name, digest in manifest['sha256'].items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f'Missing or invalid package path: {name}')
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Hash mismatch: {name}')
    print(f"HANDOFF_HASH_PASS files={len(manifest['sha256'])}")
    return manifest


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = next((p for p in Path(__file__).resolve().parents
                         if (p/'HANDOFF_MANIFEST.json').is_file()), Path.cwd())
    parser.add_argument('--root', type=Path, default=default_root)
    verify(parser.parse_args().root)
