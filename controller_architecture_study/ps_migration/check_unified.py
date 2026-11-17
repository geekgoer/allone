"""Run all packaged offline acceptance checks without the previous project."""
import argparse
import json
from pathlib import Path
import os
import subprocess
import sys


def run(root):
    root = root.resolve()
    env = dict(os.environ, PYTHONPATH=str(root/'source'), PYTHONDONTWRITEBYTECODE='1',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    checks = [
        ('hashes', root, [str(root/'tools/verify_bundle.py'), '--root', str(root)]),
        ('python_units', root/'source', ['-m', 'unittest', 'discover', '-s', 'tests']),
        ('numeric_fixtures', root/'source', ['-m', 'verification.check_ps_scope']),
        ('evidence_and_full_deployment', root, [str(root/'tools/audit_evidence.py'), '--root', str(root),
                                              '--deployment', str(root/'reference_assets/deployment')]),
        ('cpp_python_contracts', root, [str(root/'tools/check_unified_contracts.py'), '--root', str(root)]),
    ]
    results = []
    for name, cwd, args in checks:
        value = subprocess.run([sys.executable, '-B', *args], cwd=cwd, env=env, capture_output=True, text=True)
        print(f'{name}: {"PASS" if value.returncode == 0 else "FAIL"}', flush=True)
        print(value.stdout+value.stderr, flush=True)
        results.append(dict(name=name, returncode=value.returncode))
        if value.returncode:
            raise SystemExit(value.returncode)
    print(json.dumps(dict(checks=results, scope='offline acceptance; not a new DPU or AirSim test'), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    run(parser.parse_args().root)
