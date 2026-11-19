"""Package existing 237D artifacts without altering their bytes or source paths."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT/'controller_architecture_study/results'
OUT = RESULTS/'gain7_handoff_20261109_v1'


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    groups = [RESULTS/'decision_convex_recovery_scenic_falcon18_v1/runs',
              RESULTS/'decision_convex_residual4_v1/runs']
    model_dirs = [p for group in groups for p in sorted(group.iterdir()) if p.is_dir()]
    old = RESULTS/'decision_convex_recovery_v1/runs/features_recovery'
    model_dirs += [p for p in sorted(old.iterdir()) if p.is_dir() and p.name.endswith('_seed0')]
    model_dirs += [RESULTS/'decision_convex_residual33_long5000_v1/residual_mlp_h33_w1024_seed0',
                   RESULTS/'incremental_supernet_237d_v4_grid5000']
    if len(model_dirs) != 28 or any(not (p/'best_model.pt').is_file() for p in model_dirs):
        raise RuntimeError('expected 28 existing model checkpoints; refusing incomplete package')
    files = [ROOT/'controller_architecture_study/docs/237D增益7微调交接_20261109.md']
    for directory in model_dirs:
        files += [directory/name for name in ('best_model.pt', 'config.json', 'completion.json')
                  if (directory/name).is_file()]
    dataset = RESULTS/'decision_convex_recovery_v1/datasets/features_recovery'
    files += [dataset/name for name in ('train.npz', 'validation.npz', 'normalization.json', 'split_manifest.json')]
    for run in ('decision_convex_96k_v1', 'decision_convex_recovery_v1'):
        files += [RESULTS/run/name for name in ('manifest.json', 'data_audit.json') if (RESULTS/run/name).is_file()]
    base = RESULTS/'decision_convex_96k_v1/dnn_dataset'
    files += [base/name for name in ('normalization.json', 'split_manifest.json')]
    if any(not p.is_file() for p in files):
        raise RuntimeError('required input missing')
    archive = OUT/'237D_gain7_handoff.zip'
    manifest = []
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        for i, path in enumerate(files, 1):
            rel = path.relative_to(ROOT).as_posix()
            digest = hashlib.sha256()
            with path.open('rb') as source, z.open(rel, 'w', force_zip64=True) as dest:
                while chunk := source.read(4*1024*1024):
                    digest.update(chunk)
                    dest.write(chunk)
            manifest.append(dict(path=rel, bytes=path.stat().st_size, sha256=digest.hexdigest()))
            print(f'Packaged {i}/{len(files)} {rel}', flush=True)
        z.writestr('MANIFEST_SHA256.json', json.dumps({'models': 28, 'files': manifest}, indent=2))
    print('Verifying every archived file against manifest...', flush=True)
    with zipfile.ZipFile(archive) as z:
        for item in manifest:
            digest = hashlib.sha256()
            with z.open(item['path']) as source:
                while chunk := source.read(4*1024*1024):
                    digest.update(chunk)
            if digest.hexdigest() != item['sha256']:
                raise RuntimeError(f"archive hash mismatch: {item['path']}")
    digest = hashlib.sha256()
    with archive.open('rb') as source:
        while chunk := source.read(4*1024*1024):
            digest.update(chunk)
    (OUT/'SHA256.txt').write_text(f'{digest.hexdigest()}  {archive.name}\n', encoding='ascii')
    (OUT/'MANIFEST_SHA256.json').write_text(json.dumps({'models': 28, 'files': manifest}, indent=2)+'\n', encoding='utf-8')
    print(f'PASS {archive} bytes={archive.stat().st_size} sha256={digest.hexdigest()}', flush=True)


if __name__ == '__main__':
    main()
