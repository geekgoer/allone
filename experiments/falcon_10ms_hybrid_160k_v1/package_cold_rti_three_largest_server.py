"""Create a self-contained upload ZIP for the three largest cold-RTI models."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PACKAGE = 'falcon_cold_rti_12d_three_largest_2000_seed0_v2'
DATASET = 'dnn_dataset_v4_cold_rti_labels_v1'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=HERE / 'server_transfer_cold_rti')
    args = parser.parse_args()
    data = HERE / DATASET
    required = ('train.npz', 'validation.npz', 'test.npz', 'normalization.json', 'split_manifest.json')
    missing = [name for name in required if not (data / name).is_file()]
    if missing:
        raise FileNotFoundError(f'Cold-RTI dataset is incomplete: {missing}')
    root = args.output_root.resolve()
    folder, archive = root / PACKAGE, root / f'{PACKAGE}.zip'
    if folder.exists() or archive.exists():
        raise FileExistsError(f'Package already exists: {folder}. Use a new --output-root; no files were replaced.')
    for name in ('train_dnn_controller.py', 'dnn_model_utils.py'):
        copy(ROOT / name, folder / name)
    copy(HERE / 'train_server_three_largest_2000.py', folder / 'train_server_three_largest_2000.py')
    (folder / 'requirements.txt').write_text(
        '--extra-index-url https://download.pytorch.org/whl/cu121\n'
        'numpy==1.26.4\n'
        'torch==2.4.1+cu121\n', encoding='utf-8')
    for name in required:
        copy(data / name, folder / DATASET / name)
    (folder / 'README.md').write_text(
        '# FALCON cold-RTI 12-D: three-largest-model server package\n\n'
        'Order: 34x1024, 22x1024, 34x512, strictly descending by parameter count. All runs are fixed to seed 0, train exactly 2000 epochs, use fixed LR 1e-4, and have no early stopping or LR scheduler.\n\n'
        '```bash\n'
        'python3 -m venv .venv && source .venv/bin/activate\n'
        'python -m pip install --upgrade pip\n'
        'python -m pip install -r requirements.txt\n'
        'sha256sum -c SHA256SUMS.txt\n'
        'python train_server_three_largest_2000.py --device cuda --batch-size 512 --workers 2\n'
        '```\n\n'
        'If the server runs out of GPU memory, use `--batch-size 256`. Re-run the same command after interruption; each model resumes from `latest_checkpoint.pt`.\n', encoding='utf-8')
    files = sorted(path for path in folder.rglob('*') if path.is_file())
    (folder / 'SHA256SUMS.txt').write_text(''.join(f'{sha256(path)}  {path.relative_to(folder).as_posix()}\n' for path in files), encoding='utf-8')
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_STORED) as z:
        for path in sorted(path for path in folder.rglob('*') if path.is_file()):
            z.write(path, Path(PACKAGE) / path.relative_to(folder))
    print(f'Upload: {archive}')
    print(f'Bytes: {archive.stat().st_size}')


if __name__ == '__main__':
    main()
