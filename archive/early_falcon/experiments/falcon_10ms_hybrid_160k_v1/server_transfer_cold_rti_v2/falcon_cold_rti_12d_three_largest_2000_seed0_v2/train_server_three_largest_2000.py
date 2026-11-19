"""Linux runner: train the three largest 12-D cold-RTI models for exactly 2000 epochs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] if (HERE.parents[1] / 'train_dnn_controller.py').is_file() else HERE
ARCHITECTURES = ((34, 1024), (22, 1024), (34, 512))
POLICY = 'fully_initialized_acado_then_exactly_one_rti_step_v1'


def complete(path: Path) -> bool:
    try:
        return bool(json.loads((path / 'metrics.json').read_text(encoding='utf-8'))['training_complete'])
    except (FileNotFoundError, KeyError, ValueError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=HERE / 'dnn_dataset_v4_cold_rti_labels_v1')
    parser.add_argument('--output-root', type=Path, default=HERE / 'runs_server_three_largest_2000_seed0')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.seed != 0:
        raise ValueError('This transfer experiment is fixed to seed=0 for cross-model comparability.')
    data, output = args.data_dir.resolve(), args.output_root.resolve()
    manifest = json.loads((data / 'split_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('teacher_policy') != POLICY or int(manifest.get('input_dimensions', 0)) != 12:
        raise ValueError('Requires the 12-D cold-RTI dataset created on Windows.')
    for split in ('train', 'validation', 'test'):
        if not (data / f'{split}.npz').is_file():
            raise FileNotFoundError(f'Missing {split}.npz')
    for depth, width in ARCHITECTURES:
        label = f'seed_{args.seed}/depth{depth}_width{width}'
        run = output / f'seed_{args.seed}' / f'depth{depth}_width{width}'
        if complete(run):
            print(f'SKIP complete: {label}', flush=True)
            continue
        command = [sys.executable, str(ROOT / 'train_dnn_controller.py'), '--data-dir', str(data), '--output-dir', str(run),
            '--depth', str(depth), '--width', str(width), '--epochs', '2000', '--batch-size', str(args.batch_size),
            '--learning-rate', '1e-4', '--gradient-clip-norm', '1.0', '--workers', str(args.workers),
            '--selection-metric', 'normalized_mse', '--min-epochs', '2000', '--patience', '0', '--lr-plateau-patience', '0',
            '--max-validation-baseline-ratio', '0.25', '--log-every', '20', '--seed', str(args.seed), '--device', args.device,
            '--skip-curve']
        if (run / 'latest_checkpoint.pt').exists():
            command.append('--resume')
        print(f"{'RESUME' if '--resume' in command else 'START'} {label}; fixed seed=0, 2000 epochs, LR=1e-4.", flush=True)
        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode:
            raise SystemExit(f'FAILED {label}; checkpoint preserved: {run / "latest_checkpoint.pt"}')
        time.sleep(10)
    print(f'COMPLETE: {output}', flush=True)


if __name__ == '__main__':
    main()
