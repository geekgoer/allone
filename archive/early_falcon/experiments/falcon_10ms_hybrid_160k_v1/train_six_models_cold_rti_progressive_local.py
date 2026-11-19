"""Fresh local cold-RTI sweep with a fixed progressive epoch budget per model.

The output directory is deliberately distinct from every earlier 300/2000
experiment.  A repeated invocation resumes only this experiment's own
checkpoint, never an older run.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
POLICY = 'fully_initialized_acado_then_exactly_one_rti_step_v1'

# Parameter count ascending; the 8.4M tie uses shallower-before-deeper order.
RUNS = (
    (10, 512, 200),
    (22, 512, 300),
    (10, 1024, 400),
    (34, 512, 500),
    (22, 1024, 600),
    (34, 1024, 700),
)


def completed(path: Path, epochs: int) -> bool:
    try:
        metrics = json.loads((path / 'metrics.json').read_text(encoding='utf-8'))
        return bool(metrics.get('training_complete')) and int(metrics.get('epochs_completed', 0)) >= epochs
    except (FileNotFoundError, KeyError, ValueError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=HERE / 'dnn_dataset_v4_cold_rti_labels_v1')
    parser.add_argument('--output-root', type=Path, default=HERE / 'runs_cold_rti_progressive_200_to_700_batch512_seed0')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--cooldown-seconds', type=float, default=60.0)
    args = parser.parse_args()
    data_dir, output_root = args.data_dir.resolve(), args.output_root.resolve()
    manifest = json.loads((data_dir / 'split_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('teacher_policy') != POLICY or int(manifest.get('input_dimensions', 0)) != 12:
        raise ValueError('Requires the published 12-D cold-RTI dataset.')
    for split in ('train', 'validation', 'test'):
        if not (data_dir / f'{split}.npz').is_file():
            raise FileNotFoundError(f'Missing {split}.npz')
    if output_root == HERE / 'runs_cold_rti_2000_seed0':
        raise ValueError('Refusing to mix the progressive experiment with the older 2000-epoch run.')

    for depth, width, epochs in RUNS:
        label = f'seed_0/depth{depth}_width{width}'
        output = output_root / 'seed_0' / f'depth{depth}_width{width}'
        if completed(output, epochs):
            print(f'SKIP complete: {label}', flush=True)
            continue
        command = [
            sys.executable, str(ROOT / 'train_dnn_controller.py'),
            '--data-dir', str(data_dir), '--output-dir', str(output),
            '--depth', str(depth), '--width', str(width), '--epochs', str(epochs),
            '--batch-size', '512', '--learning-rate', '1e-4',
            '--gradient-clip-norm', '1.0', '--workers', '0', '--disable-pin-memory',
            '--selection-metric', 'normalized_mse', '--min-epochs', str(epochs),
            '--patience', '0', '--lr-plateau-patience', '0',
            '--max-validation-baseline-ratio', '0.25', '--log-every', '20',
            '--seed', '0', '--device', args.device, '--skip-curve',
        ]
        if (output / 'latest_checkpoint.pt').is_file():
            command.append('--resume')
        mode = 'RESUME' if '--resume' in command else 'START'
        print(f'{mode} {label}; fixed epochs={epochs}, batch=512, seed=0, LR=1e-4.', flush=True)
        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode:
            raise SystemExit(f'FAILED {label}; checkpoint retained. Re-run this same command to resume.')
        gc.collect()
        if args.cooldown_seconds > 0:
            print(f'Cooling down CUDA for {args.cooldown_seconds:.0f}s.', flush=True)
            time.sleep(args.cooldown_seconds)
    print(f'COMPLETE: {output_root}', flush=True)


if __name__ == '__main__':
    main()
