"""Serial, resumable six-model cold-RTI ablation: exactly 2000 epochs/model."""
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
ARCHITECTURES = ((10, 512), (34, 1024), (10, 1024), (22, 512), (22, 1024), (34, 512))
POLICY = 'fully_initialized_acado_then_exactly_one_rti_step_v1'


def complete(path: Path) -> bool:
    try:
        return bool(json.loads((path / 'metrics.json').read_text(encoding='utf-8'))['training_complete'])
    except (FileNotFoundError, KeyError, ValueError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=HERE / 'dnn_dataset_v4_cold_rti_labels_v1')
    parser.add_argument('--output-root', type=Path, default=HERE / 'runs_cold_rti_2000_seed0')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cooldown-seconds', type=float, default=60.0)
    args = parser.parse_args()
    data, output = args.data_dir.resolve(), args.output_root.resolve()
    manifest = json.loads((data / 'split_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('teacher_policy') != POLICY or int(manifest.get('input_dimensions', 0)) != 12:
        raise ValueError('This runner requires the published cold-one-step 12-D dataset.')
    for split in ('train', 'validation', 'test'):
        if not (data / f'{split}.npz').is_file():
            raise FileNotFoundError(f'Missing {split}.npz')
    for depth, width in ARCHITECTURES:
        label, run = f'seed_{args.seed}/depth{depth}_width{width}', output / f'seed_{args.seed}' / f'depth{depth}_width{width}'
        if complete(run):
            print(f'SKIP complete: {label}', flush=True)
            continue
        command = [sys.executable, str(ROOT / 'train_dnn_controller.py'), '--data-dir', str(data), '--output-dir', str(run),
            '--depth', str(depth), '--width', str(width), '--epochs', '2000', '--batch-size', str(args.batch_size),
            '--learning-rate', '1e-4', '--gradient-clip-norm', '1.0', '--workers', '0', '--disable-pin-memory',
            '--selection-metric', 'normalized_mse', '--min-epochs', '2000', '--patience', '0', '--lr-plateau-patience', '0',
            '--max-validation-baseline-ratio', '0.25', '--log-every', '20', '--seed', str(args.seed), '--device', args.device,
            '--skip-curve']
        if (run / 'latest_checkpoint.pt').exists():
            command.append('--resume')
        print(f"{'RESUME' if '--resume' in command else 'START'} {label}; fixed 2000 epochs, fixed LR 1e-4.", flush=True)
        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode:
            raise SystemExit(f'FAILED {label}; checkpoint retained at {run}. Re-run the same command to resume.')
        gc.collect()
        if args.cooldown_seconds:
            print(f'Cooling down CUDA for {args.cooldown_seconds:.0f}s.', flush=True)
            time.sleep(args.cooldown_seconds)
    print(f'COMPLETE: {output}', flush=True)


if __name__ == '__main__':
    main()
