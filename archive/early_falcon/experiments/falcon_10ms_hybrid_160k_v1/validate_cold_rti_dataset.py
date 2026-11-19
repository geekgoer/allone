"""Validate that cold-RTI relabeling changed targets only, not v4 split membership."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from relabel_v4_cold_rti import OUTPUT, SOURCE, read_npz

SPLITS = ('train', 'validation', 'test')
POLICY = 'fully_initialized_acado_then_exactly_one_rti_step_v1'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=OUTPUT)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    manifest = json.loads((data_dir / 'split_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('teacher_policy') != POLICY:
        raise ValueError('Not a published cold-one-step dataset.')
    report = dict(status='valid', data_dir=str(data_dir), splits={})
    for split in SPLITS:
        original, relabeled = read_npz(SOURCE / f'{split}.npz'), read_npz(data_dir / f'{split}.npz')
        for key in original:
            if key == 'U':
                continue
            if key not in relabeled or not np.array_equal(original[key], relabeled[key]):
                raise ValueError(f'{split}: state/split metadata changed for {key}')
        if not np.array_equal(original['U'], relabeled['U_original']):
            raise ValueError(f'{split}: U_original does not equal v4 target')
        if relabeled['X'].shape != (len(original['X']), 12) or relabeled['U'].shape != (len(original['U']), 4):
            raise ValueError(f'{split}: invalid X/U shapes')
        if not np.isfinite(relabeled['U']).all():
            raise ValueError(f'{split}: non-finite labels')
        report['splits'][split] = dict(samples=len(relabeled['U']), environments=np.unique(relabeled['environment_id']).tolist(),
            changed_target_mse=float(np.mean((relabeled['U'] - original['U']) ** 2)),
            kkt_p99=float(np.quantile(relabeled['cold_rti_kkt'], .99)))
    normal = json.loads((data_dir / 'normalization.json').read_text(encoding='utf-8'))
    if len(normal['x_mean']) != 12 or len(normal['u_mean']) != 4:
        raise ValueError('Invalid normalization dimensions')
    print(json.dumps(report, indent=2) if args.json else report)


if __name__ == '__main__':
    main()
