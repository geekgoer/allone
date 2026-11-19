"""Read-only CPU diagnosis of a saved 300-D model; never evaluates test."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dnn_model_utils import build_model, load_normalization


def metrics(errors, std):
    squared = errors.astype(np.float64) ** 2
    row = squared.mean(axis=1)
    order = np.sort(row)[::-1]
    return {
        "n": len(errors), "normalized_mse": float(row.mean()),
        "normalized_mse_per_output": squared.mean(axis=0).tolist(),
        "physical_mse_per_output": (squared * std ** 2).mean(axis=0).tolist(),
        "row_mse_quantiles": np.quantile(row, [.5, .9, .95, .99, .999, 1]).tolist(),
        "top_1pct_sse_share": float(order[:max(1, int(len(order)*.01))].sum() / max(row.sum(), 1e-30)),
    }


def main():
    import torch
    ap = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parent
    ap.add_argument('--data', type=Path, default=base/'dnn_dataset_300d_140k_v1_within_environment')
    ap.add_argument('--checkpoint', type=Path, default=base/'runs_pilot/seed_0/depth10_width512/best_model.pt')
    ap.add_argument('--output', type=Path, default=base/'audit_generalization')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    norm = load_normalization(args.data/'normalization.json', expected_input_dim=300)
    norm_diff = {k: float(np.max(np.abs(v - np.asarray(checkpoint['normalization'][k], dtype=np.float32)))) for k,v in norm.items()}
    model = build_model(torch, depth=checkpoint['depth'], width=checkpoint['width'], input_dim=300).eval()
    model.load_state_dict(checkpoint['model_state'])
    report = {'checkpoint': str(args.checkpoint), 'epoch': checkpoint['epoch'],
              'checkpoint_normalization_diff': norm_diff, 'split_metrics': {}, 'subgroups': {},
              'normalization_output_std': norm['u_std'].tolist()}
    data = {}
    for name in ('train', 'validation'):
        with np.load(args.data/f'{name}.npz', allow_pickle=False) as z:
            data[name] = {k:z[k] for k in z.files}
        d = data[name]
        x = ((d['X'] - norm['x_mean']) / norm['x_std']).astype(np.float32)
        y = ((d['U'] - norm['u_mean']) / norm['u_std']).astype(np.float32)
        pred = []
        with torch.inference_mode():
            for start in range(0, len(x), 512):
                pred.append(model(torch.from_numpy(x[start:start+512])).numpy())
        pred = np.concatenate(pred)
        err = pred-y
        report['split_metrics'][name] = metrics(err, norm['u_std'])
        report['subgroups'][name] = {}
        for column in ('source_kind', 'sample_family', 'control_bucket', 'environment_id', 'target_gate_id', 'teacher_iterations'):
            report['subgroups'][name][column] = {
                str(int(value)): metrics(err[d[column]==value], norm['u_std']) for value in np.unique(d[column])
            }
        loss = np.mean(err.astype(np.float64)**2, axis=1)
        worst = np.argsort(loss)[-100:][::-1]
        np.savez_compressed(args.output/f'{name}_errors.npz', error_normalized=err,
                            sample_id=d['sample_id'], worst_indices=worst)
        report['subgroups'][name]['worst_10'] = [dict(
            index=int(i), sample_id=int(d['sample_id'][i]), source=int(d['source_kind'][i]),
            family=int(d['sample_family'][i]), environment=int(d['environment_id'][i]),
            bucket=int(d['control_bucket'][i]), teacher_delta=float(d['teacher_final_delta'][i]),
            teacher_iterations=int(d['teacher_iterations'][i]),
            target=d['U'][i].tolist(), predicted=(pred[i]*norm['u_std']+norm['u_mean']).tolist(),
            error_normalized=err[i].tolist(),
        ) for i in worst[:10]]
        print(name, json.dumps(report['split_metrics'][name]), flush=True)
    tr, va = data['train'], data['validation']
    normalized_mean_difference = np.abs(tr['X'].mean(axis=0)-va['X'].mean(axis=0))/norm['x_std']
    report['distribution'] = {
        'max_input_mean_difference_in_train_std': float(normalized_mean_difference.max()),
        'input_mean_difference_p95': float(np.quantile(normalized_mean_difference,.95)),
        'target_mean_train': tr['U'].mean(axis=0).tolist(),
        'target_mean_validation': va['U'].mean(axis=0).tolist(),
        'target_std_train': tr['U'].std(axis=0).tolist(),
        'target_std_validation': va['U'].std(axis=0).tolist(),
        'overlapping_query_group_ids': int(len(np.intersect1d(tr['query_group_id'], va['query_group_id']))),
        'overlapping_base_episode_ids': int(len(np.intersect1d(tr['episode_id'],va['episode_id']))),
        'overlapping_base_rows': int(len(np.intersect1d(tr['base_row_id'],va['base_row_id']))),
    }
    for name,d in data.items():
        rows=np.ascontiguousarray(d['X']).view(np.dtype((np.void, d['X'].dtype.itemsize*300))).ravel()
        data[name]['row_bytes']=rows
    report['distribution']['exact_shared_X'] = int(len(np.intersect1d(data['train']['row_bytes'],data['validation']['row_bytes'])))
    report['generalization_ratio_same_checkpoint'] = report['split_metrics']['validation']['normalized_mse']/report['split_metrics']['train']['normalized_mse']
    (args.output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('report',args.output/'report.json',flush=True)


if __name__ == '__main__':
    main()
