"""Bounded CPU audit of hybrid-v4 checkpoints and teacher labels; no test use."""
from pathlib import Path
import json
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from airsim_interface import RawDroneState
from dnn_model_utils import build_model, load_normalization
from falcon_paper_fidelity.contracts import build_trajectory_input
from falcon_paper_fidelity.motion_primitive import MotionPrimitiveConfig, QuinticMotionPrimitivePlanner
from falcon_paper_fidelity.phase_recovery_dataset import _build_trajectories, _controller, _load_gates


def main():
    import torch
    torch.set_num_threads(4)
    directory = HERE / 'dnn_dataset_hybrid_160k_10ms_v4_environment_balanced'
    norm = load_normalization(directory / 'normalization.json')
    data = {}
    for split in ('train', 'validation'):
        with np.load(directory / f'{split}.npz', allow_pickle=False) as z:
            data[split] = {k: z[k] for k in z.files}
    rng = np.random.default_rng(1809)
    # 1000 random rows per source per split, with no selection on model error.
    indices = {s: np.concatenate([rng.choice(np.flatnonzero(d['source_kind'] == k), 1000, False)
                                 for k in range(4)]) for s, d in data.items()}
    report = {'sampling': '4000 rows per split; 1000 per source; seed 1809', 'models': {}}
    for depth, width in ((10, 512), (34, 1024)):
        name = f'depth{depth}_width{width}'
        path = HERE / 'runs_local_pilot_300_seed0_v4_fresh' / 'seed_0' / name / 'best_model.pt'
        ck = torch.load(path, map_location='cpu', weights_only=False)
        for key, value in norm.items():
            np.testing.assert_allclose(value, ck['normalization'][key], rtol=1e-6, atol=1e-7)
        model = build_model(torch, depth=depth, width=width, input_dim=12).eval()
        model.load_state_dict(ck['model_state'])
        result = {'epoch': ck['epoch']}
        for split, d in data.items():
            idx = indices[split]
            x = ((d['X'][idx] - norm['x_mean']) / norm['x_std']).astype(np.float32)
            y = ((d['U'][idx] - norm['u_mean']) / norm['u_std']).astype(np.float32)
            with torch.inference_mode():
                pred = np.concatenate([model(torch.from_numpy(x[j:j+256])).numpy() for j in range(0, len(x), 256)])
            err = (pred - y).astype(np.float64) ** 2
            part = {'normalized_mse': float(err.mean()), 'per_output': err.mean(0).tolist()}
            for column in ('source_kind', 'control_bucket', 'environment_id'):
                part[column] = {str(int(v)): {'n': int(np.sum(d[column][idx] == v)),
                    'mse': float(err[d[column][idx] == v].mean())} for v in np.unique(d[column][idx])}
            part['top_1pct_error_share'] = float(np.sort(err.mean(1))[-40:].sum() / err.mean(1).sum())
            result[split] = part
        report['models'][name] = result
        print(name, json.dumps(result), flush=True)

    config_file = HERE / 'hybrid_extension_config.json'
    cfg = json.loads(config_file.read_text())
    with np.load(HERE / cfg['source']['dataset']) as z:
        core = {k: z[k] for k in z.files}
    planner = QuinticMotionPrimitivePlanner(MotionPrimitiveConfig(**cfg['planner']))
    trajectories = _build_trajectories(core, _load_gates(HERE / cfg['source']['environments']), planner,
        start_velocity_mode=cfg['trajectory']['start_velocity_mode'],
        nominal_speed=cfg['planner']['nominal_speed'], final_exit_distance=cfg['trajectory']['final_exit_distance'])
    ctl = _controller(cfg, config_file)
    queries = []
    for split, d in data.items():
        for kind in range(4):
            for i in indices[split][kind*1000:kind*1000+16]:
                refs = planner.reference_window_for_trajectory(trajectories[int(d['episode_id'][i])],
                    elapsed=float(d['step_id'][i])*.05, dt=.1, horizon=21)
                state = RawDroneState(position=d['position'][i], quaternion=d['quaternion'][i],
                    velocity=d['velocity'][i], acceleration=d['acceleration'][i], angular_velocity=np.zeros(3))
                x_error = float(np.max(np.abs(build_trajectory_input(state, refs[0]) - d['X'][i])))
                if x_error > 1e-4:
                    raise ValueError(f'reconstructed input mismatch: {split}/{i}: {x_error}')
                ctl.reset()
                first = ctl.solve(state, refs)
                previous = first
                for _ in range(63):
                    refined = ctl.solve(state, refs)
                    delta = float(np.max(np.abs(refined - previous)))
                    previous = refined
                queries.append({'split': split, 'source': kind, 'index': int(i),
                    'stored': d['U'][i].tolist(), 'reset_first': first.tolist(), 'refined64': refined.tolist(),
                    'stored_vs_first_max': float(np.max(np.abs(d['U'][i]-first))),
                    'stored_vs_refined_max': float(np.max(np.abs(d['U'][i]-refined))),
                    'stored_vs_refined_normalized_mse': float(np.mean(((d['U'][i]-refined)/norm['u_std'])**2)),
                    'first_vs_refined_max': float(np.max(np.abs(first-refined))),
                    'final_delta': delta, 'kkt64': ctl.last_diagnostics['kkt']})
            print('teacher replay', split, kind, 'done', flush=True)
    report['teacher_queries'] = queries
    report['teacher_summary'] = {}
    for kind in range(4):
        subset = [q for q in queries if q['source'] == kind]
        report['teacher_summary'][str(kind)] = {key: {'mean': float(np.mean([q[key] for q in subset])),
            'max': float(np.max([q[key] for q in subset]))} for key in
            ('stored_vs_first_max', 'stored_vs_refined_max', 'stored_vs_refined_normalized_mse',
             'first_vs_refined_max', 'final_delta', 'kkt64')}
    report['coverage'] = {s: {'low_speed_fraction': float(np.mean(np.linalg.norm(d['velocity'], axis=1)<=3.5)),
        'max_mean_difference_train_std': float(np.max(np.abs(d['X'].mean(0)-data['train']['X'].mean(0))/norm['x_std']))}
        for s,d in data.items()}
    output = HERE / 'diagnostics' / 'v4_generalization_audit.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report['teacher_summary'], indent=2), flush=True)
    print(output)


if __name__ == '__main__':
    main()
