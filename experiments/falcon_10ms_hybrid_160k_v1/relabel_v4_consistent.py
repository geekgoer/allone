"""Isolated hybrid-v4 label ablation; preserves states and original split membership.

No AirSim/GPU required. Each query starts from a fully initialized ACADO workspace.
Failed solver queries are recorded, never replaced by stale/one-iteration labels.
"""
from pathlib import Path
import argparse
import ctypes
import hashlib
import json
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from airsim_interface import RawDroneState
from falcon_paper_fidelity.contracts import build_trajectory_input
from falcon_paper_fidelity.motion_primitive import MotionPrimitiveConfig, QuinticMotionPrimitivePlanner
from falcon_paper_fidelity.phase_recovery_dataset import _build_trajectories, _controller, _load_gates
from prepare_dnn_dataset import _atomic_npz, normalization_from_train

SOURCE = HERE / 'dnn_dataset_hybrid_160k_10ms_v4_environment_balanced'
OUTPUT = HERE / 'dnn_dataset_v4_consistent_labels_v1'
POLICY = dict(min_iterations=4, max_iterations=64, delta_tolerance=1e-3,
              kkt_tolerance=1e-3, stable_iterations=3, bound_tolerance=1e-3)


def read_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def write_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def full_reset(controller):
    # initialize() also restores default weights: reapply ALL actual configuration.
    dll, c = controller._dll, controller.config
    controller._raise_status(int(dll.falcon_acado_initialize()), 'full initialize')
    values = ([c.q_position_xy, c.q_position_z, c.q_attitude, c.q_velocity, c.q_perception],
              [c.r_thrust, c.r_pitch_roll, c.r_yaw],
              [c.min_thrust, c.max_thrust, c.max_body_rate_xy, c.max_body_rate_z])
    arrays = [np.ascontiguousarray(v, dtype=np.float64) for v in values]
    status = dll.falcon_acado_configure(*[v.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) for v in arrays])
    controller._raise_status(int(status), 'configure')
    controller.reset()


def refine(controller, state, references):
    c = controller.config
    lower = np.array([c.min_thrust, -c.max_body_rate_xy, -c.max_body_rate_xy, -c.max_body_rate_z])
    upper = np.array([c.max_thrust, c.max_body_rate_xy, c.max_body_rate_xy, c.max_body_rate_z])
    previous, stable = None, 0
    delta = kkt = float('inf')
    output = np.full(4, np.nan)
    for iteration in range(1, POLICY['max_iterations'] + 1):
        try:
            output = controller.solve(state, references)
        except RuntimeError as exc:
            return output, False, iteration, delta, kkt, str(exc)
        delta = float(np.max(np.abs(output - previous))) if previous is not None else float('inf')
        kkt = abs(controller.last_diagnostics['kkt'])
        bounded = np.all(output >= lower - POLICY['bound_tolerance']) and np.all(output <= upper + POLICY['bound_tolerance'])
        good = (iteration >= POLICY['min_iterations'] and bounded and
                np.isfinite(output).all() and np.isfinite(kkt) and
                delta <= POLICY['delta_tolerance'] and kkt <= POLICY['kkt_tolerance'])
        stable = stable + 1 if good else 0
        if stable >= POLICY['stable_iterations']:
            return output, True, iteration, delta, kkt, ''
        previous = output.copy()
    return output, False, iteration, delta, kkt, 'iteration limit without stable KKT/control'


def query(controller, state, references):
    full_reset(controller)
    return (*refine(controller, state, references), 'cold_start')


def problem(data, index, trajectories, planner, cfg):
    refs = planner.reference_window_for_trajectory(trajectories[int(data['episode_id'][index])],
        elapsed=float(data['step_id'][index]) * cfg['timing']['dt'],
        dt=cfg['mpc']['prediction_dt'], horizon=cfg['mpc']['horizon'] + 1)
    state = RawDroneState(position=data['position'][index], quaternion=data['quaternion'][index],
        velocity=data['velocity'][index], acceleration=data['acceleration'][index], angular_velocity=np.zeros(3))
    np.testing.assert_allclose(build_trajectory_input(state, refs[0]), data['X'][index], atol=1e-4, rtol=0)
    return state, refs


def preflight(controller, trajectories, planner, cfg):
    """Always include rare launch states; global random sampling hides this regime."""
    rng = np.random.default_rng(77)
    rows = []
    for split in ('train', 'validation'):
        data = read_npz(SOURCE / f'{split}.npz')
        low = np.flatnonzero(np.linalg.norm(data['velocity'], axis=1) <= 3.5)
        indices = np.unique(np.concatenate([
            rng.choice(low, min(16, len(low)), False),
            *[rng.choice(np.flatnonzero(data['source_kind'] == k), 16, False) for k in range(4)]]))
        for i in indices:
            _, ok, iterations, delta, kkt, reason, _ = query(controller, *problem(data, int(i), trajectories, planner, cfg))
            rows.append(dict(split=split, source_index=int(i), source_kind=int(data['source_kind'][i]),
                speed=float(np.linalg.norm(data['velocity'][i])), accepted=ok, iterations=iterations,
                delta=delta if np.isfinite(delta) else None, kkt=kkt if np.isfinite(kkt) else None, reason=reason))
        print(f'Preflight {split} done', flush=True)
    failed = [r for r in rows if not r['accepted']]
    return dict(status='pass' if not failed else 'blocked_teacher_convergence',
                samples=len(rows), failures=len(failed), rows=rows,
                note='Does not use test data. A pass is not a closed-loop guarantee.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke-test', action='store_true', help='Stratified and low-speed train/validation preflight only')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    out = OUTPUT.with_name(OUTPUT.name + '_smoke') if args.smoke_test else OUTPUT
    if args.output_dir is not None:
        out = args.output_dir.resolve()
    if out.resolve() == SOURCE.resolve():
        raise ValueError('Must not overwrite the original v4 dataset')
    out.mkdir(parents=True, exist_ok=True)
    config_path = HERE / 'hybrid_extension_config.json'
    cfg = json.loads(config_path.read_text())
    controller = _controller(cfg, config_path)
    splits = ('train', 'validation') if args.smoke_test else ('train', 'validation', 'test')
    fingerprint = {'policy': POLICY, 'smoke': args.smoke_test, 'files': {str(p): digest(p) for p in
        [Path(__file__), config_path, controller.library_path, SOURCE / 'split_manifest.json',
         HERE.parents[1] / 'falcon_paper_fidelity' / 'acado_mpc.py',
         HERE.parents[1] / 'falcon_paper_fidelity' / 'motion_primitive.py',
         HERE.parents[1] / 'falcon_paper_fidelity' / 'phase_recovery_dataset.py',
         HERE.parents[1] / 'falcon_paper_fidelity' / 'contracts.py',
         HERE / cfg['source']['dataset'], HERE / cfg['source']['environments'],
         *[SOURCE / f'{s}.npz' for s in splits]]}}
    stamp = out / 'provenance.json'
    if stamp.exists():
        if not args.resume:
            raise RuntimeError('Output already exists; use --resume. Original data/models are unchanged.')
        if json.loads(stamp.read_text()) != fingerprint:
            raise RuntimeError('Source/config/code changed: refusing to mix labels in this output directory.')
    else:
        write_json(stamp, fingerprint)
    planner = QuinticMotionPrimitivePlanner(MotionPrimitiveConfig(**cfg['planner']))
    trajectories = _build_trajectories(read_npz(HERE / cfg['source']['dataset']),
        _load_gates(HERE / cfg['source']['environments']), planner,
        start_velocity_mode=cfg['trajectory']['start_velocity_mode'], nominal_speed=cfg['planner']['nominal_speed'],
        final_exit_distance=cfg['trajectory']['final_exit_distance'])
    check = preflight(controller, trajectories, planner, cfg)
    write_json(out / 'preflight.json', check)
    print(f'Preflight: {check["status"]}, failures={check["failures"]}/{check["samples"]}', flush=True)
    if check['status'] != 'pass':
        raise SystemExit('Teacher quality gate failed. No relabeled training dataset published; do not train this ablation yet.')
    if args.smoke_test:
        return
    summary, datasets = {}, {}
    rng = np.random.default_rng(1809)
    for split in splits:
        data = read_npz(SOURCE / f'{split}.npz')
        indices = (np.concatenate([rng.choice(np.flatnonzero(data['source_kind'] == k), 16, False) for k in range(4)])
                   if args.smoke_test else np.arange(len(data['X'])))
        parts = []
        for start in range(0, len(indices), 1000):
            path = out / 'chunks' / f'{split}_{start:06d}.npz'
            ids = indices[start:start+1000]
            if path.exists():
                part = read_npz(path)
                np.testing.assert_array_equal(part['source_index'], ids)
            else:
                rows = [query(controller, *problem(data, int(i), trajectories, planner, cfg)) for i in ids]
                u, ok, it, delta, kkt, reason, method = zip(*rows)
                part = dict(source_index=ids, U=np.array(u), accepted=np.array(ok), iterations=np.array(it),
                            delta=np.array(delta), kkt=np.array(kkt), rejection_reason=np.array(reason), method=np.array(method))
                _atomic_npz(path, part)
            parts.append(part)
            print(f'{split}: {start+len(ids)}/{len(indices)} | chunk rejected={int((~part["accepted"]).sum())}', flush=True)
        result = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        accepted = result['accepted']
        ids = result['source_index']
        low_speed = np.linalg.norm(data['velocity'][ids], axis=1) <= 3.5
        new_u = result['U'][accepted]
        c = controller.config
        saturated = ((new_u[:, 0] <= c.min_thrust+1e-3) | (new_u[:, 0] >= c.max_thrust-1e-3) |
                     np.any(np.abs(new_u[:, 1:3]) >= c.max_body_rate_xy-1e-3, axis=1) |
                     (np.abs(new_u[:, 3]) >= c.max_body_rate_z-1e-3))
        summary[split] = dict(queried=len(ids), accepted=int(accepted.sum()), rejected=int((~accepted).sum()),
            low_speed_before=int(low_speed.sum()), low_speed_after=int((low_speed & accepted).sum()),
            saturated_fraction=float(saturated.mean()) if len(new_u) else None,
            mean_iterations=float(result['iterations'].mean()),
            old_vs_new_physical_mse=float(np.mean((data['U'][ids[accepted]]-new_u)**2)) if len(new_u) else None,
            per_source={str(k): dict(queried=int((data['source_kind'][ids]==k).sum()),
                rejected=int(((data['source_kind'][ids]==k)&~accepted).sum())) for k in range(4)})
        _atomic_npz(out / f'{split}_label_audit.npz', result)
        # Preserve original IDs and input states. Rejection uses solver quality only, not DNN error.
        datasets[split] = {k: v[ids[accepted]] for k,v in data.items()}
        datasets[split]['U_original'] = datasets[split]['U'].copy()
        datasets[split]['U'] = new_u
        datasets[split]['source_index'] = ids[accepted]
        # Historical bucket/solve-time values are retained under explicitly historical names.
        for key in ('control_bucket', 'mpc_solve_time'):
            if key in datasets[split]:
                datasets[split]['original_' + key] = datasets[split].pop(key)
    # Predeclared engineering guards, not statistical guarantees; never silently remove a regime.
    passed = all(v['accepted'] > 0 and v['rejected']/v['queried'] <= .02 and
        v['low_speed_after'] >= .9*v['low_speed_before'] and
        all(s['rejected'] <= .02*s['queried'] for s in v['per_source'].values()) for v in summary.values())
    report = dict(status='smoke_only' if args.smoke_test else ('ready' if passed else 'blocked_quality'),
                  quality_guards_passed=passed, policy=POLICY, splits=summary,
                  note='Same split membership, 12-D states; labels changed. This is not the original RTI imitation task.')
    write_json(out / 'label_report.json', report)
    print(json.dumps(report, indent=2), flush=True)
    if args.smoke_test:
        return
    if not passed:
        raise RuntimeError('Too many rejected queries or low-speed coverage loss; no trainable dataset published. Inspect label_report.json.')
    for split, data in datasets.items():
        _atomic_npz(out / f'{split}.npz', data)
    norm = normalization_from_train(datasets['train']['X'], datasets['train']['U'])
    write_json(out / 'normalization.json', {k: v.tolist() for k,v in norm.items()})
    old = json.loads((SOURCE / 'split_manifest.json').read_text())
    manifest = {k: old[k] for k in ('input_contract', 'input_dimensions', 'input_features', 'output_dimensions',
                                    'reference_speed_values', 'split_unit', 'explicit_environment_split')}
    manifest.update(teacher_policy='fully_reset_refined_acado_v4_label_ablation_v1',
        source=str(SOURCE), provenance_sha256=digest(stamp), status='ready', splits=summary,
        normalization_fit='accepted train rows only', stored_arrays=list(datasets['train']))
    write_json(out / 'split_manifest.json', manifest)
    print(f'Ready: {out}', flush=True)


if __name__ == '__main__':
    main()
