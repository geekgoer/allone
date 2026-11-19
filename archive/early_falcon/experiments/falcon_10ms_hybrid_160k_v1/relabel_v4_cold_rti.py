"""Make hybrid-v4 labels a deterministic one-RTI-step ACADO policy.

The original v4 data mixes cold one-step queries with history-warm continuous
queries.  This ablation keeps every stored state, reference window, 12-D input,
and environment split, but recomputes *every* target from the same initialized
solver state.  It does not run AirSim and never overwrites the source dataset.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from airsim_interface import RawDroneState
from falcon_paper_fidelity.contracts import build_trajectory_input
from falcon_paper_fidelity.motion_primitive import MotionPrimitiveConfig, QuinticMotionPrimitivePlanner
from falcon_paper_fidelity.phase_recovery_dataset import _build_trajectories, _controller, _load_gates
from prepare_dnn_dataset import _atomic_npz, normalization_from_train

SOURCE = HERE / 'dnn_dataset_hybrid_160k_10ms_v4_environment_balanced'
OUTPUT = HERE / 'dnn_dataset_v4_cold_rti_labels_v1'
SPLITS = ('train', 'validation', 'test')


def read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def full_reset(controller) -> None:
    """Erase primal/dual/workspace state, then restore the configured problem."""
    config, dll = controller.config, controller._dll
    controller._raise_status(int(dll.falcon_acado_initialize()), 'full initialize')
    arrays = [
        np.ascontiguousarray([config.q_position_xy, config.q_position_z, config.q_attitude,
                              config.q_velocity, config.q_perception], dtype=np.float64),
        np.ascontiguousarray([config.r_thrust, config.r_pitch_roll, config.r_yaw], dtype=np.float64),
        np.ascontiguousarray([config.min_thrust, config.max_thrust,
                              config.max_body_rate_xy, config.max_body_rate_z], dtype=np.float64),
    ]
    pointer = ctypes.POINTER(ctypes.c_double)
    status = dll.falcon_acado_configure(*[array.ctypes.data_as(pointer) for array in arrays])
    controller._raise_status(int(status), 'configure after full initialize')
    controller.reset()


def reconstruct_problem(data: dict[str, np.ndarray], index: int, trajectories, planner, config):
    references = planner.reference_window_for_trajectory(
        trajectories[int(data['episode_id'][index])],
        elapsed=float(data['step_id'][index]) * config['timing']['dt'],
        dt=config['mpc']['prediction_dt'], horizon=config['mpc']['horizon'] + 1,
    )
    state = RawDroneState(
        position=data['position'][index], quaternion=data['quaternion'][index],
        velocity=data['velocity'][index], acceleration=data['acceleration'][index],
        angular_velocity=np.zeros(3),
    )
    np.testing.assert_allclose(build_trajectory_input(state, references[0]), data['X'][index], atol=1e-4, rtol=0)
    return state, references


def one_step(controller, state, references):
    full_reset(controller)
    output = controller.solve(state, references)
    config = controller.config
    lower = np.array([config.min_thrust, -config.max_body_rate_xy, -config.max_body_rate_xy, -config.max_body_rate_z])
    upper = np.array([config.max_thrust, config.max_body_rate_xy, config.max_body_rate_xy, config.max_body_rate_z])
    valid = bool(np.isfinite(output).all() and np.all(output >= lower - 1e-3) and np.all(output <= upper + 1e-3))
    return output, valid, float(controller.last_diagnostics['kkt']), float(controller.last_diagnostics['objective'])


def deterministic_check(controller, data, trajectories, planner, config, indices: np.ndarray) -> dict[str, object]:
    rows, failures = [], []
    for index in indices:
        state, references = reconstruct_problem(data, int(index), trajectories, planner, config)
        first = one_step(controller, state, references)
        second = one_step(controller, state, references)
        difference = float(np.max(np.abs(first[0] - second[0])))
        row = dict(index=int(index), source_kind=int(data['source_kind'][index]),
                   speed=float(np.linalg.norm(data['velocity'][index])),
                   valid=bool(first[1] and second[1]), repeat_max_abs=difference,
                   kkt=float(first[2]), objective=float(first[3]))
        rows.append(row)
        if not row['valid'] or difference > 2e-5:
            failures.append(row)
    return dict(samples=len(rows), failures=len(failures), max_repeat_abs=max((r['repeat_max_abs'] for r in rows), default=0.0),
                rows=rows, failed_rows=failures)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke-test', action='store_true', help='Only verify 160 train/validation states; publish nothing.')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output == SOURCE.resolve():
        raise ValueError('The original v4 dataset is read-only for this experiment.')
    output.mkdir(parents=True, exist_ok=True)

    config_path = HERE / 'hybrid_extension_config.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    controller = _controller(config, config_path)
    source_files = [SOURCE / f'{split}.npz' for split in SPLITS]
    provenance = dict(
        policy='fully_initialized_acado_then_exactly_one_rti_step_v1',
        input_contract='falcon_track_p_trajectory_relative_12d_v1',
        config_sha256=digest(config_path), dll_sha256=digest(controller.library_path),
        source_sha256={path.name: digest(path) for path in source_files},
    )
    provenance_path = output / 'provenance.json'
    if provenance_path.exists():
        if not args.resume:
            raise RuntimeError('Output exists. Use --resume to continue this exact relabel run.')
        if json.loads(provenance_path.read_text(encoding='utf-8')) != provenance:
            raise RuntimeError('Source/config/DLL differs; refusing to mix labels.')
    else:
        write_json(provenance_path, provenance)

    source_core = read_npz(HERE / config['source']['dataset'])
    planner = QuinticMotionPrimitivePlanner(MotionPrimitiveConfig(**config['planner']))
    trajectories = _build_trajectories(
        source_core, _load_gates(HERE / config['source']['environments']), planner,
        start_velocity_mode=config['trajectory']['start_velocity_mode'],
        nominal_speed=config['planner']['nominal_speed'], final_exit_distance=config['trajectory']['final_exit_distance'],
    )

    rng = np.random.default_rng(20260918)
    preflight = {}
    for split in ('train', 'validation'):
        data = read_npz(SOURCE / f'{split}.npz')
        low = np.flatnonzero(np.linalg.norm(data['velocity'], axis=1) <= 3.5)
        sampled = [rng.choice(low, min(16, len(low)), replace=False)]
        sampled.extend(rng.choice(np.flatnonzero(data['source_kind'] == kind), 16, replace=False) for kind in range(4))
        preflight[split] = deterministic_check(controller, data, trajectories, planner, config,
                                               np.unique(np.concatenate(sampled)))
        print(f'Preflight {split}: failures={preflight[split]["failures"]}/{preflight[split]["samples"]}', flush=True)
    write_json(output / 'preflight.json', preflight)
    if any(value['failures'] for value in preflight.values()):
        raise RuntimeError('Cold one-step teacher is not deterministic/valid on preflight; no dataset published.')
    if args.smoke_test:
        print(f'PASS: {output / "preflight.json"}', flush=True)
        return

    final, report = {}, dict(policy=provenance['policy'], preflight=preflight, splits={})
    for split in SPLITS:
        data = read_npz(SOURCE / f'{split}.npz')
        labels, kkt, objective = [], [], []
        for start in range(0, len(data['X']), 1000):
            chunk_path = output / 'chunks' / f'{split}_{start:06d}.npz'
            count = min(1000, len(data['X']) - start)
            if chunk_path.exists():
                chunk = read_npz(chunk_path)
                if int(chunk['start'][0]) != start or len(chunk['U']) != count:
                    raise RuntimeError(f'Invalid resume chunk: {chunk_path}')
            else:
                values = [one_step(controller, *reconstruct_problem(data, i, trajectories, planner, config))
                          for i in range(start, start + count)]
                if not all(value[1] for value in values):
                    raise RuntimeError(f'Non-finite/out-of-bound RTI output in {split} chunk {start}; no labels hidden or clipped.')
                chunk = dict(start=np.array([start]), U=np.asarray([value[0] for value in values], dtype=np.float32),
                             kkt=np.asarray([value[2] for value in values], dtype=np.float32),
                             objective=np.asarray([value[3] for value in values], dtype=np.float32))
                _atomic_npz(chunk_path, chunk)
            labels.append(chunk['U']); kkt.append(chunk['kkt']); objective.append(chunk['objective'])
            print(f'{split}: {start + count}/{len(data["X"])}', flush=True)
        data['U_original'] = data['U'].copy()
        data['U'] = np.concatenate(labels)
        data['cold_rti_kkt'] = np.concatenate(kkt)
        data['cold_rti_objective'] = np.concatenate(objective)
        final[split] = data
        report['splits'][split] = dict(samples=len(data['U']),
            original_vs_cold_physical_mse=float(np.mean((data['U_original'] - data['U']) ** 2)),
            kkt_p99=float(np.quantile(data['cold_rti_kkt'], .99)),
            saturation_fraction=float(np.mean((data['U'][:, 0] <= controller.config.min_thrust + 1e-3) |
                (data['U'][:, 0] >= controller.config.max_thrust - 1e-3) |
                np.any(np.abs(data['U'][:, 1:3]) >= controller.config.max_body_rate_xy - 1e-3, axis=1) |
                (np.abs(data['U'][:, 3]) >= controller.config.max_body_rate_z - 1e-3))))
    for split, data in final.items():
        _atomic_npz(output / f'{split}.npz', data)
    normalization = normalization_from_train(final['train']['X'], final['train']['U'])
    write_json(output / 'normalization.json', {key: value.tolist() for key, value in normalization.items()})
    old_manifest = json.loads((SOURCE / 'split_manifest.json').read_text(encoding='utf-8'))
    manifest = {key: old_manifest[key] for key in ('input_contract', 'input_dimensions', 'input_features', 'output_dimensions',
                                                     'reference_speed_values', 'split_unit', 'explicit_environment_split')}
    manifest.update(teacher_policy=provenance['policy'], source_dataset=str(SOURCE),
                    split_membership='unchanged from hybrid v4', normalization_fit='train only',
                    stored_arrays=list(final['train']), report='label_report.json')
    write_json(output / 'split_manifest.json', manifest)
    write_json(output / 'label_report.json', report)
    print(f'Ready: {output}', flush=True)


if __name__ == '__main__':
    main()
