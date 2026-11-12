"""Compare frozen main18 models with matched sampling and command timing."""
import argparse
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from airsim_interface import RawDroneState
from control_adapter import send_control
from environment import Gate
from falcon_paper_fidelity.motion_primitive import _rotation_matrix
from .decision_dataset import ROOT, atomic_json, sha256
from .eval_offline import load_checkpoint, load_normalization
from .fast_convex_preprocess import FastReferenceWindow, encode
from .fly_decision_dnn import StoredRoutePlanner
from .held_control_safety import GateHoldAnalyzer, HoldBounds
from .state_mode_calibration import moderate_case
from .teacher_attitude_probe import integrate_rotation


BUNDLE = ROOT/'ubuntu_transfer/falcon_independent_237d_dpu_pool_v1_20261016'
MODELS = ('main18_plain_mlp_h4_w1024_seed0', 'main18_plain_mlp_h33_w1024_seed0')
DURATION, PERIOD, DELAY, WIDTH = 3., .05, .02, 2.
TIMING_TOLERANCE = .012
CASES = tuple(dict(id=f'vy{v:g}', critical_index=0, position_y_m=-.05 if v else 0.,
                   velocity_y_mps=-v) for v in (0., .6, 1., 1.45))


def load_models(bundle, selection):
    import torch

    torch.set_num_threads(1)
    registry = json.loads((bundle/'model_registry.json').read_text(encoding='utf-8'))
    norm_dir = bundle/'shared'
    norm_sha = sha256(norm_dir/'normalization.json')
    if registry['input_dim'] != 237 or registry['normalization_sha256'] != norm_sha:
        raise ValueError('main18 normalization or input identity differs from registry')
    norm = load_normalization(norm_dir)
    registry_models = {row['id']: row for row in registry['models']}
    names = (tuple(name for name, row in registry_models.items() if row['cohort'] == 'main18')
             if selection == ['all'] else tuple(selection))
    if len(set(names)) != len(names) or MODELS[0] not in names:
        raise ValueError('select distinct main18 models including the h4/w1024 reference')
    models, hashes = {}, {}
    for name in names:
        entry = registry_models[name]
        checkpoint = bundle/entry['directory']/'best_model.pt'
        hashes[name] = sha256(checkpoint)
        if hashes[name] != entry['deployment_checkpoint_sha256'] or entry['cohort'] != 'main18':
            raise ValueError(f'main18 checkpoint identity differs: {name}')
        models[name] = load_checkpoint(checkpoint, device='cpu')

    @torch.inference_mode()
    def predict(name, feature):
        x = (feature.astype(np.float32)-norm['x_mean'])/norm['x_std']
        y = models[name](torch.from_numpy(x[None, :])).numpy()[0]
        return np.clip(y*norm['u_std']+norm['u_mean'], [2., -3., -3., -2.], [20., 3., 3., 2.])

    return predict, dict(models=hashes, normalization_sha256=norm_sha, input_contract=registry['input_contract'])


def geometry(gates, width=WIDTH, height=3.85):
    return GateHoldAnalyzer(gates, width, HoldBounds(2., .01, .05, .15, 10., 'paired recovery screening'),
                            net_height=height, body_radius=.5, clearance=.025)


def tracking(trace):
    pe = np.array([r['position_error_m'] for r in trace])
    ve = np.array([r['velocity_error_mps'] for r in trace])
    times = np.array([r['t'] for r in trace])
    outside = np.flatnonzero((pe > .3) | (ve > .6))
    i = int(outside[-1])+1 if len(outside) else 0
    recovered = i < len(times) and times[-1]-times[i] >= .25-1e-9
    return dict(position_rmse_m=float(np.sqrt(np.mean(pe**2))), peak_error_m=float(pe.max()),
                terminal_error_m=float(pe[-1]), terminal_velocity_error_mps=float(ve[-1]),
                tracking_recovered=bool(recovered), recovery_s=float(times[i]) if recovered else None)


def offline_trial(window, phase, analyzer, predict, name, case, step):
    initial = window.window(phase, horizon=1)[0]
    p, v, q = initial.position.copy(), initial.velocity.copy(), initial.orientation.copy()
    p[1] += case['position_y_m']
    v[1] += case['velocity_y_mps']
    gravity = np.array([0., 0., 9.80665])
    held = np.r_[np.linalg.norm(initial.acceleration-gravity), np.zeros(3)]
    a, omega = np.zeros(3), np.zeros(3)
    initial_budget = analyzer.analyze(RawDroneState(p, q, v, a, omega), held)['budget_s']
    release_ticks, delay_ticks = round(PERIOD/step), round(DELAY/step)
    trace, commands = [], []
    for k in range(round(DURATION/step)+1):
        t = k*step
        ref = window.window(phase+t, horizon=1, include_body_rates=False)[0]
        if k < round(DURATION/step) and k % release_ticks == 0:
            refs = window.window(phase+t, include_body_rates=False)
            pending = predict(name, encode(RawDroneState(p, q, v, a, omega), refs))
        elif k % release_ticks == delay_ticks:
            held = pending
            commands.append(dict(sample_s=t-DELAY, apply_s=t, control=held.tolist()))
        margin = (analyzer.normals@p-analyzer.offsets).max(axis=1).min()
        trace.append(dict(t=t, position=p.tolist(), velocity=v.tolist(),
                          position_error_m=float(np.linalg.norm(p-ref.position)),
                          velocity_error_mps=float(np.linalg.norm(v-ref.velocity)), frame_margin_m=float(margin)))
        omega = held[1:]
        midpoint = integrate_rotation(q, omega, step/2)
        a = gravity-held[0]*_rotation_matrix(midpoint)[:, 2]
        p, v = p+v*step+.5*a*step**2, v+a*step
        q = integrate_rotation(q, omega, step)
    return dict(**tracking(trace), initial_budget_s=initial_budget,
                minimum_frame_margin_m=min(r['frame_margin_m'] for r in trace), step_s=step), commands, trace


def airsim_trial(world, predict, name, case, output, duration=DURATION, *, release_delays=None,
                 period=PERIOD, skip_releases=(), state_controller=None):
    from .probe_controller_step_recoverability import advance

    output.mkdir(parents=True, exist_ok=False)
    initial = world.initialize(case, output)['Drone1']
    initial_ns = initial.timestamp_ns
    sample = world.snapshot('Drone1')
    initial = sample[0]
    world.observe('Drone1', *sample)
    commands = []

    def elapsed():
        return (sample[0].timestamp_ns-initial_ns)/1e9

    def step_to(target):
        remaining = target-elapsed()
        if remaining > 1e-6:
            state = advance(world, remaining)
            world.observe('Drone1', *state)
            return state
        return sample

    phase = world.phases['Drone1']
    window = world.windows['Drone1']
    analyzer = geometry(world.gates['Drone1'], world.net_width)
    held = np.asarray(world.initial['Drone1']['held_control'])
    initial_budget = analyzer.analyze(initial, held)['budget_s']
    delays = release_delays or {}
    intervention_start = None
    dropped, skipped = [], []
    for k in range(round(duration/period)):
        if world.failures['Drone1']:
            break
        scheduled = k*period
        sample_s = elapsed()
        if k in delays and intervention_start is None:
            raw = sample[0]
            intervention_start = dict(sample_s=sample_s, position=raw.position.tolist(),
                velocity=raw.velocity.tolist(), quaternion=raw.quaternion.tolist(),
                acceleration=raw.acceleration.tolist(), angular_velocity=raw.angular_velocity.tolist(),
                held_control=held.tolist())
        if k in skip_releases:
            sample = step_to(scheduled+delays.get(k, DELAY))
            skipped.append(k)
            if not world.failures['Drone1']:
                sample = step_to((k+1)*period)
            continue
        refs = window.window(phase+sample_s, include_body_rates=False)
        action = (predict(name, encode(sample[0], refs)) if state_controller is None
                  else state_controller(sample[0], refs))
        apply_target = scheduled+delays.get(k, DELAY)
        sample = step_to(apply_target)
        if world.failures['Drone1']:
            break
        apply_s = elapsed()
        if delays and apply_s >= (k+1)*period:
            dropped.append(k)
        else:
            send_control(world.drones['Drone1'], action, world.calibration, duration=1.)
            world.client.ping()
            held = action
            commands.append(dict(sample_s=sample_s, apply_s=apply_s, scheduled_sample_s=scheduled,
                                 scheduled_apply_s=apply_target, control=action.tolist()))
        sample = step_to((k+1)*period)
    # AirSim can stop one physics tick short of the requested target.
    # Cover the endpoint under the last command, without another DNN release.
    terminal_step_s = 0.
    if not world.failures['Drone1'] and elapsed() < duration:
        terminal_step_s = duration-elapsed()+.005
        sample = advance(world, terminal_step_s)
        world.observe('Drone1', *sample)
    trace = world.traces['Drone1']
    result = dict(**tracking(trace), physical=world.physical_report(duration, realtime=False)['Drone1'],
                  terminal_step_s=terminal_step_s,
                  initial_budget_s=initial_budget,
                  initial_readback=np.r_[initial.position, initial.velocity, initial.quaternion,
                                         initial.acceleration, initial.angular_velocity].tolist(),
                  schedule_within_tolerance=bool(commands) and all(
                      abs(r[f'{key}_s']-r[f'scheduled_{key}_s']) <= TIMING_TOLERANCE
                      for r in commands for key in ('sample', 'apply')))
    if delays:
        result.update(intervention_start=intervention_start, dropped_deadline_releases=dropped)
    if skip_releases:
        result['intentional_skips'] = skipped
    return result, commands, trace


def compare_timing(timing, variants):
    reference = timing[MODELS[0]]
    initial_tolerance = np.repeat([.02, .05, .01, .10, .05], [3, 3, 4, 3, 3])
    initial = np.asarray(variants[MODELS[0]]['initial_readback'])
    matches = {}
    for name, commands in timing.items():
        drift = max((abs(x[key]-y[key]) for x, y in zip(reference, commands)
                     for key in ('sample_s', 'apply_s')), default=None)
        matched = (drift is not None and drift <= TIMING_TOLERANCE
                   and bool(np.all(np.abs(initial-variants[name]['initial_readback']) <= initial_tolerance))
                   and variants[MODELS[0]]['schedule_within_tolerance']
                   and variants[name]['schedule_within_tolerance'])
        matches[name] = dict(matched=matched, drift_s=drift, command_count=len(commands),
                             common_commands=min(len(reference), len(commands)))
    return matches


def run(output, bundle, backend, selection=MODELS):
    from .framework_runtime_world import FrameworkWorld

    predict, identity = load_models(bundle, selection)
    names = tuple(identity['models'])
    if len(names) < 2:
        raise ValueError('the matched-model screen needs at least two models')
    route, layout = moderate_case()
    phase = route['reference_phase_s']
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output/'setup.json', dict(identity=identity, route=route, cases=CASES, period_s=PERIOD,
                delay_s=DELAY, duration_s=DURATION, net_width_m=WIDTH, backend=backend,
                timing_tolerance_s=TIMING_TOLERANCE, source_sha256=sha256(Path(__file__))))
    world = None
    pairs = []
    try:
        if backend == 'airsim':
            world = FrameworkWorld(layout, phase, 1, WIDTH, task_duration=DURATION)
        else:
            planner = StoredRoutePlanner(layout)
            window = FastReferenceWindow(planner.trajectory, planner.config)
            window.window = lru_cache(maxsize=None)(window.window)
            analyzer = geometry([Gate(g['index'], g['center'], g['orientation']) for g in layout['gates']])
        for i, case in enumerate(CASES):
            variants, timing = {}, {}
            for name in names if i % 2 == 0 else names[::-1]:
                directory = output/case['id']/name
                if world is not None:
                    result, commands, trace = airsim_trial(world, predict, name, case, directory)
                else:
                    result, commands, trace = offline_trial(window, phase, analyzer, predict, name, case, .005)
                    refined, _, _ = offline_trial(window, phase, analyzer, predict, name, case, .0025)
                    result['refined_step_metrics'] = refined
                    directory.mkdir(parents=True)
                atomic_json(directory/'trace.json', trace)
                atomic_json(directory/'commands.json', commands)
                atomic_json(directory/'metrics.json', result)
                variants[name], timing[name] = result, commands
            if world is not None:
                matches = compare_timing(timing, variants)
            else:
                a = timing[MODELS[0]]
                matches = {}
                for name, b in timing.items():
                    drift = max((abs(x[key]-y[key]) for x, y in zip(a, b)
                                 for key in ('sample_s', 'apply_s')), default=None)
                    assert drift == 0. and len(a) == len(b) == round(DURATION/PERIOD)
                    matches[name] = dict(matched=True, drift_s=drift, command_count=len(b))
            drifts = [m['drift_s'] for m in matches.values() if m['drift_s'] is not None]
            paired = all(m['matched'] for m in matches.values())
            pairs.append(dict(case=case['id'], maximum_matched_timing_difference_s=max(drifts, default=None),
                              matched=paired, command_counts=[len(timing[n]) for n in names], metrics=variants,
                              comparisons_to_h4_w1024=matches))
            atomic_json(output/'summary.json', dict(backend=backend, pairs=pairs, complete=len(pairs)==len(CASES)))
            print(f"{case['id']} matched={paired} RMSE "
                  f"{[round(variants[n]['position_rmse_m'], 4) for n in names]}", flush=True)
    finally:
        if world is not None:
            atomic_json(output/'cleanup_report.json', world.close())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('offline', 'airsim'), default='offline')
    parser.add_argument('--bundle', type=Path, default=BUNDLE)
    parser.add_argument('--models', nargs='+', default=MODELS, help='registered model names, or all')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.output, args.bundle, args.backend, args.models)
