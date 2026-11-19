"""Matched AirSim recovery labels with prescribed simulation-time delays."""

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import time

import numpy as np

from airsim_interface import RawDroneState
from control_adapter import control_to_airsim_command, throttle_to_thrust, send_control
from state_adapter import quaternion_multiply
from ..decision_dataset import ROOT, atomic_json, sha256
from ..dynamic_probe_common import read_json
from ..fair_static_dynamic import MODEL_BANK
from ..framework_campaign import layout_for
from ..framework_runtime import ControllerPool
from ..framework_runtime_features import reference_point, encode_direct
from .data import VERSION, PROTOCOL, architecture, recovery_label, paired_groups


SPLITS = dict(train=range(2000, 2032), validation=range(2100, 2108),
              calibration=range(2200, 2220), test=range(2300, 2308))


def source_hashes():
    base = Path(__file__).parent
    files = [base/'data.py', base/'collect.py', base.parent/'framework_runtime_world.py',
             base.parent/'framework_runtime_features.py', base.parent/'framework_runtime.py',
             ROOT/'control_adapter.py', ROOT/'gate_visualization.py']
    return {str(p.relative_to(ROOT)):sha256(p) for p in files}


def physical_control(action, calibration):
    action = np.clip(action, [2., -3., -3., -2.], [20., 3., 3., 2.])
    command = control_to_airsim_command(action, calibration, 1.)
    return np.array([throttle_to_thrust(command.throttle, calibration), command.roll_rate,
                     -command.pitch_rate, -command.yaw_rate])


def prepare(root):
    from config import load_calibration
    from ..fast_convex_preprocess import FastReferenceWindow
    from ..fly_decision_dnn import StoredRoutePlanner
    pool = ControllerPool(model_names=MODEL_BANK)
    calibration = load_calibration()
    root.mkdir(parents=True, exist_ok=False)
    cases = []
    for split, seeds in SPLITS.items():
        for seed in seeds:
            rng = np.random.default_rng(seed)
            family = ('smooth', 'continuous', 'late')[seed % 3]
            layout = layout_for(0, 3, seed, family)
            gate = min(8, layout['hard_onset_gate']) if family == 'late' else int(rng.integers(2, 8))
            segment = layout['segments'][gate]
            phase = segment['start_time']+segment['duration']-.7
            planner = StoredRoutePlanner(layout)
            window = FastReferenceWindow(planner.trajectory, planner.config)
            ref = reference_point(window, phase)
            nominal = RawDroneState(ref.position, ref.orientation, ref.velocity, ref.acceleration, np.zeros(3))
            held = physical_control(pool.predict('plain_mlp_h9_w1024_seed0', encode_direct(window, nominal, phase)), calibration)
            rotation = rng.uniform(-.07, .07, 3)
            angle = np.linalg.norm(rotation)
            dq = np.r_[np.cos(angle/2), rotation*np.sin(angle/2)/angle] if angle else np.array([1., 0., 0., 0.])
            cases.append(dict(id=f'{split}_s{seed}', split=split, route_seed=seed, family=family,
                phase_s=phase, layout=layout, relative_initial=dict(
                    position_error=[float(rng.uniform(-.25, .25)), float(rng.uniform(-.25, .25)), float(rng.uniform(-.12, .12))],
                    velocity_error=[float(rng.uniform(-.6, .6)), float(rng.uniform(-1., 1.)), float(rng.uniform(-.4, .4))],
                    quaternion=quaternion_multiply(ref.orientation, dq).tolist(), held_control=held.tolist())))
    plan = dict(version=VERSION, protocol=PROTOCOL, models=list(MODEL_BANK), delays_s=[.020, .050, .080],
        provenance=pool.provenance(), model_cost_s={m:sum(p['segment_estimates_s']) for m,p in pool.profile.items()},
        descriptors={m:architecture(m).tolist() for m in MODEL_BANK}, cases=cases,
        source_sha256=source_hashes(), state_pairing='same full observable state and effective held command; independently reset plant',
        delay_semantics='measured points near requested delay, not a certificate for every smaller delay',
        future_contract='50ms simulation actuation calendar, <=60ms measured gap, 10ms imposed recurring input age, <=20ms measured age',
        test_used_for_training=False, online_policy_modified=False)
    for name in plan['source_sha256']:
        destination = root/'frozen_source'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, destination)
    atomic_json(root/'plan.json', plan)
    print('Prepared', len(cases), 'independent scene groups,', len(cases)*len(MODEL_BANK)*3, 'paired trials:', root, flush=True)


def record_world(world, raw, hit, sampled, analyzer):
    row = world.observe('Drone1', raw, hit, sampled)
    # Unit normal half-spaces: a conservative signed clearance to finite
    # inflated boxes, not a moving-route error masquerading as mesh clearance.
    row['box_margin_m'] = float((analyzer.normals@raw.position-analyzer.offsets).max(axis=1).min())
    return row


def run_trial(world, pool, model, delay, case, directory, *, protocol=None):
    from ..held_control_safety import GateHoldAnalyzer, HoldBounds
    directory.mkdir(parents=True, exist_ok=False)
    p = PROTOCOL if protocol is None else protocol
    trace, commands = [], []
    error = error_kind = None
    feature = None
    actual_delay = None
    try:
        raw = world.initialize(dict(critical_index=-1, relative_initial=case['relative_initial']), directory)['Drone1']
        _, hit, sampled = world.snapshot('Drone1')
        analyzer = GateHoldAnalyzer(world.gates['Drone1'], world.net_width,
            HoldBounds(2., .01, .05, .15, 10., 'geometry only in this data collector; no barrier safety claim'))
        first = record_world(world, raw, hit, sampled, analyzer)
        trace = world.traces['Drone1']
        start_ns = raw.timestamp_ns
        z = encode_direct(world.windows['Drone1'], raw, world.phases['Drone1'])
        held = np.array(world.initial['Drone1']['held_control'])
        feature = np.r_[z, held].astype(np.float32).tolist()
        began = time.perf_counter()

        def elapsed():
            return (raw.timestamp_ns-start_ns)/1e9

        def move_to(target):
            nonlocal raw
            while elapsed() < target-1e-8 and not world.failures['Drone1']:
                if time.perf_counter()-began > 40.:
                    raise TimeoutError('stepped recovery exceeded its 40s wall guard')
                before = raw.timestamp_ns
                for retry in range(8):
                    world.client.simContinueForTime(min(p['observation_s'], max(.003, target-elapsed())))
                    until = time.perf_counter()+3.
                    while not world.client.simIsPause():
                        if time.perf_counter() > until:
                            raise TimeoutError('AirSim did not pause after the physical step')
                        time.sleep(.001)
                    raw, collision, sampled = world.snapshot('Drone1')
                    if raw.timestamp_ns > before:
                        break
                if raw.timestamp_ns <= before:
                    raise RuntimeError('AirSim physics clock did not advance after eight physical-step attempts')
                record_world(world, raw, collision, sampled, analyzer)

        def apply(action, input_s, target_s):
            nonlocal actual_delay
            move_to(target_s)
            if world.failures['Drone1']:
                return False
            effective = physical_control(action, world.calibration)
            # Physics is paused here. RPC completion is a command-acceptance
            # proxy; the following tick applies it, not a measured motor event.
            send_control(world.drones['Drone1'], effective, world.calibration, duration=p['command_lease_s'])
            world.client.ping()
            when = elapsed()
            commands.append(dict(input_s=input_s, target_s=target_s, applied_proxy_s=when,
                input_age_s=when-input_s, control=effective.tolist()))
            if actual_delay is None:
                actual_delay = when
            return True

        # First command uses the t=0 state, then waits while the old command
        # stays active. Its delay includes all imposed computation/queue time.
        action = pool.predict(model, np.array(feature[:237], np.float32))
        if apply(action, 0., delay):
            for k in range(1, 1000):
                target = delay+k*p['cadence_s']
                if target >= p['duration_s']:
                    break
                move_to(target-p['recurring_input_age_s'])
                if world.failures['Drone1']:
                    break
                input_s = elapsed()
                z = encode_direct(world.windows['Drone1'], raw, world.phases['Drone1']+input_s)
                action = pool.predict(model, z)
                if not apply(action, input_s, target):
                    break
            move_to(p['duration_s'])
        if world.failures['Drone1']:
            error_kind = 'flight'
            error = str(world.failures['Drone1'])
        times = [c['applied_proxy_s'] for c in commands]
        gaps = np.diff(times).tolist()
        if commands:
            gaps.append(elapsed()-times[-1])
        delay_ok = actual_delay is None or abs(actual_delay-delay) <= p['initial_delay_tolerance_s']+1e-8
        cadence_ok = all(g <= p['update_bound_s']+1e-8 for g in gaps)
        ages_ok = all(c['input_age_s'] <= p['input_age_bound_s']+1e-8 for c in commands[1:])
        if not (delay_ok and cadence_ok and ages_ok):
            error_kind, error = 'timing', 'prescribed simulation delay/update/age protocol not met'
    except KeyboardInterrupt:
        error_kind, error = 'interrupted', 'KeyboardInterrupt'
    except Exception as exc:
        error_kind, error = 'infrastructure', f'{type(exc).__name__}: {exc}'
    label = recovery_label(trace, commands, error_kind=error_kind, protocol=p)
    result = dict(group=case['id'], split=case['split'], model=model, requested_delay_s=delay,
        actual_delay_s=actual_delay, state_feature=feature, label=label, error_kind=error_kind, error=error,
        simulation_stepped=True, realtime_claimed=False, command_time_proxy=True,
        uniform_delay_range_certified=False, actuator_internal_state_exactly_matched=False)
    if protocol is not None:result.update(cadence_s=p['cadence_s'],protocol=p)
    atomic_json(directory/'report.json', result)
    atomic_json(directory/'commands.json', commands)
    atomic_json(directory/'trace.json', trace)
    return result


def gather(root):
    plan = read_json(root/'plan.json')
    rows = []
    for directory in sorted((root/'trials').glob('*')):
        attempts = [read_json(p) for p in sorted(directory.glob('attempt_*/report.json'))]
        if attempts:
            rows.append(next((r for r in attempts if r['label']['valid']), attempts[-1]))
    valid, rejected = paired_groups(rows, plan['models'], plan['delays_s'])
    summary = dict(expected_groups=len(plan['cases']), complete_valid_groups=len({r['group'] for r in valid}),
        completed_trials=len(rows), valid_trials=sum(r['label']['valid'] for r in rows),
        qualified_trials=sum(r['label'].get('qualified') is True for r in rows),
        valid_groups_by_split=dict(Counter(r['split'] for r in {r['group']:r for r in valid}.values())),
        invalid_reasons=dict(Counter(r['label']['reason'] for r in rows if not r['label']['valid'])),
        rejected_groups=rejected, full_dataset_complete=len({r['group'] for r in valid}) == len(plan['cases']))
    atomic_json(root/'collection_summary.json', summary)
    atomic_json(root/'paired_rows.json', valid)
    return summary


def collect(root, max_groups=0):
    from ..fly_convex_policy import load_airsim_client
    from ..framework_cpu_affinity import CpuPlacement
    from ..framework_runtime_world import FrameworkWorld
    plan = read_json(root/'plan.json')
    if plan['version'] != VERSION or plan['source_sha256'] != source_hashes():
        raise ValueError('collector source changed; prepare a new dataset directory, keep previous evidence')
    load_airsim_client()
    with CpuPlacement() as placement:
        pool = ControllerPool(model_names=plan['models'])
        if pool.provenance() != plan['provenance']:
            raise ValueError('frozen controller artifacts changed')
        cases = plan['cases'][:max_groups] if max_groups else plan['cases']
        for index, case in enumerate(cases):
            world = FrameworkWorld(case['layout'], case['phase_s'], 1, PROTOCOL['net_width_m'],
                                   task_duration=PROTOCOL['duration_s'])
            try:
                for di, delay in enumerate(plan['delays_s']):
                    order = list(plan['models'])
                    np.random.default_rng(case['route_seed']+di).shuffle(order)
                    for model in order:
                        parent = root/'trials'/f"{case['id']}_d{int(delay*1000):03d}_{model}"
                        attempts = sorted(parent.glob('attempt_*/report.json'))
                        if any(read_json(p)['label']['valid'] for p in attempts):
                            continue
                        for attempt in range(len(attempts)+1, 3):
                            directory = parent/f'attempt_{attempt:03d}'
                            r = run_trial(world, pool, model, delay, case, directory)
                            print(case['id'], model, 'delay', delay, 'attempt', attempt, 'valid', r['label']['valid'],
                                  'margin', r['label'].get('margin'), 'error', r['error'], flush=True)
                            gather(root)
                            if r['error_kind'] in ('infrastructure', 'interrupted'):
                                raise RuntimeError(r['error'])
                            if r['label']['valid']:
                                break
            finally:
                world.hover_all()
                atomic_json(root/f"cleanup_{case['id']}.json", world.close())
    atomic_json(root/'cpu_placement.json', placement.report)
    print(json.dumps(gather(root), indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('prepare', 'collect', 'summarize'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-groups', type=int, default=0)
    args = p.parse_args()
    if args.action == 'prepare':
        prepare(args.output)
    elif args.action == 'collect':
        collect(args.output, args.max_groups)
    else:
        print(json.dumps(gather(args.output), indent=2))


if __name__ == '__main__':
    main()
