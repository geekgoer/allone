"""Resumable eight-environment FPGA collection. Does not train or monitor DNNs."""
from pathlib import Path
import argparse
import os
import time
import json
import hashlib

import numpy as np

import experiment as exp
from airsim_interface import require_airsim
from config import load_calibration
from guarded_udp import GuardedFalcon12Client as Falcon12FpgaUdpClient
from initial_state_generator import load_initial_states
from run_fpga_closed_loop import _run_trial
from run_fpga_shadow import _load_shadow_config, _model_table, _distribution


def profile(root, config, catalog):
    path = root / 'latency_profile.json'
    if path.exists():
        return exp.read_json(path)
    network = exp.read_json(exp.resolve(config['fpga_config']))
    package = exp.resolve(config['model_manifest']).parent
    models = []
    with Falcon12FpgaUdpClient(host=network['host'], port=network['port'],
            timeout_ms=config['profile_udp_timeout_ms'], retries=0,
            input_scales=config.get('input_quantization_scales')) as udp:
        for model in catalog['models']:
            vectors = package / 'models' / model['name'] / 'export/golden_vectors.npz'
            with np.load(vectors, allow_pickle=False) as archive:
                inputs = archive['X_normalized'][:config['profile_requests_per_model']].astype(np.float32)
            if len(inputs) != config['profile_requests_per_model']:
                raise ValueError('Not enough frozen inputs for latency profiling')
            warm = udp.warmup(inputs[0], count=config['profile_warmup_requests'], model_id=model['model_id'])
            results = [udp.infer_normalized(x, model_id=model['model_id']) for x in inputs]
            if any(not r.fresh for r in warm + results):
                raise RuntimeError('UDP profiling failed; fix the board/network before any flights')
            item = dict(model_id=model['model_id'], name=model['name'], vectors_sha256=exp.sha256(vectors),
                rtt_us=[r.rtt_us for r in results], board_dpu_plus_io_us=[r.dpu_time_us for r in results])
            item['rtt_distribution_us'] = _distribution(item['rtt_us'])
            models.append(item)
            print(f"PROFILE model={model['model_id']} RTT max={max(item['rtt_us'])/1000:.3f} ms", flush=True)
        report = dict(endpoint=dict(host=network['host'], port=network['port']), models=models,
            client_stats=dict(udp.stats), definition='isolated UDP RTT, not full host-loop R and not proven WCET',
            host_guard_ms=config['profile_guard_ms'])
    floor = max(max(m['rtt_us']) for m in models) / 1000 + config['profile_guard_ms']
    report['empirical_target_floor_ms'] = floor
    if floor >= min(config['target_responses_ms']):
        exp.write_json(root / 'latency_profile_rejected.json', report)
        raise RuntimeError(f'Observed UDP RTT plus host guard is {floor:.3f} ms, too close to the lowest target R. Fix latency or choose a new configuration/output root.')
    exp.write_json(path, report)
    return report


def next_attempt(directory):
    previous = []
    for path in directory.glob('attempt_*'):
        previous.append(int(path.name.split('.')[0].split('_')[-1]))
    return max(previous, default=-1) + 1


def progress(root, plan, rows):
    recorded = len(rows)
    seconds = sum(r['wall_seconds'] for r in rows.values()) / recorded if recorded else None
    payload = dict(planned=len(plan), recorded=recorded, pending=len(plan)-recorded,
        completed_routes=sum(r['completed'] for r in rows.values()),
        collisions=sum(r['outcome'] == 'collision' for r in rows.values()),
        missed_gates=sum(r['outcome'] == 'missed_gate' for r in rows.values()),
        mission_timeouts=sum(r['outcome'] == 'mission_timeout' for r in rows.values()),
        input_range_exceeded=sum(r['outcome'] == 'input_range_exceeded' for r in rows.values()),
        valid_success_targets=sum(r['valid_for_success_svr'] for r in rows.values()),
        valid_failure_zero_targets=sum(r['valid_for_failure_zero_svr'] for r in rows.values()),
        measured_seconds_per_trial=seconds,
        estimated_remaining_hours=seconds * (len(plan)-recorded) / 3600 if seconds else None,
        status='complete' if recorded == len(plan) else 'partial')
    exp.write_json(root / 'progress.json', payload)
    return payload


def collect(root, config, benchmark, catalog, plan, limit):
    by_id = {int(m['model_id']): m for m in catalog['models']}
    scenario_map = {s['id']: s for s in benchmark['scenarios']}
    rows = {}
    for job in plan:
        result = root / 'trials' / job['id'] / 'result.json'
        if result.exists():
            record = exp.read_json(result)
            if record['job'] != job:
                raise ValueError('Stored job disagrees with frozen plan')
            rows[job['id']] = exp.outcome_row(record, by_id[job['model_id']])
    progress(root, plan, rows)
    if len(rows) == len(plan):
        return progress(root, plan, rows)
    profile(root, config, catalog)
    airsim = require_airsim()
    client = airsim.MultirotorClient(timeout_value=config['rpc_timeout_seconds'])
    client.confirmConnection()
    if 'Drone1' not in client.listVehicles():
        raise ValueError('This frozen experiment requires Drone1')
    settings = json.loads(client.getSettingsString())
    if float(settings.get('ClockSpeed', 1.0)) != 1.0:
        raise ValueError('This wall-clock T/R protocol requires AirSim ClockSpeed=1')
    runtime = dict(clock_speed=settings.get('ClockSpeed', 1.0),
        settings_sha256=hashlib.sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest(),
        client_version=client.getClientVersion(), server_version=client.getServerVersion())
    runtime_path = root / 'runtime_identity.json'
    if runtime_path.exists() and exp.read_json(runtime_path) != runtime:
        raise ValueError('AirSim settings/version changed; use a new experiment output root')
    exp.write_json(runtime_path, runtime)
    models = {m['model_id']: m for m in _model_table(benchmark)}
    states = {s.initial_state_id: s for s in load_initial_states(root / 'initial_states.json')}
    network = _load_shadow_config(exp.resolve(config['fpga_config']))
    calibration = load_calibration()
    new = 0
    consecutive_timing_invalid = 0
    with Falcon12FpgaUdpClient(host=network['host'], port=network['port'],
            timeout_ms=network['udp_timeout_ms'], retries=0,
            input_scales=config.get('input_quantization_scales')) as udp:
        for job in plan:
            if job['id'] in rows:
                continue
            if limit is not None and new >= limit:
                break
            directory = root / 'trials' / job['id']
            directory.mkdir(parents=True, exist_ok=True)
            attempt = next_attempt(directory)
            raw_path = directory / f'attempt_{attempt:04d}.jsonl'
            started = time.perf_counter()
            scenario = scenario_map[job['scenario_id']]
            with raw_path.open('x', encoding='utf-8') as handle:
                trial = _run_trial(client=client, airsim=airsim, scenario=scenario,
                    repeat_index=job['variant'], benchmark=benchmark, model=models[job['model_id']],
                    udp_client=udp, shadow=network, period_s=job['period_ms']/1000, actuation_delay_s=0,
                    calibration=calibration, vehicle_name='Drone1', initial_states=states,
                    output_handle=handle, experiment_id=job['id'], keep_gates=False,
                    reference_profile='constant', target_response_ms=job['target_R_ms'],
                    response_tolerance_ms=config['response_tolerance_ms'],
                    max_episode_seconds=config['max_episode_seconds'], cycle_log_stride=0,
                    reset_after_trial=True)
                handle.flush()
                os.fsync(handle.fileno())
            record = dict(job=job, scenario=scenario, trial=trial,
                raw_log=str(raw_path.relative_to(root)), wall_seconds=time.perf_counter()-started)
            exp.write_json(directory / f'attempt_{attempt:04d}.summary.json', record)
            row = exp.outcome_row(record, by_id[job['model_id']])
            if row['outcome'] == 'infrastructure_error' or trial.get('timeout_count', 0) or trial.get('cleanup_error'):
                raise RuntimeError(f"Infrastructure failure in {job['id']}; attempt retained, not a collision-zero label. Fix AirSim/UDP and rerun to resume. {trial.get('error')} {trial.get('cleanup_error')}")
            exp.write_json(directory / 'result.json', record)
            rows[job['id']] = row
            new += 1
            p = progress(root, plan, rows)
            eta = p['estimated_remaining_hours']
            print(f"{p['recorded']}/{p['planned']} {job['id']} outcome={row['outcome']} gates={row['crossings']}/5 Rmax={row['R_max_ms']} J={row['J_failure_zero']} ETA={eta:.2f}h", flush=True)
            consecutive_timing_invalid = consecutive_timing_invalid + 1 if trial.get('deadline_miss_count', 0) else 0
            if consecutive_timing_invalid >= 3:
                raise RuntimeError('Three consecutive timing-invalid trials were recorded. Check host/AirSim load before resuming; these records are not silently retried.')
    return progress(root, plan, rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=exp.HERE / 'collection_config_moderate_v2.json')
    parser.add_argument('--output-root', type=Path, default=exp.HERE / 'results/moderate_v2')
    parser.add_argument('--enable-fpga-actuation', action='store_true')
    parser.add_argument('--profile-only', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--limit', type=int, default=None, help='maximum new trials this invocation; rerun to continue')
    args = parser.parse_args()
    config, _, _ = exp.load_experiment(args.config)
    plan = exp.make_plan(config)
    if args.limit is not None and args.limit < 1:
        raise ValueError('--limit must be positive')
    if args.dry_run:
        print(dict(environments=len(config['environments']), models=6,
            periods=config['periods_ms'], target_responses=config['target_responses_ms'],
            variants=config['variants'], flights=len(plan), coordinates_per_cube=len(config['periods_ms'])*len(config['target_responses_ms'])*6,
            estimated_hours_at_18_to_32_seconds_per_trial=[round(len(plan)*s/3600,1) for s in (18,32)],
            first_jobs=plan[:3]))
        return
    if not args.profile_only and not args.enable_fpga_actuation:
        raise SystemExit('Add --enable-fpga-actuation for AirSim flights; --profile-only sends UDP but does not fly.')
    root = args.output_root.resolve()
    config, benchmark, catalog, plan = exp.freeze(root, args.config)
    if args.profile_only:
        profile(root, config, catalog)
        return
    try:
        collect(root, config, benchmark, catalog, plan, args.limit)
    except KeyboardInterrupt:
        print('\nInterrupted. Recorded results and partial attempts are retained; rerun the same command.', flush=True)
    finally:
        from export_and_plot import export_dataset
        export_dataset(root)
    print('Saved progress and datasets:', root, flush=True)


if __name__ == '__main__':
    main()
