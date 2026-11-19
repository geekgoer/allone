"""Frozen geometry, trial plan, and outcome semantics for the eight-env extension."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import math
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LEGACY = ROOT / 'experiments/paper_track_p_12d_5ms_v1/fpga_shadow_historical5ms'
for directory in (ROOT, LEGACY):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_experiment(config_path):
    config = read_json(config_path)
    if config['format_version'] != 1 or len(config['environments']) < 8:
        raise ValueError('Expected version 1 and at least eight frozen environments')
    if config['variants'] != 2 or len(config['initial_state_ids']) != 2:
        raise ValueError('This protocol uses two fixed trajectory/initial-state variants')
    for key in ('profile_requests_per_model', 'profile_warmup_requests', 'profile_guard_ms',
                'profile_udp_timeout_ms', 'response_tolerance_ms', 'rpc_timeout_seconds', 'max_episode_seconds'):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    for values in (config['periods_ms'], config['target_responses_ms']):
        if len(values) != len(set(values)) or any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError('Timing coordinates must be unique, finite and positive')
    if max(config['target_responses_ms']) + config['response_tolerance_ms'] >= min(config['periods_ms']):
        raise ValueError('The rectangular cube must satisfy max R + tolerance < min T')
    ids = [e['id'] for e in config['environments']]
    if len(ids) != len(set(ids)):
        raise ValueError('Environment IDs must be unique')
    for env in config['environments']:
        values = [env[k] for k in ('x_spacing_m', 'lateral_step_m', 'vertical_step_m')]
        if not np.isfinite(values).all() or values[0] <= 0 or min(values[1:]) < 0:
            raise ValueError('Invalid gate geometry')
    benchmark = read_json(resolve(config['benchmark_config']))
    catalog = read_json(resolve(config['model_manifest']))
    if sorted(int(m['model_id']) for m in catalog['models']) != list(range(1, 7)):
        raise ValueError('Exactly the six deployed model IDs are required')
    return config, benchmark, catalog


def scenarios(config):
    result = []
    for env in config['environments']:
        dx, dy, dz = [float(env[k]) for k in ('x_spacing_m', 'lateral_step_m', 'vertical_step_m')]
        for variant in range(2):
            sign = 1 if variant == 0 else -1
            centers = [[config['x_start'] + dx * i, sign * dy * (i % 2),
                        config['base_z'] - dz * (i % 2)] for i in range(config['gate_count'])]
            steps = np.abs(np.diff(np.asarray(centers)[:, 1:], axis=0)).sum(axis=1)
            result.append(dict(id=f"{env['id']}_v{variant}", environment_id=env['id'],
                environment_name=env['name'], variant=variant,
                epsilon=float(steps.mean()), initial_state_id=config['initial_state_ids'][variant],
                seed=config['plan_seed'] + variant, gate_centers=centers,
                x_spacing_m=dx, lateral_step_m=dy, vertical_step_m=dz,
                transverse_to_forward_ratio=float(math.hypot(dy, dz) / dx),
                gate_overrides=dict(approach_distance=min(2.0, .35 * dx), exit_distance=min(1.0, .2 * dx),
                                    completion_radius=min(.8, .8 * min(1.0, .2 * dx)))))
    return result


def make_plan(config):
    rng = np.random.default_rng(config['plan_seed'])
    coordinates = [(t, r) for t in config['periods_ms'] for r in config['target_responses_ms']]
    anchor = (min(config['periods_ms']), min(config['target_responses_ms']))
    others = [x for x in coordinates if x != anchor]
    rng.shuffle(others)
    plan = []
    for t, r in [anchor] + others:
        block = [dict(scenario_id=s['id'], environment_id=s['environment_id'], variant=s['variant'],
                      model_id=m, period_ms=t, target_R_ms=r)
                 for s in scenarios(config) for m in range(1, 7)]
        rng.shuffle(block)
        for job in block:
            job['id'] = f"{job['scenario_id']}_m{job['model_id']}_T{t:g}_R{r:g}"
        plan.extend(block)
    return plan


def freeze(root, config_path):
    from config import load_calibration

    config, source_benchmark, catalog = load_experiment(config_path)
    package = resolve(config['model_manifest']).parent
    sources = [Path(config_path).resolve(), resolve(config['benchmark_config']),
        resolve(config['model_manifest']), resolve(source_benchmark['initial_states']),
        HERE / 'experiment.py', HERE / 'collect.py', HERE / 'guarded_udp.py', LEGACY / 'run_fpga_closed_loop.py',
        LEGACY / 'run_fpga_shadow.py', ROOT / 'airsim_interface.py', ROOT / 'config.py', ROOT / 'dnn_model_utils.py',
        ROOT / 'gate_route.py', ROOT / 'reference_generator.py', ROOT / 'control_adapter.py', ROOT / 'state_adapter.py',
        ROOT / 'gate_visualization.py', ROOT / 'mpc_controller.py', ROOT / 'falcon12_fpga_udp_client.py',
        resolve(config['fpga_config'])]
    sources += [package / 'models' / m['name'] / 'export/normalization.json' for m in catalog['models']]
    identity = dict(config=config, source_sha256={str(p.resolve()): sha256(p) for p in sources},
        actuator_calibration=asdict(load_calibration()),
        input_range_policy='abort_trial_no_clipping_v1; current5ms scale32 unless explicitly configured',
        execution_mode='one whole DPU task, nonpreemptive; constant send-release delay injection',
        performance_definition='original J only on completed pure-FPGA routes; separate explicit failure_zero_v1 label',
        reference_profile='constant_speed_polyline_v1')
    manifest_path = root / 'experiment_manifest.json'
    if manifest_path.exists():
        if read_json(manifest_path) != identity:
            raise ValueError('Configuration/source fingerprint changed. Preserve this run; choose a new output root.')
        for name, digest in read_json(root / 'frozen_files_sha256.json').items():
            if sha256(root / name) != digest:
                raise ValueError(f'Frozen experiment file changed: {name}')
        return config, read_json(root / 'benchmark.json'), catalog, read_json(root / 'plan.json')
    root.mkdir(parents=True, exist_ok=True)
    initial = deepcopy(read_json(resolve(source_benchmark['initial_states'])))
    dz = config['base_z'] - source_benchmark['gate']['base_z']
    for state in initial:
        state['position'][2] += dz
    write_json(root / 'initial_states.json', initial)
    benchmark = deepcopy(source_benchmark)
    benchmark['benchmark_name'] = 'falcon_8env_safety_stress_extension_v1'
    benchmark['purpose'] = 'Frozen broad geometry extension, not the original paper environments'
    benchmark['initial_states'] = str((root / 'initial_states.json').resolve())
    benchmark['gate'].update(count=config['gate_count'], base_z=config['base_z'])
    benchmark['scenarios'] = scenarios(config)
    benchmark['scenario_sets'] = {'full': [s['id'] for s in benchmark['scenarios']]}
    for model in benchmark['models']:
        model['normalization'] = str((package / 'models' / model['name'] / 'export/normalization.json').resolve())
    plan = make_plan(config)
    write_json(root / 'benchmark.json', benchmark)
    write_json(root / 'model_manifest.json', catalog)
    write_json(root / 'plan.json', plan)
    write_json(root / 'frozen_files_sha256.json', {name: sha256(root / name) for name in
        ('benchmark.json', 'initial_states.json', 'model_manifest.json', 'plan.json')})
    write_json(manifest_path, identity)
    return config, benchmark, catalog, plan


def failure_kind(trial):
    error = trial.get('error') or ''
    if trial.get('input_range_violation') or error.startswith('InputRangeError:'):
        return 'input_range_exceeded'
    if trial.get('collision'):
        return 'collision'
    if error.startswith('GatePassageError:'):
        return 'missed_gate'
    if error.startswith(('TimeoutError: gate ', 'TimeoutError: episode ')):
        return 'mission_timeout'
    if trial.get('completed') and not error:
        return 'success'
    return 'infrastructure_error'


def outcome_row(record, model):
    trial, job = record['trial'], record['job']
    error = trial.get('error') or ''
    kind = failure_kind(trial)
    physical = kind in ('collision', 'missed_gate', 'mission_timeout')
    pure = bool(trial.get('samples', 0) > 0 and not trial.get('fallback_count', 0)
                and not trial.get('timeout_count', 0) and not trial.get('deadline_miss_count', 0))
    r = trial.get('measured_R_max_ms')
    timing = r is not None and math.isfinite(r) and 0 < r < job['period_ms']
    j = trial.get('J_paper_style')
    valid_j = j is not None and math.isfinite(j) and j > 0
    success = bool(trial.get('completed') and not error and not trial.get('collision'))
    valid_success = success and pure and timing and valid_j
    valid_failure = physical and pure and timing and not success
    scenario = record['scenario']
    latency = np.asarray(trial.get('state_to_command_dispatch_us', []), dtype=float) / 1000.0
    return dict(record_id=job['id'], environment_id=job['environment_id'], scenario_id=job['scenario_id'],
        variant=job['variant'], model_id=job['model_id'], model_name=model['name'],
        beta_test_normalized_mse=model['test_normalized_mse'], c=1.0/model['test_normalized_mse'],
        period_ms=job['period_ms'], target_R_ms=job['target_R_ms'], R_max_ms=r,
        R_mean_ms=float(latency.mean()) if latency.size else None,
        R_median_ms=float(np.median(latency)) if latency.size else None,
        R_std_ms=float(latency.std()) if latency.size else None,
        R_p95_ms=trial.get('measured_R_p95_ms'), R_p99_ms=trial.get('measured_R_p99_ms'),
        epsilon_transverse_m=scenario['epsilon'], x_spacing_m=scenario['x_spacing_m'],
        lateral_step_m=scenario['lateral_step_m'], vertical_step_m=scenario['vertical_step_m'],
        outcome=kind, completed=success, collision=bool(trial.get('collision')),
        collision_object=trial.get('collision'), crossings=len(trial.get('crossings', [])),
        error=error, fallback_count=trial.get('fallback_count', 0), timeout_count=trial.get('timeout_count', 0),
        deadline_miss_count=trial.get('deadline_miss_count', 0),
        response_target_violation_count=trial.get('response_target_violation_count', 0),
        input_range_violation=trial.get('input_range_violation'),
        partial_J_diagnostic=j, J_paper_style_success=j if valid_success else None,
        J_failure_zero=(j if valid_success else 0.0 if valid_failure else None),
        valid_for_success_svr=valid_success, valid_for_failure_zero_svr=valid_success or valid_failure,
        wall_seconds=record['wall_seconds'], raw_log=record['raw_log'],
        cleanup_error=trial.get('cleanup_error'),
        execution_mode='whole_model_nonpreemptive')
