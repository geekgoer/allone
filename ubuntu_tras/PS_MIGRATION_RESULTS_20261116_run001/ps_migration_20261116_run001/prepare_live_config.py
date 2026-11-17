#!/usr/bin/env python3
"""Freeze an empirical PS calendar profile, never retrain or select on final480."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bounded_ms(values, factor):
    if not values or any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError('Need positive actual measured phase costs')
    return math.ceil(max(values) * factor * 1000) / 1000


def build(bundle, local, runs, bench, factor):
    package = bundle / 'reference_assets/deployment'
    manifest = load(package / 'payload_manifest.json')
    for name, sha in manifest['files'].items():
        path = (package / name).resolve()
        if package.resolve() not in path.parents or digest(path) != sha:
            raise ValueError('Frozen deployment hash mismatch: ' + name)
    registry = load(package / 'dpu_model_registry.json')
    profile = {r['model_id']: r for r in load(local / 'cost_profile.json')}
    identity = load(local / 'actual_identity.json')
    if identity['wire_tag_hex'] != registry['wire_tag_hex']:
        raise ValueError('Local DPU and model registry identity mismatch')
    quality = load(bundle / 'evidence/release001/results/release001_board480/summary.json')
    checks = {r['model_id']: r for r in quality['models']}
    golden = load(local / 'golden_summary.json')
    if len(golden) != 18 or any(r['status'] != 'PASS' or r['rows'] != 32 for r in golden):
        raise ValueError('Missing all-model local golden proof')
    modes = []
    for item in registry['models']:
        numeric_id = item['numeric_id']
        row = profile[numeric_id]
        if row['service_us']['count'] < 300:
            raise ValueError('Missing local model cost repetitions')
        modes.append(dict(numeric_id=numeric_id, name=item['id'],
                          service_ms=bounded_ms([row['service_us']['max']/1000], factor),
                          selection_mse=item['train_selection']['teacher_std_mse'],
                          check_mse=checks[numeric_id]['board_vs_teacher']['standardized_mse']))
    if sorted(m['numeric_id'] for m in modes) != list(range(1, 19)):
        raise ValueError('Expected exactly 18 model identities')
    prep, post, rearm, sources = [], [], [], []
    for run in runs:
        summary = load(run / 'summary.json')
        if summary.get('interrupted') or summary.get('journal_failed') or summary.get('early_stop') or (run/'failure.json').exists():
            raise ValueError('Cannot derive profile from incomplete/failed run: ' + str(run))
        rows = [json.loads(line) for line in (run/'requests.jsonl').read_text().splitlines() if line.strip()]
        for row in rows:
            if row['status'] != 'delivered' or row.get('error'):
                raise ValueError('Profile source must be a clean fixed trial')
            prep.append((row['feature_ready_us']-row['sample_start_us'])/1000)
            post.append((row['command_return_us']-row['output_ready_us'])/1000)
            rearm.append((row['slot_return_us']-row['command_return_us'])/1000)
        sources.append(dict(run=str(run.resolve()), rows=len(rows),
                            requests_sha256=digest(run/'requests.jsonl')))
    with bench.open() as stream:
        measured = [float(r['elapsed_ms']) for r in csv.DictReader(stream)]
    if len(measured) < 60:
        raise ValueError('Need repeated actual ARM calendar planner benchmark')
    costs = dict(prep_ms=bounded_ms(prep, factor), post_ms=bounded_ms(post, factor),
                 rearm_ms=bounded_ms(rearm, factor),
                 decision_ms=max(3., bounded_ms(measured, factor)))
    return dict(schema='ps-obligation-config/v1', names=['Drone1', 'Drone2'], origin_ms=0.,
                model_id=11, candidate_ids=[5, 10, 13, 8, 11], period_ms=50.,
                periods_ms=[20., 30., 40., 50., 60., 80.], fresh_ms=80., maximum_update_ms=160.,
                planner_interval_ms=10., enable_skip=False, adaptive=True,
                command_reserve_ms=costs['post_ms'], modes=modes,
                wire_tag_hex=registry['wire_tag_hex'], freeze_sha256=registry['freeze_sha256'],
                normalization_sha256=registry['normalization_sha256'],
                empirical_factor=factor,
                profile_scope='Local serialized DNNDK wall service + measured PS/AirSim RPC envelopes; NOT certified WCET',
                period_scope='Original exploratory supported periods; NOT a certified period-admissibility map',
                selection_scope='Unchanged frozen TRAIN512 teacher_std_mse ONLY; final480 check_mse descriptive ONLY',
                provenance=dict(local_profile=str(local.resolve()),
                                local_profile_sha256=digest(local/'cost_profile.json'),
                                registry_sha256=digest(package/'dpu_model_registry.json'),
                                raw_fixed_runs=sources, planner_benchmark=str(bench.resolve()),
                                planner_benchmark_sha256=digest(bench),
                                measured_prep_max_ms=max(prep), measured_post_max_ms=max(post),
                                measured_rearm_max_ms=max(rearm), measured_planner_max_ms=max(measured)), **costs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--local', type=Path, required=True)
    parser.add_argument('--fixed-runs', type=Path, nargs='+', required=True)
    parser.add_argument('--planner-bench', type=Path, required=True)
    parser.add_argument('--factor', type=float, default=1.25)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.factor) or args.factor < 1.:
        parser.error('factor must be finite >=1')
    result = build(args.bundle, args.local, args.fixed_runs, args.planner_bench, args.factor)
    # Exclusive creation: profile selection must not mutate a running condition.
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print('PS_PROFILE_FROZEN models={} prep_ms={} post_ms={} rearm_ms={} decision_ms={}'.format(
        len(result['modes']), result['prep_ms'], result['post_ms'], result['rearm_ms'], result['decision_ms']))


if __name__ == '__main__':
    main()
