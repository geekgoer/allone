"""18 FPGA flights at T=60 ms, target R=25 ms, on frozen scenario S02."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MODEL_MANIFEST = HERE.parent / 'ubuntu_transfer/falcon_current_5ms_12d_six_20260920_202227/package_manifest.json'
PERIOD_MS = 60
TARGET_MS = 25.0
TOLERANCE_MS = 3.0
MAX_MEDIAN_SPREAD_MS = 1.0


def _plan():
    rng = np.random.default_rng(42)
    return [rng.permutation(np.arange(1, 7)).tolist() for _ in range(3)]


def _command(directory, order):
    command = [sys.executable, str(HERE / 'run_fpga_closed_loop.py'),
        '--enable-fpga-actuation', '--fpga-config', str(HERE / 'fpga_current5ms_config.json'),
        '--scenario-set', 'S02', '--model-ids', ','.join(map(str, order)), '--repeats', '1',
        '--period-ms', str(PERIOD_MS), '--actuation-delay-ms', '0', '--reference-profile', 'constant',
        '--target-response-ms', str(TARGET_MS), '--response-tolerance-ms', str(TOLERANCE_MS),
        '--output-dir', str(directory)]
    if (directory / 'closed_loop_manifest.json').exists():
        command.append('--resume')
    return command


def summarize(root, catalog):
    trials = []
    model_latencies = {model_id: [] for model_id in range(1, 7)}
    for block in range(3):
        directory = root / f'repeat_{block:02d}'
        summary_path = directory / 'fpga_closed_loop_summary.json'
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        for trial in summary['trials']:
            model_id = int(trial['model_id'])
            rows = [json.loads(line) for line in (directory / 'trial_logs' / f"{trial['trial_key']}.jsonl").read_text(encoding='utf-8').splitlines() if line.strip()]
            latencies = np.array([(r['command_dispatch_us'] - r['state_acquired_us']) / 1000.0 for r in rows])
            clock_fields_valid = bool(rows and all(
                r.get('target_response_ms') == TARGET_MS and r.get('response_tolerance_ms') == TOLERANCE_MS
                and r.get('model_id') == model_id and r.get('scenario_id') == 'S02'
                and r.get('period_ms') == PERIOD_MS
                and r.get('command_send_started_us', 0) >= r['state_acquired_us'] + round(TARGET_MS * 1000)
                for r in rows))
            missed_target = int(np.sum(np.abs(latencies - TARGET_MS) > TOLERANCE_MS + 1e-9))
            costs = np.array([r['paper_state_error_squared_sum'] for r in rows])
            used = len(costs) // 21 * 21
            window_mean = float(costs[:used].reshape(-1, 21).sum(axis=1).mean()) if used else 0.0
            reported_j = trial.get('J_paper_style')
            metric_ok = bool(window_mean > 0 and reported_j is not None
                and np.isclose(1.0 / window_mean, reported_j, rtol=1e-9, atol=1e-12))
            clean = bool(rows and len(rows) == trial['samples'] and trial['completed']
                and len(trial['crossings']) == 5 and not trial['error'] and not trial['collision']
                and trial['fallback_count'] == trial['deadline_miss_count'] == trial['timeout_count'] == 0
                and metric_ok and all(r['command_source'] == 'fpga_fresh' and not any(r['clip_flags']) for r in rows))
            target_ok = bool(clock_fields_valid and missed_target == 0)
            model_latencies[model_id].extend(latencies.tolist())
            trials.append(dict(block=block, model_id=model_id, trial_key=trial['trial_key'],
                completed=trial['completed'], clean_fpga=clean, response_target_pass=target_ok,
                metric_consistent=metric_ok,
                response_target_violation_cycles=missed_target, samples=len(rows),
                J=trial.get('J_paper_style'), error=trial['error'],
                measured_R_median_ms=float(np.median(latencies)) if len(latencies) else None,
                measured_R_max_ms=float(np.max(latencies)) if len(latencies) else None))
    groups = []
    for model_id, model in catalog.items():
        selected = [t for t in trials if t['model_id'] == model_id]
        values = [t['J'] for t in selected if t['completed'] and t['J'] is not None and np.isfinite(t['J']) and t['J'] > 0]
        latency = model_latencies[model_id]
        groups.append(dict(model_id=model_id, model_name=model['name'],
            c=1.0 / model['test_normalized_mse'], test_normalized_mse=model['test_normalized_mse'],
            trials_recorded=len(selected), clean_and_target_pass=sum(t['clean_fpga'] and t['response_target_pass'] for t in selected),
            J_mean_all_completed=float(np.mean(values)) if values else None,
            J_std_all_completed=float(np.std(values, ddof=1)) if len(values) > 1 else None,
            measured_R_median_ms=float(np.median(latency)) if latency else None,
            measured_R_p99_ms=float(np.percentile(latency, 99)) if latency else None,
            measured_R_max_ms=float(np.max(latency)) if latency else None))
    groups.sort(key=lambda g: g['c'])
    medians = [g['measured_R_median_ms'] for g in groups if g['measured_R_median_ms'] is not None]
    spread = max(medians) - min(medians) if len(medians) == 6 else None
    comparable = bool(len(trials) == 18 and all(g['trials_recorded'] == g['clean_and_target_pass'] == 3 for g in groups)
        and spread is not None and spread <= MAX_MEDIAN_SPREAD_MS)
    trend = None
    if comparable:
        means = [g['J_mean_all_completed'] for g in groups]
        if all(v is not None for v in means):
            trend = dict(model_ids_by_increasing_c=[g['model_id'] for g in groups],
                highest_vs_lowest_c_J_change_percent=100.0 * (means[-1] / means[0] - 1),
                adjacent_mean_J_change_percent=[100.0 * (b / a - 1) for a, b in zip(means, means[1:])],
                note='Descriptive evidence on S02 only, not a significance test or a guarantee for other T/R/environments.')
    return dict(episodes_requested=18, episodes_recorded=len(trials),
        clean_fpga_episodes=sum(t['clean_fpga'] for t in trials),
        clean_and_target_pass=sum(t['clean_fpga'] and t['response_target_pass'] for t in trials),
        target_R_ms=TARGET_MS, tolerance_ms=TOLERANCE_MS,
        maximum_allowed_model_median_spread_ms=MAX_MEDIAN_SPREAD_MS,
        observed_model_median_spread_ms=spread, timing_comparable=comparable,
        trend=trend, models_by_increasing_c=groups, trials=trials,
        interpretation='J means include all completed trials for transparency; do not compare c-axis trends unless timing_comparable=true. No monotonic fit, model selection, or SVR training is performed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--enable-fpga-actuation', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--output-root', type=Path, default=HERE / 'paper_style_sweep/capacity_axis_fixedR25_T60_v1')
    args = parser.parse_args()
    if not args.dry_run and not args.enable_fpga_actuation:
        raise SystemExit('Add --enable-fpga-actuation for flights, or --dry-run to inspect commands.')
    root = args.output_root.resolve()
    catalog = {int(m['model_id']): m for m in json.loads(MODEL_MANIFEST.read_text(encoding='utf-8'))['models']}
    plan = _plan()
    signature = dict(scenario='S02', period_ms=PERIOD_MS, target_R_ms=TARGET_MS, tolerance_ms=TOLERANCE_MS,
        reference_profile='constant_speed_polyline_v1', seed_for_run_order=42, model_order_by_repeat=plan,
        max_model_median_spread_ms=MAX_MEDIAN_SPREAD_MS,
        model_manifest_sha256=hashlib.sha256(MODEL_MANIFEST.read_bytes()).hexdigest())
    if not args.dry_run:
        root.mkdir(parents=True, exist_ok=True)
        manifest = root / 'comparison_manifest.json'
        if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8')) != signature:
            raise ValueError('Comparison configuration changed; use a new output directory.')
        manifest.write_text(json.dumps(signature, indent=2) + '\n', encoding='utf-8')
    for block, order in enumerate(plan):
        directory = root / f'repeat_{block:02d}'
        summary_path = directory / 'fpga_closed_loop_summary.json'
        previous = json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.exists() else {}
        # Do not selectively re-run a recorded failure to make the comparison pass.
        if len(previous.get('trials', [])) == 6:
            print('SKIP recorded repeat:', block, flush=True)
        else:
            command = _command(directory, order)
            print(subprocess.list2cmdline(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, cwd=ROOT, check=True)
        if not args.dry_run:
            report = summarize(root, catalog)
            (root / 'capacity_axis_report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    if not args.dry_run:
        print(json.dumps({key: report[key] for key in ('episodes_recorded', 'clean_and_target_pass', 'timing_comparable', 'trend', 'models_by_increasing_c')}, indent=2), flush=True)
        print('Report:', root / 'capacity_axis_report.json', flush=True)
    else:
        print('Dry run complete; no AirSim or FPGA requests sent.', flush=True)


if __name__ == '__main__':
    main()
