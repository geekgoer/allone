#!/usr/bin/env python3
"""Read immutable per-stage analyses and export a compact cross-stage CSV."""
import argparse
import csv
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--analyses', nargs='+', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
rows = []
for folder in a.analyses:
    analysis = json.loads((folder / 'analysis.json').read_text())
    for run in analysis['runs']:
        timing = run['timing_ms']
        ident = run['execution_identity']
        counts = run['counts']
        statuses = counts.get('status', {})
        life = run.get('calendar_lifecycle', {})
        vehicles = list(run['vehicles'].values())
        tracker = [v['tracking'] for v in vehicles]
        row = dict(analysis=folder.name, run=Path(run['data']).name,
                   policy=ident.get('policy', 'fixed_single'),
                   audit_passed=run['audit_passed'],
                   observed_complete_flight=run.get('complete_flight_observed'),
                   calendar_timing_acceptance=run.get('calendar_timing_acceptance'),
                   skip_recovery_experiment_validated=run.get('skip_recovery_experiment_validated', False),
                   statuses=json.dumps(statuses, sort_keys=True),
                   actual_models=json.dumps(life.get('actual_inference_model_histogram',
                       {str(ident['model_id']): counts.get('recorded_slots', 0)} if 'model_id' in ident else {}), sort_keys=True),
                   released_periods_ms=json.dumps(life.get('released_period_histogram_ms', {}), sort_keys=True),
                   installed_actions=json.dumps(run.get('planner', {}).get('installed_actions', {}), sort_keys=True),
                   installed_skip_count=life.get('installed_skip_count', 0),
                   observed_planned_skips=life.get('observed_planned_skips', 0),
                   recoveries_on_time=life.get('recoveries_on_time', 0),
                   planner_wall_max_ms=run.get('planner', {}).get('wall_ms', {}).get('max'),
                   full_window_rmse_min_m=min(t['full_window_tracking_rmse_m'] for t in tracker if t.get('full_window_tracking_rmse_m') is not None),
                   full_window_rmse_max_m=max(t['full_window_tracking_rmse_m'] for t in tracker if t.get('full_window_tracking_rmse_m') is not None),
                   sim_wall_ratio_min=min(t['sim_wall_ratio'] for t in tracker if t.get('sim_wall_ratio') is not None))
        for phase in ['input_preparation_quantization', 'dpu_call', 'dpu_output_processing',
                      'local_dpu_service', 'release_to_command_proxy', 'release_to_slot_return']:
            stats = timing.get(phase, {})
            for quantile in ['count', 'p50', 'p95', 'p99', 'max']:
                row[phase + '_' + quantile + ('_ms' if quantile != 'count' else '')] = stats.get(quantile)
        rows.append(row)
with a.out.open('x', newline='', encoding='utf-8') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print('MIGRATION_SUMMARY runs={} out={}'.format(len(rows), a.out))
