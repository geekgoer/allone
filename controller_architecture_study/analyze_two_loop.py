"""Summarize every two-loop repetition, including failures and dropped jobs."""
import json
from pathlib import Path
import numpy as np
from .decision_dataset import atomic_json


def stats(values):
    return dict(count=len(values), mean=float(np.mean(values)), p99=float(np.percentile(values, 99)),
                maximum=float(max(values))) if values else dict(count=0)


def main():
    root = Path('controller_architecture_study/results/two_loop_mixed_speed_20261108_v1')
    output = []
    phases = dict(release_wait=('release_s', 'sample_start_s'),
                  preparation=('sample_start_s', 'feature_ready_s'),
                  queue=('queue_s', 'board_start_s'), board_roundtrip=('board_start_s', 'board_done_s'),
                  command=('command_start_s', 'command_return_s'), response=('release_s', 'command_return_s'))
    for p in sorted(root.glob('*/result.json')):
        result = json.loads(p.read_text(encoding='utf-8'))
        jobs = json.loads((p.parent/'requests.json').read_text(encoding='utf-8'))
        setup = json.loads((p.parent/'setup.json').read_text(encoding='utf-8'))
        for name, vehicle in result['vehicles'].items():
            rows = [r for r in jobs if r['vehicle'] == name]
            delivered = [r for r in rows if r['status'] == 'delivered']
            timing = {key: stats([1000*(r[b]-r[a]) for r in rows if a in r and b in r])
                      for key, (a, b) in phases.items()}
            services = [r['board_timing']['board_service_us'] for r in rows
                        if isinstance(r.get('board_timing'), dict) and 'board_service_us' in r['board_timing']]
            margins = [g['body_margin_m'] for g in vehicle['crossings'] if g['body_margin_m'] is not None]
            output.append(dict(trial=p.parent.name, vehicle=name, requests=len(rows), delivered=len(delivered),
                other_statuses={s: sum(r['status'] == s for r in rows) for s in sorted({r['status'] for r in rows})},
                body_passed=vehicle['body_clearance_passed'], contact=vehicle['contact'], error=vehicle['error'],
                min_crossing_margin_m=min(margins) if margins else None, rmse_m=vehicle.get('tracking_rmse_m'),
                timing_ms=timing, board_service_ms=sum(services)/1000,
                board_service_fraction=sum(services)/1e6/setup['duration_s']))
    atomic_json(root/'per_vehicle_analysis.json', output)
    for r in output:
        print(r['trial'], r['vehicle'], 'delivered', r['delivered'], '/', r['requests'],
              'margin', r['min_crossing_margin_m'], 'R', r['timing_ms']['response'],
              'board_fraction', r['board_service_fraction'])


if __name__ == '__main__':
    main()
