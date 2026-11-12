"""Compare the two IP37 revisions, FP32 and MPC on one recorded flight."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from .ip37_shadow import fp32_reference, make_client


def metrics(a, b):
    if not len(a):
        return None
    error = np.abs(a - b)
    return {
        'n': len(error),
        'rmse': np.sqrt(np.mean(error**2, axis=0)).tolist(),
        'p95_abs': np.percentile(error, 95, axis=0).tolist(),
        'max_abs': error.max(axis=0).tolist(),
    }


def compare(run, candidate_bundle, old_bundle, board):
    output = run / 'paired_output_quality.json'
    old_output = run / 'old_on_identical_inputs.json'
    if output.exists() or old_output.exists():
        raise FileExistsError(f'Paired report already exists: {run}')
    meta = json.loads((run / 'report.json').read_text(encoding='utf-8'))
    manifest = json.loads((candidate_bundle / 'manifest.json').read_text(encoding='utf-8'))
    endpoint = meta['ip37_endpoint']
    if endpoint['wire_tag_hex'] != manifest['wire_tag_hex'] or endpoint['model']['numeric_id'] != 11:
        raise ValueError('Recorded endpoint does not match the model-11 candidate')
    rows = json.loads((run / 'ip37_shadow_requests.json').read_text(encoding='utf-8'))
    ready = [r for r in rows if r['status'] == 'ready']
    x = np.asarray([r['raw237'] for r in ready], dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 237 or not np.isfinite(x).all():
        raise ValueError('Expected finite recorded 237D inputs')
    new = np.asarray([r['output'] for r in ready])
    reference = fp32_reference(x, manifest)[11]
    with (run / 'requests.jsonl').open(encoding='utf-8') as stream:
        mpc = {(r['vehicle'], r['cycle']): r['mpc_output'] for line in stream
               if 'mpc_output' in (r := json.loads(line))}
    teacher = np.asarray([mpc[r['vehicle'], r['cycle']] for r in ready])

    client, old_manifest = make_client(old_bundle, board, 5237, 100)
    responses, consecutive_errors = [], 0
    try:
        for feature in x:
            if consecutive_errors >= 3:
                responses.append({'status': 'not_attempted'})
                continue
            try:
                responses.append(client.infer(11, feature.tolist()))
                consecutive_errors = 0
            except (OSError, ValueError) as exc:
                responses.append({'status': 'transport_error', 'error': str(exc)})
                consecutive_errors += 1
    finally:
        client.close()
    status = np.array([str(r['status']) for r in responses])
    old = np.asarray([r['U_raw_physical'] if r['status'] == 0 else [np.nan] * 4 for r in responses])
    seconds = np.asarray([r['cycle'] for r in ready]) * meta['period_ms'] / 1000.
    slices = {}
    for name, included in (
        ('all', np.ones(len(ready), dtype=bool)), ('first_3s', seconds < 3.),
        ('mid_flight_3_to_10s', (seconds >= 3.) & (seconds < 10.)),
        ('old_rejected', status == '3'),
    ):
        common = included & (status == '0')
        slices[name] = {
            'candidate_vs_fp32': metrics(new[included], reference[included]),
            'fp32_vs_mpc': metrics(reference[included], teacher[included]),
            'candidate_vs_mpc': metrics(new[included], teacher[included]),
            'candidate_vs_fp32_on_common': metrics(new[common], reference[common]),
            'old_vs_fp32_on_common': metrics(old[common], reference[common]),
        }
    report = {
        'scope': 'Identical recorded raw237; candidate online in MPC shadow; old sequential replay',
        'environment_id': meta['environment_id'], 'route_index': meta['route_index'],
        'candidate_tag': endpoint['wire_tag_hex'], 'old_tag': old_manifest['wire_tag_hex'],
        'candidate_status_counts': dict(Counter(r['status'] for r in rows)),
        'old_status_counts': dict(Counter(status.tolist())),
        'x7_range_m': [float(x[:, 7].min()), float(x[:, 7].max())], 'slices': slices,
    }
    old_rows = [{'vehicle': r['vehicle'], 'cycle': r['cycle'], **reply}
                for r, reply in zip(ready, responses)]
    old_output.write_text(json.dumps(old_rows, indent=2), encoding='utf-8')
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Paired report: {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--candidate-bundle', type=Path, required=True)
    parser.add_argument('--old-bundle', type=Path, required=True)
    parser.add_argument('--board', default='192.168.77.2')
    args = parser.parse_args()
    compare(args.run, args.candidate_bundle, args.old_bundle, args.board)
