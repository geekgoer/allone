"""Read-only audit of route-A traces. Wall time is not processor demand."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


def stats(values):
    a = np.asarray(values, dtype=float)
    return dict(count=len(a), mean=float(a.mean()), p99=float(np.percentile(a, 99)),
                max=float(a.max())) if len(a) else dict(count=0)


def overlap(intervals, start, end):
    """Union length, so accidental overlap cannot double count resource time."""
    total = 0.; cursor = start
    for left, right in sorted((max(start, a), min(end, b)) for a, b in intervals
                              if b > start and a < end):
        total += max(0., right - max(cursor, left)); cursor = max(cursor, right)
    return total


def audit(root):
    evidence_path = root / 'rt_splitio_stage_20261105/evidence.json'
    evidence = json.loads(evidence_path.read_text())
    result = dict(scope='Archived Windows measurements; no FPGA timing or new flight',
                  timing_units='ms', source_sha256={}, runs=[])
    def read(path, lines=False):
        raw = path.read_bytes()
        result['source_sha256'][str(path)] = hashlib.sha256(raw).hexdigest()
        return [json.loads(x) for x in raw.splitlines()] if lines else json.loads(raw)

    # Predeclared last two batches: 1.75m and 1.60m, both static/dynamic pairs.
    for batch in evidence['runs'][-2:]:
        base = Path(batch['directory'])
        for old in batch['rows']:
            folder = base / old['name']
            requests = read(folder / 'requests.json')
            segments = read(folder / 'segments.json')
            by_job = {}
            for s in segments:
                by_job.setdefault(s['sample_id'], []).append((s['start_s'], s['end_s']))
            all_segments = [(s['start_s'], s['end_s']) for s in segments]
            applied = [r for r in requests if r.get('status') == 'applied']
            stages = {}
            for label, start, end in [
                ('release_to_sample_submit', 'planned_release_s', 'sample_submitted_s'),
                ('sample_handoff', 'sample_submitted_s', 'sample_helper_received_s'),
                ('sample_rpc', 'sample_io_started_s', 'sample_io_finished_s'),
                ('sample_return', 'sample_io_finished_s', 'sample_response_received_s'),
                ('frontend_to_input', 'sample_response_received_s', 'frontend_state_start_s'),
                ('input_wall', 'frontend_state_start_s', 'enqueue_s'),
                ('backend_wall', 'enqueue_s', 'ready_s'),
                ('output_handoff', 'ready_s', 'io_submitted_s'),
                ('command_handoff', 'io_submitted_s', 'command_helper_received_s'),
                ('command_rpc', 'command_helper_received_s', 'rpc_return_s'),
                ('response', 'planned_release_s', 'rpc_return_s')]:
                stages[label] = stats([1000*(r[end]-r[start]) for r in applied if start in r and end in r])
            partition = []
            for r in applied:
                if not all(k in r for k in ('enqueue_s', 'ready_s', 'rpc_return_s')): continue
                a, b = r['enqueue_s'], r['ready_s']
                own = overlap(by_job.get(r['sample_id'], []), a, b)
                busy = overlap(all_segments, a, b)
                partition.append(dict(id=r['sample_id'], own=1000*own,
                    other=1000*max(0., busy-own), nonexecution=1000*max(0., b-a-busy),
                    response=1000*(r['rpc_return_s']-r['planned_release_s'])))
            row = dict(directory=str(folder), commands=len(applied), requests=len(requests),
                status_counts=dict(Counter(r.get('status', 'missing') for r in requests)),
                stages=stages, backend_partition={k:stats([p[k] for p in partition])
                    for k in ('own', 'other', 'nonexecution')},
                tail_commands=sorted(partition, key=lambda r:r['response'], reverse=True)[:5],
                installed_groups=(old.get('planner') or {}).get('installed_groups', 0))
            planner = base / (old['name'] + '_planner')
            if planner.exists():
                decisions = read(planner / 'decisions.jsonl', True)
                barriers = [b for d in decisions for b in d['barriers'].values() if b.get('available')]
                proposals = [r for d in decisions for r in (d.get('proposal') or {}).values()]
                physical = [b for b in barriers if 'physical_deadline_s' in b and 'deadline_s' in b]
                row['planning'] = dict(decisions=len(decisions),
                    reasons=dict(Counter(d['report'].get('reason') for d in decisions)),
                    worker_total_ms=stats([d['total_worker_ms'] for d in decisions]),
                    snapshot_to_result_ms=stats([d['snapshot_to_result_ms'] for d in decisions]),
                    barriers=len(barriers), budget_ms=stats([1000*b['budget_s'] for b in barriers]),
                    physical_deadline_binding=sum(b['physical_deadline_s']<=b['deadline_s']+1e-6 for b in physical),
                    binding_denominator=len(physical),
                    state_age_ms=stats([1000*b['state_age_s'] for b in barriers if 'state_age_s' in b]),
                    proposed_period_ms=stats([1000*r['requested_period_s'] for r in proposals]),
                    nonrenewed_obligation_note='Repeated observations are correlated; binding count is not independent trials')
            result['runs'].append(row)
    result['interpretation_limits'] = [
        'Stage summaries condition on delivered commands; all request status counts are retained separately.',
        'Backend nonexecution includes IPC, dispatch, OS descheduling and unlogged work; it is not all scheduling interference.',
        'DNN service is CPU measured, not PL service. Existing segment_estimates_s include host dispatch.',
        'Physical deadline binding concerns the recorded analyzer under its unverified model bounds.',
        'No hypothetical change to timing predicts a closed-loop collision outcome.']
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    report = audit(Path('controller_architecture_study/results'))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x', encoding='utf-8') as f: json.dump(report, f, indent=2)
    for r in report['runs']:
        print(r['directory'], {k:round(v.get('mean',0),3) for k,v in r['backend_partition'].items()},
              'physical_binding',r.get('planning',{}).get('physical_deadline_binding'))
    print('Report:', a.output)
