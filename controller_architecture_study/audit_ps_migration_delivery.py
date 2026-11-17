"""Read-only audit of the Ubuntu PS migration delivery; no board or RPC access."""
import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import subprocess


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_rows(path):
    if not path.is_file():
        return []
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def stats(values):
    ordered = sorted(values)
    if not ordered:
        return {'count': 0}
    def quantile(fraction):
        index = fraction * (len(ordered) - 1)
        low = int(index)
        high = min(low + 1, len(ordered) - 1)
        return ordered[low] + (ordered[high] - ordered[low]) * (index - low)
    return dict(count=len(ordered), mean=sum(ordered)/len(ordered),
                p50=quantile(.5), p99=quantile(.99), max=ordered[-1])


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit(delivery, output, planner=None):
    delivery = delivery.resolve()
    output.mkdir(parents=True, exist_ok=False)
    mismatches, missing, checked = [], [], 0
    for line in (delivery/'DELIVERY_FILE_SHA256.txt').read_text(encoding='utf-8').splitlines():
        digest, relative = line.split(None, 1)
        path = (delivery/relative.strip()).resolve()
        if not path.is_relative_to(delivery):
            raise ValueError('Manifest path outside delivery')
        if not path.is_file():
            missing.append(relative)
        elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            mismatches.append(relative)
        checked += 1
    manifest = dict(checked=checked, missing=missing, mismatches=mismatches)
    (output/'manifest_audit.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    if missing or mismatches:
        raise ValueError('Delivery integrity failed; inspect manifest_audit.json')

    with (delivery/'CROSS_STAGE_FINAL_001.csv').open(encoding='utf-8', newline='') as stream:
        declared = list(csv.DictReader(stream))
    assert len(declared) == 17
    runs, skip_rows, accepted_skips, replays = [], [], [], []
    eligible20, target20 = 0, []
    largest_plan = None
    lifecycle_fields = [
        ('calendar_release', 'published_ms'), ('calendar_queued', 'queue_ms'),
        ('calendar_dispatch', 'board_start_ms'), ('calendar_after_dpu', 'board_done_ms'),
        ('calendar_finished', 'command_begin_ms'), ('calendar_finished', 'command_return_ms'),
        ('calendar_finished', 'finished_ms')]
    for entry in declared:
        name = entry['run']
        folder = delivery/name
        rows = read_rows(folder/'requests.jsonl')
        assert len({(r['vehicle'], r['cycle']) for r in rows}) == len(rows), name
        statuses = Counter(r['status'] for r in rows)
        assert dict(statuses) == json.loads(entry['statuses']), name
        models = Counter(str(r['model_id']) for r in rows if r['status'] == 'delivered')
        assert dict(models) == json.loads(entry['actual_models']), name
        deadlines = [r for r in rows if r['status'] == 'delivered' and
                     r['command_return_us'] > min(r['period_deadline_us'], r['actuation_deadline_us'])]
        slot_overruns = [r for r in rows if r.get('slot_return_us', r['finished_us'] if 'finished_us' in r else 0)
                         > r['period_deadline_us']]
        response = stats([(r['command_return_us']-r['planned_release_us'])/1000
                          for r in rows if r['status'] == 'delivered'])
        assert response['count'] == int(entry['release_to_command_proxy_count'])
        assert abs(response['max']-float(entry['release_to_command_proxy_max_ms'])) < 1e-7

        # These logged events are witnesses of calendar changes, not a complete
        # revision trace. Multiple witnesses do not identify one unique cause.
        events = set()
        for row in rows:
            key = '{}:{}'.format(row['vehicle'], row['cycle'])
            for field, stamp in lifecycle_fields:
                value = row.get(field, {}).get(stamp)
                if value is not None:
                    events.add((value*1000, key, stamp))
            if row.get('observe_accepted'):
                events.add((row['analysis_ready_us'], key, 'initial_observation_lower_time_proxy'))
            if row.get('paired_observe_accepted'):
                events.add((row['paired_analysis_ready_us'], key, 'paired_observation_lower_time_proxy'))
        events = sorted(events)
        event_times = [event[0] for event in events]
        plans = read_rows(folder/'plans.jsonl')
        rejected = Counter()
        cpu_values, wall_values = [], []
        for index, row in enumerate(plans):
            if row.get('event') != 'obligation_plan':
                continue
            rejected[row['install']['status']] += 1
            wall_values.append(row['wall_ms'])
            cpu_values.append(row['cpu_ms'])
            if largest_plan is None or row['wall_ms'] > largest_plan['wall_ms']:
                largest_plan = dict(run=name, index=index, wall_ms=row['wall_ms'], cpu_ms=row['cpu_ms'],
                                    install_status=row['install']['status'],
                                    checks=row['decision'].get('checks'), action=row['decision'].get('action'))
            snap, decision = row['snapshot'], row['decision']
            if snap and name == 'pressure20_skip_on_001':
                for permission in snap['permissions']:
                    if permission['eligible']:
                        eligible20 += 1
                        target20.append(permission['valid_until'])
            if not decision.get('accepted') or not decision.get('skipped'):
                continue
            start = snap['snapshot_ms']*1000
            elapsed = row['end_us']/1000-snap['snapshot_ms']
            witnesses = events[bisect_left(event_times, start):bisect_right(event_times, row['end_us'])]
            permission = decision['permission']
            item = dict(run=name, plan_index=index, skipped=decision['skipped'],
                        wall_ms=row['wall_ms'], cpu_ms=row['cpu_ms'],
                        elapsed_from_snapshot_ms=elapsed, commit_by_ms=decision['commit_by'],
                        over_commit_budget=elapsed > decision['commit_by']+1e-8,
                        target_release_remaining_ms=permission['valid_until']-elapsed,
                        install_status=row['install']['status'], checks=decision['checks'],
                        witnessed_lifecycle_changes=len(witnesses),
                        witnessed_command_changes=sum(e[2] in ('command_begin_ms', 'command_return_ms') for e in witnesses),
                        witnessed_releases=sum(e[2] == 'published_ms' for e in witnesses))
            skip_rows.append(item)
            accepted_skips.append((row['wall_ms'], name, index, row))

        final = read_json(folder/'calendar_final.json') if (folder/'calendar_final.json').is_file() else {}
        run = dict(run=name, stage='formal' if name.startswith(('fixed_single_', 'dual_')) else 'pressure',
                   recorded=len(rows), statuses=json.dumps(statuses, sort_keys=True),
                   delivered=statuses['delivered'], command_deadline_misses=len(deadlines),
                   slot_return_overruns=len(slot_overruns), models=json.dumps(models, sort_keys=True),
                   response_p50_ms=response['p50'], response_p99_ms=response['p99'], response_max_ms=response['max'],
                   planned_skips=statuses['planned_skip'],
                   pending_recoveries=len(final.get('pending_recovery', {})),
                   planner_records=len(plans), planner_wall_max_ms=max(wall_values, default=0),
                   planner_cpu_max_ms=max(cpu_values, default=0),
                   install_statuses=json.dumps(rejected, sort_keys=True),
                   requests_sha256=hashlib.sha256((folder/'requests.jsonl').read_bytes()).hexdigest())
        runs.append(run)

    write_csv(output/'runs.csv', runs)
    write_csv(output/'skip_candidates.csv', skip_rows)
    # Preserve a few complete real scheduling inputs for deterministic replay.
    examples = []
    def replay_request(row):
        snap = row['snapshot']
        return dict(command='plan', calendar=snap['calendar'], modes=snap['modes'],
                    options=dict(now=0., decision_ms=snap['config']['decision_ms'],
                                 blocked_until=snap['blocked_until_ms'], permissions=snap['permissions'],
                                 max_checks=8, max_candidates=12, enable_skip=snap['config']['enable_skip']))

    for label, selected in [('fastest', min(accepted_skips)), ('slowest', max(accepted_skips))]:
        _, name, index, row = selected
        request = replay_request(row)
        (output/(label+'_skip_request.json')).write_text(json.dumps(request, indent=2), encoding='utf-8')
        (output/(label+'_skip_original.json')).write_text(json.dumps(row, indent=2), encoding='utf-8')
        examples.append(dict(label=label, run=name, plan_index=index))

    if planner is not None:
        replay_dir = output/'cpp_replay'
        replay_dir.mkdir()
        keys = ['accepted', 'skipped', 'action', 'checks', 'choice', 'order', 'pending', 'commit_by']
        for _, name, index, row in accepted_skips:
            filename = '{}_{}'.format(name, index)
            request_path = replay_dir/(filename+'_request.json')
            request_path.write_text(json.dumps(replay_request(row)), encoding='utf-8')
            result = subprocess.run([str(planner.resolve()), '--request', str(request_path)],
                                    capture_output=True, text=True, check=True)
            actual, original = json.loads(result.stdout), row['decision']
            differences = {k:dict(original=original[k], replay=actual[k])
                           for k in keys if original[k] != actual[k]}
            assert not differences, (name, index, differences)
            (replay_dir/(filename+'_result.json')).write_text(json.dumps(actual), encoding='utf-8')
            replays.append(dict(run=name, index=index, matched=True, keys_checked=keys))
        (output/'cpp_replays.json').write_text(json.dumps(replays, indent=2), encoding='utf-8')

    snapshot_hashes = {}
    for name in ['scheduler/scheduler.cpp', 'scheduler/live_calendar.cpp',
                 'scheduler/scheduler.hpp', 'scheduler/live_calendar.hpp', 'adaptive_runtime.cpp']:
        top = hashlib.sha256((delivery/name).read_bytes()).hexdigest()
        measured = hashlib.sha256((delivery/'build_adaptive_005/source'/name).read_bytes()).hexdigest()
        snapshot_hashes[name] = dict(top_sha256=top, measured_v5_sha256=measured, equal=top == measured)
    summary = dict(scope='Read-only Windows reproduction of delivered PS raw logs and scheduling inputs; no live board/flight',
                   manifest=manifest, formal_delivered=sum(r['delivered'] for r in runs if r['stage']=='formal'),
                   pressure_delivered=sum(r['delivered'] for r in runs if r['stage']=='pressure'),
                   command_deadline_misses=sum(r['command_deadline_misses'] for r in runs),
                   slot_return_overruns=sum(r['slot_return_overruns'] for r in runs),
                   actual_planned_skips=sum(r['planned_skips'] for r in runs),
                   final_pending_recoveries=sum(r['pending_recoveries'] for r in runs),
                   candidate_count=len(skip_rows), candidate_install_statuses=dict(Counter(r['install_status'] for r in skip_rows)),
                   candidate_wall_ms=stats([r['wall_ms'] for r in skip_rows]),
                   candidate_cpu_ms=stats([r['cpu_ms'] for r in skip_rows]),
                   candidate_cpu_wall_ratio=stats([r['cpu_ms']/r['wall_ms'] for r in skip_rows]),
                   candidates_with_commit_time_also_exceeded=sum(r['over_commit_budget'] for r in skip_rows),
                   candidates_with_target_release_also_passed=sum(r['target_release_remaining_ms']<=0 for r in skip_rows),
                   candidates_with_lifecycle_witness=sum(r['witnessed_lifecycle_changes']>0 for r in skip_rows),
                   candidates_with_command_witness=sum(r['witnessed_command_changes']>0 for r in skip_rows),
                   candidates_with_release_witness=sum(r['witnessed_releases']>0 for r in skip_rows),
                   eligible_pressure20=eligible20, eligible_pressure20_target_lead_ms=stats(target20),
                   largest_plan=largest_plan, v5_source_parity=snapshot_hashes, replay_examples=examples,
                   native_replayed_skip_candidates=len(replays))
    assert summary['formal_delivered'] == 6000 and summary['pressure_delivered'] == 12004
    assert summary['candidate_count'] == 89
    assert summary['actual_planned_skips'] == summary['command_deadline_misses'] == 0
    (output/'audit.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('v5_source_parity',)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--delivery', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--planner', type=Path, help='Optional locally built scheduler_parity executable')
    args = parser.parse_args()
    audit(args.delivery, args.output, args.planner)
