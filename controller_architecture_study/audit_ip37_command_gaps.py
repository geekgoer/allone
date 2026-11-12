"""Locate missing-command intervals and the recorded stages of skipped jobs."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

from .decision_dataset import atomic_json, sha256
from .dynamic_probe_common import read_json


def missing_stage(row):
    if row.get('queue_s') is None:
        return 'before_board_queue'
    if row.get('board_start_s') is None:
        return 'queued_without_recorded_dispatch'
    if row.get('board_done_s') is None:
        return 'dispatched_without_recorded_reply'
    return 'after_board_reply'


def inspect(requests, origin):
    groups = defaultdict(list)
    for row in requests:
        groups[row['vehicle']].append(row)
    gaps, stages, statuses, proxies = [], Counter(), Counter(), Counter()
    all_gaps = 0
    for vehicle, rows in groups.items():
        rows.sort(key=lambda row: row['cycle'])
        if len({r['cycle'] for r in rows}) != len(rows):
            raise ValueError(f'duplicate calendar cycles for {vehicle}')
        # Older runs only expose ACK. Keep each interval on one clock/marker.
        field = ('command_submitted_s' if any(r.get('command_submitted_s') is not None for r in rows)
                 else 'command_return_s')
        previous = None
        for i, row in enumerate(rows):
            stamp = row.get(field)
            if stamp is None:
                if field == 'command_submitted_s' and row.get('command_return_s') is not None:
                    raise ValueError('cannot combine incomplete submission and ACK timestamps')
                continue
            proxies[field] += 1
            if previous is not None:
                j, prior = previous
                gap_ms = 1000*(stamp-prior[field])
                skipped = rows[j+1:i]
                if gap_ms <= 0:
                    raise ValueError(f'non-increasing {field} for {vehicle}')
                all_gaps += 1
                if skipped:
                    missing = [dict(cycle=r['cycle'], release_offset_s=r['release_s']-origin,
                                    status=r['status'], stage=missing_stage(r)) for r in skipped]
                    stages.update(r['stage'] for r in missing)
                    statuses.update(r['status'] for r in missing)
                    gaps.append(dict(vehicle=vehicle, proxy=field, gap_ms=gap_ms,
                        start_s=prior[field]-origin, end_s=stamp-origin,
                        first_cycle=prior['cycle'], last_cycle=row['cycle'],
                        first_period_ms=prior['period_ms'], last_period_ms=row['period_ms'], skipped=missing))
            previous = i, row
    return dict(all_bracketed_intervals=all_gaps, marker_counts=dict(proxies),
                skipped_stage_counts=dict(stages), skipped_status_counts=dict(statuses), gaps=gaps)


def run(base, output):
    paths = [(cohort, f) for cohort, roots in (
        ('focused', [*base.glob('ip37_kernel_*'), *base.glob('ip37_native_*')]),
        ('campaign', [base/'ip37_campaign_2h_20261108_v2']))
        for root in roots if root.is_dir() for f in root.rglob('requests.json')]
    reports, exclusions, seen = [], [], set()
    for cohort, file in sorted(paths):
        directory = file.parent
        rel = directory.relative_to(base).as_posix()
        if not all((directory/name).is_file() for name in ('result.json', 'setup.json')):
            exclusions.append((rel, 'missing reports'))
            continue
        result = read_json(directory/'result.json')
        setup = read_json(directory/'setup.json')
        if not result.get('board_used') or setup.get('origin_s') is None:
            exclusions.append((rel, 'board or time origin unavailable'))
            continue
        digest = sha256(file)
        if digest in seen:
            exclusions.append((rel, 'duplicate request bytes'))
            continue
        seen.add(digest)
        try:
            result = inspect(read_json(file), setup['origin_s'])
        except ValueError as exc:
            exclusions.append((rel, str(exc)))
            continue
        reports.append(dict(run=rel, cohort=cohort, source_sha256=digest, **result))
    summary = {}
    for cohort in ('focused', 'campaign'):
        runs = [r for r in reports if r['cohort'] == cohort]
        long = [g for r in runs for g in r['gaps'] if g['gap_ms'] >= 85.]
        counts = Counter(s['stage'] for g in long for s in g['skipped'])
        summary[cohort] = dict(runs=len(runs), skipped_gap_count=sum(len(r['gaps']) for r in runs),
            at_least_85ms=len(long), skipped_in_long_gaps_by_stage=dict(counts),
            longest_ms=max((g['gap_ms'] for g in long), default=None))
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output/'summary.json', dict(cohorts=summary, excluded=exclusions, runs=reports))
    print(summary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.results, args.output)
