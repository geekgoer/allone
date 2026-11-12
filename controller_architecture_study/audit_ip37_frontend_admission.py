"""Audit critical releases and estimate direct rearm opportunities from saved traces."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

from .decision_dataset import atomic_json, sha256
from .dynamic_probe_common import read_json


CRITICAL_S = 4.245  # frozen natural-start target, between T50 and T80 phases
REARM_MS = (0.5, 1., 2.)


def inspect(rows, setup, critical):
    groups = defaultdict(list)
    for row in rows:
        groups[row['vehicle']].append(row)
    origin = setup['origin_s']
    decision = setup.get('decision_budget_ms', 3.)
    critical_rows, blocked, decisions = [], [], []
    counts = Counter()
    for vehicle, jobs in groups.items():
        jobs.sort(key=lambda row: row['cycle'])
        if critical and vehicle == 'Drone1':
            i = min(range(len(jobs)), key=lambda j: abs(jobs[j]['release_s']-origin-CRITICAL_S))
            window = jobs[i:i+2] if abs(jobs[i]['release_s']-origin-CRITICAL_S) <= .045 else ()
            for row in window:
                stamps = ('sample_start_s', 'native_sample_finished_s', 'native_sample_consumed_s',
                          'feature_ready_s', 'queue_s', 'board_start_s', 'board_done_s',
                          'command_return_s', 'finished_s')
                relative = {key: 1000*(row[key]-row['release_s']) for key in stamps if row.get(key) is not None}
                critical_rows.append(dict(cycle=row['cycle'], phase_s=row['release_s']-origin,
                    status=row['status'], period_ms=row['period_ms'], timeline_ms=relative,
                    queue_wait_ms=1000*(row['board_start_s']-row['queue_s'])
                    if row.get('board_start_s') is not None and row.get('queue_s') is not None else None))
        old, history, prior_decision = None, None, None
        for row in jobs:
            published = row.get('published_s', row['release_s'])
            if row['status'] == 'dropped_busy_io':
                earlier = old or {}
                ack, done = earlier.get('command_return_s'), earlier.get('finished_s')
                paired = earlier.get('paired_sample_start_s')
                if ack is not None and done is not None and ack < published <= done and paired is not None:
                    category = 'post_ack_optional_refresh'
                    unchanged = abs(earlier['release_s']+earlier['period_ms']/1000-row['release_s']) < 1e-6
                    headroom = 1000*(published-ack)
                    blocked.append(dict(vehicle=vehicle, cycle=row['cycle'], previous_cycle=earlier['cycle'],
                        phase_s=row['release_s']-origin, paired_refresh_started_at_drop=paired <= published,
                        ack_to_publish_ms=headroom, unchanged_next_release=unchanged,
                        past_max_defers=prior_decision if unchanged else None))
                elif ack is not None and ack >= published:
                    category = 'pre_ack'
                elif done is not None and done >= published:
                    category = 'active_without_ack_or_refresh_marker'
                else:
                    category = 'unresolved'
                counts[category] += 1
            elif row.get('sample_start_s') is not None or row.get('finished_s') is not None:
                old, prior_decision = row, None
                ack, done, paired = (row.get(key) for key in
                                     ('command_return_s', 'finished_s', 'paired_sample_start_s'))
                if ack is not None and done is not None and paired is not None and ack <= paired <= done:
                    remaining = 1000*(row['release_s']-ack)+row['period_ms']
                    # Same causal estimate used by --refresh-when-slack before observing this refresh.
                    prior_decision = remaining <= (history if history is not None else 5.)+decision
                    decisions.append(prior_decision)
                    history = max(history or 0., 1000*(done-paired))
    return dict(critical=critical_rows, busy_by_stage=dict(counts), refresh_blocks=blocked,
                observed_refreshes=len(decisions), past_max_would_defer=sum(decisions))


def run(results, gap_audit, output):
    reports = []
    for entry in read_json(gap_audit)['runs']:
        directory = results/entry['run']
        file = directory/'requests.json'
        if sha256(file) != entry['source_sha256']:
            raise ValueError(f'{file}: changed after gap audit')
        reports.append(dict(run=entry['run'], cohort=entry['cohort'],
            **inspect(read_json(file), read_json(directory/'setup.json'), entry['cohort'] == 'focused')))
    cohorts = {}
    for cohort in ('focused', 'campaign'):
        runs = [r for r in reports if r['cohort'] == cohort]
        busy = Counter()
        for r in runs:
            busy.update(r['busy_by_stage'])
        blocks = [b for r in runs for b in r['refresh_blocks']]
        cohorts[cohort] = dict(runs=len(runs), busy_by_stage=dict(busy),
            observed_refreshes=sum(r['observed_refreshes'] for r in runs),
            past_max_would_defer=sum(r['past_max_would_defer'] for r in runs),
            direct_rearm_opportunities=[dict(reserve_ms=ms,
                always_defer=sum(b['ack_to_publish_ms'] >= ms for b in blocks),
                past_max_guard=sum(b['ack_to_publish_ms'] >= ms and b['past_max_defers'] is True for b in blocks))
                for ms in REARM_MS])
    critical = [row for r in reports for row in r['critical']]
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output/'summary.json', dict(source_sha256=sha256(Path(__file__)), audit_sha256=sha256(gap_audit),
        critical_phase_s=CRITICAL_S, critical_status_counts=dict(Counter(row['status'] for row in critical)),
        maximum_critical_response_ms=max((row['timeline_ms']['command_return_s']
            for row in critical if 'command_return_s' in row['timeline_ms']), default=None),
        maximum_critical_board_queue_ms=max((row['queue_wait_ms'] for row in critical
                                            if row['queue_wait_ms'] is not None), default=None),
        cohorts=cohorts, runs=reports))
    print(cohorts, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--gap-audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.results, args.gap_audit, args.output)
