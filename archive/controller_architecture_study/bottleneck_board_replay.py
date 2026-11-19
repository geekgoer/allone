"""Audit AXU3EGB run001 and replay its measured single-worker service."""
import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import replace
from hashlib import file_digest
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from .bottleneck_abc import abc_repair
from .bottleneck_joint import Loop, Mode, Option, Platform, Scenario, simulate


PERIOD_NS = 50_000_000
SOURCES = {
    'single': 1, 'open_1_staggered': 1, 'open_6_staggered': 6,
    'open_12_staggered': 12, 'open_6_synchronized': 6, 'open_12_synchronized': 12,
}
STAMPS = (
    'release_ns', 'enqueue_ns', 'prep_start_ns', 'prep_end_ns', 'submit_ns',
    'done_ns', 'post_start_ns', 'output_ready_ns',
)
METRICS = ('release_jitter', 'wait', 'prep', 'dispatch_gap', 'dpu_call',
           'return_gap', 'post', 'service', 'response')
DERIVED = ('release_jitter_ns', 'software_queue_wait_ns', 'prep_wall_ns',
           'prep_to_submit_gap_ns', 'dpu_call_ns', 'done_to_post_gap_ns',
           'post_wall_ns', 'service_ns', 'response_ns')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def load_board(root):
    root = root.resolve()
    hashes = {}
    for line in (root / 'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        expected, name = line.split(maxsplit=1)
        path = (root / name).resolve()
        require(path.is_relative_to(root) and digest(path) == expected, f'Bad hash: {name}')
        hashes[name] = expected
    require({'requests.csv', 'summary.json', 'environment.json', 'inputs/golden.txt',
             'output_check.json', 'board_capture/formal/process_meta.txt'} <= hashes.keys(),
            'Missing hashes for formal data')
    meta = dict(line.split('=', 1) for line in
                (root / 'board_capture/formal/process_meta.txt').read_text().splitlines())
    require(meta['backend'] == 'REAL_DNNDK_DPU', 'Expected real DPU run')
    env = json.loads((root / 'environment.json').read_text(encoding='utf-8'))
    summary = json.loads((root / 'summary.json').read_text(encoding='utf-8'))
    tolerance = json.loads((root / 'output_check.json').read_text())['int8_tolerance']
    golden = np.loadtxt(root / 'inputs/golden.txt')
    require(golden.shape == (32, 245), 'Unexpected golden shape')
    runs = defaultdict(list)
    with (root / 'requests.csv').open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            request = row['request_id']
            seq, source, sample = (int(row[k]) for k in ('sequence', 'source_id', 'input_id'))
            condition, round_id = row['condition'], int(row['round'])
            require(condition in SOURCES and round_id in (1, 2, 3), request)
            n = SOURCES[condition]
            tick = seq // n
            require(row['run_id'] == f'{condition}_r{round_id}' and
                    request == f"{row['run_id']}:{seq}" and source == seq % n and
                    sample == (tick + source) % 32, request)
            require(row['model_id'] == row['actual_model_id'] == meta['model_id'] and
                    row['status'] == 'success' and row['completed'] == row['output_valid'] == '1' and
                    row['deadline_miss'] == '0', request)
            times = [int(row[k]) for k in STAMPS]
            require(all(a <= b for a, b in zip(times, times[1:])), request)
            require(all(not row[k] for k in ('hw_start_ns', 'hw_end_ns', 'hw_duration_ns',
                                             'feature_start_ns', 'feature_end_ns')), request)
            release, enqueue, start, _, _, _, _, ready = times
            if condition != 'single':
                require(int(row['period_ns']) == PERIOD_NS and
                        int(row['deadline_ns']) == release + PERIOD_NS and ready <= release + PERIOD_NS,
                        request)
            else:
                require(not row['period_ns'] and not row['deadline_ns'], request)
            output = np.array([float(row[f'u{i}']) for i in range(4)])
            require(np.max(np.abs(output - golden[sample, 237:241])) <= tolerance, request)
            stages = [b - a for a, b in zip(times, times[1:])] + [ready - start, ready - release]
            require(stages == [int(row[key]) for key in DERIVED], request)
            runs[(condition, round_id)].append((seq, release, enqueue, start, ready, stages))
    require(len(runs) == 18 and sum(map(len, runs.values())) == summary['formal_request_records'] == 69600,
            'Unexpected run or request count')
    for (condition, _), rows in runs.items():
        rows.sort()
        n = SOURCES[condition]
        expected = 1000 if condition == 'single' else 600 * n
        epoch = rows[0][1]
        prev_ready = 0
        require(len(rows) == expected, condition)
        for i, (seq, release, enqueue, start, ready, _) in enumerate(rows):
            require(i == seq and prev_ready <= start and (i == 0 or rows[i-1][2] <= enqueue), condition)
            if condition != 'single':
                phase = (i % n) * PERIOD_NS // n if condition.endswith('staggered') else 0
                require(release == epoch + i // n * PERIOD_NS + phase, condition)
            prev_ready = ready
    return runs, env, hashes


def stats_ms(ns):
    return dict(n=len(ns), **{k: float(v) / 1e6 for k, v in zip(
        ('p50_ms', 'p95_ms', 'p99_ms', 'max_ms'), np.percentile(ns, (50, 95, 99, 100)))})


def fifo_calibration(runs, single_service):
    """Recompute queueing from actual enqueue times, using two service inputs."""
    rng = np.random.default_rng(237)
    groups = defaultdict(lambda: defaultdict(list))
    for (condition, _), rows in sorted(runs.items()):
        actual_ready = isolated_ready = 0
        sampled = rng.choice(single_service, size=len(rows))
        group = groups[condition]
        for (_, release, enqueue, start, ready, stages), sample in zip(rows, sampled):
            for name, duration in zip(METRICS, stages):
                group[name].append(duration)
            if condition == 'single':
                continue
            replay_start = max(enqueue, actual_ready)
            isolated_start = max(enqueue, isolated_ready)
            actual_ready = replay_start + ready - start
            isolated_ready = isolated_start + int(sample)
            group['trace_service_fifo_wait'].append(replay_start - enqueue)
            group['trace_service_fifo_response'].append(actual_ready - release)
            group['single_sample_fifo_wait'].append(isolated_start - enqueue)
            group['single_sample_fifo_response'].append(isolated_ready - release)
    return {condition: {name: stats_ms(values) for name, values in group.items()}
            for condition, group in groups.items()}


def deadline_replay(runs, service_ms, model_id):
    """Compare policies on board arrivals with explicitly hypothetical deadlines."""
    options = (Option(50., Mode(model_id, (service_ms,))),)
    # Service represents the entire serial worker, including PS work and DPU API.
    # External feature construction, network and actuation are outside this model.
    platform = Platform(prep=0., dispatch=0., post=0., ingress=0., tail=0.,
                        switch=0., decision=0.)
    records = []
    for (condition, round_id), rows in sorted(runs.items()):
        n = SOURCES[condition]
        if n not in (6, 12):
            continue
        for batch, tick in enumerate(range(0, 600, 30)):
            jobs = rows[tick*n:(tick+1)*n]
            epoch = jobs[0][1]
            for urgent_count, tight_ms in ((0, 50), (1, 10), (1, 5), (4, 5), (1, 1)):
                urgent = {(batch + round_id + i) % n for i in range(urgent_count)}
                loops = tuple(
                    Loop(str(i), (enqueue-epoch)/1e6 - 50., (enqueue-epoch)/1e6,
                         (release-epoch)/1e6 + (tight_ms if i in urgent else 50.),
                         100., options, release_locked=True)
                    for i, (_, release, enqueue, _, _, _) in enumerate(jobs)
                )
                scenario = Scenario(loops, platform, horizon=50., initial_policy=('fifo', 4),
                                    legal_groups=(4,))
                choice = (0,) * n
                fifo = simulate(scenario, choice, ('fifo', 4))
                edf = simulate(scenario, choice, ('edf', 4))
                planned = replace(scenario, platform=replace(platform, decision=.5))
                start = perf_counter_ns()
                abc = abc_repair(planned)
                wall_ms = (perf_counter_ns() - start) / 1e6
                charged = replace(scenario, platform=replace(platform, decision=max(.5, wall_ms)))
                accepted = abc['accepted'] and simulate(
                    charged, abc['choice'], abc['policy'])['feasible']
                records.append(dict(condition=condition, round=round_id, tick=tick,
                                    urgent=sorted(urgent), deadline_ms=tight_ms,
                                    fifo=fifo['feasible'], edf=edf['feasible'], abc=accepted,
                                    abc_precharge=abc['accepted'], abc_type=abc['initial_type'],
                                    abc_policy=abc['policy'], abc_wall_ms=wall_ms))
    by_group = defaultdict(list)
    for row in records:
        by_group[f"{row['condition']}/urgent{len(row['urgent'])}_D{row['deadline_ms']}"].append(row)
    return records, {k: dict(cases=len(v), fifo=sum(r['fifo'] for r in v),
                             edf=sum(r['edf'] for r in v), abc=sum(r['abc'] for r in v),
                             initial_type=dict(Counter(r['abc_type'] for r in v)))
                     for k, v in by_group.items()}


def run(root, output):
    runs, env, hashes = load_board(root)
    single = [row[-1][-2] for (condition, _), rows in runs.items()
              if condition == 'single' for row in rows]
    profile = fifo_calibration(runs, single)
    by_condition = defaultdict(list)
    for (condition, _), rows in runs.items():
        if condition != 'single':
            by_condition[condition].extend(row[-1][-1] for row in rows)
    measured_misses_at_other_limits = {
        condition: {str(ms): sum(response > ms * 1e6 for response in responses)
                    for ms in (5, 10, 15, 50)}
        for condition, responses in by_condition.items()
    }
    p99 = profile['single']['service']['p99_ms']
    records, decisions = deadline_replay(runs, p99, env['model']['id'])
    report = dict(
        source=str(root.resolve()), source_hashes=hashes, script_sha256=digest(Path(__file__)),
        model_id=env['model']['id'], requests=sum(map(len, runs.values())), runs=len(runs),
        measured_profile_and_fifo_replays=profile, stress_serial_service_ms=p99,
        measured_misses_at_other_limits=measured_misses_at_other_limits,
        stress_scope=('Measured enqueue times; isolated serial service p99 fixed per request; '
                      'one 50ms release window with no initial backlog; hypothetical deadlines; '
                      'fixed model and fixed releases; FIFO/EDF no online decision cost, '
                      'ABC with 0.5ms reserve or observed Python wall time, whichever is larger'),
        stress_results=decisions,
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (output / 'deadline_cases.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    print(f'Audited {len(hashes)} files, {report["requests"]} requests, {len(records)} deadline cases.')
    print(f'Report: {output / "summary.json"}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--board-data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.board_data, args.output)
