"""237D paired-cost audit and exact model allocation for an EDF-ready batch."""
import argparse
from collections import defaultdict
import csv
from dataclasses import replace
import json
from math import ceil
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from .bottleneck_abc import abc_repair
from .bottleneck_board_replay import digest, load_board, require, stats_ms
from .bottleneck_joint import Loop, Mode, Option, Platform, Scenario, simulate


def audit_pair(root):
    hashes = {}
    for line in (root / 'SHA256SUMS.txt').read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        require(digest(root / name) == expected, f'Bad hash: {name}')
        hashes[name] = expected
    blocks = defaultdict(list)
    timings = defaultdict(list)
    paired = {}
    references = {
        m: np.loadtxt(root / 'inputs' / f'{m}.txt')[:, 237:241] for m in (11, 6)
    }
    names = {
        11: 'main18_plain_mlp_h4_w1024_seed0',
        6: 'main18_plain_mlp_h33_w1024_seed0',
    }
    for process in ('solo_A', 'solo_B', 'dual'):
        require('backend=REAL_DNNDK_DPU' in
                (root / 'capture' / process / 'process_meta.txt').read_text(), process)
    with (root / 'requests.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            scenario, run_id, request_id = (row[key] for key in ('scenario', 'run_id', 'request_id'))
            model, occurrence, index, seq, round_id = (int(row[key]) for key in
                ('numeric_model_id', 'model_occurrence', 'input_id', 'sequence', 'round'))
            require(round_id in (1, 2, 3) and run_id == f'{scenario}_r{round_id}' and
                    request_id == f'{run_id}:{seq}' and index == occurrence % 32, request_id)
            if scenario == 'dual_alternate':
                require(model == (11 if seq % 2 == 0 else 6) and occurrence == seq // 2, request_id)
            else:
                require(scenario in ('solo_A', 'solo_B', 'dual_A_repeat', 'dual_B_repeat') and
                        model == (11 if '_A' in scenario else 6) and occurrence == seq, request_id)
            require(row['model_id'] == row['actual_model_id'] == names[model] and
                    row['status'] == 'success' and row['completed'] == row['output_valid'] == '1',
                    request_id)
            times = [int(row[key + '_ns']) for key in ('release', 'dispatch_start', 'dispatch_end',
                     'prep_start', 'prep_end', 'submit', 'done', 'post_start', 'output_ready')]
            require(all(a <= b for a, b in zip(times, times[1:])), request_id)
            service, response = times[-1] - times[3], times[-1] - times[0]
            require(service == int(row['service_ns']) and response == int(row['response_ns']), request_id)
            output = [float(row[f'u{i}']) for i in range(4)]
            require(np.max(np.abs(output - references[model][index])) <= 1e-4, request_id)
            blocks[run_id].append((seq, times[0], times[-1]))
            timings[(scenario, model)].append((service, response))
            key = (scenario, model, round_id, occurrence)
            require(key not in paired, request_id)
            paired[key] = (request_id, service, index)
    require(len(blocks) == 15 and len(paired) == 18000, 'Expected 18,000 formal rows')
    for name, rows in blocks.items():
        rows.sort()
        require(len(rows) == (2000 if name.startswith('dual_alternate') else 1000), name)
        require(all(i == seq and (i == 0 or rows[i-1][2] <= release)
                    for i, (seq, release, _) in enumerate(rows)), name)
    deltas = defaultdict(list)
    seen = set()
    with (root / 'paired_deltas.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            model, rnd, occurrence = (int(row[key]) for key in ('numeric_model_id', 'round', 'model_occurrence'))
            key = (model, rnd, occurrence)
            require(key not in seen, str(key))
            seen.add(key)
            repeat = paired[('dual_A_repeat' if model == 11 else 'dual_B_repeat', *key)]
            alternate = paired[('dual_alternate', *key)]
            delta = alternate[1] - repeat[1]
            require((row['repeat_request_id'], row['alternate_request_id']) == (repeat[0], alternate[0])
                    and int(row['input_id']) == repeat[2] == alternate[2]
                    and int(row['delta_service_ns']) == delta, str(key))
            deltas[model].append(delta)
    require(len(seen) == 6000, 'Expected 6000 matched pairs')
    profile = {f'{scenario}/{model}': dict(service=stats_ms([s for s, _ in rows]),
               response=stats_ms([r for _, r in rows]))
               for (scenario, model), rows in timings.items()}
    return hashes, profile, {m: dict(stats_ms(d), mean_ms=float(np.mean(d))/1e6)
                            for m, d in deltas.items()}


def minimum_downgrades(deadlines, fast_allowed, start, fast_ms, slow_ms):
    """Exact for a serial EDF batch ready at start, with equal savings per switch."""
    savings = slow_ms - fast_ms
    require(savings > 0, 'Expected faster mode to have lower cost')
    choices = [0] * len(deadlines)
    eligible = []
    count = 0
    for prefix, i in enumerate(sorted(range(len(deadlines)), key=deadlines.__getitem__), 1):
        if fast_allowed[i]:
            eligible.append(i)
        needed = max(0, ceil((start + prefix * slow_ms - deadlines[i] - 1e-8) / savings))
        while count < needed:
            if not eligible:
                return None
            choices[eligible.pop()] = 1
            count += 1
    return tuple(choices)


def compare_batch(runs, fast_ms, slow_ms):
    cases = []
    platform = Platform(prep=0., dispatch=0., post=0., ingress=0., tail=0.,
                        switch=0., decision=.5)
    modes = Mode('model6', (slow_ms,)), Mode('model11', (fast_ms,))
    for (condition, rnd), rows in sorted(runs.items()):
        if not condition.endswith('_synchronized'):
            continue
        n = 6 if condition.startswith('open_6') else 12
        for batch, tick in enumerate(range(0, 600, 30)):
            samples = rows[tick*n:(tick+1)*n]
            epoch = samples[0][1]
            arrivals = [(r[2] - epoch)/1e6 for r in samples]
            # Wait until the sampled burst is fully known. This is shared by all methods.
            start = max(platform.decision, *arrivals)
            for name, urgent_count, tight, slow_only in (
                ('D50_all_eligible', 0, 50., 0), ('D50_three_slow_only', 0, 50., 3),
                ('D50_four_slow_only', 0, 50., 4), ('D5_one_urgent', 1, 5., 0),
                ('D8_four_urgent', 4, 8., 0), ('D5_four_urgent', 4, 5., 0),
                ('D5_urgent_slow_only', 1, 5., 1),
            ):
                order = [(batch + rnd + j) % n for j in range(n)]
                urgent, mandatory = set(order[:urgent_count]), set(order[:slow_only])
                deadlines = [tight if i in urgent else 50. for i in range(n)]
                allowed = [i not in mandatory for i in range(n)]
                loops = tuple(Loop(str(i), arrival - 50., arrival, deadlines[i], 100.,
                              (Option(50., modes[0]), Option(50., modes[1], allowed[i])), True)
                              for i, arrival in enumerate(arrivals))
                scenario = Scenario(loops, replace(platform, decision=start), horizon=50.,
                                    initial_policy=('edf', 4), legal_groups=(4,))
                base = simulate(scenario, (0,) * n, ('edf', 4))['feasible']
                floor = simulate(scenario, tuple(map(int, allowed)), ('edf', 4))['feasible']
                t = perf_counter_ns()
                exact = minimum_downgrades(deadlines, allowed, start, fast_ms, slow_ms)
                exact_ms = (perf_counter_ns() - t) / 1e6
                require((exact is not None) == floor, 'Batch oracle disagreed with fastest legal EDF')
                if exact is not None:
                    require(simulate(scenario, exact, ('edf', 4))['feasible'], 'Batch choice failed replay')
                t = perf_counter_ns()
                abc = abc_repair(scenario)
                abc_ms = (perf_counter_ns() - t) / 1e6
                cases.append(dict(n=n, condition=name, round=rnd, tick=tick, fixed_slow=base,
                                  fastest_eligible=floor, prefix=exact is not None,
                                  abc=abc['accepted'], prefix_downgrades=sum(exact) if exact is not None else None,
                                  abc_downgrades=sum(abc['choice']) if abc['accepted'] else None,
                                  abc_edits=len(abc['attempts']), prefix_ms=exact_ms, abc_ms=abc_ms,
                                  batch_start_ms=start))
    groups = defaultdict(list)
    for row in cases:
        groups[f'{row["n"]}/{row["condition"]}'].append(row)
    summary = {k: dict(cases=len(v), **{method: sum(r[method] for r in v)
                 for method in ('fixed_slow', 'fastest_eligible', 'prefix', 'abc')},
                 prefix_downgrades=sorted({r['prefix_downgrades'] for r in v if r['prefix']}),
                 abc_downgrades=sorted({r['abc_downgrades'] for r in v if r['abc']}))
               for k, v in groups.items()}
    return cases, summary


def run(pair_dir, arrivals_dir, output):
    hashes, profile, deltas = audit_pair(pair_dir)
    runs, _, arrival_hashes = load_board(arrivals_dir)
    # Profile response from the synchronous local pair runner covers each full
    # serial request including dispatch; alternate samples include model changes.
    costs = {m: max(profile[f'dual_{c}/{m}']['response']['p99_ms']
                    for c in (('A_repeat' if m == 11 else 'B_repeat'), 'alternate'))
             for m in (11, 6)}
    rows, summary = compare_batch(runs, costs[11], costs[6])
    report = dict(pair_hashes=hashes, arrivals_csv_sha256=arrival_hashes['requests.csv'],
                  pair_requests=18000, pair_profile=profile, matched_service_delta_ms=deltas,
                  replay_cost_ms=costs, groups=summary, script_sha256=digest(Path(__file__)),
                  assumptions='Single atomic worker; observed synchronized enqueues; shared gate until '
                              'all jobs ready and at least 0.5ms; no initial carry; deadlines and mode '
                              'eligibility are hypothetical. Fixed decision reserve for all methods.')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (output / 'cases.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print(f'Checked {len(hashes)} hashes, 18000 pair requests and {len(rows)} replay cases.')
    print(f'Report: {output / "summary.json"}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pair-data', type=Path, required=True)
    parser.add_argument('--arrivals-data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.pair_data, args.arrivals_data, args.output)
