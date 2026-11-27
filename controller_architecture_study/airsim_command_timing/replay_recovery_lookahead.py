"""Compare action scheduling with short, explicitly revealed capacity windows."""

import argparse
from collections import Counter
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path

from .replay_recovery_windows import DONE, UNSTARTED, Recovery, dispatch, future_slots, oracle, outcome, ready, replay


LOOKAHEADS = (0, 1, 2, 4)
POLICIES = ('edf_visible', 'finish_first', 'complete_visible')


def visible_candidates(requests, state, now, possible):
    """Use known closures to tighten EDF, keeping unobserved slots possible."""
    candidates = []
    for i, due in ready(requests, state, now):
        request = requests[i]
        first = state[i]
        if first >= 0:
            due = max(t for t in future_slots(request, first, now - 1) if possible(t))
        elif request.updates == 2:
            valid_starts = [t for t in range(now, due + 1) if possible(t)
                            and any(possible(s) for s in future_slots(request, t, t))]
            if not valid_starts or valid_starts[0] != now:
                continue
            due = valid_starts[-1]
        candidates.append((due, first == UNSTARTED, i))
    return [i for _, _, i in sorted(candidates)]


def score_actions(requests, state, now, capacity, candidates):
    """Count the requests that can finish in this fixed, short visible prefix."""
    announced = tuple(r.release <= now for r in requests)
    end = now + len(capacity)

    @lru_cache(maxsize=None)
    def score(t, remaining):
        if t == end:
            return remaining.count(DONE)
        best = score(t + 1, remaining)
        if capacity[t - now]:
            for i, _ in ready(requests, remaining, t):
                if announced[i]:
                    best = max(best, score(t + 1, dispatch(requests, remaining, t, i)))
        return best

    values = {i: score(now + 1, dispatch(requests, state, now, i)) for i in candidates}
    return values, score.cache_info().misses


def play(requests, available, horizon, policy, announced=None):
    promised = available if announced is None else announced
    if len(promised) != len(available):
        raise ValueError('The announced and actual windows must have the same length')
    state, schedule = (UNSTARTED,) * len(requests), []
    changes = nodes = 0
    for now, opening in enumerate(available):
        if not opening:
            schedule.append(None)
            continue
        end = min(now + horizon + 1, len(available))
        capacity = (True, *promised[now+1:end])

        def possible(t):
            return t < len(available) and (t >= end or capacity[t-now])

        candidates = visible_candidates(requests, state, now, possible)
        if not candidates:
            schedule.append(None)
            continue
        choice = candidates[0]
        if policy == 'finish_first':
            choice = next((i for i in candidates if state[i] >= 0), choice)
        if policy in ('complete_visible', 'one_loss') and len(candidates) > 1:
            values, visited = score_actions(requests, state, now, capacity, candidates)
            worst = values.copy()
            if policy == 'one_loss':
                for j, free in enumerate(capacity[1:], 1):
                    if not free:
                        continue
                    lost = (*capacity[:j], False, *capacity[j+1:])
                    after_loss, checked = score_actions(requests, state, now, lost, candidates)
                    worst = {i: min(worst[i], after_loss[i]) for i in candidates}
                    visited += checked
            choice = max(candidates, key=lambda i: (worst[i], values[i]))
            changes += choice != candidates[0]
            nodes = max(nodes, visited)
        schedule.append(choice)
        state = dispatch(requests, state, now, choice)
    return dict(outcome(requests, state, schedule), changed=changes, max_nodes=nodes)


def summarize(rows):
    edf = [r['edf'] for r in rows]
    measures = ('complete', 'complete_pairs', 'timely', 'incomplete_started')
    base = {k: sum(x[k] for x in edf) for k in measures}
    summary = {}
    for k in LOOKAHEADS:
        pairs = [r['horizons'][k] for r in rows]
        counts = {}
        for policy in POLICIES:
            outcomes = [p[policy] for p in pairs]
            counts[policy] = {key: sum(x[key] for x in outcomes) for key in measures}
        comparison = {}
        for baseline in ('edf_visible', 'finish_first'):
            deltas = Counter(p['complete_visible']['complete'] - p[baseline]['complete'] for p in pairs)
            comparison[baseline] = dict(win=sum(n for d, n in deltas.items() if d > 0),
                loss=sum(n for d, n in deltas.items() if d < 0), tie=deltas[0],
                same_timely_wins=sum(p['complete_visible']['complete'] > p[baseline]['complete']
                                     and p['complete_visible']['timely'] == p[baseline]['timely'] for p in pairs))
        summary[k] = dict(**counts, comparison=comparison,
                          max_nodes=max(p['complete_visible']['max_nodes'] for p in pairs))
    return dict(cases=len(rows), edf=base, oracle=sum(r['oracle']['complete'] for r in rows), horizons=summary)


def run(source):
    raw = source.read_bytes()
    studies = []
    for case in json.loads(raw):
        available = tuple(case['available'])
        for mix in ('mixed', 'pairs_only'):
            if mix == 'pairs_only' and case['load'] != 'trace':
                continue
            selected = [r for r in case['requests'] if mix == 'mixed' or r['updates'] == 2]
            requests = tuple(Recovery(**r) for r in selected)
            edf = replay(requests, available, 'edf')
            optimum = case['results']['oracle'] if mix == 'mixed' else oracle(requests, available)
            if mix == 'mixed':
                assert edf == case['results']['edf']
            horizons = {k: {policy: play(requests, available, k, policy) for policy in POLICIES} for k in LOOKAHEADS}
            assert all(result['complete'] <= optimum['complete']
                       for policies in horizons.values() for result in policies.values())
            studies.append(dict(phase_ms=case['phase_ms'], source_origin_ns=case['source_origin_ns'],
                                load=case['load'], fleet=len(requests), mix=mix, requests=selected, available=available,
                                edf=edf, oracle=optimum, horizons=horizons))
    summary = {label: summarize([r for r in studies if (r['load'], r['mix']) == case])
               for label, case in {'trace': ('trace', 'mixed'), 'all_open': ('all_open', 'mixed'),
                                   'pairs_only': ('trace', 'pairs_only')}.items()}
    return dict(source=str(source.resolve()), cases_sha256=sha256(raw).hexdigest(), summary=summary), studies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True)
    summaries = []
    for source in args.cases:
        report, rows = run(source)
        summaries.append(report)
        (args.output/f'{source.parent.name}_cases.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    (args.output/'summary.json').write_text(json.dumps(summaries, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
