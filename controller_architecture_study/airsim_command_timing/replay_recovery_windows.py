"""Replay recovery sequences with one correction per available opportunity."""

import argparse
from bisect import bisect_right
import csv
from dataclasses import asdict, dataclass
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
from random import Random


TICK_NS = 75_000_000
WINDOW_NS = 30_000_000
PREP_NS = 3_500_000
START_RESERVE_NS = 4_500_000
DPU_NS = 1_350_000
GAPS = (1, 2, 4)
HORIZON = 12
PHASES_MS = (0, 25, 50)
POLICIES = ('edf', 'finish_first', 'guard', 'guard_one_loss', 'oracle')
UNSTARTED = -1
DONE = -2


@dataclass(frozen=True)
class Recovery:
    release: int
    first_by: int
    finish_by: int
    updates: int


def future_slots(request, first, after):
    return tuple(first+gap for gap in GAPS if after < first+gap <= request.finish_by)


def ready(requests, state, now):
    for i, (r, first) in enumerate(zip(requests, state)):
        if first == UNSTARTED and r.release <= now <= r.first_by:
            if r.updates == 1 or future_slots(r, now, now):
                yield i, r.first_by
        if first >= 0:
            slots = future_slots(r, first, first)
            if now in slots:
                yield i, slots[-1]


def dispatch(requests, state, now, i):
    changed = now if state[i] == UNSTARTED and requests[i].updates == 2 else DONE
    return (*state[:i], changed, *state[i+1:])


def matchable(choices, removed=None):
    """Each remaining follow-up needs its own future slot."""
    if not choices:
        return True
    first, *rest = choices
    return any(t != removed and matchable([tuple(x for x in row if x != t) for row in rest], removed)
               for t in first)


def select(requests, state, now, candidates, policy):
    by_deadline = sorted(candidates, key=lambda p: (p[1], state[p[0]] == UNSTARTED, p[0]))
    if policy == 'edf':
        return by_deadline[0][0]
    finish_first = next((i for i, _ in by_deadline if state[i] >= 0), by_deadline[0][0])
    if policy == 'finish_first':
        return finish_first
    for i, _ in by_deadline:
        nxt = dispatch(requests, state, now, i)
        choices = sorted([future_slots(r, first, now) for r, first in zip(requests, nxt)
                          if first >= 0 and future_slots(r, first, now-1)], key=len)
        if policy == 'guard':
            admitted = matchable(choices)
        else:
            losses = {t for options in choices for t in options}
            admitted = matchable(choices) and all(matchable(choices, t) for t in losses)
        if admitted:
            return i
    return finish_first


def replay(requests, available, policy):
    state, out = (UNSTARTED,)*len(requests), []
    for now, free in enumerate(available):
        candidates = list(ready(requests, state, now)) if free else []
        if not candidates:
            out.append(None)
            continue
        i = select(requests, state, now, candidates, policy)
        out.append(i)
        state = dispatch(requests, state, now, i)
    return outcome(requests, state, out)


def outcome(requests, state, out):
    return dict(complete=state.count(DONE),
                complete_pairs=sum(r.updates == 2 and status == DONE for r, status in zip(requests, state)),
                timely=sum(i is not None for i in out),
                incomplete_started=sum(i >= 0 for i in state), schedule=out)


def oracle(requests, available):
    """Exact upper bound with full future releases and availability in this finite replay."""
    initial = (UNSTARTED,)*len(requests)

    @lru_cache(maxsize=None)
    def best(now, state):
        if now == len(available):
            return state.count(DONE), 0, ()
        done, sent, suffix = best(now+1, state)
        result = done, sent, (None, *suffix)
        if not available[now]:
            return result
        for i, _ in ready(requests, state, now):
            done, sent, suffix = best(now+1, dispatch(requests, state, now, i))
            if (done, sent+1) > result[:2]:
                result = done, sent+1, (i, *suffix)
        return result

    _, _, schedule = best(0, initial)
    state = initial
    for now, i in enumerate(schedule):
        if i is not None:
            state = dispatch(requests, state, now, i)
    return outcome(requests, state, schedule)


def opens_window(intervals, starts, stamp):
    begin = stamp+PREP_NS
    last_start = stamp+WINDOW_NS-START_RESERVE_NS
    i = max(0, bisect_right(starts, begin)-1)
    while i < len(intervals) and begin <= last_start:
        a, b = intervals[i]
        if begin+DPU_NS <= a:
            return True
        begin = max(begin, b)
        i += 1
    return begin <= last_start


def trace_masks(source):
    with source.open(newline='', encoding='utf-8') as f:
        intervals = sorted((int(r['begin_ns']), int(r['end_ns'])) for r in csv.DictReader(f) if int(r['status']) == 0)
    if not intervals or any(a >= b for a, b in intervals) or any(
            b > c for (_, b), (c, _) in zip(intervals, intervals[1:])):
        raise ValueError('Expected non-overlapping successful DNNDK calls')
    start, end = intervals[0][0], intervals[-1][1]
    starts = [a for a, _ in intervals]
    horizon_ns = (HORIZON-1)*TICK_NS + WINDOW_NS
    for phase in PHASES_MS:
        for origin in range(start+phase*1_000_000, end-horizon_ns+1, HORIZON*TICK_NS):
            yield phase, origin, tuple(opens_window(intervals, starts, origin+t*TICK_NS) for t in range(HORIZON))


def generated_requests(rng, fleet):
    req = []
    for k in range(fleet):
        release = rng.randrange(5)
        first_by = release+rng.randint(1, 3)
        if k == fleet-1:  # one single-update opportunity competing with the sequences
            req.append(Recovery(release, first_by, first_by, 1))
        else:
            req.append(Recovery(release, first_by, first_by+rng.randint(2, 4), 2))
    rng.shuffle(req)
    return tuple(req)


def summarize(rows):
    result = dict(cases=len(rows), available=sum(sum(r['available']) for r in rows),
                  potential_sequences=sum(len(r['requests']) for r in rows), policies={})
    for name in POLICIES:
        counts = [r['results'][name] for r in rows]
        edf = [r['results']['edf'] for r in rows]
        result['policies'][name] = dict(
            **{k: sum(r[k] for r in counts) for k in ('complete', 'complete_pairs', 'timely', 'incomplete_started')},
            more_cases_than_edf=sum(a['complete'] > b['complete'] for a, b in zip(counts, edf)),
            fewer_cases_than_edf=sum(a['complete'] < b['complete'] for a, b in zip(counts, edf)),
            more_with_same_timely=sum(a['complete'] > b['complete'] and a['timely'] == b['timely']
                                      for a, b in zip(counts, edf)))
    return result


def indistinguishable_prefix():
    """Both traces offer the same first two slots, with different capacity after that."""
    requests = (Recovery(0, 0, 2, 2), Recovery(1, 1, 3, 2), Recovery(4, 4, 4, 1))
    worlds = {'open': (True, True, True, True, True), 'two_lost': (True, True, False, False, True)}
    return dict(requests=[asdict(r) for r in requests], worlds={
        name: dict(available=available, online={p: replay(requests, available, p) for p in POLICIES[:-1]},
                   oracle=oracle(requests, available))
        for name, available in worlds.items()
    })


def run(source, output, seed):
    rng = Random(seed)
    output.mkdir(parents=True)
    rows = []
    for phase, origin, trace in trace_masks(source):
        for fleet in (3, 4):
            requests = generated_requests(rng, fleet)
            for load, available in (('trace', trace), ('all_open', (True,)*HORIZON)):
                results = {p: replay(requests, available, p) for p in POLICIES if p != 'oracle'}
                results['oracle'] = oracle(requests, available)
                assert all(r['complete'] <= results['oracle']['complete'] for r in results.values())
                rows.append(dict(phase_ms=phase, source_origin_ns=origin, load=load, fleet=fleet,
                                 available=available, requests=[asdict(r) for r in requests], results=results))
    if not rows:
        raise ValueError('Trace too short for one replay block')
    report = dict(source=str(source.resolve()), source_sha256=sha256(source.read_bytes()).hexdigest(),
                  seed=seed, horizon_ticks=HORIZON, phase_offsets_ms=PHASES_MS,
                  tick_ms=TICK_NS/1e6, calibrated_followup_offsets_ticks=GAPS,
                  window_ms=WINDOW_NS/1e6, assumed_prep_ms=PREP_NS/1e6,
                  assumed_start_reserve_ms=START_RESERVE_NS/1e6, assumed_dpu_ms=DPU_NS/1e6,
                  indistinguishable_prefix=indistinguishable_prefix(),
                  summary={f'{load}_{fleet}': summarize([r for r in rows if r['load'] == load and r['fleet'] == fleet])
                           for load in ('trace', 'all_open') for fleet in (3, 4)})
    examples = [r for r in rows if r['load'] == 'trace'
                and r['results']['finish_first']['complete'] > r['results']['edf']['complete']
                and r['results']['finish_first']['timely'] == r['results']['edf']['timely']]
    report['example_equal_timely'] = examples[0] if examples else None
    (output/'summary.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    (output/'cases.json').write_text(json.dumps(rows, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261127)
    args = parser.parse_args()
    run(args.trace, args.output, args.seed)
