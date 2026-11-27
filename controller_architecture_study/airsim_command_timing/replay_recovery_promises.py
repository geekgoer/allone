"""Stress the capacity announcements while holding the real DPU trace fixed."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from .replay_recovery_lookahead import play
from .replay_recovery_windows import Recovery


POLICIES = ('edf_visible', 'complete_visible', 'one_loss')
HORIZONS = (1, 2, 4)


def breach_slots(requests, actual):
    lo = min(r.release for r in requests)
    hi = min(len(actual) - 1, max(r.finish_by for r in requests))
    lookahead = max(HORIZONS)
    for slot in range(lo + 1, hi + 1):
        if not actual[slot] and any(actual[max(lo, slot-lookahead):slot]):
            yield slot


def policies(requests, actual, horizon, promised):
    return {p: play(requests, actual, horizon, p, announced=promised) for p in POLICIES}


def stats(runs):
    sums = {}
    for policy in POLICIES:
        own = [row[policy] for row in runs]
        deltas = [a['complete'] - row['edf_visible']['complete'] for row, a in zip(runs, own)]
        sums[policy] = {key: sum(r[key] for r in own)
                        for key in ('complete', 'complete_pairs', 'timely', 'incomplete_started')}
        sums[policy].update(max_nodes=max((r['max_nodes'] for r in own), default=0), delta=sum(deltas),
                            wins=sum(x > 0 for x in deltas), losses=sum(x < 0 for x in deltas))
    return sums


def run(sources):
    rows = []
    sources_used = []
    for source in sources:
        raw = source.read_bytes()
        sources_used.append(dict(path=str(source.resolve()), sha256=sha256(raw).hexdigest()))
        for sample, case in enumerate(json.loads(raw)):
            if case['load'] != 'trace':
                continue
            actual = tuple(case['available'])
            requests = tuple(Recovery(**r) for r in case['requests'])
            clean = {h: policies(requests, actual, h, actual) for h in HORIZONS}
            breaches = {}
            for lost in breach_slots(requests, actual):
                promised = (*actual[:lost], True, *actual[lost+1:])
                breaches[lost] = {h: policies(requests, actual, h, promised) for h in HORIZONS}
            rows.append(dict(source=source.parent.name, sample=sample, requests=case['requests'], actual=actual,
                             clean=clean, breaches=breaches, oracle=case['results']['oracle']['complete']))
    summary = {}
    for h in HORIZONS:
        perfect = [r['clean'][h] for r in rows]
        paired = [r['clean'][h] for r in rows for _ in r['breaches']]
        stressed = [result[h] for r in rows for result in r['breaches'].values()]
        effects = {}
        for policy in POLICIES:
            deltas = Counter(b[policy]['complete'] - a[policy]['complete'] for a, b in zip(paired, stressed))
            effects[policy] = dict(total_change=sum(k*v for k, v in deltas.items()),
                                   worsens=sum(v for k,v in deltas.items() if k < 0),
                                   improves=sum(v for k,v in deltas.items() if k > 0))
        summary[h] = dict(original=stats(perfect), paired_perfect=stats(paired),
                          one_breach=stats(stressed), breach_effect=effects)
    return dict(sources=sources_used, cases=len(rows), variants=sum(len(r['breaches']) for r in rows),
                oracle=sum(r['oracle'] for r in rows), summary=summary), rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report, rows = run(args.cases)
    args.output.mkdir(parents=True)
    (args.output/'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (args.output/'variants.json').write_text(json.dumps(rows, separators=(',', ':')), encoding='utf-8')


if __name__ == '__main__':
    main()
