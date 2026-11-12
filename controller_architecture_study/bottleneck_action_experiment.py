"""Paired finite timing experiments for whole-job DPU service."""
import argparse
from dataclasses import asdict, replace
from itertools import product
import json
from pathlib import Path
import random
from time import perf_counter_ns

from .bottleneck_abc import abc_repair, diagnose
from .bottleneck_joint import Job, Loop, Mode, Option, Platform, Scenario, edit_cost, simulate
from .replay_remaining_chain import pipeline_passed


SMALL = Mode('11', (1.333814,))
LARGE = Mode('6', (11.787358,))
PERIODS = (30., 60., 120.)
ORDERS = ('edf', 'fifo', 'guarded_edf')
STRATA = ('A', 'B', 'C', 'Mixed')
ACTIONS = {1: 'T', 2: 'c', 4: 'schedule', 3: 'T+c', 5: 'T+schedule', 6: 'c+schedule', 7: 'all'}
METHODS = ('fixed_order', 'local', 'mode_only', 'abc', 'guarded_abc')


def loop(name, release, deadline, mode=SMALL, *, current=60., periods=PERIODS, locked=False, both=True):
    initial = Option(current, mode)
    options = tuple(Option(t, m) for t, m in product(periods, (SMALL, LARGE) if both else (mode,)))
    return Loop(name, release-current, release, deadline, 240.,
                (initial, *(o for o in options if o != initial)), locked)


def case(kind, seed):
    rng = random.Random(seed)
    platform = Platform(prep=rng.uniform(.5, 1.5), dispatch=.1, post=rng.uniform(.25, 1.),
                        ingress=.1, tail=.25, switch=.011, decision=rng.choice((.5, 1., 2.)),
                        initial_pl_block=rng.choice((0., 1.5, 4.)), rearm=rng.uniform(.2, 1.5),
                        front_wait=rng.uniform(0., 2.))
    if kind == 'A':
        r = rng.uniform(31., 39.)
        d = r+rng.uniform(-3., 6.) if seed % 10 else rng.uniform(.1, 1.5)
        loops = (loop('a', r, d), loop('b', 12., 48.))
    elif kind == 'B':
        r = rng.uniform(1., 4.)
        d = r+rng.uniform(16., 26.)
        loops = (loop('a', r, d, LARGE, current=30., locked=True, both=seed % 5 != 0),
                 loop('b', r, d, LARGE, current=30., locked=True, both=seed % 5 != 0))
    elif kind == 'C':
        r = rng.uniform(1., 4.)
        u = r+rng.uniform(2., 6.)
        loops = (loop('a', r, r+rng.uniform(20., 28.), LARGE, locked=True),
                 loop('b', u, u+rng.uniform(5., 12.), locked=True))
    else:
        r = rng.uniform(6., 11.)
        d = r+rng.uniform(-1., 3.)
        loops = (loop('a', r, d, LARGE), loop('b', rng.uniform(1., 6.), 34., LARGE, locked=True))
    loops += (loop('c', rng.uniform(13., 29.), 70., LARGE if seed % 3 == 0 else SMALL,
                   current=60., locked=seed % 4 == 0),)
    return Scenario(loops, platform, initial_policy=('edf', 1), legal_groups=(1,))


def action_mask(scenario, choice, policy):
    mask = 4 if policy != scenario.initial_policy else 0
    for loop_, index in zip(scenario.loops, choice):
        initial, option = loop_.options[0], loop_.options[index]
        mask |= int(initial.period != option.period) | (2*int(initial.mode != option.mode))
    return mask


def passed(report):
    return pipeline_passed({'accepted': report['feasible'], 'forecast': report})


def enumerate_actions(scenario, initial):
    best = {}
    for choice in product(*(range(len(item.options)) for item in scenario.loops)):
        for order in ORDERS:
            policy = (order, 1)
            mask = action_mask(scenario, choice, policy)
            report = initial if mask == 0 else simulate(scenario, choice, policy)
            if not passed(report):
                continue
            key = edit_cost(scenario, choice, policy)
            if mask not in best or key < best[mask]['cost']:
                best[mask] = dict(choice=choice, policy=policy, cost=key, slack=report['slack'])
    return best


def sanity():
    p = Platform(prep=0., dispatch=0., post=0., ingress=0., tail=0., switch=0., decision=0.)
    fixtures = (
        ('A', (loop('a', 34.5, 35., periods=(50., 60.), both=False),), {1}),
        ('B', (loop('a', 0., 20., LARGE, locked=True), loop('b', 0., 20., LARGE, locked=True)), {2}),
        ('C', (loop('a', 0., 25., LARGE, locked=True, both=False),
               loop('b', 2., 6., locked=True, both=False)), {4}),
        ('Mixed', (loop('a', 34.5, 35., LARGE, periods=(50., 60.)),), {3}),
    )
    results = {}
    for name, loops, expected in fixtures:
        s = Scenario(loops, p, horizon=0., initial_policy=('edf', 1), legal_groups=(1,))
        base = simulate(s, (0,)*len(loops), s.initial_policy)
        best = enumerate_actions(s, base)
        minimal = {m for m in best if not any(other != m and other & m == other for other in best)}
        assert not passed(base) and minimal == expected, (name, minimal)
        results[name] = {'baseline_slack': base['slack'], 'minimal_action': ACTIONS[next(iter(minimal))],
                         'repaired': best[next(iter(minimal))]}
    return results


def local_search(scenario, budget, method, seed):
    choice, policy = (0,)*len(scenario.loops), scenario.initial_policy
    report = simulate(scenario, choice, policy)
    seen = {(choice, policy)}
    rng = random.Random(seed)
    while len(seen) < budget and not passed(report):
        neighbors = []
        for i, loop_ in enumerate(scenario.loops):
            before = loop_.options[choice[i]]
            for k, option in enumerate(loop_.options):
                kind = int(option.period != before.period) | (2*int(option.mode != before.mode))
                if kind and (method != 'mode_only' or kind == 2):
                    neighbors.append((kind, choice[:i]+(k,)+choice[i+1:], policy))
        if method != 'mode_only':
            neighbors.extend((4, choice, (order, 1)) for order in ORDERS)
        neighbors = [(kind, c, p) for kind, c, p in neighbors if (c, p) not in seen]
        if not neighbors:
            break
        if method == 'fixed_order':
            neighbors.sort(key=lambda item: (item[0] == 3, item[0]))
        else:
            rng.shuffle(neighbors)
        _, candidate, order = neighbors[0]
        seen.add((candidate, order))
        revised = simulate(scenario, candidate, order)
        if passed(revised) or min(revised['slack'].values()) > min(report['slack'].values()):
            choice, policy, report = candidate, order, revised
    return choice, policy, report, len(seen)


def evaluate(scenario, initial, method, budget, seed):
    start = perf_counter_ns()
    if method in ('abc', 'guarded_abc'):
        r = abc_repair(scenario, max_checks=budget, allow_guarded_edf=method == 'guarded_abc')
        choice, policy, report, checks = (r[k] for k in ('choice', 'policy', 'forecast', 'checks'))
    else:
        choice, policy, report, checks = local_search(scenario, budget, method, seed)
    wall = (perf_counter_ns()-start)/1e6
    charged = replace(scenario, platform=replace(scenario.platform, decision=max(wall, scenario.platform.decision)))
    verified = simulate(charged, choice, policy)
    target = next(j.loop for j in initial['jobs'] if j.name == initial['worst'])
    # Match first obligations. Future releases may change with T.
    delta = {l.name: verified['slack'][f'{l.name}:0']-initial['slack'][f'{l.name}:0'] for l in scenario.loops}
    accepted = passed(report)
    return dict(checks=checks, wall_ms=wall, accepted=accepted, charged=accepted and passed(verified),
                choice=choice, policy=policy, min_slack=min(verified['slack'].values()), misses=len(verified['misses']),
                target_delta=delta[target], peer_loss=max(0., *(-v for k, v in delta.items() if k != target)))


def run(output, seed_start, per_stratum):
    fixtures = sanity()
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, kind in enumerate(STRATA):
        for n in range(per_stratum):
            seed = seed_start+index*per_stratum+n
            s = case(kind, seed)
            base = simulate(s, (0,)*len(s.loops), s.initial_policy)
            fixed = passed(base)
            best = enumerate_actions(s, base) if not fixed else {}
            methods = {m: [] for m in METHODS}
            if not fixed:
                for budget in range(1, 6):
                    offset = (n+budget) % len(METHODS)
                    for m in METHODS[offset:]+METHODS[:offset]:
                        methods[m].append(evaluate(s, base, m, budget, seed))
            assert not any(v['accepted'] and not best for results in methods.values() for v in results)
            rows.append(dict(stratum=kind, seed=seed, scenario=asdict(s), fixed=fixed,
                             diagnosis=None if fixed else diagnose(s, base['jobs']),
                             baseline_min_slack=min(base['slack'].values()), oracle=best, methods=methods))
        print(f'{kind}: completed {per_stratum} scenarios', flush=True)
    (output/'cases.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    (output/'sanity.json').write_text(json.dumps(fixtures, indent=2), encoding='utf-8')
    (output/'protocol.json').write_text(json.dumps(dict(seed_start=seed_start, per_stratum=per_stratum,
        periods=PERIODS, orders=ORDERS, service_ms={m.name:m.segments[0] for m in (SMALL, LARGE)},
        service_source='board_237d_timing_20261106_pair_run001 RESULTS.md, dual_alternate service P99',
        scope='controlled timing replay; host ranges, deadlines, initial blocking and mode eligibility are experimental inputs',
        charging='replay chosen configuration at max(planned decision delay, measured search wall time)'), indent=2), encoding='utf-8')
    summarize(output, rows)


def replay_trace(output, trace):
    """Hold gate-window releases, deadlines, host cost and carry-in at their logged values."""
    origin = json.loads((trace/'setup.json').read_text(encoding='utf-8'))['origin_s']
    log = json.loads((trace/'decisions.json').read_text(encoding='utf-8'))
    rows = []
    for i, record in enumerate(log):
        t = record['snapshot_s']-origin
        if not 4.7 <= t <= 5.4:
            continue
        loops = []
        for info in record['loop_context']:
            options = tuple(Option(o['period'], Mode(o['mode']['name'], tuple(o['mode']['segments'])), o['qualified'])
                            for o in info['options'])
            loops.append(Loop(**dict(info, options=options)))
        carry = tuple(Job(**info) for info in record['carry_context'])
        s = Scenario(tuple(loops), Platform(**record['platform']), horizon=240., carry=carry,
                     initial_policy=('edf', 1), legal_groups=(1,))
        slow = loops[-1]
        big = Mode(LARGE.name, (slow.options[0].mode.segments[0]+LARGE.segments[0]-SMALL.segments[0],))
        big_options = tuple(replace(o, mode=big) for o in slow.options)
        mixed = replace(s, loops=(*loops[:-1], replace(slow, options=big_options+slow.options)))
        for profile, scenario in (('logged', s), ('hypothetical_slow_peer', mixed)):
            base = simulate(scenario, (0,)*len(scenario.loops), scenario.initial_policy)
            fixed = passed(base)
            oracle = enumerate_actions(scenario, base) if not fixed else {}
            methods = {m: evaluate(scenario, base, m, 5, i) for m in METHODS} if not fixed else {}
            rows.append(dict(index=i, time_s=t, profile=profile, fixed=fixed, oracle=oracle, methods=methods,
                             expired=any(l.deadline <= 0. for l in scenario.loops), carry=len(carry),
                             initial_type=None if fixed else diagnose(scenario, base['jobs'])['type']))
    if not rows:
        raise ValueError('no logged planning snapshots in the gate window')
    stats = {}
    for profile in ('logged', 'hypothetical_slow_peer'):
        subset = [r for r in rows if r['profile'] == profile]
        failed = [r for r in subset if not r['fixed']]
        stats[profile] = dict(total=len(subset), baseline_failed=len(failed), expired=sum(r['expired'] for r in failed),
                              oracle_feasible=sum(bool(r['oracle']) for r in failed),
                              carry_cases=sum(r['carry']>0 for r in subset),
                              charged={m:sum(r['methods'][m]['charged'] for r in failed) for m in METHODS})
    output.mkdir(parents=True, exist_ok=False)
    (output/'trace_cases.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    (output/'trace_summary.json').write_text(json.dumps(dict(source=str(trace), window_s=[4.7, 5.4],
        scope='logged timing envelopes; slow peer is hypothetical, adds board P99 service difference, small mode permitted',
        profiles=stats), indent=2), encoding='utf-8')
    print(json.dumps(stats), flush=True)


def summarize(output, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    failures = [r for r in rows if not r['fixed']]
    feasible = [r for r in failures if r['oracle']]
    counts, denominators = [], []
    for kind in STRATA:
        subset = [r for r in failures if r['stratum'] == kind]
        denominators.append(len(subset))
        counts.append([sum(any(int(m) & allowed == int(m) for m in r['oracle']) for r in subset) for allowed in ACTIONS])
    curves = {m: [sum(r['methods'][m][i]['accepted'] for r in feasible) for i in range(5)] for m in METHODS}
    stats = {}
    for m in METHODS:
        values = [r['methods'][m][-1] for r in failures]
        wall = [v['wall_ms'] for v in values]
        stats[m] = dict(found=curves[m][-1], charged=sum(v['charged'] for v in values),
                        mean_ms=float(np.mean(wall)), p99_ms=float(np.percentile(wall, 99)), max_ms=max(wall),
                        mean_checks=float(np.mean([v['checks'] for v in values])))
    summary = dict(total=len(rows), baseline_failed=len(failures), oracle_feasible=len(feasible),
                   joint_only=sum(not {int(m) for m in r['oracle']}.intersection((1, 2, 4)) for r in feasible),
                   action_columns=list(ACTIONS.values()), strata=STRATA, counts=counts,
                   denominators=denominators, curves=curves, methods=stats, matplotlib=matplotlib.__version__)
    (output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    labels = ('Fixed order', 'Local search', 'Model only', 'ABC', 'ABC + dispatch guard')
    fig, ax = plt.subplots(figsize=(8, 3.5), layout='constrained')
    rates = np.divide(counts, np.array(denominators)[:, None], out=np.zeros((4, 7)),
                      where=np.array(denominators)[:, None] != 0)
    ax.imshow(rates, cmap='Blues', vmin=0, vmax=1)
    ax.set(xticks=range(7), xticklabels=('$T$', '$c$', '$\\sigma$', '$T+c$', '$T+\\sigma$', '$c+\\sigma$', 'All'),
           yticks=range(4), yticklabels=STRATA,
           title='Whole-job replay: repairs among initial failures', xlabel='Allowed actions')
    for i, n in enumerate(denominators):
        for j, value in enumerate(counts[i]):
            ax.text(j, i, f'{value}/{n}', ha='center', va='center', color='white' if rates[i,j]>.55 else 'black')
    fig.savefig(output/'action_effectiveness.png', dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4), layout='constrained')
    for m, label, marker, style in zip(METHODS, labels, ('o', 's', '^', 'D', 'x'), ('-', '--', ':', '-.', '-')):
        ax.plot(range(1, 6), np.array(curves[m])/len(feasible), marker=marker, linestyle=style, label=label)
    ax.set(xticks=range(1, 6), ylim=(0, 1.05), xlabel='Full timing checks allowed (including baseline)',
           ylabel='Fraction repaired', title=f'Discovery among {len(feasible)} finite-oracle feasible failures')
    ax.legend(loc='best')
    fig.savefig(output/'discovery_budget.png', dpi=180)
    plt.close(fig)

    fig, (ax, worst) = plt.subplots(1, 2, figsize=(10.5, 4), layout='constrained')
    for ok, label, marker in ((False, 'Returned configuration still fails', 'x'), (True, 'Passes full check', 'o')):
        subset = [r for r in failures if r['methods']['guarded_abc'][-1]['charged'] == ok]
        points = [r['methods']['guarded_abc'][-1] for r in subset]
        ax.scatter([v['target_delta'] for v in points], [v['peer_loss'] for v in points],
                   marker=marker, alpha=.7, label=f'{label} (n={len(points)})')
        worst.scatter([r['baseline_min_slack'] for r in subset], [v['min_slack'] for v in points],
                      marker=marker, alpha=.7)
    ax.axvline(0, color='0.5', linewidth=.7)
    ax.set(xlabel='Target first-obligation slack change (ms)', ylabel='Largest peer slack loss (ms)')
    ax.legend(fontsize='small')
    worst.axhline(0, color='0.5', linewidth=.7)
    worst.set(xlabel='Baseline minimum slack (ms)', ylabel='Returned minimum slack (ms)')
    fig.suptitle('ABC + dispatch guard, after charging decision time')
    fig.savefig(output/'peer_slack.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seed-start', default=73000, type=int)
    parser.add_argument('--per-stratum', default=40, type=int)
    parser.add_argument('--trace', type=Path, help='replay the prior split-arm gate window instead of controlled scenarios')
    args = parser.parse_args()
    if args.trace:
        replay_trace(args.output, args.trace)
    else:
        run(args.output, args.seed_start, args.per_stratum)
