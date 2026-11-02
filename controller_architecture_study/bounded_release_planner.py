"""Release-first finite planning on the existing two-FC-segment simulator.

T is the installed release-to-release interval, not a 20/60 ms risk label.
No controller/EDF permutation search: modes are fixed before this function.
"""

import math
import time

from .framework_runtime import simulate_pipeline
from .heterogeneous_load_guard import resource_loads


def plan_releases(now, future, pending, profile, costs, *, inflight_limit, max_checks=5,
                  minimum_interval_s=.020, reserve_s=0., planning_budget_s=.005):
    if (not math.isfinite(now) or max_checks < 1 or minimum_interval_s <= 0.
            or reserve_s < 0. or not math.isfinite(reserve_s)
            or not math.isfinite(planning_budget_s) or planning_budget_s <= 0.):
        raise ValueError('finite time and positive bounded planning limits required')
    if not future:
        return {}, dict(feasible=True, checked=0, historical_deadlines_repaired=False)
    started = time.perf_counter()
    ids = {r['sample_id'] for r in pending}
    if any(r['sample_id'] in ids for r in future.values()):
        raise ValueError('released jobs cannot be reconfigured')
    expired = [r['sample_id'] for r in [*pending, *future.values()]
               if r.get('command_completed_s') is None and r['deadline_s'] <= now]
    if expired:
        return None, dict(feasible=False, checked=0, already_expired_ids=expired,
                          historical_deadlines_repaired=False, reason='a committed obligation has expired')
    base_cost = sum(costs.get(k, 0.) for k in
                    ('sample_io_s', 'input_s', 'ingress_s', 'egress_s', 'handoff_s', 'command_s'))
    earliest, latest = {}, {}
    for v, row in future.items():
        earliest[v] = max(now+planning_budget_s, row['minimum_release_s'],
                          row.get('previous_release_s', now-minimum_interval_s)+minimum_interval_s)
        service = base_cost + sum(profile[row['model']]['segment_estimates_s'])
        latest[v] = row['deadline_s']-service-reserve_s
    if any(latest[v] < earliest[v] for v in future):
        return None, dict(feasible=False, checked=0, reason='even isolated response cannot fit the current deadline',
                          historical_deadlines_repaired=False)
    # A common interpolation towards the earliest legal release makes this
    # O(K) WHOLE-GROUP trials, never K^N independent latest-release choices.
    attempts = []
    for alpha in ([1.] if max_checks == 1 else [1.-j/(max_checks-1) for j in range(max_checks)]):
        proposal = {v: {**r, 'planned_release_s': earliest[v]+alpha*(latest[v]-earliest[v])}
                    for v, r in future.items()}
        intervals = {v:r['planned_release_s']-r.get('previous_release_s', now-minimum_interval_s)
                     for v,r in proposal.items()}
        demands = resource_loads(intervals, {v:r['model'] for v,r in proposal.items()},
            {m:sum(p['segment_estimates_s']) for m,p in profile.items()}, costs)
        if max(demands.values()) > .95+1e-12:
            attempts.append(dict(alpha=alpha, feasible=False, resource_loads=demands,
                                 reason='sustained demand screen failed; not a scheduling theorem'))
            continue
        remaining = {**costs, 'frontend_reserved_until_s': now+planning_budget_s}
        feasible, finishes, events = simulate_pipeline(now, list(proposal.values()), pending, profile, remaining,
                                             policy='edf', inflight_limit=inflight_limit)
        # The full planning reserve is already in the forecast. A Python
        # operation can overrun; reject then, never move a checked release.
        total_lag = time.perf_counter()-started
        obligations = [*pending, *proposal.values()]
        fits = feasible and total_lag <= planning_budget_s and all(r['sample_id'] in finishes and finishes[r['sample_id']] <= r['deadline_s']
                   for r in obligations)
        attempts.append(dict(alpha=alpha, feasible=fits, planning_s=total_lag, resource_loads=demands))
        if fits:
            for v, row in proposal.items():
                row['requested_period_s'] = intervals[v]
            return proposal, dict(feasible=True, checked=len(attempts), attempts=attempts,
                                  events=events, planning_s=total_lag, is_WCET=False,
                                  historical_deadlines_repaired=False,
                                  scope='known finite jobs only; atomic publication validation still required')
        if total_lag > planning_budget_s:
            break
    return None, dict(feasible=False, checked=len(attempts), attempts=attempts,
                      planning_s=time.perf_counter()-started, historical_deadlines_repaired=False,
                      reason='bounded joint release candidates exhausted; not a proof of global infeasibility')
