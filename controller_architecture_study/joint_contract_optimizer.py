"""Explainable joint mode/release decisions using resource prices and EDF.

The dual bounds the supplied finite utility table, not true future flight J.
Execution costs remain empirical unless a caller supplies certified bounds.
"""

import math
import time

import numpy as np

from .framework_runtime import simulate_pipeline


def compile_options(now, future, options, profile, costs, budget_s, minimum_interval_s):
    """Freeze commitments; a candidate contains only a new mode and release."""
    vehicles = list(future)
    service = sum(costs.get(k, 0.) for k in
        ('sample_io_s', 'input_s', 'ingress_s', 'egress_s', 'handoff_s', 'command_s'))
    frontend = sum(costs.get(k, 0.) for k in ('input_s', 'output_s', 'planner_reserve_s', 'handoff_s'))
    choices, rejected = [], {}
    for v in vehicles:
        row, good, bad = future[v], [], []
        for action in options[v]:
            model, release, value = action['model'], action['release_s'], action['value']
            if not np.isfinite([release, value]).all() or model not in profile:
                raise ValueError('finite utilities/releases and a known model required')
            compute = sum(profile[model]['segment_estimates_s'])
            period = release-row['previous_release_s']
            reason = None
            if release < max(now+budget_s, row['minimum_release_s'])-1e-12 or period < minimum_interval_s-1e-12:
                reason = 'release_before_planning_or_minimum_interval'
            elif release+service+compute+costs.get('completion_guard_s', 0.) > row['deadline_s']+1e-12:
                reason = 'isolated_response_exceeds_unchanged_deadline'
            if reason:
                bad.append(dict(id=action['id'], reason=reason))
            else:
                good.append({**action, 'period_s':period, 'demand':np.array([compute/period, frontend/period])})
        choices.append(good)
        rejected[v] = bad
    return vehicles, choices, rejected


def priced_candidates(choices, capacities, iterations):
    """O(iterations*N*M) dual updates; no product over task option counts."""
    n, width = len(choices), max(map(len, choices))
    values = np.full((n, width), -np.inf)
    demand = np.zeros((n, width, 2))
    valid = np.zeros((n, width), bool)
    for i, row in enumerate(choices):
        for j, action in enumerate(row):
            values[i,j], demand[i,j], valid[i,j] = action['value'], action['demand'], True
    index = np.arange(n)
    prices = np.zeros(2)
    scale = max(sum(max(a['value'] for a in row)-min(a['value'] for a in row) for row in choices), 1e-4)
    candidates, trace = {}, []
    upper, best_dual_choice, best_prices = float('inf'), None, prices.copy()
    first = values.argmax(axis=1)

    def keep(selected, origin):
        load = demand[index, selected].sum(axis=0)
        if np.all(load <= capacities+1e-10):
            key = tuple(map(int, selected))
            candidates.setdefault(key, dict(indices=selected.copy(), value=float(values[index,selected].sum()),
                                           load=load.tolist(), origin=origin))

    for k in range(iterations):
        selected = (values-np.sum(demand*prices, axis=2)).argmax(axis=1)
        load = demand[index, selected].sum(axis=0)
        # L(lambda) = lambda*b + sum max(value-lambda*demand).
        bound = float(values[index,selected].sum()+prices@(capacities-load))
        if bound < upper:
            upper, best_dual_choice, best_prices = bound, selected.copy(), prices.copy()
        keep(selected, 'resource_price_iteration')
        trace.append(dict(iteration=k, prices=prices.tolist(), load=load.tolist(), dual_upper=bound))
        prices = np.maximum(0., prices+scale/math.sqrt(k+1)*(load-capacities))

    # Bounded primal repair: free resources with the least utility loss. It
    # changes whole-loop contracts, never a running job or its deadline.
    repairs = []
    for seed in (first, best_dual_choice):
        selected = seed.copy()
        for _ in range(2*n):
            load = demand[index,selected].sum(axis=0)
            violation = np.maximum(load-capacities, 0.)
            if not violation.any():
                break
            delta = demand[index,selected][:,None,:]-demand
            relief = np.sum(delta*violation, axis=2)
            eligible = valid & np.all(delta >= -1e-12, axis=2) & (relief > 1e-12)
            loss = values[index,selected][:,None]-values
            ratios = np.divide(loss, relief, out=np.full_like(loss, np.inf), where=eligible)
            if not np.isfinite(ratios).any():
                break
            i,j = np.unravel_index(np.argmin(ratios), ratios.shape)
            repairs.append(dict(loop=int(i), before=choices[i][selected[i]]['id'], after=choices[i][j]['id'],
                                utility_loss=float(loss[i,j]), resource_relief=delta[i,j].tolist()))
            selected[i] = j
        keep(selected, 'bounded_least_loss_resource_repair')
    return sorted(candidates.values(), key=lambda r:-r['value']), dict(
        resource_dual_upper=upper, resource_prices=best_prices.tolist(), trace=trace, repairs=repairs,
        selected_without_resource_constraints=first.tolist())


def optimize(now, future, pending, options, profile, costs, *, inflight_limit,
             capacities=(.95, .95), price_iterations=12, max_checks=3,
             planning_budget_s=.006, minimum_interval_s=.020, recover_after_miss=False):
    """Return a whole-group recommendation; caller must atomically validate it.

    Scope: finite next jobs, fixed segment EDF, and the provided utility prior.
    This function never modifies native calendars, promises or actual control.
    """
    start = time.perf_counter()
    if (not future or future.keys() != options.keys() or price_iterations < 1 or max_checks < 1
            or not math.isfinite(planning_budget_s) or planning_budget_s <= 0.
            or not math.isfinite(minimum_interval_s) or minimum_interval_s <= 0.
            or not math.isfinite(now) or len(capacities) != 2
            or not np.isfinite(capacities).all() or min(capacities) <= 0.):
        raise ValueError('matching nonempty loop/options and finite bounded planning settings required')
    ids = {r['sample_id'] for r in future.values()}
    if len(ids) != len(future) or ids.intersection(r['sample_id'] for r in pending):
        raise ValueError('a released/in-flight job cannot be reconfigured')
    if any(r.get('stage','input') != 'input' for r in future.values()):
        raise ValueError('only unpublished future jobs may change configuration')
    expired = [r['sample_id'] for r in [*future.values(), *pending]
               if 'command_completed_s' not in r and r['deadline_s'] <= now]
    expired_future = [r['sample_id'] for r in future.values() if r['deadline_s'] <= now]
    missed_pending = {r['sample_id'] for r in pending
                      if 'command_completed_s' not in r and r['deadline_s'] <= now}
    common = dict(historical_deadlines_repaired=False, deadlines_relaxed=False, installed=False,
                  scheduler='two-FC-segment EDF', physical_safety_certified=False, is_WCET=False,
                  previous_completed_misses=[r['sample_id'] for r in pending if
                      r.get('command_completed_s',-math.inf) > r['deadline_s']+1e-10],
                  recovery_enabled=recover_after_miss, existing_missed_ids=expired,
                  future_feasible=False, strict_timing_passed=False)
    if expired and (not recover_after_miss or expired_future):
        return None, {**common, 'accepted':False, 'reason':'existing_unfinished_deadline_expired', 'expired_ids':expired}
    ignored = missed_pending if recover_after_miss else set()
    vehicles, choices, rejected = compile_options(now, future, options, profile, costs,
                                                  planning_budget_s, minimum_interval_s)
    if any(not row for row in choices):
        return None, {**common, 'accepted':False, 'reason':'no_individually_time_feasible_contract',
                      'excluded':rejected, 'empty_loops':[v for v,r in zip(vehicles,choices) if not r]}
    caps = np.asarray(capacities)
    minimum_demand = np.sum([np.min([a['demand'] for a in row],axis=0) for row in choices],axis=0)
    if np.any(minimum_demand > caps+1e-10):
        return None, {**common, 'accepted':False, 'reason':'even_resource_lower_bound_exceeds_capacity',
                      'minimum_demand':minimum_demand.tolist(), 'excluded':rejected}
    candidates, dual = priced_candidates(choices, caps, price_iterations)
    # Retain an exact current-contract option as a checked fallback when it is
    # genuinely in the candidate set; never invent a fallback's utility.
    current = []
    for v, row in zip(vehicles, choices):
        matches = [j for j,a in enumerate(row) if a['model'] == future[v]['model']
                   and abs(a['release_s']-future[v]['planned_release_s']) < 1e-10]
        if not matches:
            current = []
            break
        current.append(matches[0])
    if current:
        load = np.sum([row[j]['demand'] for row,j in zip(choices,current)],axis=0)
        if np.all(load <= caps+1e-10):
            fallback = dict(indices=np.array(current), value=sum(row[j]['value'] for row,j in zip(choices,current)),
                            load=load.tolist(), origin='unchanged_current_contracts')
            if not any(np.array_equal(c['indices'],current) for c in candidates[:max_checks]):
                candidates = candidates[:max_checks-1]+[fallback]
    candidates.sort(key=lambda c:-c['value'])
    checks, best = [], None
    for candidate in candidates[:max_checks]:
        if time.perf_counter()-start >= planning_budget_s:
            break
        chosen = {v:row[j] for v,row,j in zip(vehicles, choices, candidate['indices'])}
        proposal = {v:{**future[v], 'model':a['model'], 'planned_release_s':a['release_s'],
                       'requested_period_s':a['period_s']} for v,a in chosen.items()}
        residual = {**costs, 'frontend_reserved_until_s':max(now+planning_budget_s,
                                                              costs.get('frontend_reserved_until_s',now))}
        feasible, finish, events = simulate_pipeline(now, list(proposal.values()), pending, profile,
            residual, policy='edf', inflight_limit=inflight_limit, missed_deadline_ids=ignored)
        covered = [*proposal.values(), *pending]
        passed = feasible and all('command_completed_s' in r or (r['sample_id'] in finish and
            (r['sample_id'] in ignored or
             finish[r['sample_id']]+costs.get('completion_guard_s',0.) <= r['deadline_s']+1e-10)) for r in covered)
        checks.append(dict(value=candidate['value'], origin=candidate['origin'], load=candidate['load'],
                           passed=passed, finish=finish, events=events))
        if passed and (best is None or candidate['value'] > best[0]):
            best = candidate['value'], proposal, chosen, finish
            # Candidates are ordered by utility. Later ones cannot improve a
            # feasible result; avoid paying for redundant full EDF forecasts.
            break
    elapsed = time.perf_counter()-start
    report = {**common, 'accepted':best is not None and elapsed <= planning_budget_s,
        'planning_ms':1000*elapsed, 'planning_budget_ms':1000*planning_budget_s,
        'checks':checks, 'excluded':rejected, 'candidate_groups':len(candidates), **dual,
        'scope':'finite supplied contracts and empirical pipeline; dual gap is NOT a bound on actual flight J'}
    if not report['accepted']:
        report['reason'] = 'planning_budget_exceeded' if elapsed > planning_budget_s else 'bounded_candidates_not_timing_feasible'
        return None, report
    value, proposal, chosen, finish = best
    report.update(value=value, duality_gap=max(0.,dual['resource_dual_upper']-value),
        future_feasible=True, strict_timing_passed=not expired and not common['previous_completed_misses'],
        acceptance_scope='future obligations only; past misses retained' if ignored else 'supplied finite obligations',
        missed_work_still_charged=sorted(ignored),
        decisions={v:dict(action=a['id'], model=a['model'], interval_ms=1000*a['period_s'],
            release_s=a['release_s'], unchanged_deadline_s=future[v]['deadline_s'], utility=a['value'],
            resource_demand=a['demand'].tolist(), source=a.get('source','caller_supplied'),
            quality_scope=a.get('quality_scope','not_certified_by_this_optimizer')) for v,a in chosen.items()})
    return proposal, report
