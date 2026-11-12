"""Bounded A/B/C repair for the deterministic PS/PL replay model."""
from dataclasses import replace
from functools import cache

from .bottleneck_joint import (
    blocking_contributions, chain, edit_cost, isolated_finish, make_jobs, simulate, window_demand, witness,
)


EPS = 1e-8


def path_cost(job, platform):
    return sum(cost for _, cost in chain(job, platform))


def path_deficit(job, platform):
    if job.command_completed:
        return float('-inf')
    return isolated_finish(job, platform) - job.deadline


def _service(job, resource, platform):
    cost = sum(duration for kind, duration in chain(job, platform) if kind == resource)
    return cost + platform.rearm if resource == 'ps' else cost


def diagnose(scenario, jobs):
    """Prefer an isolated path failure, then an overloaded resource window."""
    target = max(jobs, key=lambda job: path_deficit(job, scenario.platform), default=None)
    if target is not None:
        deficit = path_deficit(target, scenario.platform)
        if deficit > EPS:
            return {
                'type': 'A', 'job': target.name, 'deficit_ms': deficit,
                'latest_start_ms': target.deadline - path_cost(target, scenario.platform),
            }
    window = witness(scenario, jobs)
    if window['kind'] in ('ps', 'pl') and window['deficit'] > EPS:
        return {'type': 'B', **window}
    return {'type': 'C'}


def _schedule_candidates(scenario, policy, forecast, allow_guarded_edf):
    order, group = policy
    waits = forecast['waits'][forecast['worst']]
    alternatives = []
    if order == 'fifo':
        alternatives.append(('edf', group))
    if allow_guarded_edf and waits['pl'] > EPS and order == 'edf':
        alternatives.append(('guarded_edf', group))
    if waits['pl'] > EPS:
        alternatives.extend((order, g) for g in scenario.legal_groups if g < group)
    if waits['ps'] > EPS:
        alternatives.extend((order, g) for g in reversed(scenario.legal_groups) if g > group)
    return alternatives


def _lateness(forecast):
    return max((-slack for slack in forecast['slack'].values()), default=0.)


def abc_repair(scenario, *, max_checks=5, max_edits=8, allow_guarded_edf=False):
    """Return only a fully checked choice as accepted; retain rejected attempts."""
    platform = scenario.platform
    initial = (tuple(0 for _ in scenario.loops), scenario.initial_policy)
    plan = initial
    checked = {plan: simulate(scenario, *plan)}
    seen = {plan}
    schedule_tried = set()
    attempts = []
    diagnosis = None
    initial_kind = 'none'
    edits = 0
    stop_reason = 'no_effective_action'
    index = {loop.name: i for i, loop in enumerate(scenario.loops)}

    @cache
    def local(i, option, group):
        return make_jobs(replace(scenario, loops=(scenario.loops[i],), carry=()),
                         (option,), group)

    def jobs_for(choice, group):
        return scenario.carry + tuple(
            job for i, option in enumerate(choice) for job in local(i, option, group)
        )

    while True:
        choice, policy = plan
        order, group = policy
        forecast = checked.get(plan)
        if forecast and forecast['feasible']:
            stop_reason = 'accepted'
            break
        jobs = forecast['jobs'] if forecast else jobs_for(choice, group)
        diagnosis = diagnose(scenario, jobs)
        kind = diagnosis['type']
        if initial_kind == 'none':
            initial_kind = kind
        if len(checked) >= max_checks or edits >= max_edits:
            stop_reason = 'budget_exhausted'
            break

        if kind == 'C' and forecast is None:
            forecast = checked[plan] = simulate(scenario, *plan)
            if forecast['feasible']:
                stop_reason = 'accepted'
                break
            if len(checked) >= max_checks:
                stop_reason = 'budget_exhausted'
                break

        if kind == 'C' and choice not in schedule_tried:
            schedule_tried.add(choice)
            # Give schedule-only repair one forecast before changing T or c.
            for alternative in _schedule_candidates(scenario, policy, forecast, allow_guarded_edf):
                probe = (choice, alternative)
                if probe in seen:
                    continue
                seen.add(probe)
                result = checked[probe] = simulate(scenario, *probe)
                attempts.append({
                    'type': 'C', 'action': 'schedule_only', 'target': forecast['worst'],
                    'choice': choice, 'policy': alternative, 'passed': result['feasible'],
                    'lateness_gain_ms': _lateness(forecast) - _lateness(result),
                })
                if result['feasible'] or _lateness(result) < _lateness(forecast) - EPS:
                    plan = probe
                    edits += 1
                break
            if plan != (choice, policy):
                continue
            if len(checked) >= max_checks:
                stop_reason = 'budget_exhausted'
                break

        by_name = {job.name: job for job in jobs}
        target = by_name.get(diagnosis.get('job') if kind == 'A'
                             else forecast['worst'] if kind == 'C' else None)
        blockers = {}
        wait_end = {}
        if kind == 'C':
            for (name, resource), duration in blocking_contributions(forecast, target.name).items():
                blocker = by_name[name]
                blockers.setdefault(blocker.loop, []).append((blocker, resource, duration))
            for operation in forecast['timeline']:
                if operation['job'] == target.name and operation['ready'] < operation['start']:
                    resource = operation['resource']
                    wait_end[resource] = max(wait_end.get(resource, 0.), operation['start'])

        best = None

        def consider(candidate, gain, action, *, still_late=0.):
            nonlocal best
            if gain <= EPS or candidate in seen:
                return
            candidate_choice, candidate_policy = candidate
            disturbance = edit_cost(scenario, candidate_choice, candidate_policy)
            # For A, prefer the smallest change that clears the isolated path.
            if kind == 'A' and still_late <= EPS:
                key = (0, disturbance, -gain)
            else:
                key = (1, -gain, disturbance)
            if best is None or key < best[0]:
                best = (key, candidate, gain, action)

        for i, loop in enumerate(scenario.loops):
            if kind == 'A' and index.get(target.loop) != i:
                continue
            if kind == 'C' and loop.name != target.loop and loop.name not in blockers:
                continue
            old_jobs = local(i, choice[i], group)
            old_deficit = max(path_deficit(job, platform) for job in old_jobs)
            old_demand = window_demand(old_jobs, platform, diagnosis) if kind == 'B' else 0.
            for k, option in enumerate(loop.options):
                if k == choice[i] or not option.qualified:
                    continue
                before = loop.options[choice[i]]
                action = ('rate' if option.mode == before.mode else
                          'mode' if option.period == before.period else 'rate_mode')
                new_jobs = local(i, k, group)
                new_deficit = max(path_deficit(job, platform) for job in new_jobs)
                if new_deficit > max(0., old_deficit) + EPS:
                    continue
                candidate = (choice[:i] + (k,) + choice[i+1:], policy)
                if kind == 'B':
                    consider(candidate, old_demand - window_demand(new_jobs, platform, diagnosis), action)
                    continue
                new_by_name = {job.name: job for job in new_jobs}
                changed = new_by_name.get(target.name)
                if kind == 'A':
                    if changed is not None:
                        remaining = path_deficit(changed, platform)
                        consider(candidate, diagnosis['deficit_ms'] - remaining,
                                 action, still_late=max(remaining, new_deficit))
                    continue
                gain = path_deficit(target, platform) - path_deficit(changed, platform) if changed else 0.
                for blocker, resource, duration in blockers.get(loop.name, ()):
                    successor = new_by_name.get(blocker.name)
                    if successor is None or successor.release >= wait_end[resource]:
                        gain += duration
                    else:
                        # Rank only. Moving a task can change every downstream wait.
                        old = _service(blocker, resource, platform)
                        new = _service(successor, resource, platform)
                        if old > EPS:
                            gain += duration * (old - new) / old
                consider(candidate, gain, action)

        if kind == 'A' and target.loop in index:
            for larger in scenario.legal_groups:
                if larger > group:
                    revised = next(job for job in local(index[target.loop], choice[index[target.loop]], larger)
                                   if job.name == target.name)
                    remaining = path_deficit(revised, platform)
                    consider((choice, (order, larger)), diagnosis['deficit_ms'] - remaining,
                             'merge_segments', still_late=remaining)

        if kind == 'B' and diagnosis['kind'] == 'ps':
            old_demand = window_demand(jobs, platform, diagnosis)
            for larger in scenario.legal_groups:
                if larger > group:
                    gain = old_demand - window_demand(jobs_for(choice, larger), platform, diagnosis)
                    consider((choice, (order, larger)), gain, 'merge_segments')

        if best is None:
            break
        _, plan, gain, action = best
        seen.add(plan)
        edits += 1
        attempts.append({**diagnosis, 'action': action, 'gain_ms': gain,
                         'choice': plan[0], 'policy': plan[1]})

    # An unchecked plan is never returned with the result of an older plan.
    if plan not in checked:
        plan = min(checked, key=lambda candidate: _lateness(checked[candidate]))
    forecast = checked[plan]
    return {
        'accepted': forecast['feasible'], 'choice': plan[0], 'policy': plan[1],
        'checks': len(checked), 'initial_type': initial_kind, 'attempts': attempts,
        'stop_reason': stop_reason, 'forecast': forecast,
    }
