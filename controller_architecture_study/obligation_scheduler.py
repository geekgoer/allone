"""Bounded finite-calendar scheduling for one non-preemptive DPU and one PS.

All timestamps are caller-supplied milliseconds on one clock. A successful
plan is conditional on its supplied costs and control qualifications.
"""
from dataclasses import dataclass, replace
import math

from .bottleneck_joint import Job, Platform, Scenario, simulate, witness


EPS = 1e-8


@dataclass(frozen=True)
class ControlJob:
    name: str
    vehicle: str
    release: float
    prep_ready: float
    prep_ms: float
    post_ms: float
    transport_ms: float
    slot_deadline: float
    hold_deadline: float
    fresh_deadline: float
    allowed_modes: tuple
    current_mode: int
    release_options: tuple = ()  # (absolute release, absolute slot deadline)
    recovery: bool = False
    mode_locked: bool = False
    periodic_edit: bool = False  # retime following supplied slots at the new period

    @property
    def deadline(self):
        return min(self.slot_deadline, self.hold_deadline, self.fresh_deadline)


@dataclass(frozen=True)
class SkipPermission:
    job: str
    successor: str
    hold_until: float
    valid_until: float
    eligible: bool
    evidence: str


@dataclass(frozen=True)
class Calendar:
    jobs: tuple
    revision: int = 0
    pending: tuple = ()  # (vehicle, successor job name)

    def commit(self, decision, now):
        """Atomic replacement in this immutable offline state machine."""
        if not decision['accepted'] or decision['revision'] != self.revision:
            raise ValueError('Rejected or stale decision')
        if now > decision['commit_by'] + EPS:
            raise ValueError('Decision expired')
        jobs = decision['jobs']
        if decision['skipped']:
            permission = decision['permission']
            before = {job.name: job for job in self.jobs}
            target = before[permission.job]
            if target.release <= now + EPS or now > permission.valid_until:
                raise ValueError('Skip is no longer retractable')
        # Slot edits only apply to unreleased work.
        changed = {job.name: job for job in jobs}
        for old in self.jobs:
            if old.name in changed and old.release != changed[old.name].release and old.release <= now + EPS:
                raise ValueError('Cannot retime a published job')
        return Calendar(jobs, self.revision + 1, decision['pending'])

    def delivered(self, name, actuation_time):
        job = next(job for job in self.jobs if job.name == name)
        if not math.isfinite(actuation_time) or not job.release <= actuation_time <= job.deadline + EPS:
            raise ValueError('Late delivery cannot discharge a control obligation')
        pending = tuple(item for item in self.pending if item[1] != name)
        return Calendar(tuple(j for j in self.jobs if j.name != name), self.revision + 1, pending)

    def tighten_hold(self, vehicle, deadline):
        """External state/control analysis supplies a new absolute hold bound."""
        if not math.isfinite(deadline) or not any(j.vehicle == vehicle for j in self.jobs):
            raise ValueError('Expected a finite bound for an existing vehicle')
        jobs = tuple(replace(j, hold_deadline=min(j.hold_deadline, deadline))
                     if j.vehicle == vehicle else j for j in self.jobs)
        return Calendar(jobs, self.revision+1, self.pending)


def fastest(job, modes):
    pool = (job.current_mode,) if job.mode_locked else job.allowed_modes
    return min(pool, key=lambda key: (modes[key].service_ms, modes[key].selection_mse, key))


def evaluate(jobs, order, choice, modes, now, decision_ms, blocked_until):
    """Reuse the PS/PL simulator, charging shared PS before and after inference."""
    service = []
    for job in jobs:
        model = choice[job.name]
        operations = (('ps', job.prep_ms), ('pl', modes[model].service_ms),
                      ('ps', job.post_ms), ('delay', job.transport_ms))
        service.append(Job(job.name, job.vehicle, max(0., job.prep_ready-now),
                           job.deadline-now, (), str(model), operations))
    platform = Platform(decision=decision_ms, switch=0., rearm=0.,
                        initial_pl_block=max(0., blocked_until-now))
    scenario = Scenario((), platform, horizon=0., carry=tuple(service), legal_groups=(1,))
    report = simulate(scenario, (), ('edf', 1), pl_order=order)
    if set(report['command_finish']) != {job.name for job in jobs}:
        raise RuntimeError('Dispatch order failed to drain')
    overlap = []
    for vehicle in {job.vehicle for job in jobs}:
        loop = sorted((j for j in jobs if j.vehicle == vehicle), key=lambda j: j.release)
        for old, new in zip(loop, loop[1:]):
            if now+report['finish'][old.name] > new.release+EPS:
                overlap.append(old.name)
    return dict(accepted=report['feasible'] and not overlap, overlap=overlap, forecast=report)


def tail_budget(jobs, order, choice, modes, now):
    """Optimistic PS-ready estimate only screens candidates; evaluate approves."""
    by_name = {job.name: job for job in jobs}
    latest = math.inf
    for name in reversed(order):
        job = by_name[name]
        latest = min(job.deadline-job.post_ms-job.transport_ms, latest) - modes[choice[name]].service_ms
        if max(now, job.prep_ready)+job.prep_ms > latest+EPS:
            return None
    return latest-now


def dispatch_budget(jobs, order, choice, modes, now):
    head = next(job for job in jobs if job.name == order[0])
    tail = tail_budget(jobs, order[1:], choice, modes, now)
    if tail is None:
        return None
    end = min(head.deadline-head.post_ms-head.transport_ms, now+tail)
    return end-(max(now, head.prep_ready)+head.prep_ms)


def diagnosis(jobs, choice, modes, now, decision_ms, blocked_until):
    platform = Platform(prep=0., dispatch=0., post=0., ingress=0., tail=0.,
                        decision=decision_ms, initial_pl_block=max(0., blocked_until-now))
    demand = tuple(Job(j.name, j.vehicle, max(0., j.prep_ready-now), j.deadline-now, (), '',
                       (('ps', j.prep_ms), ('pl', modes[choice[j.name]].service_ms),
                        ('ps', j.post_ms), ('delay', j.transport_ms))) for j in jobs)
    result = witness(Scenario((), platform, carry=demand), demand)
    kind = 'A' if result['kind'] == 'path' else 'B' if result['deficit'] > EPS else 'C'
    return dict(type=kind, **result)


def validate(calendar, modes):
    names = [job.name for job in calendar.jobs]
    if len(names) != len(set(names)) or not names:
        raise ValueError('Expected nonempty uniquely named obligations')
    for job in calendar.jobs:
        numbers = (job.release, job.prep_ready, job.prep_ms, job.post_ms,
                   job.transport_ms, job.slot_deadline, job.hold_deadline, job.fresh_deadline)
        if (not all(math.isfinite(x) for x in numbers)
                or min(job.prep_ms, job.post_ms, job.transport_ms) < 0
                or job.prep_ready < job.release or not job.allowed_modes
                or job.current_mode not in job.allowed_modes
                or not set(job.allowed_modes) <= modes.keys()):
            raise ValueError(f'Invalid obligation {job.name}')
        if any(not all(math.isfinite(x) for x in pair) or pair[1] < pair[0]
               for pair in job.release_options):
            raise ValueError('Invalid release option')
    for model in modes.values():
        if (not math.isfinite(model.service_ms) or model.service_ms <= 0
                or not math.isfinite(model.selection_mse) or model.selection_mse < 0):
            raise ValueError('Invalid model profile')
    for vehicle, successor in calendar.pending:
        if not any(j.name == successor and j.vehicle == vehicle and j.recovery for j in calendar.jobs):
            raise ValueError('Missing recovery obligation')


def plan(calendar, modes, *, now, decision_ms=.5, blocked_until=0.,
         permissions=(), max_checks=8, max_candidates=12, enable_skip=True,
         enable_retime=True, enable_reorder=True, enable_modes=True):
    """Keep-first repair with bounded candidate and full PS/PL check counts.

    Candidate order: EDF/current, EDF/fastest, up to two dispatch alternatives,
    legal one-slot release edits, then at most one skip plus its successor.
    This ordering is explicit and intentionally heuristic.
    """
    validate(calendar, modes)
    if max_checks < 1 or max_candidates < 1 or not all(
            math.isfinite(x) for x in (now, decision_ms, blocked_until)) or decision_ms < 0:
        raise ValueError('Invalid decision budget')
    modes = dict(modes)
    initial = tuple(sorted(calendar.jobs, key=lambda j: (j.deadline, j.release, j.name)))
    order = tuple(job.name for job in initial)
    base = {job.name: job.current_mode for job in initial}
    fast = {job.name: fastest(job, modes) if enable_modes else job.current_mode for job in initial}
    checks = candidates = 0
    trace = []
    commit_time = now+decision_ms
    approved = None
    initial_diagnosis = diagnosis(initial, base, modes, now, decision_ms, blocked_until)

    def attempt(jobs, sequence, choice, action, permission=None):
        nonlocal checks, candidates, approved
        if candidates >= max_candidates or checks >= max_checks:
            return
        candidates += 1
        # Envelopes include PS competition only in the full evaluation.
        screen = dispatch_budget(jobs, sequence, choice, modes, max(commit_time, blocked_until))
        delta = None
        if permission:
            original = (sequence[0],) + tuple(name for name in order if name != sequence[0])
            original_choice = {**fast, **choice}
            before = dispatch_budget(initial, original, original_choice, modes,
                                     max(commit_time, blocked_until))
            if before is not None and screen is not None and min(before, screen) >= 0.:
                delta = screen-before
        checks += 1
        result = evaluate(jobs, sequence, choice, modes, now, decision_ms, blocked_until)
        trace.append(dict(action=action, accepted=result['accepted'], optimistic_head_budget_ms=screen,
                          skip_net_head_budget_ms=delta,
                          misses=result['forecast']['misses'], overlap=result['overlap']))
        if not result['accepted']:
            return
        pending = calendar.pending
        if permission:
            target = next(j for j in calendar.jobs if j.name == permission.job)
            pending += ((target.vehicle, permission.successor),)
        approved = dict(accepted=True, revision=calendar.revision,
                        jobs=tuple(replace(job, current_mode=choice[job.name]) for job in jobs),
                        order=sequence, choice=choice, pending=pending,
                        skipped=permission.job if permission else None, permission=permission,
                        commit_by=min((permission.valid_until if permission else math.inf),
                                      min((job.release for job in jobs
                                           if any(old.name == job.name and old.release != job.release
                                                  for old in calendar.jobs)), default=math.inf),
                                      commit_time),
                        forecast=result['forecast'], action=action)

    attempt(initial, order, base, 'keep_current')
    if approved is None and fast != base:
        attempt(initial, order, fast, 'faster_modes')
    if approved is None and enable_reorder and initial_diagnosis['type'] == 'C':
        # At most two alternative heads, then deterministic EDF tail.
        heads = sorted(initial, key=lambda j: (max(commit_time, j.prep_ready)+j.prep_ms, j.name))[:2]
        for head in heads:
            sequence = (head.name,) + tuple(name for name in order if name != head.name)
            if sequence != order:
                attempt(initial, sequence, fast, 'reorder')
            if approved:
                break
    if approved is None and enable_retime:
        for target in initial:
            if target.release <= commit_time+EPS or target.recovery:
                continue
            for release, slot in target.release_options:
                if release <= commit_time+EPS:
                    continue
                moved = replace(target, release=release, slot_deadline=slot,
                                prep_ready=release+(target.prep_ready-target.release),
                                fresh_deadline=release+(target.fresh_deadline-target.release))
                jobs = tuple(moved if job.name == target.name else job for job in initial)
                if target.periodic_edit:
                    following = sorted((j for j in initial if j.vehicle == target.vehicle
                                        and j.release > target.release), key=lambda j: j.release)
                    mapping = {}
                    for k, job in enumerate(following, start=1):
                        new_release = release+k*(slot-release)
                        mapping[job.name] = replace(
                            job, release=new_release,
                            prep_ready=new_release+(job.prep_ready-job.release),
                            slot_deadline=new_release+(slot-release),
                            fresh_deadline=new_release+(job.fresh_deadline-job.release))
                    jobs = tuple(mapping.get(job.name, job) for job in jobs)
                sequence = tuple(j.name for j in sorted(jobs, key=lambda j: (j.deadline, j.release, j.name)))
                attempt(jobs, sequence, fast, 'release_edit')
                if approved or checks >= max_checks or candidates >= max_candidates:
                    break
            if approved or checks >= max_checks or candidates >= max_candidates:
                break
    if approved is None and enable_skip:
        by_name = {job.name: job for job in initial}
        eligible = []
        for permission in permissions:
            target, successor = by_name.get(permission.job), by_name.get(permission.successor)
            if (not permission.eligible or not permission.evidence or target is None or successor is None
                    or not math.isfinite(permission.hold_until) or not math.isfinite(permission.valid_until)
                    or target.recovery or target.release <= commit_time+EPS
                    or permission.valid_until < commit_time
                    or target.vehicle != successor.vehicle or successor.release <= target.release
                    or any(j.vehicle == target.vehicle and target.release < j.release < successor.release
                           for j in initial)
                    or any(v == target.vehicle for v, _ in calendar.pending)):
                continue
            eligible.append(permission)
        # Greatest actual saved service first; qualification and all successors
        # must pass. Only one skip is committed by a decision.
        eligible.sort(key=lambda p: (-modes[fast[p.job]].service_ms, p.job))
        for permission in eligible:
            jobs = tuple(
                replace(job, recovery=True, hold_deadline=min(job.hold_deadline, permission.hold_until))
                if job.name == permission.successor else job
                for job in initial if job.name != permission.job)
            sequence = tuple(job.name for job in sorted(jobs, key=lambda j: (j.deadline, j.release, j.name)))
            attempt(jobs, sequence, {j.name: fast[j.name] for j in jobs}, 'skip_recover', permission)
            if approved is None and enable_reorder:
                for head in sorted(jobs, key=lambda j: (max(commit_time, j.prep_ready)+j.prep_ms, j.name))[:2]:
                    alternative = (head.name,) + tuple(name for name in sequence if name != head.name)
                    if alternative != sequence:
                        attempt(jobs, alternative, {j.name: fast[j.name] for j in jobs},
                                'skip_recover_reorder', permission)
                    if approved or checks >= max_checks or candidates >= max_candidates:
                        break
            if approved or checks >= max_checks or candidates >= max_candidates:
                break
    # One quality upgrade at the next dispatched job, with full revalidation.
    if approved and enable_modes and checks < max_checks and candidates < max_candidates:
        saved = approved
        head = next(j for j in saved['jobs'] if j.name == saved['order'][0])
        if not head.mode_locked:
            budget = dispatch_budget(saved['jobs'], saved['order'], saved['choice'], modes,
                                     max(commit_time, blocked_until))
            fitting = [key for key in head.allowed_modes
                       if budget is not None and modes[key].service_ms <= budget+EPS]
            best = min(fitting, key=lambda key: (modes[key].selection_mse, modes[key].service_ms, key),
                       default=saved['choice'][head.name])
            if modes[best].selection_mse < modes[saved['choice'][head.name]].selection_mse:
                upgraded = {**saved['choice'], head.name: best}
                attempt(saved['jobs'], saved['order'], upgraded, 'quality_upgrade', saved['permission'])
                if not trace[-1]['accepted']:
                    approved = saved
    if approved is None:
        approved = dict(accepted=False, revision=calendar.revision, action='no_feasible_checked_plan')
    approved.update(checks=checks, candidates=candidates, trace=trace,
                    diagnosis=initial_diagnosis,
                    scope='finite PS/PL replay under supplied service and control qualifications')
    return approved
