"""Live adapter for the finite-calendar controller-obligation scheduler."""
from dataclasses import replace
import time

from .ip37_joint_runtime import JointRuntime
from .obligation_scheduler import Calendar, ControlJob, SkipPermission, plan


class ObligationRuntime(JointRuntime):
    def __init__(self, *args, backend, candidate_ids, fresh_ms=80., network_ms=3.,
                 enable_skip=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.backend = backend
        self.candidate_ids = tuple(candidate_ids)
        self.fresh_ms = fresh_ms
        self.enable_skip = enable_skip
        self.modes = {key: replace(backend.modes[key], service_ms=backend.modes[key].service_ms+network_ms)
                      for key in self.candidate_ids}
        self.future_modes = {}
        self.dispatch_order = ()
        self.dispatch_expires_s = 0.
        self.pending_recovery = {}
        self.planned_skip = {}
        self.qualification = {}

    def observe_phase(self, vehicle, sample_s, stable, evidence):
        with self.cv:
            self.qualification[vehicle] = dict(sample_s=sample_s, stable=stable, evidence=evidence,
                                               control_version=self.loops[vehicle]['control_version'])

    def observe(self, name, sample_s, deadline_s, analysis, version):
        with self.cv:
            old = self.loops[name]['observation']
            accepted = super().observe(name, sample_s, deadline_s, analysis, version)
            if accepted and old is not None and deadline_s < old['deadline_s']:
                self.dispatch_order = ()
            return accepted

    def release_due(self, now, *, stop_before=float('inf')):
        with self.cv:
            for vehicle, loop in self.loops.items():
                if (not loop['retired'] and loop['next'] <= now and loop['next'] < stop_before
                        and self.planned_skip.get(vehicle) == loop['cycle']):
                    self.rows.append(dict(vehicle=vehicle, cycle=loop['cycle'], release_s=loop['next'],
                                          published_s=now, period_ms=loop['period_ms'],
                                          deadline_s=loop['next']+loop['period_ms']/1000,
                                          model_id=self.model_id, status='planned_skip'))
                    loop.update(previous=loop['next'], next=loop['next']+loop['period_ms']/1000,
                                cycle=loop['cycle']+1)
                    del self.planned_skip[vehicle]
                    self.revision += 1
            rows = super().release_due(now, stop_before=stop_before)
            for row in rows:
                key = f"{row['vehicle']}:{row['cycle']}"
                row['model_id'] = self.future_modes.pop(key, self.model_id)
                recovery = self.pending_recovery.get(row['vehicle'])
                if recovery and recovery['cycle'] == row['cycle']:
                    row['recovery_due_s'] = recovery['deadline_s']
            return rows

    def finish(self, row):
        with self.cv:
            recovery = self.pending_recovery.get(row['vehicle'])
            if recovery and recovery['cycle'] == row['cycle'] and 'command_return_s' in row:
                row['recovery_on_time'] = row['command_return_s'] <= recovery['deadline_s']
                if row['recovery_on_time']:
                    del self.pending_recovery[row['vehicle']]
            super().finish(row)

    def board_done(self, row, now, **kwargs):
        with self.cv:
            key = row['model_id']
            self.modes[key] = replace(self.modes[key], service_ms=max(
                self.modes[key].service_ms, 1000*(now-row['board_start_s'])))
            super().board_done(row, now, **kwargs)

    def take(self, now):
        with self.cv:
            if self.active is not None:
                return None
            live = []
            for row in self.queue:
                due = row.get('actuation_deadline_s', row['deadline_s'])
                if row['future'].cancelled() or due <= now:
                    row['future'].cancel()
                else:
                    live.append(row)
            self.queue = live
            if not live:
                return None
            if now > self.dispatch_expires_s:
                self.dispatch_order = ()
            by_name = {f"{r['vehicle']}:{r['cycle']}": r for r in live}
            if self.adaptive and self.dispatch_order:
                head = self.dispatch_order[0]
                if head in by_name:
                    row = by_name[head]
                else:
                    # Do not wait for an expired or retired promised job.
                    vehicle, cycle = head.rsplit(':', 1)
                    loop = self.loops[vehicle]
                    obligation_exists = (not loop['retired'] and
                                         (loop['cycle'] <= int(cycle) or
                                          loop['busy'] is not None and loop['busy']['cycle'] == int(cycle)))
                    if obligation_exists and all(r.get('actuation_deadline_s', r['deadline_s']) >
                                                 now+self.modes[r['model_id']].service_ms/1000
                                                 for r in live):
                        return None
                    self.dispatch_order = ()
                    row = min(live, key=lambda r: r.get('actuation_deadline_s', r['deadline_s']))
            else:
                row = min(live, key=lambda r: r.get('actuation_deadline_s', r['deadline_s']))
            name = f"{row['vehicle']}:{row['cycle']}"
            self.dispatch_order = tuple(key for key in self.dispatch_order if key != name)
            row['model_id'] = self.future_modes.pop(name, row['model_id'])
            # Every dispatch also checks its own remaining real response budget.
            budget = 1000*(row.get('actuation_deadline_s', row['deadline_s'])-now)-self.command_reserve_ms
            affordable = [key for key in self.candidate_ids if self.modes[key].service_ms <= budget]
            if not affordable or (not self.adaptive and row['model_id'] not in affordable):
                self.queue.remove(row)
                row['dispatch_rejection'] = 'no_model_fits_remaining_delivery_budget'
                row['future'].cancel()
                self.revision += 1
                return None
            if self.adaptive and self.modes[row['model_id']].service_ms > budget and affordable:
                row['model_id'] = min(affordable, key=lambda key: self.modes[key].selection_mse)
            self.queue.remove(row)
            row.update(board_start_s=now, phase='board', configuration_id=row['model_id'])
            self.active = row
            self.revision += 1
            return row

    def _calendar(self, now):
        jobs, permissions = [], []
        blocked_until = 0.
        if self.active is not None and 1000*(now-self.active['board_start_s']) > self.modes[self.active['model_id']].service_ms:
            # An exceeded empirical envelope gives no valid remaining-time bound.
            return None
        for vehicle, loop in self.loops.items():
            if loop['retired']:
                continue
            observation = loop['observation']
            if observation is None or observation['deadline_s'] <= now:
                return None
            hold = observation['deadline_s']
            busy = loop['busy']
            next_release = loop['next']
            if next_release <= now:
                return None
            if busy is not None:
                stage = busy.get('phase', 'sampling')
                if stage in ('board', 'post'):
                    # Reserve an active request and its remaining delivery before
                    # the next per-vehicle slot, rather than simulating it twice.
                    remaining = (max(0., self.modes[busy['model_id']].service_ms-
                                     1000*(now-busy['board_start_s'])) if stage == 'board' else 0.)
                    if 'command_return_s' not in busy and (
                            now+(remaining+self.post_ms)/1000 > min(next_release, hold, busy['deadline_s'])):
                        return None
                    if 'command_return_s' in busy and now+self.rearm_ms/1000 > next_release:
                        return None
                    blocked_until = max(blocked_until, remaining)
                elif 'command_return_s' not in busy:
                    ready = 0.
                    prep = 0. if stage == 'queued' else self.prep_ms
                    fresh = busy.get('sample_start_s', busy['release_s'])+self.fresh_ms/1000
                    jobs.append(ControlJob(
                        f"{vehicle}:{busy['cycle']}", vehicle, 1000*(busy['release_s']-now),
                        ready, prep, self.post_ms, 0., 1000*(busy['deadline_s']-now),
                        1000*(min(hold, busy.get('recovery_due_s', hold))-now),
                        1000*(fresh-now), self.candidate_ids, busy['model_id'],
                        recovery=self.pending_recovery.get(vehicle, {}).get('cycle') == busy['cycle']))
            period = loop['period_ms']/1000
            first_name = f"{vehicle}:{loop['cycle']}"
            pending = self.pending_recovery.get(vehicle)
            recovery = pending is not None and pending['cycle'] == loop['cycle']
            first = ControlJob(first_name, vehicle, 1000*(next_release-now), 1000*(next_release-now),
                               self.prep_ms, self.post_ms, 0., 1000*(next_release+period-now),
                               1000*(min(hold, pending['deadline_s'] if recovery else hold)-now),
                               1000*(next_release-now)+self.fresh_ms, self.candidate_ids,
                               self.future_modes.get(first_name, self.model_id), recovery=recovery)
            if not recovery:
                first = replace(first, periodic_edit=True, release_options=tuple(
                    (1000*(loop['previous']+t/1000-now),
                     1000*(loop['previous']+2*t/1000-now))
                    for t in self.periods if t != loop['period_ms']
                    and loop['previous']+t/1000 > now+self.decision_ms/1000))
            jobs.append(first)
            # Future hold is conditional on delivery of the first update. A skip
            # explicitly replaces this with the currently held command's bound.
            successor = replace(first, name=f"{vehicle}:{loop['cycle']+1}",
                                release=first.release+1000*period, prep_ready=first.release+1000*period,
                                slot_deadline=first.slot_deadline+1000*period,
                                hold_deadline=first.slot_deadline+1000*period,
                                fresh_deadline=first.fresh_deadline+1000*period,
                                release_options=(), recovery=False, periodic_edit=False)
            jobs.append(successor)
            phase = self.qualification.get(vehicle)
            eligible = (self.enable_skip and not pending and busy is None and phase is not None
                        and phase['stable'] and now-phase['sample_s'] <= self.fresh_ms/1000
                        and phase['control_version'] == loop['control_version']
                        and vehicle not in self.planned_skip)
            permissions.append(SkipPermission(first.name, successor.name, 1000*(hold-now),
                                               first.release, bool(eligible),
                                               phase['evidence'] if phase else 'unknown phase'))
        pending_keys = tuple((v, f"{v}:{p['cycle']}") for v, p in self.pending_recovery.items()
                             if any(j.name == f"{v}:{p['cycle']}" for j in jobs))
        return Calendar(tuple(jobs), self.revision, pending_keys), permissions, blocked_until

    def plan_once(self):
        if time.perf_counter() < self.origin:
            return None
        started = time.perf_counter()
        with self.cv:
            snapshot = self._calendar(started)
            modes = dict(self.modes)
        if snapshot is None:
            self.skips['obligation_context_not_ready'] += 1
            return None
        calendar, permissions, blocked = snapshot
        decision = plan(calendar, modes, now=0., decision_ms=self.decision_ms,
                        blocked_until=blocked, permissions=permissions, max_checks=8)
        updates, model_updates, status = [], [], 'no_feasible_checked_plan'
        with self.cv:
            done = time.perf_counter()
            if decision['accepted']:
                if self.revision != calendar.revision:
                    status = 'context_changed'
                elif (done-started)*1000 > decision['commit_by']:
                    status = 'decision_budget_exceeded'
                else:
                    installed = calendar.commit(decision, (done-started)*1000)
                    old_jobs = {job.name: job for job in calendar.jobs}
                    for job in installed.jobs:
                        vehicle, cycle = job.name.rsplit(':', 1)
                        loop = self.loops[vehicle]
                        self.future_modes[job.name] = job.current_mode
                        if job.current_mode != old_jobs[job.name].current_mode:
                            model_updates.append(dict(job=job.name, previous=old_jobs[job.name].current_mode,
                                                      selected=job.current_mode))
                        if int(cycle) != loop['cycle']:
                            continue
                        release = started+job.release/1000
                        if abs(release-loop['next']) > 1e-7:
                            period = 1000*(release-loop['previous'])
                            loop.update(next=release, period_ms=period)
                            updates.append(dict(vehicle=vehicle, period_ms=period,
                                                next_release_s=release, first_cycle=int(cycle)))
                    if decision['skipped']:
                        permission = decision['permission']
                        vehicle, cycle = permission.job.rsplit(':', 1)
                        self.planned_skip[vehicle] = int(cycle)
                        self.pending_recovery[vehicle] = dict(cycle=int(permission.successor.rsplit(':', 1)[1]),
                                                             deadline_s=started+permission.hold_until/1000)
                    self.dispatch_order = decision['order']
                    self.dispatch_expires_s = done+max(0., min(decision['forecast']['slack'].values()))/1000
                    self.revision += 1
                    status = ('installed' if updates or model_updates or decision['skipped']
                              or decision['action'] == 'reorder' else 'plan_refreshed')
                    self.cv.notify_all()
            record = dict(snapshot_s=started, done_s=done, elapsed_ms=1000*(done-started),
                          status=status, updates=updates, checks=decision['checks'],
                          model_updates=model_updates,
                          action=decision['action'], initial_type=decision['diagnosis']['type'],
                          attempts=decision['trace'], choice=decision.get('choice'),
                          planned_skip=decision.get('skipped'), pending_recovery=dict(self.pending_recovery))
            self.decisions.append(record)
        return record
