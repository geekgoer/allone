"""Live calendar, whole-task EDF and bounded ABC adapter for one IP37 worker.

Times at the live boundary are absolute perf_counter seconds; the replay uses
milliseconds relative to its snapshot. Costs are empirical, not certified WCET.
"""
from concurrent.futures import Future
from collections import Counter
from dataclasses import asdict
import threading
import time

from .bottleneck_abc import abc_repair
from .bottleneck_joint import Job, Loop, Mode, Option, Platform, Scenario


class JointRuntime:
    def __init__(self, names, origin, period_ms, periods_ms, model_id, cost_ms,
                 *, adaptive=True, order='edf', prep_ms=3., post_ms=2., decision_ms=3., initial_periods=None, qualified_periods=None):
        costs = (period_ms, cost_ms, prep_ms, post_ms, decision_ms, *periods_ms)
        if (not names or len(set(names)) != len(names) or order not in ('edf', 'fifo')
                or any(not 0. < x < float('inf') for x in costs)
                or period_ms not in periods_ms):
            raise ValueError('unique vehicles, positive costs and a supported initial period required')
        self.cv = threading.Condition(threading.RLock())
        self.origin, self.model_id, self.cost_ms = origin, model_id, cost_ms
        self.periods = tuple(sorted(set(periods_ms)))
        self.adaptive, self.order = adaptive, order
        self.prep_ms, self.post_ms, self.decision_ms = prep_ms, post_ms, decision_ms
        self.front_wait_ms = 0.
        self.command_reserve_ms = post_ms  # operational guard, not the whole post-refresh phase
        self.rearm_ms = post_ms  # empirical seed until a worker completion has been observed
        self.loops = {v: dict(period_ms=period_ms, previous=origin+i*period_ms/1000/len(names)-period_ms/1000,
                             next=origin+i*period_ms/1000/len(names), cycle=0, busy=None,
                             observation=None, control_version=0, retired=False)
                      for i, v in enumerate(names)}
        self.rows, self.decisions, self.queue = [], [], []
        self.planner_calls = []
        self.skips = Counter()
        self.active = None
        self.revision = 0
        self.closed = False
        self.native_io = self.native_end = None
        if initial_periods is not None:
            if (set(initial_periods) != set(names) or
                    any(t not in self.periods for t in initial_periods.values())):
                raise ValueError('initial periods must cover exactly the selected vehicles and supported periods')
            for i, (name, loop) in enumerate(self.loops.items()):
                period = initial_periods[name]
                release = origin+i*period/1000/len(names)
                loop.update(period_ms=period, previous=release-period/1000, next=release)
        self.qualified_periods = qualified_periods
        if qualified_periods is not None:
            if (set(qualified_periods) != set(names) or
                    any(not values or any(t not in self.periods for t in values) or
                        self.loops[name]['period_ms'] not in values for name, values in qualified_periods.items())):
                raise ValueError('qualified periods must cover every loop and its starting period')

    def release_due(self, now, *, stop_before=float('inf')):
        """Called by the independent calendar; busy IO drops a release, never moves it."""
        admitted = []
        with self.cv:
            for name, loop in self.loops.items():
                if loop['retired']:
                    continue
                while loop['next'] <= now and loop['next'] < stop_before-1e-9:
                    release, period = loop['next'], loop['period_ms']/1000
                    row = dict(vehicle=name, cycle=loop['cycle'], release_s=release,
                               published_s=now, period_ms=loop['period_ms'],
                               deadline_s=release+period, model_id=self.model_id, status='sampling')
                    self.rows.append(row)
                    loop.update(previous=release, next=release+period, cycle=loop['cycle']+1)
                    old = loop['busy']
                    if old is not None and old['deadline_s'] <= now:
                        old['expired'] = True
                        if 'future' in old:
                            old['future'].cancel()
                    if row['deadline_s'] <= now:
                        row['status'] = 'dropped_release_late'
                    elif old is not None:
                        row['status'] = 'dropped_busy_io'
                    else:
                        loop['busy'] = row
                        admitted.append(row)
                    self.revision += 1
            self.cv.notify_all()
        return admitted

    def observe(self, name, sample_s, deadline_s, analysis, version):
        with self.cv:
            loop = self.loops[name]
            if version != loop['control_version'] or loop['retired']:
                return False
            if loop['observation'] is not None and sample_s <= loop['observation']['sample_s']:
                return False
            loop['observation'] = dict(sample_s=sample_s, deadline_s=deadline_s,
                                       analysis=analysis, control_version=version)
            self.revision += 1
            self.cv.notify_all()
            return True

    def accept_native_release(self, event):
        """Import a published slot against the calendar last committed to C++."""
        name = event['vehicle']
        release, period, cycle = event['planned_release_s'], event['requested_period_s'], event['slot']
        status = {0: 'sampling', 1: 'dropped_release_late', 4: 'dropped_busy_io'}[event['native_release_status']]
        with self.cv:
            loop = self.loops[name]
            if (cycle != loop['cycle'] or abs(release-loop['next']) > 1e-7 or
                    abs(period*1000-loop['period_ms']) > 1e-7):
                raise RuntimeError('native release does not match the committed calendar')
            row = dict(vehicle=name, cycle=cycle, release_s=release, published_s=event['native_release_s'],
                       period_ms=period*1000, deadline_s=release+period, model_id=self.model_id, status=status)
            if 'native_calendar_version' in event:
                row['native_calendar_version'] = event['native_calendar_version']
            self.rows.append(row)
            loop.update(previous=release, next=release+period, cycle=cycle+1)
            if status == 'sampling':
                if loop['busy'] is not None:
                    raise RuntimeError('native and Python worker ownership disagree')
                loop['busy'] = row
            self.revision += 1
            self.cv.notify_all()
            return row

    def command_changed(self, name):
        with self.cv:
            loop = self.loops[name]
            loop['control_version'] += 1
            loop['observation'] = None  # old state/held-control certificate is no longer usable
            self.revision += 1
            return loop['control_version']

    def update_cpu_costs(self, *, prep_ms=0., post_ms=0., rearm_ms=0., front_wait_ms=0.):
        with self.cv:
            prep, post = max(self.prep_ms, prep_ms), max(self.post_ms, post_ms)
            rearm = max(self.rearm_ms, rearm_ms)
            front_wait = max(self.front_wait_ms, front_wait_ms)
            if (prep, post, rearm, front_wait) != (self.prep_ms, self.post_ms, self.rearm_ms, self.front_wait_ms):
                self.prep_ms, self.post_ms, self.rearm_ms, self.front_wait_ms = prep, post, rearm, front_wait
                self.revision += 1

    def enqueue(self, row, feature):
        with self.cv:
            future = row['future'] = Future()
            if self.closed or row['deadline_s'] <= time.perf_counter():
                future.cancel()
                return future
            row.update(feature=feature, queue_s=time.perf_counter(), phase='queued')
            self.queue.append(row)
            self.revision += 1
            self.cv.notify_all()
            return future

    def take(self, now):
        """Nonpreemptive: a newer urgent job cannot replace the active UDP request."""
        with self.cv:
            if self.active is not None:
                return None
            live = []
            for row in self.queue:
                if row['future'].cancelled() or row['deadline_s'] <= now:
                    row['future'].cancel()
                else:
                    live.append(row)
            self.queue = live
            if not live:
                return None
            key = (lambda r: (min(r['deadline_s'], r.get('safety_deadline_s', r['deadline_s']))
                              if self.adaptive else r['deadline_s'],
                              r['release_s'], r['vehicle'])) if self.order == 'edf' else (
                                  lambda r: (r['queue_s'], r['vehicle']))
            row = min(live, key=key)
            self.queue.remove(row)
            row.update(board_start_s=now, phase='board')
            self.active = row
            self.revision += 1
            return row

    def board_done(self, row, now, *, action=None, timing=None, error=None):
        with self.cv:
            row.update(board_done_s=now, phase='post', board_timing=timing, board_error=error)
            elapsed_ms = 1000*(now-row['board_start_s'])
            self.cost_ms = max(self.cost_ms, elapsed_ms)  # never silently shrink the cost envelope
            self.active = None
            future = row['future']
            if not future.cancelled():
                if error is None:
                    future.set_result(action)
                else:
                    future.set_exception(RuntimeError(error))
            self.revision += 1
            self.cv.notify_all()

    def finish(self, row):
        with self.cv:
            loop = self.loops[row['vehicle']]
            if loop['busy'] is row:
                loop['busy'] = None
            self.revision += 1
            self.cv.notify_all()

    def retire(self, name):
        with self.cv:
            self.loops[name]['retired'] = True
            row = self.loops[name]['busy']
            if row is not None and 'future' in row:
                row['future'].cancel()
            self.revision += 1
            self.cv.notify_all()

    def snapshot(self, now):
        with self.cv:
            if self.active is not None and 1000*(now-self.active['board_start_s']) >= self.cost_ms:
                # Once the envelope is exceeded, no finite remainder is known.
                self.skips['active_cost_envelope_exceeded'] += 1
                return None
            loops, carry = [], []
            mode = Mode(str(self.model_id), (self.cost_ms,))
            for name, live in self.loops.items():
                if live['retired']:
                    continue
                old = live['busy']
                command_completed = old is not None and 'command_return_s' in old
                if old is not None and old['deadline_s'] <= now and not command_completed:
                    self.skips['expired_io_still_running'] += 1
                    return None
                observation = live['observation']
                if observation is None or live['next'] <= now:
                    self.skips['unpaired_state' if observation is None else 'release_due'] += 1
                    return None
                current = Option(live['period_ms'], mode)
                allowed = self.periods if self.qualified_periods is None else self.qualified_periods[name]
                options = (current,) + tuple(Option(t, mode) for t in self.periods if t != current.period and t in allowed)
                loops.append(Loop(name, 1000*(live['previous']-now), 1000*(live['next']-now),
                                  1000*(observation['deadline_s']-now),
                                  2*max(self.periods), options))
                if old is not None:
                    active = self.active is old
                    remaining = max(0., self.cost_ms-1000*(now-old['board_start_s'])) if active else 0.
                    # Active service is reserved separately and cannot be reordered.
                    cost = 0. if active or old.get('phase') == 'post' else self.cost_ms
                    deadline = min(old['deadline_s'], old.get('safety_deadline_s', old['deadline_s']))
                    operations = None
                    if command_completed:
                        operations = ()
                    elif active or old.get('phase') == 'post':
                        operations = (('ps', self.post_ms),)
                    elif old.get('phase') == 'queued':
                        operations = (('pl', cost), ('ps', self.post_ms))
                    carry.append(Job(f"carry:{name}:{old['cycle']}", name, remaining,
                                     1000*(deadline-now), (cost,), str(self.model_id), operations, command_completed))
            if not loops:
                return None
            block = (max(0., self.cost_ms-1000*(now-self.active['board_start_s']))
                     if self.active is not None else 0.)
            platform = Platform(prep=self.prep_ms, dispatch=0., post=self.post_ms,
                                ingress=0., tail=0., switch=0., decision=self.decision_ms,
                                initial_pl_block=block, rearm=self.rearm_ms, front_wait=self.front_wait_ms)
            return self.revision, Scenario(tuple(loops), platform, horizon=2*max(self.periods),
                                           carry=tuple(carry), initial_policy=(self.order, 1), legal_groups=(1,))

    def install(self, revision, scenario, result, snapshot_s, now=None):
        """CAS plus clock check; never alter released rows or reset a past deadline."""
        with self.cv:
            now = time.perf_counter() if now is None else now
            if not result['accepted']:
                return 'no_feasible_repair', []
            if revision != self.revision:
                return 'context_changed', []
            if (now-snapshot_s)*1000 > scenario.platform.decision:
                return 'decision_budget_exceeded', []
            forecast = result['forecast']
            for loop in scenario.loops:
                jobs = sorted((j for j in forecast['jobs'] if j.loop == loop.name), key=lambda j: j.release)
                if any(forecast['finish'][old.name] > new.release+1e-8 for old, new in zip(jobs, jobs[1:])):
                    # Each vehicle owns one IO pipeline. Advancing its future release
                    # must not overlap the carry-in task still occupying that worker.
                    return 'per_vehicle_pipeline_overlap', []
            updates, expected, proposal = [], {}, {}
            for loop, index in zip(scenario.loops, result['choice']):
                option = loop.options[index]
                live = self.loops[loop.name]
                release = (loop.next_release if index == 0 else
                           max(scenario.platform.decision, loop.previous_release+option.period))
                absolute = snapshot_s+release/1000
                if absolute <= now:
                    return 'release_already_due', []
                if self.native_io is not None:
                    name = loop.name
                    expected[name] = dict(vehicle=name, sample_id=f"{name}:{live['cycle']}", model=self.model_id,
                                          planned_release_s=live['next'], requested_period_s=live['period_ms']/1000,
                                          deadline_s=live['next']+live['period_ms']/1000)
                    proposal[name] = dict(expected[name], planned_release_s=absolute,
                                          requested_period_s=option.period/1000, deadline_s=absolute+option.period/1000)
                if option.period != live['period_ms']:
                    updates.append(dict(vehicle=loop.name, previous_period_ms=live['period_ms'],
                                        period_ms=option.period, next_release_s=absolute, first_cycle=live['cycle']))
            changed_order = self.order != result['policy'][0]
            if updates and self.native_io is not None:
                if not self.adaptive:
                    return 'planning_stopped', []
                for name in proposal:
                    busy = self.loops[name]['busy']
                    if busy is not None and 'command_start_s' in busy and 'command_return_s' not in busy:
                        return 'command_in_flight', []
                # This DLL only samples. Its command tokens remain zero. Python's
                # revision and in-flight check above guard the real command path.
                guard = self.native_io.guarded_batch(self.native_io.calendar_basis(expected), proposal,
                    dict.fromkeys(proposal, 0), self.native_end, snapshot_s+scenario.platform.decision/1000,
                    commit=True)
                if not guard['installed']:
                    return 'native_'+guard['reason'], []
            for update in updates:
                self.loops[update['vehicle']].update(period_ms=update['period_ms'], next=update['next_release_s'])
            self.order = result['policy'][0]
            if updates or changed_order:
                self.revision += 1
                self.cv.notify_all()
                return 'installed', updates
            return 'already_feasible', []

    def plan_once(self):
        start = time.perf_counter()
        if start < self.origin:
            return None
        with self.cv:
            snapshot = self.snapshot(start)
            ages = {name: 1000*(start-live['observation']['sample_s'])
                    for name, live in self.loops.items()
                    if not live['retired'] and live['observation'] is not None}
        if snapshot is None:
            return None
        revision, scenario = snapshot
        result = abc_repair(scenario)
        status, updates = self.install(revision, scenario, result, start)
        done = time.perf_counter()
        record = dict(snapshot_s=start, done_s=done, elapsed_ms=1000*(done-start),
                      status=status, updates=updates, initial_type=result['initial_type'],
                      attempts=result['attempts'], checks=result['checks'],
                      stop_reason=result['stop_reason'], forecast_passed=result['accepted'],
                      minimum_slack_ms=min(result['forecast']['slack'].values()),
                      worst_predicted_job=result['forecast']['worst'],
                      predicted_deadline_misses=result['forecast']['misses'],
                      choice=result['choice'], policy=result['policy'],
                      loop_context=[asdict(loop) for loop in scenario.loops],
                      carry_context=[asdict(job) for job in scenario.carry],
                      carry_jobs=len(scenario.carry), platform=asdict(scenario.platform),
                      state_ages_ms=ages)
        self.decisions.append(record)
        return record

    def public_rows(self):
        return [{**{k: v for k, v in r.items() if k not in ('future', 'feature')},
                 **({'raw237': r['feature'].tolist()} if 'feature' in r else {})} for r in self.rows]
