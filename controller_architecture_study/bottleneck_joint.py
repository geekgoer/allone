"""Finite-window PS/PL experiment, independent of the live route-A runtime.

All times are milliseconds. Profiles are deterministic supplied service demands,
NOT measured FPGA WCETs. Passing is conditional finite-horizon timing feasibility.
"""
from dataclasses import dataclass, replace
import heapq
import itertools
import math


@dataclass(frozen=True)
class Mode:
    name: str
    segments: tuple


@dataclass(frozen=True)
class Option:
    period: float
    mode: Mode
    qualified: bool = True  # caller owns controller/state/delay qualification


@dataclass(frozen=True)
class Loop:
    name: str
    previous_release: float
    next_release: float
    deadline: float  # absolute first actuation deadline relative to snapshot t=0
    hold_bound: float  # supplied lower bound valid over forecast; never inferred from one B
    options: tuple  # option 0 is the current configuration
    release_locked: bool = False


@dataclass(frozen=True)
class Platform:
    prep: float = .25
    dispatch: float = .04
    post: float = .15
    ingress: float = .15
    tail: float = .25
    switch: float = .03
    decision: float = .5  # reserved PS service / commit delay for every method
    initial_pl_block: float = 0.  # immutable currently executing segment remainder
    rearm: float = 0.  # host work after command return, before the vehicle worker is reusable
    front_wait: float = 0.  # independently overlapping observation/host wait before PS service


@dataclass(frozen=True)
class Job:
    name: str
    loop: str
    release: float
    deadline: float
    segments: tuple
    mode: str
    remaining_operations: tuple | None = None
    command_completed: bool = False


@dataclass(frozen=True)
class Scenario:
    loops: tuple
    platform: Platform
    horizon: float = 120.
    carry: tuple = ()  # already admitted jobs, never erased or reconfigured
    initial_policy: tuple = ('fifo', 2)
    legal_groups: tuple = (1,2)  # (4,) models whole-job execution for these <=4-segment modes


def group_segments(segments, group):
    return tuple(sum(segments[i:i+group]) for i in range(0, len(segments), group))


def make_jobs(scenario, choice, group):
    jobs = list(scenario.carry)
    for loop, index in zip(scenario.loops, choice):
        option = loop.options[index]
        if not option.qualified: raise ValueError('unqualified controller option')
        if option.period <= 0 or loop.hold_bound <= 0: raise ValueError('invalid contract')
        release = (loop.next_release if loop.release_locked or option.period == loop.options[0].period else
                   max(scenario.platform.decision, loop.previous_release + option.period))
        # Committed releases can happen while planning; that work waits for PS.
        if release < 0: raise ValueError('past release belongs in immutable carry-in')
        segments=group_segments(option.mode.segments, group)
        k = 0
        # The current obligation remains even if its chosen release falls beyond
        # the forecast window; postponing work must not hide an existing deadline.
        while k == 0 or release < scenario.horizon - 1e-9:
            relative = min(option.period, loop.hold_bound-option.period)
            deadline = min(release+option.period, loop.deadline) if k == 0 else release+relative
            jobs.append(Job(f'{loop.name}:{k}', loop.name, release, deadline,
                segments, option.mode.name))
            k += 1; release += option.period
    return tuple(jobs)


def chain(job, platform):
    if job.remaining_operations is not None:
        return list(job.remaining_operations)
    operations = [('ps', platform.prep), ('delay', platform.ingress)]
    if platform.front_wait:
        operations.insert(0, ('delay', platform.front_wait))
    for segment in job.segments:
        operations.extend((('ps', platform.dispatch), ('pl', segment)))
    operations.extend((('ps', platform.post), ('delay', platform.tail)))
    return operations


def isolated_finish(job, platform):
    operations = chain(job, platform)
    release = job.release
    if operations and operations[0][0] == 'delay':
        release += operations[0][1]
        operations = operations[1:]
    return max(release, platform.decision)+sum(cost for _, cost in operations)


def simulate(scenario, choice, policy):
    """Two concurrent resources; PS/PL atomic operations, no intra-segment preemption.

    PS dispatch handshakes are charged per PL segment. Every job is drained even
    after a miss; drop is not a way to obtain an accepted timing certificate.
    External ingress/tail bounds assume contention is included by the provider.
    """
    order, group = policy
    if group not in scenario.legal_groups:raise ValueError('unsupported accelerator preemption boundary')
    jobs = make_jobs(scenario, choice, group); p = scenario.platform
    pending_pl = {}
    if order == 'guarded_edf':
        for job in jobs:
            stages = chain(job, p)
            first = next((i for i, (resource, _) in enumerate(stages) if resource == 'pl'), None)
            if first is None:
                continue
            if any(resource == 'pl' for resource, _ in stages[first+1:]):
                raise ValueError('guarded EDF supports whole accelerator jobs only')
            prefix = stages[:first]
            head = prefix.pop(0)[1] if prefix and prefix[0][0] == 'delay' else 0.
            ready = max(job.release+head, p.decision)+sum(cost for _, cost in prefix)
            pending_pl[job.name] = (ready, sum(cost for _, cost in stages[first:]))
    queues = {'ps': [], 'pl': []}
    busy = {'ps': True, 'pl': True}; last_pl = None
    events = []; serial = itertools.count()
    command_paths = {j.name: chain(j, p) for j in jobs}
    operations = {name: path + ([('ps', p.rearm)] if p.rearm else [])
                  for name, path in command_paths.items()}
    progress = {j.name:0 for j in jobs}; by_name = {j.name:j for j in jobs}
    if len(by_name) != len(jobs): raise ValueError('duplicate job identity')
    waits = {j.name:{'ps':0., 'pl':0.} for j in jobs}
    finish = {}; command_finish = {}; timeline = []; switches = 0
    def event(t, resource, name): heapq.heappush(events,(t,next(serial),resource,name))
    def advance(name, now):
        k = progress[name]
        if k == len(command_paths[name]) and not by_name[name].command_completed:
            command_finish[name] = now
        if k == len(operations[name]): finish[name] = now; return
        resource, duration = operations[name][k]
        if duration < 0: raise ValueError('negative execution demand')
        if resource == 'delay':
            progress[name] += 1; event(now+duration, 'delay', name)
        else:
            j = by_name[name]
            priority = j.release if order == 'fifo' else j.deadline
            heapq.heappush(queues[resource], (priority, j.release, name, now, duration))
    for j in jobs: event(max(0.,j.release), 'release', j.name)
    event(p.decision, 'ps', None); event(p.initial_pl_block, 'pl', None)
    while events:
        now = events[0][0]
        while events and events[0][0] <= now+1e-10:
            _, _, resource, name = heapq.heappop(events)
            if resource in busy: busy[resource] = False
            if name is not None: advance(name,now)
        for resource in ('ps','pl'):
            if busy[resource] or not queues[resource]: continue
            if resource == 'pl' and order == 'guarded_edf':
                _, _, next_job, _, cost = queues['pl'][0]
                end = now+cost+(p.switch if last_pl is not None and last_pl != next_job else 0.)
                # A published future release and its path give a predicted PL-ready time.
                # A PS completion at this same instant can still put the urgent job in the queue.
                if any(by_name[name].deadline < by_name[next_job].deadline
                       and max(now, ready)+remaining+p.switch <= by_name[name].deadline+1e-8
                       and end+remaining+p.switch > by_name[name].deadline+1e-8
                       for name, (ready, remaining) in pending_pl.items()):
                    continue
                pending_pl.pop(next_job, None)
            _, _, name, ready, duration = heapq.heappop(queues[resource])
            waits[name][resource] += now-ready
            if resource == 'pl':
                if last_pl is not None and last_pl != name:
                    duration += p.switch; switches += 1
                last_pl = name
            busy[resource] = True; progress[name] += 1
            timeline.append(dict(resource=resource, job=name, ready=ready, start=now, end=now+duration))
            event(now+duration,resource,name)
    slack = {j.name:j.deadline-command_finish[j.name] for j in jobs if not j.command_completed}
    misses = [name for name, value in slack.items() if value < -1e-8]
    worst = min(slack,key=slack.get) if slack else None
    return dict(feasible=not misses, misses=misses, worst=worst, finish=finish, command_finish=command_finish, slack=slack,
                waits=waits, timeline=timeline, switches=switches, jobs=jobs,
                scope='finite deterministic service envelope; not plant or infinite-horizon certificate')


def witness(scenario, jobs):
    """Resource demand and isolated-path necessary conditions (relaxation).

    Each counted job must execute its WHOLE remaining path inside [a,b]. Carry-in
    jobs use release=0 and remaining demand. Cross-boundary jobs are omitted,
    never arbitrarily charged wholly to the interval. Passing is NOT sufficient.
    """
    p = scenario.platform
    jobs = [j for j in jobs if not j.command_completed]
    demands = {j.name:{resource:sum(cost for kind, cost in chain(j, p) if kind == resource)
                        for resource in ('ps', 'pl')} for j in jobs}
    worst = dict(deficit=0., kind='none', start=0., end=0., jobs=[])
    for j in jobs:
        earliest = isolated_finish(j, p)
        if earliest-j.deadline > worst['deficit']:
            worst = dict(deficit=earliest-j.deadline, kind='path', start=j.release,
                         end=j.deadline, jobs=[j.name])
    by_deadline=sorted(jobs,key=lambda j:j.deadline)
    for a in sorted({max(0.,j.release) for j in jobs} | {0.}):
        enclosed=[];totals={'ps':0.,'pl':0.}
        for j in by_deadline:
            if j.release<a-1e-9:continue
            enclosed.append(j)
            for resource in totals:totals[resource]+=demands[j.name][resource]
            b=j.deadline
            if b<=a:continue
            for resource, initial in [('ps',p.decision),('pl',p.initial_pl_block)]:
                blocked = max(0.,min(b,initial)-a)
                deficit = totals[resource]-(b-a-blocked)
                if deficit > worst['deficit']+1e-9:
                    worst = dict(deficit=deficit, kind=resource, start=a, end=b,
                                 jobs=[j.name for j in enclosed])
    return worst


def window_demand(jobs, p, interval):
    a,b,kind = interval['start'],interval['end'],interval['kind']
    return sum(sum(cost for resource, cost in chain(j, p) if resource == kind)
               for j in jobs if not j.command_completed and j.release>=a-1e-9 and j.deadline<=b+1e-9)


def blocking_contributions(forecast, target):
    """Executed service intersecting target wait intervals, by job/resource.

    Used only for causal candidate ranking when no positive demand witness is
    available. Changed arrivals can reorder the schedule, so this is no bound.
    """
    result={}
    for wait in forecast['timeline']:
        if wait['job']!=target or wait['ready']>=wait['start']:continue
        for operation in forecast['timeline']:
            if operation['resource']!=wait['resource'] or operation['job']==target:continue
            amount=max(0.,min(wait['start'],operation['end'])-max(wait['ready'],operation['start']))
            if amount:
                key=(operation['job'],operation['resource'])
                result[key]=result.get(key,0.)+amount
    return result


def edit_cost(scenario, choice, policy):
    """Transparent disruption objective; explicitly not FALCON J or MSE."""
    modes=0; cadence=0.
    for loop, k in zip(scenario.loops,choice):
        before,after=loop.options[0],loop.options[k]
        modes += before.mode.name != after.mode.name
        cadence += abs(math.log(after.period/before.period))
    return modes, cadence, policy!=scenario.initial_policy


def neighbors(scenario, choice, policy, allowed):
    for i,loop in enumerate(scenario.loops):
        before=loop.options[choice[i]]
        for k,after in enumerate(loop.options):
            if k==choice[i] or not after.qualified:continue
            changed=set()
            if after.period!=before.period:changed.add('rate')
            if after.mode!=before.mode:changed.add('mode')
            if not changed or not changed<=allowed:continue
            c=list(choice);c[i]=k
            yield tuple(c),policy
    if 'schedule' in allowed:
        for order in ('fifo','edf'):
            for group in scenario.legal_groups:
                if (order,group)!=policy:yield choice,(order,group)


def heuristic(scenario, *, max_checks=5, max_edits=8,
              allowed=frozenset(('rate','mode','schedule')), guided=True):
    """Greedy deficit repair; no learned score or Cartesian online search.

    For a positive demand witness, maximize relief in that very window. Otherwise
    use EDF/fine segments or isolated completion relief for the failed job. A
    new combination is committed only after the full PS/PL forecast succeeds.
    One trajectory is explored, so complementary moves may be missed.
    """
    choice=tuple(0 for _ in scenario.loops);policy=scenario.initial_policy
    forecast=simulate(scenario,choice,policy);checks=1;edits=[];seen={(choice,policy)}
    # Window-demand contributions are separable by loop. Build each legal local
    # option once, rather than reconstructing every loop for every local move.
    table={}
    if not forecast['feasible']:
        for i,loop in enumerate(scenario.loops):
            local=replace(scenario,loops=(loop,),carry=())
            for k,option in enumerate(loop.options):
                if option.qualified:
                    for group in scenario.legal_groups:table[i,k,group]=make_jobs(local,(k,),group)
    def build(c,group):
        return scenario.carry+tuple(j for i,k in enumerate(c) for j in table[i,k,group])
    w=witness(scenario,forecast['jobs']) if not forecast['feasible'] else None
    while not forecast['feasible'] and checks<max_checks and len(edits)<max_edits:
        jobs=build(choice,policy[1])
        worst=forecast['worst'];target=next(j for j in forecast['jobs'] if j.name==worst)
        interference=blocking_contributions(forecast,worst) if w['kind']=='none' else {}
        old_by_name={j.name:j for j in forecast['jobs']}
        def rank(candidate):
            c,s=candidate;new=build(c,s[1]);p=scenario.platform
            if guided and w['kind'] in ('ps','pl'):
                gain=window_demand(jobs,p,w)-window_demand(new,p,w)
            elif guided and w['kind']=='path':
                name=w['jobs'][0]
                old=next(j for j in jobs if j.name==name)
                changed=next((j for j in new if j.name==name),old)
                duration=lambda j: max(p.decision,j.release)+p.prep+p.post+p.ingress+p.tail+len(j.segments)*p.dispatch+sum(j.segments)-j.deadline
                gain=duration(old)-duration(changed)
            else:
                # No capacity certificate: inspect the observed failed path.
                changed=next((j for j in new if j.name==target.name),target)
                duration=lambda j:j.release+sum(j.segments)+len(j.segments)*p.dispatch-j.deadline
                gain=duration(target)-duration(changed)
                if guided:
                    by_name={j.name:j for j in new}
                    for (name,resource),amount in interference.items():
                        before=old_by_name[name];after=by_name.get(name)
                        demand=lambda j:sum(j.segments) if resource=='pl' else p.prep+p.post+len(j.segments)*p.dispatch
                        old_cost=demand(before)
                        if old_cost>0:
                            gain+=amount*(1-(demand(after) if after else 0.)/old_cost)
                if guided and policy[0]=='fifo' and s[0]=='edf':
                    gain+=sum(forecast['waits'][worst].values())
                if guided and policy[1]>s[1]:
                    gain+=min(forecast['waits'][worst]['pl'],max(target.segments))
                if not guided:
                    gain=sum(sum(j.segments) for j in jobs)-sum(sum(j.segments) for j in new)
            return (-gain,edit_cost(scenario,c,s),c,s)
        options=[v for v in neighbors(scenario,choice,policy,allowed) if v not in seen]
        if not options:break
        chosen=min(options,key=rank);score=rank(chosen)
        if -score[0]<=1e-9:break
        choice,policy=chosen;seen.add(chosen)
        edits.append(dict(witness=w,estimated_relief=-score[0],choice=choice,policy=policy))
        # Cheap necessary checks avoid simulating obviously deficient plans.
        after=witness(scenario,build(choice,policy[1]));w=after
        if after['deficit']>1e-9:continue
        forecast=simulate(scenario,choice,policy);checks+=1
    # Do not expose an unchecked intermediate edit as an accepted plan.
    if edits and not forecast['feasible'] and checks<max_checks:
        forecast=simulate(scenario,choice,policy);checks+=1
    return dict(accepted=forecast['feasible'],choice=choice,policy=policy,checks=checks,
                edits=edits,forecast=forecast)


def exhaustive(scenario):
    """Complete declared finite option space, not continuous/global optimum."""
    best=None;checked=0;pruned=0
    for choice in itertools.product(*(tuple(k for k,o in enumerate(l.options) if o.qualified) for l in scenario.loops)):
        for policy in itertools.product(('fifo','edf'),scenario.legal_groups):
            checked+=1
            w=witness(scenario,make_jobs(scenario,choice,policy[1]))
            if w['deficit']>1e-9:pruned+=1;continue
            report=simulate(scenario,choice,policy)
            if report['feasible'] and (best is None or edit_cost(scenario,choice,policy)<best['cost']):
                best=dict(choice=choice,policy=policy,cost=edit_cost(scenario,choice,policy))
    return dict(accepted=best is not None,best=best,configurations=checked,pruned=pruned)
