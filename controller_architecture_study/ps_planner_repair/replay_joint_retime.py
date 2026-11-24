"""Offline pinned-release joint retime for the recorded ten-UAV incident."""

import argparse
from collections import Counter, defaultdict
from itertools import product
import json
from pathlib import Path

from ..bottleneck_joint import Job, Platform, Scenario, simulate, witness
from .audit_retime_preparation import audit
from .critical_update_screen import load, rows


EPS = 1e-8
DECISION_MS = 1.


def snapshot(log, budgets, config, now_us):
    now = now_us / 1000
    loops, transition, carry_overrun = {}, [], []
    for vehicle, post in budgets['fleet_post_ms'].items():
        past = [r for r in log[vehicle] if r['calendar_release']['published_ms'] <= now]
        last = max(past, key=lambda r: r['cycle'])
        period = last['period_ms']
        release = last['calendar_release']['release_ms'] + period - now

        returned = [r for r in past if r.get('command_return_us', float('inf')) <= now_us
                    and 'new_held_version' in r]
        ack = max(returned, key=lambda r: r['command_return_us'])
        version = ack['new_held_version']
        hold = ack['command_return_us'] / 1000 + config['maximum_update_ms']
        observations = [(r['sample_start_us'], r['hold_deadline_us'] / 1000) for r in past
                        if r.get('observe_accepted') and r.get('held_version') == version
                        and r.get('feature_ready_us', float('inf')) <= now_us]
        if ack.get('paired_observe_accepted') and ack['slot_return_us'] <= now_us:
            sample = ack['paired_sample_start_us']
            observations.append((sample, sample / 1000 + 1000 * ack['paired_hold_budget_s']))
        if observations:
            hold = min(hold, max(observations)[1])

        local_rearm = max((r['slot_return_us'] - r['command_return_us']) / 1000
                          for r in returned if r['slot_return_us'] <= now_us)
        loops[vehicle] = dict(period=period, release=release, hold=hold-now, post=post,
                              rearm=max(config['rearm_ms'], local_rearm), observed_hold=bool(observations))

        for r in past:
            if r['calendar_release']['status'] != 'sampling' or r['slot_return_us'] <= now_us:
                continue
            job = r['calendar_finished']
            board_start = job.get('board_start_ms', float('inf'))
            if board_start <= now and r['slot_return_us'] > now_us + DECISION_MS * 1000:
                # A board/post job owns its loop through finish(), also after ACK.
                transition.append(f'{vehicle}:{r["cycle"]}')
            if (r['slot_return_us'] - now_us) / 1000 > release + EPS:
                carry_overrun.append(f'{vehicle}:{r["cycle"]}')
    return loops, transition, carry_overrun


def check(loops, periods, prep, critical, dpu, fresh, horizon, rearm, serial):
    # The first job has every deadline of its successors plus the old hold bound.
    # Test that isolated path before building the shared-resource calendar.
    blockers = []
    for vehicle, loop in loops.items():
        period = periods[vehicle]
        limits = dict(slot=period, fresh=fresh, hold=loop['hold'] - loop['release'], worker_reuse=period-rearm[vehicle])
        kind = min(limits, key=limits.get)
        path = prep + dpu + loop['post']
        blockers.append((path - limits[kind], dict(vehicle=vehicle, constraint=kind, path_ms=path,
                                                  available_ms=limits[kind])))
    local_deficit, blocker = max(blockers, key=lambda item: item[0])
    if local_deficit > EPS:
        return local_deficit, 'isolated_path_or_rearm', blocker

    jobs, next_release = [], {}
    for vehicle, loop in loops.items():
        period = periods[vehicle]
        for k in range(1 + int(horizon / period)):
            release = loop['release'] + k * period
            if release >= horizon:
                break
            if serial:
                chain = (('ps', prep), ('pl', dpu), ('ps', loop['post']))
            else:
                chain = (('delay', prep-critical), ('ps', critical), ('pl', dpu), ('delay', loop['post']))
            deadline = min(release + min(period, fresh), loop['hold'] if k == 0 else float('inf'))
            name = f'{vehicle}:{k}'
            jobs.append(Job(name, vehicle, release, deadline, (), 'fixed', chain))
            next_release[name] = release + period

    platform = Platform(decision=DECISION_MS, initial_pl_block=DECISION_MS, switch=0.)
    scenario = Scenario((), platform, carry=tuple(jobs), legal_groups=(1,))
    if serial:
        bound = witness(scenario, jobs)
        if bound['deficit'] > EPS:
            kind = f'{bound["kind"]}_interval' if bound['kind'] != 'path' else 'isolated_path'
            return bound['deficit'], kind, bound
    forecast = simulate(scenario, (), ('edf', 1))
    slack, name = min((min(j.deadline, next_release[j.name] - rearm[j.loop]) - forecast['command_finish'][j.name],
                       j.name) for j in jobs)
    return -slack, 'replay', dict(job=name)


def search(loops, changed, grid, prep, critical, dpu, fresh, horizon, rearm, serial):
    current = {vehicle: loop['period'] for vehicle, loop in loops.items()}
    menus = [[period for period in grid if period > current[v]] for v in changed]
    best, rejections = None, Counter()
    for periods in product(*menus):
        assignment = dict(zip(changed, periods))
        deficit, kind, blocker = check(loops, dict(current, **assignment), prep, critical, dpu, fresh, horizon, rearm, serial)
        candidate = dict(periods_ms=assignment, deficit_ms=deficit, check=kind, blocker=blocker)
        if best is None or deficit < best['deficit_ms']:
            best = candidate
        if deficit <= EPS:
            # Ascending period grid. Use this witness only for existence.
            return dict(feasible=True, witness=candidate, rejected_candidates=rejections)
        rejections[kind] += 1
    return dict(feasible=False, closest=best, rejected_candidates=rejections)


def run(trial):
    budgets = audit(trial, include_all_posts=True)['attempts']
    identity = load(trial / 'execution_identity.json')
    config, origin = identity['live_config'], identity['origin_us']
    grid = sorted(config['periods_ms'])
    horizon = max(grid)  # Same release window for every candidate, including the baseline.
    log = defaultdict(list)
    for row in rows(trial):
        log[row['vehicle']].append(row)
    results = []
    profile_names = ('serial_shared', 'overlap_shared', 'overlap_local', 'overlap_no_rearm')
    for attempt in budgets:
        loops, transition, overrun = snapshot(log, attempt, config, origin + round(attempt['target_time_ms'] * 1000))
        affected = sorted(set(attempt['fleet_post_over_period']) | {attempt['target']})
        prep, critical, dpu = attempt['prep_ms']['global_max'], attempt['prep_ms']['shared_critical'], attempt['dpu_ms']
        isolated_cost = {v: prep + dpu + loop['post'] for v, loop in loops.items()}
        shared = {v: max(loop['rearm'] for loop in loops.values()) for v in loops}
        local = {v: loop['rearm'] for v, loop in loops.items()}
        profiles = dict(serial_shared=(True, shared), overlap_shared=(False, shared),
                        overlap_local=(False, local), overlap_no_rearm=(False, dict.fromkeys(loops, 0.)))
        commit_room = min(loop['release'] for loop in loops.values())
        comparisons = {}
        if commit_room > 0:
            for name, (serial, rearm) in profiles.items():
                comparisons[name] = {policy: search(loops, group, grid, prep, critical, dpu, config['fresh_ms'],
                                                    horizon, rearm, serial)
                                     for policy, group in (('fixed', []), ('single', [attempt['target']]), ('joint', affected))}
        results.append(dict(time_ms=attempt['target_time_ms'], target=attempt['target'], cycle=attempt['target_cycle'],
                            affected=affected, loops=loops, in_flight_board_or_post=transition, carry_overrun=overrun,
                            isolated_over_period=sorted(v for v, cost in isolated_cost.items()
                                                        if cost > loops[v]['period'] + EPS),
                            isolated_over_hold=sorted(v for v, cost in isolated_cost.items()
                                                      if cost > loops[v]['hold'] - loops[v]['release'] + EPS),
                            commit_room_ms=commit_room, comparisons=comparisons))
    return dict(scope='Trace-derived finite windows; simulations contain new work only, carry-in reported separately',
                horizon_ms=horizon, decision_reserve_ms=DECISION_MS, windows=len(results),
                blocked_by_current_transition_gate=sum(bool(r['in_flight_board_or_post']) for r in results),
                historical_carry_overruns=sum(bool(r['carry_overrun']) for r in results),
                past_release_windows=sum(r['commit_room_ms'] <= 0 for r in results),
                commit_room_below_reserve=sum(r['commit_room_ms'] <= DECISION_MS for r in results),
                feasible={name: {policy: sum(r['comparisons'][name][policy]['feasible'] for r in results if r['comparisons'])
                                 for policy in ('fixed', 'single', 'joint')} for name in profile_names}, windows_detail=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.trial)
    with args.output.open('x', encoding='utf-8') as out:
        json.dump(result, out, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in result.items() if k != 'windows_detail'}, indent=2))


if __name__ == '__main__':
    main()
