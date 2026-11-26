"""Run the native Zynq PS controller against the isolated onboard Blocks."""

import argparse
from bisect import bisect_right
import json
from math import ceil, inf
from pathlib import Path
from random import Random
import shlex
import subprocess

from config import DEFAULT_CALIBRATION_PATH
from .run_onboard_fleet_value import nominal_schedule, trial_margins
from .run_onboard_load_replay import load_windows
from .run_onboard_pilot import scenario, trial
from .run_onboard_stagger_validation import LEADING, ROUTES
from .run_onboard_value import BAND_M, PORT, SLOTS, SLOT_S, read, save


STAGGER_S = .90
WINDOW_STARTS = (1, 9, 17)
REQUEST_BUDGET = 18
STATIC_CANDIDATES = ((2, 5), (2, 11), (3, 5), (3, 11))
BOARD_PACKAGE = '/home/root/board_package_requant18_release001'


def ssh_command(args):
    remote = shlex.join([args.binary, args.windows_ip, args.package])
    command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5']
    board = [*command, args.board, 'exec ' + remote]
    return [*command, args.ssh_hop, shlex.join(board)] if args.ssh_hop else board


def fly(args, client, gates, path, amplitudes, distances, slots, probes, policy, period, model, *,
        stagger=STAGGER_S, table=None, background_us=0):
    spec = dict(policy='route_table' if table is not None else policy, slots=slots,
                request_budget=REQUEST_BUDGET, fixed_period_slots=period,
                fixed_model=model, port=PORT, calibration=read(DEFAULT_CALIBRATION_PATH), probes=probes,
                fleet=[dict(vehicle=v, amplitude=a) for v, a in amplitudes.items()])
    if table is not None:
        spec.update(route_table=table, request_budget=len(slots))
    elif args.load_study or background_us:
        spec['request_budget'] = len(slots)
    if background_us:
        spec['background_period_us'] = background_us
    log = path.with_suffix('.ps.log')
    with log.open('w', encoding='utf-8') as diagnostic:
        with subprocess.Popen(ssh_command(args), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              stderr=diagnostic, text=True, encoding='utf-8') as ps:
            try:
                ps.stdin.reconfigure(newline='\n')
                ps.stdin.write(json.dumps(spec)+'\n')
                ps.stdin.flush()
                while True:
                    line = ps.stdout.readline()
                    if not line:
                        raise RuntimeError(f'PS did not reach READY. See {log}')
                    if line.startswith('{'):
                        ready = json.loads(line)
                        if ready.get('ready'):
                            break
                    diagnostic.write(line)

                def feed(stop):
                    stdout, _ = ps.communicate(input='GO\n', timeout=12)
                    if ps.returncode:
                        raise RuntimeError(f'PS trial failed. See {log}')
                    native = json.loads(stdout.splitlines()[-1])
                    save(path/'native_ps.json', native)
                    return {}, native['requests']

                result = trial(client, PORT, path, gates, 3, False, feed, leading=LEADING, stagger_s=stagger,
                               amplitudes=amplitudes)
            finally:
                if ps.poll() is None:
                    ps.kill()
    native = read(path/'native_ps.json')
    summary = inspect(path, result, native, probes)
    if background_us:
        summary['contention'] = contention_metrics(native)
    summary.update(policy=policy, fixed_period_ms=period*180, fixed_model=model, golden_max_abs=ready['goldens'],
                   stagger_s=stagger, available_slots=len(slots), request_budget=spec['request_budget'],
                   **trial_margins(path, amplitudes=amplitudes, distances=distances))
    save(path/'joined_report.json', summary)
    print(f"{path.name} requests={summary['requests']} adopted={summary['adopted']} "
          f"key={summary['key_adopted']} margin={summary['fleet_min_m']} pass={summary['pass']}", flush=True)
    return summary


def inspect(path, result, native, probes):
    traces = {v: read(path/f'{v}.json') for v in result}
    requests = {(r['vehicle'], r['slot'], r['sample_ns']): r for r in native['requests']}
    applied = []
    for v, trace in traces.items():
        for actual, x, _, _, source, seq, sample in trace:
            if source != 3:
                continue
            row = requests.get((v, seq, sample))
            if not row or not row['accepted'] or not 0 <= actual-sample <= 55_000_000:
                raise RuntimeError('Found an unpaired or expired adopted command')
            timely = any(abs(x-probes[p]) <= .25 for p in ('g2_early', 'g3_early', 'g3_near'))
            applied.append(dict(vehicle=v, slot=int(seq), age_ms=(actual-sample)/1e6, x_m=x, timely=timely))
    in_between = {}
    for v, trace in traces.items():
        used = [r for r in trace if r[4] == 3]
        if len(used) >= 2:
            in_between[v] = sum(r[4] == 1 and used[0][0] < r[0] < used[1][0] for r in trace)
    return dict(vehicles=result, requests=len(requests), accepted=sum(r['accepted'] for r in requests.values()),
                adopted=len(applied), key_adopted=sum(r['timely'] for r in applied),
                dpu_run_ms=sum(r.get('dpu_run_us',0) for r in requests.values())/1000,
                ps_active_ms=sum(r['elapsed_ms'] for r in native['slots']),
                applied=applied, local_between_first_two=in_between,
                **{'pass': all(v['pass_gates'] and not v['contact'] for v in result.values())})


def contention_metrics(native):
    background = native['background']
    start = background['origin_us']
    stop = start+background['horizon_us']
    jobs = background['jobs']
    if [row[0] for row in jobs] != list(range(0, background['horizon_us'], background['period_us'])):
        raise RuntimeError('Background release sequence differs from its offered workload')
    intervals = [(row[3], row[4]) for row in jobs if row[5] == 0]
    if len(intervals) != len(jobs) or any(end <= begin for begin, end in intervals):
        raise RuntimeError('Background DNNDK failed or produced invalid timing')
    foreground = [r for r in native['requests'] if r.get('dpu_run_us', 0) > 0]
    uses = sorted(intervals+[(r['dpu_run_start_us'], r['dpu_run_end_us']) for r in foreground])
    if any(a[1] > b[0] for a, b in zip(uses, uses[1:])):
        raise RuntimeError('DNNDK intervals overlap on the serialized device')
    starts = [a for a, _ in intervals]
    blocked = []
    for r in foreground:
        at = r['device_requested_us']
        before = bisect_right(starts, at)-1
        if before >= 0 and at < intervals[before][1] and r['device_acquired_us'] >= intervals[before][1]:
            blocked.append(dict(slot=r['slot'], vehicle=r['vehicle'], wait_ms=r['device_wait_ms']))
    busy = sum(max(0, min(b, stop)-max(a, start)) for a, b in intervals)
    critical_candidates = {s['slot']: [c['vehicle'] for c in s.get('candidates', []) if c['critical'] and c['eligible']]
                           for s in native['slots']}
    release_lateness = sorted(row[1]-row[0] for row in jobs)
    return dict(background_requests=len(jobs), background_call_fraction=busy/(stop-start),
                background_release_lateness_p99_ms=release_lateness[ceil(.99*len(jobs))-1]/1000,
                background_blocked=blocked, max_device_wait_ms=max((r['device_wait_ms'] for r in foreground), default=0.),
                dual_critical={slot: vehicles for slot, vehicles in critical_candidates.items() if len(vehicles) >= 2})


def route_table(plan, slots, stagger=STAGGER_S, *, prefer_smaller_margin=False):
    forecast = plan['schedules'][str(stagger)]
    centers = [(p, plan['probe_x_m'][p]) for p in ('g2_early', 'g3_early', 'g3_near')]
    last = dict.fromkeys(forecast[0]['nominal_x_m'], 0)
    table = {}
    for seq in slots:
        i = seq-1
        positions = forecast[i]['nominal_x_m']
        previous = forecast[max(0, i-1)]['nominal_x_m']
        following = forecast[min(len(forecast)-1, i+1)]['nominal_x_m']
        choices = []
        for vehicle, x in positions.items():
            phase, end = next(((p, center+BAND_M) for p, center in centers if abs(x-center) <= BAND_M), (None, inf))
            critical = phase is not None
            period = 1 if critical else 4
            if x < 9.95 and (not last[vehicle] or seq-last[vehicle] >= period):
                span = min(len(forecast)-1, i+1)-max(0, i-1)
                velocity = (following[vehicle]-previous[vehicle])/(span*SLOT_S)
                margin = (plan['baseline_m'][vehicle][f'gate{phase[1]}']
                          if prefer_smaller_margin and critical else inf)
                key = (margin, (end-x)/max(.2, velocity), last[vehicle], vehicle)
                choices.append((key, period, 11 if critical else 5))
        if choices:
            (_, _, _, vehicle), period, model = min(choices)
            table[str(seq)] = dict(vehicle=vehicle, period_slots=period, model=model)
            last[vehicle] = seq
    return table


def contention_pair(args, plan, amplitudes, distances):
    sources = sorted((args.replay_from/'shadow'/'independent').glob('cal_*_shadow'))
    if len(sources) != 2:
        raise ValueError('Need the two frozen independent-route calibration flights')
    slots = list(range(1, SLOTS+1))
    cases = {'overlap': 0.} if args.contention_margin_check else {'staggered': STAGGER_S, 'overlap': 0.}
    period_us = 3000
    plan['schedules']['0.0'] = nominal_schedule(sources, LEADING, plan['gains_m'], plan['baseline_m'],
                                                stagger=0., distances=distances, phases=plan['probe_x_m'])
    tables = {name: route_table(plan, slots, stagger, prefer_smaller_margin=args.contention_margin_check)
              for name, stagger in cases.items()}
    with scenario(PORT, args.output, amplitudes=amplitudes, distances=distances) as (client, gates):
        save(args.output/'contention_plan.json', dict(cases=cases, slots=slots, tables=tables,
                                                    table_priority=('calibrated_gate_margin' if args.contention_margin_check
                                                                    else 'time_to_band_exit'),
                                                    background_period_us=period_us, repetitions=3))

        def run_one(group, name, policy):
            return fly(args, client, gates, args.output/name, amplitudes, distances, slots, plan['probe_x_m'],
                       policy, 2, 11, stagger=cases[group],
                       table=tables[group] if policy == 'route_table' else None, background_us=period_us)

        validation = {name: run_one(name, f'verify_{name}', 'adaptive') for name in cases}
        save(args.output/'setup_validation.json', validation)
        verification = {case: row['contention'] for case, row in validation.items()}
        if (any(not row['pass'] or not row['adopted'] for row in validation.values()) or
                any(not row['background_blocked'] for row in verification.values()) or
                verification.get('staggered', {}).get('dual_critical') or not verification['overlap']['dual_critical']):
            raise RuntimeError('Pilot did not establish shared-device blocking and the intended phase contrast')
        evaluation = {}
        for rep in range(1, 4):
            order = ('route_table', 'adaptive') if rep % 2 else ('adaptive', 'route_table')
            for group in cases if rep % 2 else reversed(cases):
                for policy in order:
                    name = f'{group}_r{rep}_{policy}'
                    evaluation[name] = run_one(group, name, policy)
                    save(args.output/'summary.json', evaluation)


def load_study(args, client, gates, plan, slots, amplitudes, distances):
    choice = read(args.load_study)
    period, model = choice['period_slots'], choice['model']
    # Restore a seeded subset of the historical busy slots without looking at
    # flight phases or results. All three levels retain the original open slots.
    blocked = sorted(set(range(1, SLOTS+1))-set(slots))
    Random(20261126).shuffle(blocked)
    levels = {'low': sorted([*slots, *blocked[:16]]),
              'medium': sorted([*slots, *blocked[:8]]), 'high': slots}
    arms = ('route_table', 'adaptive', 'lock_T', 'lock_c', 'lock_order')
    ablation = [(f'nominal_r{rep}_{policy}', slots, STAGGER_S, policy)
                for rep, order in enumerate((arms, arms[::-1]), 1) for policy in order]
    shifted_arms = ('route_table', 'adaptive', 'route_table_aligned')
    shifted = [(f'shifted_{level}_{policy}', levels[level], 1.20, policy)
               for index, level in enumerate(levels) for policy in shifted_arms[index:]+shifted_arms[:index]]
    tables = {level: {'route_table': route_table(plan, opens),
                      'route_table_aligned': route_table(plan, opens, 1.20)}
              for level, opens in levels.items()}
    save(args.output/'study_plan.json', dict(levels=levels, masks_source='w17', seed=20261126, tables=tables,
                                           static_period_slots=period, static_model=model, runs=ablation+shifted))
    evaluation = {}
    for name, opens, stagger, policy in ablation+shifted:
        level = name.split('_')[1] if name.startswith('shifted_') else 'high'
        evaluation[name] = fly(args, client, gates, args.output/name, amplitudes, distances, opens,
                               plan['probe_x_m'], policy, period, model, stagger=stagger,
                               table=tables[level].get(policy))
        save(args.output/'summary.json', evaluation)


def run(args):
    plan = read(args.replay_from/'departure_plans.json')['independent']
    amplitudes, distances = ROUTES['independent']
    if args.contention_pair or args.contention_margin_check:
        contention_pair(args, plan, amplitudes, distances)
        return
    recorded = read(args.replay_from/'load_masks.json')
    if args.load_masks:
        masks = read(args.load_masks)
        if any(masks['windows'][name] != rows for name, rows in recorded['windows'].items()):
            raise ValueError('Frozen load masks differ in shared windows')
    else:
        source = args.trace_source or Path(recorded['source'])
        masks = load_windows(source, window_starts=WINDOW_STARTS)
    if masks['source_sha256'] != recorded['source_sha256']:
        raise ValueError('DPU trace does not match the frozen replay')
    available = {name: [row['slot'] for row in rows if not row['covering_call_ns']]
                 for name, rows in masks['windows'].items()}
    with scenario(PORT, args.output, amplitudes=amplitudes, distances=distances) as (client, gates):
        save(args.output/'load_masks.json', masks)
        probes = plan['probe_x_m']
        probe = fly(args, client, gates, args.output/'probe', amplitudes, distances, [8,16], probes, 'probe', 1, 11)
        if not (probe['pass'] and probe['adopted'] == 2 and probe['local_between_first_two'].get('Drone1', 0) > 0):
            raise RuntimeError('Fixed PS bridge must adopt both commands and locally control during the gap')
        if args.probe_only:
            return
        if args.load_study:
            load_study(args, client, gates, plan, available['w17'], amplitudes, distances)
            return
        training = [fly(args, client, gates, args.output/f'tune_T{period}_c{model}', amplitudes, distances,
                        available['w01'], probes, 'fixed', period, model) for period, model in STATIC_CANDIDATES]
        viable = [r for r in training if r['pass'] and r['fleet_min_m'] is not None]
        if not viable:
            raise RuntimeError('No viable static candidate')
        best = max(viable, key=lambda r: (r['fleet_min_m'], -r['dpu_run_ms']))
        period, model = best['fixed_period_ms']//180, best['fixed_model']
        save(args.output/'static_choice.json', dict(period_slots=period, model=model,
                                                   tuning_window='w01', training=training))
        evaluation = {}
        for window in ('w09','w17'):
            for rep, order in enumerate((('fixed','adaptive'), ('adaptive','fixed')), 1):
                for policy in order:
                    name = f'{window}_r{rep}_{policy}'
                    evaluation[name] = fly(args, client, gates, args.output/name, amplitudes, distances,
                                           available[window], probes, policy, period, model)
                    save(args.output/'summary.json', evaluation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replay-from', type=Path, required=True)
    loads = parser.add_mutually_exclusive_group()
    loads.add_argument('--trace-source', type=Path, help='Local path to the frozen historical DPU profile')
    loads.add_argument('--load-masks', type=Path, help='Frozen three-window output from this comparison')
    parser.add_argument('--binary', required=True, help='Fresh executable on the board')
    parser.add_argument('--package', default=BOARD_PACKAGE)
    parser.add_argument('--windows-ip', default='192.168.77.1')
    parser.add_argument('--board', default='axu3egb')
    parser.add_argument('--ssh-hop', default='vitis-vm')
    parser.add_argument('--probe-only', action='store_true')
    parser.add_argument('--load-study', type=Path, metavar='STATIC_CHOICE',
                        help='Run route table, ablations and an opportunity sweep using the previously tuned static choice')
    contention = parser.add_mutually_exclusive_group()
    contention.add_argument('--contention-pair', action='store_true',
                        help='Compare staggered and coincident critical phases under periodic real DNNDK background work')
    contention.add_argument('--contention-margin-check', action='store_true',
                        help='Repeat coincident phases with a static table prioritizing the lower calibrated gate margin')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
