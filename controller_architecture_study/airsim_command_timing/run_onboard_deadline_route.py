"""Test one versus two optional updates under real DPU work and a common deadline."""

import argparse
import json
import math
from pathlib import Path
import subprocess
from time import perf_counter

from airsim_interface import require_airsim
from config import DEFAULT_CALIBRATION_PATH
from ..framework_runtime_wait import DeadlineWaiter
from .probe_blocks import LocalRpc
from .run_onboard_dpu import sample
from .run_onboard_fleet_value import trial_margins
from .run_onboard_pilot import scenario, trial
from .run_onboard_single_pulse import kick, matched
from .run_onboard_value import PORT, read, save
from .run_ps_onboard_pair import BOARD_PACKAGE, ssh_command


AMPLITUDES = {'Drone1': .535, 'Drone2': -.57}
GATES = (3.15, 6.40, 9.45)
THRESHOLD = GATES[-1]-1.75
STEP_MPS = .18
ROOMY = 6
DEADLINE_MS = 30.


def trigger(target, stop):
    world = require_airsim().MultirotorClient(ip='127.0.0.1', port=PORT, timeout_value=3)
    event = None
    try:
        with LocalRpc(PORT) as rpc, DeadlineWaiter() as timer:
            while not stop.is_set():
                timer.wait([], perf_counter()+.004)
                positions = {v: sample(rpc, v)[1].position[0] for v in AMPLITUDES}
                if target and event is None and positions[target] >= THRESHOLD-.26:
                    center = AMPLITUDES[target]*math.sin(.6*GATES[-1])
                    event = kick(world, target, center, STEP_MPS)
                    continue
                if min(positions.values()) >= THRESHOLD:
                    return event
    finally:
        world.client.close()
    raise RuntimeError('Control window did not trigger')


def run_case(args, client, gates, name, *, policy, calls, reserve, target=None, fixed='Drone2',
             overlap=False, recovery_commands=1, recovery_schedule=None):
    path = args.output/name
    spec = dict(policy='deadline_pair', fixed_period_slots=1, fixed_model=11, request_budget=2,
                pair_policy=policy, pair_static=fixed, pair_background_calls=calls, pair_reserve_ms=reserve,
                pair_overlap_sample=overlap, pair_recovery_commands=recovery_commands,
                gate_x_m=GATES[-1], port=PORT, calibration=read(DEFAULT_CALIBRATION_PATH),
                probes=dict(g2_early=0., g3_early=THRESHOLD, g3_near=GATES[-1]-.8),
                fleet=[dict(vehicle=v, amplitude=a) for v, a in AMPLITUDES.items()])
    if recovery_schedule is not None:
        spec['recovery_schedule'] = dict(recovery_schedule, background_calls=calls, gate_x_m=GATES[-1],
                                         reserve_ms=reserve, policy=policy)
    with path.with_suffix('.ps.log').open('w', encoding='utf-8') as diagnostic:
        with subprocess.Popen(ssh_command(args), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=diagnostic,
                              text=True, encoding='utf-8') as ps:
            try:
                ps.stdin.reconfigure(newline='\n')
                ps.stdin.write(json.dumps(spec)+'\n')
                ps.stdin.flush()
                for line in ps.stdout:
                    if line.startswith('{') and json.loads(line).get('ready'):
                        break
                    diagnostic.write(line)
                else:
                    raise RuntimeError(f'PS failed to warm in {name}')

                def feed(stop):
                    event = trigger(target, stop)
                    stdout, _ = ps.communicate(input='GO\n', timeout=5)
                    if ps.returncode:
                        raise RuntimeError(f'PS window failed in {name}')
                    native = json.loads(stdout.splitlines()[-1])
                    native['velocity_step'] = event
                    save(path/'native_ps.json', native)
                    return {}, native['requests']

                flight = trial(client, PORT, path, gates, 3, False, feed, amplitudes=AMPLITUDES)
            finally:
                if ps.poll() is None:
                    ps.kill()
    native = read(path/'native_ps.json')
    requests = {r['slot']: r for r in native['requests']}
    adopted = {}
    for v in AMPLITUDES:
        for absolute, _, _, _, source, slot, sample_ns in read(path/f'{v}.json'):
            if source == 3:
                row = requests[slot]
                if not row['accepted'] or row['vehicle'] != v or row['sample_ns'] != sample_ns or slot in adopted:
                    raise RuntimeError('Unmatched central command')
                adopted[int(slot)] = dict(vehicle=v, since_start_ms=(absolute-native['start_ns'])/1e6,
                                         age_ms=(absolute-sample_ns)/1e6)
                if recovery_schedule is not None:
                    window_ms = (absolute-row['window_start_ns'])/1e6
                    adopted[int(slot)].update(window=row['window'], window_ms=window_ms,
                                             on_time=0 <= window_ms <= DEADLINE_MS)
    intervals = sorted(native['background']+[r['dpu_interval_us'] for r in requests.values() if 'dpu_interval_us' in r])
    expected_calls = calls if recovery_schedule is None else calls*recovery_schedule['windows']
    if len(native['background']) != expected_calls or any(a[1] > b[0] for a, b in zip(intervals, intervals[1:])):
        raise RuntimeError('Real DPU execution trace differs from the requested workload')
    summary = dict(native, policy=policy, target=target, background_calls=calls, adopted=adopted,
                   timely=sum(a['since_start_ms'] <= DEADLINE_MS for a in adopted.values()),
                   contacts={v: r['contact'] for v, r in flight.items()},
                   pass_gates=all(r['pass_gates'] for r in flight.values()),
                   **trial_margins(path, amplitudes=AMPLITUDES, distances=GATES))
    if recovery_schedule is not None:
        summary['timely'] = sum(a['on_time'] for a in adopted.values())
        summary['background_calls'] = expected_calls
    save(path/'deadline_report.json', summary)
    print(name, 'background', expected_calls, 'ready_ms', round(native['background_done_ms'], 2)
          if recovery_schedule is None else [round(w['ready_ms'], 2) for w in native['windows']],
          'adopted', adopted, 'gate3_mm', {v: round(m['gate3']*1000, 2) if m['gate3'] is not None else None
                                         for v, m in summary['by_gate_m'].items()}, flush=True)
    return summary


def gate3(report):
    return {v: r['gate3'] for v, r in report['by_gate_m'].items()}


def evaluate(runs, baseline):
    audit = {}
    for name, (fixed, online) in runs.items():
        target = fixed['target']
        base = baseline[target]
        matches = {r['policy']: {v: matched(base['states'][v], r['states'][v]) for v in AMPLITUDES}
                   for r in (fixed, online)}
        base_margins = {v: m for v, m in gate3(base).items() if m is not None}
        valid = (all(m['matched'] for arm in matches.values() for m in arm.values())
                 and fixed['timely'] == online['timely']
                 and len(fixed['adopted']) == len(online['adopted']) == fixed['timely']
                 and fixed['timely'] > 0)
        audit[name] = dict(valid=valid, target=target, matches=matches,
            base_limiting_vehicle=min(base_margins, key=base_margins.get, default=None),
            fixed_timely=fixed['timely'], online_timely=online['timely'],
            fixed_contacts=fixed['contacts'], online_contacts=online['contacts'],
            fixed_adopted=[a['vehicle'] for a in fixed['adopted'].values()],
            online_adopted=[a['vehicle'] for a in online['adopted'].values()],
            fixed_gate3_m=gate3(fixed), online_gate3_m=gate3(online),
            delta_m=min(gate3(online).values())-min(gate3(fixed).values())
            if None not in (*gate3(fixed).values(), *gate3(online).values()) else None)
    return audit


def run(args):
    prior = read(args.calibration/'allocation_plan.json')['calibrated_gain_m']
    with scenario(PORT, args.output, amplitudes=AMPLITUDES, distances=GATES) as (client, gates):
        def fly(name, **spec):
            return run_case(args, client, gates, name, **spec)

        save(args.output/'design.json', dict(amplitudes=AMPLITUDES, gates_m=GATES, phase_x_m=THRESHOLD,
             hidden_velocity_step_mps=STEP_MPS, adoption_target_ms=DEADLINE_MS, roomy_background_calls=ROOMY))
        if args.timing_from:
            calibration = read(args.timing_from)
        else:
            calibration = calibrate_timing(fly)
        reserve, tight = calibration['reserve_ms'], calibration['tight_calls']
        save(args.output/'timing_calibration.json', calibration)
        if tight is None:
            raise RuntimeError('No measured one-command window met the timing conditions')
        baseline = {v: fly(f'local_{v}', policy='local', target=v, calls=tight, reserve=reserve) for v in AMPLITUDES}
        priors = {choice: min(min(m+(prior[v] if v == choice else 0.) for v,m in gate3(b).items())
                              for b in baseline.values()) for choice in AMPLITUDES}
        fixed = max(priors, key=priors.__getitem__)
        planned = [(load, target, rep) for rep in (1, 2) for target in AMPLITUDES for load in ('tight', 'roomy')
                   if load == 'tight' or rep == 1]
        save(args.output/'plan.json', dict(tight_calls=tight, roomy_calls=ROOMY, reserve_ms=reserve,
             static=fixed, static_priors_m=priors, evaluation=planned))
        runs = {}
        for load, target, rep in planned:
            name = f'{load}_{target}_r{rep}'
            order = ('static','online') if ((target == 'Drone1') == (rep == 1)) else ('online','static')
            options = {p: fly(f'{name}_{p}', policy=p, calls=tight if load == 'tight' else ROOMY,
                              reserve=reserve, target=target, fixed=fixed) for p in order}
            runs[name] = (options['static'], options['online'])
            save(args.output/'paired.json', evaluate(runs, baseline))


def calibrate_timing(fly):
    base = fly('timing_roomy', policy='timing', calls=ROOMY, reserve=0)
    if base['timely'] != 2:
        raise RuntimeError('Two timely real commands did not pass the roomy timing probe')
    timing = [base]
    reserve = math.ceil(2*(.5+max(base['adopted'][r['slot']]['since_start_ms']-r['before_ms']
                              for r in base['requests'])))/2
    cost = sum(b-a for a, b in base['background'])/1000/ROOMY
    non_dpu = base['background_done_ms']-cost*ROOMY
    center = round((DEADLINE_MS-reserve-1.-non_dpu)/cost)
    tight = None
    for calls in dict.fromkeys(max(1, min(25, center+i)) for i in (0, 1, -1, 2, -2, 3, 4, 5)):
        observed = fly(f'timing_b{calls}', policy='timing', calls=calls, reserve=reserve)
        timing.append(observed)
        rows = observed['requests']
        if (len(observed['adopted']) == 2 and observed['timely'] == 1
                and rows[0]['before_ms']+reserve <= DEADLINE_MS < rows[1]['before_ms']+reserve
                and observed['adopted'][1]['since_start_ms'] <= DEADLINE_MS):
            tight = calls
            break
    return dict(reserve_ms=reserve, tight_calls=tight, trials=timing)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True, help='Previous allocation result directory')
    parser.add_argument('--timing-from', type=Path, help='Reuse the frozen timing calibration on the same board')
    parser.add_argument('--binary', required=True)
    parser.add_argument('--package', default=BOARD_PACKAGE)
    parser.add_argument('--windows-ip', default='192.168.77.1')
    parser.add_argument('--board', default='axu3egb')
    parser.add_argument('--ssh-hop', default='vitis-vm')
    run(parser.parse_args())
