"""Measure the closed-loop effect of one timely or delayed DPU command."""

import argparse
from contextlib import nullcontext
import math
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

from airsim_interface import require_airsim
from config import load_calibration
from falcon_paper_fidelity.motion_primitive import QuinticMotionPrimitivePlanner
from ..framework_runtime_wait import DeadlineWaiter
from ..ip37_shadow import make_client
from .probe_blocks import LocalRpc
from .run_onboard_dpu import Route, assess, infer_command, sample, verify_model
from .run_onboard_fleet_value import interpolated_margin
from .run_onboard_pilot import Z, scenario, trial
from .run_onboard_stagger_validation import LEADING, ROUTES
from .run_onboard_value import PORT, read, save


CASES = {'early_out': (7.60, .65), 'early_in': (7.60, -.65), 'near_out': (8.55, .65)}
BOUNDARY_CASES = {'mild_early': (7.60, .18), 'mild_near': (8.55, .18)}
ARMS = ('local', 'now', 'late')
DELAY_NS = 40_000_000
BOUNDARY_DELAY_NS = 30_000_000
RECOVERY_BAND_M = .20
BENEFIT_M = .006
STAGGER_S = .90


def kick(client, vehicle, gate_y, delta):
    kin = client.simGetGroundTruthKinematics(vehicle_name=vehicle)
    side = kin.position.y_val - gate_y
    before = kin.linear_velocity.y_val
    change = delta if side >= 0. else -delta
    kin.linear_velocity.y_val += change
    client.simSetKinematics(kin, ignore_collision=False, vehicle_name=vehicle)
    after = client.simGetGroundTruthKinematics(vehicle_name=vehicle)
    return dict(x_m=float(kin.position.x_val), y_m=float(kin.position.y_val),
                requested_delta_mps=change, observed_delta_mps=float(after.linear_velocity.y_val)-before)


def snapshot(state):
    return {name: getattr(state, name).tolist()
            for name in ('position', 'velocity', 'quaternion', 'acceleration', 'angular_velocity')}


def send(rpc, vehicle, slot, stamp, command, wait_ns, stop, timer):
    remaining = stamp + wait_ns - rpc.call('falconTraceClock')
    if remaining > 0:
        if timer is None:
            stop.wait(remaining/1e9)
        else:
            timer.wait([], perf_counter()+remaining/1e9)
    if stop.is_set():
        return False
    return rpc.call('falconPilotRates', vehicle, slot, stamp, *command)


def feed(board, calibration, amplitudes, gate_x, threshold, delta, arm, delay_ns, stop):
    planner = QuinticMotionPrimitivePlanner()
    routes = {v: Route(a) for v, a in amplitudes.items()}
    kicks, done, records = {}, set(), []
    world = require_airsim().MultirotorClient(ip='127.0.0.1', port=PORT, timeout_value=3)
    try:
        # Keep the original pilot's wait behavior for reproducing its measured ages.
        with LocalRpc(PORT) as rpc, (DeadlineWaiter() if delay_ns == BOUNDARY_DELAY_NS else nullcontext()) as timer:
            while len(done) < len(routes) and not stop.is_set():
                if timer is None:
                    if stop.wait(.004):
                        break
                else:
                    timer.wait([], perf_counter()+.004)
                for vehicle, route in routes.items():
                    if vehicle in done:
                        continue
                    stamp, state = sample(rpc, vehicle)
                    x = state.position[0]
                    if vehicle not in kicks and x >= threshold-.26:
                        kicks[vehicle] = kick(world, vehicle, route.amplitude*math.sin(.6*gate_x), delta)
                        continue
                    if x < threshold or vehicle not in kicks:
                        continue
                    done.add(vehicle)
                    row = dict(vehicle=vehicle, slot=len(records)+1, sample_ns=stamp,
                               state=snapshot(state), kick=kicks[vehicle], accepted=False, status='error')
                    records.append(row)
                    try:
                        command, reply = infer_command(board, calibration, planner, route, state)
                        row.update(reply)
                        if arm != 'local':
                            row['accepted'] = send(rpc, vehicle, row['slot'], stamp, command,
                                                   delay_ns if arm == 'late' else 0, stop, timer)
                    except (OSError, ValueError) as exc:
                        row.update(status='error', error=str(exc))
    finally:
        world.client.close()
    return {}, records


def outcome(trace, row, amplitude, gate):
    window = [t for t in trace if t[0] >= row['sample_ns'] and t[1] <= gate.center[0]]
    error = [abs(t[2]-amplitude*math.sin(.6*t[1])) for t in window]
    peak = max(error, default=None)
    recovery = None
    if window and error[0] > RECOVERY_BAND_M:
        start = None
        for t, e in zip(window, error):
            start = (start or t[0]) if e <= RECOVERY_BAND_M else None
            if start is not None and t[0]-start >= 60_000_000:
                recovery = (start-row['sample_ns'])/1e6
                break
    adopted = next((t for t in trace if t[4] == 3 and t[5] == row['slot']), None)
    resumed = next((t for t in trace if adopted is not None and t[0] > adopted[0] and t[4] == 1), None)
    return dict(margin_m=interpolated_margin(trace, gate.center), peak_error_m=peak,
                recovery_ms=recovery, recovery_needed=bool(error and error[0] > RECOVERY_BAND_M),
                local_resume_after_adoption_ms=None if resumed is None else (resumed[0]-adopted[0])/1e6,
                adopted_age_ms=None if adopted is None else (adopted[0]-adopted[6])/1e6)


def matched(a, b):
    distance = float(np.linalg.norm(np.subtract(a['position'], b['position'])))
    velocity = float(np.linalg.norm(np.subtract(a['velocity'], b['velocity'])))
    q = [np.asarray(s['quaternion']) for s in (a, b)]
    cosine = abs(float(q[0]@q[1]))/(np.linalg.norm(q[0])*np.linalg.norm(q[1]))
    attitude = math.degrees(2*math.acos(min(1., cosine)))
    return dict(position_m=distance, velocity_mps=velocity, attitude_deg=attitude,
                matched=bool(distance <= .022 and velocity <= .055 and attitude <= 1.))


def measure(client, gates, path, amplitudes, distance, board, calibration, threshold, delta, arm, delay_ns):
    def pulse(stop):
        return feed(board, calibration, amplitudes, distance, threshold, delta, arm, delay_ns, stop)

    result = trial(client, PORT, path, gates, 3, False, pulse, leading=LEADING,
                   stagger_s=STAGGER_S, amplitudes=amplitudes)
    protocol = assess(path, result, require_gap=False)
    values = {}
    for row in read(path/'board_requests.json'):
        vehicle = row['vehicle']
        values[vehicle] = dict(state=row['state'], kick=row['kick'], ready=row['status'] == 'ready',
                               contact=result[vehicle]['contact'],
                               **outcome(read(path/f'{vehicle}.json'), row, amplitudes[vehicle], gates[vehicle][-1]))
    report = dict(protocol=protocol, vehicles=values)
    save(path/'pulse_report.json', report)
    print(path.name, [(v, round(r['margin_m']*1000, 2) if r['margin_m'] is not None else None,
                       r['adopted_age_ms']) for v, r in values.items()], flush=True)
    return report


def comparisons(runs, cases, delay_ns):
    details, candidates = [], []
    for case in cases:
        for vehicle in ROUTES['independent'][0]:
            group = []
            for repetition in range(1, 4):
                names = [f'{case}_r{repetition}_{arm}' for arm in ARMS]
                rows = [runs.get(n, {}).get('vehicles', {}).get(vehicle) for n in names]
                if any(r is None for r in rows):
                    continue
                local, now, late = rows
                match = [matched(local['state'], r['state']) for r in (now, late)]
                timing = (local['adopted_age_ms'] is None and
                          now['adopted_age_ms'] is not None and now['adopted_age_ms'] <= 20 and
                          late['adopted_age_ms'] is not None and delay_ns/1e6-2 <= late['adopted_age_ms'] < 55)
                if delay_ns == BOUNDARY_DELAY_NS:
                    timing = timing and late['adopted_age_ms'] <= 35
                valid = (timing and all(x['matched'] for x in match) and
                         all(r['ready'] and
                             abs(r['kick']['observed_delta_mps']-r['kick']['requested_delta_mps']) <= .15
                             for r in rows))
                margin = None if any(r['margin_m'] is None for r in rows) else [
                    now['margin_m']-local['margin_m'], late['margin_m']-local['margin_m']]
                entry = dict(case=case, repetition=repetition, vehicle=vehicle, matched=match,
                             timing_pass=timing, valid=valid, margin_delta_m=margin,
                             contacts=[r['contact'] for r in rows])
                details.append(entry)
                if valid and margin is not None and not now['contact']:
                    group.append(entry)
            if len(group) == 3:
                gains = [r['margin_delta_m'][0] for r in group]
                delay_loss = [r['margin_delta_m'][0]-r['margin_delta_m'][1] for r in group]
                if min(gains) > 0 and median(gains) >= BENEFIT_M:
                    candidates.append(dict(case=case, vehicle=vehicle, gains_m=gains, delay_loss_m=delay_loss))
    return dict(triplets=details, candidates=candidates)


def allocation_feed(board, calibration, amplitudes, gate_x, threshold, delta, disturbed, policy, static, stop):
    planner = QuinticMotionPrimitivePlanner()
    gate_y = {v: a*math.sin(.6*gate_x) for v, a in amplitudes.items()}
    world = require_airsim().MultirotorClient(ip='127.0.0.1', port=PORT, timeout_value=3)
    try:
        with LocalRpc(PORT) as rpc, DeadlineWaiter() as timer:
            applied = None
            while not stop.is_set():
                timer.wait([], perf_counter()+.004)
                samples = {v: sample(rpc, v) for v in amplitudes}
                if applied is None and samples[disturbed][1].position[0] >= threshold-.26:
                    applied = kick(world, disturbed, gate_y[disturbed], delta)
                    continue
                if applied is None or any(s.position[0] < threshold for _, s in samples.values()):
                    continue
                # Rank only from observed state, with the same lookahead for both aircraft.
                risk = {v: float(abs(s.position[1]+.25*s.velocity[1]-gate_y[v])) for v, (_, s) in samples.items()}
                recipient = max(risk, key=risk.__getitem__) if policy == 'online' else static
                stamp, state = samples[recipient]
                row = dict(vehicle=recipient, slot=1, sample_ns=stamp, state=snapshot(state),
                           states={v: snapshot(s) for v, (_, s) in samples.items()},
                           projected_error_m=risk, kick=applied, accepted=False, status='error')
                try:
                    command, reply = infer_command(board, calibration, planner, Route(amplitudes[recipient]), state)
                    row.update(reply)
                    if policy != 'local':
                        row['accepted'] = send(rpc, recipient, 1, stamp, command, 0, stop, timer)
                except (OSError, ValueError) as exc:
                    row.update(status='error', error=str(exc))
                return {}, [row]
    finally:
        world.client.close()
    return {}, []


def allocation(client, gates, args, board, calibration, amplitudes, distances, plan):
    calibration_dir = args.allocation_from
    calibration_runs = read(calibration_dir/'summary.json')
    positive = read(calibration_dir/'paired.json')['candidates']
    if not positive:
        raise ValueError('Single-command calibration does not qualify any allocation state')
    case = max(positive, key=lambda r: median(r['gains_m']))['case']
    threshold, delta = read(calibration_dir/'pulse_plan.json')['cases'][case]
    # Freeze the strongest fixed recipient under the measured two possible
    # disturbed-vehicle cases. Calibration gives credit only for qualified gains.
    historical = read(args.frozen/'departure_plans.json')['independent']['baseline_m']
    base = {v: historical[v]['gate3'] for v in amplitudes}
    pulse = {v: median(calibration_runs[f'{case}_r{i}_local']['vehicles'][v]['margin_m']
                       for i in range(1, 4)) for v in amplitudes}
    gain = {v: next((median(r['gains_m']) for r in positive if r['case'] == case and r['vehicle'] == v), 0.)
            for v in amplitudes}
    predicted = {choice: min(min(pulse[v]+(gain[v] if choice == v else 0.) if v == target else base[v]
                                 for v in amplitudes) for target in amplitudes) for choice in amplitudes}
    static = max(predicted, key=predicted.__getitem__)
    schedule = [(target, arm) for target, order in zip(amplitudes, (('local', 'static', 'online'),
                                                                    ('local', 'online', 'static')))
                for arm in order]
    save(args.output/'allocation_plan.json', dict(case=case, threshold_m=threshold, velocity_step_mps=delta,
         one_command_per_run=True, disturbance_unknown_before_launch=True, static=static, static_prior_m=predicted,
         historical_gate3_m=base, local_calibrated_pulse_m=pulse, calibrated_gain_m=gain, schedule=schedule,
         model=plan))
    runs = {}
    for target, arm in schedule:
        def pulse_feed(stop):
            return allocation_feed(board, calibration, amplitudes, distances[-1], threshold, delta, target,
                                   arm, static, stop)
        name = f'{target}_{arm}'
        path = args.output/name
        result = trial(client, PORT, path, gates, 3, False, pulse_feed, leading=LEADING, amplitudes=amplitudes)
        protocol = assess(path, result, require_gap=False)
        rows = read(path/'board_requests.json')
        traces = {v: read(path/f'{v}.json') for v in amplitudes}
        margins = {v: interpolated_margin(traces[v], gates[v][-1].center) for v in amplitudes}
        ages = [(r[0]-r[6])/1e6 for trace in traces.values() for r in trace if r[4] == 3]
        runs[name] = dict(target=target, policy=arm, protocol=protocol, request=rows[0] if rows else None,
                          by_vehicle_m=margins, adopted_ages_ms=ages,
                          contacts={v: r['contact'] for v, r in result.items()})
        save(args.output/'summary.json', runs)
        print(name, rows[0]['vehicle'] if rows else None, margins, flush=True)
    evaluation = {}
    for target in amplitudes:
        local, fixed, online = (runs[f'{target}_{arm}'] for arm in ('local', 'static', 'online'))
        if any(r['request'] is None for r in (local, fixed, online)):
            evaluation[target] = dict(valid=False, reason='trial has no trigger sample')
            continue
        matches = {arm['policy']: {v: matched(local['request']['states'][v], arm['request']['states'][v])
                                   for v in amplitudes} for arm in (fixed, online)}
        valid = (all(m['matched'] for arm in matches.values() for m in arm.values())
                 and all(r['protocol']['ready'] == 1 for r in (local, fixed, online))
                 and not local['adopted_ages_ms']
                 and all(len(r['adopted_ages_ms']) == 1 and r['adopted_ages_ms'][0] <= 20 for r in (fixed, online)))
        minima = {r['policy']: min(r['by_vehicle_m'].values()) if None not in r['by_vehicle_m'].values() else None
                  for r in (local, fixed, online)}
        evaluation[target] = dict(valid=valid, states_matched=matches, minima_m=minima,
             local_limiting_vehicle=min(local['by_vehicle_m'], key=local['by_vehicle_m'].get)
             if minima['local'] is not None else None, static_recipient=fixed['request']['vehicle'],
             online_recipient=online['request']['vehicle'])
    save(args.output/'allocation_audit.json', evaluation)


def run(args):
    cases = BOUNDARY_CASES if args.boundary else CASES
    delay_ns = BOUNDARY_DELAY_NS if args.boundary else DELAY_NS
    amplitudes, distances = ROUTES['independent']
    calibration = load_calibration()
    board, manifest = make_client(args.bundle, args.board, args.board_port, 35.)
    try:
        model = verify_model(board, manifest, args.bundle)
        with scenario(PORT, args.output, amplitudes=amplitudes, distances=distances) as (client, gates):
            if args.allocation_from:
                return allocation(client, gates, args, board, calibration, amplitudes, distances, model)
            schedule = [(case, rep, arm) for rep in range(1, 4) for case in cases
                        for arm in ARMS[rep-1:]+ARMS[:rep-1]]
            save(args.output/'pulse_plan.json', dict(model=model, cases=cases, schedule=schedule,
                 late_target_ns=delay_ns, benefit_threshold_m=BENEFIT_M, stagger_s=STAGGER_S))
            runs = {}
            for case, repetition, arm in schedule:
                threshold, delta = cases[case]
                name = f'{case}_r{repetition}_{arm}'
                runs[name] = measure(client, gates, args.output/name, amplitudes, distances[-1], board,
                                     calibration, threshold, delta, arm, delay_ns)
                save(args.output/'summary.json', runs)
                save(args.output/'paired.json', comparisons(runs, cases, delay_ns))
    finally:
        board.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--board', default='192.168.77.2')
    parser.add_argument('--board-port', type=int, default=5240)
    parser.add_argument('--boundary', action='store_true', help='Repeat the comparison with smaller lateral velocity pulses')
    parser.add_argument('--allocation-from', type=Path, help='Use qualified states from the single-command calibration')
    parser.add_argument('--frozen', type=Path, help='Frozen route calibration for the static allocation baseline')
    args = parser.parse_args()
    if args.allocation_from and not args.frozen:
        parser.error('--allocation-from requires --frozen')
    run(args)
