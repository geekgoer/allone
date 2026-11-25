"""Freeze optional-slot deferral scores, then test both recipients in AirSim."""

import argparse
from functools import partial
from hashlib import sha256
from pathlib import Path

from config import load_calibration
from ..ip37_shadow import make_client
from .run_onboard_dpu import verify_model
from .run_onboard_fleet_value import run_trial, trial_margins
from .run_onboard_load_replay import STAGGER
from .run_onboard_pilot import scenario
from .run_onboard_slot_pair import TARGET_SLOT, audit_trial, state_difference
from .run_onboard_stagger_validation import LEADING, ROUTES
from .run_onboard_value import BAND_M, PORT, SLOT_S, allocate, read, save


MARGIN_FLOOR_M = .0008
MAX_DEFERRAL_GAP_SLOTS = 14


def score(positions, donor, receiver, calibration, future_gap_slots):
    gates = {v: next((f'gate{g}' for g, x in enumerate(calibration['gate_x_m'], 1)
                     if positions[v] < x), None) for v in (donor, receiver)}
    phase = next((k for k, x in calibration['probe_x_m'].items()
                  if abs(positions[receiver]-x) <= BAND_M), None)
    baseline = calibration['baseline_m']
    if (any(gates[v] not in baseline[v] for v in gates) or
            phase is None or f'gate{phase[1]}' != gates[receiver]):
        return None
    samples = calibration['matched_deltas_m']
    benefit = min(samples[phase])
    cost = max(abs(x) for k, values in samples.items() if f'gate{k[1]}' == gates[donor] for x in values)
    m_donor, m_receiver = (baseline[v][gates[v]] for v in (donor, receiver))
    reserve = m_donor-cost
    delta = min(reserve, m_receiver+benefit)-min(m_donor, m_receiver)
    allow = future_gap_slots <= MAX_DEFERRAL_GAP_SLOTS and reserve > 0 and delta > MARGIN_FLOOR_M
    return dict(gates=gates, phase=phase, donor_baseline_m=m_donor, receiver_baseline_m=m_receiver,
                receiver_gain_m=benefit, donor_cost_reserve_m=cost, predicted_min_delta_m=delta, allow=allow)


def freeze(source):
    plans, masks = (read(source / f'{name}.json') for name in ('departure_plans', 'load_masks'))
    paths = [source / f'{name}.json' for name in ('departure_plans', 'load_masks', 'deployment')]
    cases = {}
    for route, calibration in plans.items():
        for window, slots in masks['windows'].items():
            occupied = [r['slot'] for r in slots if r['covering_call_ns']]
            traces = [source / route / f'{window}_r{i}_live_position' / 'board_requests.json' for i in (1, 2)]
            paths.extend(traces)
            rows = [read(path) for path in traces]
            schedules = [{r['slot']: r['vehicle'] for r in trial} for trial in rows]
            schedule = schedules[0]
            if (schedule != schedules[1] or
                    list(schedule) != [r['slot'] for r in slots if r['slot'] not in occupied]):
                raise ValueError('Expected two reproducible historical recipient sequences')
            previous = {}
            for i, (slot, donor) in enumerate(schedule.items()):
                past = previous.get(donor)
                previous[donor] = slot
                receiver = next(v for v in calibration['amplitudes'] if v != donor)
                following = next((n for n, v in schedule.items() if n > slot and v == donor), None)
                if (slot == TARGET_SLOT or past is None or slot-past < 4 or following is None or
                        receiver not in previous or not any(n > slot and v == receiver for n, v in schedule.items())):
                    continue
                gap = following-past
                predictions = [score(trial[i]['positions_x_m'], donor, receiver, calibration, gap) for trial in rows]
                if None in predictions or predictions[0] != predictions[1]:
                    continue
                cases[f'{route}_{window}_s{slot}'] = dict(
                    route=route, window=window, slot=slot, donor=donor, receiver=receiver, schedule=schedule,
                    occupied=occupied, donor_previous_slot=past, donor_reserved_slot=following,
                    gap_slots=gap, prediction=predictions[0])
    return dict(cases=cases, minimum_effect_m=MARGIN_FLOOR_M, slot_s=SLOT_S,
                max_gap_slots=MAX_DEFERRAL_GAP_SLOTS,
                source_sha256={p.relative_to(source).as_posix(): sha256(p.read_bytes()).hexdigest() for p in paths})


def paired_results(keep, swap, case):
    donor, receiver = case['donor'], case['receiver']
    initial = {v: state_difference(keep['target_states'][v], swap['target_states'][v])
               for v in keep['target_states']}
    gates = case['prediction']['gates']
    margins = [{v: row['by_gate_m'][v][g] for v, g in gates.items()} for row in (keep, swap)]
    delta = {v: margins[1][v]-margins[0][v] if None not in (margins[0][v], margins[1][v]) else None
             for v in gates}
    delta_floor = min(margins[1].values())-min(margins[0].values()) if None not in delta.values() else None
    fleet = (swap['fleet_min_m']-keep['fleet_min_m']
             if None not in (keep['fleet_min_m'], swap['fleet_min_m']) else None)
    return dict(matched=all(r['matched'] for r in initial.values()), pre_slot_difference=initial,
                donor_delta_m=delta[donor], receiver_delta_m=delta[receiver], two_gate_delta_m=delta_floor,
                fleet_delta_m=fleet,
                beneficial=bool(delta_floor is not None and delta_floor > MARGIN_FLOOR_M and
                                all(v['pass_gates'] for row in (keep, swap) for v in row['vehicles'].values())))


def run(args):
    frozen = freeze(args.replay_from)
    args.output.mkdir(parents=True)
    save(args.output / 'frozen_experiment.json', frozen)
    board, manifest = make_client(args.bundle, args.board, args.board_port, 35.)
    summary = {}
    try:
        identity = verify_model(board, manifest, args.bundle)
        if identity != read(args.replay_from / 'deployment.json'):
            raise ValueError('Expected the source replay board deployment')
        calib = load_calibration()
        route_calibration = read(args.replay_from / 'departure_plans.json')
        for route, (amplitudes, distances) in ROUTES.items():
            with scenario(PORT, args.output / route, amplitudes=amplitudes, distances=distances) as (client, gates):
                for name, case in frozen['cases'].items():
                    if case['route'] != route:
                        continue
                    slot = case['slot']
                    donor, receiver = case['donor'], case['receiver']
                    (args.output / route / name).mkdir()
                    for rep, order in enumerate(((donor, receiver), (receiver, donor)), 1):
                        trials = {}
                        for recipient in order:
                            plan = dict(case['schedule'])
                            plan[slot] = recipient
                            path = args.output / route / name / f'r{rep}_{recipient}'
                            def selector(seq, positions, last):
                                return plan[seq], dict.fromkeys(positions, 0.)
                            feed = partial(allocate, board, calib, None, selector=selector, amplitudes=amplitudes,
                                           occupied=case['occupied'], capture_state_slot=slot)
                            result = run_trial(client, gates, path, feed, LEADING, STAGGER, amplitudes=amplitudes)
                            audit = audit_trial(path, plan, result, target_slot=slot)
                            observed = {v: state['position'][0] for v, state in audit['target_states'].items()}
                            trials[recipient] = dict(
                                board=result['board'], vehicles=result['vehicles'],
                                shadow_prediction=score(observed, donor, receiver, route_calibration[route], case['gap_slots']),
                                **trial_margins(path, amplitudes=amplitudes, distances=distances), **audit)
                        summary[f'{name}/r{rep}'] = dict(
                            trials=trials, **paired_results(trials[donor], trials[receiver], case))
                        save(args.output / 'summary.json', summary)
    finally:
        board.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay-from', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--board', default='192.168.77.2')
    parser.add_argument('--board-port', type=int, default=5240)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
