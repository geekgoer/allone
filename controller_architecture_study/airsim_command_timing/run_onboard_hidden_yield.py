"""Compare online yielding against training-tuned static dispatch and EDF."""

import argparse
from itertools import product
from pathlib import Path
from statistics import mean

from .run_onboard_deadline_route import AMPLITUDES, DEADLINE_MS, GATES, PORT, STEP_MPS, THRESHOLD, run_case, scenario
from .run_onboard_reserved_recovery import CALLS, HEALTHY_TOLERANCE_M
from .run_onboard_stage_recovery import RESERVE_MS
from .run_onboard_value import save
from .run_onboard_yield_competition import WINDOWS, audit_group
from .run_ps_onboard_pair import BOARD_PACKAGE


STATIC = {f's{a}{b}': ('Drone2', f'Drone{a}', f'Drone{b}') for a, b in product((1, 2), repeat=2)}
# Peer velocity step (m/s), trigger offset from the first window (m), preceding sample window.
TRAIN = {'quiet': (0., .075, 0), 'peer75': (.256, .075, 0)}
HOLDOUT = {'quiet': (0., .09, 0), 'early': (.255, .060, 0),
           'near_second': (.257, .095, 0), 'after_second': (.256, .20, 1)}
ORDERS = (
    ('static', 'deadline', 'yield'),
    ('deadline', 'yield', 'static'),
    ('yield', 'static', 'deadline'),
    ('static', 'yield', 'deadline'),
)


def static_scores(training):
    scores = {}
    for name in STATIC:
        rows = [group[name] for group in training.values()]
        margins = [min(r['gate3_m'].values()) for r in rows if None not in r['gate3_m'].values()]
        if all(r['comparable'] for r in rows) and len(margins) == len(rows):
            scores[name] = (sum(r['pass_gates'] for r in rows), min(margins), mean(margins))
    return scores


def summarize(holdout):
    summary = {}
    for name in ORDERS[0]:
        groups = [g for g in holdout.values() if all(r['comparable'] for r in g.values())]
        rows = [g[name] for g in groups]
        margins = [min(r['gate3_m'].values())*1000 for r in rows if None not in r['gate3_m'].values()]
        summary[name] = dict(valid_cases=len(rows), team_passes=sum(r['pass_gates'] for r in rows),
            contacts={v: sum(bool(r['contacts'][v]) for r in rows) for v in AMPLITUDES},
            mean_min_gate3_mm=mean(margins) if margins else None, worst_gate3_mm=min(margins) if margins else None)
    return summary


def run(args):
    with scenario(PORT, args.output, amplitudes=AMPLITUDES, distances=GATES) as (client, gates):
        save(args.output/'design.json', dict(aircraft=AMPLITUDES, gates_m=GATES, windows_ms=WINDOWS,
            primary_step_mps=STEP_MPS, training=TRAIN, holdout=HOLDOUT, holdout_order=ORDERS,
            static_candidates=STATIC, static_objective=('team_passes', 'worst_min_gate3', 'mean_min_gate3'),
            background_per_window=CALLS, adoption_target_ms=DEADLINE_MS, reserve_ms=RESERVE_MS,
            deadline_ms=dict(Drone1=WINDOWS[1]+DEADLINE_MS, Drone2=WINDOWS[2]+DEADLINE_MS)))

        def fly(name, policy, case, recipients=(), peer_trigger_m=0.):
            step, offset, _ = case
            return run_case(args, client, gates, name, policy=policy, calls=CALLS, reserve=RESERVE_MS,
                recovery_schedule=dict(windows=len(WINDOWS), window_offsets_ms=WINDOWS, healthy_error_m=[],
                                       fixed_recipients=recipients, peer_trigger_m=peer_trigger_m),
                staged=(AMPLITUDES, GATES, THRESHOLD, (STEP_MPS, step, offset)))

        training = {}
        for name, case in TRAIN.items():
            order = tuple(STATIC) if name == 'quiet' else tuple(reversed(STATIC))
            reports = {p: fly(f'train_{name}_{p}', 'fixed', case, STATIC[p]) for p in order}
            training[name] = audit_group(reports, 's21')
            if name == 'quiet':
                peer_trigger = reports['s21']['windows'][1]['risk']['Drone1'] + HEALTHY_TOLERANCE_M
            save(args.output/'training.json', training)
        scores = static_scores(training)
        # The static reference must be chosen using equally complete training evidence.
        fixed = max(scores, key=scores.__getitem__) if set(scores) == set(STATIC) else None
        save(args.output/'plan.json', dict(selected_static=fixed, training_scores=scores, peer_trigger_m=peer_trigger))
        if fixed is None:
            raise RuntimeError('Static candidates were not all comparable in training.')

        holdout = {}
        for (name, case), order in zip(HOLDOUT.items(), ORDERS):
            reports = {p: fly(f'test_{name}_{p}', 'fixed' if p == 'static' else p, case,
                              STATIC[fixed] if p == 'static' else (), peer_trigger) for p in order}
            holdout[name] = audit_group(reports, 'static', kick_after_window=case[2])
            save(args.output/'holdout.json', holdout)
            save(args.output/'summary.json', summarize(holdout))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--binary', required=True)
    parser.add_argument('--package', default=BOARD_PACKAGE)
    parser.add_argument('--windows-ip', default='192.168.77.1')
    parser.add_argument('--board', default='axu3egb')
    parser.add_argument('--ssh-hop', default='vitis-vm')
    run(parser.parse_args())
