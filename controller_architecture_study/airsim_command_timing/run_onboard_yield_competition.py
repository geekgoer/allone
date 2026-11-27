"""Compare three ways to share a recovery window before a measured follow-up."""

import argparse
from pathlib import Path
from statistics import median

from .run_onboard_crossing_risk import staged_timing_ok
from .run_onboard_deadline_route import AMPLITUDES, DEADLINE_MS, GATES, PORT, STEP_MPS, THRESHOLD, gate3, run_case, scenario
from .run_onboard_reserved_recovery import CALLS, HEALTHY_TOLERANCE_M, state_match
from .run_onboard_stage_recovery import RESERVE_MS
from .run_onboard_value import save
from .run_ps_onboard_pair import BOARD_PACKAGE


WINDOWS = (0, 75, 300)
RESERVED = ('Drone2', 'Drone2', 'Drone1')
YIELDED = ('Drone2', 'Drone1', 'Drone2')
PEER_STEPS = (.27, .285, .30, .26, .25)
ORDERS = (
    ('reserve', 'risk', 'yield'),
    ('risk', 'yield', 'reserve'),
    ('yield', 'reserve', 'risk'),
)


def audit_trial(report):
    by_window = {r['window']: r for r in report['adopted'].values()}
    adoption = [by_window[w]['vehicle'] if w in by_window else None for w in range(len(WINDOWS))]
    sampled = [w['start_ns'] for w in report['windows']]
    timing_ok = (len(report['adopted']) == len(by_window) == len(WINDOWS) and
                 set(by_window) == set(range(len(WINDOWS))) and all(r['on_time'] for r in by_window.values()) and
                 all(abs((ns-sampled[0])/1e6-ms) <= 5. for ns, ms in zip(sampled, WINDOWS)))
    planned = [phase['first_candidate'] for phase in report['windows']]
    return dict(adoption=adoption, timing_ok=timing_ok and adoption == planned,
                stage_ok=staged_timing_ok(report), pass_gates=report['pass_gates'],
                contacts=report['contacts'], gate3_m=gate3(report),
                background_calls=report['background_calls'],
                background_dpu_ms=sum(end-start for start, end in report['background'])/1000,
                adoption_since_trigger_ms=[by_window[w]['since_start_ms'] for w in sorted(by_window)],
                followup_owed=[phase.get('followup_owed') for phase in report['windows']])


def audit_group(reports, reference):
    base = reports[reference]
    rows = {}
    for name, report in reports.items():
        row = audit_trial(report)
        state_ok = all(state_match(base['windows'][w], report['windows'][w]) for w in (0, 1))
        kick_dx = {v: abs(base['velocity_step'][v]['x_m']-report['velocity_step'][v]['x_m']) for v in AMPLITUDES}
        row.update(state_matched=state_ok, kick_dx_m=kick_dx,
                   comparable=state_ok and max(kick_dx.values()) <= .022 and row['timing_ok'] and row['stage_ok']
                   and row['adoption'][0] == 'Drone2' and row['background_calls'] == len(WINDOWS)*CALLS)
        rows[name] = row
    if not rows[reference]['comparable']:
        for row in rows.values():
            row['comparable'] = False
    return rows


def summarize(groups):
    summary = {}
    for name in ORDERS[0]:
        rows = [group[name] for group in groups.values() if all(r['comparable'] for r in group.values())]
        margins = {v: [r['gate3_m'][v]*1000 for r in rows if r['gate3_m'][v] is not None] for v in AMPLITUDES}
        summary[name] = dict(comparable_groups=len(rows), all_gates=sum(r['pass_gates'] for r in rows),
            contacts={v: sum(bool(r['contacts'][v]) for r in rows) for v in AMPLITUDES},
            median_gate3_mm={v: median(m) if m else None for v, m in margins.items()},
            recipients=[r['adoption'] for r in rows])
    return summary


def run(args):
    with scenario(PORT, args.output, amplitudes=AMPLITUDES, distances=GATES) as (client, gates):
        save(args.output/'design.json', dict(gates_m=GATES, amplitudes=AMPLITUDES, trigger_m=THRESHOLD,
             windows_ms=WINDOWS, background_calls_per_window=CALLS, reserve_ms=RESERVE_MS, adoption_target_ms=DEADLINE_MS,
             drone2_step_mps=STEP_MPS, peer_steps_mps=args.peer_steps, order=ORDERS,
             screen='first matched pair with Drone1 contact in reserve and both aircraft clear in yield'))

        def fly(name, policy, peer_step, recipients=(), peer_trigger_m=0.):
            return run_case(args, client, gates, name, policy=policy, calls=CALLS, reserve=RESERVE_MS,
                recovery_schedule=dict(windows=len(WINDOWS), window_offsets_ms=WINDOWS, healthy_error_m=[],
                                       fixed_recipients=recipients, peer_trigger_m=peer_trigger_m),
                staged=(AMPLITUDES, GATES, THRESHOLD, (STEP_MPS, peer_step)))

        nominal = fly('peer_nominal', 'fixed', 0., RESERVED)
        peer_trigger = nominal['windows'][1]['risk']['Drone1'] + HEALTHY_TOLERANCE_M
        screens = {}
        selected = None
        for i, step in enumerate(args.peer_steps, 1):
            reports = {arm: fly(f'screen_{i}_{arm}', 'fixed', step, recipients)
                       for arm, recipients in (('reserve', RESERVED), ('yield', YIELDED))}
            rows = audit_group(reports, 'reserve')
            peer_risk = min(r['windows'][1]['risk']['Drone1'] for r in reports.values())
            screens[i] = dict(peer_step_mps=step, peer_risk_m=peer_risk, arms=rows)
            save(args.output/'screen.json', screens)
            if (all(r['comparable'] for r in rows.values()) and peer_risk > peer_trigger and
                    rows['reserve']['contacts']['Drone1'] and not rows['reserve']['contacts']['Drone2'] and
                    rows['yield']['pass_gates']):
                selected = step
                break
        save(args.output/'plan.json', dict(peer_step_mps=selected, peer_trigger_m=peer_trigger))
        if selected is None:
            print('No eligible peer disturbance in the frozen screen.', flush=True)
            return

        results = {}
        for rep, order in enumerate(ORDERS, 1):
            reports = {p: fly(f'r{rep}_{p}', 'fixed' if p == 'reserve' else p, selected,
                              RESERVED if p == 'reserve' else (), peer_trigger) for p in order}
            results[rep] = audit_group(reports, 'reserve')
            save(args.output/'groups.json', results)
            save(args.output/'summary.json', summarize(results))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--binary', required=True)
    parser.add_argument('--package', default=BOARD_PACKAGE)
    parser.add_argument('--windows-ip', default='192.168.77.1')
    parser.add_argument('--board', default='axu3egb')
    parser.add_argument('--ssh-hop', default='vitis-vm')
    parser.add_argument('--peer-steps', type=float, nargs='+', default=PEER_STEPS,
                        help='Ordered peer perturbation magnitudes for the calibration screen, in m/s.')
    args = parser.parse_args()
    if not all(0. < step <= .5 for step in args.peer_steps):
        parser.error('--peer-steps must contain magnitudes in (0, 0.5] m/s')
    run(args)
