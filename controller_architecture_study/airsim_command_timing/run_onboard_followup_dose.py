"""Measure two follow-up opportunities against matched closed-loop controls."""

import argparse
from pathlib import Path
from statistics import median

from .run_onboard_crossing_risk import AMPLITUDES, GATES, THRESHOLD, staged_timing_ok
from .run_onboard_deadline_route import DEADLINE_MS, PORT, gate3, run_case, scenario
from .run_onboard_reserved_recovery import CALLS, state_match
from .run_onboard_stage_recovery import RESERVE_MS
from .run_onboard_value import save
from .run_ps_onboard_pair import BOARD_PACKAGE


STEPS = (.16, .35)
ARMS = {
    'none': ('', ''),
    'd1_early': ('Drone1', ''),
    'd1_late': ('', 'Drone1'),
    'd1_both': ('Drone1', 'Drone1'),
    'd2_early': ('Drone2', ''),
    'd2_late': ('', 'Drone2'),
    'd2_both': ('Drone2', 'Drone2'),
}
ORDERS = (
    tuple(ARMS),
    tuple(reversed(ARMS)),
    ('d1_both', 'd2_early', 'd1_early', 'd2_both', 'none', 'd1_late', 'd2_late'),
)


def verify_delivery(report, recipients):
    planned = {i: v for i, v in enumerate(recipients) if v}
    adopted = {row['window']: row['vehicle'] for row in report['adopted'].values() if row['on_time']}
    return (staged_timing_ok(report) and report['background_calls'] == len(recipients)*CALLS
            and adopted == planned and len(report['adopted']) == len(planned))


def audit_round(reports):
    base = reports['none']
    output = {}
    for name, report in reports.items():
        recipients = ('Drone2', *ARMS[name])
        delivery = verify_delivery(report, recipients)
        match = all(state_match(base['windows'][w], report['windows'][w]) for w in (0, 1))
        dx = {v: abs(base['velocity_step'][v]['x_m']-report['velocity_step'][v]['x_m']) for v in AMPLITUDES}
        comparable = delivery and match and max(dx.values()) <= .022
        margins = gate3(report)
        baseline_margins = gate3(base)
        deltas = {v: None if None in (margins[v], baseline_margins[v]) else margins[v]-baseline_margins[v]
                  for v in AMPLITUDES}
        output[name] = dict(comparable=comparable, matched=match, delivery_ok=delivery, kick_dx_m=dx,
                            planned_recipients=recipients, gate3_m=margins, delta_vs_none_m=deltas,
                            contacts=report['contacts'], pass_gates=report['pass_gates'],
                            background_calls=report['background_calls'],
                            adoption_ms={r['window']: r['window_ms'] for r in report['adopted'].values()})
    if not output['none']['comparable']:
        for row in output.values():
            row['comparable'] = False
    return output


def summarize(rounds):
    summary = {}
    for name in ARMS:
        rows = [r[name] for r in rounds.values() if r[name]['comparable']]
        summary[name] = dict(comparable=len(rows), planned=len(rounds),
                             all_gates=sum(r['pass_gates'] for r in rows), by_aircraft={})
        for v in AMPLITUDES:
            margins = [r['gate3_m'][v]*1000 for r in rows if r['gate3_m'][v] is not None]
            delta = [r['delta_vs_none_m'][v]*1000 for r in rows if r['delta_vs_none_m'][v] is not None]
            summary[name]['by_aircraft'][v] = dict(contacts=sum(bool(r['contacts'][v]) for r in rows),
                median_margin_mm=median(margins) if margins else None,
                median_delta_vs_none_mm=median(delta) if delta else None)
    return summary


def run(args):
    with scenario(PORT, args.output, amplitudes=AMPLITUDES, distances=GATES) as (client, gates):
        save(args.output/'design.json', dict(gates_m=GATES, amplitudes=AMPLITUDES, steps_mps=STEPS,
             first_recipient='Drone2', followup_recipients=ARMS, order=ORDERS, windows_ms=(0, 75, 150),
             calls_per_window=CALLS, deadline_ms=DEADLINE_MS, reserve_ms=RESERVE_MS))
        audits = {}
        for rep, order in enumerate(ORDERS, 1):
            reports = {}
            for name in order:
                recipients = ('Drone2', *ARMS[name])
                reports[name] = run_case(args, client, gates, f'r{rep}_{name}', policy='fixed',
                    calls=CALLS, reserve=RESERVE_MS,
                    recovery_schedule=dict(windows=3, healthy_error_m=[], fixed_recipients=recipients),
                    staged=(AMPLITUDES, GATES, THRESHOLD, STEPS))
            audits[rep] = audit_round(reports)
            save(args.output/'rounds.json', audits)
            save(args.output/'summary.json', summarize(audits))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--binary', required=True)
    parser.add_argument('--package', default=BOARD_PACKAGE)
    parser.add_argument('--windows-ip', default='192.168.77.1')
    parser.add_argument('--board', default='axu3egb')
    parser.add_argument('--ssh-hop', default='vitis-vm')
    run(parser.parse_args())
