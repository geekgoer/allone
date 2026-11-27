"""Challenge a protected recovery slot with ordinary DPU job-boundary priority."""

import argparse
from pathlib import Path
from statistics import median

from .run_onboard_deadline_route import AMPLITUDES, DEADLINE_MS, GATES, PORT, STEP_MPS, gate3, run_case, scenario
from .run_onboard_reserved_recovery import state_match
from .run_onboard_stage_recovery import RESERVE_MS, TARGET
from .run_onboard_value import save
from .run_ps_onboard_pair import BOARD_PACKAGE


THRESHOLD_M = .589773940381774  # Frozen from the earlier undisturbed calibration.
POLICIES = ('boundary', 'fixed_reserve', 'on_demand')
CASES = (
    ('critical_high', 19, True, .26),
    ('quiet_low', 14, False, .26),
    ('critical_mid', 17, True, .26),
    ('quiet_high', 19, False, .26),
    ('critical_low', 14, True, .26),
    ('quiet_mid', 17, False, .26),
    ('shifted_high', 19, True, .23),
)


def trial_stats(report, count):
    phase = report['windows'][1]
    deadline = phase['background_deadline_us']
    response = [(b-deadline)/1000+DEADLINE_MS for _, b in phase['background']]
    assert len(response) == count and response[0] > 0.
    foreground = {row['window']: row for row in report['adopted'].values()}
    prefix = count-phase['deferred_background_calls']
    request = report['requests'][1]
    if 'dpu_interval_us' in request:
        start, end = request['dpu_interval_us']
        assert not prefix or phase['background'][prefix-1][1] <= start
        assert prefix == count or end <= phase['background'][prefix][0]
    followup = foreground.get(1)
    return dict(critical=phase['followup_required'], team_pass=report['pass_gates'], contacts=report['contacts'],
                first_on_time=0 in foreground and foreground[0]['on_time'],
                followup_on_time=bool(followup and followup['on_time']),
                followup_ms=followup['window_ms'] if followup else None,
                update_gap_ms=followup['since_start_ms']-foreground[0]['since_start_ms']
                if followup and 0 in foreground else None,
                second_status=request['status'], gate3_m=gate3(report), prefix=prefix,
                bg_response_ms=response, bg_late=sum(r > DEADLINE_MS for r in response))


def audit_group(reports, count, disturbed):
    rows = {p: trial_stats(r, count) for p, r in reports.items()}
    comparable = all(state_match(reports['boundary'], r) for r in reports.values())
    if disturbed:
        steps = [r['velocity_step'] for r in reports.values()]
        comparable &= (max(s['x_m'] for s in steps)-min(s['x_m'] for s in steps) <= .022 and
                       all(abs(s['requested_delta_mps']-s['observed_delta_mps']) < .05 for s in steps))
    comparable &= all(r['first_on_time'] for r in rows.values())
    comparable &= len({r['critical'] for r in rows.values()}) == 1
    return dict(comparable=comparable, second_window_background=count, pulse=disturbed, arms=rows)


def summarize(groups):
    valid = [c for c in groups.values() if c['comparable']]
    summary = {}
    for p in POLICIES:
        rows = [c['arms'][p] for c in valid]
        required = [r for r in rows if r['critical']]
        unnecessary = [r for r in rows if not r['critical']]
        summary[p] = dict(cases=len(rows), team_passes=sum(r['team_pass'] for r in rows),
            required_followups={'timely':sum(r['followup_on_time'] for r in required), 'total':len(required)},
            optional_followups={'timely':sum(r['followup_on_time'] for r in unnecessary), 'total':len(unnecessary)},
            background_late=sum(r['bg_late'] for r in rows),
            background_total=sum(len(r['bg_response_ms']) for r in rows),
            median_background_batch_end_ms=median(r['bg_response_ms'][-1] for r in rows) if rows else None,
            max_background_batch_end_ms=max((r['bg_response_ms'][-1] for r in rows), default=None))
    return summary


def run(args):
    with scenario(PORT, args.output, amplitudes=AMPLITUDES, distances=GATES) as (client, gates):
        save(args.output/'design.json', dict(cases=CASES, policies=POLICIES, model=11,
             trigger_m=THRESHOLD_M, pulse_mps=STEP_MPS, first_background_calls=17,
             windows_ms=(0, 75), deadline_ms=DEADLINE_MS, reserve_ms=RESERVE_MS))
        groups = {}
        for i, (name, count, disturbed, offset) in enumerate(CASES):
            shift = i % len(POLICIES)
            order = POLICIES[shift:]+POLICIES[:shift]
            reports = {p: run_case(args, client, gates, f'{name}_{p}', policy='fixed', calls=17, reserve=RESERVE_MS,
                target=TARGET if disturbed else None, pulse_offset_m=offset,
                recovery_schedule=dict(windows=2, healthy_error_m=[], fixed_recipients=(TARGET, TARGET),
                                       second_background_calls=count, capacity_policy=p, capacity_trigger_m=THRESHOLD_M))
                       for p in order}
            groups[name] = audit_group(reports, count, disturbed)
            save(args.output/'groups.json', groups)
            save(args.output/'summary.json', summarize(groups))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--binary', required=True)
    parser.add_argument('--package', default=BOARD_PACKAGE)
    parser.add_argument('--windows-ip', default='192.168.77.1')
    parser.add_argument('--board', default='axu3egb')
    parser.add_argument('--ssh-hop', default='vitis-vm')
    run(parser.parse_args())
