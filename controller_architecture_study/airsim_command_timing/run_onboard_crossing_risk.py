"""Compare re-ranking and reservation when the two aircraft trade risk ranks."""

import argparse
from pathlib import Path

from .run_onboard_deadline_route import DEADLINE_MS, PORT, gate3, run_case, scenario
from .run_onboard_reserved_recovery import CALLS, HEALTHY_TOLERANCE_M, state_match
from .run_onboard_stage_recovery import RESERVE_MS
from .run_onboard_value import save
from .run_ps_onboard_pair import BOARD_PACKAGE


AMPLITUDES = {'Drone1': .575, 'Drone2': -.58}
GATES = (2.75, 5.90, 9.38)
THRESHOLD = GATES[-1] - 1.75
SCREEN = ((.18, .34), (.16, .35), (.18, .42), (.16, .47), (.20, .51))
MIN_GAP_M = .012


def adopted_order(report):
    return [row['vehicle'] for row in sorted(report['adopted'].values(), key=lambda row: row['window'])]


def timing_ok(report):
    return (report['background_calls'] == 2*CALLS and len(report['adopted']) == 2
            and {row['window'] for row in report['adopted'].values() if row['on_time']} == {0, 1})


def staged_timing_ok(report):
    first, second = report['windows'][:2]
    step = report['velocity_step']
    early, late = step['Drone2'], step['Drone1']
    return (step['start_ns'] <= first['start_ns']
            and max(s['sample_ns'] for s in first['states'].values()) < late['begin_ns']
            <= late['end_ns'] <= second['start_ns']
            and all(abs(x['requested_delta_mps']-x['observed_delta_mps']) < .05 for x in (early, late)))


def stage_audit(report, healthy):
    first, second = report['windows']
    a, b = (first['risk'][v] for v in AMPLITUDES)
    c, d = (second['risk'][v] for v in AMPLITUDES)
    return dict(first_gap_m=b-a, second_gap_m=c-d, still_eligible=d > healthy[1]['Drone2'],
                stage_ok=staged_timing_ok(report), timing_ok=timing_ok(report), adoption=adopted_order(report),
                gate3_m=gate3(report), contacts=report['contacts'], pass_gates=report['pass_gates'])


def run(args):
    with scenario(PORT, args.output, amplitudes=AMPLITUDES, distances=GATES) as (client, gates):
        save(args.output/'design.json', dict(amplitudes=AMPLITUDES, gates_m=GATES, trigger_m=THRESHOLD,
             screen_steps_mps=SCREEN, minimum_rank_gap_m=MIN_GAP_M, calls_per_window=CALLS,
             deadline_ms=DEADLINE_MS, reserve_ms=RESERVE_MS, screening_rule='first valid rank reversal'))

        def fly(name, policy, *, steps=(0., 0.), healthy=()):
            return run_case(args, client, gates, name, policy=policy, calls=CALLS, reserve=RESERVE_MS,
                            recovery_schedule=dict(windows=2, healthy_error_m=healthy),
                            staged=(AMPLITUDES, GATES, THRESHOLD, steps))

        warm = [fly(f'healthy_{i+1}', 'observe') for i in range(2)]
        healthy = [{v: max(r['windows'][w]['risk'][v] for r in warm)+HEALTHY_TOLERANCE_M for v in AMPLITUDES}
                   for w in range(2)]
        screening, selected = {}, None
        for i, steps in enumerate(SCREEN, 1):
            name = f'screen_{i}'
            report = stage_audit(fly(name, 'reserve', steps=steps, healthy=healthy), healthy)
            screening[name] = report
            save(args.output/'screen.json', screening)
            if (min(report['first_gap_m'], report['second_gap_m']) >= MIN_GAP_M and report['still_eligible']
                    and report['stage_ok'] and report['timing_ok'] and report['adoption'] == ['Drone2', 'Drone2']):
                selected = steps
                break
        save(args.output/'plan.json', dict(steps_mps=selected, healthy_error_m=healthy, repetitions=4))
        if selected is None:
            print('No candidate met the frozen rank-reversal screen.', flush=True)
            return

        pairs = {}
        for rep in range(1, 5):
            order = ('risk', 'reserve') if rep % 2 else ('reserve', 'risk')
            reports = {p: fly(f'pair_{rep}_{p}', p, steps=selected, healthy=healthy) for p in order}
            arms = {p: stage_audit(r, healthy) for p, r in reports.items()}
            matched = state_match(reports['risk'], reports['reserve'])
            exposure_delta = abs(reports['risk']['velocity_step']['Drone1']['x_m'] -
                                 reports['reserve']['velocity_step']['Drone1']['x_m'])
            comparable = (matched and exposure_delta <= .022
                          and all(r['stage_ok'] and r['timing_ok'] for r in arms.values()))
            divergent = arms['risk']['adoption'] == ['Drone2', 'Drone1'] and arms['reserve']['adoption'] == ['Drone2', 'Drone2']
            pairs[rep] = dict(comparable=comparable, state_matched=matched, late_step_position_delta_m=exposure_delta,
                              divergent=divergent, arms=arms)
            save(args.output/'pairs.json', pairs)
            print('pair', rep, 'comparable', comparable, 'divergent', divergent,
                  'pass', {p: r['pass_gates'] for p, r in arms.items()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--binary', required=True)
    parser.add_argument('--package', default=BOARD_PACKAGE)
    parser.add_argument('--windows-ip', default='192.168.77.1')
    parser.add_argument('--board', default='axu3egb')
    parser.add_argument('--ssh-hop', default='vitis-vm')
    run(parser.parse_args())
