"""Cross MPC attitude gain, release period and command age on one frozen route."""

import argparse
import csv
from pathlib import Path
import random

import numpy as np

from .decision_dataset import ROOT, atomic_json, sha256
from .dynamic_probe_common import describe, read_json
from .probe_main18_full_routes import window_metrics
from .probe_main18_matched_recovery import airsim_trial
from .safety_comparison_core import BODY_RADIUS, CLEARANCE
from .teacher_attitude_probe import DiagnosticTeacher, simulate


CASE = ROOT/'controller_architecture_study/results/ip37_native_prep_worker_wall_20261109_120420/split/case_frozen.json'
DURATION = 7.2
TARGET_GATE = 4
GAINS = (3.5, 7., 10.)
PERIODS = (30, 60, 120)
DELAYS = (9, 18, 24)
LONG_DELAYS = (54, 99)


def grid():
    return [(gain, period, delay) for gain in GAINS for period in PERIODS
            for delay in DELAYS + (LONG_DELAYS if period == max(PERIODS) else ())]


def airsim_metrics(world, gain, period, delay, directory):
    teacher = DiagnosticTeacher(attitude_gain=gain)
    infos = []

    def control(state, references):
        command, info = teacher.query(state, references)
        infos.append(info)
        return command

    releases = round(DURATION/period)
    result, commands, trace = airsim_trial(world, None, None, {}, directory, DURATION,
        release_delays=dict.fromkeys(range(releases), delay), period=period, state_controller=control)
    physical = result['physical']
    failure = physical['failure'] or {}
    gate = next((g for g in physical['crossings'] if g['gate'] == TARGET_GATE), None)
    clearance = BODY_RADIUS+CLEARANCE
    margin = min(world.net_width/2-clearance-abs(gate['lateral']),
                 2.-clearance-abs(gate['vertical'])) if gate else None
    release_delays_ms = [1000*(cmd['apply_s']-cmd['scheduled_sample_s']) for cmd in commands]
    ages_ms = [1000*(cmd['apply_s']-cmd['sample_s']) for cmd in commands]
    command_gaps_ms = 1000*np.diff([cmd['apply_s'] for cmd in commands])
    apply_errors_ms = [abs(r-delay*1000) for r in release_delays_ms]
    sample_errors_ms = [1000*abs(cmd['sample_s']-cmd['scheduled_sample_s']) for cmd in commands]
    dropped = result['dropped_deadline_releases']
    timing_valid = (bool(commands) and not dropped and
                    max(apply_errors_ms + sample_errors_ms) <= 6.5 and
                    abs(float(np.median(release_delays_ms))-delay*1000) <= 3.5)
    covered = physical['window_covered']
    atomic_json(directory/'details.json', dict(
        result=result, teacher=teacher.signature(), expected_releases=releases,
        planned_to_apply_ms=describe(release_delays_ms), sample_age_ms=describe(ages_ms),
        command_gap_ms=describe(command_gaps_ms), max_apply_error_ms=max(apply_errors_ms, default=None),
        max_sample_error_ms=max(sample_errors_ms, default=None),
        query_ms=describe([i['solve_ms'] for i in infos])))
    atomic_json(directory/'commands.json', commands)
    return dict(gate4_margin_m=margin, gate4_crossed=gate is not None,
                collision=failure.get('kind') == 'collision', failure_kind=failure.get('kind'),
                collision_object=failure.get('object'), failure_time_s=failure.get('t'),
                window_covered=covered, physical_pass=physical['physical_task_passed'],
                valid_gates=len({g['gate'] for g in physical['crossings'] if g['valid']}),
                expected_gates=len(physical['expected_gate_ids']),
                timing_valid=timing_valid, full_timing_valid=timing_valid and covered and len(commands) == releases,
                delivered=len(commands), dropped=len(dropped),
                actual_R_median_ms=float(np.median(release_delays_ms)) if commands else None,
                rate_clipped_fraction=float(np.mean([i['rate_clipped'] for i in infos])) if infos else None), trace


def offline_metrics(case, gain, period, delay, directory):
    directory.mkdir()
    result, trace = simulate(case, 'baseline', period=period, delay=delay, attitude_gain=gain)
    atomic_json(directory/'details.json', result)
    gate = next(g for g in result['crossings'] if g['gate'] == TARGET_GATE)
    for row in trace:
        row['position_error_m'] = row.pop('error_m')
    return dict(gate4_margin_m=gate.get('body_margin_m'), gate4_crossed=gate['crossed'],
                window_covered=True, rate_clipped_fraction=result['rate_clipped_fraction']), trace


def run(case_path, output, backend, repeats):
    case = read_json(case_path)
    case['duration_s'] = DURATION
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output/'case.json', case)
    setup = dict(backend=backend, case_source=str(case_path.resolve()), case_source_sha256=sha256(case_path),
                 source_sha256=sha256(Path(__file__)), duration_s=DURATION, gate=TARGET_GATE,
                 net_width_m=case['net_width_m'], repeats=repeats, settings=grid(),
                 feedforward='none', initial_state='route reference', disturbance=False,
                 real_time_guarantee=False, board_inference=False)
    atomic_json(output/'setup.json', setup)
    rows = []
    world = None
    try:
        if backend == 'airsim':
            from .framework_runtime_world import FrameworkWorld
            world = FrameworkWorld(case['layouts']['Drone1'], 0., 1, case['net_width_m'], task_duration=DURATION)
            if TARGET_GATE not in world.expected['Drone1']:
                raise ValueError('target gate is not in the observation window')
        for rep in range(1, repeats+1):
            settings = grid()
            random.Random(20261110+rep).shuffle(settings)
            for gain, t_ms, r_ms in settings:
                directory = output/f'rep{rep:02d}_g{gain:g}_t{t_ms}_r{r_ms}'
                period, delay = t_ms/1000, r_ms/1000
                if world is None:
                    result, trace = offline_metrics(case, gain, period, delay, directory)
                else:
                    result, trace = airsim_metrics(world, gain, period, delay, directory)
                stats = window_metrics(trace, DURATION) if result['window_covered'] else None
                row = dict(repeat=rep, gain=gain, T_ms=t_ms, R_target_ms=r_ms,
                           group='long_delay' if r_ms in LONG_DELAYS else 'factorial',
                           **result, rmse_m=stats['rmse_m'] if stats else None,
                           peak_error_m=stats['peak_m'] if stats else None)
                atomic_json(directory/'trace.json', trace)
                rows.append(row)
                atomic_json(output/'summary.json', dict(**setup, complete=len(rows)==len(grid())*repeats, rows=rows))
                print(f'{directory.name} margin={row["gate4_margin_m"]} rmse={row["rmse_m"]}'
                      f' timing_valid={row.get("timing_valid")} collision={row.get("collision")}', flush=True)
    finally:
        if rows:
            with (output/'summary.csv').open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        if world is not None:
            atomic_json(output/'cleanup_report.json', world.close())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('offline', 'airsim'), default='offline')
    parser.add_argument('--case', type=Path, default=CASE)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=1)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    run(args.case, args.output, args.backend, args.repeats)
