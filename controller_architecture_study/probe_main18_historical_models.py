"""Compare model recovery from historical turn states at matched command times."""
import argparse
from math import ceil
from pathlib import Path

import numpy as np

from .decision_dataset import atomic_json, sha256
from .dynamic_probe_common import read_json
from .framework_runtime_world import FrameworkWorld
from .probe_main18_local_delay import MATCH_TOLERANCE, state_difference
from .probe_main18_matched_recovery import BUNDLE, DURATION as LOCAL_DURATION, MODELS, WIDTH, airsim_trial, load_models
from .replay_ip37_held_gap import AIRSIM_CASES, prepare
from .safety_comparison_core import BODY_RADIUS, CLEARANCE
from .state_mode_calibration import moderate_case
from .teacher_attitude_probe import DiagnosticTeacher


REFERENCE = MODELS[0]
CONTROLLERS = ('main18_plain_mlp_h2_w128_seed0', REFERENCE, 'main18_plain_mlp_h9_w512_seed0', 'mpc35')
PERIOD, DELAY, DURATION, TIMING_TOLERANCE = .050, .020, 1.20, .0061
GATE, REPEATS = 4, 3
SOURCES = (*AIRSIM_CASES[:3], ('T50_mid', AIRSIM_CASES[0][1], 93))


def historical_case(source, cycle):
    info, window, _, _ = prepare(source, cycle, GATE)
    phase, initial = info['trace_phase_s'], info['initial']
    ref = window.window(phase, horizon=1)[0]
    scenario = dict(relative_initial=dict(
        position_error=(np.asarray(initial['position'])-ref.position).tolist(),
        velocity_error=(np.asarray(initial['velocity'])-ref.velocity).tolist(),
        quaternion=initial['quaternion'], acceleration=initial['acceleration'],
        angular_velocity=initial['angular_velocity'], held_control=info['held_control']))
    return info, read_json(source/'case_frozen.json')['layouts']['Drone1'], scenario


def cases(results, extended):
    sources = (SOURCES[0], SOURCES[2]) if extended else SOURCES
    for label, source, cycle in sources:
        info, layout, scenario = historical_case(results/source, cycle)
        gates = (GATE, GATE+1) if extended else (GATE,)
        end = next(g['reference_time_s'] for g in layout['gates'] if g['index'] == gates[-1])
        duration = ceil((end-info['trace_phase_s']+.35)/.10)*.10 if extended else DURATION
        yield label, info, layout, scenario, gates, duration
    if not extended:
        return

    archive = results/'main18_full_routes_20261109_201022'
    plan_path = archive/'plan.json'
    layout = read_json(plan_path)['layouts']['turns']
    gate_times = {g['index']: g['reference_time_s'] for g in layout['gates']}
    trial = archive/'turns'/REFERENCE
    trace_path, commands_path = trial/'trace.json', trial/'commands.json'
    trace, commands = read_json(trace_path), read_json(commands_path)
    initial = min(trace, key=lambda r: abs(r['t']-(gate_times[4]-.95)))
    phase = initial['t']
    held = next(row['control'] for row in reversed(commands) if row['apply_s'] < phase)
    scenario = dict(relative_initial=dict(
        position_error=np.subtract(initial['position'], initial['reference_position']).tolist(),
        velocity_error=np.subtract(initial['velocity'], initial['reference_velocity']).tolist(),
        quaternion=initial['quaternion'], acceleration=initial['acceleration'],
        angular_velocity=initial['angular_velocity'], held_control=held))
    info = dict(trace_phase_s=phase, width_m=2.4, height_m=4.,
                sources={p.name: sha256(p) for p in (plan_path, trace_path, commands_path)})
    yield 'independent_reverse_turn', info, layout, scenario, (4, 5), ceil((gate_times[5]-phase+.35)/.1)*.1

    route, layout = moderate_case()
    info = dict(trace_phase_s=route['reference_phase_s'], width_m=WIDTH, height_m=4., source=route)
    for speed in (1.20, 1.35):
        scenario = dict(critical_index=0, position_y_m=-.05, velocity_y_mps=-speed)
        yield f'controlled_vy{speed:g}', info, layout, scenario, (3, 4), LOCAL_DURATION


def compare(reference, candidate, ref_commands, commands):
    delta = state_difference(reference['intervention_start'], candidate['intervention_start'])
    common = list(zip(ref_commands, commands))
    error = max((abs(a[k]-b[k]) for a, b in common for k in ('sample_s', 'apply_s')), default=None)
    matched = (delta is not None and all(delta[k] <= v for k, v in MATCH_TOLERANCE.items())
               and bool(common) and error <= TIMING_TOLERANCE
               and all(trial['schedule_within_tolerance'] and not trial['dropped_deadline_releases']
                       for trial in (reference, candidate)))
    return dict(matched=matched, initial_difference=delta, timing_max_difference_s=error,
                common_commands=len(common))


def run(results, output, extended=False):
    controllers = (*CONTROLLERS[:-1], MODELS[1], CONTROLLERS[-1]) if extended else CONTROLLERS
    predict, identity = load_models(BUNDLE, controllers[:-1])
    teacher = DiagnosticTeacher()
    plan = list(cases(results, extended))
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output/'setup.json', dict(models=identity, teacher=teacher.signature(),
        period_s=PERIOD, delay_s=DELAY, repeats=REPEATS, extended=extended,
        cases=[dict(name=name, gates=gates, duration_s=duration) for name, _, _, _, gates, duration in plan],
        match_tolerance=MATCH_TOLERANCE, pair_timing_tolerance_s=TIMING_TOLERANCE,
        source_sha256=sha256(Path(__file__)), stepped_airsim=True, board_inference=False))
    rows = []
    for idx, (label, info, layout, scenario, gates, duration) in enumerate(plan):
        atomic_json(output/label/'historical_setup.json', dict(info=info, scenario=scenario))
        world = FrameworkWorld(layout, info['trace_phase_s'], 1, info['width_m'], task_duration=duration)
        try:
            if not set(gates).issubset(world.expected['Drone1']):
                raise ValueError(f'{label}: target gate missing from reference window')
            for repeat in range(REPEATS):
                offset = (idx+repeat) % len(controllers)
                order = controllers[offset:] + controllers[:offset]
                trials = {}
                for name in order:
                    directory = output/label/f'rep{repeat+1}'/name
                    teacher.reset()
                    callback = (lambda raw, refs: teacher.query(raw, refs)[0]) if name == 'mpc35' else None
                    trial, commands, trace = airsim_trial(world, predict, name, scenario, directory, duration,
                        release_delays={0: DELAY}, period=PERIOD, state_controller=callback)
                    physical = trial['physical']
                    margins = {}
                    for gate in gates:
                        crossing = next((x for x in physical['crossings'] if x['gate'] == gate), None)
                        margins[gate] = (min(info['width_m']/2-abs(crossing['lateral']),
                                             info['height_m']/2-abs(crossing['vertical']))-BODY_RADIUS-CLEARANCE
                                         if crossing else None)
                    margin = margins[gates[0]]
                    for key, data in (('metrics', trial), ('commands', commands), ('trace', trace)):
                        atomic_json(directory/f'{key}.json', data)
                    trials[name] = (trial, commands, dict(controller=name, margin_m=margin, gate_margins_m=margins,
                        contact=(physical['failure'] or {}).get('kind') == 'collision',
                        failure=physical['failure'], target_pass=all(m is not None and m >= 0 for m in margins.values())
                        and physical['failure'] is None and physical['window_covered'],
                        tracking_recovered=trial['tracking_recovered'], commands=len(commands)))
                    print(f'{label} rep{repeat+1} {name} margin={margin} failure={physical["failure"]}', flush=True)
                ref, ref_commands, _ = trials[REFERENCE]
                rows.append(dict(case=label, repeat=repeat+1, order=order,
                    controllers=[dict(**row, **compare(ref, trial, ref_commands, commands))
                                 for trial, commands, row in trials.values()]))
                atomic_json(output/'summary.json', dict(complete=len(rows)==len(plan)*REPEATS, pairs=rows))
        finally:
            atomic_json(output/label/'cleanup.json', world.close())
        if read_json(output/label/'cleanup.json')['status'] != 'CLEANUP_COMPLETE':
            raise RuntimeError(f'{label}: see cleanup.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--extended', action='store_true')
    args = parser.parse_args()
    run(args.results, args.output, args.extended)
