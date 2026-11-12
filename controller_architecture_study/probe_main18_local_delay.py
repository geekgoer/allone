"""Paired local command-delay probe on natural AirSim reference routes."""
import argparse
import csv
from math import ceil
from pathlib import Path

import numpy as np

from .decision_dataset import atomic_json, sha256
from .probe_main18_full_routes import HEIGHT, WIDTH, make_plan
from .probe_main18_matched_recovery import BUNDLE, DELAY, MODELS, PERIOD, airsim_trial, geometry, load_models
from .safety_comparison_core import BODY_RADIUS, CLEARANCE


MODEL = MODELS[0]
TARGET_GATE = 4
ORDERS = ((20, 30, 40), (30, 40, 20), (40, 20, 30))
MATCH_TOLERANCE = dict(time_s=.0061, position_m=.04, velocity_mps=.12,
                       angle_deg=2., held_thrust_mps2=.25, held_rate_radps=.12)


def protocol():
    source = make_plan()
    windows = []
    layouts = {name: source['layouts'][name] for name in ('smooth', 'turns')}
    for route in source['routes'][:2]:
        gate_s = route['gate_times_s'][TARGET_GATE-1]
        first = round((gate_s-.30)/PERIOD)
        end = ceil((gate_s+.45)/PERIOD)*PERIOD
        windows.append(dict(route=route['id'], source_sha256=route['source_sha256'], target_gate=TARGET_GATE,
                            gate_reference_s=gate_s, release_indices=[first, first+1],
                            window_start_s=first*PERIOD, window_end_s=end))
    return dict(model=MODEL, period_s=PERIOD, normal_delay_s=DELAY, net_width_m=WIDTH,
                orders=ORDERS, windows=windows, layouts=layouts, match_tolerance=MATCH_TOLERANCE,
                source_sha256=sha256(Path(__file__)),
                step_probe_sha256=sha256(Path(__file__).with_name('probe_main18_matched_recovery.py')))


def state_difference(a, b):
    if a is None or b is None:
        return None
    qa, qb = np.asarray(a['quaternion']), np.asarray(b['quaternion'])
    cos = np.clip(abs(qa @ qb)/(np.linalg.norm(qa)*np.linalg.norm(qb)), 0., 1.)
    return dict(time_s=abs(a['sample_s']-b['sample_s']),
                position_m=float(np.linalg.norm(np.subtract(a['position'], b['position']))),
                velocity_mps=float(np.linalg.norm(np.subtract(a['velocity'], b['velocity']))),
                angle_deg=float(np.rad2deg(2*np.arccos(cos))),
                held_thrust_mps2=abs(a['held_control'][0]-b['held_control'][0]),
                held_rate_radps=float(np.max(np.abs(np.subtract(a['held_control'][1:], b['held_control'][1:])))))


def local_metrics(trace, start, end, analyzer):
    times = np.asarray([r['t'] for r in trace])
    if times[0] > start or times[-1] < end:
        return dict(min_frame_margin_m=None, peak_lateral_error_m=None)
    grid = np.linspace(start, end, ceil((end-start)/.01)+1)

    def at(key):
        values = np.asarray([r[key] for r in trace])
        return np.stack([np.interp(grid, times, values[:, axis]) for axis in range(3)], axis=1)

    position, reference, velocity = at('position'), at('reference_position'), at('reference_velocity')
    normal = np.c_[-velocity[:, 1], velocity[:, 0]]
    signed = np.einsum('ni,bsi->nbs', position, analyzer.normals)-analyzer.offsets
    lateral = np.sum((position-reference)[:, :2]*normal, axis=1)/np.linalg.norm(normal, axis=1)
    return dict(min_frame_margin_m=float(signed.max(axis=-1).min()),
                peak_lateral_error_m=float(np.abs(lateral).max()))


def summarize(window, report, commands, trace, analyzer):
    selected = window['release_indices']
    by_release = {round(row['scheduled_sample_s']/PERIOD): row for row in commands}
    age = [1000*(by_release[k]['apply_s']-by_release[k]['sample_s']) for k in selected if k in by_release]
    response = [1000*(by_release[k]['apply_s']-by_release[k]['scheduled_sample_s'])
                for k in selected if k in by_release]
    around = [by_release[k]['apply_s'] for k in range(selected[0]-1, selected[-1]+2) if k in by_release]
    physical = report['physical']
    crossing = next((g for g in physical['crossings'] if g['gate'] == TARGET_GATE), None)
    clearance = (min(WIDTH/2-abs(crossing['lateral']), HEIGHT/2-abs(crossing['vertical']))-BODY_RADIUS-CLEARANCE
                 if crossing else None)
    failure = physical['failure']
    delivered = (len(age) == len(selected) and not report['dropped_deadline_releases']
                 and report['schedule_within_tolerance'])
    return dict(measured_age_ms=age, measured_response_ms=response,
                mean_age_ms=float(np.mean(age)) if age else None,
                max_command_gap_ms=float(max(np.diff(around))*1000) if len(around) >= 2 else None,
                treatment_delivered=delivered, dropped_deadline_releases=report['dropped_deadline_releases'],
                window_covered=physical['window_covered'], target_gate_valid=crossing['valid'] if crossing else None,
                crossing_margin_m=clearance, collision=bool(failure and failure['kind'] == 'collision'),
                failure=failure, **local_metrics(trace, window['window_start_s'], window['window_end_s'], analyzer))


def pair_rows(rows):
    baseline = rows[20]
    result = []
    for delay in (30, 40):
        row = rows[delay]
        state = state_difference(baseline['intervention_start'], row['intervention_start'])
        matched = state is not None and all(state[k] <= v for k, v in MATCH_TOLERANCE.items())
        age_delta = (row['mean_age_ms']-baseline['mean_age_ms']
                     if row['mean_age_ms'] is not None and baseline['mean_age_ms'] is not None else None)
        delivered = row['treatment_delivered'] and baseline['treatment_delivered']
        per_command_age_delta = [x-y for x, y in zip(row['measured_age_ms'], baseline['measured_age_ms'])]
        contrast = delivered and matched and min(per_command_age_delta) >= (delay-20)-6.1
        values = {key: (row[key]-baseline[key] if contrast and row[key] is not None
                        and baseline[key] is not None else None)
                  for key in ('crossing_margin_m', 'min_frame_margin_m', 'peak_lateral_error_m')}
        result.append(dict(route=row['route'], repeat=row['repeat'], nominal_delay_ms=delay,
                           matched_pre_intervention=matched, applied_contrast_valid=contrast,
                           age_delta_ms=age_delta, per_command_age_delta_ms=per_command_age_delta,
                           state_difference=state, delta=values))
    return result


def run(output, bundle, plan_only):
    from .framework_runtime_world import FrameworkWorld

    plan = protocol()
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output/'plan.json', plan)
    for window in plan['windows']:
        print(f"PLAN {window['route']} gate={TARGET_GATE} releases={window['release_indices']} "
              f"window={window['window_start_s']:.2f}-{window['window_end_s']:.2f}s", flush=True)
    if plan_only:
        return
    predict, identity = load_models(bundle, (MODEL,))
    atomic_json(output/'model_identity.json', identity)
    rows, pairs = [], []
    for window in plan['windows']:
        name, end = window['route'], window['window_end_s']
        world = FrameworkWorld(plan['layouts'][name], 0., 1, WIDTH, task_duration=end)
        try:
            gate = next(g for g in world.gates['Drone1'] if g.index == TARGET_GATE)
            analyzer = geometry([gate], WIDTH, HEIGHT)
            for repeat, order in enumerate(ORDERS, 1):
                group = {}
                for delay in order:
                    directory = output/name/f'r{repeat}_R{delay}'
                    delays = dict.fromkeys(window['release_indices'], delay/1000)
                    report, commands, trace = airsim_trial(world, predict, MODEL, dict(id=name),
                                                           directory, duration=end, release_delays=delays)
                    for key, value in (('metrics', report), ('commands', commands), ('trace', trace)):
                        atomic_json(directory/f'{key}.json', value)
                    row = dict(route=name, repeat=repeat, nominal_delay_ms=delay,
                               intervention_start=report['intervention_start'],
                               **summarize(window, report, commands, trace, analyzer))
                    rows.append(row)
                    group[delay] = row
                    print(f"{name} r{repeat} R{delay}: age_ms={row['measured_age_ms']} "
                          f"crossing_margin_m={row['crossing_margin_m']} collision={row['collision']}", flush=True)
                pairs.extend(pair_rows(group))
                atomic_json(output/'summary.json', dict(complete=len(rows) == 18, trials=rows, contrasts=pairs))
        finally:
            cleanup = world.close()
            atomic_json(output/name/'cleanup_report.json', cleanup)
        if cleanup['status'] != 'CLEANUP_COMPLETE':
            raise RuntimeError(f'{name}: check cleanup_report.json before restarting Play')
    flat = [dict(route=r['route'], repeat=r['repeat'], nominal_delay_ms=r['nominal_delay_ms'],
                 valid=r['applied_contrast_valid'], age_delta_ms=r['age_delta_ms'], **r['delta']) for r in pairs]
    with (output/'contrasts.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    print(f'Summary {output / "summary.json"}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, default=BUNDLE)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    run(args.output, args.bundle, args.plan_only)
