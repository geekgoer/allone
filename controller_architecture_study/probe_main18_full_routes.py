"""AirSim four-model screen on three complete, frozen reference routes."""
import argparse
import csv
from math import ceil
from pathlib import Path

import numpy as np

from environment import Gate
from .decision_dataset import atomic_json, sha256
from .fast_convex_preprocess import FastReferenceWindow
from .fly_decision_dnn import StoredRoutePlanner
from .ip37_safety_cases import add_turn_gates, audit, evaluate, time_gates
from .probe_airsim_three import load_layout
from .probe_main18_matched_recovery import (
    BUNDLE, DELAY, MODELS as MATCHED_MODELS, PERIOD, TIMING_TOLERANCE,
    airsim_trial, compare_timing, geometry, load_models,
)
from .safety_comparison_core import BODY_RADIUS, CLEARANCE
from .safety_comparison_flight import reference_crossings


MODELS = ('main18_plain_mlp_h2_w128_seed0', MATCHED_MODELS[0],
          'main18_plain_mlp_h6_w1024_seed0', 'main18_plain_mlp_h9_w512_seed0')
WIDTH, HEIGHT = 2.4, 4.
ROUTES = (('smooth', 50, 4), ('turns', 40, 2), ('dense_turns', 40, 2))


def densify_turns(layout):
    gates = sorted(layout['gates'], key=lambda g: g['reference_time_s'])
    times = [g['reference_time_s'] for g in gates]
    additions = set()
    for i, g in enumerate(gates[1:-1], 1):
        if g['kind'] == 'turn':
            left, center, right = times[i-1:i+2]
            a, b = (left, center) if center-left >= right-center else (center, right)
            additions.add((a+b)/2)
    for t in sorted(additions):
        segment = next(s for s in layout['segments'] if s['start_time'] <= t <= s['start_time']+s['duration'])
        local = t-segment['start_time']
        p, v = evaluate(segment, local), evaluate(segment, local, 1)
        yaw = np.arctan2(v[1], v[0])
        layout['gates'].append(dict(center=p.tolist(), orientation=[float(np.cos(yaw/2)), 0., 0.,
            float(np.sin(yaw/2))], reference_time_s=t, kind='dense'))


def make_plan():
    layouts, entries = {}, []
    for i, (name, environment, route) in enumerate(ROUTES):
        layout, digest = load_layout(environment, route)
        time_gates(layout)
        if name != 'smooth':
            add_turn_gates(layout)
        if name == 'dense_turns':
            densify_turns(layout)
        layout['gates'].sort(key=lambda g: g['reference_time_s'])
        for index, gate in enumerate(layout['gates'], 1):
            gate['index'] = index
        layouts[name] = layout
        duration = ceil((layout['duration']+.5)/PERIOD)*PERIOD
        planner = StoredRoutePlanner(layout)
        window = FastReferenceWindow(planner.trajectory, planner.config)
        _, v, a, _ = window._evaluate(np.linspace(0., duration, ceil(duration/.01)+1))
        gates = [Gate(g['index'], g['center'], g['orientation']) for g in layout['gates']]
        expected = [g.index for g in gates]
        # Also audit the exact point grid used when FrameworkWorld spawns the gates.
        spawn_points = window._evaluate(np.arange(0., duration+.001, .02))[0]
        if reference_crossings(spawn_points, gates) != expected:
            raise ValueError(f'{name}: reference does not cross every planned gate')
        motion = np.linalg.norm(v[:, :2], axis=1)
        moving = motion > .5
        lateral_accel = np.cross(v[moving, :2], a[moving, :2])/motion[moving]
        heading = np.unwrap(np.arctan2(v[moving, 1], v[moving, 0]))
        centers = np.asarray([g.center for g in gates])
        gaps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
        gate_times = np.array([g['reference_time_s'] for g in layout['gates']])
        entries.append(dict(id=name, source_environment=environment, source_route=route, source_sha256=digest,
            duration_s=duration, reference_duration_s=layout['duration'], gate_count=len(gates),
            expected_gate_ids=expected, added_gate_ids=[g['index'] for g in layout['gates'] if g['kind'] != 'endpoint'],
            gate_times_s=gate_times.tolist(), minimum_gate_gap_m=float(gaps.min()),
            mean_gate_gap_m=float(gaps.mean()), minimum_gate_gap_s=float(np.diff(gate_times).min()),
            mean_gate_gap_s=float(np.diff(gate_times).mean()),
            peak_speed_mps=float(np.linalg.norm(v, axis=1).max()),
            moving_heading_span_deg=float(np.rad2deg(np.ptp(heading))),
            lateral_accel_range_mps2=[float(lateral_accel.min()), float(lateral_accel.max())],
            model_order=MODELS[i:]+MODELS[:i]))
    report = audit(dict(layouts=layouts, net_width_m=WIDTH, gate_center_height_m=HEIGHT+.15,
                        roles={name: name for name in layouts}))
    if not report['passed'] or layouts['turns']['segments'] != layouts['dense_turns']['segments']:
        raise ValueError('reference audit or identical compact-route dynamics check failed')
    return dict(routes=entries, layouts=layouts, geometry_audit=report, period_s=PERIOD, delay_s=DELAY,
                timing_tolerance_s=TIMING_TOLERANCE, net_width_m=WIDTH, net_height_m=HEIGHT,
                reference_phase_s=0., nominal_initial_state=True, expected_trials=len(entries)*len(MODELS),
                runner_sha256=sha256(Path(__file__)),
                step_probe_sha256=sha256(Path(__file__).with_name('probe_main18_matched_recovery.py')))


def window_metrics(trace, end):
    t, error = [np.array([r[key] for r in trace]) for key in ('t', 'position_error_m')]
    if not 0. < end <= t[-1]:
        return None
    earlier = t < end
    values = np.r_[error[earlier], np.interp(end, t, error)]
    dt = np.diff(np.r_[t[earlier], end])
    a, b = values[:-1], values[1:]
    return dict(rmse_m=float(np.sqrt(dt @ (a*a+a*b+b*b)/(3*end))), peak_m=float(values.max()))


def summarize(route, name, trial, timing, trace, analyzer, shared_end):
    physical = trial['physical']
    failure = physical['failure'] or {}
    crossings = physical['crossings']
    valid = sorted({gate['gate'] for gate in crossings if gate['valid']})
    clear = BODY_RADIUS+CLEARANCE
    margins = [min(WIDTH/2-clear-abs(g['lateral']), HEIGHT/2-clear-abs(g['vertical'])) for g in crossings]
    added = [m for g, m in zip(crossings, margins) if g['gate'] in route['added_gate_ids']]
    duration = route['duration_s']
    positions = np.array([row['position'] for row in trace if row['t'] <= duration])
    signed = np.einsum('ni,bsi->nbs', positions, analyzer.normals)-analyzer.offsets
    sampled_margin = signed.max(axis=-1).min()
    whole = window_metrics(trace, duration) or {}
    prefix = window_metrics(trace, shared_end) or {}
    return dict(route=route['id'], model=name, collision=failure.get('kind') == 'collision',
                failure_kind=failure.get('kind'), failure_time_s=failure.get('t'),
                expected_gates=route['gate_count'], valid_gate_count=len(valid),
                all_gates_in_order=[g['gate'] for g in crossings if g['valid']] == route['expected_gate_ids'],
                physical_passed=physical['physical_task_passed'], recovery_passed=physical['recovery_passed'],
                covered=physical['window_covered'], trace_end_s=trace[-1]['t'],
                terminal_step_s=trial['terminal_step_s'], commands=timing['command_count'],
                expected_commands=round(duration/PERIOD), timing_prefix_matched=timing['matched'],
                reference_pair_common_commands=timing['common_commands'], timing_drift_s=timing['drift_s'],
                minimum_crossing_margin_m=min(margins, default=None),
                minimum_added_gate_margin_m=min(added, default=None),
                minimum_sampled_frame_margin_m=float(sampled_margin),
                fixed_window_rmse_m=whole.get('rmse_m'), fixed_window_peak_error_m=whole.get('peak_m'),
                common_window_s=shared_end, common_window_rmse_m=prefix.get('rmse_m'),
                common_window_peak_error_m=prefix.get('peak_m'))


def run(output, bundle, plan_only):
    from .framework_runtime_world import FrameworkWorld

    plan = make_plan()
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output/'plan.json', plan)
    for route in plan['routes']:
        print(f"PLAN {route['id']} gates={route['gate_count']} horizon={route['duration_s']:.2f}s "
              f"mean_gap={route['mean_gate_gap_m']:.2f}m", flush=True)
    if plan_only:
        return
    predict, identity = load_models(bundle, MODELS)
    atomic_json(output/'model_identity.json', identity)
    rows = []
    for route in plan['routes']:
        name, duration = route['id'], route['duration_s']
        directory = output/name
        world = FrameworkWorld(plan['layouts'][name], 0., 1, WIDTH, task_duration=duration)
        trials, commands, traces = {}, {}, {}
        try:
            if world.expected['Drone1'] != route['expected_gate_ids']:
                raise RuntimeError(f'{name}: spawned gates differ from the frozen plan')
            analyzer = geometry(world.gates['Drone1'], WIDTH, HEIGHT)
            for model in route['model_order']:
                trial, timeline, trace = airsim_trial(world, predict, model, dict(id=name), directory/model,
                                                      duration=duration)
                for key, value in (('metrics', trial), ('commands', timeline), ('trace', trace)):
                    atomic_json(directory/model/f'{key}.json', value)
                trials[model], commands[model], traces[model] = trial, timeline, trace
                physical = trial['physical']
                valid = len({gate['gate'] for gate in physical['crossings'] if gate['valid']})
                print(f"{name} {model} gates={valid}/{route['gate_count']} covered={physical['window_covered']} "
                      f"recovered={physical['recovery_passed']} failure={physical['failure']}", flush=True)
        finally:
            cleanup = world.close()
            atomic_json(directory/'cleanup_report.json', cleanup)
        if cleanup['status'] != 'CLEANUP_COMPLETE':
            raise RuntimeError(f'{name}: check {directory}/cleanup_report.json before restarting Play')
        matches = compare_timing(commands, trials)
        shared_end = min(duration, *(trace[-1]['t'] for trace in traces.values()))
        rows.extend(summarize(route, model, trials[model], matches[model], traces[model], analyzer, shared_end)
                    for model in MODELS)
        atomic_json(output/'summary.json', dict(complete=len(rows) == plan['expected_trials'], trials=rows))
        with (output/'trials.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f'Summary {output / "summary.json"}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, default=BUNDLE)
    parser.add_argument('--plan-only', action='store_true', help='audit the reference without opening AirSim')
    args = parser.parse_args()
    run(args.output, args.bundle, args.plan_only)
