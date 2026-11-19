"""Offline held-command contrasts initialized from real IP37 flight samples."""
import argparse
from math import ceil
from pathlib import Path

import numpy as np

from airsim_interface import RawDroneState
from environment import Gate
from falcon_paper_fidelity.motion_primitive import _rotation_matrix
from gate_route import segment_gate_crossing
from .decision_dataset import atomic_json, sha256
from .dynamic_probe_common import read_json
from .fast_convex_preprocess import FastReferenceWindow, encode
from .fly_decision_dnn import StoredRoutePlanner
from .probe_main18_matched_recovery import BUNDLE, MODELS, TIMING_TOLERANCE, airsim_trial, geometry, load_models
from .teacher_attitude_probe import integrate_rotation


MODEL = MODELS[0]
CASES = (
    ('recorded_gap_gate3', 'ip37_kernel_20261109_105424', 76, 3),
    ('tight_gate4_T50', 'ip37_kernel_20261109_105424', 101, 4),
    ('tight_gate4_T80', 'ip37_kernel_20261109_112606', 65, 4),
)
LEAD_TIMES = (.15, .35, .55, .75, .95)
AIRSIM_CASES = (('T50_early', CASES[1][1], 85), ('T50_late', CASES[1][1], 101),
                ('T80_early', CASES[2][1], 55), ('T80_late', CASES[2][1], 65))
DELAY = .020
WINDOW = 1.2


def prepare(directory, cycle, gate_id):
    files = ('case_frozen.json', 'setup.json', 'requests.json', 'Drone1_trace.json', 'result.json')
    case, setup, requests, trace, result = [read_json(directory/name) for name in files]
    row = next(r for r in requests if r['vehicle'] == 'Drone1' and r['cycle'] == cycle)
    phase = row['sample_start_s']-setup['origin_s']
    state = min(trace, key=lambda r: abs(r['t']-phase))
    if abs(state['t']-phase) > .005 or row['model_id'] != 11:
        raise ValueError('recorded state or model does not match the selected request')
    layout = case['layouts']['Drone1']
    planner = StoredRoutePlanner(layout)
    window = FastReferenceWindow(planner.trajectory, planner.config)
    shift = np.asarray(trace[0]['position'])-window.window(0., horizon=1)[0].position
    item = next(g for g in layout['gates'] if g['index'] == gate_id)
    gate = Gate(gate_id, item['center'], item['orientation'])
    analyzer = geometry([gate], case['net_width_m'], case['gate_center_height_m']-.15)
    start = dict(position=(np.asarray(state['position'])-shift).tolist(),
                 **{key: state[key] for key in ('velocity', 'quaternion', 'acceleration', 'angular_velocity')})
    crossing = next(g for g in result['vehicles']['Drone1']['crossings'] if g['gate'] == gate_id)
    info = dict(run=directory.name, cycle=cycle, gate=gate_id, request_status=row['status'],
                sample_phase_s=phase, trace_phase_s=state['t'], recorded_gate_margin_m=crossing['body_margin_m'],
                period_s=row['period_ms']/1000, width_m=case['net_width_m'],
                height_m=case['gate_center_height_m']-.15, held_control=row['held_control'], initial=start,
                sources={name: sha256(directory/name) for name in files})
    return info, window, gate, analyzer


def simulate(info, window, gate, analyzer, predict, skip, step):
    initial = info['initial']
    p, v, q, acceleration, rate = (np.array(initial[key]) for key in
        ('position', 'velocity', 'quaternion', 'acceleration', 'angular_velocity'))
    held = np.asarray(info['held_control'])
    pending, crossing = None, None
    release_ticks, delay_ticks = round(info['period_s']/step), round(DELAY/step)
    path, applications = [], []
    for k in range(round(WINDOW/step)+1):
        t = k*step
        if k % release_ticks == 0 and k//release_ticks >= skip:
            references = window.window(info['trace_phase_s']+t, include_body_rates=False)
            pending = predict(MODEL, encode(RawDroneState(p, q, v, acceleration, rate), references))
        elif k % release_ticks == delay_ticks and pending is not None:
            held, pending = pending, None
            applications.append(t)
        if path and crossing is None:
            crossing = segment_gate_crossing(np.asarray(path[-1]['position']), p, gate)
        margin = float((analyzer.normals@p-analyzer.offsets).max(axis=1).min())
        path.append(dict(t=t, position=p.tolist(), sampled_frame_margin_m=margin))
        rate = held[1:]
        middle = integrate_rotation(q, rate, step/2)
        acceleration = np.array([0., 0., 9.80665])-held[0]*_rotation_matrix(middle)[:, 2]
        p, v = p+v*step+.5*acceleration*step**2, v+acceleration*step
        q = integrate_rotation(q, rate, step)
    crossing_margin = (float(min(info['width_m']/2-.525-abs(crossing.lateral_offset),
                                 info['height_m']/2-.525-abs(crossing.vertical_offset)))
                       if crossing else None)
    return dict(skip=skip, step_s=step, first_new_apply_s=applications[0], crossing_margin_m=crossing_margin,
                min_sampled_frame_margin_m=min(row['sampled_frame_margin_m'] for row in path)), path


def run(results, output):
    predict, identity = load_models(BUNDLE, (MODEL,))
    output.mkdir(parents=True, exist_ok=False)
    cases = []
    selections = list(CASES)
    for _, run_id, _, gate_id in CASES[1:]:
        directory = results/run_id
        case = read_json(directory/'case_frozen.json')
        origin = read_json(directory/'setup.json')['origin_s']
        target = next(g['reference_time_s'] for g in case['layouts']['Drone1']['gates'] if g['index'] == gate_id)
        requests = [r for r in read_json(directory/'requests.json')
                    if r['vehicle'] == 'Drone1' and r.get('held_control') is not None
                    and r['status'] == 'delivered']
        for lead in LEAD_TIMES:
            row = min(requests, key=lambda r: abs(r['sample_start_s']-origin-(target-lead)))
            selections.append((f'{run_id}_lead{round(lead*1000)}ms', run_id, row['cycle'], gate_id))
    for name, run_id, cycle, gate_id in selections:
        info, window, gate, analyzer = prepare(results/run_id, cycle, gate_id)
        variants = []
        for step in (.005, .0025):
            for skip in (0, 1, 2):
                metrics, path = simulate(info, window, gate, analyzer, predict, skip, step)
                atomic_json(output/name/f'{round(step*1e6)}us_skip{skip}.json', path)
                variants.append(metrics)
        cases.append(dict(case=name, historical=info, variants=variants))
        print(name, 'historical margin', info['recorded_gate_margin_m'],
              'modeled margins', [m['crossing_margin_m'] for m in variants if m['step_s'] == .0025], flush=True)
    atomic_json(output/'summary.json', dict(scope='ideal body-rate offline model, FP32, no AirSim or FPGA',
                model=identity, delay_s=DELAY, window_s=WINDOW, phase_screen_leads_s=LEAD_TIMES,
                source_sha256=sha256(Path(__file__)), cases=cases))


def run_airsim(results, output, *, natural_start=False):
    from .framework_runtime_world import FrameworkWorld
    from .probe_main18_local_delay import MATCH_TOLERANCE, state_difference

    predict, identity = load_models(BUNDLE, (MODEL,))
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output/'model_identity.json', identity)
    summary = []
    for label, run_id, cycle in AIRSIM_CASES:
        info, window, gate, _ = prepare(results/run_id, cycle, 4)
        directory = output/label
        atomic_json(directory/'historical_setup.json', info)
        phase, period, initial = info['trace_phase_s'], info['period_s'], info['initial']
        ref = window.window(phase, horizon=1)[0]
        case = dict(relative_initial=dict(
            position_error=(np.asarray(initial['position'])-ref.position).tolist(),
            velocity_error=(np.asarray(initial['velocity'])-ref.velocity).tolist(),
            quaternion=initial['quaternion'], acceleration=initial['acceleration'],
            angular_velocity=initial['angular_velocity'], held_control=info['held_control']))
        trigger = round(phase/period) if natural_start else 0
        start_phase = 0. if natural_start else phase
        trigger_s = trigger*period
        duration = ceil((trigger_s+WINDOW)/period-1e-9)*period
        if natural_start:
            case = {}
        layout = read_json(results/run_id/'case_frozen.json')['layouts']['Drone1']
        world = FrameworkWorld(layout, start_phase, 1, info['width_m'], task_duration=duration)
        try:
            if gate.index not in world.expected['Drone1']:
                raise RuntimeError(f'{label}: target gate missing from reference window')
            for repeat, order in enumerate(((0, 2), (2, 0)), 1):
                records = {}
                for skipped in order:
                    trial_dir = directory/f'r{repeat}_skip{skipped}'
                    trial, commands, trace = airsim_trial(
                        world, predict, MODEL, case, trial_dir, duration=duration,
                        release_delays={trigger: DELAY, trigger+1: DELAY}, period=period,
                        skip_releases=range(trigger, trigger+skipped))
                    for key, value in (('metrics', trial), ('commands', commands), ('trace', trace)):
                        atomic_json(trial_dir/f'{key}.json', value)
                    crossing = next((c for c in trial['physical']['crossings'] if c['gate'] == gate.index), None)
                    margin = (float(min(info['width_m']/2-.525-abs(crossing['lateral']),
                                        info['height_m']/2-.525-abs(crossing['vertical']))) if crossing else None)
                    first = next((c['apply_s']-trigger_s for c in commands
                                  if c['scheduled_sample_s'] >= trigger_s-1e-9), None)
                    records[skipped] = dict(crossing_margin_m=margin, first_new_apply_s=first,
                        failure=trial['physical']['failure'], window_covered=trial['physical']['window_covered'],
                        observed_skips=trial.get('intentional_skips', []),
                        deadline_drops=trial['dropped_deadline_releases'], state=trial['intervention_start'],
                        applied_as_planned=(first is not None and abs(first-(skipped*period+DELAY)) <= TIMING_TOLERANCE
                                            and trial.get('intentional_skips', []) == list(range(trigger, trigger+skipped))
                                            and not trial['dropped_deadline_releases']))
                a, b = records[0], records[2]
                difference = state_difference(a['state'], b['state'])
                matched = difference is not None and all(difference[k] <= v for k, v in MATCH_TOLERANCE.items())
                summary.append(dict(case=label, repeat=repeat, matched=matched, state_difference=difference,
                                    trigger_s=trigger_s, reference_phase_at_trigger_s=start_phase+trigger_s,
                                    applied_as_planned=a['applied_as_planned'] and b['applied_as_planned'],
                                    normal=a, two_missed=b))
                atomic_json(output/'summary.json', dict(complete=len(summary) == 2*len(AIRSIM_CASES),
                                                      natural_start=natural_start, pairs=summary))
                print(label, repeat, 'matched', matched, 'margins',
                      a['crossing_margin_m'], b['crossing_margin_m'], flush=True)
        finally:
            cleanup = world.close()
            atomic_json(directory/'cleanup_report.json', cleanup)
        if cleanup['status'] != 'CLEANUP_COMPLETE':
            raise RuntimeError(f'{label}: see cleanup_report.json')
    print(f'Summary {output / "summary.json"}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--airsim', action='store_true', help='paired confirmation from a historical state')
    mode.add_argument('--natural-airsim', action='store_true', help='fly from the nominal route start before intervention')
    args = parser.parse_args()
    if args.airsim or args.natural_airsim:
        run_airsim(args.results, args.output, natural_start=args.natural_airsim)
    else:
        run(args.results, args.output)
