"""Route B live multi-UAV control: one real IP37 worker, fixed or bounded ABC."""
import argparse
from collections import Counter
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor, TimeoutError, CancelledError
from dataclasses import asdict
import json
from pathlib import Path
import re
import shutil
import threading
import time
from time import thread_time

import numpy as np

from .decision_dataset import atomic_json, sha256
from .dynamic_probe_common import describe
from .fast_convex_preprocess import FastReferenceWindow, encode
from .fly_convex_policy import load_airsim_client
from .fly_decision_dnn import StoredRoutePlanner
from .held_control_safety import GateHoldAnalyzer, HoldBounds, safe_deadline
from .ip37_joint_runtime import JointRuntime
from .ip37_shadow import make_client, predict
from .probe_airsim_three import load_layout, translate_layout
from .control_value_probe import tracking_metrics


ROUTES = ((48, 0), (49, 2), (50, 4))


def prepare_gate_scene(client, output):
    """Remove only our generated bars, then verify before resetting the trial."""
    found = client.simListSceneObjects('Falcon.*Gate.*')
    owned = re.compile(r'FalconJointGate_Drone[1-9][0-9]*_[1-9][0-9]*_(left|right|top|bottom)(_[0-9]+)?')
    foreign = [name for name in found if owned.fullmatch(name) is None]
    report = dict(found=found, foreign=foreign, destroy_results={}, remaining=found)
    atomic_json(output/'gate_preflight.json', report)
    if foreign:
        raise ValueError(f'other generated gates remain: {foreign}; stop and restart Play')
    for name in found:
        try:
            report['destroy_results'][name] = bool(client.simDestroyObject(name))
        except Exception as exc:
            report['destroy_results'][name] = f'{type(exc).__name__}: {exc}'
    # Unreal can defer actual destruction to its next frame.
    if found:
        for _ in range(10):
            report['remaining'] = client.simListSceneObjects('Falcon.*Gate.*')
            if not report['remaining']:
                break
            time.sleep(.02)
    atomic_json(output/'gate_preflight.json', report)
    if report['remaining']:
        raise ValueError(f"gate cleanup not verified: {report['remaining']}; stop and restart Play")
    if found:
        print(f'Cleared {len(found)} previous experiment gate bars.', flush=True)
    return report


def load_flight_layouts(args):
    case_path = getattr(args, 'case', None)
    if case_path is None:
        args.gate_width = args.gate_width if args.gate_width is not None else 4.
        layouts = [load_layout(*ROUTES[i % len(ROUTES)]) for i in range(len(args.vehicles))]
        sources = [dict(vehicle=v, environment=ROUTES[i % len(ROUTES)][0],
                        route=ROUTES[i % len(ROUTES)][1], sha256=layouts[i][1])
                   for i, v in enumerate(args.vehicles)]
        return layouts, sources, None
    case = json.loads(case_path.read_text(encoding='utf-8'))
    if case['version'] != 'ip37_natural_turn_gates_v1' or case['gate_center_height_m'] != 4.:
        raise ValueError('unsupported frozen flight case')
    if args.gate_width is not None and not np.isclose(args.gate_width, case['gate_center_width_m']):
        raise ValueError('gate width must match the frozen case')
    if args.duration is not None and args.duration < case['duration_s']:
        raise ValueError('a frozen case cannot be shortened')
    args.gate_width = case['gate_center_width_m']
    args.duration = args.duration or case['duration_s']
    digest = sha256(case_path)
    layouts = [(case['layouts'][v], digest) for v in args.vehicles]
    sources = [dict(vehicle=v, **case['sources'][v], case_sha256=digest) for v in args.vehicles]
    return layouts, sources, case


class Contact(RuntimeError):
    pass


class Pilot:
    """One thread owns each AirSim RPC client, from connect through cleanup."""
    def __init__(self, args, name, planner, gates, offset, runtime, calibration, origin, bounds):
        from airsim_interface import AirSimDrone, require_airsim
        from control_adapter import send_control
        self.client = require_airsim().MultirotorClient(ip=args.airsim_host, timeout_value=2.)
        self.client.ping()
        self.drone = AirSimDrone(self.client, name)
        self.name, self.args, self.runtime = name, args, runtime
        self.calibration, self.offset, self.origin = calibration, offset, origin
        self.window = FastReferenceWindow(planner.trajectory, planner.config)
        self.analyzer = GateHoldAnalyzer(gates, args.gate_width-.15, bounds,
                                         net_height=4.-.15, body_radius=.5, clearance=.025)
        self.held = np.array([calibration.hover_thrust_accel, 0., 0., 0.])
        self.actuation_s, self.version = origin, 0
        self.command_pending = False
        self.trace, self.error, self.contact = [], None, None
        self.stamp = int(self.client.simGetCollisionInfo(vehicle_name=name).time_stamp)
        raw = self.drone.get_raw_state()  # paused initial state; its clock starts on launch
        first = self.window.window(0.)[0]
        if (np.linalg.norm(raw.position-first.position) > .05 or np.linalg.norm(raw.velocity) > .2
                or abs(float(np.dot(raw.quaternion, first.orientation))) < .999):
            raise RuntimeError('paused initial state does not match route start')
        self.initial = raw
        self.teacher = None
        if getattr(args, 'reference_teacher', False):
            if getattr(args, 'teacher_variant', None) is not None:
                from .teacher_attitude_probe import DiagnosticTeacher
                self.teacher = DiagnosticTeacher(args.teacher_variant)
            else:
                from .convex_teacher import ConvexTeacher
                self.teacher = ConvexTeacher()
            self.teacher.query(raw, self.window.window(0.))  # setup while paused
        # hoverAsync is not evidence of an active physical rate/throttle hold.
        # Prime the same physical command used by the held-control analyzer.
        send_control(self.drone, self.held, calibration, duration=1.)
        self.client.ping()

    def publish(self, raw, sample_s, *, mark=None):
        refs = self.window.window(max(0., sample_s-self.origin), include_body_rates=self.teacher is not None)
        if mark is not None:
            mark('reference_ready')
        error = float(np.linalg.norm(raw.position-refs[0].position))
        if (not all(np.isfinite(v).all() for v in (raw.position, raw.velocity, raw.quaternion))
                or raw.position[2]+self.offset[2] > -.3 or error > 8. or np.linalg.norm(raw.velocity) > 15.):
            raise RuntimeError('flight guard exceeded')
        self.trace.append(dict(t=max(0., sample_s-self.origin), position=raw.position.tolist(),
                               velocity=raw.velocity.tolist(), error_m=error,
                               quaternion=raw.quaternion.tolist(), acceleration=raw.acceleration.tolist(),
                               angular_velocity=raw.angular_velocity.tolist(),
                               sim_ns=int(raw.timestamp_ns or 0)))
        if mark is not None:
            mark('flight_trace_ready')
        start = time.perf_counter()
        analysis = self.analyzer.analyze(raw, self.held)
        now = time.perf_counter()
        # During initial pause, the initial state is unchanged until launch.
        deadline = safe_deadline(sample_s, max(now, sample_s), self.actuation_s,
                                 analysis, self.args.maximum_update_ms/1000)
        if mark is not None:
            mark('safety_deadline_ready')
        self.runtime.observe(self.name, sample_s, deadline['deadline_s'], analysis, self.version)
        if mark is not None:
            mark('observation_published')
        return refs, analysis, deadline, (now-start)*1000

    def collision(self, raw=None, sampled_s=None, *, hit=None):
        c = hit if hit is not None else self.client.simGetCollisionInfo(vehicle_name=self.name)
        if c.has_collided and int(c.time_stamp) != self.stamp:
            self.contact = dict(object=c.object_name, timestamp_ns=int(c.time_stamp))
            if raw is not None:
                self.contact.update(position=raw.position.tolist(), velocity=raw.velocity.tolist(),
                                    observation_t_s=sampled_s-self.origin,
                                    state_timestamp_ns=int(raw.timestamp_ns or 0))
            raise Contact(f'collision with {c.object_name}')

    def run_job(self, row, native_sample=None):
        from control_adapter import send_control, thrust_to_throttle, throttle_to_thrust
        trace_cycles = getattr(self.args, 'trace_thread_cycles', False)
        split_prep = native_sample is not None and getattr(self.args, 'split_native_prep', False)
        if trace_cycles:
            from .thread_cycle_trace import cycles
            row['native_thread_id'] = threading.get_native_id()
        def mark(stage):
            if trace_cycles:
                row.setdefault('thread_marks_s', {})[stage] = time.perf_counter()
                row.setdefault('thread_cycles', {})[stage] = cycles()
        deadline = row['deadline_s']
        try:
            prep_cpu = thread_time()
            if native_sample is None:
                sample = row['sample_start_s'] = time.perf_counter()
                mark('sample')
                if sample >= deadline:
                    row['status'] = 'dropped_before_sample'
                    return
                raw = self.drone.get_raw_state()
                row['state_ready_s'] = time.perf_counter()
                mark('state_ready')
                self.collision(raw, sample)
                row['collision_ready_s'] = time.perf_counter()
            else:
                row['native_sample_consumed_s'] = time.perf_counter()
                row['front_wait_ms'] = 1000*(row['native_sample_consumed_s']-row['release_s'])
                if split_prep:
                    self.runtime.update_cpu_costs(front_wait_ms=row['front_wait_ms'])
                raw, hit, sample = native_sample['packet']
                row['sample_start_s'] = sample
                self.collision(raw, sample, hit=hit)
                if time.perf_counter() >= deadline:
                    row['status'] = 'dropped_native_handoff_late'
                    return
            mark('collision_ready')
            if trace_cycles:
                refs, analysis, physical, safety_ms = self.publish(raw, sample, mark=mark)
            else:
                refs, analysis, physical, safety_ms = self.publish(raw, sample)
            row.update(safety_deadline_s=physical['deadline_s'], budget_ms=analysis['budget_s']*1000,
                       safety_ms=safety_ms, held_version=self.version,
                       held_control=self.held.tolist())
            mark('encode_start')
            feature = encode(raw, refs)
            row['feature_ready_s'] = time.perf_counter()
            mark('feature_ready')
            row['prep_thread_cpu_ms'] = 1000*(thread_time()-prep_cpu)
            if native_sample is not None:
                # Windows thread CPU time can move in 15.625 ms ticks. Retain the
                # measured wall time in the worker as the shared-service allowance.
                row['input_worker_wall_ms'] = 1000*(row['feature_ready_s']-row['native_sample_consumed_s'])
            if split_prep:
                self.runtime.update_cpu_costs(prep_ms=row['input_worker_wall_ms'])
            else:
                self.runtime.update_cpu_costs(prep_ms=1000*(row['feature_ready_s']-sample))
            if row['feature_ready_s'] >= deadline:
                row['status'] = 'dropped_preprocess'
                return
            post_reserve = self.runtime.command_reserve_ms/1000
            if getattr(self, 'teacher', None) is not None:
                row['feature'] = feature
                row['teacher_start_s'] = time.perf_counter()
                action, diagnostics = self.teacher.query(raw, refs)
                row['teacher_solve_ms'] = diagnostics['solve_ms']
                row['teacher_output'] = action.tolist()
                row['teacher_diagnostics'] = {k: v.tolist() if isinstance(v, np.ndarray) else v
                                              for k, v in diagnostics.items()}
            else:
                future = self.runtime.enqueue(row, feature)
                try:
                    action = future.result(timeout=max(0., deadline-time.perf_counter()-post_reserve))
                except (TimeoutError, CancelledError):
                    future.cancel()
                    row['status'] = 'dropped_inference'
                    return
                row['board_output'] = action.tolist()
            row['output_ready_s'] = time.perf_counter()
            delay_ms = getattr(self.args, 'injected_delay_ms', 0.)
            if delay_ms:
                time.sleep(delay_ms/1000.)
                row['injected_delay_end_s'] = time.perf_counter()
            if time.perf_counter()+post_reserve >= deadline:
                row['status'] = 'dropped_before_command'
                return
            applied = np.clip(action, [2., -3., -3., -2.], [20., 3., 3., 2.])
            command_cpu = thread_time()
            row['command_start_s'] = time.perf_counter()
            mark('command_start')
            with self.runtime.cv if hasattr(self.runtime, 'cv') else nullcontext():
                self.command_pending = True
            send_control(self.drone, applied, self.calibration, duration=1.)
            row['command_submitted_s'] = time.perf_counter()
            mark('command_submitted')
            self.client.ping()  # submission completion proxy, not a hardware actuator timestamp
            ack = time.perf_counter()
            mark('command_return')
            row['command_thread_cpu_ms'] = 1000*(thread_time()-command_cpu)
            with self.runtime.cv if hasattr(self.runtime, 'cv') else nullcontext():
                self.runtime.update_cpu_costs(post_ms=1000*(ack-row['command_start_s']))
                row.update(command_return_s=ack, status='delivered' if ack < deadline else 'command_return_late',
                           applied_control=applied.tolist(), clipped=bool(np.any(applied != action)),
                           physical_deadline_missed=ack > physical['deadline_s'])
                self.held = applied.copy()
                self.held[0] = throttle_to_thrust(thrust_to_throttle(applied[0], self.calibration), self.calibration)
                self.actuation_s, self.version = ack, self.runtime.command_changed(self.name)
                self.command_pending = False
            defer_refresh = (getattr(self.args, 'refresh_at_release', False) or
                             getattr(self.args, 'independent_observer', False))
            if getattr(self.args, 'refresh_when_slack', False):
                with self.runtime.cv:
                    next_release = self.runtime.loops[self.name]['next']
                budget_ms = getattr(self, 'refresh_budget_ms', self.args.prep_ms+self.args.post_ms)
                row['refresh_available_ms'] = 1000*(next_release-time.perf_counter())
                row['refresh_budget_ms'] = budget_ms
                defer_refresh = row['refresh_available_ms'] <= budget_ms+self.args.decision_ms
            if defer_refresh:
                # The next scheduled sample pairs its state with this held control.
                # command_changed has invalidated the previous observation; planning
                # must wait for a new pair instead of reusing a stale certificate.
                row['paired_refresh'] = 'next_scheduled_sample'
                return
            # Pair a NEW state with the NEW held control; do not attach an old B to it.
            sample = row['paired_sample_start_s'] = time.perf_counter()
            mark('paired_sample')
            paired_cpu = thread_time()
            raw = self.drone.get_raw_state()
            row['paired_state_ready_s'] = time.perf_counter()
            mark('paired_state_ready')
            _, _, _, row['paired_safety_ms'] = self.publish(raw, sample)
            self.collision(raw, sample)
            row['paired_thread_cpu_ms'] = 1000*(thread_time()-paired_cpu)
            self.refresh_budget_ms = max(getattr(self, 'refresh_budget_ms', 0.), 1000*(time.perf_counter()-sample))
        except Exception as exc:
            self.error = f'{type(exc).__name__}: {exc}'
            row.update(status='collision' if isinstance(exc, Contact) else 'error', error=self.error)
            self.runtime.retire(self.name)  # peer loops continue under the reported reduced load
            try:
                self.drone.hover()
                self.client.ping()
            except Exception:
                pass
        finally:
            if hasattr(self.runtime, 'cv'):
                with self.runtime.cv:
                    self.command_pending = False
            row['finished_s'] = time.perf_counter()
            mark('finished')
            if 'command_return_s' in row:
                self.runtime.update_cpu_costs(rearm_ms=1000*(row['finished_s']-row['command_return_s']))
            elif 'command_start_s' in row:
                self.runtime.update_cpu_costs(post_ms=1000*(row['finished_s']-row['command_start_s']))
            if native_sample is None:
                self.runtime.finish(row)
            else:
                with self.runtime.cv:
                    self.runtime.finish(row)
                    try:
                        self.native_io.complete_sample(self.name, row['cycle'])
                    except Exception as exc:
                        self.error = f'native completion: {exc}'
                        self.runtime.retire(self.name)

    def close(self):
        try:
            if self.error is None and time.perf_counter() >= self.origin:
                sample = time.perf_counter()
                raw = self.drone.get_raw_state()
                self.publish(raw, sample)
                self.collision(raw, sample)
        except Exception as exc:
            self.error = f'{type(exc).__name__}: {exc}'
        finally:
            self.drone.hover()
            self.client.ping()


def observe_independently(pilot, stop, stats):
    """Own RPC client; reject a state if command overlaps its sampling."""
    from airsim_interface import AirSimDrone, require_airsim
    try:
        client = require_airsim().MultirotorClient(ip=pilot.args.airsim_host, timeout_value=2.)
        client.ping()
        drone = AirSimDrone(client, pilot.name)
        while not stop.is_set():
            if time.perf_counter() < pilot.origin:
                stop.wait(.01)
                continue
            with pilot.runtime.cv:
                if pilot.command_pending or pilot.runtime.loops[pilot.name]['retired']:
                    stats['pending_skip'] += 1
                    held = None
                else:
                    held, version, applied = pilot.held.copy(), pilot.version, pilot.actuation_s
            if held is not None:
                sampled = time.perf_counter()
                raw = drone.get_raw_state()
                state_ready = time.perf_counter()
                analysis = pilot.analyzer.analyze(raw, held)
                now = time.perf_counter()
                deadline = safe_deadline(sampled, now, applied, analysis, pilot.args.maximum_update_ms/1000)
                with pilot.runtime.cv:
                    if not pilot.command_pending and version == pilot.version:
                        accepted = pilot.runtime.observe(pilot.name, sampled, deadline['deadline_s'], analysis, version)
                    else:
                        accepted = False
                stats['accepted' if accepted else 'stale_rejected'] += 1
                stats['rpc_ms'].append(1000*(state_ready-sampled))
                stats['total_ms'].append(1000*(now-sampled))
            stop.wait(pilot.args.observer_ms/1000.)
    except Exception as exc:
        stats['error'] = f'{type(exc).__name__}: {exc}'


def consume_native_calendar(io, runtime, pilots, executors, pending):
    # Serialize Python's template/event bookkeeping with planner commits.
    with runtime.cv:
        for name, loop in runtime.loops.items():
            if loop['retired'] and name in io.scheduled:
                io.cancel(name)
        releases, samples = io.drain_calendar_samples()
        for event in releases:
            row = runtime.accept_native_release(event)
            if row['status'] == 'sampling':
                pending[(row['vehicle'], row['cycle'])] = row
        for sample in samples:
            name, cycle = sample['vehicle'], sample['cycle']
            row = pending.pop((name, cycle))
            row.update({k: v for k, v in sample.items() if k != 'packet'})
            if sample['sample_status'] == 1 or runtime.loops[name]['retired']:
                row['status'] = 'dropped_before_sample' if sample['sample_status'] == 1 else 'dropped_retired'
                row['finished_s'] = sample['native_sample_finished_s']
                runtime.finish(row)
                io.complete_sample(name, cycle)
            else:
                executors[name].submit(pilots[name].run_job, row, sample)


def serve_board(runtime, board):
    while True:
        with runtime.cv:
            if runtime.closed:
                return
            row = runtime.take(time.perf_counter())
            if row is None:
                runtime.cv.wait(.002)
                continue
        try:
            action, timing = predict(board, row['model_id'], row['feature'])
            runtime.board_done(row, time.perf_counter(), action=action, timing=timing)
        except Exception as exc:
            runtime.board_done(row, time.perf_counter(), error=f'{type(exc).__name__}: {exc}')


def plan_live(runtime, failures):
    try:
        while True:
            with runtime.cv:
                if runtime.closed or not runtime.adaptive:
                    return
            started, cpu_started = time.perf_counter(), thread_time()
            runtime.plan_once()
            runtime.planner_calls.append(dict(wall_ms=1000*(time.perf_counter()-started),
                                              cpu_ms=1000*(thread_time()-cpu_started)))
            # A bounded service interval; no dependency on a foreground idle slot.
            time.sleep(.01)
    except Exception as exc:
        failures.append(f'planner: {type(exc).__name__}: {exc}')


def run(args):
    from airsim_interface import AirSimDrone, require_airsim, vehicle_local_to_world_offset
    from config import load_calibration
    from environment import Gate
    from gate_visualization import spawn_gate_sequences
    from experiments.falcon_paper_fidelity_acado_v3_continuous.demo_acado_five_gates import _crossing_results
    from types import SimpleNamespace

    layouts, route_sources, case = load_flight_layouts(args)
    initial_periods = (json.loads(args.initial_periods.read_text(encoding='utf-8'))
                       if getattr(args, 'initial_periods', None) is not None else None)
    qualification = None
    if getattr(args, 'period_qualification', None) is not None:
        qualification = json.loads(args.period_qualification.read_text(encoding='utf-8'))
        if (case is None or qualification['case_sha256'] != sha256(args.case) or
                qualification['model_id'] != args.model_id or not qualification['evidence']):
            raise ValueError('empirical period qualification does not match frozen case/model/evidence')
    evaluation_ms = getattr(args, 'evaluation_ms', 0.)
    if getattr(args, 'independent_observer', False) and (not np.isfinite(args.observer_ms) or args.observer_ms < 10.):
        raise ValueError('observer-ms must be finite and at least 10ms')
    injected_delay_ms = getattr(args, 'injected_delay_ms', 0.)
    if not np.isfinite(injected_delay_ms) or injected_delay_ms < 0.:
        raise ValueError('injected-delay-ms must be finite and nonnegative')
    if not np.isfinite(evaluation_ms) or evaluation_ms < 0.:
        raise ValueError('evaluation-ms must be finite and nonnegative')
    if initial_periods is not None and (set(initial_periods) != set(args.vehicles) or
            any(t not in args.periods_ms for t in initial_periods.values())):
        raise ValueError('initial-periods must specify one supported period for every selected vehicle')
    reference_teacher = getattr(args, 'reference_teacher', False)
    native_sampling = getattr(args, 'native_sampling', False)
    split_native_prep = getattr(args, 'split_native_prep', False)
    if split_native_prep and not native_sampling:
        raise ValueError('split-native-prep requires native-sampling')
    if native_sampling and (reference_teacher or min(args.periods_ms) < 20.):
        raise ValueError('native sampling requires board control with periods >=20ms')
    if getattr(args, 'teacher_variant', None) is not None and not reference_teacher:
        raise ValueError('teacher-variant requires reference-teacher; board models are unchanged')
    if reference_teacher and (len(args.vehicles) != 1 or args.policy != 'fixed'):
        raise ValueError('reference-teacher is a single-vehicle fixed-period diagnostic only')
    if (args.period_ms not in args.periods_ms or not 1 <= len(args.vehicles) <= 12
            or len(set(args.vehicles)) != len(args.vehicles) or args.duration is not None and args.duration <= 0
            or not 0 < max(args.periods_ms) < args.maximum_update_ms or args.gate_width <= 1.05):
        raise ValueError('invalid periods, vehicles, duration, update cap or gate geometry')
    args.output.mkdir(parents=True, exist_ok=False)
    if case is not None:
        shutil.copyfile(args.case, args.output/'case_frozen.json')
    board = client = scene = runtime = native_io = None
    native_pending = {}
    executors, pilots, threads, errors = {}, {}, [], []
    observer_stop = threading.Event()
    observer_stats = {}
    evaluators = {}
    setup, result = {}, dict(flight_passed=False)
    timer = simulation_touched = False
    try:
        warm = []
        entry, manifest = None, dict(wire_tag_hex=None, normalization_sha256=None)
        if not reference_teacher:
            board, manifest = make_client(args.bundle, args.board, args.board_port, 100.)
            entry = next(m for m in manifest['models'] if m['numeric_id'] == args.model_id)
            golden = args.bundle / entry.get('golden_file', f'inputs/{args.model_id}.txt')
            if entry.get('golden_file_sha256') and sha256(golden) != entry['golden_file_sha256']:
                raise ValueError('golden input hash mismatch')
            values = np.loadtxt(golden)
            for i in range(40):
                x = values[i % len(values)]
                start = time.perf_counter()
                action, _ = predict(board, args.model_id, x[:237].astype(np.float32))
                warm.append((time.perf_counter()-start)*1000)
                if np.max(abs(action-x[237:241])) > 1e-4:
                    raise ValueError('board golden output mismatch')
        cost = max(warm)+1. if warm else 1.  # unused replay estimate in fixed teacher diagnostic
        planners0 = [StoredRoutePlanner(layout) for layout, _ in layouts]
        duration = args.duration or max(20., max(p.trajectory.duration for p in planners0)+2.)
        extents = [np.array([r.position for r in p.reference_window_for_trajectory(
            p.trajectory, elapsed=0., dt=duration/200, horizon=201)]) for p in planners0]
        spacing = max(12., max(float(np.ptp(e[:, 1])) for e in extents)+10.)
        load_airsim_client()
        airsim = require_airsim()
        client = airsim.MultirotorClient(ip=args.airsim_host, timeout_value=3.)
        client.confirmConnection()
        if not set(args.vehicles) <= set(client.listVehicles()):
            raise ValueError(f'Play must contain {args.vehicles}')
        prepare_gate_scene(client, args.output)
        client.reset()
        simulation_touched = True
        client.simPause(True)
        offsets = {v: vehicle_local_to_world_offset(client, v) for v in args.vehicles}
        planners, gates = {}, {}
        for i, name in enumerate(args.vehicles):
            shift = np.array([0., (i-(len(args.vehicles)-1)/2)*spacing-np.mean(extents[i][:, 1]), 0.])-offsets[name]
            local = translate_layout(layouts[i][0], shift)
            planner = planners[name] = StoredRoutePlanner(local)
            gates[name] = [Gate(g['index'], g['center'], g['orientation']) for g in local['gates']]
            first = planner.reference_window_for_trajectory(planner.trajectory, elapsed=0., dt=.1, horizon=21)[0]
            drone = AirSimDrone(client, name)
            drone.enable_api_control(True)
            drone.arm()
            kin = airsim.KinematicsState()
            kin.position = airsim.Vector3r(*map(float, first.position))
            q = first.orientation
            kin.orientation = airsim.Quaternionr(float(q[1]), float(q[2]), float(q[3]), float(q[0]))
            kin.linear_velocity = kin.angular_velocity = airsim.Vector3r(0., 0., 0.)
            kin.linear_acceleration = kin.angular_acceleration = airsim.Vector3r(0., 0., 0.)
            client.simSetKinematics(kin, ignore_collision=True, vehicle_name=name)
            drone.hover()
        scene = spawn_gate_sequences(client, gates, width=args.gate_width, height=4., thickness=.15,
                                     asset_name='1M_Cube', object_prefix='FalconJointGate_', vehicle_world_offsets=offsets)
        calibration = load_calibration()
        bounds = HoldBounds(2., .01, .05, .15, 10., 'Route-B development assumptions; not plant-certified')
        # Thread/RPC initialization takes place while physics is paused.
        origin = 0.
        runtime = JointRuntime(args.vehicles, origin, args.period_ms, args.periods_ms, args.model_id, cost,
                               adaptive=args.policy == 'adaptive', order=args.order,
                               prep_ms=args.prep_ms, post_ms=args.post_ms, decision_ms=args.decision_ms,
                               initial_periods=initial_periods,
                               qualified_periods=qualification['periods_ms'] if qualification else None)
        for name in args.vehicles:
            ex = executors[name] = ThreadPoolExecutor(max_workers=1, thread_name_prefix=name)
            pilots[name] = ex.submit(Pilot, args, name, planners[name], gates[name], offsets[name],
                                    runtime, calibration, origin, bounds).result(timeout=10.)
        if native_sampling:
            from .framework_native_io import LIBRARY, NativeCommandIO
            native_io = NativeCommandIO(args.vehicles, calibration, args.post_ms/1000.,
                                        host=args.airsim_host, auto_sample=True)
            for pilot in pilots.values():
                pilot.native_io = native_io
        if evaluation_ms:
            from .ip37_evaluation import Evaluator
            for name, pilot in pilots.items():
                evaluators[name] = Evaluator(args.airsim_host, name, planners[name], pilot.initial,
                                            0., duration, evaluation_ms)
        origin = time.perf_counter()+.5
        shift = origin-runtime.origin
        runtime.origin = origin
        for name, loop in runtime.loops.items():
            loop.update(previous=loop['previous']+shift, next=loop['next']+shift, observation=None)
            pilot = pilots[name]
            pilot.origin = pilot.actuation_s = origin
            pilot.trace.clear()
            executors[name].submit(pilot.publish, pilot.initial, origin).result(timeout=3.)
        for evaluator in evaluators.values():
            evaluator.origin = origin
            evaluator.trace[0]['sample_s'] = origin
        setup = dict(policy=args.policy, order=args.order, vehicles=args.vehicles, routes=route_sources,
            endpoint=board.peer if board else None, model=entry, wire_tag_hex=manifest['wire_tag_hex'],
            controller='CPU convex MPC teacher' if reference_teacher else 'real IP37 board DNN',
            reference_teacher=reference_teacher,
            normalization_sha256=manifest['normalization_sha256'], period_ms=args.period_ms,
            initial_periods_ms={v: loop['period_ms'] for v, loop in runtime.loops.items()},
            evaluation_period_ms=evaluation_ms,
            paired_refresh_policy=('independent_observer' if getattr(args, 'independent_observer', False) else
                                   'slack_guard' if getattr(args, 'refresh_when_slack', False) else
                                   'next_release' if getattr(args, 'refresh_at_release', False) else 'immediate'),
            trace_thread_cycles=getattr(args, 'trace_thread_cycles', False),
            calendar_thread_id=threading.get_native_id(),
            release_sampling_backend='cpp_airlib' if native_sampling else 'python_rpc',
            split_native_prep=split_native_prep,
            prep_cost_model='native_wait_plus_worker_wall' if split_native_prep else 'sample_to_input_wall',
            native_io_sha256=sha256(LIBRARY) if native_sampling else None,
            independent_observer=getattr(args, 'independent_observer', False),
            observer_ms=getattr(args, 'observer_ms', 50.),
            periods_ms=args.periods_ms, duration_s=duration, origin_s=origin, warm_rtt_ms=describe(warm),
            initial_board_cost_ms=None if reference_teacher else cost,
            gate_width_m=args.gate_width, hold_bounds=asdict(bounds),
            expected_gates={v: len(gates[v]) for v in args.vehicles},
            case_sha256=sha256(args.case) if case is not None else None,
            command_reserve_ms=runtime.command_reserve_ms,
            maximum_update_ms=args.maximum_update_ms, decision_budget_ms=args.decision_ms,
            model_switching_enabled=False, mpc_fallbacks=0, outputs_applied=True,
            startup_hold='calibrated hover thrust and zero body rates, primed while paused',
            period_qualification=qualification or 'exploratory allowed periods; not a certified admissibility map',
            forecast_scope='first physical obligation plus finite future period deadlines; refreshed online',
            execution='single CPU teacher diagnostic' if reference_teacher else
                      'host EDF, one outstanding real whole-task IP37 request, no DPU preemption')
        if reference_teacher:
            setup['teacher_signature'] = pilots[args.vehicles[0]].teacher.signature()
            if getattr(args, 'teacher_variant', None) is not None:
                setup['source_sha256_diagnostic'] = sha256(Path(__file__).with_name('teacher_attitude_probe.py'))
        setup['source_sha256'] = {name: sha256(Path(__file__).with_name(name)) for name in (
            'fly_ip37_joint.py', 'ip37_joint_runtime.py', 'bottleneck_abc.py',
            'bottleneck_joint.py', 'held_control_safety.py', 'convex_teacher.py', 'fast_convex_preprocess.py')}
        atomic_json(args.output/'setup.json', setup)
        if __import__('os').name == 'nt':
            import ctypes
            timer = ctypes.windll.winmm.timeBeginPeriod(1) == 0
        if not reference_teacher:
            thread = threading.Thread(target=serve_board, args=(runtime, board), name='IP37-worker')
            threads.append(thread)
            thread.start()
        for evaluator in evaluators.values():
            evaluator.thread.start()
        if getattr(args, 'independent_observer', False):
            for name, pilot in pilots.items():
                observer_stats[name] = dict(accepted=0, stale_rejected=0, pending_skip=0, rpc_ms=[], total_ms=[])
                thread = threading.Thread(target=observe_independently, args=(pilot, observer_stop, observer_stats[name]), name=f'observer-{name}')
                threads.append(thread)
                thread.start()
        if runtime.adaptive and not native_sampling:
            thread = threading.Thread(target=plan_live, args=(runtime, errors), name='ABC-planner')
            threads.append(thread)
            thread.start()
        # The initial trace and deadlines are anchored at origin. Do NOT let
        # physics advance through the reserved startup interval before release 0.
        launch_wait = origin-time.perf_counter()
        if launch_wait <= 0.:
            raise RuntimeError('launch reserve exhausted while physics was still paused')
        time.sleep(launch_wait)
        setup['unpause_start_s'] = time.perf_counter()
        client.simPause(False)
        setup['unpause_return_s'] = time.perf_counter()
        if native_sampling:
            for name, loop in runtime.loops.items():
                period, release = loop['period_ms']/1000., loop['next']
                native_io.schedule(dict(vehicle=name, sample_id=f'{name}:0', model=args.model_id,
                    planned_release_s=release, requested_period_s=period, deadline_s=release+period),
                    origin+duration, static=True)
            runtime.native_io, runtime.native_end = native_io, origin+duration
            if runtime.adaptive:
                thread = threading.Thread(target=plan_live, args=(runtime, errors), name='ABC-planner')
                threads.append(thread)
                thread.start()
        print(f"{setup['controller']}: {len(args.vehicles)} UAVs, T={args.period_ms:g}ms, policy={args.policy}.", flush=True)
        while time.perf_counter() < origin+duration:
            now = time.perf_counter()
            if native_sampling:
                consume_native_calendar(native_io, runtime, pilots, executors, native_pending)
            else:
                for row in runtime.release_due(now, stop_before=origin+duration):
                    executors[row['vehicle']].submit(pilots[row['vehicle']].run_job, row)
            with runtime.cv:
                if not evaluators and all(loop['retired'] for loop in runtime.loops.values()):
                    break
            time.sleep(.001)
        # Account for every planned release inside the window even when the
        # final timer wake occurs just past its endpoint. No release at t=end.
        if native_sampling:
            with runtime.cv:
                runtime.adaptive = False
                runtime.revision += 1
            drain_deadline = time.perf_counter()+3.5
            while True:
                consume_native_calendar(native_io, runtime, pilots, executors, native_pending)
                if not native_pending and all(loop['retired'] or loop['next'] >= origin+duration-1e-9
                                              for loop in runtime.loops.values()):
                    break
                if time.perf_counter() >= drain_deadline:
                    raise RuntimeError('native calendar/sample drain timed out')
                time.sleep(.001)
        else:
            for row in runtime.release_due(time.perf_counter(), stop_before=origin+duration):
                executors[row['vehicle']].submit(pilots[row['vehicle']].run_job, row)
    except (Exception, KeyboardInterrupt) as exc:
        errors.append(f'{type(exc).__name__}: {exc}')
    finally:
        observer_stop.set()
        if runtime is not None:
            with runtime.cv:
                runtime.adaptive = False
                runtime.revision += 1
        if native_io is not None:
            for name in list(native_io.scheduled):
                try:
                    native_io.cancel(name)
                except Exception as exc:
                    errors.append(f'native calendar cancel {name}: {type(exc).__name__}: {exc}')
        if runtime is not None:
            # Stop releasing, but finish the final valid jobs before closing the board worker.
            for ex in executors.values():
                try:
                    ex.submit(lambda: None).result(timeout=4.)
                except Exception as exc:
                    errors.append(f'drain: {type(exc).__name__}: {exc}')
            with runtime.cv:
                runtime.closed = True
                for row in runtime.queue:
                    row['future'].cancel()
                runtime.cv.notify_all()
        for thread in threads:
            thread.join()  # board socket has a finite timeout; drain before closing it
        if native_io is not None:
            try:
                native_io.close()
            except Exception as exc:
                errors.append(f'native shutdown: {exc}')
        for name, stats in observer_stats.items():
            if stats.get('error'):
                errors.append(f'observer {name}: {stats["error"]}')
        for name, evaluator in evaluators.items():
            if time.perf_counter() < evaluator.origin+evaluator.duration:
                evaluator.stop.set()  # interrupted run: never fabricate a full evaluation window
            if evaluator.thread.ident is not None:
                evaluator.thread.join(timeout=3.)
            if evaluator.thread.is_alive():
                evaluator.stop.set()
                evaluator.thread.join()
            atomic_json(args.output/f'evaluation_{name}.json', evaluator.trace)
        for name, ex in executors.items():
            try:
                if name in pilots:
                    ex.submit(pilots[name].close).result(timeout=5.)
            except Exception as exc:
                errors.append(f'{name} cleanup: {exc}')
            ex.shutdown(wait=True)
        try:
            if client is not None and simulation_touched:
                client.simPause(False)
            if scene is not None:
                scene.destroy()
        except Exception as exc:
            errors.append(f'scene cleanup: {type(exc).__name__}: {exc}')
        if board is not None:
            board.close()
        if timer:
            import ctypes
            ctypes.windll.winmm.timeEndPeriod(1)
        per_vehicle = {}
        for name, pilot in pilots.items():
            if pilot.error is not None:
                print(f'{name}: {pilot.error}', flush=True)
                if pilot.contact is None:
                    errors.append(f'{name}: {pilot.error}')
            trace = pilot.trace
            geometry = dict(width=args.gate_width, height=4., thickness=.15, clearance=.25)
            crossings = [dict(gate=g, crossed=p is not None, valid=valid,
                              lateral_offset_m=float(p.lateral_offset) if p is not None else None,
                              vertical_offset_m=float(p.vertical_offset) if p is not None else None,
                              body_margin_m=min(args.gate_width/2-.075-.525-abs(p.lateral_offset),
                                                2.-.075-.525-abs(p.vertical_offset)) if p is not None else None)
                         for g, p, _, valid in
                         _crossing_results([r['position'] for r in trace], [r['velocity'] for r in trace],
                                           SimpleNamespace(gates=gates[name]), geometry)] if len(trace) > 1 else []
            ratio = ((trace[-1]['sim_ns']-trace[0]['sim_ns'])/1e9/(trace[-1]['t']-trace[0]['t'])
                     if len(trace) > 1 and trace[-1]['t'] > trace[0]['t'] else None)
            per_vehicle[name] = dict(error=pilot.error, contact=pilot.contact, crossings=crossings,
                                     expected_gates=len(gates[name]),
                                     body_clearance_passed=len(crossings) == len(gates[name]) and
                                         all(c['crossed'] and c['body_margin_m'] > 0. for c in crossings),
                                     sim_wall_ratio=ratio, **tracking_metrics(trace, setup.get('duration_s', 20.)))
            atomic_json(args.output/f'{name}_trace.json', trace)
        rows = runtime.public_rows() if runtime is not None else []
        decisions = runtime.decisions if runtime is not None else []
        result = dict(flight_passed=bool(per_vehicle) and not errors and all(
            v['error'] is None and v['window_covered'] and len(v['crossings']) == v['expected_gates']
            and all(c['valid'] for c in v['crossings']) for v in per_vehicle.values()),
            errors=errors, vehicles=per_vehicle, requests=len(rows),
            controller='CPU convex MPC teacher' if reference_teacher else 'real IP37 board DNN',
            board_used=not reference_teacher,
            body_clearance_passed=bool(per_vehicle) and all(v['body_clearance_passed'] for v in per_vehicle.values()),
            calendar_coverage_passed=bool(runtime) and not errors and all(
                loop['retired'] or loop['next'] >= runtime.origin+setup.get('duration_s', 20.)-1e-9
                for loop in runtime.loops.values()),
            outcomes=dict(Counter(r['status'] for r in rows)),
            timing_passed=bool(rows) and all(r['status'] == 'delivered' for r in rows),
            timing_before_contact_passed=bool(rows) and all(r['status'] == 'delivered' or (
                r['status'] == 'collision' and r['finished_s'] < r['deadline_s'] and
                r.get('command_return_s', r['release_s']) < r['deadline_s']) for r in rows),
            sim_clock_passed=bool(per_vehicle) and all(v['sim_wall_ratio'] is not None and
                .95 <= v['sim_wall_ratio'] <= 1.05 for v in per_vehicle.values()),
            conditional_physical_deadline_misses=sum(r.get('physical_deadline_missed', False) for r in rows),
            decisions=dict(Counter(d['status'] for d in decisions)),
            snapshot_skips=dict(runtime.skips) if runtime else {},
            period_updates=sum(len(d['updates']) for d in decisions),
            decision_ms=describe([d['elapsed_ms'] for d in decisions]),
            planner_service_ms=describe([d['wall_ms'] for d in runtime.planner_calls]) if runtime else None,
            planner_thread_cpu_ms=describe([d['cpu_ms'] for d in runtime.planner_calls]) if runtime else None,
            planner_calls=len(runtime.planner_calls) if runtime else 0,
            release_jitter_ms=describe([1000*(r['published_s']-r['release_s']) for r in rows]),
            release_to_sample_ms=describe([1000*(r['sample_start_s']-r['release_s']) for r in rows
                                           if 'state_ready_s' in r]),
            response_ms=describe([1000*(r['command_return_s']-r['release_s']) for r in rows if 'command_return_s' in r]),
            board_queue_ms=describe([1000*(r['board_start_s']-r['queue_s']) for r in rows if 'board_start_s' in r]),
            safety_ms=describe([r['safety_ms'] for r in rows if 'safety_ms' in r]),
            state_rpc_ms=describe([1000*(r['state_ready_s']-r['sample_start_s']) for r in rows if 'state_ready_s' in r]),
            sample_to_input_ms=describe([1000*(r['feature_ready_s']-r['sample_start_s']) for r in rows if 'feature_ready_s' in r]),
            teacher_solve_ms=describe([r['teacher_solve_ms'] for r in rows if 'teacher_solve_ms' in r]),
            command_rpc_ms=describe([1000*(r['command_return_s']-r['command_start_s']) for r in rows if 'command_return_s' in r]),
            command_submit_ms=describe([1000*(r['command_submitted_s']-r['command_start_s']) for r in rows if 'command_submitted_s' in r]),
            command_ping_ms=describe([1000*(r['command_return_s']-r['command_submitted_s']) for r in rows if 'command_return_s' in r]),
            post_refresh_ms=describe([1000*(r['finished_s']-r['command_return_s']) for r in rows if 'command_return_s' in r]),
            prep_thread_cpu_ms=describe([r['prep_thread_cpu_ms'] for r in rows if 'prep_thread_cpu_ms' in r]),
            command_thread_cpu_ms=describe([r['command_thread_cpu_ms'] for r in rows if 'command_thread_cpu_ms' in r]),
            paired_thread_cpu_ms=describe([r['paired_thread_cpu_ms'] for r in rows if 'paired_thread_cpu_ms' in r]),
            final_empirical_board_cost_ms=runtime.cost_ms if runtime and not reference_teacher else None,
            final_empirical_cpu_cost_ms=dict(prep=runtime.prep_ms, post=runtime.post_ms, rearm=runtime.rearm_ms,
                                             front_wait=runtime.front_wait_ms) if runtime else None,
            guarantee=('CPU teacher capability diagnostic; not board deployment or a safety certificate'
                       if reference_teacher else
                       'exploratory real-board control; empirical costs and uncertified plant bounds'))
        if native_sampling:
            result['native_sampling_ms'] = {
                'sample_rpc': describe([1000*(r['native_sample_finished_s']-r['sample_start_s'])
                                        for r in rows if 'collision_ready_s' in r]),
                'sample_ready_to_poll': describe([1000*(r['native_sample_received_s']-r['native_sample_finished_s'])
                                                  for r in rows if 'collision_ready_s' in r]),
                'sample_ready_to_worker': describe([1000*(r['native_sample_consumed_s']-r['native_sample_finished_s'])
                                                    for r in rows if 'native_sample_consumed_s' in r]),
                'input_worker_wall': describe([r['input_worker_wall_ms'] for r in rows if 'input_worker_wall_ms' in r]),
                'front_wait': describe([r['front_wait_ms'] for r in rows if 'front_wait_ms' in r]),
            }
        if evaluators:
            evaluation = {v: e.report() for v, e in evaluators.items()}
            valid = set(evaluation) == set(args.vehicles) and all(e['valid'] for e in evaluation.values())
            result['evaluation'] = dict(vehicles=evaluation, valid=valid,
                total_J=sum(e['score']['J'] for e in evaluation.values()) if valid else None,
                scope='Eq4-form, Wx=I10, common20ms interpolation; fixed-rate independent observations; includes post-contact hover')
        if getattr(args, 'trace_thread_cycles', False):
            stages = ('collision_ready', 'reference_ready', 'flight_trace_ready',
                      'safety_deadline_ready', 'observation_published', 'encode_start', 'feature_ready')
            marks = [row['thread_marks_s'] for row in rows if 'thread_marks_s' in row]
            result['prep_phases_ms'] = {
                f'{start}_to_{end}': describe([1000*(m[end]-m[start]) for m in marks if start in m and end in m])
                for start, end in zip(stages, stages[1:])
            }
        atomic_json(args.output/'requests.json', rows)
        atomic_json(args.output/'decisions.json', decisions)
        if observer_stats:
            atomic_json(args.output/'observer_stats.json', observer_stats)
        if setup:
            atomic_json(args.output/'setup.json', setup)
        atomic_json(args.output/'result.json', result)
        print(f"flight_passed={result['flight_passed']} timing_passed={result['timing_passed']} "
              f"period_updates={result['period_updates']} outcomes={result['outcomes']}", flush=True)
        print(f'Report: {args.output / "result.json"}', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', type=Path, help='frozen natural-route case.json')
    parser.add_argument('--initial-periods', type=Path, help='JSON vehicle->period ms, used identically for fixed/adaptive starts')
    parser.add_argument('--native-sampling', action='store_true', help='board control with a C++ calendar and release-triggered AirLib sampling')
    parser.add_argument('--split-native-prep', action='store_true', help='experimental: charge measured native frontend waits as latency instead of shared CPU service')
    parser.add_argument('--period-qualification', type=Path, help='case-bound empirical allowed period sets; not safety certification')
    parser.add_argument('--evaluation-ms', type=float, default=0., help='independent evaluation sampling interval; 0 disables')
    parser.add_argument('--reference-teacher', action='store_true', help='single-vehicle CPU MPC diagnostic; board is not used')
    parser.add_argument('--teacher-variant', choices=('baseline', 'gain5', 'gain7', 'qp_ff', 'qp_ff_gain7'),
                        help='isolated CPU teacher diagnostic; requires --reference-teacher')
    parser.add_argument('--board', default='192.168.77.2')
    parser.add_argument('--board-port', type=int, default=5238)
    parser.add_argument('--model-id', type=int, default=11, choices=(11,))
    parser.add_argument('--airsim-host', default='127.0.0.1')
    parser.add_argument('--vehicles', nargs='+', default=['Drone1', 'Drone2', 'Drone3'])
    parser.add_argument('--policy', choices=('fixed', 'adaptive'), default='adaptive')
    parser.add_argument('--order', choices=('edf', 'fifo'), default='edf')
    parser.add_argument('--period-ms', type=float, default=50.)
    parser.add_argument('--refresh-at-release', action='store_true', help='diagnostic: pair new held control at next scheduled sample, no stale observation reuse')
    parser.add_argument('--refresh-when-slack', action='store_true', help='experimental: use empirical refresh budget to avoid starting late duplicate reads; not WCET')
    parser.add_argument('--independent-observer', action='store_true', help='experimental independent version-checked observation; defers inline refresh')
    parser.add_argument('--observer-ms', type=float, default=50.)
    parser.add_argument('--trace-thread-cycles', action='store_true', help='Windows CPU-cycle diagnostic, not WCET')
    parser.add_argument('--injected-delay-ms', type=float, default=0., help='diagnostic delay after inference; expired outputs are dropped')
    parser.add_argument('--periods-ms', type=float, nargs='+', default=[20., 30., 40., 50., 60., 80.])
    parser.add_argument('--maximum-update-ms', type=float, default=160.)
    parser.add_argument('--gate-width', type=float, help='gate frame center-line spacing; default 4m without --case')
    parser.add_argument('--duration', type=float)
    parser.add_argument('--prep-ms', type=float, default=3.)
    parser.add_argument('--post-ms', type=float, default=2.)
    parser.add_argument('--decision-ms', type=float, default=3.)
    result = run(parser.parse_args())
    raise SystemExit(0 if result['flight_passed'] and result['sim_clock_passed'] else 1)


if __name__ == '__main__':
    main()
