"""Paired short closed-loop sensitivity trials; manual Play, no training or FPGA claims."""

import argparse
import csv
from dataclasses import asdict
import itertools
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np


MODELS = ("plain4x256", "plain4x512", "plain6x1024", "residual21x1024")
SCENARIOS = ("startup", "tracking", "turn", "lateral_recovery", "velocity_recovery", "attitude_recovery")


def trial_matrix(models, scenarios, periods, delays, repeats, seed):
    """Shuffle matched blocks; rotate model order across repeats to reduce order bias."""
    if not models or not scenarios or repeats < 1:
        raise ValueError("empty matrix or invalid repeats")
    if any(not np.isfinite(t) or t <= 0 for t in periods):
        raise ValueError("periods must be positive and finite")
    if any(not np.isfinite(d) or d < 0 for d in delays) or max(delays) >= min(periods):
        raise ValueError("extra delay must be finite, nonnegative and smaller than every period")
    rng = np.random.default_rng(seed)
    blocks = list(itertools.product(scenarios, periods, delays))
    order = list(models)
    rng.shuffle(order)
    trials = []
    for repeat in range(repeats):
        rng.shuffle(blocks)
        rotated = order[repeat % len(order):] + order[:repeat % len(order)]
        for scenario, period, delay in blocks:
            for model in rotated:
                item = dict(model=model, scenario=scenario, period_ms=period,
                            extra_delay_ms=delay, repeat=repeat)
                item["id"] = f"r{repeat}_{scenario}_T{period:g}_L{delay:g}_{model}"
                trials.append(item)
    return trials


def due_time(inference_ready, extra_delay_ms):
    # ADDITIVE delay, never a claimed fixed end-to-end latency.
    return inference_ready + extra_delay_ms / 1000.


def next_slot(origin, now, period):
    """No burst catch-up; callers separately count skipped releases."""
    return max(0, int(np.floor((now-origin) / period)) + 1)


def tracking_metrics(trace, duration):
    if len(trace) < 2 or trace[-1]["t"] < duration:
        return {"window_covered": False, "tracking_mse_m2": None}
    t = np.asarray([r["t"] for r in trace])
    error = np.asarray([r["error_m"] for r in trace])
    if t[0] > 0 or np.any(np.diff(t) <= 0):
        raise ValueError("invalid observation clock")
    inside = t < duration
    grid = np.r_[t[inside], duration]
    squared = np.r_[error[inside] ** 2, np.interp(duration, t, error ** 2)]
    integral = np.sum(np.diff(grid) * (squared[:-1] + squared[1:]) / 2)
    return {"window_covered": True, "tracking_mse_m2": float(integral / duration),
            "tracking_rmse_m": float(np.sqrt(integral / duration)),
            "max_tracking_error_m": float(error.max()),
            "max_observation_gap_ms": float(np.diff(t).max() * 1000)}


def paired_summary(reports, baseline=MODELS[0]):
    """Never let early failures appear better because their cost integral is shorter."""
    index = {(r["trial"]["scenario"], r["trial"]["period_ms"],
              r["trial"]["extra_delay_ms"], r["trial"]["repeat"], r["trial"]["model"]): r
             for r in reports}
    pairs = []
    for key, report in index.items():
        if key[-1] == baseline:
            continue
        other = index.get((*key[:-1], baseline))
        if other is None:
            continue
        comparable = report["comparable"] and other["comparable"]
        pairs.append({"trial": report["trial"]["id"], "baseline_trial": other["trial"]["id"],
                      "comparable": comparable,
                      "delta_tracking_mse_m2": (report["tracking_mse_m2"] - other["tracking_mse_m2"])
                      if comparable else None})
    groups = {}
    for report in reports:
        t = report["trial"]
        key = (t["scenario"], t["period_ms"], t["extra_delay_ms"], t["model"])
        groups.setdefault(key, []).append(report)
    aggregates = []
    for key, rows in sorted(groups.items()):
        values = [r["tracking_mse_m2"] for r in rows if r["comparable"]]
        aggregates.append(dict(scenario=key[0], period_ms=key[1], extra_delay_ms=key[2], model=key[3],
            trials=len(rows), comparable=len(values), invalid=len(rows)-len(values),
            comparable_mean_mse_m2=float(np.mean(values)) if values else None,
            comparable_min_mse_m2=float(min(values)) if values else None,
            comparable_max_mse_m2=float(max(values)) if values else None))
    effects = []
    # Same model/scenario/repeat; change only one requested timing factor.
    for key, report in index.items():
        for factor, position in (("extra_delay_ms", 2), ("period_ms", 1)):
            matched = [(k, r) for k, r in index.items()
                       if all(k[j] == key[j] for j in range(5) if j != position)]
            reference_key, reference = min(matched, key=lambda item: item[0][position])
            if reference_key == key:
                continue
            valid = report["comparable"] and reference["comparable"]
            effects.append(dict(factor=factor, trial=report["trial"]["id"],
                reference_trial=reference["trial"]["id"], comparable=valid,
                delta_tracking_mse_m2=report["tracking_mse_m2"]-reference["tracking_mse_m2"] if valid else None))
    return {"completed_trials": len(reports), "comparable_trials": sum(r["comparable"] for r in reports),
            "failed_or_timing_invalid_trials": sum(not r["comparable"] for r in reports),
            "paired_against": baseline, "negative_delta_favors_candidate": True,
            "pairs": pairs, "timing_effects": effects, "aggregates": aggregates, "trials": reports,
            "aggregation_warning": "means condition on valid trials; inspect invalid counts before any ranking; two repeats do not establish significance",
            "scope": "validation-route sensitivity pilot, not a safety certificate or final held-out evaluation"}


def check_resume(stored, current, resume):
    if stored is not None and (not resume or stored != current):
        raise ValueError("existing experiment differs or --resume missing; use a NEW output directory")


def save_summary(output, reports, baseline):
    from .decision_dataset import atomic_json
    summary = paired_summary(reports, baseline)
    atomic_json(output / "summary.json", summary)
    for filename, key in (("summary.csv", "aggregates"), ("model_pairs.csv", "pairs"),
                          ("timing_effects.csv", "timing_effects")):
        rows = summary[key]
        temporary = output / (filename + ".tmp")
        with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
            if rows:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        temporary.replace(output / filename)


def make_start(window, route_duration, scenario):
    phase = 0. if scenario == "startup" else min(2., route_duration / 4)
    if scenario == "turn":
        times = np.linspace(1., max(1., route_duration-4.), 80)
        phase = float(max(times, key=lambda t: np.linalg.norm(window.window(float(t))[0].acceleration[1:])))
    ref = window.window(phase)[0]
    position, velocity, quaternion = ref.position.copy(), ref.velocity.copy(), ref.orientation.copy()
    if scenario == "startup":
        velocity[:] = 0.
    elif scenario == "lateral_recovery":
        position[1] += .8
    elif scenario == "velocity_recovery":
        velocity += np.array([1., 1., 0.])
    elif scenario == "attitude_recovery":
        # Quaternion multiplication q_ref * roll(0.15 rad), wxyz convention.
        w, x, y, z = quaternion
        c, s = np.cos(.075), np.sin(.075)
        quaternion = np.array([w*c-x*s, w*s+x*c, y*c+z*s, z*c-y*s])
    return dict(reference_phase_s=phase, position=position.tolist(), velocity=velocity.tolist(),
                quaternion=quaternion.tolist())


def flight(client, pool, layout, start, trial, args, directory):
    from airsim_interface import AirSimDrone, require_airsim, vehicle_local_to_world_offset
    from config import load_calibration
    from control_adapter import send_control
    from environment import Gate
    from gate_visualization import spawn_gate_sequences
    from .decision_dataset import atomic_json
    from .dynamic_probe_common import describe
    from .fast_convex_preprocess import FastReferenceWindow, encode
    from .fly_decision_dnn import StoredRoutePlanner
    from .probe_airsim_three import GATE, translate_layout
    from experiments.falcon_paper_fidelity_acado_v3_continuous.demo_acado_five_gates import _crossing_results

    airsim = require_airsim()
    trace, commands, error, scene, drone = [], [], None, None, None
    interrupted = infrastructure_error = False
    initial_match = False
    period = trial["period_ms"] / 1000.
    skipped = 0
    try:
        client.reset()
        client.simPause(True)
        offset = vehicle_local_to_world_offset(client, args.vehicle)
        local_layout = translate_layout(layout, -offset)
        planner = StoredRoutePlanner(local_layout)
        window = FastReferenceWindow(planner.trajectory, planner.config)
        gates = [Gate(g["index"], g["center"], g["orientation"]) for g in local_layout["gates"]]
        drone = AirSimDrone(client, args.vehicle)
        drone.enable_api_control(True)
        drone.arm()
        kin = airsim.KinematicsState()
        desired_p = np.array(start["position"]) - offset
        kin.position = airsim.Vector3r(*map(float, desired_p))
        q = start["quaternion"]
        kin.orientation = airsim.Quaternionr(float(q[1]), float(q[2]), float(q[3]), float(q[0]))
        kin.linear_velocity = airsim.Vector3r(*map(float, start["velocity"]))
        kin.linear_acceleration = airsim.Vector3r(0., 0., 0.)
        kin.angular_velocity = airsim.Vector3r(0., 0., 0.)
        kin.angular_acceleration = airsim.Vector3r(0., 0., 0.)
        client.simSetKinematics(kin, ignore_collision=True, vehicle_name=args.vehicle)
        drone.hover()
        scene = spawn_gate_sequences(client, {args.vehicle: gates}, width=4., height=4., thickness=.15,
            asset_name="1M_Cube", object_prefix="FalconValueGate_", vehicle_world_offsets={args.vehicle: offset})
        collision_stamp = client.simGetCollisionInfo(vehicle_name=args.vehicle).time_stamp
        raw = drone.get_raw_state()
        initial_match = bool(np.linalg.norm(raw.position-desired_p) < .05 and
            np.linalg.norm(raw.velocity-start["velocity"]) < .2 and
            abs(float(np.dot(raw.quaternion, q))) > .999)
        atomic_json(directory / "initial_state.json", {"requested_world": start, "offset": offset.tolist(),
            "actual_local_position": raw.position.tolist(), "actual_velocity": raw.velocity.tolist(),
            "actual_quaternion": raw.quaternion.tolist(), "matched": initial_match})
        if not initial_match:
            raise RuntimeError("initial placement did not match requested state")
        calibration = load_calibration()
        phase = start["reference_phase_s"]

        def observe(state, elapsed):
            refs = window.window(phase + elapsed)
            e = float(np.linalg.norm(state.position-refs[0].position))
            observation = {"t": elapsed, "position": state.position.tolist(), "velocity": state.velocity.tolist(),
                           "quaternion": state.quaternion.tolist(),
                           "reference_position": refs[0].position.tolist(),
                           "reference_velocity": refs[0].velocity.tolist(),
                           "reference_quaternion": refs[0].orientation.tolist(),
                           "reference_phase_s": phase + elapsed, "error_m": e,
                           "sim_ns": int(state.timestamp_ns or 0)}
            if trace and elapsed == trace[-1]["t"]:
                trace[-1] = observation
            else:
                trace.append(observation)
            if not all(np.isfinite(v).all() for v in (state.position, state.velocity, state.quaternion)):
                raise RuntimeError("nonfinite state")
            if state.position[2]+offset[2] > -.3 or e > 8. or np.linalg.norm(state.velocity) > 15.:
                raise RuntimeError("flight guard exceeded")
            collision = client.simGetCollisionInfo(vehicle_name=args.vehicle)
            if collision.has_collided and collision.time_stamp != collision_stamp:
                raise RuntimeError(f"collision with {collision.object_name}")
            return refs

        observe(raw, 0.)
        client.simPause(False)
        origin = time.perf_counter()
        slot, next_observation, pending = 0, 0., None
        while True:
            now = time.perf_counter()
            elapsed = now-origin
            if elapsed >= args.duration:
                observe(drone.get_raw_state(), time.perf_counter()-origin)
                break
            # An observation clock independent of T avoids silently averaging
            # many more observations for fast controllers. Actual gaps are saved.
            if elapsed >= next_observation or (pending is None and elapsed >= slot*period):
                sample_start = time.perf_counter()
                raw = drone.get_raw_state()
                refs = observe(raw, sample_start-origin)
                next_observation = float((np.floor((time.perf_counter()-origin)/.02)+1)*.02)
                if pending is None and time.perf_counter()-origin >= slot*period:
                    row = {"slot": slot, "release_s": slot*period, "sample_s": sample_start-origin}
                    commands.append(row)
                    action, timing = pool.predict(trial["model"], encode(raw, refs))
                    ready = time.perf_counter()
                    row.update(timing, inference_ready_s=ready-origin,
                               delay_due_s=due_time(ready, trial["extra_delay_ms"])-origin)
                    pending = (action, row)
            if pending is not None and time.perf_counter()-origin >= pending[1]["delay_due_s"]:
                action, row = pending
                clipped = np.clip(action, [2., -3., -3., -2.], [20., 3., 3., 2.])
                dispatch = time.perf_counter()-origin
                if dispatch >= row["release_s"]+period:
                    row["deadline_missed"] = True
                    raise RuntimeError("response exceeded committed T; output NOT applied")
                # Commands persist across injection gaps, and are overwritten at
                # the next dispatch. A finite lease bounds hold if the host dies.
                send_control(drone, clipped, calibration, duration=1.)
                client.ping()  # Pump async submission; this is NOT actuator acknowledgement.
                end = time.perf_counter()-origin
                row.update(dispatch_s=end, sample_to_dispatch_ms=(end-row["sample_s"])*1000,
                           release_to_dispatch_ms=(end-row["release_s"])*1000,
                           deadline_missed=bool(end >= row["release_s"]+period),
                           control=clipped.tolist(), clipped=bool(np.any(clipped != action)))
                if row["deadline_missed"]:
                    raise RuntimeError("dispatch completed after committed T")
                new_slot = max(slot+1, next_slot(0., end, period))
                skipped += new_slot-slot-1
                slot, pending = new_slot, None
            targets = [origin+args.duration, origin+next_observation]
            targets.append(origin+pending[1]["delay_due_s"] if pending else origin+slot*period)
            time.sleep(max(0., min(targets)-time.perf_counter()))
    except KeyboardInterrupt:
        interrupted, error = True, "KeyboardInterrupt"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        infrastructure_error = type(exc).__module__.startswith(("msgpackrpc", "tornado"))
    finally:
        try:
            client.simPause(False)
            if drone is not None:
                drone.hover()
                client.ping()
            if scene is not None:
                scene.destroy()
        except Exception as exc:
            error = f"{error or ''}; cleanup: {exc}"
            infrastructure_error = True
    metrics = tracking_metrics(trace, args.duration)
    ratio = None
    if len(trace) > 1 and trace[-1]["t"] > 0:
        ratio = (trace[-1]["sim_ns"]-trace[0]["sim_ns"])/1e9/trace[-1]["t"]
    crossings = []
    if len(trace) > 1 and "gates" in locals():
        crossings = [{"gate": g, "crossed": c is not None, "valid": valid, "speed_mps": speed}
            for g, c, speed, valid in _crossing_results([r["position"] for r in trace],
                [r["velocity"] for r in trace], SimpleNamespace(gates=gates), GATE)]
    invalid_crossings = sum(c["crossed"] and not c["valid"] for c in crossings)
    comparable = bool(error is None and not invalid_crossings and initial_match and metrics["window_covered"] and
                      ratio is not None and .95 <= ratio <= 1.05 and skipped == 0 and
                      metrics["max_observation_gap_ms"] <= 100.)
    report = dict(trial=trial, status="interrupted" if interrupted or infrastructure_error else "finished", error=error,
        infrastructure_error=infrastructure_error,
        comparable=comparable, initial_state_matched=initial_match, sim_wall_ratio=ratio,
        skipped_releases=skipped, **metrics, crossings=crossings,
        short_window_not_five_gate_acceptance=True,
        response_ms=describe([r["sample_to_dispatch_ms"] for r in commands if "dispatch_s" in r]),
        deadline_misses=sum(r.get("deadline_missed", False) for r in commands),
        clipped_commands=sum(r.get("clipped", False) for r in commands))
    atomic_json(directory / "trace.json", trace)
    atomic_json(directory / "commands.json", commands)
    atomic_json(directory / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--periods-ms", nargs="+", type=float, default=[50., 100.])
    parser.add_argument("--extra-delays-ms", nargs="+", type=float, default=[0., 20.])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--duration", type=float, default=4.)
    parser.add_argument("--environment-id", type=int, choices=range(48, 54), default=48)
    parser.add_argument("--route-index", type=int, choices=range(20), default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=41451)
    parser.add_argument("--vehicle", default="Drone1")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results/control_value_237d_v1")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if not np.isfinite(args.duration) or args.duration < 1.:
        parser.error("duration must be finite and at least one second")
    for items in (args.models, args.scenarios, args.periods_ms, args.extra_delays_ms):
        if len(items) != len(set(items)):
            parser.error("duplicate matrix values")
    trials = trial_matrix(args.models, args.scenarios, args.periods_ms, args.extra_delays_ms, args.repeats, args.seed)
    if args.plan_only:
        print(json.dumps({"trials": len(trials), "flight_minutes_without_setup": len(trials)*args.duration/60,
                          "output": str(args.output.resolve()), "models": args.models}, indent=2))
        return
    from config import load_calibration
    from .decision_dataset import atomic_json, sha256
    from .dynamic_probe_common import ModelPool, provenance, read_json
    from .probe_airsim_three import load_layout
    from .fly_convex_policy import load_airsim_client
    from .fly_decision_dnn import StoredRoutePlanner
    from .fast_convex_preprocess import FastReferenceWindow
    from airsim_interface import require_airsim

    evidence = provenance(args.models)
    layout, route_hash = load_layout(args.environment_id, args.route_index)
    planner = StoredRoutePlanner(layout)
    window = FastReferenceWindow(planner.trajectory, planner.config)
    starts = {s: make_start(window, planner.trajectory.duration, s) for s in args.scenarios}
    pool = ModelPool(args.models, args.device)
    runtime = pool.runtime_info()
    runtime = {k: v for k, v in runtime.items() if not k.startswith("process_cuda_")}
    setup = {"version": 1, "provenance": evidence, "route_sha256": route_hash,
        "source_sha256": sha256(Path(__file__)), "runtime": runtime,
        "dependency_sha256": {name: sha256(Path(__file__).parent / name) for name in
            ("fast_convex_preprocess.py", "fly_decision_dnn.py", "dynamic_probe_common.py")},
        "adapter_sha256": sha256(Path(__file__).parent.parent / "control_adapter.py"),
        "calibration": asdict(load_calibration()), "starts": starts, "trials": trials,
        "duration": args.duration, "host": args.host, "port": args.port, "vehicle": args.vehicle,
        "extra_delay_semantics": "added AFTER real inference; simulator continues; previous command held",
        "reference_clock": "fixed initial phase + monotonic wall time; no gate-clock reset",
        "state_trace_contract": {"psi_order": "position[3], velocity[3], quaternion[4]",
            "coordinate_frame": "vehicle-local NED; actual and reference share the same frame",
            "quaternion_order": "w,x,y,z; normalize and align signs before computing error",
            "metric_weights": "not inferred; declare Wx and aggregation before comparisons",
            "legacy_traces": "missing velocity/orientation reference fields cannot be recovered"},
        "observation_period_ms": 20., "command_lease_ms": 1000., "test_read": False,
        "state_pairing_limit": "matched observable initial kinematics, not a bit-exact snapshot of hidden physics state",
        "timestamp_limit": "host state-request/async dispatch proxies, not actuator timestamps"}
    args.output.mkdir(parents=True, exist_ok=True)
    setup_path = args.output / "setup.json"
    if setup_path.exists():
        check_resume(read_json(setup_path), setup, args.resume)
    else:
        atomic_json(setup_path, setup)
    for model in args.models:
        for _ in range(20):
            pool.predict(model, pool.norm["x_mean"])
    load_airsim_client()
    client = require_airsim().MultirotorClient(ip=args.host, port=args.port, timeout_value=3.)
    client.confirmConnection()
    if args.vehicle not in client.listVehicles():
        raise ValueError("vehicle missing from manual Play session")
    foreign = [n for n in client.simListSceneObjects("Falcon.*Gate.*") if not n.startswith("FalconValueGate_")]
    if foreign:
        raise RuntimeError("Stop/Play to clear previous generated gates; foreign gates will not be deleted")
    reports = []
    for number, trial in enumerate(trials, 1):
        directory = args.output / trial["id"]
        reports_path = sorted(directory.glob("attempt_*/report.json"))
        previous = read_json(reports_path[-1]) if reports_path else None
        if previous and previous["status"] == "finished":
            reports.append(previous)  # Failed trials are retained, never silently retried to success.
            continue
        attempt = directory / f"attempt_{len(list(directory.glob('attempt_*')))+1:03d}"
        attempt.mkdir(parents=True, exist_ok=False)
        print(f"[{number}/{len(trials)}] {trial['id']}", flush=True)
        report = flight(client, pool, layout, starts[trial["scenario"]], trial, args, attempt)
        if report["status"] == "finished":
            reports.append(report)
        save_summary(args.output, reports, args.models[0])
        print(f"  comparable={report['comparable']} MSE={report.get('tracking_mse_m2')} error={report['error']}", flush=True)
        if report["status"] == "interrupted":
            raise SystemExit(1 if report["infrastructure_error"] else 130)
        if report["error"] and (not report["initial_state_matched"] or not (attempt / "trace.json").exists()):
            raise RuntimeError("setup failed; results preserved; fix the simulator before continuing")
    save_summary(args.output, reports, args.models[0])
    print(f"Summary: {args.output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
