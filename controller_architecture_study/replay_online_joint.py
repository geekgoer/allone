"""Shared inference: fixed, period-only and joint feedback policies.

No AirSim or training. Completed arms are resumable. Period changes preserve the
next committed release; subsequent releases use the new T. Each request keeps
its own D=T even after a later policy change. Late outputs are counted/discarded.
"""

import argparse
import json
from pathlib import Path
import threading
import time

import numpy as np

from .decision_dataset import atomic_json, sha256
from .dynamic_probe_common import (DATA, DEFAULT_MODELS, OUTPUT, InferenceService,
                                  ModelPool, describe, read_json)
from .joint_controller_sweep import RUNTIME_KEYS
from .online_joint_policy import DEFAULT_SUMMARY, ResourceSelector, load_profile


DEFAULT_OUTPUT = OUTPUT / "online_joint_pool_v2"


def trial_cases(rounds):
    modes = ("fixed", "period_only", "adaptive")
    return [(r+1, mode) for r in range(rounds)
            for mode in (modes if r % 2 == 0 else modes[::-1])]


def summarize(rows):
    ready = [r for r in rows if r.get("status") == "ready"]
    timely = [r for r in ready if r["ready_s"] <= r["scheduled_s"]+r["deadline_ms"]/1000]
    common_misses = {str(limit): len(rows)-sum(r["ready_s"] <= r["scheduled_s"]+limit/1000
                     for r in ready) for limit in (50, 75, 100)}
    return {"released": len(rows), "ready": len(ready), "timely_outputs": len(timely),
            "deadline_misses": len(rows)-len(timely),
            "miss_fraction": (len(rows)-len(timely))/len(rows) if rows else None,
            "dropped_busy": sum(r.get("status") == "dropped_busy" for r in rows),
            "skipped_release": sum(r.get("status") == "skipped_release" for r in rows),
            "inference_errors": sum(r.get("status") == "inference_error" for r in rows),
            "common_response_limit_misses": common_misses,
            "common_response_limit_miss_fraction": {k: v/len(rows) if rows else None
                                                     for k, v in common_misses.items()},
            "response_ms": describe([(r["ready_s"]-r["scheduled_s"])*1000 for r in ready]),
            "queue_ms": describe([(r["start_s"]-r["enqueue_s"])*1000 for r in ready]),
            "service_ms": describe([r["service_ms"] for r in ready]),
            "period_counts": {str(t): sum(r["period_ms"] == t for r in rows)
                              for t in sorted({r["period_ms"] for r in rows})},
            "model_counts": {m: sum(r["model"] == m for r in rows)
                             for m in sorted({r["model"] for r in rows})}}


def active_intervals(client, counts, stage_seconds, end):
    intervals = []
    for stage, count in enumerate(counts):
        a, b = stage*stage_seconds, min((stage+1)*stage_seconds, end)
        if client < count and a < b:
            if intervals and abs(intervals[-1][1]-a) < 1e-9:
                intervals[-1][1] = b
            else:
                intervals.append([a, b])
    return intervals


def output_age(rows, start, counts, stage_seconds, duration, *, vehicle_names=None):
    """Age of timely delivered sampled inputs, while each client is active.

    Before its first usable output, time is recorded as unserved (never zero age).
    Request/sample timestamps are host times, not physical sensor acquisition.
    """
    area = served = active = unserved = maximum = 0.
    names = list(vehicle_names) if vehicle_names is not None else [f"client{i}" for i in range(max(counts))]
    if len(names) != max(counts):
        raise ValueError("vehicle names must match the maximum active client count")
    for client in range(max(counts)):
        events = sorted((r for r in rows if r["vehicle"] == names[client]
                        and r.get("status") == "ready"
                        and r["ready_s"] <= r["scheduled_s"]+r["deadline_ms"]/1000),
                        key=lambda r: r["ready_s"])
        for a, b in active_intervals(client, counts, stage_seconds, duration):
            active += b-a
            last_sample, last_time = None, a
            for row in events:
                stamp, ready = row["sample_start_s"]-start, row["ready_s"]-start
                if stamp < a or ready < a or ready > b or (last_sample is not None and stamp <= last_sample):
                    continue
                if last_sample is None:
                    unserved += ready-last_time
                else:
                    area += ((last_time-last_sample)+(ready-last_sample))*(ready-last_time)/2
                    served += ready-last_time
                    maximum = max(maximum, ready-last_sample)
                last_sample, last_time = stamp, ready
            if last_sample is None:
                unserved += b-a
            else:
                area += ((last_time-last_sample)+(b-last_sample))*(b-last_time)/2
                served += b-last_time
                maximum = max(maximum, b-last_sample)
    return {"mean_age_when_served_ms": 1000*area/served if served else None,
            "max_age_when_served_ms": 1000*maximum if served else None,
            "unserved_fraction": unserved/active if active else None,
            "active_client_seconds": active}


def run_arm(args, profile, pool, x, ids, order, mode, path):
    path.mkdir(parents=True, exist_ok=False)
    selector = ResourceSelector(profile, utilization_limit=args.utilization_limit,
                                dwell_seconds=args.dwell_seconds)
    service = None
    stop = threading.Event()
    decisions, errors = [], []
    next_release, cycles = {}, {}
    counts, stage_seconds = args.clients, args.stage_seconds
    duration = len(counts)*stage_seconds
    start = end = None
    metadata = {"schema_version": 2, "mode": mode,
                "scope": "dynamic-client open-loop inference replay; no aircraft controlled",
                "clients_by_stage": counts, "stage_seconds": stage_seconds,
                "planned_duration_seconds": duration, "runtime": pool.runtime_info(),
                "test_read": False, "teacher_labels_read": False,
                "quality_prior": profile["quality_scope"], "profile_sha256": profile["summary_sha256"],
                "resident_models": list(DEFAULT_MODELS), "parameter_bytes": pool.parameter_bytes,
                "input_stream": "seed=20261004, 50ms logical ticks, row index=(tick*max_clients+client) modulo validation size",
                "period_change": "keep next committed release; new T thereafter; existing request D never changes",
                "timing_scope": "host release to returned inference output; excludes AirSim RPC/reference/actuation",
                "feedback": "bounded last-2s completion window, capacity=4096; full history only archived",
                "limitations": ["Route quality prior is from one three-drone route, not measured control quality at N clients.",
                    "P95 service/load thresholds are heuristic resource estimates, not schedulability proofs.",
                    "Only load feedback implemented; no state-dependent T_max or state-aware model benefit yet.",
                    "Late requests still consume compute but outputs are discarded by the replay consumer.",
                    "Different T observes different subsets of the same logical input stream.",
                    "A central producer represents resource requests, not N full AirSim preprocessing threads."]}
    atomic_json(path/"setup.json", metadata)
    try:
        def warmup():
            for name in DEFAULT_MODELS:
                for _ in range(20):
                    pool.predict(name, pool.norm["x_mean"])
        service = InferenceService(pool.predict, warmup, feedback=True)
        start = time.perf_counter()
        next_decision, next_stage, stage = start, start, -1
        chosen = None
        with (path/"decisions.jsonl").open("w", encoding="utf-8") as log:
            while True:
                # Process scheduled events chronologically even after a host
                # scheduling delay. Departures must not erase earlier missed slots.
                event = min(next_stage, next_decision, min(next_release.values(), default=float("inf")))
                if event >= start+duration:
                    stop.wait(max(0., start+duration-time.perf_counter()))
                    break
                stop.wait(max(0., event-time.perf_counter()))
                changed_stage = event == next_stage
                if changed_stage:
                    stage += 1
                    next_stage = start+(stage+1)*stage_seconds
                n = counts[stage]
                if changed_stage or event == next_decision:
                    decision_start = time.perf_counter()
                    snapshot = service.feedback_snapshot(decision_start)
                    if snapshot["inference_error"]:
                        raise RuntimeError("inference error; inspect requests.jsonl")
                    chosen = selector.select(n, decision_start, snapshot, mode)
                    chosen.update(elapsed_seconds=decision_start-start, stage=stage,
                                  decision_ms=(time.perf_counter()-decision_start)*1000)
                    decisions.append(chosen)
                    log.write(json.dumps(chosen, allow_nan=False)+"\n")
                    print(f"{mode} t={decision_start-start:5.1f}s N={n:2d} {chosen['case']} "
                          f"estimated_feasible={chosen['estimated_feasible']} "
                          f"load={chosen['estimated_load']:.2f} pending={chosen['pending_service_estimate_ms']:.1f}ms", flush=True)
                    chosen["producer_pause_ms"] = (time.perf_counter()-decision_start)*1000
                    next_decision = decision_start+args.decision_seconds
                if changed_stage:
                    next_release = {i: t for i, t in next_release.items() if i < n}
                    for i in range(n):
                        if i not in next_release:
                            next_release[i] = event+i*chosen["period_ms"]/1000/n
                            cycles.setdefault(i, 0)
                    continue
                if event < min(next_release.values()):
                    continue
                # Preserve already scheduled arrivals across a T update, with no
                # catch-up inference bursts. Late slots remain in the denominator.
                i = min(next_release, key=next_release.get)
                scheduled = next_release[i]
                sample = time.perf_counter()
                period_ms = chosen["period_ms"]
                row = {"vehicle": f"client{i}", "cycle": cycles[i], "stage": stage,
                       "scheduled_s": scheduled, "period_ms": period_ms, "deadline_ms": period_ms,
                       "decision_index": len(decisions)-1, "model": chosen["model"]}
                if sample >= scheduled+period_ms/1000:
                    row["status"] = "skipped_release"
                    with service.lock:
                        service.rows.append(row)
                else:
                    tick = max(0, int(np.floor((scheduled-start)/.05+1e-8)))
                    index = order[(tick*max(counts)+i) % len(order)]
                    row.update(sample_start_s=sample, row_id=int(ids[index]), logical_tick=tick)
                    service.submit(chosen["model"], x[index], row)
                cycles[i] += 1
                next_release[i] = scheduled+period_ms/1000
        end = start+duration
    except BaseException as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        end = time.perf_counter()
    finally:
        if service is not None:
            service.close()
        rows = service.rows if service else []
        if any(r.get("status") == "inference_error" for r in rows) and not errors:
            errors.append("inference error in final requests")
        with (path/"requests.jsonl").open("w", encoding="utf-8") as log:
            for row in sorted(rows, key=lambda r: (r["scheduled_s"], r["vehicle"])):
                log.write(json.dumps(row, allow_nan=False)+"\n")
        # Include total host pause (statistics + selection + logging/printing).
        with (path/"decisions.jsonl").open("w", encoding="utf-8") as log:
            for decision in decisions:
                log.write(json.dumps(decision, allow_nan=False)+"\n")
        measured = min(duration, max(0., end-start)) if start is not None and end is not None else 0.
        report = {**metadata, "status": "INCOMPLETE" if errors else "COMPLETE", "errors": errors,
                  "replay_start_s": start, "measurement_end_s": end,
                  "measured_duration_seconds": measured, "timing": summarize(rows),
                  "per_stage": [{"stage": k, "active_clients": count,
                      **summarize([r for r in rows if r["stage"] == k])} for k, count in enumerate(counts)],
                  "output_age": output_age(rows, start, counts, stage_seconds, measured) if start else None,
                  "decision_ms": describe([d["decision_ms"] for d in decisions]),
                  "producer_pause_ms": describe([d["producer_pause_ms"] for d in decisions]),
                  "switches": sum(d["changed"] for d in decisions),
                  "infeasible_decisions": sum(not d["estimated_feasible"] for d in decisions),
                  "model_switches": sum(a["model"] != b["model"] for a, b in zip(decisions, decisions[1:])),
                  "period_switches": sum(a["period_ms"] != b["period_ms"] for a, b in zip(decisions, decisions[1:]))}
        atomic_json(path/"report.json", report)
        print(f"Report: {path/'report.json'}", flush=True)
    if errors:
        raise RuntimeError(errors[0])
    return report


def run(args):
    profile = load_profile(args.summary)
    if args.device != profile["device"]:
        raise ValueError("use the same device type as the measured flight profile")
    if args.preflight:
        print(json.dumps({"status": "PASS", "candidate_count": len(profile["candidates"]),
                          "static_candidate": ResourceSelector(profile).static,
                          "initial_service_p95_ms": profile["initial_service_p95_ms"],
                          "gpu_inference_started": False, "airsim_started": False}, indent=2))
        return 0
    config = {k: getattr(args, k) for k in ("device", "clients", "stage_seconds", "rounds",
                                          "decision_seconds", "utilization_limit", "dwell_seconds")}
    sources = {n: sha256(Path(__file__).with_name(n)) for n in
               ("online_joint_policy.py", "replay_online_joint.py", "dynamic_probe_common.py")}
    path = args.output_dir.resolve()
    if (path/"summary.json").exists():
        if not args.resume:
            raise ValueError("output exists; use --resume or a new --output-dir")
        suite = read_json(path/"summary.json")
        if suite["config"] != config or suite["profile"] != profile or suite["sources"] != sources:
            raise ValueError("resume inputs/code changed; choose a new output directory")
        for trial in suite["trials"]:
            for name, expected in trial["artifact_sha256"].items():
                if sha256(Path(trial["report"]).parent/name) != expected:
                    raise ValueError("completed replay artifact changed")
    else:
        if path.exists() and any(path.iterdir()):
            raise ValueError("nonempty output directory without manifest")
        path.mkdir(parents=True, exist_ok=True)
        suite = {"schema_version": 2, "status": "INCOMPLETE", "config": config, "profile": profile,
                 "sources": sources, "trials": [], "test_read": False}
    atomic_json(path/"summary.json", suite)
    # Forward/reverse order; identical underlying feature stream and load stages.
    cases = trial_cases(args.rounds)
    pool = None
    with np.load(DATA/"validation.npz", allow_pickle=False) as data:
        x, ids = data["X"].astype(np.float32), data["row_id"]
    order = np.random.default_rng(20261004).permutation(len(x))
    try:
        for repeat, mode in cases:
            case = f"round{repeat}_{mode}"
            if any(r["case"] == case for r in suite["trials"]):
                print(f"Reuse {case}", flush=True)
                continue
            folder = path/case
            attempts = sorted(folder.glob("attempt_*"))
            prior = attempts[-1]/"report.json" if attempts else None
            report = read_json(prior) if prior and prior.is_file() else None
            if report is None or report["status"] != "COMPLETE":
                if report and not any("KeyboardInterrupt" in e for e in report["errors"]):
                    raise RuntimeError("previous arm has an inference/setup error; inspect it before retrying")
                if pool is None:
                    pool = ModelPool(DEFAULT_MODELS, args.device)
                    runtime_info = pool.runtime_info()
                    runtime = {k: runtime_info.get(k) for k in RUNTIME_KEYS}
                    if runtime != {k: profile["runtime"].get(k) for k in RUNTIME_KEYS}:
                        raise ValueError("runtime/device/precision differs from flight profile")
                prior = folder/f"attempt_{len(attempts)+1:03d}"/"report.json"
                report = run_arm(args, profile, pool, x, ids, order, mode, prior.parent)
            suite["trials"].append({"case": case, "round": repeat, "mode": mode,
                "report": str(prior), "artifact_sha256": {n: sha256(prior.parent/n) for n in
                    ("report.json", "requests.jsonl", "decisions.jsonl")},
                "timing": report["timing"], "output_age": report["output_age"],
                "switches": report["switches"], "model_switches": report["model_switches"],
                "period_switches": report["period_switches"], "decision_ms": report["decision_ms"]})
            suite["trials"][-1]["producer_pause_ms"] = report["producer_pause_ms"]
            atomic_json(path/"summary.json", suite)
        suite["status"] = "COMPLETE"
    finally:
        atomic_json(path/"summary.json", suite)
        print(f"Summary: {path/'summary.json'}", flush=True)
    for row in suite["trials"]:
        t = row["timing"]
        age = row["output_age"]["mean_age_when_served_ms"]
        print(f"{row['case']:17s} misses={t['deadline_misses']}/{t['released']} "
              f"model_switches={row['model_switches']} period_switches={row['period_switches']} "
              f"mean_output_age_ms={age} common50_misses={t['common_response_limit_misses']['50']} "
              f"max_producer_pause_ms={row['producer_pause_ms']['max_observed']}", flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--clients", type=int, nargs="+", default=[3, 8, 16, 32, 8, 3])
    parser.add_argument("--stage-seconds", type=float, default=6.)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--decision-seconds", type=float, default=1.)
    parser.add_argument("--utilization-limit", type=float, default=.75)
    parser.add_argument("--dwell-seconds", type=float, default=2.)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight", action="store_true", help="Validate existing files only; no inference or AirSim")
    args = parser.parse_args()
    if (any(n < 1 or n > 64 for n in args.clients) or args.rounds < 1
            or not 0 < args.utilization_limit <= 1
            or not all(np.isfinite(v) and v > 0 for v in
                       (args.stage_seconds, args.decision_seconds, args.dwell_seconds))):
        parser.error("clients 1..64; rounds and durations positive; utilization limit in (0,1]")
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
