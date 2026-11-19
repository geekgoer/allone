"""Independent periodic replay clients sharing one resident 237D model pool (not flight)."""

import argparse
import threading
import time

import numpy as np

from .dynamic_probe_common import (DATA, RUNS, DEFAULT_MODELS, InferenceService, ModelPool, describe,
                                   new_output, provenance, release_schedule, save_timing)


def run(args):
    names = list(dict.fromkeys(getattr(args, "models", DEFAULT_MODELS)))
    if args.model != "cycle" and args.model not in names:
        names.append(args.model)
    evidence = provenance(names, validation=True)
    with np.load(DATA / "validation.npz", allow_pickle=False) as data:
        x = data["X"].astype(np.float32)
        ids = data["row_id"]
    path = new_output("replay")
    pool = ModelPool(names, args.device)
    def warmup():
        for name in names:
            for _ in range(20):
                pool.predict(name, x[0])
    service = InferenceService(pool.predict, warmup)
    period, stop = args.period_ms / 1000, threading.Event()
    start = time.perf_counter()+.5
    failures = []
    order = np.random.default_rng(20261004).permutation(len(x))
    def producer(index):
        name = f"Drone{index+1}"
        phase = index*period/args.clients if args.phase == "staggered" else 0.
        try:
            for k, scheduled in release_schedule(start, args.duration, period, stop, service, name, phase):
                # Exactly reproducible states/choices, independent of which requests finished.
                row_index = order[(k*args.clients+index) % len(x)]
                model = args.model if args.model != "cycle" else names[(k+index) % len(names)]
                row = {"vehicle": name, "cycle": k, "scheduled_s": scheduled,
                       "sample_start_s": time.perf_counter(), "row_id": int(ids[row_index])}
                service.submit(model, x[row_index], row)
        except BaseException as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            stop.set()
    threads = [threading.Thread(target=producer, args=(i,)) for i in range(args.clients)]
    try:
        for thread in threads:
            thread.start()
        while any(t.is_alive() for t in threads):
            time.sleep(.1)
    except KeyboardInterrupt:
        failures.append("interrupted")
        stop.set()
    finally:
        for thread in threads:
            if thread.ident is not None:
                thread.join()
        service.close()
    ready = [r for r in service.rows if r.get("status") == "ready"]
    report = save_timing(path, service.rows, args.deadline_ms, {
        "status": "INCOMPLETE" if failures else "COMPLETE", "errors": failures,
        "scope": "open-loop periodic inference clients, not N-drone flight or closed-loop success",
        "provenance": evidence, "device": args.device, "phase": args.phase,
        "runtime": pool.runtime_info(), "input_order_seed": 20261004,
        "input_order_version": 2, "resident_models": names,
        "clients": args.clients, "period_ms": args.period_ms, "duration_seconds": args.duration,
        "planned_releases": int(np.ceil(args.duration/period))*args.clients,
        "resident_model_count": len(pool.models), "resident_parameter_bytes": pool.parameter_bytes,
        "request_selection": args.model, "selection_is_not_state_adaptive": True,
        "per_model_service_ms": {n: describe([r["service_ms"] for r in ready if r["model"] == n]) for n in names},
        "training_interference_excluded": "operator must verify no training or other GPU workload",
    })
    print(f"ready={report['timing']['ready']} misses={report['timing']['output_deadline_misses']} "
          f"of released={report['timing']['released']}", flush=True)
    if failures or report["timing"]["inference_errors"]:
        raise SystemExit(1)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--clients", type=int, choices=range(1, 65), default=3)
    parser.add_argument("--model", choices=(*RUNS, "cycle"), default="residual21x1024")
    parser.add_argument("--models", nargs="+", choices=tuple(RUNS), default=list(DEFAULT_MODELS),
                        help="Resident pool / cycle order; fixed --model is appended if missing")
    parser.add_argument("--phase", choices=("sync", "staggered"), default="sync")
    parser.add_argument("--duration", type=float, default=60.)
    parser.add_argument("--period-ms", type=float, default=50.)
    parser.add_argument("--deadline-ms", type=float, default=50.)
    args = parser.parse_args()
    if not all(np.isfinite(v) and v > 0 for v in (args.duration, args.period_ms, args.deadline_ms)):
        parser.error("duration, period and deadline must be finite and positive")
    run(args)


if __name__ == "__main__":
    main()
