"""User-run periodic-client capacity sweep; no training or AirSim launch.

Every completed trial and the partial suite summary survive interruption.
Deadline misses are experimental results, not reasons to skip harder cases.
"""

import argparse
import gc
import itertools
from types import SimpleNamespace

import numpy as np

from .decision_dataset import atomic_json
from .dynamic_probe_common import RUNS, new_output
from .probe_shared_response import run as replay


def trial_cases(clients, models, rounds):
    cases = list(itertools.product(clients, models, ("sync", "staggered")))
    for repeat in range(rounds):
        # Keep order deterministic; rotate repeated trials to expose order effects.
        offset = repeat % len(cases)
        for n, model, phase in cases[offset:] + cases[:offset]:
            yield repeat + 1, n, model, phase


def compact_result(report, repeat):
    t = report["timing"]
    return {"round": repeat, "clients": report["clients"], "model": report["request_selection"],
            "phase": report["phase"], "report": report["report_path"], "status": report["status"],
            "realtime_passed": report["realtime_passed"],
            **{key: t[key] for key in ("released", "ready", "output_deadline_misses",
                                       "output_deadline_miss_fraction", "skipped_release", "dropped_busy")},
            "service_mean_ms": t["service_ms"]["mean"] if t["service_ms"] else None,
            "response_p99_ms": t["release_to_output_ms"]["p99"] if t["release_to_output_ms"] else None,
            "worst_client_miss_fraction": max(
                v["output_deadline_miss_fraction"] for v in report["per_vehicle"].values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clients", type=int, nargs="+", default=[1, 3, 6, 7, 8, 12, 16])
    parser.add_argument("--models", nargs="+", choices=tuple(RUNS),
                        default=["plain6x1024", "residual33x1024"])
    parser.add_argument("--duration", type=float, default=15.)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--period-ms", type=float, default=50.)
    parser.add_argument("--deadline-ms", type=float, default=50.)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if (any(n < 1 or n > 64 for n in args.clients) or args.rounds < 1
            or not all(np.isfinite(v) and v > 0 for v in (args.duration, args.period_ms, args.deadline_ms))):
        parser.error("clients: 1..64; rounds positive; duration/period/deadline finite and positive")
    args.clients = list(dict.fromkeys(args.clients))
    args.models = list(dict.fromkeys(args.models))
    path = new_output("capacity")
    cases = list(trial_cases(args.clients, args.models, args.rounds))
    summary = {"status": "INCOMPLETE", "config": vars(args), "resident_models": list(RUNS),
               "scope": "periodic inference capacity pilot, not N-drone flight or a safety bound",
               "planned_trials": len(cases), "trials": [], "test_read": False,
               "limitations": ["No dynamic arrivals or online state-aware selector yet.",
                               "No interference-free claim without operator confirmation.",
                               "Observed misses at one load are not a certified capacity limit."]}
    print(f"Close training/AirSim. {len(cases)} serial trials, {len(cases)*args.duration/60:.1f} "
          "minutes measured time plus loading/warmup. No retraining.", flush=True)
    try:
        for index, (repeat, n, model, phase) in enumerate(cases, 1):
            summary["current_trial"] = dict(round=repeat, clients=n, model=model, phase=phase)
            atomic_json(path / "summary.json", summary)
            print(f"Trial {index}/{len(cases)}: round={repeat} N={n} {model} {phase}", flush=True)
            report = replay(SimpleNamespace(clients=n, model=model, models=list(RUNS), phase=phase,
                            duration=args.duration, period_ms=args.period_ms,
                            deadline_ms=args.deadline_ms, device=args.device))
            summary["trials"].append(compact_result(report, repeat))
            atomic_json(path / "summary.json", summary)
            gc.collect()
        summary["status"] = "COMPLETE"
        print("N  model             phase       ready/released  misses  response_p99_ms", flush=True)
        for row in summary["trials"]:
            p99 = row["response_p99_ms"]
            p99_text = "n/a" if p99 is None else f"{p99:.3f}"
            print(f"{row['clients']:2d} {row['model']:17s} {row['phase']:10s} "
                  f"{row['ready']:5d}/{row['released']:<5d} {row['output_deadline_misses']:6d} "
                  f"{p99_text:>16s}", flush=True)
    finally:
        atomic_json(path / "summary.json", summary)
        print(f"Summary: {path / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
