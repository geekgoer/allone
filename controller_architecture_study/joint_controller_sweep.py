"""User-run model/period pilot on the existing manual-Play three-drone harness.

No training, Test data, FPGA claims or unmeasured state-dependent safety bounds.
All drones use the same candidate; this is a static joint baseline, not online
adaptation or the complete FALCON hardware/priority co-design optimizer.
"""

import argparse
import itertools
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from .decision_dataset import ROOT, atomic_json, sha256
from .dynamic_probe_common import DEFAULT_MODELS, OUTPUT, provenance, read_json
from .probe_airsim_three import VEHICLES, load_layout


DEFAULT_OUTPUT = OUTPUT / "joint_model_period_v1"
RUNTIME_KEYS = ("platform", "torch", "torch_cuda", "gpu", "cpu_threads", "dtype",
                "autocast", "tf32_matmul_allowed")
DEADLINE_ERRORS = ("RuntimeError: late DNN output not applied",
                   "RuntimeError: DNN deadline expired;")


def case_key(model, period_ms):
    return f"{model}_T{period_ms:g}"


def time_mean_squared_error(trace, start, duration):
    """Integrate squared position error over the SAME reference-clock interval.

    Endpoint interpolation holds the first value back to time zero (the three
    producers have slightly different first sample times). The last measurement
    must cover the interval; never extend an aborted flight through its tail.
    """
    if len(trace) < 2:
        return None
    t = np.array([row["sample_s"] - start for row in trace], dtype=float)
    e2 = np.array([row["tracking_error_m"]**2 for row in trace], dtype=float)
    if (not np.isfinite(t).all() or not np.isfinite(e2).all()
            or np.any(np.diff(t) <= 0) or t[-1] < duration):
        return None
    grid = np.concatenate(([0.], t[(t > 0) & (t < duration)], [duration]))
    values = np.interp(grid, t, e2)
    return float(np.sum(np.diff(grid) * (values[:-1] + values[1:]) / 2) / duration)


def check_report(report, summary):
    cfg, evidence = summary["config"], summary["provenance"]
    expected = {"mode": "dnn", "device": cfg["device"], "phase": cfg["phase"],
                "environment_id": cfg["environment_id"], "route_index": cfg["route_index"],
                "preprocess_policy": "vectorized", "saved_route_sha256": summary["saved_route_sha256"],
                "preprocess_source_sha256": summary["source_sha256"]["fast_convex_preprocess.py"]}
    if any(report.get(k) != v for k, v in expected.items()):
        raise ValueError("flight report configuration/route/preprocess mismatch")
    if report["model_selection"] not in cfg["models"] or report["period_ms"] not in cfg["periods_ms"]:
        raise ValueError("report is not a requested model/period candidate")
    if report["deadline_ms"] != report["period_ms"]:
        raise ValueError("joint pilot requires D=T, not a deadline-only relaxation")
    actual = report["provenance"]
    for key in ("input_contract", "normalization_sha256", "validation_sha256"):
        if actual.get(key) != evidence[key]:
            raise ValueError(f"report {key} mismatch")
    if set(report["resident_models"]) != set(DEFAULT_MODELS):
        raise ValueError("resident model pool changed between candidates")
    for name in DEFAULT_MODELS:
        for key in ("checkpoint_sha256", "config_sha256"):
            if actual["models"][name][key] != evidence["models"][name][key]:
                raise ValueError(f"report weights/config changed: {name}")
    runtime = {key: report["runtime"].get(key) for key in RUNTIME_KEYS}
    if "runtime" in summary and runtime != summary["runtime"]:
        raise ValueError("runtime/device/precision changed between candidates")
    summary.setdefault("runtime", runtime)


def compact_result(report_path, summary):
    report_path = report_path.resolve()
    report = read_json(report_path)
    check_report(report, summary)
    folder = report_path.parent
    traces = read_json(folder / "flight_trace.json")
    with (folder / "requests.jsonl").open(encoding="utf-8") as handle:
        requests = [json.loads(line) for line in handle if line.strip()]
    period = report["period_ms"] / 1000
    clocks = [r["scheduled_s"] - r["cycle"]*period
              - (VEHICLES.index(r["vehicle"])*period/3 if report["phase"] == "staggered" else 0.)
              for r in requests]
    if clocks and max(clocks)-min(clocks) > 1e-6:
        raise ValueError("requests do not share the saved global trajectory clock")
    costs = {v: time_mean_squared_error(traces.get(v, []), min(clocks), summary["route_duration_seconds"])
             if clocks else None for v in VEHICLES}
    mse = float(np.mean(list(costs.values()))) if all(x is not None for x in costs.values()) else None
    timing = report["timing"]
    ratios = [report["flight"].get(v, {}).get("sim_seconds_per_host_second") for v in VEHICLES]
    clock_ok = all(r is not None and .95 <= r <= 1.05 for r in ratios)
    eligible = bool(report["acceptance_passed"] and mse is not None and clock_ok)
    latency = timing["release_to_output_ms"]
    service = timing["service_ms"]
    errors = report["errors"]
    unsafe_failure = bool(errors and not all(any(token in e for token in DEADLINE_ERRORS) for e in errors))
    return {"case": case_key(report["model_selection"], report["period_ms"]),
            "model": report["model_selection"], "period_ms": report["period_ms"],
            "deadline_ms": report["deadline_ms"], "report": str(report_path),
            "artifact_sha256": {name: sha256(folder/name) for name in
                                ("report.json", "flight_trace.json", "requests.jsonl")},
            "status": report["status"], "errors": errors, "unsafe_failure": unsafe_failure,
            "interrupted": any("KeyboardInterrupt" in e for e in errors),
            "flight_passed": report["flight_passed"], "realtime_passed": report["realtime_passed"],
            "acceptance_passed": report["acceptance_passed"], "eligible": eligible,
            "crossings": {v: report["flight"].get(v, {}).get("crossings", 0) for v in VEHICLES},
            "clock_ratio_passed": clock_ok, "tracking_mse_m2": mse,
            "tracking_rmse_m": float(np.sqrt(mse)) if mse is not None else None,
            "tracking_mse_m2_per_vehicle": costs,
            "response_p99_ms": latency["p99"] if latency else None,
            "response_max_ms": latency["max_observed"] if latency else None,
            "mean_service_load_estimate": 3*service["mean"]/report["period_ms"] if service else None,
            "clipped_commands": sum(bool(r.get("clipped")) for r in requests),
            **{k: timing[k] for k in ("released", "ready", "dnn_commands_dispatched",
                                     "output_deadline_misses", "control_deadline_aborts")}}


def choose(rows, period_cap_ms=None):
    """Observed static winner only. The optional cap is supplied, not inferred."""
    feasible = [r for r in rows if r["eligible"]
                and (period_cap_ms is None or r["period_ms"] <= period_cap_ms)]
    return min(feasible, key=lambda r: (r["tracking_mse_m2"], r["response_p99_ms"])) if feasible else None


def selection(summary, period_cap_ms=None):
    best = choose(summary["trials"], period_cap_ms)
    return {"status": "OBSERVED_CANDIDATE" if best else "NO_OBSERVED_FEASIBLE_CANDIDATE",
            "period_cap_ms": period_cap_ms, "candidate": best,
            "coverage_complete": summary["status"] == "COMPLETE",
            "metric": summary["performance_proxy"], "scope": summary["config"],
            "not_an_online_policy": True, "not_a_safety_certificate": True,
            "note": "Single-route pilot ranking, not statistical superiority or a full FALCON optimum."}


def save(path, summary):
    atomic_json(path / "summary.json", summary)
    atomic_json(path / "selection.json", selection(summary))


def run(args):
    path = args.output_dir.resolve()
    layout, route_hash = load_layout(args.environment_id, args.route_index)
    cfg = {"models": args.models, "periods_ms": args.periods_ms, "device": args.device,
           "phase": args.phase, "environment_id": args.environment_id, "route_index": args.route_index,
           "host": args.host, "port": args.port}
    sources = {name: sha256(Path(__file__).with_name(name)) for name in
               ("probe_airsim_three.py", "dynamic_probe_common.py", "fast_convex_preprocess.py")}
    evidence = provenance(DEFAULT_MODELS)
    if (path / "summary.json").exists():
        if not args.resume:
            raise ValueError("suite exists; use --resume or a new --output-dir")
        summary = read_json(path / "summary.json")
        if (summary["config"] != cfg or summary["source_sha256"] != sources
                or summary["provenance"] != evidence or summary["saved_route_sha256"] != route_hash):
            raise ValueError("resume configuration/model/source changed; use a new output directory")
        for row in summary["trials"]:
            for name, expected in row["artifact_sha256"].items():
                if sha256(Path(row["report"]).parent/name) != expected:
                    raise ValueError("completed flight artifact changed")
    else:
        if path.exists() and any(path.iterdir()):
            raise ValueError("nonempty output directory has no suite manifest")
        path.mkdir(parents=True, exist_ok=True)
        summary = {"schema_version": 1, "status": "INCOMPLETE", "config": cfg,
                   "source_sha256": sources, "provenance": evidence, "saved_route_sha256": route_hash,
                   "route_duration_seconds": float(layout["duration"]), "planned_trials": len(args.models)*len(args.periods_ms),
                   "performance_proxy": {"metric": "equal-weight time-mean squared position error",
                       "unit": "m^2", "window_seconds": [0., float(layout["duration"])],
                       "smaller_is_better": True, "not_original_falcon_J": True},
                   "test_read": False, "trials": [],
                   "limitations": ["One validation route, one flight per configuration; no statistical guarantee.",
                       "Feasibility is observed at the candidate's D=T, not a WCET certificate.",
                       "All three drones use the same model/period and the same four-model resident pool.",
                       "Reference prediction dt remains 0.1s; control period AND command duration change together.",
                       "Tracking integral excludes post-route hover; first error held back to time zero.",
                       "Mean service load is a descriptive estimate, not a schedulability test.",
                       "No online state-conditioned selector, dynamic T_max, SVR, supernet or FPGA implementation yet."]}
    save(path, summary)
    try:
        for report_path in args.reuse_report:
            row = compact_result(report_path, summary)
            if not any(r["case"] == row["case"] for r in summary["trials"]):
                row["reused"] = True
                summary["trials"].append(row)
                save(path, summary)
        if any(r["unsafe_failure"] for r in summary["trials"]):
            raise RuntimeError("suite contains a collision/setup/guard error; inspect before continuing")
        cases = list(itertools.product(args.models, args.periods_ms))
        print(f"Joint pilot: {len(cases)} configurations, D=T, existing manual-Play AirSim. "
              "No training or new editor instance.", flush=True)
        for index, (model, period) in enumerate(cases, 1):
            key = case_key(model, period)
            if any(r["case"] == key for r in summary["trials"]):
                print(f"Reuse {index}/{len(cases)}: {key}", flush=True)
                continue
            folder = path / "trials" / key
            attempts = sorted(folder.glob("attempt_*"))
            recovered = None
            if attempts and (attempts[-1] / "report.json").is_file():
                old = read_json(attempts[-1] / "report.json")
                if not any("KeyboardInterrupt" in e for e in old["errors"]):
                    recovered = compact_result(attempts[-1] / "report.json", summary)
            if recovered is None:
                output = folder / f"attempt_{len(attempts)+1:03d}"
                summary["current_trial"] = {"case": key, "output_dir": str(output)}
                save(path, summary)
                command = [sys.executable, "-m", "controller_architecture_study.probe_airsim_three",
                           "--mode", "dnn", "--model", model, "--device", args.device,
                           "--phase", args.phase, "--preprocess-policy", "vectorized",
                           "--environment-id", str(args.environment_id), "--route-index", str(args.route_index),
                           "--period-ms", str(period), "--deadline-ms", str(period),
                           "--host", args.host, "--port", str(args.port), "--output-dir", str(output)]
                print(f"Flight {index}/{len(cases)}: {key}", flush=True)
                result = subprocess.run(command, cwd=ROOT, check=False)
                if result.returncode not in (0, 1, 2) or not (output / "report.json").is_file():
                    raise RuntimeError(f"child stopped with exit {result.returncode}; completed results retained")
                recovered = compact_result(output / "report.json", summary)
                recovered["exit_code"] = result.returncode
            if recovered["case"] != key:
                raise ValueError("trial output belongs to a different candidate")
            if recovered["interrupted"]:
                raise KeyboardInterrupt("child flight interrupted; keep its files and retry on resume")
            summary["trials"].append(recovered)
            save(path, summary)
            if recovered["unsafe_failure"]:
                raise RuntimeError("collision/setup/guard error; remaining flights not started")
        summary["status"] = "COMPLETE"
        summary.pop("current_trial", None)
        summary.pop("interruption", None)
    except BaseException as exc:
        summary["status"] = "INCOMPLETE"
        summary["interruption"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        save(path, summary)
        print(f"Summary: {path / 'summary.json'}\nSelection: {path / 'selection.json'}", flush=True)
    print("model                T=D ms eligible tracking_RMSE_m response_p99_ms misses", flush=True)
    for row in summary["trials"]:
        rmse = "n/a" if row["tracking_rmse_m"] is None else f"{row['tracking_rmse_m']:.4f}"
        p99 = "n/a" if row["response_p99_ms"] is None else f"{row['response_p99_ms']:.3f}"
        print(f"{row['model']:20s} {row['period_ms']:6g} {str(row['eligible']):8s} "
              f"{rmse:>15s} {p99:>15s} {row['output_deadline_misses']:6d}", flush=True)
    best = choose(summary["trials"])
    if best:
        print(f"Observed static candidate: {best['model']}, T=D={best['period_ms']:g}ms; "
              "not automatically applied and not a safety bound.", flush=True)
    return 0 if best else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reuse-report", type=Path, nargs="*", default=[])
    parser.add_argument("--models", choices=DEFAULT_MODELS, nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--periods-ms", type=float, nargs="+", default=[50., 75., 100.])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--phase", choices=("sync", "staggered"), default="staggered")
    parser.add_argument("--environment-id", type=int, choices=range(48, 54), default=48)
    parser.add_argument("--route-index", type=int, choices=range(20), default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=41451)
    parser.add_argument("--select-only", action="store_true", help="Read measured table; never connect to AirSim")
    parser.add_argument("--period-cap-ms", type=float, help="Externally supplied cap; not a derived T_max")
    args = parser.parse_args()
    if (not all(np.isfinite(p) and p > 0 for p in args.periods_ms)
            or (args.period_cap_ms is not None and (not np.isfinite(args.period_cap_ms) or args.period_cap_ms <= 0))):
        parser.error("periods and cap must be finite and positive")
    if args.select_only:
        summary = read_json(args.output_dir / "summary.json")
        result = selection(summary, args.period_cap_ms)
        print(json.dumps(result, indent=2, allow_nan=False))
        raise SystemExit(0 if result["candidate"] else 2)
    if args.period_cap_ms is not None:
        parser.error("--period-cap-ms is only for --select-only")
    args.models = list(dict.fromkeys(args.models))
    args.periods_ms = list(dict.fromkeys(args.periods_ms))
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
