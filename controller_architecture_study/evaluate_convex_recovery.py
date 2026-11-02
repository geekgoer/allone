"""Manual Play, resumable combined-kick tests on two frozen Validation routes."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from .convex_dataset import EXPERIMENT
from .convex_recovery import OUTPUT, read_json, verify_arm
from .convex_recovery_flight import PROFILES, PROTOCOL
from .decision_dataset import atomic_json, sha256
from .fly_convex_policy import fly, verify_dataset
from .run_convex_recovery import WIDTHS, selection
from .train_convex_capacity import run_state


CASES = [(env, route, profile) for env, route in ((48, 3), (51, 6)) for profile in PROFILES]


def plan(base, output, seeds, vehicle_name):
    selection(("features_recovery",), seeds)
    verify_dataset(base, full=False)
    verify_arm(output/"datasets/features_recovery", base, full=False)
    models = []
    for seed in seeds:
        for width in WIDTHS:
            run = output/"runs/features_recovery"/f"plain_mlp_h4_w{width}_seed{seed}"
            config = read_json(run/"config.json")
            if (run_state(config, run) != "complete" or config["input_dim"] != 237
                    or config["hidden_layers"] != 4 or config["hidden_width"] != width
                    or config["seed"] != seed or config.get("ablation_arm") != "features_recovery"):
                raise ValueError(f"requires a completed matching recovery model: {run}")
            models.append({"model": run.name, "checkpoint": str((run/"best_model.pt").resolve()),
                           "sha256": sha256(run/"best_model.pt")})
    collection = read_json(base/"manifest.json")
    cases = []
    for env, route, profile in CASES:
        name = f"env{env:02d}_route{route:02d}"
        digest = sha256(base/"routes"/f"{name}.json")
        if not collection["complete"] or digest != collection["chunks"][name]["route_sha256"]:
            raise ValueError(f"saved route changed: {name}")
        cases.append({"case": f"{name}_{profile}", "environment_id": env, "route_index": route,
                      "disturbance": profile, "route_sha256": digest})
    return {"base": str(base), "models": models, "cases": cases, "vehicle_name": vehicle_name,
            "protocol": PROTOCOL, "profiles": PROFILES, "test_evaluated": False,
            "source_sha256": {name: sha256(Path(__file__).with_name(name)) for name in
                              ("evaluate_convex_recovery.py", "convex_recovery_flight.py", "fly_convex_policy.py")}}


def summarize_trials(results, models, cases):
    rows = []
    for model in ["teacher", *[item["model"] for item in models]]:
        trials = [results[f"{case['case']}/{model}"] for case in cases if f"{case['case']}/{model}" in results]
        attempted = [trial for trial in trials if trial["status"] != "SKIP"]
        times = [trial["recovery"]["confirmation_time_after_kick_seconds"] for trial in attempted
                 if trial.get("recovery", {}).get("recovered")]
        peaks = [trial["recovery"]["post_kick_peak_position_error_m"] for trial in attempted
                 if "post_kick_peak_position_error_m" in trial.get("recovery", {})]
        rows.append({"model": model, "completed_trials": len(attempted),
                     "skipped_teacher_failed": len(trials)-len(attempted),
                     "successes": sum(trial["status"] == "PASS" for trial in attempted),
                     "failures": sum(trial["status"] == "FAIL" for trial in attempted),
                     "recovery_confirmation_mean_seconds": None if not times else sum(times)/len(times),
                     "post_kick_peak_position_error_max_m": None if not peaks else max(peaks),
                     "success_rate": None if not attempted else sum(trial["status"] == "PASS" for trial in attempted)/len(attempted)})
    return rows


def execute(config, report_path, *, resume=False, fly_fn=fly):
    if report_path.exists():
        if not resume:
            raise ValueError("report exists; use --resume to retain completed trials")
        report = read_json(report_path)
        if report["config"] != config:
            raise ValueError("suite/model/protocol changed; use a new --report path")
    else:
        report = {"status": "running", "config": config, "results": {}}
    results = report["results"]

    def save():
        report["summary"] = summarize_trials(results, config["models"], config["cases"])
        atomic_json(report_path, report)

    for case in config["cases"]:
        for model in [{"model": "teacher", "checkpoint": None}, *config["models"]]:
            key = f"{case['case']}/{model['model']}"
            if key in results:
                continue
            teacher = results.get(f"{case['case']}/teacher")
            if model["model"] != "teacher" and teacher["status"] != "PASS":
                results[key] = {"status": "SKIP", "reason": "QP teacher did not pass the identical recovery case"}
                save()
                continue
            print(f"Recovery trial: {key}", flush=True)
            args = SimpleNamespace(output=Path(config["base"]),
                checkpoint=None if model["checkpoint"] is None else Path(model["checkpoint"]),
                environment_id=case["environment_id"], route_index=case["route_index"],
                disturbance=case["disturbance"], kick_time=config["protocol"]["kick_time_seconds"],
                vehicle_name=config["vehicle_name"], device="cpu", keep_gates=False)
            trial = fly_fn(args)
            if trial.get("error") == "interrupted by user":
                report.update(status="interrupted", last_interruption=trial)
                save()
                return report
            # Stop on connection/readback/startup failures, rather than falsely calling the case infeasible.
            event = trial.get("recovery", {}).get("event") or {}
            if (not trial.get("samples") or (event and not event.get("applied"))
                    or (trial.get("five_gate_flight_passed") and not event)):
                report.update(status="needs_attention", last_infrastructure_failure=trial)
                save()
                return report
            results[key] = trial
            save()
    report["status"] = "complete"
    save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=EXPERIMENT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--vehicle-name", default="Drone1")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    base, output = args.base.resolve(), args.output.resolve()
    config = plan(base, output, args.seeds, args.vehicle_name)
    path = args.report or output/"runs"/("recovery_closed_loop_seeds"+"_".join(map(str, args.seeds))+".json")
    print(f"Four fixed cases: {len(config['models'])} DNNs + QP teacher; "
          f"up to {len(CASES)*(len(config['models'])+1)} flights; manual UE4 Play only", flush=True)
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return
    report = execute(config, path, resume=args.resume)
    print(json.dumps({"status": report["status"], "summary": report["summary"]}, indent=2))
    print(f"Report: {path}", flush=True)
    if report["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
