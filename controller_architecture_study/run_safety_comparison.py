"""Finite paired discovery: static aggregate-J search versus dynamic deadlines."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import sys

from .decision_dataset import ROOT, atomic_json, sha256
from .dynamic_probe_common import read_json
from .safety_comparison_core import MODELS, PERIODS_MS, LayerPool, full_job_cost_profile
from .safety_comparison_flight import SafetyWorld
from .state_mode_calibration import moderate_case


BASE = ROOT/"controller_architecture_study/results/safety_framework_comparison_v5"


def config(count, model=MODELS[0], period=100., phase="staggered"):
    names = [f"Drone{i+1}" for i in range(count)]
    return {"models": {v: model for v in names}, "periods_ms": {v: period for v in names},
            "priority_order": names, "phase": phase}


def score(report):
    return report["total_J"] if (report["mission_passed"] and report["clock_valid"]
        and report["expired_not_applied"] == 0 and report["rpc_after_deadline"] == 0) else None


def scenarios(count):
    # Both priority extremes and both disturbance directions: no winner-only selection.
    return [{"id": f"i{i}_sign{sign:+d}_v{v:g}", "critical_index": i,
        "position_y_m": .05*sign, "velocity_y_mps": v*sign}
        for i in (0, count-1) for sign in (-1, 1) for v in (.6, 1., 1.4, 1.8)]


def clean_split(pair):
    baseline, framework = pair["static"], pair["framework"]
    return bool(baseline["collisions"] > 0 and framework["mission_passed"]
        and all(r["clock_valid"] and r["expired_not_applied"] == 0 and r["rpc_after_deadline"] == 0
            and not r["error"] for r in (baseline, framework)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "calibrate", "discover", "confirm"), required=True)
    parser.add_argument("--vehicles", type=int, choices=(3, 8, 12), default=8)
    parser.add_argument("--net-width", type=float, choices=(2., 1.5, 1., .75), default=2.)
    args = parser.parse_args()
    # Python RPC/feature work must not hold the inference thread behind a5ms GIL slice.
    # Identical process-local setting for both policies; this is not an OS realtime guarantee.
    sys.setswitchinterval(.0005)
    root = BASE/f"n{args.vehicles}_net{args.net_width:g}"
    if args.stage == "smoke":
        root = root/("smoke_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    root.mkdir(parents=True, exist_ok=True)
    case, layout = moderate_case()
    pool = LayerPool()
    metadata = {"version": "safety_framework_comparison_v5", "actual_vehicles": args.vehicles,
        "net_width_m": args.net_width, "reference_environment": 48, "reference_route": 1,
        "reference_phase_s": case["reference_phase_s"], "route_sha256": case["route_sha256"],
        "physical_phase_span_s": 2.1,
        "reference_phases_s": {f"Drone{i+1}": case["reference_phase_s"]-2.1*i/max(1,args.vehicles-1)
            for i in range(args.vehicles)},
        "models": pool.provenance(), "periods_ms": list(PERIODS_MS),
        "python_thread_switch_interval_s": sys.getswitchinterval(),
        "resource_model": "one exclusive full job incl frontend/IPC/inference/command; nominal p95 predictive cost, not WCET",
        "frontend_inflight_limit": 1, "intra_job_preemption_benefit_claimed": False,
        "ipc_polling": "nonblocking Pipe.poll plus high-resolution sleep; no timed Pipe wait in dispatch path",
        "source_sha256": {name: sha256(Path(__file__).with_name(name)) for name in
            ("run_safety_comparison.py", "safety_comparison_core.py", "safety_comparison_flight.py")},
        "duration_s": 3., "fixed_scenarios": scenarios(args.vehicles),
        "baseline": "empirical bounded static local search of sum J; no SVM, fixed CPU, restricted candidate family; NOT full FALCON",
        "framework": "unchanged physical deadline + collective EDF slack release planning for all unreleased full jobs; SAME selected models",
        "metric": "Eq4-form, Wx=I(10), common20ms grid, average nonoverlap K20 window cost then reciprocal; sum equally weighted per-vehicle J",
        "failure_metric": "collision is primary; failure/partial mission J is missing, never rewards early abort",
        "scope": "finite exploratory prototype, real AirSim vehicles, real segmented CPU inference, NOT FPGA/CBF/SVM or complete framework validation",
        "test_read": False, "training_performed": False}
    if (root/"setup.json").exists() and read_json(root/"setup.json") != metadata:
        raise ValueError("frozen comparison source/configuration changed; preserve results and use a new version")
    atomic_json(root/"setup.json", metadata)
    atomic_json(root/("profile_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")+".json"), pool.profile)
    nominal = {"id": "nominal", "critical_index": -1, "position_y_m": 0., "velocity_y_mps": 0.}
    world = None

    def run(key, cfg, scenario, policy):
        nonlocal world
        directory = root/key
        if (directory/"report.json").exists():
            return read_json(directory/"report.json")
        if directory.exists():
            raise ValueError(f"partial trial retained; inspect before another attempt: {directory}")
        if world is None:
            world = SafetyWorld(layout, case["reference_phase_s"], args.vehicles, args.net_width, phase_span=2.1)
        print(f"FLIGHT {key} N={args.vehicles} net={args.net_width:g} policy={policy}", flush=True)
        try:
            report = world.trial(pool, cfg, scenario, policy, directory)
        except Exception as exc:
            atomic_json(directory/"infrastructure_error.json", {"error": f"{type(exc).__name__}: {exc}",
                "valid_policy_comparison": False, "do_not_count_as_collision_or_controller_failure": True})
            raise
        print(f"  mission={report['mission_passed']} collisions={report['collisions']} J={report['total_J']} "
              f"expired={report['expired_not_applied']} error={report['error']}", flush=True)
        if report["error"]:
            raise RuntimeError("infrastructure/control execution error retained; no blind retry")
        return report

    def validate_candidate(best):
        framework_nominal = run("calibration/framework_nominal", best["configuration"], nominal, "framework")
        result = {"candidate": best, "framework_nominal": framework_nominal,
            "not_global_optimum": True, "not_complete_FALCON": True, "nominal_only_selection": True}
        result["status"] = "BEST_OBSERVED_LOCAL_CANDIDATE" if score(framework_nominal) is not None else "FRAMEWORK_NOMINAL_FAILED"
        atomic_json(root/"selection.json", result)
        if result["status"] != "BEST_OBSERVED_LOCAL_CANDIDATE":
            raise RuntimeError("selected static configuration has no passing framework nominal run; retain this failure")

    try:
        if args.stage == "smoke":
            passed = True
            for policy in ("static", "framework"):
                report = run(policy, config(args.vehicles, period=150.), nominal, policy)
                passed = passed and score(report) is not None
            if not passed:
                raise RuntimeError("nominal smoke did not pass under both policies; do not interpret challenge results yet")
        elif args.stage == "calibrate":
            # First compare common operating points. Then test one-loop/phase/priority neighbors.
            candidates = [config(args.vehicles, model=m, period=t) for m in MODELS for t in PERIODS_MS]
            observed = []
            for i, candidate in enumerate(candidates):
                report = run(f"calibration/base{i:02d}", candidate, nominal, "static")
                observed.append({"id": f"base{i:02d}", "configuration": candidate, "score": score(report), "report": report})
                atomic_json(root/"calibration_results.json", observed)
            valid = [r for r in observed if r["score"] is not None]
            if not valid:
                atomic_json(root/"selection.json", {"status": "NO_FEASIBLE_BASELINE", "observed": observed})
                raise RuntimeError("no nominal static baseline passed; do not search for a safety advantage yet")
            current = deepcopy(max(valid, key=lambda r: r["score"])["configuration"])
            neighbors = []
            flipped = deepcopy(current)
            flipped["phase"] = "sync" if current["phase"] == "staggered" else "staggered"
            neighbors.append(flipped)
            reversed_priority = deepcopy(current)
            reversed_priority["priority_order"].reverse()
            neighbors.append(reversed_priority)
            last = f"Drone{args.vehicles}"
            for m in MODELS[1:]:
                c = deepcopy(current)
                c["models"][last] = m
                neighbors.append(c)
            index = list(PERIODS_MS).index(current["periods_ms"][last])
            for j in (index-1, index+1):
                if 0 <= j < len(PERIODS_MS):
                    c = deepcopy(current)
                    c["periods_ms"][last] = PERIODS_MS[j]
                    neighbors.append(c)
            for i, candidate in enumerate(neighbors):
                report = run(f"calibration/neighbor{i:02d}", candidate, nominal, "static")
                observed.append({"id": f"neighbor{i:02d}", "configuration": candidate, "score": score(report), "report": report})
                atomic_json(root/"calibration_results.json", observed)
            best = max((r for r in observed if r["score"] is not None), key=lambda r: r["score"])
            calibration_requests = []
            for r in observed:
                if r["score"] is not None:
                    calibration_requests.extend(read_json(root/"calibration"/r["id"]/"requests.json"))
            cost_profile = full_job_cost_profile(calibration_requests)
            atomic_json(root/"full_job_cost_profile.json", cost_profile)
            best = deepcopy(best)
            best["configuration"]["job_costs_s"] = {v: cost_profile[m]["estimated_cost_s"]
                for v, m in best["configuration"]["models"].items()}
            validate_candidate(best)
        else:
            selected = read_json(root/"selection.json")
            if selected["status"] != "BEST_OBSERVED_LOCAL_CANDIDATE":
                raise ValueError("calibrate and freeze a feasible static candidate first")
            chosen = selected["candidate"]["configuration"]
            if args.stage == "discover":
                rows = []
                for i, scenario in enumerate(scenarios(args.vehicles)):
                    pair = {"scenario": scenario}
                    for policy in (("static", "framework") if i % 2 == 0 else ("framework", "static")):
                        pair[policy] = run(f"discovery/{scenario['id']}/{policy}", chosen, scenario, policy)
                    pair["desired_split"] = clean_split(pair)
                    rows.append(pair)
                    atomic_json(root/"discovery_summary.json", {"pairs": rows, "complete": len(rows) == 16,
                        "selection_sha256": sha256(root/"selection.json"), "desired_splits": sum(r["desired_split"] for r in rows)})
            else:
                discovery = read_json(root/"discovery_summary.json")
                if not discovery["complete"]:
                    raise ValueError("finish the entire declared discovery before choosing confirmation")
                splits = [r for r in discovery["pairs"] if r["desired_split"]]
                if not splits:
                    print("NO discovery split; no confirmation invented. Preserve the negative result.", flush=True)
                    return
                scenario = deepcopy(splits[0]["scenario"])
                nominal_candidates = [r for r in read_json(root/"calibration_results.json") if r["score"] is not None]
                fastest = min(nominal_candidates, key=lambda r: sum(r["configuration"]["periods_ms"].values()))["configuration"]
                rows = []
                for repeat, perturbation in enumerate((-.01, 0., .01)):
                    case = deepcopy(scenario)
                    case["position_y_m"] += perturbation
                    arms = [(p, chosen, p) for p in (("framework", "static") if repeat % 2 == 0 else ("static", "framework"))]
                    if fastest != chosen:
                        arms.append(("fastest_nominal_static", fastest, "static"))
                        if repeat % 2 == 1:
                            arms.reverse()
                    for arm, cfg, policy in arms:
                        r = run(f"confirmation/r{repeat}_{arm}", cfg, case, policy)
                        rows.append({"repeat": repeat, "arm": arm, "report": r})
                        atomic_json(root/"confirmation_summary.json", {"trials": rows, "complete": len(rows) == 3*len(arms),
                            "extra_static_control": "fastest nominally feasible observed static configuration, if distinct",
                            "discovery_sha256": sha256(root/"discovery_summary.json"),
                            "scope": "three NEW repeats with fixed+-1cm neighboring initial positions; not a reliability certificate"})
    finally:
        if world is not None:
            cleanup = world.close()
            atomic_json(root/"cleanup_report.json", cleanup)
            if cleanup["status"] != "CLEANUP_COMPLETE":
                print(f"Cleanup warning: {cleanup}", flush=True)
    print(f"Stage complete: {root}", flush=True)


if __name__ == "__main__":
    main()
