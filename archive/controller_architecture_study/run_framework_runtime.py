"""Run the framework's CPU analogue, preserving every accepted/rejected trial."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import sys
import shutil
import time

import numpy as np

from .decision_dataset import ROOT, atomic_json, sha256
from .dynamic_probe_common import read_json
from .framework_runtime import CONTRACT, COLLISION_CONTRACT, TIMING_CONTRACT, ControllerPool, MODELS, ModeMap
from .framework_runtime_flight import DEFAULT_COSTS, fly
from .framework_runtime_world import FrameworkWorld
from .framework_cpu_affinity import CpuPlacement
from .framework_runtime_editor import read_background_throttle
from .run_safety_comparison import config, scenarios
from .safety_comparison_core import SegmentServer
from .state_mode_calibration import moderate_case


OUTPUT = ROOT/"controller_architecture_study/results/framework_runtime_v43_paired_ingress"
PERIODS = (25.,30.,35.,40.,45.,50.,75.,100.)


def preemption_check(pool, directory):
    """Actual two-context execution, not a timing trace replay or artificial load."""
    rows,eager_errors = [],{}
    raw = pool.norm["x_mean"]+.1*pool.norm["x_std"]
    x = pool.torch.from_numpy(((raw-pool.norm["x_mean"])/pool.norm["x_std"])[None].astype(np.float32))
    for model in MODELS:
        with pool.torch.inference_mode():
            expected = pool.models[model](x).numpy()[0]*pool.norm["u_std"]+pool.norm["u_mean"]
        eager_errors[model] = float(np.max(abs(pool.predict(model,raw)-expected)))
    for repeat in range(3):
        server = SegmentServer(pool,"edf")
        inputs = [pool.norm["x_mean"]+sign*.1*pool.norm["x_std"] for sign in (1.,-1.)]
        origin = time.perf_counter()
        low = {"sample_id":"low","vehicle":"low","model":MODELS[-1],"deadline_s":origin+.500,"priority":1}
        urgent = {"sample_id":"urgent","vehicle":"urgent","model":MODELS[0],"deadline_s":origin+.150,"priority":0}
        result = {}
        try:
            server.submit(inputs[0],low)
            while not server.segments and time.perf_counter()-origin < .100:
                time.sleep(.0001)
            server.submit(inputs[1],urgent)
            while len(result)<2 and time.perf_counter()-origin < .500:
                for action,row in server.drain():
                    result[row["sample_id"]] = action
                time.sleep(.0001)
        finally:
            server.close()
        ids = [s["sample_id"] for s in server.segments]
        observed = bool("urgent" in ids and "low" in ids[:ids.index("urgent")]
            and "low" in ids[ids.index("urgent")+1:])
        errors = {key:float(np.max(abs(result[key]-pool.predict(model,x)))) if key in result and result[key] is not None else None
            for key,model,x in (("low",MODELS[-1],inputs[0]),("urgent",MODELS[0],inputs[1]))}
        rows.append({"repeat":repeat,"preemption_observed":observed,"segment_order":ids,
            "max_physical_output_errors":errors,"completed":len(result)==2,"segments":server.segments})
    report = {"repeats":rows,"actual_segment_preemptions_observed":sum(r["preemption_observed"] for r in rows),
        "numerics_passed":all(e<=2e-6 for e in eager_errors.values()) and all(r["completed"] and all(e is not None and e<=2e-6 for e in r["max_physical_output_errors"].values()) for r in rows),
        "segmented_vs_original_eager_errors":eager_errors,
        "context":"actual shared CPU worker; per-job activations isolated, no model changes mid-job; not FPGA"}
    atomic_json(directory/"preemption_check.json",report)
    if not report["numerics_passed"]:
        raise RuntimeError("segmented shared execution changed the frozen controller function or lost a job")
    return report


def clean_win(pair):
    return bool(pair.get("initial_state_match",False) and pair["static"]["collisions"]>0 and pair["adaptive"]["accepted"]
        and pair["static"]["timing_passed"] and pair["static"]["clock_valid"]
        and pair["static"]["error"] is None)


def initial_states_match(left, right):
    for filename,fields in (("initial_state.json",("position","velocity","quaternion","held_control","matched")),
            ("traces.json",("position","velocity","quaternion"))):
        a,b = read_json(left/filename),read_json(right/filename)
        if a.keys() != b.keys():
            return False
        for v in a:
            x,y = (a[v][0],b[v][0]) if filename == "traces.json" else (a[v],b[v])
            if any(x[k] != y[k] for k in fields):
                return False
    return True


def repeated_static_selection(observed, check, output, excluded=()):
    shortlist = sorted((r for r in observed if r["score"] is not None and r["name"] not in excluded),
        key=lambda r:r["score"],reverse=True)[:3]
    validated = []
    for i,candidate in enumerate(shortlist):
        checks = [check(i,repeat,candidate["configuration"]) for repeat in range(2)]
        valid = all(r["accepted"] for r in checks)
        validated.append({**candidate,"validation_reports":checks,"all_runs_passed":valid,
            "score":float(np.mean([candidate["score"],*[r["total_J"] for r in checks]])) if valid else None})
        atomic_json(output,validated)
    stable = [r for r in validated if r["score"] is not None]
    return max(stable,key=lambda r:r["score"]) if stable else None


def frozen_baseline(selection, source_setup, current_setup):
    fields = ("vehicles","net_width_m","provenance","physical_phase_span_s",
        "phase_s","route_sha256","duration_s","cpu_placement")
    if any(source_setup[k] != current_setup[k] for k in fields):
        raise ValueError("frozen static search controller/resource/physical domain differs")
    if source_setup.get("editor_background_throttle_enabled",True) != current_setup.get("editor_background_throttle_enabled",True):
        raise ValueError("editor timing conditions changed; repeat the finite static search instead of reusing its claimed optimum")
    if not selection["candidate"]["report"]["accepted"]:
        raise ValueError("source static candidate did not pass its declared conditions")
    return deepcopy(selection["candidate"]["configuration"])


def run(args, placement):
    sys.setswitchinterval(.0005)
    editor = read_background_throttle()
    atomic_json(args.output_dir/"editor_preflight.json",editor)
    if editor["background_throttle_enabled"]:
        raise RuntimeError("Unreal background CPU throttling is enabled; disable Use Less CPU when in Background and keep the window unminimized before timing comparisons")
    costs = dict(DEFAULT_COSTS)
    cost_profile_sha256 = None
    if args.cost_profile:
        profile = read_json(args.cost_profile)
        if (profile["contract"] != "framework_cpu_empirical_costs_v1" or profile["vehicles"] != args.vehicles
                or profile["cpu_placement"] != {"enabled":placement.enabled,"layout":placement.layout if placement.enabled else None}
                or profile["costs"]["is_WCET"]):
            raise ValueError("empirical cost profile CPU layout/concurrency/contract mismatch")
        if profile.get("editor_background_throttle_enabled",True) != editor["background_throttle_enabled"]:
            raise ValueError("cost profile was measured under different editor throttling; collect a new bootstrap smoke first")
        if profile.get("timing_contract") != TIMING_CONTRACT:
            raise ValueError("cost profile ends at the helper instead of parent visibility; collect a matching bootstrap")
        costs = dict(profile["costs"])
        if profile.get("statistic") == "p95":
            tail_statistics = {**profile["statistics_s"],**profile.get("visibility_tail_statistics_s",{})}
            costs["tail_costs"] = {k:max(.001,1.25*s["max"]+.0005) for k,s in tail_statistics.items()}
            costs["tail_policy"] = "observed nominal overrun: use padded development maximum minus actual elapsed; never extend deadline"
        cost_profile_sha256 = sha256(args.cost_profile)
    root = args.output_dir/f"n{args.vehicles}_net2"
    if args.stage in ("smoke","preflight"):
        root = root/(args.stage+"_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    root.mkdir(parents=True,exist_ok=True)
    placement.result_path = root/"cpu_placement.json"
    atomic_json(placement.result_path,placement.report)
    pool = ControllerPool()
    modes = ModeMap(args.mode_map,pool.provenance(),2.)
    case,layout = moderate_case()
    envelope = modes.data.get("reference_acceleration_envelope")
    if envelope is not None and envelope["route_sha256"]!=case["route_sha256"]:
        raise ValueError("mode-map reference envelope belongs to another saved route")
    source_names = ("framework_runtime_wait.py","framework_runtime_io_wire.py","framework_runtime_wire.py","framework_runtime_editor.py","framework_cpu_affinity.py","framework_runtime.py","framework_runtime_worker.py","framework_runtime_world.py",
        "framework_runtime_flight.py","framework_runtime_io.py","framework_runtime_features.py","framework_runtime_calibrate.py","run_framework_runtime.py","safety_comparison_core.py",
        "safety_comparison_flight.py")
    metadata = {"contract":CONTRACT,"collision_observation":COLLISION_CONTRACT,"timing_contract":TIMING_CONTRACT,"version":"framework_runtime_v43_paired_ingress","vehicles":args.vehicles,"net_width_m":2.,
        "editor_background_throttle_enabled":editor["background_throttle_enabled"],"smoke_period_ms":50.,
        "cpu_placement":{"enabled":placement.enabled,"layout":placement.layout if placement.enabled else None},
        "provenance":pool.provenance(),"mode_map_sha256":modes.sha256,"physical_phase_span_s":2.1,
        "phase_s":case["reference_phase_s"],"route_sha256":case["route_sha256"],"duration_s":3.,
        "source_sha256":{n:sha256(Path(__file__).with_name(n)) for n in source_names},"costs":costs,
        "source_search":str(args.source_search.resolve()) if args.source_search else None,
        "source_search_sha256":sha256(args.source_search/"calibration_results.json") if args.source_search else None,
        "source_screening_sha256":sha256(args.source_search/"baseline_validation.json") if args.source_search else None,
        "cost_profile_sha256":cost_profile_sha256,
        "baseline_selection_sha256":sha256(args.baseline_selection) if args.baseline_selection else None,
        "static_candidate_periods_ms":list(PERIODS),"scenarios":scenarios(args.vehicles),
        "static_shortlist_validation":{"candidates":3,"additional_normal_runs":2,
            "rule":"all three runs must pass; select highest mean J"},
        "scope":"CPU analogue of mode selection, segmented EDF and bounded pipeline release; not a CBF/FPGA guarantee",
        "baseline":"FALCON-style finite empirical sum-J static search, six models, no SVM or full hardware search",
        "failure_handling":"both stop on missed committed deadlines; hover never counts as safe task completion",
        "test_read":False,"training_performed":False}
    if (root/"setup.json").exists() and read_json(root/"setup.json") != metadata:
        raise ValueError("frozen runtime identity changed; retain results and choose a new output directory")
    frozen = root/"frozen_source"
    frozen.mkdir(exist_ok=True)
    for name in source_names:
        destination = frozen/name
        if not destination.exists():
            shutil.copy2(Path(__file__).with_name(name),destination)
        if sha256(destination) != metadata["source_sha256"][name]:
            raise ValueError("retained source snapshot differs; choose a new output directory")
    atomic_json(root/"setup.json",metadata)
    if not (root/"preemption_check.json").exists():
        check = preemption_check(pool,root)
        print(f"PREEMPTION numerics={check['numerics_passed']} observed={check['actual_segment_preemptions_observed']}/3",flush=True)
    if args.stage == "preflight":
        print(f"PREFLIGHT_READY {root}",flush=True)
        return
    world = None
    nominal = {"id":"nominal","critical_index":-1,"position_y_m":0.,"velocity_y_mps":0.}

    def run(key,configuration,scenario,policy):
        nonlocal world
        directory = root/key
        if (directory/"report.json").exists():
            return read_json(directory/"report.json")
        if directory.exists():
            raise ValueError(f"partial trial retained; inspect before retry: {directory}")
        if world is None:
            world = FrameworkWorld(layout,case["reference_phase_s"],args.vehicles,2.,phase_span=2.1)
        print(f"FLIGHT {key} N={args.vehicles} policy={policy}",flush=True)
        r = fly(world,pool,configuration,scenario,policy,directory,mode_map=modes if policy=="adaptive" else None,
            cpu_layout=placement.layout if placement.enabled else None,costs=costs)
        print(f"  accepted={r['accepted']} physical={r['physical_task_passed']} recovery={r['recovery_passed']} "
              f"collisions={r['collisions']} J={r['total_J']} switches={r['model_switches']} "
              f"resumes={r['segment_resumptions']} abort={r['abort']} error={r['error']}",flush=True)
        if r["error"]:
            raise RuntimeError("infrastructure error retained; no automatic retry")
        return r

    try:
        if args.stage == "smoke":
            baseline = config(args.vehicles,model=MODELS[0],period=50.)
            reports = [run(p,baseline,nominal,p) for p in ("static","adaptive")]
            atomic_json(root/"summary.json",{"reports":reports,"all_accepted":all(r["accepted"] for r in reports)})
            if not all(r["accepted"] for r in reports):
                raise RuntimeError("nominal integration has an explicit failure; inspect its report before expanding")
        elif args.stage == "revalidate":
            if args.baseline_selection is None:
                raise ValueError("revalidate requires an explicit frozen baseline selection")
            source = read_json(args.baseline_selection)
            cfg = frozen_baseline(source,read_json(args.baseline_selection.parent/"setup.json"),metadata)
            static = run("calibration/frozen_static",cfg,nominal,"static")
            adaptive = run("calibration/adaptive_nominal",cfg,nominal,"adaptive")
            candidate = {"name":"frozen_"+source["candidate"]["name"],"configuration":cfg,"report":static,
                "score":static["total_J"] if static["accepted"] else None}
            atomic_json(root/"calibration_results.json",[candidate])
            status = "READY" if static["accepted"] and adaptive["accepted"] else "NOMINAL_REVALIDATION_FAILED"
            atomic_json(root/"selection.json",{"candidate":candidate,"adaptive_nominal":adaptive,"status":status,
                "source_static_search":str(args.baseline_selection.resolve()),
                "source_selection_sha256":metadata["baseline_selection_sha256"],
                "source_observed_score":source["candidate"]["score"],"not_global_optimum":True,"not_complete_FALCON":True})
            if status != "READY":
                raise RuntimeError("frozen baseline/current framework normal pair failed; no challenge was started")
        elif args.stage == "stabilize":
            if args.source_search is None:
                raise ValueError("stabilize requires a retained source search")
            if read_json(args.source_search/"selection.json").get("status") != "NO_REPEATEDLY_VALID_STATIC_CANDIDATE":
                raise ValueError("continue screening only after the source shortlist explicitly failed")
            source_setup = read_json(args.source_search/"setup.json")
            if (source_setup.get("collision_observation") != COLLISION_CONTRACT
                    or source_setup.get("cost_profile_sha256") != metadata["cost_profile_sha256"]):
                raise ValueError("source search collision/cost domain differs")
            observed = read_json(args.source_search/"calibration_results.json")
            previous = read_json(args.source_search/"baseline_validation.json")
            excluded = [r["name"] for r in previous]
            for candidate in observed:
                if candidate["score"] is not None:
                    frozen_baseline({"candidate":candidate},source_setup,metadata)
            atomic_json(root/"screening_source.json",{"search":metadata["source_search"],
                "excluded_already_screened":excluded,"original_failures_preserved":True,
                "additional_candidates":3,"initial_candidates_not_reflown":True})
            best = repeated_static_selection(observed,lambda i,j,c:run(
                f"baseline_validation/c{i}_r{j}",c,nominal,"static"),root/"baseline_validation.json",excluded)
            if best is None:
                atomic_json(root/"selection.json",{"status":"NO_REPEATEDLY_VALID_STATIC_CANDIDATE"})
                raise RuntimeError("the next three unscreened candidates did not all pass; no challenge started")
            r = run("calibration/adaptive_nominal",best["configuration"],nominal,"adaptive")
            atomic_json(root/"selection.json",{"candidate":best,"adaptive_nominal":r,
                "status":"READY" if r["accepted"] else "ADAPTIVE_NOMINAL_FAILED",
                "source_static_search":str((args.source_search/"calibration_results.json").resolve()),
                "not_global_optimum":True,"not_complete_FALCON":True})
            if not r["accepted"]:
                raise RuntimeError("adaptive nominal integration failed; no challenge started")
        elif args.stage == "calibrate":
            observed = []
            for model in MODELS:
                for period in PERIODS:
                    name = f"{model}_T{period:g}"
                    cfg = config(args.vehicles,model=model,period=period)
                    r = run(f"calibration/{name}",cfg,nominal,"static")
                    observed.append({"name":name,"configuration":cfg,"report":r,
                        "score":r["total_J"] if r["accepted"] else None})
                    atomic_json(root/"calibration_results.json",observed)
            valid = [r for r in observed if r["score"] is not None]
            if not valid:
                raise RuntimeError("no valid nominal static baseline; challenge claims are not supported")
            best = max(valid,key=lambda r:r["score"])
            neighbors = []
            reverse = deepcopy(best["configuration"])
            reverse["priority_order"].reverse()
            neighbors.append(reverse)
            simultaneous = deepcopy(best["configuration"])
            simultaneous["phase"] = "sync"
            neighbors.append(simultaneous)
            last = f"Drone{args.vehicles}"
            index = PERIODS.index(best["configuration"]["periods_ms"][last])
            for i in (index-1,index+1):
                if 0<=i<len(PERIODS):
                    c = deepcopy(best["configuration"])
                    c["periods_ms"][last] = PERIODS[i]
                    neighbors.append(c)
            for i,c in enumerate(neighbors):
                r = run(f"calibration/neighbor{i}",c,nominal,"static")
                observed.append({"name":f"neighbor{i}","configuration":c,"report":r,
                    "score":r["total_J"] if r["accepted"] else None})
            atomic_json(root/"calibration_results.json",observed)
            best = repeated_static_selection(observed,lambda i,j,c:run(
                f"baseline_validation/c{i}_r{j}",c,nominal,"static"),root/"baseline_validation.json")
            if best is None:
                atomic_json(root/"selection.json",{"status":"NO_REPEATEDLY_VALID_STATIC_CANDIDATE"})
                raise RuntimeError("no shortlisted baseline passed all three normal runs; no challenge started")
            r = run("calibration/adaptive_nominal",best["configuration"],nominal,"adaptive")
            atomic_json(root/"selection.json",{"candidate":best,"adaptive_nominal":r,
                "status":"READY" if r["accepted"] else "ADAPTIVE_NOMINAL_FAILED",
                "not_global_optimum":True,"not_complete_FALCON":True})
            if not r["accepted"]:
                raise RuntimeError("adaptive nominal integration failed; do not interpret challenges as safety validation")
        else:
            selection = read_json(root/"selection.json")
            if selection["status"] != "READY":
                raise ValueError("both nominal paths must pass first")
            cfg = selection["candidate"]["configuration"]
            if args.stage == "discover":
                rows = []
                for i,scenario in enumerate(scenarios(args.vehicles)):
                    pair = {"scenario":scenario}
                    for p in (("static","adaptive") if i%2==0 else ("adaptive","static")):
                        pair[p] = run(f"discovery/{scenario['id']}/{p}",cfg,scenario,p)
                    pair["initial_state_match"] = initial_states_match(root/f"discovery/{scenario['id']}/static",
                        root/f"discovery/{scenario['id']}/adaptive")
                    pair["clean_safety_win"] = clean_win(pair)
                    rows.append(pair)
                    atomic_json(root/"discovery_summary.json",{"pairs":rows,"complete":len(rows)==16,
                        "clean_safety_wins":sum(p["clean_safety_win"] for p in rows)})
            else:
                discovery = read_json(root/"discovery_summary.json")
                if not discovery["complete"]:
                    raise ValueError("finish the complete declared discovery first")
                winners = [p for p in discovery["pairs"] if p["clean_safety_win"]]
                if not winners:
                    print("NO_CLEAN_SAFETY_WIN; negative results preserved",flush=True)
                    return
                rows = []
                for repeat,delta in enumerate((-.01,0.,.01)):
                    scenario = deepcopy(winners[0]["scenario"])
                    scenario["position_y_m"] += delta
                    for p in (("adaptive","static","deadline_only") if repeat%2==0 else ("deadline_only","static","adaptive")):
                        r = run(f"confirmation/r{repeat}_{p}",cfg,scenario,p)
                        rows.append({"repeat":repeat,"policy":p,"report":r})
                        atomic_json(root/"confirmation_summary.json",{"rows":rows,"complete":len(rows)==9,
                            "ablation":"deadline_only keeps the selected static modes but shares the same EDF/release mechanism"})
    finally:
        if world is not None:
            atomic_json(root/"cleanup_report.json",world.close())
    print(f"STAGE_COMPLETE {args.stage} {root}",flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("preflight","smoke","revalidate","calibrate","stabilize","discover","confirm"),required=True)
    parser.add_argument("--vehicles",type=int,choices=(3,7,8,12),default=8)
    parser.add_argument("--mode-map",type=Path,default=ROOT/"controller_architecture_study/results/framework_runtime_refinement_v2_direct_collision/mode_map.json")
    parser.add_argument("--output-dir",type=Path,default=OUTPUT)
    parser.add_argument("--cpu-isolation",choices=("physical","off"),default="physical")
    parser.add_argument("--cost-profile",type=Path)
    parser.add_argument("--baseline-selection",type=Path)
    parser.add_argument("--source-search",type=Path)
    args = parser.parse_args()
    placement = CpuPlacement(args.cpu_isolation == "physical")
    try:
        with placement:
            run(args,placement)
    finally:
        args.output_dir.mkdir(parents=True,exist_ok=True)
        atomic_json(args.output_dir/"last_cpu_placement.json",placement.report)
        if hasattr(placement,"result_path"):
            atomic_json(placement.result_path,placement.report)


if __name__ == "__main__":
    main()
