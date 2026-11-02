"""Exploratory timing ablation after the complete v23 negative discovery set."""

from copy import deepcopy
from pathlib import Path
import sys

from .decision_dataset import ROOT, atomic_json, sha256
from .dynamic_probe_common import read_json
from .framework_cpu_affinity import CpuPlacement
from .framework_runtime import ControllerPool
from .framework_runtime_editor import read_background_throttle
from .framework_runtime_flight import fly
from .framework_runtime_world import FrameworkWorld
from .state_mode_calibration import moderate_case


OUTPUT = ROOT/"controller_architecture_study/results/state_deadline_isolation_v1"
SOURCE = ROOT/"controller_architecture_study/results/framework_runtime_v23_precise_release/n7_net2"
# Cost development was frozen with the same execution engine before v23's timer fix.
COSTS = ROOT/"controller_architecture_study/results/framework_runtime_v22_event_executor/cpu_costs_n7_p95_event_development_v1.json"


def run(placement):
    sys.setswitchinterval(.0005)
    editor = read_background_throttle()
    if editor["background_throttle_enabled"]:
        raise RuntimeError("editor background throttle must remain disabled")
    selection,source_setup = read_json(SOURCE/"selection.json"),read_json(SOURCE/"setup.json")
    if selection["status"] != "READY":
        raise ValueError("complete the full-framework normal gate first")
    for name,digest in source_setup["source_sha256"].items():
        if sha256(Path(__file__).with_name(name)) != digest:
            raise ValueError("core runtime changed; this ablation requires the frozen v23 engine")
    profile = read_json(COSTS)
    if profile["vehicles"] != 7 or profile["cpu_placement"]["layout"] != placement.layout:
        raise ValueError("CPU prediction domain differs")
    costs = {**profile["costs"],"tail_costs":{
        k:max(.001,1.25*s["max"]+.0005) for k,s in profile["statistics_s"].items()}}
    cfg = deepcopy(selection["candidate"]["configuration"])
    cases = [{"id":f"i{i}_sign{sign:+d}_v{speed:g}","critical_index":i,
        "position_y_m":.05*sign,"velocity_y_mps":speed*sign}
        for i in (0,6) for sign in (-1,1) for speed in (1.45,1.50,1.55)]
    metadata = {"purpose":"timing-only diagnostic, NOT full-framework validation or complete FALCON",
        "baseline_selection_sha256":sha256(SOURCE/"selection.json"),"costs_sha256":sha256(COSTS),
        "core_source_sha256":source_setup["source_sha256"],"runner_sha256":sha256(Path(__file__)),
        "vehicles":7,"net_width_m":2.,"duration_s":3.,"fine_periods_ms":[50.,40.,30.,45.,35.],
        "cases":cases,"editor":editor,"models_unchanged":True,"mode_map_bypassed_for_ablation":True,
        "no_global_timer_change":True,"no_MPC_fallback":True,"test_read":False}
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if (OUTPUT/"setup.json").exists() and read_json(OUTPUT/"setup.json") != metadata:
        raise ValueError("ablation protocol changed; preserve results and choose a new experiment")
    atomic_json(OUTPUT/"setup.json",metadata)
    pool = ControllerPool()
    if pool.provenance() != source_setup["provenance"]:
        raise ValueError("controller provenance differs")
    case,layout = moderate_case()
    world = None
    nominal = {"id":"nominal","critical_index":-1,"position_y_m":0.,"velocity_y_mps":0.}

    def trial(key,config,scenario,policy):
        nonlocal world
        directory = OUTPUT/key
        if (directory/"report.json").exists():
            return read_json(directory/"report.json")
        if directory.exists():
            raise ValueError(f"partial trial retained: {directory}")
        if world is None:
            world = FrameworkWorld(layout,case["reference_phase_s"],7,2.,phase_span=2.1)
        print(f"ISOLATION {key} policy={policy}",flush=True)
        report = fly(world,pool,config,scenario,policy,directory,cpu_layout=placement.layout,costs=costs)
        print(f"  accepted={report['accepted']} collisions={report['collisions']} J={report['total_J']} abort={report['abort']}",flush=True)
        if report["error"]:
            raise RuntimeError(report["error"])
        return report

    try:
        candidates = []
        for period in metadata["fine_periods_ms"]:
            current = deepcopy(cfg)
            current["periods_ms"] = {v:period for v in current["periods_ms"]}
            report = trial(f"normal/static_T{period:g}",current,nominal,"static")
            candidates.append({"configuration":current,"report":report,"score":report["total_J"] if report["accepted"] else None})
            atomic_json(OUTPUT/"baseline_refinement.json",candidates)
        valid = [r for r in candidates if r["score"] is not None]
        if not valid:
            raise RuntimeError("no feasible refined static baseline; no challenge started")
        best = max(valid,key=lambda r:r["score"])
        normal = trial("normal/state_deadline",best["configuration"],nominal,"deadline_only")
        atomic_json(OUTPUT/"selection.json",{"candidate":best,"state_deadline_nominal":normal,
            "not_global_optimum":True,"not_complete_FALCON":True})
        if not normal["accepted"]:
            raise RuntimeError("timing-only normal control failed; no challenge started")
        pairs = []
        for index,scenario in enumerate(cases):
            pair = {"scenario":scenario}
            for policy in (("static","deadline_only") if index%2==0 else ("deadline_only","static")):
                pair[policy] = trial(f"discovery/{scenario['id']}/{policy}",best["configuration"],scenario,policy)
            left,right = (read_json(OUTPUT/"discovery"/scenario["id"]/p/"initial_state.json") for p in ("static","deadline_only"))
            pair["initial_state_match"] = all(left[v][k]==right[v][k] for v in left
                for k in ("position","velocity","quaternion","held_control","matched"))
            baseline,dynamic = pair["static"],pair["deadline_only"]
            pair["clean_timing_only_win"] = bool(pair["initial_state_match"] and baseline["collisions"]>0
                and baseline["timing_passed"] and baseline["clock_valid"] and dynamic["accepted"])
            pairs.append(pair)
            atomic_json(OUTPUT/"discovery_summary.json",{"pairs":pairs,"complete":len(pairs)==len(cases),
                "clean_timing_only_wins":sum(p["clean_timing_only_win"] for p in pairs),"not_full_framework":True})
    finally:
        if world is not None:
            atomic_json(OUTPUT/"cleanup_report.json",world.close())
    print(f"TIMING_ISOLATION_COMPLETE {OUTPUT}",flush=True)


def main():
    placement = CpuPlacement()
    try:
        with placement:
            run(placement)
    finally:
        OUTPUT.mkdir(parents=True,exist_ok=True)
        atomic_json(OUTPUT/"cpu_placement.json",placement.report)


if __name__=="__main__":
    main()
