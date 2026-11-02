"""Paired audit of a rejected reachable state before adding map dimensions."""

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np

from .decision_dataset import ROOT, atomic_json, sha256
from .dynamic_probe_common import read_json
from .framework_runtime import COLLISION_CONTRACT, ControllerPool, MODELS, CONTROL_LIMIT
from .framework_runtime_calibrate import geometries, rollout
from .framework_runtime_world import FrameworkWorld
from .state_mode_calibration import moderate_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failure-report",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,default=ROOT/"controller_architecture_study/results/framework_runtime_refinement_v2_direct_collision")
    parser.add_argument("--base-map",type=Path,default=ROOT/"controller_architecture_study/results/gate_task_delay_refinement_v1/runtime_mode_map.json")
    args = parser.parse_args()
    source = args.failure_report.parent
    report = read_json(args.failure_report)
    if not report["abort"] or report["abort"]["kind"] not in ("no_admissible_mode","deadline_expired_before_release"):
        raise ValueError("a captured mode-map rejection or restricted-wait expiry is required")
    setup = read_json(source.parent/"setup.json")
    analyses = read_json(source/"analyses.json")
    traces = read_json(source/"traces.json")
    vehicle = report["abort"]["vehicle"]
    analysis = next(a for a in reversed(analyses) if a["vehicle"]==vehicle)
    if (report["abort"]["kind"] == "deadline_expired_before_release"
            and analysis.get("mode_selection",{}).get("delay_limit_s",CONTROL_LIMIT) >= CONTROL_LIMIT):
        raise ValueError("the expiry was not caused by a stricter empirical mode wait cap")
    state = min(traces[vehicle],key=lambda r:abs(r["sample_s"]-analysis["analysis_s"]))
    index = int(vehicle.removeprefix("Drone"))-1
    phase = setup["phase_s"]-setup["physical_phase_span_s"]*index/max(1,setup["vehicles"]-1)+state["t"]
    _,layout = moderate_case()
    actual = {"relative_initial":{"position_error":(np.array(state["position"])-state["reference_position"]).tolist(),
        "velocity_error":(np.array(state["velocity"])-state["reference_velocity"]).tolist(),
        "quaternion":state["quaternion"],"held_control":analysis["held_control"]}}
    nominal = deepcopy(actual)
    nominal["relative_initial"]["quaternion"] = state["reference_quaternion"]
    nominal["relative_initial"].pop("held_control")
    root = args.output_dir
    root.mkdir(parents=True,exist_ok=True)
    pool = ControllerPool()
    base = read_json(args.base_map)
    if base.get("collision_observation") != COLLISION_CONTRACT or base.get("safety_predicate") != "gate_task":
        raise ValueError("local refinement requires the direct-collision gate-task map")
    plan = {"source_report_sha256":sha256(args.failure_report),"source_report":str(args.failure_report),
        "phase_s":phase,"initial_risk":state["risk"],"conditions":{"captured_attitude_control":actual,
        "nominal_attitude_control":nominal},"delays_s":[0.,CONTROL_LIMIT],"provenance":pool.provenance(),
        "purpose":"same h/hdot with different attitude/held control; decide whether2D coarseness or omitted state matters",
        "base_map_sha256":sha256(args.base_map),"collision_observation":COLLISION_CONTRACT,
        "safety_predicate":"gate_task","not_independent_final_validation":True,"test_read":False}
    if (root/"plan.json").exists() and read_json(root/"plan.json") != plan:
        raise ValueError("refinement identity changed; preserve the previous output")
    atomic_json(root/"plan.json",plan)
    world = FrameworkWorld(layout,phase,1,2.)
    rows = []
    try:
        for condition,scenario in plan["conditions"].items():
            for delay in (0.,CONTROL_LIMIT):
                for model in MODELS:
                    directory = root/f"{condition}_d{int(delay*1000)}_{model}"
                    r = read_json(directory/"report.json") if (directory/"report.json").exists() else rollout(
                        world,pool,model,(state["risk"]["h"],state["risk"]["hdot"]),delay,directory,scenario,enforce_tube=False)
                    rows.append({"condition":condition,"model":model,"delay_s":delay,"report":r})
                    print(f"REFINE {len(rows)}/24 {condition} {model} d={delay} recovery={r['recovery_passed']} timing={r['timing_comparable']}",flush=True)
                    atomic_json(root/"summary.json",{"rows":rows,"complete":len(rows)==24,"source_plan_sha256":sha256(root/"plan.json")})
    finally:
        atomic_json(root/"cleanup_report.json",world.close())
    context_ok = {m for m in MODELS if all(r["report"]["recovery_passed"] and r["report"]["timing_comparable"]
        for r in rows if r["model"]==m)}
    if not context_ok:
        print("LOCAL_REFINEMENT_NOT_JUSTIFIED: no model passed all captured/nominal contexts",flush=True)
        return
    # Preserve the coarse failure label. Only a separately tested small 2D
    # region gets an additional empirical entry; no third state feature.
    local_plan = {"h_range":[.225,.25],"hdot_range":[-.35,-.2],"representative_risk":[.225,-.35],
        "geometry":[g for g,_,_,_ in geometries()],"delays_s":[0.,CONTROL_LIMIT],
        "source_audit_sha256":sha256(root/"summary.json"),"not_a_region_certificate":True}
    atomic_json(root/"local_cell_plan.json",local_plan)
    local_rows = []
    for g,layout,phase,digest in geometries():
        world = FrameworkWorld(layout,phase,1,2.)
        try:
            for delay in (0.,CONTROL_LIMIT):
                for model in MODELS:
                    directory = root/f"local_{g}_d{int(delay*1000)}_{model}"
                    r = read_json(directory/"report.json") if (directory/"report.json").exists() else rollout(
                        world,pool,model,(.225,-.35),delay,directory,enforce_tube=False)
                    local_rows.append({"model":model,"geometry":g,"delay_s":delay,"report":r})
                    print(f"LOCAL_CELL {len(local_rows)}/24 {g} {model} d={delay} recovery={r['recovery_passed']} timing={r['timing_comparable']}",flush=True)
                    atomic_json(root/"local_cell_summary.json",{"rows":local_rows,"complete":len(local_rows)==24})
        finally:
            atomic_json(root/"cleanup_report.json",world.close())
    modes = {}
    for model in MODELS:
        found = [r["report"] for r in local_rows if r["model"]==model]
        modes[model] = {"status":"empirical_admissible" if model in context_ok and len(found)==4 and all(r["recovery_passed"] and r["timing_comparable"] for r in found)
            else "observed_failure" if any(not r["recovery_passed"] and r["timing_comparable"] for r in found) else "unknown",
            "samples":len(found),"successes":sum(r["recovery_passed"] for r in found)}
    if not any(m["status"]=="empirical_admissible" for m in modes.values()):
        print("NO_ADMISSIBLE_LOCAL_MODE; coarse failures preserved",flush=True)
        return
    base.setdefault("local_cells",[]).append({"id":"refined_h225_250_vminus350_200",**local_plan,"modes":modes,
        "evidence_sha256":sha256(root/"local_cell_summary.json")})
    base["base_map_sha256"] = sha256(args.base_map)
    base["refinement_audit_sha256"] = sha256(root/"summary.json")
    base["development_only_not_final_validation"] = True
    atomic_json(root/"mode_map.json",base)
    print(f"REFINED_MAP_READY {root/'mode_map.json'}",flush=True)


if __name__ == "__main__":
    main()
