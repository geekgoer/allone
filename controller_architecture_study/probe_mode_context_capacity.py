"""Compare the six frozen controllers in the two ambiguous reference contexts."""

from pathlib import Path

from .decision_dataset import ROOT, atomic_json, sha256
from .dynamic_probe_common import read_json
from .fast_convex_preprocess import FastReferenceWindow
from .fly_decision_dnn import StoredRoutePlanner
from .framework_cpu_affinity import CpuPlacement
from .framework_runtime import COLLISION_CONTRACT, ControllerPool, MODELS
from .framework_runtime_world import FrameworkWorld
from .probe_controller_step_recoverability import trial
from .state_mode_calibration import moderate_case


SOURCE = ROOT/"controller_architecture_study/results/mode_projection_probe_v1"
PERIOD = ROOT/"controller_architecture_study/results/mode_period_probe_v1"
OUTPUT = ROOT/"controller_architecture_study/results/mode_context_capacity_v1"


def run():
    plan,source = read_json(SOURCE/"plan.json"),read_json(PERIOD/"summary.json")
    if not source["complete"] or not all(r["initial_state_match"] for r in source["rows"]):
        raise ValueError("complete matched40ms context trials required")
    pool = ControllerPool()
    if pool.provenance() != plan["provenance"]:
        raise ValueError("controller identity differs")
    _,layout = moderate_case()
    planner = StoredRoutePlanner(layout)
    window = FastReferenceWindow(planner.trajectory,planner.config)
    OUTPUT.mkdir(parents=True,exist_ok=False)
    atomic_json(OUTPUT/"plan.json",{"source_plan_sha256":sha256(SOURCE/"plan.json"),
        "source_summary_sha256":sha256(PERIOD/"summary.json"),"models":list(MODELS),
        "period_s":.040,"initial_wait_s":0.,"contexts":plan["phases"],"sides":[1,-1],
        "provenance":pool.provenance(),"collision_observation":COLLISION_CONTRACT,
        "reused_model":plan["model"],"simulation_stepped":True,"end_to_end_realtime_claimed":False,
        "purpose":"controller recoverability and context dependence; no mode map approval",
        "source_sha256":{n:sha256(Path(__file__).with_name(n)) for n in
            ("probe_mode_context_capacity.py","probe_controller_step_recoverability.py",
             "framework_runtime.py","framework_runtime_world.py")},"test_read":False})
    rows = []
    for ci,(context,phase) in enumerate(plan["phases"].items()):
        ref = window.window(phase,horizon=1)[0]
        world = FrameworkWorld(layout,phase,1,2.)
        try:
            for si,side in enumerate((1,-1)):
                name = f"{context}_side{side:+d}_delay0"
                previous = read_json(PERIOD/name/"report.json")
                initial = read_json(PERIOD/name/"initial_state.json")["Drone1"]
                measured = read_json(PERIOD/name/"traces.json")["Drone1"][0]
                shift = (ci*2+si)%len(MODELS)
                for model in MODELS[shift:]+MODELS[:shift]:
                    reused = model == plan["model"]
                    directory = PERIOD/name if reused else OUTPUT/f"{context}_side{side:+d}_{model}"
                    report = previous if reused else trial(world,pool,model,previous["scenario"],.040,directory)
                    other = read_json(directory/"initial_state.json")["Drone1"]
                    observed = read_json(directory/"traces.json")["Drone1"][0]
                    matched = all(initial[k] == other[k] for k in
                        ("position","velocity","quaternion","held_control")) and all(
                        measured[k] == observed[k] for k in ("position","velocity","quaternion"))
                    rows.append({"context":context,"side":side,"model":model,"reused":reused,
                        "directory":str(directory),"report_sha256":sha256(directory/"report.json"),
                        "reference_acceleration":ref.acceleration.tolist(),"initial_state_match":matched,
                        "report":report})
                    atomic_json(OUTPUT/"summary.json",{"rows":rows,"complete":len(rows)==24,
                        "simulation_stepped":True,"end_to_end_realtime_claimed":False,
                        "mode_map_changed":False})
                    print(f"CONTEXT {len(rows)}/24 {context} side={side} {model} recovery={report['control_recovery_passed']} matched={matched} reused={reused}",flush=True)
                    if report["error"] or not matched:
                        raise RuntimeError(report["error"] or "model comparison initial conditions differ")
        finally:
            atomic_json(OUTPUT/f"cleanup_{context}.json",world.close())


def main():
    placement = CpuPlacement()
    try:
        with placement:
            run()
    finally:
        if OUTPUT.exists():
            atomic_json(OUTPUT/"cpu_placement.json",placement.report)


if __name__ == "__main__":
    main()
