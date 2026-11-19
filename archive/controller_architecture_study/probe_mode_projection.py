"""Control-only test of whether identical(h,hdot) hides recovery differences."""

from pathlib import Path
import numpy as np

from .decision_dataset import ROOT, atomic_json, sha256
from .dynamic_probe_common import read_json
from .fast_convex_preprocess import FastReferenceWindow
from .fly_decision_dnn import StoredRoutePlanner
from .framework_cpu_affinity import CpuPlacement
from .framework_runtime import COLLISION_CONTRACT, ControllerPool
from .framework_runtime_world import FrameworkWorld
from .probe_controller_step_recoverability import trial
from .state_mode_calibration import moderate_case
from .state_mode_probe import normal_frame


SOURCE = ROOT/"controller_architecture_study/results/framework_runtime_v27_stable_static/n7_net2"
OUTPUT = ROOT/"controller_architecture_study/results/mode_projection_probe_v1"


def run(placement):
    setup,selection,discovery = (read_json(SOURCE/f"{n}.json") for n in ("setup","selection","discovery_summary"))
    if setup.get("collision_observation") != COLLISION_CONTRACT or not discovery["complete"]:
        raise ValueError("a complete direct-collision comparison is required")
    pair = next(p for p in discovery["pairs"] if p["scenario"]["critical_index"]==6
        and abs(p["scenario"]["velocity_y_mps"])==1. and p["static"]["accepted"])
    source = SOURCE/"discovery"/pair["scenario"]["id"]/"static"
    trace = read_json(source/"traces.json")["Drone7"]
    commands = read_json(source/"requests.json")
    post_samples = {r["post_command_observation_s"] for r in commands
        if r["vehicle"]=="Drone7" and r["status"]=="applied" and "post_command_observation_s" in r}
    rows = [r for r in trace if r["sample_s"] in post_samples and .05<r["t"]<.8
        and 0.<r["risk"]["h"]<.25 and -.8<r["risk"]["hdot"]<-.2]
    state = min(rows,key=lambda r:((r["risk"]["h"]-.15)/.1)**2+((r["risk"]["hdot"]+.5)/.5)**2)
    h,hdot = state["risk"]["h"],state["risk"]["hdot"]
    captured_phase = setup["phase_s"]-setup["physical_phase_span_s"]+state["t"]
    case,layout = moderate_case()
    phases = {"captured_phase":captured_phase,"calibration_phase":case["reference_phase_s"]}
    planner = StoredRoutePlanner(layout)
    window = FastReferenceWindow(planner.trajectory,planner.config)
    pool = ControllerPool()
    if pool.provenance() != setup["provenance"]:
        raise ValueError("controller identity differs")
    model = selection["candidate"]["configuration"]["models"]["Drone7"]
    OUTPUT.mkdir(parents=True,exist_ok=False)
    atomic_json(OUTPUT/"plan.json",{"purpose":"test omitted phase/side information in the2D mode projection",
        "source_case":pair["scenario"]["id"],"source_report_sha256":sha256(source/"report.json"),
        "source_trace_sha256":sha256(source/"traces.json"),"source_state":state,"target_h_hdot":[h,hdot],
        "phases":phases,"sides":[1,-1],"initial_waits_s":[0.,.080],"subsequent_period_s":.080,
        "model":model,"provenance":pool.provenance(),"simulation_stepped":True,
        "end_to_end_realtime_claimed":False,"test_read":False,
        "source_sha256":{n:sha256(Path(__file__).with_name(n)) for n in
            ("probe_mode_projection.py","probe_controller_step_recoverability.py","framework_runtime_world.py")}})
    reports = []
    for name,phase in phases.items():
        ref = window.window(phase,horizon=1)[0]
        normal,derivative = normal_frame(ref)
        world = FrameworkWorld(layout,phase,1,2.)
        try:
            for side in (1,-1):
                error = side*normal*(2./2-.5-.025-h)
                dv = normal*(-side*hdot-float(error@derivative))
                scenario = {"relative_initial":{"position_error":error.tolist(),"velocity_error":dv.tolist()}}
                for delay in (0.,.080):
                    directory = OUTPUT/f"{name}_side{side:+d}_delay{1000*delay:g}"
                    r = trial(world,pool,model,scenario,.080,directory,initial_wait=delay)
                    first = read_json(directory/"traces.json")["Drone1"][0]
                    difference = max(abs(first["risk"]["h"]-h),abs(first["risk"]["hdot"]-hdot))
                    r.update(phase_name=name,side=side,initial_projection=first["risk"],
                        initial_projection_max_delta=difference,projection_matched=difference<1e-4)
                    atomic_json(directory/"report.json",r)
                    reports.append(r)
                    atomic_json(OUTPUT/"summary.json",{"rows":reports,"complete":len(reports)==8,
                        "simulation_stepped":True,"end_to_end_realtime_claimed":False})
                    print(f"PROJECTION {len(reports)}/8 {name} side={side} wait={delay} recovery={r['control_recovery_passed']} matched={r['projection_matched']} error={r['error']}",flush=True)
                    if r["error"]:
                        raise RuntimeError(r["error"])
        finally:
            atomic_json(OUTPUT/"cleanup_report.json",world.close())


def main():
    placement = CpuPlacement()
    try:
        with placement:
            run(placement)
    finally:
        if OUTPUT.exists():
            atomic_json(OUTPUT/"cpu_placement.json",placement.report)


if __name__ == "__main__":
    main()
