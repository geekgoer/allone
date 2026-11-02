"""Bounded development scan between the observed1.0 and1.4m/s outcomes."""

from pathlib import Path
import argparse

import numpy as np

from .decision_dataset import ROOT, atomic_json, sha256
from .dynamic_probe_common import read_json
from .framework_cpu_affinity import CpuPlacement
from .framework_runtime import ControllerPool
from .framework_runtime_world import FrameworkWorld
from .probe_controller_step_recoverability import trial
from .state_mode_calibration import moderate_case


SOURCE = ROOT/"controller_architecture_study/results/framework_runtime_v27_stable_static/n7_net2"
PREVIOUS = ROOT/"controller_architecture_study/results/initial_recovery_window_v1"
OUTPUT = ROOT/"controller_architecture_study/results/wait_sensitive_states_v1"
REFINED = ROOT/"controller_architecture_study/results/wait_sensitive_states_refinement_v1"


def run(root, refine=False):
    setup,selection = read_json(SOURCE/"setup.json"),read_json(SOURCE/"selection.json")
    previous = read_json(PREVIOUS/"summary.json")
    if not previous["complete"]:
        raise ValueError("finish the original recovery-window comparison first")
    pool = ControllerPool()
    if pool.provenance() != setup["provenance"]:
        raise ValueError("controller identity differs")
    cases = read_json(PREVIOUS/"plan.json")["cases"]
    jobs = [(case,speed) for case in cases for speed in (1.1,1.2,1.3)]
    delays = (0.,.080)
    if refine:
        scan = read_json(OUTPUT/"summary.json")
        if not scan["complete"]:
            raise ValueError("complete development scan required")
        def sensitive(case,speed):
            rows = [v for v in scan["rows"] if v["source_case"]==case["id"] and v["velocity_magnitude"]==speed]
            return len(rows)==2 and next(v["report"]["control_recovery_passed"] for v in rows if v["delay_s"]==0.) and not next(
                v["report"]["control_recovery_passed"] for v in rows if v["delay_s"]==.080)
        jobs = [(case,speed) for case,speed in jobs if sensitive(case,speed)]
        delays = (.020,.040,.060)
    total = len(jobs)*len(delays)
    _,layout = moderate_case()
    root.mkdir(parents=True,exist_ok=False)
    atomic_json(root/"plan.json",{"source_setup_sha256":sha256(SOURCE/"setup.json"),
        "previous_summary_sha256":sha256(PREVIOUS/"summary.json"),"provenance":pool.provenance(),
        "cases":cases,"jobs":[{"source_case":c["id"],"velocity_magnitude":v} for c,v in jobs],
        "initial_waits_s":list(delays),"scan_summary_sha256":sha256(OUTPUT/"summary.json") if refine else None,
        "future_period_s":.040,"development_only":True,"independent_confirmation":False,
        "simulation_stepped":True,"end_to_end_realtime_claimed":False,"mode_map_changed":False,
        "source_sha256":{n:sha256(Path(__file__).with_name(n)) for n in
            ("probe_wait_sensitive_states.py","probe_controller_step_recoverability.py",
             "framework_runtime.py","framework_runtime_world.py")},"test_read":False})
    rows = []
    for case in cases:
        vehicle = f"Drone{case['critical_index']+1}"
        phase = setup["phase_s"]-setup["physical_phase_span_s"]*case["critical_index"]/(setup["vehicles"]-1)
        original = read_json(PREVIOUS/f"{case['id']}_DNN_delay0/report.json")["scenario"]
        model = selection["candidate"]["configuration"]["models"][vehicle]
        world = FrameworkWorld(layout,phase,1,2.)
        try:
            for speed in (v for c,v in jobs if c["id"]==case["id"]):
                scenario = {"relative_initial":dict(original["relative_initial"])}
                scenario["relative_initial"]["velocity_error"] = [0.,float(np.sign(case["velocity_y_mps"])*speed),0.]
                pair = []
                for delay in delays:
                    directory = root/f"{case['id']}_v{speed:g}_delay{1000*delay:g}"
                    r = trial(world,pool,model,scenario,.040,directory,initial_wait=delay)
                    rows.append({"source_case":case["id"],"velocity_magnitude":speed,"delay_s":delay,
                        "directory":str(directory),"report":r})
                    pair.append(read_json(directory/"initial_state.json")["Drone1"])
                    atomic_json(root/"summary.json",{"rows":rows,"complete":len(rows)==total,
                        "development_only":True,"simulation_stepped":True,"end_to_end_realtime_claimed":False})
                    print(f"WAIT_SCAN {len(rows)}/{total} {case['id']} v={speed} wait={1000*delay:g} recovery={r['control_recovery_passed']} error={r['error']}",flush=True)
                    if r["error"]:
                        raise RuntimeError(r["error"])
                if any(pair[0][k] != p[k] for p in pair[1:] for k in ("position","velocity","quaternion","held_control")):
                    raise RuntimeError("wait comparison initial states differ")
        finally:
            atomic_json(root/f"cleanup_{case['id']}.json",world.close())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refine",action="store_true")
    args = parser.parse_args()
    root = REFINED if args.refine else OUTPUT
    placement = CpuPlacement()
    try:
        with placement:
            run(root,args.refine)
    finally:
        if root.exists():
            atomic_json(root/"cpu_placement.json",placement.report)


if __name__ == "__main__":
    main()
