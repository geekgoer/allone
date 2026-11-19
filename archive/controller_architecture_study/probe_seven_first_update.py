"""Separate first-command urgency from sustained period changes, without retuning models."""

import argparse
from pathlib import Path
import shutil
import sys

from .decision_dataset import ROOT, atomic_json, sha256
from .dynamic_probe_common import read_json
from .framework_cpu_affinity import CpuPlacement
from .framework_runtime import ControllerPool
from .framework_runtime_editor import read_background_throttle
from .framework_runtime_world import FrameworkWorld
from .probe_seven_continuous_timing import flight, matched_initials, MODEL
from .state_mode_calibration import moderate_case


SOURCE=ROOT/"controller_architecture_study/results/seven_continuous_timing_20261026_v2"
POLICIES=("static","static_now","edf_only","event_edf","adaptive_now")


def run(root,placement):
    root.mkdir(parents=True,exist_ok=False)
    sys.setswitchinterval(.0005)
    source=read_json(SOURCE/"plan.json")
    selection=read_json(SOURCE/"selection.json")
    pool=ControllerPool()
    editor=read_background_throttle()
    if pool.provenance()!=source["provenance"] or placement.layout!=source["cpu_layout"] or editor["background_throttle_enabled"]:
        raise ValueError("model/CPU/editor resource domain differs")
    p=selection["selected"]["period_ms"]
    cases=[dict(id="i0_sign+1_v1.2",critical_index=0,position_y_m=.05,velocity_y_mps=1.2),
        dict(id="i6_sign-1_v1.2",critical_index=6,position_y_m=-.05,velocity_y_mps=-1.2)]
    frozen=root/"frozen_source"
    frozen.mkdir()
    files=list(source["source_sha256"])+["probe_seven_first_update.py"]
    for n in files:
        shutil.copy2(Path(__file__).with_name(n),frozen/n)
    atomic_json(root/"plan.json",{"source":str(SOURCE),"source_selection_sha256":sha256(SOURCE/"selection.json"),
        "provenance":pool.provenance(),"cpu_layout":placement.layout,"editor":editor,"period_ms":p,"cases":cases,
        "policies":list(POLICIES),"source_sha256":{n:sha256(frozen/n) for n in files},
        "static_now":"same fixed-priority worker and period, all first requests immediately released",
        "event_edf":"all first requests immediate, continuous state urgency changes FIRST deadline only, then fixed period",
        "adaptive_now":"all first requests immediate, continuous period20..40ms and EDF throughout",
        "repeat_rule":"repeat static/static_now/event_edf once for each observed event safety win",
        "fixed_model":MODEL,"no_grid":True,"no_training":True,"development_only":True})
    case,layout=moderate_case()
    world=FrameworkWorld(layout,case["reference_phase_s"],7,2.,phase_span=2.1,task_duration=source["duration_s"])
    rows,pairs=[],[]
    def execute(name,scenario,policy):
        print(f"FIRST_UPDATE {name}",flush=True)
        r=flight(world,pool,p,scenario,policy,root/name,source["duration_s"],placement.layout)
        rows.append({"name":name,"directory":str(root/name),"report":r})
        atomic_json(root/"progress.json",{"rows":rows,"complete":False})
        critical=f"Drone{scenario['critical_index']+1}" if scenario["critical_index"]>=0 else "Drone7"
        print(f"  full={r['full_flight_completed']} success={r['success']} collision={r['collisions']} first={r['per_vehicle_timing'][critical]['first_command_ms']} late={len(r['late_jobs'])} J={r['total_J']} error={r['error']}",flush=True)
        if r["error"] or r["cleanup_errors"]:
            raise RuntimeError("technical failure retained")
        return r
    try:
        normal=dict(id="normal",critical_index=-1,position_y_m=0.,velocity_y_mps=0.)
        for policy in ("static","event_edf"):
            execute(f"normal_{policy}",normal,policy)
        for i,c in enumerate(cases):
            reports,dirs={},{}
            for policy in POLICIES if i==0 else tuple(reversed(POLICIES)):
                name=f"{c['id']}_r1_{policy}"
                reports[policy]=execute(name,c,policy)
                dirs[policy]=str(root/name)
            initials=[read_json(Path(dirs[q])/"initial_state.json") for q in POLICIES]
            paired=all(matched_initials(initials[0],x) for x in initials[1:])
            win=paired and reports["static"]["collisions"]>0 and reports["event_edf"]["success"]
            pairs.append({"scenario":c,"reports":reports,"directories":dirs,"initial_states_match":paired,"event_safety_win":win})
            atomic_json(root/"pairs.json",pairs)
            if win:
                repeat,repeat_dirs={},{}
                for policy in ("event_edf","static_now","static"):
                    name=f"{c['id']}_r2_{policy}"
                    repeat[policy]=execute(name,c,policy)
                    repeat_dirs[policy]=str(root/name)
                initials=[read_json(Path(repeat_dirs[q])/"initial_state.json") for q in repeat]
                paired=all(matched_initials(initials[0],x) for x in initials[1:])
                pairs.append({"scenario":c,"reports":repeat,"directories":repeat_dirs,"repeat":2,
                    "initial_states_match":paired,"event_safety_win":paired and repeat["static"]["collisions"]>0 and repeat["event_edf"]["success"]})
        atomic_json(root/"summary.json",{"rows":rows,"pairs":pairs,"complete":True,
            "event_safety_wins":sum(x["event_safety_win"] for x in pairs),"development_only":True,"FPGA_verified":False})
    finally:
        atomic_json(root/"cleanup_report.json",world.close())
    print(f"COMPLETE {root}",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    placement=CpuPlacement()
    try:
        with placement:
            run(args.output,placement)
    finally:
        if args.output.exists():
            atomic_json(args.output/"cpu_placement.json",placement.report)


if __name__=="__main__":
    main()
