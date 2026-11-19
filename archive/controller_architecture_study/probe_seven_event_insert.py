"""One corrective request for a closing state, then the unchanged staggered calendar."""

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
from .probe_seven_continuous_timing import flight, matched_initials
from .probe_seven_first_update import SOURCE
from .state_mode_calibration import moderate_case


def run(root,placement):
    root.mkdir(parents=True,exist_ok=False)
    sys.setswitchinterval(.0005)
    source=read_json(SOURCE/"plan.json")
    pool=ControllerPool()
    editor=read_background_throttle()
    if pool.provenance()!=source["provenance"] or placement.layout!=source["cpu_layout"] or editor["background_throttle_enabled"]:
        raise ValueError("resource or model identity differs")
    period=read_json(SOURCE/"selection.json")["selected"]["period_ms"]
    case=dict(id="i6_sign-1_v1.2",critical_index=6,position_y_m=-.05,velocity_y_mps=-1.2)
    normal=dict(id="normal",critical_index=-1,position_y_m=0.,velocity_y_mps=0.)
    control=dict(id="i0_sign+1_v1.2",critical_index=0,position_y_m=.05,velocity_y_mps=1.2)
    jobs=[(normal,p,0) for p in ("static","event_insert","static_now")]
    jobs += [(case,p,rep) for rep in (1,2) for p in
        (("static","event_insert","static_now") if rep==1 else ("static_now","event_insert","static"))]
    jobs.append((control,"event_insert",3))
    frozen=root/"frozen_source"
    frozen.mkdir()
    names=list(source["source_sha256"])+["probe_seven_event_insert.py"]
    for n in names:
        shutil.copy2(Path(__file__).with_name(n),frozen/n)
    atomic_json(root/"plan.json",{"source_selection_sha256":sha256(SOURCE/"selection.json"),"period_ms":period,
        "provenance":pool.provenance(),"cpu_layout":placement.layout,"editor":editor,
        "jobs":[{"scenario":s,"policy":p,"repeat":rep} for s,p,rep in jobs],
        "event_rule":"if h+0.5*min(hdot,0)<=0 at first observed state, release one corrective job immediately; EDF first deadline from continuous urgency",
        "following_jobs":"return to original staggered 40ms calendar; minimum request interval20ms",
        "B_role":"same distance/rate budget caps later EDF deadlines; violations recorded, not a certificate",
        "development_only":True,"source_sha256":{n:sha256(frozen/n) for n in names}})
    ref,layout=moderate_case()
    world=FrameworkWorld(layout,ref["reference_phase_s"],7,2.,phase_span=2.1,task_duration=source["duration_s"])
    rows=[]
    try:
        for s,p,rep in jobs:
            name=f"{s['id']}_r{rep}_{p}"
            print(f"INSERT_EVENT {name}",flush=True)
            r=flight(world,pool,period,s,p,root/name,source["duration_s"],placement.layout)
            rows.append({"name":name,"directory":str(root/name),"repeat":rep,"report":r})
            atomic_json(root/"progress.json",{"rows":rows,"complete":False})
            v=f"Drone{s['critical_index']+1}" if s["critical_index"]>=0 else "Drone7"
            print(f"  full={r['full_flight_completed']} success={r['success']} collisions={r['collisions']} first={r['per_vehicle_timing'][v]['first_command_ms']} extra={r['extra_corrective_requests']} R={r['response_ms']} J={r['total_J']} late={len(r['late_jobs'])} error={r['error']}",flush=True)
            if r["error"] or r["cleanup_errors"]:
                raise RuntimeError("technical failure retained")
        pairs=[]
        for rep in (1,2):
            rr=[x for x in rows if x["report"]["scenario"]["id"]==case["id"] and x["repeat"]==rep]
            records={x["report"]["policy"]:x for x in rr}
            initials=[read_json(Path(x["directory"])/"initial_state.json") for x in rr]
            paired=all(matched_initials(initials[0],x) for x in initials[1:])
            pairs.append({"repeat":rep,"initial_states_match":paired,
                "observed_safety_win":paired and records["static"]["report"]["collisions"]>0 and records["event_insert"]["report"]["success"]})
        atomic_json(root/"summary.json",{"rows":rows,"pairs":pairs,"complete":True,
            "observed_safety_wins":sum(x["observed_safety_win"] for x in pairs),"FPGA_verified":False,"development_only":True})
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
