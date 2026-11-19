"""Low-load recovery diagnostic; not a multi-loop algorithm comparison."""

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
from .inflight_sensing import InflightSensing
from .probe_inflight_events import event_outcomes, SOURCE
from .probe_seven_continuous_timing import flight, matched_initials
from .state_mode_calibration import moderate_case


def run(root,placement):
    root.mkdir(parents=True,exist_ok=False)
    sys.setswitchinterval(.0005)
    source=read_json(SOURCE/"plan.json")
    previous=read_json(ROOT/"controller_architecture_study/results/inflight_events_sustained_20261026_v1/plan.json")
    pool=ControllerPool()
    if pool.provenance()!=source["provenance"] or placement.layout!=source["cpu_layout"]:
        raise ValueError("controller or resource domain changed")
    if read_background_throttle()["background_throttle_enabled"]:
        raise RuntimeError("editor background throttling enabled")
    _,layout=moderate_case()
    cases=[dict(id="source_d7",phase_s=previous["phase_s"]-2.1,delta_y_mps=-1.8),
        dict(id="source_d1",phase_s=previous["phase_s"],delta_y_mps=1.95)]
    frozen=root/"frozen_source"
    frozen.mkdir()
    names=list(source["source_sha256"])+["inflight_sensing.py","probe_inflight_events.py","probe_fast_inflight.py"]
    for name in names:
        shutil.copy2(Path(__file__).with_name(name),frozen/name)
    atomic_json(root/"plan.json",dict(cases=cases,periods_ms=[20.,40.],pulse_time_s=1.207,
        duration_s=previous["duration_s"],net_width_m=2.,source_provenance=pool.provenance(),
        scope="one active loop, same relative route phase/impulse, zero release offset; no proof of seven-loop feasibility",
        no_MPC_fallback=True,no_midflight_pause=True,
        repeat_rule="repeat40/20 once only if40collides and20fullyrecovers; no severity search",
        source_sha256={n:sha256(frozen/n) for n in names}))
    rows,pairs=[],[]
    for i,case in enumerate(cases):
        world=FrameworkWorld(layout,case["phase_s"],1,2.,phase_span=0.,task_duration=previous["duration_s"])
        pulse=dict(id=case["id"],vehicle="Drone1",time_s=1.207,delta_y_mps=case["delta_y_mps"])
        def execute(period,repeat=1,disturbed=True):
            name=f"{case['id']}_T{period:g}_r{repeat}" if disturbed else case["id"]+"_normal_T20"
            p=pulse if disturbed else None
            directory=root/name
            sensing=InflightSensing(world.vehicles,world.calibration,placement.layout["io_cpus"][1],p)
            print("FAST INFLIGHT",name,flush=True)
            report=flight(world,pool,period,dict(id=name,critical_index=-1),"static",directory,
                previous["duration_s"],placement.layout,sensing=sensing)
            outcome=event_outcomes(report,directory,p)
            row=dict(name=name,case=case,period_ms=period,directory=str(directory),report=report,event_outcomes=outcome)
            rows.append(row)
            atomic_json(directory/"event_outcomes.json",outcome)
            atomic_json(root/"progress.json",dict(complete=False,rows=rows,pairs=pairs))
            print("full",report["full_flight_completed"],"success",report["success"],"collisions",report["collisions"],
                "R",report["response_ms"],"update",report["per_vehicle_timing"]["Drone1"]["update_gap_ms"],flush=True)
            if report["error"] or report["cleanup_errors"]:
                raise RuntimeError("technical failure retained, no retry")
            return row
        def compare(repeat,periods):
            rr={period:execute(period,repeat) for period in periods}
            a,b=rr[40.],rr[20.]
            matched=matched_initials(read_json(Path(a["directory"])/"initial_state.json"),read_json(Path(b["directory"])/"initial_state.json"))
            times=abs(a["event_outcomes"]["injection"]["post_simulation_s"]-b["event_outcomes"]["injection"]["post_simulation_s"])<=.012
            rescued=matched and times and all(z["event_outcomes"]["disturbance_valid"] for z in rr.values()) and a["report"]["collisions"]>0 and b["report"]["success"]
            result=dict(case=case,repeat=repeat,initial_states_match=matched,pulse_times_match=times,observed_collision_rescue=rescued)
            pairs.append(result)
            return rescued
        try:
            if not execute(20.,disturbed=False)["report"]["success"]:
                raise RuntimeError("normal low-load baseline failed; stop challenges")
            if compare(1,(40.,20.) if i==0 else (20.,40.)):
                compare(2,(20.,40.) if i==0 else (40.,20.))
        finally:
            atomic_json(root/(case["id"]+"_cleanup.json"),world.close())
    atomic_json(root/"summary.json",dict(complete=True,rows=rows,pairs=pairs,FPGA_verified=False))


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
