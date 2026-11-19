"""Whole-mission static selection versus state-aware control; no injected motion."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys

import numpy as np

from .decision_dataset import ROOT, atomic_json, sha256
from .dynamic_probe_common import read_json
from .fly_decision_dnn import StoredRoutePlanner
from .framework_cpu_affinity import CpuPlacement
from .framework_runtime import ControllerPool, MODELS
from .framework_runtime_editor import read_background_throttle
from .framework_runtime_world import FrameworkWorld
from .inflight_sensing import InflightSensing
from .probe_airsim_three import load_layout
from .probe_natural_demo import observed_completion
from .probe_seven_continuous_timing import flight, matched_initials
from .safety_comparison_core import paper_score


PERIODS=(20.,40.,60.,80.)
SOURCES=("long_route_demo.py","probe_seven_continuous_timing.py","framework_runtime.py",
    "framework_runtime_worker.py","framework_runtime_world.py","inflight_sensing.py",
    "safety_comparison_core.py","safety_comparison_flight.py")


def nominal_timing_passed(report):
    return (not report["nominal_late_jobs"] and not report["nominal_undelivered_deadlines"]
        and report["skipped_releases"]==0 and report["error"] is None)


def build_case(prefix_gates=12, width=2., mid_curve_gates=False):
    layout,digest=load_layout(48,13)
    layout=deepcopy(layout)
    original=deepcopy(layout["segments"])
    p=np.asarray(original[1]["coefficients"],dtype=float)[:,0]
    v=np.asarray(original[1]["coefficients"],dtype=float)[:,1]
    interval=2.
    extra=prefix_gates*interval
    shift=v*extra
    segments=[original[0]]
    prefix=[]
    for i in range(prefix_gates):
        c=np.zeros((3,6));c[:,0]=p+v*i*interval;c[:,1]=v
        segments.append(dict(start_time=original[1]["start_time"]+i*interval,
            duration=interval,coefficients=c.tolist(),heading_orientation=original[1]["heading_orientation"]))
        prefix.append(dict(center=c[:,0].tolist(),orientation=layout["gates"][0]["orientation"]))
    for segment in original[1:]:
        c=np.asarray(segment["coefficients"],dtype=float);c[:,0]+=shift
        segments.append({**segment,"start_time":segment["start_time"]+extra,"coefficients":c.tolist()})
    gates=prefix+[{**g,"center":(np.asarray(g["center"])+shift).tolist()} for g in layout["gates"]]
    layout.update(segments=segments,gates=[{**g,"index":i+1} for i,g in enumerate(gates)],duration=layout["duration"]+extra)
    planner=StoredRoutePlanner(layout)
    if mid_curve_gates:
        # Gate locations come only from the reference, never a controller's trace.
        gates=layout["gates"][:prefix_gates+1]
        for segment in planner.trajectory.segments[prefix_gates+1:prefix_gates+5]:
            point,velocity,_=segment.primitive.evaluate(segment.primitive.duration/2)
            yaw=np.arctan2(velocity[1],velocity[0])
            gates.append(dict(center=point.tolist(),orientation=[float(np.cos(yaw/2)),0.,0.,float(np.sin(yaw/2))]))
        layout["gates"]=[{**g,"index":i+1} for i,g in enumerate(gates)]
    return dict(id=f"long_straight{prefix_gates}_tail13_w{width:g}"+("_midcurve" if mid_curve_gates else ""),layout=layout,source_route_sha256=digest,
        count=7,phase_s=2.1,phase_span_s=2.1,net_width_m=width,duration_s=layout["duration"]+3.,
        prefix_gates=prefix_gates,hard_start_reference_s=original[1]["start_time"]+extra,
        external_impulse=False,initial_tracking_offset=False,
        initialization="nominal reference states, staggered route phases0..2.1s; no later state writes")


def summarize(report,directory,case):
    traces=read_json(directory/"traces.json")
    # Safety is reported independently of the full-window tracking objective.
    # A contact must not turn the objective into a missing value or a shorter run.
    scores=[paper_score(traces[v],case["duration_s"]) for v in report["vehicles"]]
    complete=(report["full_flight_completed"] and report["sensor_window_covered"]
        and all(r["clock_valid"] for r in report["vehicles"].values()) and all(scores)
        and not report["error"] and not report["cleanup_errors"])
    return dict(directory=str(directory),model=report["model"],period_ms=report["period_ms"],
        complete=bool(complete),global_tracking_J=sum(s["J"] for s in scores) if complete else None,
        strict_timing=nominal_timing_passed(report),physical_acceptance=report["success"],
        timing_window="until_contact" if report["contact_stop"] else "full_mission",
        colliding_vehicles=report["collisions"],observed_no_contact_completion=observed_completion(report,case),
        nominal_misses=len(report["nominal_late_jobs"])+len(report["nominal_undelivered_deadlines"])+report["skipped_releases"],
        contact_stop=report["contact_stop"],error=report["error"],response_ms=report["response_ms"])


def execute(world,pool,placement,case,root,name,model,period,*,dynamic=False,modes=None,visible=False):
    directory=root/name
    print(f"LONG_ROUTE {name} model={model} nominal_T={period:g}ms",flush=True)
    sensing=InflightSensing(world.vehicles,world.calibration,placement.layout["io_cpus"][1],None)
    report=flight(world,pool,period,dict(id=name,critical_index=-1),
        "adaptive_twosided" if dynamic else "static",directory,case["duration_s"],placement.layout,
        sensing=sensing,response_reserve_s=case["response_reserve_s"],frontend_order="category",
        resident_worker=True,model=model,mode_policy=modes,stop_on_collision=visible)
    if report["sensor_summary"]["injections"]:
        raise RuntimeError("unexpected injection")
    result=summarize(report,directory,case)
    if result["contact_stop"]:
        result["paused_verified"]=world.client.simIsPause()
        if not result["paused_verified"]:raise RuntimeError("collision stop did not pause the simulator")
    requests=read_json(directory/"requests.json")
    result["model_usage"]={m:sum(r["model"]==m for r in requests) for m in MODELS}
    result["requested_period_range_ms"]=[1000*min(r["requested_period_s"] for r in requests),
        1000*max(r["requested_period_s"] for r in requests)] if requests else None
    print("RESULT",result,flush=True)
    if report["error"] or report["cleanup_errors"]:
        raise RuntimeError("technical failure retained; no automatic retry")
    return result


def pair_matches(root,a,b):
    return matched_initials(read_json(root/a/"initial_state.json"),read_json(root/b/"initial_state.json"))


def search(args,placement,pool):
    root=args.output
    root.mkdir(parents=True,exist_ok=False)
    case=read_json(args.case)["case"] if args.case else build_case(args.prefix_gates,args.width,args.mid_curve_gates)
    case["response_reserve_s"]=read_json(ROOT/"controller_architecture_study/results/feedback_residency_20261027_v2/full_sequence/plan.json")["response_reserve_s"]
    recovery_override=None
    calibration=case.get("recovery_calibration")
    if calibration:
        report_path=Path(calibration["report"]);case_path=Path(calibration["case_file"])
        if sha256(report_path)!=calibration["report_sha256"] or sha256(case_path)!=calibration["case_sha256"]:
            raise ValueError("recovery calibration changed")
        measured=read_json(report_path);measured_case=read_json(case_path)["case"]
        if (not measured["success"] or measured["collisions"] or not measured["full_flight_completed"]
                or any(case[k]!=measured_case[k] for k in ("layout","phase_s","phase_span_s","net_width_m","count"))):
            raise ValueError("recovery calibration does not cover this geometry")
        recovery_override=calibration["model"]
        if recovery_override not in MODELS or measured["mode_policy"]["recovery"]!=recovery_override:
            raise ValueError("calibration names a different recovery mode")
    frozen=root/"frozen_source";frozen.mkdir()
    for n in SOURCES:shutil.copy2(Path(__file__).with_name(n),frozen/n)
    plan=dict(case=case,candidates=[dict(model=m,period_ms=p) for m in MODELS for p in PERIODS],
        objective="sum of full-mission Eq4-form tracking J, common20ms grid and Wx=I; collisions separately reported",
        static_eligibility="complete, clock-valid, zero observed nominal deadline misses; no collision-based filtering",
        selection="maximum whole-mission J in this declared finite homogeneous candidate set; not a global FALCON optimum",
        source_sha256={n:sha256(frozen/n) for n in SOURCES},provenance=pool.provenance(),cpu_layout=placement.layout,
        source_case_sha256=sha256(args.case) if args.case else None,no_external_impulse=True,FPGA_verified=False)
    atomic_json(root/"plan.json",plan)
    world=FrameworkWorld(case["layout"],case["phase_s"],case["count"],case["net_width_m"],
        phase_span=case["phase_span_s"],task_duration=case["duration_s"])
    rows=[]
    try:
        # Interleave models; do not run until a favorable repetition occurs.
        for p in PERIODS:
            for m in MODELS:
                if (root/"STOP").exists():
                    atomic_json(root/"summary.json",dict(status="OPERATOR_STOP",complete=False,rows=rows));return
                name=f"static_h{m.split('_h')[1]}_T{p:g}"
                result=execute(world,pool,placement,case,root,name,m,p)
                rows.append(result)
                atomic_json(root/"progress.json",dict(complete=False,rows=rows))
        eligible=[r for r in rows if r["complete"] and r["strict_timing"]]
        if not eligible:
            atomic_json(root/"summary.json",dict(status="NO_OBSERVED_TIMING_FEASIBLE_STATIC",complete=True,rows=rows));return
        selected=max(eligible,key=lambda r:r["global_tracking_J"])
        # Calibrate the recovery controller using measured late-section tracking,
        # not by assuming greater depth implies greater recovery capability.
        tail_cost={}
        for m in MODELS:
            trials=[r for r in rows if r["model"]==m and r["complete"]]
            if not trials:continue
            trial=min(trials,key=lambda r:r["period_ms"])
            traces=read_json(Path(trial["directory"])/"traces.json")
            tail_cost[m]=float(np.mean([x["state_cost"] for v,t in traces.items() for x in t
                if x["t"]>=case["hard_start_reference_s"]-world.phases[v] and x["t"]<=case["duration_s"]-3.]))
        recovery=recovery_override or min(tail_cost,key=tail_cost.get)
        modes=dict(cruise=selected["model"],recovery=recovery,contract="relative_preview_v2")
        selection=dict(selected=selected,tail_cost=tail_cost,mode_policy=modes,
            frozen_before_confirmation=True,case=case,source_sha256=plan["source_sha256"],provenance=plan["provenance"])
        atomic_json(root/"selection.json",selection)
        print("SELECTED",selected,"RECOVERY",recovery,flush=True)
        pairs=[]
        for repeat,order in enumerate((("static","dynamic"),("dynamic","static")),1):
            results={}
            for arm in order:
                name=f"r{repeat}_{arm}"
                results[arm]=execute(world,pool,placement,case,root,name,selected["model"],selected["period_ms"],
                    dynamic=arm=="dynamic",modes=modes if arm=="dynamic" else None,visible=True)
                rows.append(results[arm])
            matched=pair_matches(root,f"r{repeat}_static",f"r{repeat}_dynamic")
            pairs.append(dict(repeat=repeat,initial_states_match=matched,
                collision_rescue=matched and results["static"]["colliding_vehicles"]>0
                    and results["dynamic"]["physical_acceptance"],
                dynamic_nominal_timing_passed=results["dynamic"]["strict_timing"]))
            atomic_json(root/"progress.json",dict(complete=False,rows=rows,pairs=pairs))
        ready=all(p["collision_rescue"] for p in pairs)
        atomic_json(root/"summary.json",dict(complete=True,status="DEMO_READY" if ready else "NO_REPEATED_RESCUE",
            rows=rows,pairs=pairs,selection=str(root/"selection.json"),full_FALCON=False))
        print("FINAL", "DEMO_READY" if ready else "NO_REPEATED_RESCUE",root/"summary.json",flush=True)
    finally:
        atomic_json(root/"cleanup_report.json",world.close())


def demo(args,placement,pool):
    selection=read_json(args.selection)
    if selection["provenance"]!=pool.provenance():raise ValueError("model artifacts changed")
    for n,digest in selection["source_sha256"].items():
        if sha256(Path(__file__).with_name(n))!=digest:raise ValueError(f"source changed: {n}")
    case=selection["case"];selected=selection["selected"]
    root=args.output;root.mkdir(parents=True,exist_ok=False)
    world=FrameworkWorld(case["layout"],case["phase_s"],case["count"],case["net_width_m"],
        phase_span=case["phase_span_s"],task_duration=case["duration_s"])
    try:
        pairs=[]
        orders=(("static","dynamic"),("dynamic","static")) if args.confirm else (("static","dynamic"),)
        for repeat,order in enumerate(orders,1):
            results={}
            for arm in order:
                if not args.confirm:input(f"准备好观察{arm}后按Enter开始：")
                name=f"r{repeat}_{arm}"
                result=execute(world,pool,placement,case,root,name,selected["model"],selected["period_ms"],
                    dynamic=arm=="dynamic",modes=selection["mode_policy"] if arm=="dynamic" else None,visible=True)
                results[arm]=result
                atomic_json(root/f"{name}_summary.json",result)
                if result["contact_stop"] and not args.confirm:
                    input("已在碰撞处暂停。查看画面后按Enter继续；不会将暂停算进控制性能：")
                world.client.simPause(False)
            contact=results["static"]["contact_stop"]
            phase=case["phase_s"]-case["phase_span_s"]*(int(contact["vehicle"].replace("Drone",""))-1)/(case["count"]-1) if contact else 0.
            pairs.append(dict(repeat=repeat,initial_states_match=pair_matches(root,f"r{repeat}_static",f"r{repeat}_dynamic"),
                static_first_contact_in_hard_tail=bool(contact and contact["observed_t_s"]+phase>=case["hard_start_reference_s"]),
                static=results["static"],dynamic=results["dynamic"]))
            atomic_json(root/"summary.json",dict(complete=repeat==len(orders),pairs=pairs,
                repeated_rescue=all(p["initial_states_match"] and p["static_first_contact_in_hard_tail"]
                    and p["dynamic"]["physical_acceptance"] for p in pairs)))
    finally:
        atomic_json(root/"cleanup_report.json",world.close())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    stage=parser.add_mutually_exclusive_group(required=True)
    stage.add_argument("--search",action="store_true")
    stage.add_argument("--selection",type=Path)
    parser.add_argument("--confirm",action="store_true",help="two noninteractive reverse-order pairs using the frozen selection")
    parser.add_argument("--width",type=float,default=2.)
    parser.add_argument("--prefix-gates",type=int,default=12)
    parser.add_argument("--mid-curve-gates",action="store_true")
    parser.add_argument("--case",type=Path,help="predeclared reference-derived geometry for static search")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if args.case and not args.search:raise ValueError("--case requires --search; demonstrations use --selection")
    args.output=args.output or ROOT/"controller_architecture_study/results"/(
        "long_route_demo_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    sys.setswitchinterval(.0005)
    placement=CpuPlacement()
    try:
        with placement:
            pool=ControllerPool()
            if read_background_throttle()["background_throttle_enabled"]:raise RuntimeError("disable editor background throttling")
            search(args,placement,pool) if args.search else demo(args,placement,pool)
    finally:
        if args.output.exists():atomic_json(args.output/"cpu_placement.json",placement.report)


if __name__=="__main__":main()
