"""Counterbalanced frontend-order ablation with the same event-driven I/O."""

import argparse
import math
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
from .probe_inflight_events import SOURCE, event_outcomes
from .probe_seven_continuous_timing import flight, matched_initials, calibrated_response_reserve
from .probe_timing_full_route import stats_ms


def normal_metrics(report,requests,period=.040):
    origin=report["wall_origin_s"];end=origin+report["duration_s"]
    by_slot={(r["vehicle"],r["slot"]):r for r in requests}
    expected,missing,late=0,0,0
    planned_response=[];by_vehicle={}
    for i,v in enumerate(report["vehicles"]):
        first=origin+i*period/len(report["vehicles"])
        last=math.floor((end-first-period)/period+1e-9)
        jitter=[]
        for slot in range(last+1):
            expected+=1
            r=by_slot.get((v,slot));deadline=first+(slot+1)*period
            if r is None or r.get("status")!="applied":
                missing+=1
            else:
                late+=r["rpc_return_s"]>deadline
                planned_response.append(r["rpc_return_s"]-(first+slot*period))
                jitter.append(r["job_released_s"]-(first+slot*period))
        by_vehicle[v]=stats_ms(jitter)
    return dict(expected=expected,missing=missing,late=late,effective_misses=missing+late,
        effective_miss_fraction=(missing+late)/expected,planned_response_ms=stats_ms(planned_response),
        release_jitter_ms=by_vehicle,worst_vehicle_release_p99_ms=max(s["p99"] for s in by_vehicle.values() if s))


def run(root,placement):
    root.mkdir(parents=True,exist_ok=False)
    sys.setswitchinterval(.0005)
    source=read_json(SOURCE/"plan.json")
    prior=read_json(ROOT/"controller_architecture_study/results/inflight_events_sustained_20261026_v1/plan.json")
    pool=ControllerPool()
    if pool.provenance()!=source["provenance"] or placement.layout!=source["cpu_layout"]:
        raise ValueError("model or resource domain differs")
    editor=read_background_throttle()
    if editor["background_throttle_enabled"]:
        raise RuntimeError("editor background throttling enabled")
    orders=("priority","category","deadline")
    frozen=root/"frozen_source";frozen.mkdir()
    names=list(source["source_sha256"])+["inflight_sensing.py","probe_inflight_events.py","probe_frontend_order.py"]
    for n in names: shutil.copy2(Path(__file__).with_name(n),frozen/n)
    atomic_json(root/"plan.json",dict(provenance=pool.provenance(),cpu_layout=placement.layout,editor=editor,
        period_ms=40.,backend="static in all normal ablations",orders=orders,
        phase_s=prior["phase_s"],phase_span_s=2.1,duration_s=prior["duration_s"],layout=prior["layout"],net_width_m=2.,
        normal_order=[list(orders),list(reversed(orders))],normal_repeats=2,
        source_sha256={n:sha256(frozen/n) for n in names},
        selection="physical/clock valid in both repeats; lowest effective misses including missing slots, then worst planned-response p99",
        challenge_gate="non-priority winner must improve combined effective misses AND worst-vehicle release p99 versus priority; otherwise stop",
        challenge=dict(vehicle="Drone1",time_s=1.207,delta_y_mps=1.95,repeats=2),
        primary_latency="original planned release through command return; skipped tasks charged as missing",
        cache_and_sensor_policy_unchanged=True,FPGA_verified=False))
    world=FrameworkWorld(prior["layout"],prior["phase_s"],7,2.,phase_span=2.1,task_duration=prior["duration_s"])
    rows=[];reserve=None
    def execute(name,order,policy="static",pulse=None):
        print("FRONTEND",name,order,policy,flush=True)
        directory=root/name
        sensing=InflightSensing(world.vehicles,world.calibration,placement.layout["io_cpus"][1],pulse)
        report=flight(world,pool,40.,dict(id=name,critical_index=-1),policy,directory,prior["duration_s"],placement.layout,
            sensing=sensing,response_reserve_s=reserve,frontend_order=order)
        outcome=event_outcomes(report,directory,pulse)
        metrics=normal_metrics(report,read_json(directory/"requests.json")) if pulse is None and policy=="static" else None
        row=dict(name=name,frontend_order=order,policy=policy,report=report,event_outcomes=outcome,normal_metrics=metrics)
        rows.append(row)
        atomic_json(directory/"event_outcomes.json",outcome)
        atomic_json(root/"progress.json",dict(complete=False,rows=rows))
        print(" full",report["full_flight_completed"],"success",report["success"],"collision",report["collisions"],
            "nominal_late",len(report["nominal_late_jobs"]),"R",report["response_ms"],
            "planned",report["planned_release_to_command_ms"],"slots",metrics,
            "fresh",outcome.get("first_fresh_control_after_impulse_ms"),flush=True)
        if report["error"] or report["cleanup_errors"]:
            raise RuntimeError("technical failure retained, no retry")
        return row
    try:
        for repeat in (1,2):
            for order in orders if repeat==1 else tuple(reversed(orders)):
                execute(f"normal_r{repeat}_{order}",order)
        initials=[read_json(root/r["name"]/"initial_state.json") for r in rows]
        normal_states_match=all(matched_initials(initials[0],x) for x in initials[1:])
        aggregate={}
        for order in orders:
            rr=[r for r in rows if r["frontend_order"]==order]
            aggregate[order]=dict(valid=normal_states_match and all(r["report"]["success"] for r in rr),
                effective_misses=sum(r["normal_metrics"]["effective_misses"] for r in rr),
                planned_p99_ms=max(r["normal_metrics"]["planned_response_ms"]["p99"] for r in rr),
                worst_vehicle_release_p99_ms=max(r["normal_metrics"]["worst_vehicle_release_p99_ms"] for r in rr))
        candidates=[o for o in orders if aggregate[o]["valid"]]
        chosen=min(candidates,key=lambda o:(aggregate[o]["effective_misses"],aggregate[o]["planned_p99_ms"])) if candidates else None
        baseline=aggregate["priority"]
        gain=(chosen is not None and chosen!="priority" and aggregate[chosen]["effective_misses"]<baseline["effective_misses"]
            and aggregate[chosen]["worst_vehicle_release_p99_ms"]<baseline["worst_vehicle_release_p99_ms"])
        atomic_json(root/"selection.json",dict(aggregate=aggregate,normal_initial_states_match=normal_states_match,
            selected=chosen,challenge_allowed=gain,frozen_before_disturbance=True))
        pairs=[]
        if gain:
            reserve=max(calibrated_response_reserve(r["report"]) for r in rows if r["frontend_order"]==chosen)
            pulse=dict(vehicle="Drone1",time_s=1.207,delta_y_mps=1.95)
            policies=("static","edf_only","adaptive_twosided")
            for repeat in (1,2):
                rr={}
                for policy in policies if repeat==1 else tuple(reversed(policies)):
                    rr[policy]=execute(f"challenge_r{repeat}_{policy}",chosen,policy,pulse)
                initials=[read_json(root/r["name"]/"initial_state.json") for r in rr.values()]
                matched=all(matched_initials(initials[0],x) for x in initials[1:])
                injections=[r["event_outcomes"]["injection"]["post_simulation_s"] for r in rr.values()]
                valid=all(r["event_outcomes"]["disturbance_valid"] for r in rr.values()) and max(injections)-min(injections)<=.012
                pairs.append(dict(repeat=repeat,initial_states_match=matched,pulses_match_and_valid=valid,
                    observed_collision_rescue=matched and valid and rr["static"]["report"]["collisions"]>0 and rr["adaptive_twosided"]["report"]["success"]))
        atomic_json(root/"summary.json",dict(complete=True,status="COMPLETE" if gain else "NO_FRONTEND_GAIN",rows=rows,
            aggregate=aggregate,normal_initial_states_match=normal_states_match,
            selected=chosen,response_reserve_s=reserve,pairs=pairs,FPGA_verified=False))
    finally:
        atomic_json(root/"cleanup_report.json",world.close())


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();placement=CpuPlacement()
    try:
        with placement:run(args.output,placement)
    finally:
        if args.output.exists():atomic_json(args.output/"cpu_placement.json",placement.report)


if __name__=="__main__":main()
