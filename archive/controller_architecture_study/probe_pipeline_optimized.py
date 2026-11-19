"""Same seven-loop scene after pipeline changes; select T before disturbances."""

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
from .probe_inflight_events import SOURCE, event_outcomes
from .probe_seven_continuous_timing import flight, matched_initials, calibrated_response_reserve, PERIODS_MS


def run(root,placement,fixed_period=None):
    root.mkdir(parents=True,exist_ok=True)
    if (root/"plan.json").exists():
        raise ValueError("experiment already started; use a new output directory")
    sys.setswitchinterval(.0005)
    source=read_json(SOURCE/"plan.json")
    prior=read_json(ROOT/"controller_architecture_study/results/inflight_events_sustained_20261026_v1/plan.json")
    pool=ControllerPool()
    if pool.provenance()!=source["provenance"] or placement.layout!=source["cpu_layout"]:
        raise ValueError("frozen model/CPU resources differ")
    editor=read_background_throttle()
    if editor["background_throttle_enabled"]:
        raise RuntimeError("editor background throttling enabled")
    pulse=dict(id="d1_confirmed",vehicle="Drone1",time_s=1.207,delta_y_mps=1.95)
    policies=("static","edf_only","adaptive_twosided")
    periods=PERIODS_MS if fixed_period is None else (fixed_period,)
    names=list(source["source_sha256"])+["inflight_sensing.py","probe_inflight_events.py","probe_pipeline_optimized.py"]
    frozen=root/"frozen_source"
    frozen.mkdir()
    for name in names:
        shutil.copy2(Path(__file__).with_name(name),frozen/name)
    atomic_json(root/"plan.json",dict(provenance=pool.provenance(),cpu_layout=placement.layout,editor=editor,
        phase_s=prior["phase_s"],duration_s=prior["duration_s"],layout=prior["layout"],net_width_m=2.,phase_span_s=2.1,
        normal_candidates_ms=list(periods),selection="highest normal J among complete/recovered/zero nominal miss candidates; fixed-c finite search" if fixed_period is None else "fixed-period pipeline ablation; no renewed period optimum claim",
        policies=policies,pulse=pulse,repeats=2,repeat_order="static/edf/adaptive then reversed regardless of result",
        calibration="freeze 1.25*selected normal capture-to-command p99 +0.5ms, empirical not WCET",
        input_cache="captured after planned release and latest command, <=20ms old at binding, monotonic state; fallback RPC",
        input_cache_max_age_s=.020,cache_must_follow_release=True,
        completion="event wake, single pipe writer; keep post-command state for budget semantics",
        stale_outputs_retained_for_explicit_exploration=True,FPGA_verified=False,
        source_sha256={n:sha256(frozen/n) for n in names}))
    world=FrameworkWorld(prior["layout"],prior["phase_s"],7,2.,phase_span=2.1,task_duration=prior["duration_s"])
    rows,pairs=[],[]
    reserve=None
    def execute(name,period,policy,disturbance=None):
        service=InflightSensing(world.vehicles,world.calibration,placement.layout["io_cpus"][1],disturbance)
        print("PIPELINE",name,"T",period,flush=True)
        directory=root/name
        report=flight(world,pool,period,dict(id=name,critical_index=-1),policy,directory,prior["duration_s"],
            placement.layout,sensing=service,response_reserve_s=reserve)
        outcome=event_outcomes(report,directory,disturbance)
        record=dict(name=name,period_ms=period,policy=policy,report=report,event_outcomes=outcome,directory=str(directory))
        rows.append(record)
        atomic_json(directory/"event_outcomes.json",outcome)
        atomic_json(root/"progress.json",dict(complete=False,rows=rows,pairs=pairs))
        print(" full",report["full_flight_completed"],"success",report["success"],"collisions",report["collisions"],
            "nominal_miss",len(report["nominal_late_jobs"]),"R",report["response_ms"],
            "capture_to_command",report["capture_to_command_ms"],"cache",report["cached_inputs"],
            "fresh_after_pulse_ms",outcome.get("first_fresh_control_after_impulse_ms"),flush=True)
        if report["error"] or report["cleanup_errors"]:
            raise RuntimeError("technical failure retained; no automatic retry")
        return record
    try:
        candidates=[execute(f"normal_static_T{p:g}",p,"static") for p in periods]
        eligible=[r for r in candidates if r["report"]["success"] and r["report"]["nominal_strict_timing_passed"]]
        if not eligible:
            atomic_json(root/"summary.json",dict(complete=True,status="NO_FEASIBLE_NORMAL_STATIC",rows=rows,pairs=pairs))
            return
        selected=max(eligible,key=lambda r:r["report"]["total_J"])
        period=selected["period_ms"]
        reserve=calibrated_response_reserve(selected["report"])
        atomic_json(root/"selection.json",dict(selected=selected["name"],period_ms=period,response_reserve_s=reserve,
            frozen_before_disturbance=True,is_WCET=False,not_global_FALCON_optimum=True))
        normals=[execute("normal_"+p,period,p) for p in policies[1:]]
        if not all(r["report"]["success"] for r in normals):
            atomic_json(root/"summary.json",dict(complete=True,status="NORMAL_POLICY_FAILED",rows=rows,pairs=pairs))
            return
        for repeat in (1,2):
            rr={}
            for policy in policies if repeat==1 else tuple(reversed(policies)):
                rr[policy]=execute(f"d1_r{repeat}_{policy}",period,policy,pulse)
            initial=[read_json(Path(r["directory"])/"initial_state.json") for r in rr.values()]
            match=all(matched_initials(initial[0],x) for x in initial[1:])
            injections=[r["event_outcomes"]["injection"]["post_simulation_s"] for r in rr.values()]
            time_match=max(injections)-min(injections)<=.012
            valid=all(r["event_outcomes"]["disturbance_valid"] for r in rr.values())
            pairs.append(dict(repeat=repeat,initial_states_match=match,pulse_times_matched_12ms=time_match,pulses_valid=valid,
                observed_collision_rescue=match and time_match and valid and rr["static"]["report"]["collisions"]>0 and rr["adaptive_twosided"]["report"]["success"]))
            atomic_json(root/"pairs.json",pairs)
        atomic_json(root/"summary.json",dict(complete=True,status="COMPLETE",rows=rows,pairs=pairs,
            selected_period_ms=period,response_reserve_s=reserve,FPGA_verified=False,physical_safety_certified=False))
    finally:
        atomic_json(root/"cleanup_report.json",world.close())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--fixed-period-ms",type=float,choices=PERIODS_MS)
    args=parser.parse_args()
    placement=CpuPlacement()
    try:
        with placement:
            run(args.output,placement,args.fixed_period_ms)
    finally:
        if args.output.exists():
            atomic_json(args.output/"cpu_placement.json",placement.report)


if __name__=="__main__":
    main()
