"""Shared20ms sensing, unknown-to-controller in-flight impulses, frozen40ms baseline."""

import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import sys

import numpy as np

from .decision_dataset import ROOT, atomic_json, sha256
from .dynamic_probe_common import read_json
from .fly_decision_dnn import StoredRoutePlanner
from .framework_cpu_affinity import CpuPlacement
from .framework_runtime import ControllerPool
from .framework_runtime_editor import read_background_throttle
from .framework_runtime_world import FrameworkWorld
from .inflight_sensing import InflightSensing, SENSOR_PERIOD
from .probe_seven_continuous_timing import flight, matched_initials, calibrated_response_reserve
from .state_mode_calibration import moderate_case


SOURCE=ROOT/"controller_architecture_study/results/seven_continuous_timing_20261026_v2"
POLICIES=("static","static_now","event_online")


def event_outcomes(report, directory, pulse):
    if pulse is None:
        return {"disturbance_valid":True}
    summary=report.get("sensor_summary") or {}
    injections=summary.get("injections",[])
    if len(injections)!=1:
        return {"disturbance_valid":False,"reason":"missing or repeated injection"}
    injection=injections[0]
    onset=injection["rpc_return_s"]
    requests=read_json(directory/"requests.json")
    corrected=[r for r in requests if r["vehicle"]==pulse["vehicle"] and r.get("sample_s",0.)>=onset and "rpc_return_s" in r]
    detected=[r for r in read_json(directory/"sensor_decisions.json") if r["vehicle"]==pulse["vehicle"] and r["capture_s"]>=onset and r["edge"]]
    return {"disturbance_valid":bool(injection["estimated_pose_writeback_bound_m"]<=.040
                and abs(injection["post_simulation_s"]-pulse["time_s"])<=.030),
        "injection":injection,"first_fresh_control_after_impulse_ms":1000*(corrected[0]["rpc_return_s"]-onset) if corrected else None,
        "first_sensor_event_after_impulse_ms":1000*(detected[0]["received_s"]-onset) if detected else None}


def run(root, placement, boundary_check=False, sustained_check=False):
    root.mkdir(parents=True,exist_ok=False)
    sys.setswitchinterval(.0005)
    pool=ControllerPool()
    source=read_json(SOURCE/"plan.json")
    if pool.provenance()!=source["provenance"] or placement.layout!=source["cpu_layout"]:
        raise ValueError("frozen controller/CPU domain differs")
    editor=read_background_throttle()
    if editor["background_throttle_enabled"]:
        raise RuntimeError("editor background throttling enabled")
    period=read_json(SOURCE/"selection.json")["selected"]["period_ms"]
    ref,layout=moderate_case()
    phase=ref["reference_phase_s"]-1.2
    duration=StoredRoutePlanner(layout).trajectory.duration-(phase-2.1)+3.
    cases=[dict(id="d7_long_wait",vehicle="Drone7",time_s=1.207,delta_y_mps=-.9),
        dict(id="d7_short_wait",vehicle="Drone7",time_s=1.227,delta_y_mps=-.9),
        dict(id="d1_long_wait",vehicle="Drone1",time_s=1.207,delta_y_mps=.9),
        dict(id="d3_long_wait",vehicle="Drone3",time_s=1.897,delta_y_mps=.9)]
    event_policy="event_twosided" if boundary_check else "event_online"
    policies=("static","static_now",event_policy)
    if boundary_check:
        cases=[dict(id="d7_boundary",vehicle="Drone7",time_s=1.207,delta_y_mps=-a) if v==7 else
               dict(id="d1_boundary",vehicle="Drone1",time_s=1.207,delta_y_mps=a)
               for a in (1.8,2.1) for v in (7,1)]
    if sustained_check:
        event_policy="adaptive_twosided"
        policies=("static","critical_fast",event_policy)
        cases=[dict(id="d7_sustained",vehicle="Drone7",time_s=1.207,delta_y_mps=-1.8),
            dict(id="d1_sustained",vehicle="Drone1",time_s=1.207,delta_y_mps=1.95)]
    names=list(source["source_sha256"])+["probe_inflight_events.py","inflight_sensing.py"]
    frozen=root/"frozen_source"
    frozen.mkdir()
    for n in names:
        shutil.copy2(Path(__file__).with_name(n),frozen/n)
    atomic_json(root/"plan.json",{"provenance":pool.provenance(),"editor":editor,"cpu_layout":placement.layout,
        "period_ms":period,"sensor_period_s":SENSOR_PERIOD,"phase_s":phase,"phase_span_s":2.1,
        "duration_s":duration,"layout":layout,"cases":cases,"policies":list(policies),
        "source_selection_sha256":sha256(SOURCE/"selection.json"),
        "sensor_information":"same service+detector for all policies; controller data pipe carries only states",
        "resource_split":{"control_io":placement.layout["io_cpus"][:1],"sensor_and_test_harness":placement.layout["io_cpus"][1:]},
        "disturbance":"simulator velocity impulse preserving last-read pose, NOT physical wind; position writeback estimate must<=40mm",
        "disturbance_time_error_limit_s":.030,"pulse_not_in_control_scenario":True,
        "bracket_rule":("two previously failing cases only; no severity search" if sustained_check else
            "D7/D1 only, test1.8/2.1; if no rescue, at most one midpoint each:1.5 when1.8static fails (previous1.2 passed), else1.95 when2.1static fails; stop otherwise"
            if boundary_check else "if no collision rescue at0.9m/s, long-wait D7/D1 only: reduce to0.6 if both tasks fail, otherwise raise to1.2; one step only"),
        "sustained_feedback":sustained_check,
        "critical_fast_contract":"fixed20ms for the a-priori known disturbed vehicle (Drone7 in normal), 40ms for others; diagnostic oracle, not a state-triggered policy" if sustained_check else None,
        "two_sided_projection":boundary_check or sustained_check,"prior_benign_run":"inflight_events_20261026_v1" if boundary_check else None,
        "repeat_rule":"repeat first up to2 collision-rescue cases once, static/event only; retain every result",
        "development_only":True,"source_sha256":{n:sha256(frozen/n) for n in names}})
    world=FrameworkWorld(layout,phase,7,2.,phase_span=2.1,task_duration=duration)
    rows,pairs=[],[]
    reserve=None

    def execute(name,pulse,policy):
        nonlocal reserve
        # The controller receives a normal initialization and state-only sensor pipe.
        scenario=dict(id=name,critical_index=-1,position_y_m=0.,velocity_y_mps=0.)
        if policy=="critical_fast":
            scenario["fast_vehicle"]=pulse["vehicle"] if pulse else "Drone7"
        service=InflightSensing(world.vehicles,world.calibration,placement.layout["io_cpus"][1],pulse)
        print(f"INFLIGHT {name} policy={policy}",flush=True)
        directory=root/name
        r=flight(world,pool,period,scenario,policy,directory,duration,placement.layout,sensing=service,response_reserve_s=reserve)
        if pulse is None and policy=="static":
            reserve=calibrated_response_reserve(r)
            atomic_json(root/"response_calibration.json",{"response_reserve_s":reserve,
                "source":name,"rule":"1.25*normal capture-to-command p99 +0.5ms","is_WCET":False})
        outcome=event_outcomes(r,directory,pulse)
        record={"name":name,"pulse":pulse,"policy":policy,"directory":str(directory),"report":r,"event_outcomes":outcome}
        rows.append(record)
        atomic_json(directory/"event_outcomes.json",outcome)
        atomic_json(root/"progress.json",{"rows":rows,"complete":False})
        print(f"  full={r['full_flight_completed']} success={r['success']} collisions={r['collisions']} J={r['total_J']} extra={r['extra_corrective_requests']} sensor_age={r.get('sensor_delivery_age_ms')} R={r['response_ms']} valid_pulse={outcome['disturbance_valid']} error={r['error']}",flush=True)
        if r["error"] or r["cleanup_errors"]:
            raise RuntimeError("technical failure retained; no automatic retry")
        return record

    def compare(pulse,index,repeat=1,selected_policies=None):
        selected_policies=policies if selected_policies is None else selected_policies
        rr={}
        for policy in selected_policies if index%2==0 else tuple(reversed(selected_policies)):
            rr[policy]=execute(f"{pulse['id']}_a{abs(pulse['delta_y_mps']):g}_r{repeat}_{policy}",pulse,policy)
        initials=[read_json(Path(v["directory"])/"initial_state.json") for v in rr.values()]
        paired=all(matched_initials(initials[0],v) for v in initials[1:])
        a,b=rr["static"],rr[event_policy]
        valid=all(r["event_outcomes"]["disturbance_valid"] for r in rr.values())
        time_match=max(r["event_outcomes"].get("injection",{}).get("post_simulation_s",float("inf")) for r in rr.values())-min(r["event_outcomes"].get("injection",{}).get("post_simulation_s",float("-inf")) for r in rr.values())<=.012
        result={"pulse":pulse,"repeat":repeat,"records":rr,"initial_states_match":paired,"pulse_times_matched_12ms":time_match,
            "observed_collision_rescue":paired and valid and time_match and a["report"]["collisions"]>0 and b["report"]["success"]}
        pairs.append(result)
        atomic_json(root/"pairs.json",pairs)
        return result

    try:
        normals=[execute("normal_"+p,None,p) for p in policies]
        if not all(r["report"]["success"] for r in normals):
            atomic_json(root/"summary.json",{"complete":True,"status":"NORMAL_SENSOR_BASELINE_FAILED","rows":rows,"pairs":pairs})
            return
        initial=[compare(c,i) for i,c in enumerate(cases)]
        if boundary_check and not any(r["observed_collision_rescue"] for r in initial):
            for i,v in enumerate(("Drone7","Drone1")):
                low=next(r for r in initial if r["pulse"]["vehicle"]==v and abs(r["pulse"]["delta_y_mps"])==1.8)
                high=next(r for r in initial if r["pulse"]["vehicle"]==v and abs(r["pulse"]["delta_y_mps"])==2.1)
                amplitude=1.5 if not low["records"]["static"]["report"]["physical_task_passed"] else (
                    1.95 if not high["records"]["static"]["report"]["physical_task_passed"] else None)
                if amplitude is not None:
                    c=deepcopy(low["pulse"])
                    c["delta_y_mps"]=float(np.sign(c["delta_y_mps"])*amplitude)
                    compare(c,i+4)
        elif not boundary_check and not sustained_check and not any(r["observed_collision_rescue"] for r in initial):
            for i in (0,2):
                base=initial[i]
                c=deepcopy(cases[i])
                both_failed=not base["records"]["static"]["report"]["physical_task_passed"] and not base["records"][event_policy]["report"]["physical_task_passed"]
                c["delta_y_mps"]=float(np.sign(c["delta_y_mps"])*(.6 if both_failed else 1.2))
                compare(c,i+4)
        wins=[p for p in pairs if p["observed_collision_rescue"]][:2]
        for i,win in enumerate(wins):
            compare(win["pulse"],i+1,repeat=2,selected_policies=("static",event_policy))
        atomic_json(root/"summary.json",{"complete":True,"rows":rows,"pairs":pairs,
            "observed_collision_rescues":sum(p["observed_collision_rescue"] for p in pairs),
            "FPGA_verified":False,"physical_safety_certified":False,"development_only":True})
    finally:
        atomic_json(root/"cleanup_report.json",world.close())
    print(f"COMPLETE {root}",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    group=parser.add_mutually_exclusive_group()
    group.add_argument("--boundary-check",action="store_true")
    group.add_argument("--sustained-check",action="store_true")
    args=parser.parse_args()
    placement=CpuPlacement()
    try:
        with placement:
            run(args.output,placement,args.boundary_check,args.sustained_check)
    finally:
        if args.output.exists():
            atomic_json(args.output/"cpu_placement.json",placement.report)


if __name__=="__main__":
    main()
