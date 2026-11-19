"""Natural-route demo/search: no injected motion and no off-reference initial state."""

import argparse
from datetime import datetime, timezone
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
from .inflight_sensing import InflightSensing
from .probe_airsim_three import load_layout
from .probe_seven_continuous_timing import flight, matched_initials
from .probe_timing_full_route import MODEL


PRIOR = ROOT/"controller_architecture_study/results/feedback_residency_20261027_v2/full_sequence"
CANDIDATES = [(15,2.),(13,2.),(15,1.6),(13,1.6),(15,1.3),(13,1.3)]


def natural_case(route, width):
    layout, digest = load_layout(48, route)
    planner = StoredRoutePlanner(layout)
    first = planner.reference_window_for_trajectory(planner.trajectory,elapsed=0.,dt=.1,horizon=1)[0]
    if np.linalg.norm(first.velocity)>1e-8 or np.linalg.norm(first.acceleration)>1e-8:
        raise ValueError("natural demo requires a stationary reference start")
    return dict(id=f"env48_route{route:02d}_width{width:g}", layout=layout,route_sha256=digest,
        count=7,phase_s=0.,phase_span_s=0.,net_width_m=width,
        duration_s=planner.trajectory.duration+3.,model=MODEL,
        static_period_ms=60.,dynamic_period_ms=40.,
        response_reserve_s=read_json(PRIOR/"plan.json")["response_reserve_s"],
        external_impulse=False,initial_tracking_offset=False,
        initialization="all seven stationary aerial reference starts; staggered control releases",
        scope="physical demonstration; fixed-c timing ablation, not an optimized FALCON comparison")


class DemoWorld(FrameworkWorld):
    def initialize(self, scenario, directory):
        self.announced=set()
        return super().initialize(scenario,directory)

    def observe(self, vehicle, raw, collision, sampled):
        row=super().observe(vehicle,raw,collision,sampled)
        if row["collision"] and vehicle not in self.announced:
            self.announced.add(vehicle)
            print(f"COLLISION {vehicle} t={row['t']:.3f}s object={row['collision_object']}",flush=True)
        return row


def observed_completion(report, case):
    # A visible no-contact completion is weaker than the existing body-clearance
    # acceptance. Keep both verdicts; never change physical_report's success.
    if (not report["full_flight_completed"] or report["error"] or report["cleanup_errors"]
            or report["collisions"] or len(report["vehicles"])!=case["count"]):
        return False
    for result in report["vehicles"].values():
        crossed={g["gate"] for g in result["crossings"]
            if abs(g["lateral"])<case["net_width_m"]/2 and abs(g["vertical"])<2.}
        if (result["failure"] or not result["window_covered"] or not result["clock_valid"]
                or not result["tail_tracking_passed"] or result["progress_ratio"] is None
                or result["progress_ratio"]<.8 or crossed!=set(result["expected_gate_ids"])):
            return False
    return True


def paired_result(rows, case):
    static,dynamic=(next(r for r in rows if r["arm"]==a) for a in ("static","dynamic"))
    left=read_json(Path(static["directory"])/"initial_state.json")
    right=read_json(Path(dynamic["directory"])/"initial_state.json")
    match=matched_initials(left,right)
    actual=True
    lt=read_json(Path(static["directory"])/"traces.json")
    rt=read_json(Path(dynamic["directory"])/"traces.json")
    for v in lt:
        for key in ("position","velocity","quaternion","angular_velocity"):
            actual &= bool(np.array_equal(lt[v][0][key],rt[v][0][key]))
    return dict(initial_states_match=match,actual_initial_states_match=actual,
        physical_rescue=match and actual and static["report"]["collisions"]>0 and dynamic["report"]["success"],
        observed_contact_avoidance=match and actual and static["report"]["collisions"]>0
            and observed_completion(dynamic["report"],case),
        dynamic_conservative_acceptance_passed=dynamic["report"]["success"],
        dynamic_strict_timing_passed=dynamic["report"]["observed_strict_timing_passed"],
        no_external_impulse=True)


def pair(case, pool, placement, root, order=("static","dynamic"), pause_between=False, live=False):
    root.mkdir(parents=True,exist_ok=False)
    atomic_json(root/"case.json",case)
    world_type=DemoWorld if live else FrameworkWorld
    world=world_type(case["layout"],case["phase_s"],case["count"],case["net_width_m"],
        phase_span=case["phase_span_s"],task_duration=case["duration_s"])
    rows=[]
    try:
        for i,arm in enumerate(order):
            if i and pause_between:
                input("上一轮已完成。准备好观察后，按Enter开始下一轮：")
            print(f"NATURAL {case['id']} arm={arm} seven drones; no velocity/pose injection",flush=True)
            directory=root/arm
            sensing=InflightSensing(world.vehicles,world.calibration,placement.layout["io_cpus"][1],None)
            report=flight(world,pool,case[f"{arm}_period_ms"],dict(id=case["id"],critical_index=-1),
                "static" if arm=="static" else "adaptive_twosided",directory,case["duration_s"],placement.layout,
                sensing=sensing,response_reserve_s=case["response_reserve_s"],frontend_order="category",
                trace_frontend=True,resident_worker=True,model=case["model"])
            if (report["sensor_summary"] or {}).get("injections"):
                raise RuntimeError("natural flight unexpectedly contained an injection")
            rows.append(dict(arm=arm,directory=str(directory),report=report))
            atomic_json(root/"progress.json",dict(complete=False,rows=rows))
            print(" full",report["full_flight_completed"],"success",report["success"],
                "collisions",report["collisions"],"late",len(report["nominal_late_jobs"]),"J",report["total_J"],
                "observed_no_contact_completion",observed_completion(report,case),flush=True)
            if report["error"] or report["cleanup_errors"]:
                raise RuntimeError("technical failure retained; no automatic retry")
        result=paired_result(rows,case) if len(rows)==2 else dict(no_external_impulse=True)
        summary=dict(complete=True,case=case,rows=rows,**result)
        atomic_json(root/"summary.json",summary)
        return summary
    finally:
        atomic_json(root/"cleanup_report.json",world.close())


def run(args, placement):
    root=args.output
    root.mkdir(parents=True,exist_ok=False)
    pool=ControllerPool()
    source=read_json(PRIOR/"plan.json")
    if pool.provenance()!=source["provenance"] or placement.layout!=source["cpu_layout"]:
        raise ValueError("controller/resource domain changed")
    if read_background_throttle()["background_throttle_enabled"]:
        raise RuntimeError("editor background throttling enabled")
    names=sorted(set(source["source_sha256"])|{Path(__file__).name,"probe_airsim_three.py"})
    frozen=root/"frozen_source"
    frozen.mkdir()
    for n in names:shutil.copy2(Path(__file__).with_name(n),frozen/n)
    atomic_json(root/"plan.json",dict(provenance=pool.provenance(),cpu_layout=placement.layout,
        source_sha256={n:sha256(frozen/n) for n in names},search=args.search,
        candidates=CANDIDATES,initial_tracking_offset=False,external_impulse=False,
        confirmation="one additional reverse-order pair; retain failed confirmation, no same-case retries",
        static_configuration="fixed 60ms from earlier CPU normal selection, not re-optimized for new geometry",
        FPGA_verified=False))
    if args.search:
        rows=[]
        for index,(route,width) in enumerate(CANDIDATES):
            if (root/"STOP").exists():break
            case=natural_case(route,width)
            result=pair(case,pool,placement,root/f"screen_{case['id']}",
                order=("static","dynamic") if index%2==0 else ("dynamic","static"))
            rows.append(result)
            atomic_json(root/"progress.json",dict(complete=False,rows=rows))
            if result["physical_rescue"]:
                confirm=pair(case,pool,placement,root/f"confirm_{case['id']}",
                    order=tuple(reversed([r["arm"] for r in result["rows"]])))
                rows.append(confirm)
                if confirm["physical_rescue"]:
                    atomic_json(root/"selected_case.json",case)
                    atomic_json(root/"summary.json",dict(complete=True,status="NATURAL_PHYSICAL_RESCUE_REPEATED",
                        rows=rows,selected_case=str(root/"selected_case.json"),
                        strict_dynamic_timing_passed=result["dynamic_strict_timing_passed"] and confirm["dynamic_strict_timing_passed"],
                        not_full_FALCON=True,FPGA_verified=False))
                    print("NATURAL_DEMO_READY",root/"selected_case.json",flush=True)
                    return
        atomic_json(root/"summary.json",dict(complete=True,status="NO_REPEATED_NATURAL_RESCUE_FOUND",rows=rows,
            not_full_FALCON=True,FPGA_verified=False))
    else:
        case=read_json(args.case)
        if case.get("external_impulse") is not False or case.get("initial_tracking_offset") is not False:
            raise ValueError("not a natural demonstration case")
        order=(("dynamic","static") if args.dynamic_first else ("static","dynamic")) if args.arm=="both" else (args.arm,)
        result=pair(case,pool,placement,root/"demo",order=order,pause_between=args.pause_between,live=True)
        atomic_json(root/"summary.json",result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--search",action="store_true")
    mode.add_argument("--case",type=Path)
    parser.add_argument("--arm",choices=("both","static","dynamic"),default="both")
    parser.add_argument("--pause-between",action="store_true")
    parser.add_argument("--dynamic-first",action="store_true",help="reverse the paired order for confirmation")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    args.output=args.output or ROOT/"controller_architecture_study/results"/(
        "natural_demo_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    sys.setswitchinterval(.0005)
    placement=CpuPlacement()
    try:
        with placement:run(args,placement)
    finally:
        if args.output.exists():atomic_json(args.output/"cpu_placement.json",placement.report)


if __name__=="__main__":main()
