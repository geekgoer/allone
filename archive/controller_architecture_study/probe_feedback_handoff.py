"""Frozen seven-loop feedback-chain diagnosis and repair comparison."""
import argparse
from pathlib import Path
import shutil
import sys
import threading

from .decision_dataset import ROOT, atomic_json, sha256
from .dynamic_probe_common import read_json
from .framework_cpu_affinity import CpuPlacement
from .framework_runtime import ControllerPool
from .framework_runtime_editor import read_background_throttle
from .framework_runtime_world import FrameworkWorld
from .inflight_sensing import InflightSensing
from .probe_frontend_order import normal_metrics
from .probe_inflight_events import event_outcomes
from .probe_seven_continuous_timing import flight, matched_initials


def run(root,placement,diagnostic,frontend_order,wait_chain=False,normal_only=False,resident_worker=False):
    root.mkdir(parents=True,exist_ok=False)
    sys.setswitchinterval(.0005)
    prior=read_json(ROOT/"controller_architecture_study/results/frontend_order_ablation_20261026_v1/plan.json")
    reserve=read_json(ROOT/"controller_architecture_study/results/frontend_order_ablation_20261026_v1/summary.json")["response_reserve_s"]
    pool=ControllerPool();editor=read_background_throttle()
    if pool.provenance()!=prior["provenance"] or placement.layout!=prior["cpu_layout"]:
        raise ValueError("model/resource domain changed")
    if editor["background_throttle_enabled"]:
        raise RuntimeError("background throttling enabled")
    names=list(prior["source_sha256"])+["probe_feedback_handoff.py","windows_wait_diagnosis.py","windows_worker_residency.py"]
    frozen=root/"frozen_source";frozen.mkdir()
    for n in names:shutil.copy2(Path(__file__).with_name(n),frozen/n)
    pulse=dict(vehicle="Drone1",time_s=1.207,delta_y_mps=1.95)
    sequence=[("normal_static","static",None)]
    sequence+=([("diagnostic_adaptive","adaptive_twosided",pulse)] if diagnostic else
        [("normal_adaptive","adaptive_twosided",None),
         ("r1_static","static",pulse),("r1_adaptive","adaptive_twosided",pulse),
         ("r2_adaptive","adaptive_twosided",pulse),("r2_static","static",pulse)])
    if wait_chain or normal_only:sequence=sequence[:1]
    atomic_json(root/"plan.json",dict(source_sha256={n:sha256(frozen/n) for n in names},
        provenance=pool.provenance(),cpu_layout=placement.layout,editor=editor,diagnostic=diagnostic,
        prior_plan=prior,period_ms=40.,frontend_order=frontend_order,response_reserve_s=reserve,
        sequence=sequence,late_outputs_applied_for_exploration=True,
        purpose="locate/fix feedback handoff; no model, geometry, pulse, or deadline relaxation",
        frontend_pid=placement.parent["pid"],frontend_tid=threading.get_native_id(),
        wait_chain_diagnostic=wait_chain,normal_only=normal_only,resident_worker=resident_worker,FPGA_verified=False))
    world=FrameworkWorld(prior["layout"],prior["phase_s"],7,2.,phase_span=2.1,task_duration=prior["duration_s"])
    rows=[]
    probe=None
    try:
        if wait_chain:
            from .windows_wait_diagnosis import WaitChainProbe
            probe=WaitChainProbe(threading.get_native_id(),root/"wait_chain_samples.json");probe.start()
        for name,policy,pulse in sequence:
            if (root/"STOP").exists():
                atomic_json(root/"summary.json",dict(complete=False,status="OPERATOR_STOP",rows=rows));return
            print("FEEDBACK",name,flush=True)
            d=root/name
            sensing=InflightSensing(world.vehicles,world.calibration,placement.layout["io_cpus"][1],pulse)
            report=flight(world,pool,40.,dict(id=name,critical_index=-1),policy,d,prior["duration_s"],placement.layout,
                sensing=sensing,response_reserve_s=reserve,frontend_order=frontend_order,trace_frontend=True,resident_worker=resident_worker)
            outcome=event_outcomes(report,d,pulse)
            metrics=normal_metrics(report,read_json(d/"requests.json")) if name=="normal_static" else None
            rows.append(dict(name=name,policy=policy,pulse=pulse,report=report,event_outcomes=outcome,normal_metrics=metrics))
            atomic_json(root/"progress.json",dict(complete=False,rows=rows))
            print("full",report["full_flight_completed"],"success",report["success"],"collisions",report["collisions"],
                "nominal_late",len(report["nominal_late_jobs"]),"R",report["response_ms"],
                "fresh_ms",outcome.get("first_fresh_control_after_impulse_ms"),"normal_metrics",metrics,flush=True)
            if report["error"] or report["cleanup_errors"]:
                raise RuntimeError("technical failure retained; no retry")
            if name.startswith("normal") and not report["success"]:
                atomic_json(root/"summary.json",dict(complete=False,status="NORMAL_BASELINE_FAILED",rows=rows));return
            if metrics is not None and metrics["effective_misses"]>178:
                atomic_json(root/"summary.json",dict(complete=False,status="NORMAL_TIMING_REGRESSION",rows=rows));return
        initials=[read_json(root/r["name"]/"initial_state.json") for r in rows]
        pairs=[]
        if not diagnostic and not wait_chain and not normal_only:
            for repeat in (1,2):
                rr={r["policy"]:r for r in rows if r["name"].startswith(f"r{repeat}_")}
                outcomes=[r["event_outcomes"] for r in rr.values()]
                times=[o["injection"]["post_simulation_s"] for o in outcomes]
                valid=all(o["disturbance_valid"] for o in outcomes) and max(times)-min(times)<=.012
                pairs.append(dict(repeat=repeat,pulses_valid_and_matched=valid,
                    collision_rescue=valid and rr["static"]["report"]["collisions"]>0 and rr["adaptive_twosided"]["report"]["success"]))
        atomic_json(root/"summary.json",dict(complete=True,rows=rows,pairs=pairs,
            initial_states_match=all(matched_initials(initials[0],x) for x in initials[1:]),FPGA_verified=False))
    finally:
        if probe is not None:probe.close()
        atomic_json(root/"cleanup_report.json",world.close())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True);parser.add_argument("--diagnostic",action="store_true")
    parser.add_argument("--frontend-order",choices=("priority","category","deadline"),default="category")
    parser.add_argument("--wait-chain",action="store_true")
    parser.add_argument("--normal-only",action="store_true")
    parser.add_argument("--resident-worker",action="store_true")
    args=parser.parse_args();placement=CpuPlacement()
    try:
        with placement:run(args.output,placement,args.diagnostic,args.frontend_order,args.wait_chain,args.normal_only,args.resident_worker)
    finally:
        if args.output.exists():atomic_json(args.output/"cpu_placement.json",placement.report)


if __name__=="__main__":main()
