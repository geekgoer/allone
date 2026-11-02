"""Validate a longer easy prefix using the previously frozen capacity domain."""

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
from .long_route_demo import SOURCES, demo, execute


SOURCE=ROOT/"controller_architecture_study/results/long_route_tail_final_search_20261027_v6"
PREDICTION=ROOT/"controller_architecture_study/results/long_route_prefix_sensitivity_20261027_v7/report.json"


def extend_prefix(source, extra):
    case=deepcopy(source)
    n=case["prefix_gates"]
    segments=case["layout"]["segments"]
    interval=segments[1]["duration"]+extra/n
    start=segments[1]["start_time"]
    p,v=np.asarray(segments[1]["coefficients"],dtype=float)[:,:2].T
    for i,segment in enumerate(segments[1:n+1]):
        c=np.asarray(segment["coefficients"],dtype=float)
        c[:,0]=p+v*i*interval
        segment.update(start_time=start+i*interval,duration=interval,coefficients=c.tolist())
    for segment in segments[n+1:]:
        c=np.asarray(segment["coefficients"],dtype=float);c[:,0]+=v*extra
        segment.update(start_time=segment["start_time"]+extra,coefficients=c.tolist())
    for i,gate in enumerate(case["layout"]["gates"]):
        gate["center"]=(p+v*i*interval if i<=n else np.asarray(gate["center"])+v*extra).tolist()
    case["layout"]["duration"]+=extra
    case["duration_s"]+=extra
    case["hard_start_reference_s"]+=extra
    case["id"]+=f"_extra{extra:g}s"
    case["extra_easy_seconds"]=extra
    # The old full-case calibration hash does not cover this longer flight.
    case.pop("recovery_calibration",None)
    StoredRoutePlanner(case["layout"])
    return case


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extra-seconds",type=float,default=336.)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.extra_seconds<=0 or not np.isclose(args.extra_seconds/42,round(args.extra_seconds/42)):
        raise ValueError("positive multiple of42s preserves release and scoring phases")
    root=args.output;root.mkdir(parents=True,exist_ok=False)
    case=extend_prefix(read_json(SOURCE/"plan.json")["case"],args.extra_seconds)
    prediction=read_json(PREDICTION)
    candidates=[c for c in prediction["candidates"] if c["eligible"]]
    estimate=next(s for s in prediction["sweep"] if s["extra_seconds"]==args.extra_seconds)
    rank={r["name"]:i for i,r in enumerate(estimate["ranking"])}
    candidates.sort(key=lambda c:rank[c["name"]])
    frozen=root/"frozen_source";frozen.mkdir()
    source_names=(*SOURCES,Path(__file__).name)
    for name in source_names:shutil.copy2(Path(__file__).with_name(name),frozen/name)
    sys.setswitchinterval(.0005)
    placement=CpuPlacement()
    with placement:
        pool=ControllerPool()
        if read_background_throttle()["background_throttle_enabled"]:
            raise RuntimeError("disable editor background throttling")
        plan=dict(case=case,candidates=[{k:c[k] for k in ("name","model","period_ms","base_J","effective_misses")} for c in candidates],
            predicted_ranking=estimate,source=str(SOURCE),source_summary_sha256=sha256(SOURCE/"summary.json"),
            prediction_sha256=sha256(PREDICTION),provenance=pool.provenance(),cpu_layout=placement.layout,
            source_sha256={n:sha256(frozen/n) for n in source_names},
            static_eligibility="frozen empirical capacity domain: zero finite-window misses in the prior24-case calibration; new overruns are reported, never erased",
            objective="same full-mission tracking J, measured on the longer flight; no contact-based candidate filtering",
            plan_revision="previous practical126s limit extended to336s after measured steady-cost extrapolation; not preregistered before that analysis",
            selection="best measured complete candidate within this3-point calibrated local domain; not a global FALCON optimum",
            full_FALCON=False,physical_safety_certificate=False)
        atomic_json(root/"plan.json",plan)
        world=FrameworkWorld(case["layout"],case["phase_s"],case["count"],case["net_width_m"],
            phase_span=case["phase_span_s"],task_duration=case["duration_s"])
        rows=[]
        try:
            for c in candidates:
                if (root/"STOP").exists():
                    atomic_json(root/"summary.json",dict(status="OPERATOR_STOP",rows=rows));return
                result=execute(world,pool,placement,case,root,c["name"],c["model"],c["period_ms"])
                rows.append(result)
                atomic_json(root/"progress.json",dict(complete=False,rows=rows))
            complete=[r for r in rows if r["complete"]]
            if len(complete)!=len(candidates):
                atomic_json(root/"summary.json",dict(status="INCOMPLETE_RANKING",rows=rows));return
            selected=max(complete,key=lambda r:r["global_tracking_J"])
            modes=dict(cruise=selected["model"],recovery="plain_mlp_h9_w1024_seed0",contract="relative_preview_v2")
            selection=dict(selected=selected,mode_policy=modes,case=case,source_sha256=plan["source_sha256"],
                provenance=plan["provenance"],frozen_before_confirmation=True,capacity_domain=plan["static_eligibility"],
                selection_scope=plan["selection"])
            atomic_json(root/"selection.json",selection)
            atomic_json(root/"summary.json",dict(status="RANKING_COMPLETE",complete=True,rows=rows,selected=selected))
            print("SELECTED",selected,flush=True)
        finally:
            atomic_json(root/"cleanup_report.json",world.close())
        # Two reverse-order paired repetitions. Each contact pauses the simulation.
        if selected["colliding_vehicles"]:
            demo(argparse.Namespace(selection=root/"selection.json",output=root/"confirmation",confirm=True),placement,pool)
        else:print("Selected static remained contact-free; no rescue demonstration claimed.",flush=True)
    atomic_json(root/"cpu_placement.json",placement.report)


if __name__=="__main__":main()
