"""Re-score all recorded candidates at one observed endpoint, then pair flights."""

import argparse
from pathlib import Path
import sys

from .decision_dataset import atomic_json, sha256
from .dynamic_probe_common import read_json
from .framework_cpu_affinity import CpuPlacement
from .framework_runtime import ControllerPool
from .framework_runtime_editor import read_background_throttle
from .framework_runtime_world import FrameworkWorld
from .long_route_demo import SOURCES, execute, pair_matches
from .safety_comparison_core import paper_score


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    source=args.source;root=args.output
    root.mkdir(parents=True,exist_ok=False)
    plan=read_json(source/"plan.json")
    records=[]
    for candidate in plan["candidates"]:
        directory=source/candidate["name"]
        report=read_json(directory/"report.json")
        if not report["full_flight_completed"] or report["error"] or report["cleanup_errors"]:
            raise RuntimeError("physical control window incomplete; endpoint repair is not applicable")
        if not all(v["clock_valid"] for v in report["vehicles"].values()):
            raise RuntimeError("invalid simulation clock")
        traces=read_json(directory/"traces.json")
        records.append((candidate,directory,report,traces))
    # Independent of score or collision outcome; includes the complete gate task.
    last=min(plan["case"]["duration_s"],*(rows[-1]["t"] for _,_,_,traces in records for rows in traces.values()))
    endpoint=int(last/.42)*.42
    if endpoint<plan["case"]["layout"]["duration"]+.25:
        raise RuntimeError("observations do not cover mission and terminal hold")
    case={**plan["case"],"duration_s":endpoint}
    rows=[]
    for candidate,directory,report,traces in records:
        scores=[paper_score(t,endpoint) for t in traces.values()]
        if not all(scores):raise RuntimeError("common evaluation window incomplete")
        rows.append(dict(directory=str(directory),model=candidate["model"],period_ms=candidate["period_ms"],
            global_tracking_J=sum(s["J"] for s in scores),colliding_vehicles=report["collisions"],
            nominal_misses=len(report["nominal_late_jobs"])+len(report["nominal_undelivered_deadlines"])+report["skipped_releases"],
            original_report_sha256=sha256(directory/"report.json"),traces_sha256=sha256(directory/"traces.json")))
    selected=max(rows,key=lambda r:r["global_tracking_J"])
    policy=dict(cruise=selected["model"],recovery="plain_mlp_h9_w1024_seed0",contract="relative_preview_v2")
    hashes={n:sha256(Path(__file__).with_name(n)) for n in (*SOURCES,Path(__file__).name)}
    selection=dict(case=case,selected=selected,mode_policy=policy,provenance=plan["provenance"],source_sha256=hashes,
        frozen_before_confirmation=True,original_plan=str(source/"plan.json"),original_plan_sha256=sha256(source/"plan.json"),
        selection_scope=plan["selection"],capacity_domain=plan["static_eligibility"],
        endpoint_repair=dict(original_duration_s=plan["case"]["duration_s"],common_duration_s=endpoint,
            rule="floor of shortest observed endpoint across ALL candidates/vehicles to a complete21-point20ms scoring window",
            reason="old sensor stopped by wall time instead of per-vehicle simulation time; no missing endpoint imputation",
            runtime_change="sensor now samples until simulation duration is covered; geometric prefilter only accelerates offline reporting"))
    atomic_json(root/"ranking.json",dict(rows=rows,selection=selection))
    atomic_json(root/"selection.json",selection)
    print("COMMON_ENDPOINT",endpoint,"RANKING",rows,"SELECTED",selected,flush=True)
    del records,traces
    if not selected["colliding_vehicles"]:
        print("Selected static is contact-free; no rescue demonstration claimed.",flush=True);return
    sys.setswitchinterval(.0005)
    placement=CpuPlacement()
    with placement:
        pool=ControllerPool()
        if pool.provenance()!=selection["provenance"]:raise RuntimeError("model provenance changed")
        if read_background_throttle()["background_throttle_enabled"]:raise RuntimeError("background throttle enabled")
        world=FrameworkWorld(case["layout"],case["phase_s"],case["count"],case["net_width_m"],
            phase_span=case["phase_span_s"],task_duration=case["duration_s"])
        try:
            results={}
            for arm in ("dynamic","static"):
                results[arm]=execute(world,pool,placement,case,root,arm,selected["model"],selected["period_ms"],
                    dynamic=arm=="dynamic",modes=policy if arm=="dynamic" else None,visible=True)
                atomic_json(root/"progress.json",dict(results=results))
                if arm=="dynamic" and not results[arm]["physical_acceptance"]:
                    atomic_json(root/"summary.json",dict(status="DYNAMIC_NOT_RECOVERED",results=results));return
                world.client.simPause(False)
            matched=pair_matches(root,"static","dynamic")
            contact=results["static"]["contact_stop"]
            phase=(case["phase_s"]-case["phase_span_s"]*(int(contact["vehicle"].replace("Drone",""))-1)/(case["count"]-1)) if contact else 0.
            tail=bool(contact and contact["observed_t_s"]+phase>=case["hard_start_reference_s"])
            ready=matched and tail and results["dynamic"]["physical_acceptance"]
            atomic_json(root/"summary.json",dict(status="DEMO_READY_ONE_PAIR" if ready else "NO_RESCUE",results=results,
                initial_states_match=matched,static_contact_in_hard_tail=tail,full_FALCON=False,FPGA_verified=False,
                physical_safety_certificate=False,replications=1))
            print("FINAL", "DEMO_READY_ONE_PAIR" if ready else "NO_RESCUE",root/"summary.json",flush=True)
        finally:atomic_json(root/"cleanup_report.json",world.close())
    atomic_json(root/"cpu_placement.json",placement.report)


if __name__=="__main__":main()
