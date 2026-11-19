"""Single-loop teacher/DNN capability check; not a multi-loop comparison."""

import argparse
from pathlib import Path

from .convex_teacher import ConvexTeacher
from .decision_dataset import atomic_json
from .dynamic_probe_common import read_json
from .framework_cpu_affinity import CpuPlacement
from .framework_runtime import ControllerPool
from .framework_runtime_world import FrameworkWorld
from .probe_timing_full_route import trial


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    case=read_json(args.selection)["case"]
    placement=CpuPlacement()
    with placement:
        pool=ControllerPool()
        teacher=ConvexTeacher()
        world=FrameworkWorld(case["layout"],case["phase_s"],1,case["net_width_m"],task_duration=case["duration_s"])
        rows=[]
        try:
            for name,period,controller in (("teacher_T10",10.,teacher),("dnn_T20",20.,None)):
                print("CAPABILITY",name,flush=True)
                report=trial(world,pool,dict(id=name,critical_index=-1),
                    dict(period_ms=period,response_floor_ms=0.,release_phase_fraction=0.),
                    args.output/name,case["duration_s"],teacher=controller)
                rows.append(dict(name=name,report=report))
                atomic_json(args.output/"summary.json",dict(complete=len(rows)==2,rows=rows,
                    multi_UAV=False,injected_response_delay=False,teacher_only_is_not_a_framework_rescue=True))
                print("RESULT",name,"success",report["success"],"contact",report["collision_observed"],
                    "R",report["response_ms"],"error",report["error"],flush=True)
                if report["error"]:break
        finally:
            atomic_json(args.output/"cleanup_report.json",world.close())
    atomic_json(args.output/"cpu_placement.json",placement.report)


if __name__=="__main__":main()
