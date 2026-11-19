"""Challenge the frozen long-route baseline with a previously excluded period."""

import argparse
from pathlib import Path
import sys

from .decision_dataset import atomic_json, sha256
from .dynamic_probe_common import read_json
from .framework_cpu_affinity import CpuPlacement
from .framework_runtime import ControllerPool
from .framework_runtime_editor import read_background_throttle
from .framework_runtime_world import FrameworkWorld
from .long_route_demo import SOURCES, execute


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--period-ms",type=float,default=60.)
    args=parser.parse_args()
    selection=read_json(args.selection);case=selection["case"]
    root=args.output;root.mkdir(parents=True,exist_ok=False)
    sys.setswitchinterval(.0005)
    placement=CpuPlacement()
    with placement:
        pool=ControllerPool()
        if pool.provenance()!=selection["provenance"]:raise RuntimeError("model provenance changed")
        if read_background_throttle()["background_throttle_enabled"]:raise RuntimeError("background throttle enabled")
        atomic_json(root/"plan.json",dict(case=case,selection=str(args.selection),selection_sha256=sha256(args.selection),
            model=selection["selected"]["model"],period_ms=args.period_ms,provenance=pool.provenance(),cpu_layout=placement.layout,
            source_sha256={n:sha256(Path(__file__).with_name(n)) for n in (*SOURCES,Path(__file__).name)},
            reason="60ms was excluded on2 calibration misses while80ms later had7 misses; test stronger static before claiming an offline optimum"))
        world=FrameworkWorld(case["layout"],case["phase_s"],case["count"],case["net_width_m"],
            phase_span=case["phase_span_s"],task_duration=case["duration_s"])
        try:
            result=execute(world,pool,placement,case,root,"static",selection["selected"]["model"],args.period_ms)
            stronger_safe=bool(result["complete"] and result["physical_acceptance"]
                and result["global_tracking_J"]>selection["selected"]["global_tracking_J"])
            atomic_json(root/"summary.json",dict(result=result,invalidates_restricted_baseline_as_general_optimum=stronger_safe,
                original_J=selection["selected"]["global_tracking_J"],full_FALCON=False))
            print("STRONGER_SAFE_STATIC",stronger_safe,flush=True)
        finally:atomic_json(root/"cleanup_report.json",world.close())
    atomic_json(root/"cpu_placement.json",placement.report)


if __name__=="__main__":main()
