"""Single-loop diagnostic: can faster updates recover the declared states?"""

from copy import deepcopy
from pathlib import Path
import sys

from .decision_dataset import ROOT, atomic_json, sha256
from .dynamic_probe_common import read_json
from .framework_cpu_affinity import CpuPlacement
from .framework_runtime import ControllerPool
from .framework_runtime_editor import read_background_throttle
from .framework_runtime_flight import fly
from .framework_runtime_world import FrameworkWorld
from .state_mode_calibration import moderate_case


OUTPUT = ROOT/"controller_architecture_study/results/controller_recoverability_v1"
SOURCE = ROOT/"controller_architecture_study/results/framework_runtime_v23_precise_release/n7_net2"
COSTS = ROOT/"controller_architecture_study/results/framework_runtime_v22_event_executor/cpu_costs_n7_p95_event_development_v1.json"


def run(placement):
    sys.setswitchinterval(.0005)
    source = read_json(SOURCE/"setup.json")
    for name,digest in source["source_sha256"].items():
        if sha256(Path(__file__).with_name(name)) != digest:
            raise ValueError("diagnostic requires the retained v23 engine")
    editor = read_background_throttle()
    if editor["background_throttle_enabled"]:
        raise RuntimeError("background throttle must remain disabled")
    cases = [{"id":"nominal","critical_index":-1,"position_y_m":0.,"velocity_y_mps":0.},
        *[{"id":f"i0_sign-1_v{v:g}","critical_index":0,"position_y_m":-.05,"velocity_y_mps":-v}
          for v in (.6,1.,1.45)]]
    metadata = {"purpose":"single-loop recoverability diagnostic; NOT multi-loop safety advantage",
        "source_sha256":source["source_sha256"],"runner_sha256":sha256(Path(__file__)),
        "vehicles":1,"net_width_m":2.,"duration_s":3.,"periods_ms":[20.,50.],"cases":cases,
        "editor":editor,"costs_sha256":sha256(COSTS),
        "cost_scope":"same retained seven-loop empirical command guard for both periods; no single-loop timing prediction claim",
        "models_unchanged":True,"mode_map_not_used":True,"test_read":False}
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if (OUTPUT/"setup.json").exists() and read_json(OUTPUT/"setup.json") != metadata:
        raise ValueError("protocol changed; retain this experiment")
    atomic_json(OUTPUT/"setup.json",metadata)
    costs = read_json(COSTS)["costs"]
    pool = ControllerPool()
    if pool.provenance() != source["provenance"]:
        raise ValueError("controller identity differs")
    original = read_json(SOURCE/"selection.json")["candidate"]["configuration"]
    cfg = {"models":{"Drone1":original["models"]["Drone1"]},"periods_ms":{"Drone1":50.},
        "priority_order":["Drone1"],"phase":"staggered"}
    case,layout = moderate_case()
    world = None
    results = []
    try:
        for i,scenario in enumerate(cases):
            reports = {}
            for period in ((20.,50.) if i%2==0 else (50.,20.)):
                current = deepcopy(cfg)
                current["periods_ms"]["Drone1"] = period
                directory = OUTPUT/scenario["id"]/f"T{period:g}"
                if (directory/"report.json").exists():
                    report = read_json(directory/"report.json")
                else:
                    if directory.exists():
                        raise ValueError(f"partial trial retained: {directory}")
                    if world is None:
                        world = FrameworkWorld(layout,case["reference_phase_s"],1,2.,phase_span=2.1)
                    print(f"RECOVERY {scenario['id']} T={period:g}",flush=True)
                    report = fly(world,pool,current,scenario,"static",directory,costs=costs,cpu_layout=placement.layout)
                    print(f"  accepted={report['accepted']} collisions={report['collisions']} abort={report['abort']}",flush=True)
                    if report["error"]:
                        raise RuntimeError(report["error"])
                reports[f"T{period:g}"] = report
            initials = [read_json(OUTPUT/scenario["id"]/f"T{p:g}"/"initial_state.json")["Drone1"] for p in (20.,50.)]
            results.append({"scenario":scenario,"reports":reports,
                "initial_state_match":all(initials[0][k]==initials[1][k]
                    for k in ("position","velocity","quaternion","held_control"))})
            atomic_json(OUTPUT/"summary.json",{"cases":results,"complete":len(results)==len(cases),
                "not_multi_loop_comparison":True})
    finally:
        if world is not None:
            atomic_json(OUTPUT/"cleanup_report.json",world.close())
    print(f"RECOVERABILITY_COMPLETE {OUTPUT}",flush=True)


def main():
    placement = CpuPlacement()
    try:
        with placement:
            run(placement)
    finally:
        OUTPUT.mkdir(parents=True,exist_ok=True)
        atomic_json(OUTPUT/"cpu_placement.json",placement.report)


if __name__ == "__main__":
    main()
