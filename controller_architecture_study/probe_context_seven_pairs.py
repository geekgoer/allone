"""Three development pairs from the wait-sensitive states, not final confirmation."""

from pathlib import Path
import sys

from .decision_dataset import ROOT, atomic_json, sha256
from .dynamic_probe_common import read_json
from .framework_cpu_affinity import CpuPlacement
from .framework_runtime import COLLISION_CONTRACT, ControllerPool, ModeMap
from .framework_runtime_flight import fly
from .framework_runtime_world import FrameworkWorld
from .run_framework_runtime import clean_win, initial_states_match
from .state_mode_calibration import moderate_case


SOURCE = ROOT/"controller_architecture_study/results/framework_runtime_v43_paired_ingress/n7_net2"
OUTPUT = ROOT/"controller_architecture_study/results/context_seven_development_pairs_v9_20261026"
MODE_MAP = ROOT/"controller_architecture_study/results/context_mode_map_v8/mode_map.json"


def run(placement):
    sys.setswitchinterval(.0005)
    setup,selection = read_json(SOURCE/"setup.json"),read_json(SOURCE/"selection.json")
    if selection["status"]!="READY" or setup["collision_observation"]!=COLLISION_CONTRACT:
        raise ValueError("current-cost normal static AND adaptive paths must pass first")
    if setup["cpu_placement"]!={"enabled":True,"layout":placement.layout}:
        raise ValueError("CPU resource domain differs")
    for name,digest in setup["source_sha256"].items():
        if sha256(Path(__file__).with_name(name))!=digest:
            raise ValueError("runtime changed since the normal pair")
    pool = ControllerPool()
    if pool.provenance()!=setup["provenance"]:
        raise ValueError("controller identity differs")
    modes = ModeMap(MODE_MAP,pool.provenance(),2.)
    if modes.sha256!=setup["mode_map_sha256"]:
        raise ValueError("mode map changed since the normal pair")
    cfg = selection["candidate"]["configuration"]
    cases = [{"id":"i0_pos_v1.1","critical_index":0,"position_y_m":.05,"velocity_y_mps":1.1},
        {"id":"i6_neg_v1.1","critical_index":6,"position_y_m":-.05,"velocity_y_mps":-1.1},
        {"id":"i6_pos_v1.2","critical_index":6,"position_y_m":.05,"velocity_y_mps":1.2}]
    OUTPUT.mkdir(exist_ok=False)
    atomic_json(OUTPUT/"setup.json",{**setup,"normal_source":str(SOURCE),"source_selection_sha256":sha256(SOURCE/"selection.json"),
        "runner_sha256":sha256(__file__),"cases":cases,"development_only":True,
        "independent_confirmation":False,"capacity_adaptation_enabled":True})
    case,layout = moderate_case()
    world = FrameworkWorld(layout,case["reference_phase_s"],7,2.,phase_span=2.1)
    pairs = []
    try:
        for i,scenario in enumerate(cases):
            pair = {"scenario":scenario}
            for policy in (("static","adaptive") if i%2==0 else ("adaptive","static")):
                directory = OUTPUT/scenario["id"]/policy
                r = fly(world,pool,cfg,scenario,policy,directory,mode_map=modes if policy=="adaptive" else None,
                    cpu_layout=placement.layout,costs=setup["costs"])
                pair[policy] = r
                print(f"SEVEN_PAIR {scenario['id']} {policy} accepted={r['accepted']} collisions={r['collisions']} J={r['total_J']} abort={r['abort']} error={r['error']}",flush=True)
                if r["error"]:
                    raise RuntimeError(r["error"])
            pair["initial_state_match"] = initial_states_match(OUTPUT/scenario["id"]/"static",OUTPUT/scenario["id"]/"adaptive")
            pair["clean_safety_win"] = clean_win(pair)
            pairs.append(pair)
            atomic_json(OUTPUT/"summary.json",{"pairs":pairs,"complete":len(pairs)==len(cases),
                "clean_safety_wins":sum(p["clean_safety_win"] for p in pairs),"development_only":True,
                "independent_confirmation":False,"FPGA_verified":False})
    finally:
        atomic_json(OUTPUT/"cleanup_report.json",world.close())


def main():
    placement = CpuPlacement()
    try:
        with placement:
            run(placement)
    finally:
        if OUTPUT.exists():
            atomic_json(OUTPUT/"cpu_placement.json",placement.report)


if __name__=="__main__":
    main()
