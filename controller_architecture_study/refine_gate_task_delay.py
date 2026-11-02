"""Profile shorter waits only where immediate recovery passed but80ms did not."""

from copy import deepcopy
from collections import Counter
from pathlib import Path
import sys

from .decision_dataset import ROOT, atomic_json, sha256
from .dynamic_probe_common import read_json
from .framework_cpu_affinity import CpuPlacement
from .framework_runtime import COLLISION_CONTRACT, ControllerPool
from .framework_runtime_calibrate import geometries, rollout
from .framework_runtime_world import FrameworkWorld


BASE = ROOT/"controller_architecture_study/results/framework_gate_task_calibration_v2_direct_collision"
OUTPUT = ROOT/"controller_architecture_study/results/gate_task_delay_refinement_v1"
WAITS = (.020,.030,.040)


def run(placement):
    sys.setswitchinterval(.0005)
    base,plan = read_json(BASE/"mode_map.json"),read_json(BASE/"plan.json")
    if not base["complete"] or base.get("collision_observation") != COLLISION_CONTRACT:
        raise ValueError("complete direct-collision calibration is required")
    pool = ControllerPool()
    if pool.provenance() != base["provenance"]:
        raise ValueError("controller provenance differs")
    geometry = geometries()
    def original(g,ci,model,delay):
        return read_json(BASE/f"{g}_c{ci:02d}_delay{delay:03d}_{model}"/"report.json")
    def passed(r):
        return r["timing_comparable"] and r["recovery_passed"]
    candidates = [(ci,model) for ci in range(len(plan["cases"])) for model in pool.models
        if all(passed(original(g,ci,model,0)) for g,_,_,_ in geometry)
        and not all(passed(original(g,ci,model,80)) for g,_,_,_ in geometry)]
    metadata = {"base_sha256":sha256(BASE/"mode_map.json"),"plan_sha256":sha256(BASE/"plan.json"),
        "source_sha256":{n:sha256(Path(__file__).with_name(n)) for n in
            ("refine_gate_task_delay.py","framework_runtime_calibrate.py","framework_runtime_world.py","framework_runtime.py")},
        "provenance":pool.provenance(),"collision_observation":COLLISION_CONTRACT,"safety_predicate":"gate_task",
        "candidates":candidates,"additional_waits_s":list(WAITS),
        "rule":"largest tested cap for which all tested waits up to that cap passed both geometries",
        "scope":"finite representatives and wait points; NOT an interval/region safety certificate",
        "development_only_not_final_validation":True,"test_read":False}
    import json
    metadata = json.loads(json.dumps(metadata))
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if (OUTPUT/"plan.json").exists() and read_json(OUTPUT/"plan.json") != metadata:
        raise ValueError("retain the previous refinement identity")
    atomic_json(OUTPUT/"plan.json",metadata)
    rows = []
    for g,layout,phase,_ in geometry:
        world = FrameworkWorld(layout,phase,1,2.)
        try:
            for ci,model in candidates:
                for delay in WAITS:
                    directory = OUTPUT/f"{g}_c{ci:02d}_delay{int(delay*1000):03d}_{model}"
                    r = read_json(directory/"report.json") if (directory/"report.json").exists() else rollout(
                        world,pool,model,plan["cases"][ci],delay,directory,enforce_tube=False)
                    rows.append({"geometry":g,"case":ci,"model":model,"delay_s":delay,"report":r})
                    print(f"WAIT_REFINEMENT {len(rows)}/{2*len(WAITS)*len(candidates)} {directory.name} recovery={r['recovery_passed']} timing={r['timing_comparable']}",flush=True)
                    atomic_json(OUTPUT/"summary.json",{"rows":rows,"complete":len(rows)==2*len(WAITS)*len(candidates)})
        finally:
            atomic_json(OUTPUT/"cleanup_report.json",world.close())
    result = deepcopy(base)
    result.update(base_map_sha256=metadata["base_sha256"],delay_refinement_plan_sha256=sha256(OUTPUT/"plan.json"),
        refinement_scope=metadata["scope"],development_only_not_final_validation=True)
    for b in result["bins"].values():
        for cell in b["modes"].values():
            if cell["status"]=="empirical_admissible":
                cell.update(delay_limit_s=.080,validated_wait_samples_s=[0.,.080])
    for ci,model in candidates:
        initial = [original(g,ci,model,0) for g,_,_,_ in geometry]
        additions = [r for r in rows if r["case"]==ci and r["model"]==model]
        cap = None
        for delay in WAITS:
            used = [r["report"] for r in additions if r["delay_s"]<=delay]
            if len(used)==2*sum(d<=delay for d in WAITS) and all(passed(r) for r in used):
                cap = delay
        if cap is not None:
            key = initial[0]["state_bin"]
            cell = result["bins"][key]["modes"][model]
            cell.update(status="empirical_admissible",delay_limit_s=cap,
                validated_wait_samples_s=[0.]+[d for d in WAITS if d<=cap],
                original_80ms_status=base["bins"][key]["modes"][model]["status"],
                out_of_cap_80ms_records_preserved=True,
                within_cap_checked_rows=2+sum(r["delay_s"]<=cap for r in additions))
    result["status_counts"] = dict(Counter(c["status"] for b in result["bins"].values() for c in b["modes"].values()))
    atomic_json(OUTPUT/"mode_map.json",result)
    print(f"WAIT_MAP_READY {OUTPUT/'mode_map.json'}",flush=True)


def main():
    placement = CpuPlacement()
    try:
        with placement:
            run(placement)
    finally:
        OUTPUT.mkdir(parents=True,exist_ok=True)
        atomic_json(OUTPUT/"cpu_placement.json",placement.report)


if __name__=="__main__":
    main()
