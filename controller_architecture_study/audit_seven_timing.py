"""Re-evaluate the declared terminal window for every retained seven-loop trial."""

import csv
from pathlib import Path
import shutil

import numpy as np

from .decision_dataset import ROOT, atomic_json, sha256
from .dynamic_probe_common import read_json
from .framework_runtime import ControllerPool


RUNS=("seven_continuous_timing_20261026_v2","seven_first_update_20261026_v1","seven_event_insert_20261026_v1")
OUTPUT=ROOT/"controller_architecture_study/results/seven_timing_stage_20261026"


def terminal_metrics(trace,duration):
    times=[r["t"] for r in trace]
    first=max(0,int(np.searchsorted(times,duration-.25,side="right"))-1)
    last=int(np.searchsorted(times,duration,side="left"))
    tail=trace[first:last+1]
    gap=float(max(np.diff([r["t"] for r in tail]),default=0.))
    coverage=bool(tail and tail[0]["t"]<=duration-.25 and tail[-1]["t"]>=duration and gap<=.100)
    bounds=bool(tail and all(r["position_error_m"]<=.3 and r["velocity_error_mps"]<=.6 for r in tail))
    return {"coverage":coverage,"state_bounds":bounds,"max_gap_s":gap,"passed":coverage and bounds}


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    provenance=ControllerPool().provenance()
    records,audits=[],[]
    for run in RUNS:
        root=OUTPUT.parent/run
        summary,plan=read_json(root/"summary.json"),read_json(root/"plan.json")
        integrity={n:sha256(root/"frozen_source"/n)==h for n,h in plan["source_sha256"].items()}
        assert summary["complete"] and all(integrity.values()) and plan["provenance"]==provenance
        rows=[]
        for row in summary["rows"]:
            original=row["report"]
            directory=Path(row["directory"])
            traces=read_json(directory/"traces.json")
            terminal={v:terminal_metrics(t,original["duration_s"]) for v,t in traces.items()}
            success=bool(original["physical_task_passed"] and all(t["passed"] for t in terminal.values())
                and all(p["clock_valid"] for p in original["vehicles"].values())
                and original["error"] is None and not original["cleanup_errors"])
            rows.append({"name":row["name"],"original_success":original["success"],"audited_success":success,
                "original_report_unchanged":True,"terminal":terminal})
            critical=f"Drone{original['scenario']['critical_index']+1}" if original["scenario"]["critical_index"]>=0 else "Drone7"
            records.append({"batch":run,"name":row["name"],"scenario":original["scenario"]["id"],
                "policy":original["policy"],"complete":original["full_flight_completed"],
                "physical_passed":original["physical_task_passed"],"original_success":original["success"],
                "audited_success":success,"collisions":original["collisions"],"J":original["total_J"],
                "first_critical_command_ms":original["per_vehicle_timing"][critical]["first_command_ms"],
                "mean_R_ms":original["response_ms"]["mean"],"p99_R_ms":original["response_ms"]["p99"],
                "max_R_ms":original["response_ms"]["max"],"deadline_late":len(original["late_jobs"]),
                "nominal_late":len(original["nominal_late_jobs"]),"B_binding":original["B_binding_count"],
                "extra_requests":original.get("extra_corrective_requests",0),"resumptions":original["segment_resumptions"],
                "mean_analysis_ms":original["selection_ms"]["mean"],"directory":str(directory.resolve())})
        audit={"source_summary_sha256":sha256(root/"summary.json"),"source_snapshot_integrity":integrity,
            "models_unchanged":True,"rows":rows,"rule":"fixed250ms terminal interval through first covering tick; ignore only later out-of-task snapshots",
            "evaluator_sha256":sha256(Path(__file__)),"runtime_window_fix_sha256":sha256(Path(__file__).with_name("framework_runtime_world.py"))}
        atomic_json(root/"terminal_window_audit.json",audit)
        audits.append({"batch":run,"trials":len(rows),"changed":[r["name"] for r in rows if r["original_success"]!=r["audited_success"]],
            "cpu_restored":read_json(root/"cpu_placement.json")["restoration_passed"]})
    with (OUTPUT/"all_trials.csv").open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    atomic_json(OUTPUT/"analysis.json",{"trials":records,"audits":audits,"all_complete":all(r["complete"] for r in records),
        "models_unchanged":True,"FPGA_verified":False,"physical_safety_certified":False})
    comparisons=[]
    latest=[r for r in records if r["batch"]==RUNS[-1] and r["scenario"]=="i6_sign-1_v1.2"]
    for repeat in (1,2):
        rr={r["policy"]:r for r in latest if f"_r{repeat}_" in r["name"]}
        a=read_json(Path(rr["static"]["directory"])/"initial_state.json")
        b=read_json(Path(rr["event_insert"]["directory"])/"initial_state.json")
        paired=a.keys()==b.keys() and all(a[v][k]==b[v][k] for v in a
            for k in ("position","velocity","quaternion","held_control"))
        comparisons.append({"repeat":repeat,"initial_states_match":paired,
            "static_collision":rr["static"]["collisions"]>0,"static_physical_passed":rr["static"]["physical_passed"],
            "event_audited_success":rr["event_insert"]["audited_success"],
            "observed_collision_rescue":paired and rr["static"]["collisions"]>0 and rr["event_insert"]["audited_success"],
            "observed_task_rescue":paired and not rr["static"]["physical_passed"] and rr["event_insert"]["audited_success"],
            "all_budget_deadlines_met":rr["event_insert"]["deadline_late"]==0})
    atomic_json(OUTPUT.parent/RUNS[-1]/"audited_summary.json",{"original_summary_unchanged":True,
        "original_summary_sha256":sha256(OUTPUT.parent/RUNS[-1]/"summary.json"),"pairs":comparisons,
        "observed_collision_rescues":sum(r["observed_collision_rescue"] for r in comparisons),
        "observed_task_rescues":sum(r["observed_task_rescue"] for r in comparisons),
        "formal_safety_certificate":False,"independent_confirmation":False})
    shutil.copy2(Path(__file__),OUTPUT/"audit_seven_timing.py")
    shutil.copy2(Path(__file__).with_name("framework_runtime_world.py"),OUTPUT/"terminal_window_evaluator_source.py")
    print({"complete_flights":sum(r["complete"] for r in records),"total":len(records),"audits":audits})
    for r in records:
        if r["batch"]==RUNS[-1]:
            print(r)


if __name__=="__main__":
    main()
