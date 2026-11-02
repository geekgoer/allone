"""Matched control-only follow-up: change the future update period, not the state."""

from pathlib import Path

from .decision_dataset import ROOT, atomic_json, sha256
from .dynamic_probe_common import read_json
from .framework_cpu_affinity import CpuPlacement
from .framework_runtime import ControllerPool
from .framework_runtime_world import FrameworkWorld
from .probe_controller_step_recoverability import trial
from .state_mode_calibration import moderate_case


SOURCE = ROOT/"controller_architecture_study/results/mode_projection_probe_v1"
OUTPUT = ROOT/"controller_architecture_study/results/mode_period_probe_v1"


def run():
    plan, summary = read_json(SOURCE/"plan.json"), read_json(SOURCE/"summary.json")
    if not summary["complete"] or not all(r["projection_matched"] for r in summary["rows"]):
        raise ValueError("complete matched projection trials required")
    pool = ControllerPool()
    if pool.provenance() != plan["provenance"]:
        raise ValueError("controller identity differs")
    _,layout = moderate_case()
    OUTPUT.mkdir(parents=True,exist_ok=False)
    atomic_json(OUTPUT/"plan.json",{"source":str(SOURCE),"source_plan_sha256":sha256(SOURCE/"plan.json"),
        "source_summary_sha256":sha256(SOURCE/"summary.json"),"future_period_s":.040,
        "unchanged_initial_waits_s":plan["initial_waits_s"],"provenance":pool.provenance(),
        "simulation_stepped":True,"end_to_end_realtime_claimed":False,"test_read":False,
        "source_sha256":{n:sha256(Path(__file__).with_name(n)) for n in
            ("probe_mode_period.py","probe_controller_step_recoverability.py","framework_runtime_world.py")}})
    rows = []
    for phase_name,phase in plan["phases"].items():
        world = FrameworkWorld(layout,phase,1,2.)
        try:
            for previous in (r for r in summary["rows"] if r["phase_name"] == phase_name):
                delay,side = previous["requested_initial_wait_s"],previous["side"]
                name = f"{phase_name}_side{side:+d}_delay{1000*delay:g}"
                directory = OUTPUT/name
                report = trial(world,pool,plan["model"],previous["scenario"],.040,directory,initial_wait=delay)
                initial = [read_json(p/"initial_state.json")["Drone1"] for p in (SOURCE/name,directory)]
                measured = [read_json(p/"traces.json")["Drone1"][0] for p in (SOURCE/name,directory)]
                match = all(initial[0][k] == initial[1][k] for k in
                    ("position","velocity","quaternion","held_control")) and all(
                    measured[0][k] == measured[1][k] for k in ("position","velocity","quaternion"))
                report.update(phase_name=phase_name,side=side,initial_state_match=match,
                    previous_recovery=previous["control_recovery_passed"])
                atomic_json(directory/"report.json",report)
                rows.append(report)
                atomic_json(OUTPUT/"summary.json",{"rows":rows,"complete":len(rows)==len(summary["rows"]),
                    "simulation_stepped":True,"end_to_end_realtime_claimed":False})
                print(f"PERIOD {len(rows)}/8 {name} matched={match} T40={report['control_recovery_passed']} T80={previous['control_recovery_passed']} error={report['error']}",flush=True)
                if report["error"] or not match:
                    raise RuntimeError(report["error"] or "initial conditions differ from the T80 trial")
        finally:
            atomic_json(OUTPUT/f"cleanup_{phase_name}.json",world.close())


def main():
    placement = CpuPlacement()
    try:
        with placement:
            run()
    finally:
        if OUTPUT.exists():
            atomic_json(OUTPUT/"cpu_placement.json",placement.report)


if __name__ == "__main__":
    main()
