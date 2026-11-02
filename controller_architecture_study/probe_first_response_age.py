"""Check whether the40ms candidate windows survive aging the first input."""

from pathlib import Path

from .decision_dataset import ROOT, atomic_json, sha256
from .dynamic_probe_common import read_json
from .framework_cpu_affinity import CpuPlacement
from .framework_runtime import ControllerPool
from .framework_runtime_world import FrameworkWorld
from .probe_controller_step_recoverability import trial
from .probe_wait_sensitive_states import SOURCE, REFINED
from .state_mode_calibration import moderate_case


OUTPUT = ROOT/"controller_architecture_study/results/mode_first_response_age_v1"


def run():
    refined,plan = read_json(REFINED/"summary.json"),read_json(REFINED/"plan.json")
    if not refined["complete"]:
        raise ValueError("completed wait refinement required")
    candidates = []
    for job in plan["jobs"]:
        rows = [v for v in refined["rows"] if all(v[k]==job[k] for k in job)]
        passed = {v["delay_s"]:v["report"]["control_recovery_passed"] for v in rows}
        if passed.get(.020) and passed.get(.040) and not passed.get(.060):
            candidates.append(next(v for v in rows if v["delay_s"]==.040))
    pool = ControllerPool()
    setup = read_json(SOURCE/"setup.json")
    if pool.provenance() != setup["provenance"]:
        raise ValueError("controller identity differs")
    _,layout = moderate_case()
    OUTPUT.mkdir(parents=True,exist_ok=False)
    atomic_json(OUTPUT/"plan.json",{"refinement_summary_sha256":sha256(REFINED/"summary.json"),
        "candidates":[{k:v[k] for k in ("source_case","velocity_magnitude")} for v in candidates],
        "initial_wait_s":.040,"initial_input_age_s":.040,"future_update_period_s":.040,
        "scope":"first-response stale input only; later steps remain ideal fresh inference",
        "source_sha256":{n:sha256(Path(__file__).with_name(n)) for n in
            ("probe_first_response_age.py","probe_controller_step_recoverability.py",
             "framework_runtime.py","framework_runtime_world.py")},
        "simulation_stepped":True,"end_to_end_realtime_claimed":False,"mode_map_changed":False,"test_read":False})
    rows = []
    for candidate in candidates:
        case = next(c for c in plan["cases"] if c["id"]==candidate["source_case"])
        phase = setup["phase_s"]-setup["physical_phase_span_s"]*case["critical_index"]/(setup["vehicles"]-1)
        world = FrameworkWorld(layout,phase,1,2.)
        try:
            previous = candidate["report"]
            directory = OUTPUT/f"{case['id']}_v{candidate['velocity_magnitude']:g}"
            report = trial(world,pool,previous["model"],previous["scenario"],.040,directory,
                initial_wait=.040,initial_input_age=.040)
            initial = [read_json(p/"initial_state.json")["Drone1"] for p in (Path(candidate["directory"]),directory)]
            match = all(initial[0][k]==initial[1][k] for k in ("position","velocity","quaternion","held_control"))
            report.update(source_case=case["id"],velocity_magnitude=candidate["velocity_magnitude"],initial_state_match=match)
            atomic_json(directory/"report.json",report)
            rows.append(report)
            atomic_json(OUTPUT/"summary.json",{"rows":rows,"complete":len(rows)==len(candidates),
                "simulation_stepped":True,"end_to_end_realtime_claimed":False,"mode_map_changed":False})
            print(f"FIRST_AGE {len(rows)}/{len(candidates)} {directory.name} recovery={report['control_recovery_passed']} matched={match}",flush=True)
            if report["error"] or not match:
                raise RuntimeError(report["error"] or "first-age comparison initial states differ")
        finally:
            atomic_json(OUTPUT/f"cleanup_{case['id']}.json",world.close())


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
