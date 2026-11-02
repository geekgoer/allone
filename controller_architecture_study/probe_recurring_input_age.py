"""Fixed actuation cadence with stale inputs on EVERY control update."""

from pathlib import Path
import time

import numpy as np

from control_adapter import send_control
from .decision_dataset import ROOT, atomic_json, sha256
from .dynamic_probe_common import read_json
from .framework_cpu_affinity import CpuPlacement
from .framework_runtime import ControllerPool
from .framework_runtime_features import encode_direct
from .framework_runtime_world import FrameworkWorld
from .probe_controller_step_recoverability import advance
from .probe_wait_sensitive_states import SOURCE, REFINED
from .state_mode_calibration import moderate_case


OUTPUT = ROOT/"controller_architecture_study/results/recurring_input_age_v2"
PROFILES = {"fresh":(0.,),"age10":(.010,),"age25":(.025,),"jitter5_30":(.005,.030)}
CADENCE = .035  # Leave physics-step granularity margin below the40ms candidate cap.


def advance_to(world, raw, target):
    origin = world.initial["Drone1"]["sim_ns"]
    for _ in range(8):
        elapsed = (raw.timestamp_ns-origin)/1e9
        if elapsed >= target or world.failures["Drone1"]:
            return raw
        # A1ms request can be smaller than one AirSim physics tick. Retry a
        # bounded number of actual ticks; never pretend physical time advanced.
        raw,hit,sampled = advance(world,max(.003,target-elapsed))
        world.observe("Drone1",raw,hit,sampled)
    if (raw.timestamp_ns-origin)/1e9 >= target or world.failures["Drone1"]:
        return raw
    raise RuntimeError("bounded physical-time target not reached")


def trial(world, pool, model, scenario, ages, directory, *, held_control_age_s=0.):
    if not 0. <= held_control_age_s < .040:
        raise ValueError("recorded held control must still have positive update slack")
    directory.mkdir(parents=True,exist_ok=False)
    raw = world.initialize(scenario,directory)["Drone1"]
    _,hit,sampled = world.snapshot("Drone1")
    world.observe("Drone1",raw,hit,sampled)
    start_ns = world.initial["Drone1"]["sim_ns"]
    commands,began = [],time.perf_counter()

    def elapsed():
        return (raw.timestamp_ns-start_ns)/1e9

    def move_to(target):
        nonlocal raw
        raw = advance_to(world,raw,target)

    try:
        previous_actuation = -held_control_age_s
        for k in range(1000):
            if time.perf_counter()-began > 60.:
                raise RuntimeError("recurring-age diagnostic exceeded60s")
            target = max(0.,previous_actuation+CADENCE)
            if target >= 3.:
                move_to(3.)
                break
            age = min(ages[k%len(ages)],target)  # no input preceding the reconstructed state
            move_to(target-age)
            if world.failures["Drone1"]:
                break
            input_s = elapsed()
            feature = encode_direct(world.windows["Drone1"],raw,world.phases["Drone1"]+input_s)
            action = np.clip(pool.predict(model,feature),[2.,-3.,-3.,-2.],[20.,3.,3.,2.])
            move_to(max(target,input_s+age))
            if world.failures["Drone1"]:
                break
            applied = elapsed()
            send_control(world.drones["Drone1"],action,world.calibration,duration=1.)
            world.client.ping()
            commands.append({"input_simulation_s":input_s,"actuation_simulation_s":applied,
                "input_age_s":applied-input_s,"requested_input_age_s":age,
                "update_gap_s":applied-previous_actuation,"control":action.tolist()})
            previous_actuation = applied
        physical = world.physical_report(3.,realtime=False)["Drone1"]
        physical["clock_valid"] = None
        gap = max([c["update_gap_s"] for c in commands]+[elapsed()-previous_actuation])
        report = {"scenario":scenario,"model":model,"age_pattern_s":list(ages),"requested_cadence_s":CADENCE,
            "initial_held_control_age_s":held_control_age_s,
            "max_actual_update_gap_s":gap,"max_actual_input_age_s":max((c["input_age_s"] for c in commands),default=None),
            "sampled_update_bound_passed":gap<=.040,"physical":physical,
            "control_recovery_passed":physical["recovery_passed"],"error":None,
            "simulation_stepped":True,"end_to_end_realtime_claimed":False,"test_read":False}
    except Exception as exc:
        report = {"error":f"{type(exc).__name__}: {exc}","control_recovery_passed":False,
            "sampled_update_bound_passed":False,"simulation_stepped":True,"end_to_end_realtime_claimed":False}
    atomic_json(directory/"report.json",report)
    atomic_json(directory/"commands.json",commands)
    atomic_json(directory/"traces.json",world.traces)
    return report


def run():
    candidates = read_json(ROOT/"controller_architecture_study/results/mode_first_response_age_v1/summary.json")
    if not candidates["complete"] or not all(r["control_recovery_passed"] for r in candidates["rows"]):
        raise ValueError("matched first-response candidates required")
    pool = ControllerPool()
    setup = read_json(SOURCE/"setup.json")
    if pool.provenance()!=setup["provenance"]:
        raise ValueError("controller identity differs")
    cases = read_json(REFINED/"plan.json")["cases"]
    _,layout = moderate_case()
    OUTPUT.mkdir(parents=True,exist_ok=False)
    atomic_json(OUTPUT/"plan.json",{"cadence_s":CADENCE,"update_cap_s":.040,"profiles":PROFILES,
        "first_response_summary_sha256":sha256(ROOT/"controller_architecture_study/results/mode_first_response_age_v1/summary.json"),
        "provenance":pool.provenance(),"development_only":True,"runtime_mode_map_changed":False,
        "source_sha256":{n:sha256(Path(__file__).with_name(n)) for n in
            ("probe_recurring_input_age.py","probe_controller_step_recoverability.py","framework_runtime.py","framework_runtime_world.py")},
        "simulation_stepped":True,"end_to_end_realtime_claimed":False,"test_read":False})
    rows = []
    for ci,candidate in enumerate(candidates["rows"]):
        case = next(c for c in cases if c["id"]==candidate["source_case"])
        phase = setup["phase_s"]-setup["physical_phase_span_s"]*case["critical_index"]/(setup["vehicles"]-1)
        world = FrameworkWorld(layout,phase,1,2.)
        initials = []
        try:
            profiles = list(PROFILES)
            for profile in profiles[ci:]+profiles[:ci]:
                directory = OUTPUT/f"{case['id']}_v{candidate['velocity_magnitude']:g}_{profile}"
                r = trial(world,pool,candidate["model"],candidate["scenario"],PROFILES[profile],directory)
                initials.append(read_json(directory/"initial_state.json")["Drone1"])
                rows.append({"source_case":case["id"],"velocity_magnitude":candidate["velocity_magnitude"],
                    "profile":profile,"directory":str(directory),"report":r})
                atomic_json(OUTPUT/"summary.json",{"rows":rows,"complete":len(rows)==12,
                    "simulation_stepped":True,"end_to_end_realtime_claimed":False})
                print(f"RECURRING {len(rows)}/12 {directory.name} recovery={r['control_recovery_passed']} update_bound={r['sampled_update_bound_passed']} error={r['error']}",flush=True)
                if r["error"]:
                    raise RuntimeError(r["error"])
            if any(a[k]!=initials[0][k] for a in initials[1:] for k in ("position","velocity","quaternion","held_control")):
                raise RuntimeError("recurring-input comparison initial states differ")
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
