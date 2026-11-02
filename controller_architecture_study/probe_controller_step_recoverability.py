"""Paused-physics recovery diagnostic; never an end-to-end timing experiment."""

import time

import numpy as np

from control_adapter import send_control
from .decision_dataset import ROOT, atomic_json, sha256
from .dynamic_probe_common import read_json
from .framework_cpu_affinity import CpuPlacement
from .framework_runtime import ControllerPool
from .framework_runtime_features import encode_direct
from .framework_runtime_world import FrameworkWorld
from .probe_controller_recoverability import SOURCE
from .state_mode_calibration import moderate_case


OUTPUT = ROOT/"controller_architecture_study/results/controller_step_recoverability_v1"


def advance(world, seconds):
    world.client.simContinueForTime(seconds)
    limit = time.perf_counter()+2.
    while not world.client.simIsPause():
        if time.perf_counter() > limit:
            raise RuntimeError("simulation did not pause after the requested step")
        time.sleep(.001)
    return world.snapshot("Drone1")


def trial(world, pool, model, scenario, period, directory, *, initial_wait=0., initial_input_age=0.):
    if not 0. <= initial_input_age <= initial_wait:
        raise ValueError("first input age must lie within the initial held-control interval")
    directory.mkdir(parents=True,exist_ok=False)
    raw = world.initialize(scenario,directory)["Drone1"]
    _,collision,sampled = world.snapshot("Drone1")
    world.observe("Drone1",raw,collision,sampled)
    commands = []
    began,error = time.perf_counter(),None
    try:
        first_action,first_input_s = None,None
        if initial_input_age:
            before_sample = initial_wait-initial_input_age
            if before_sample:
                raw,collision,sampled = advance(world,before_sample)
                world.observe("Drone1",raw,collision,sampled)
            first_input_s = (raw.timestamp_ns-world.initial["Drone1"]["sim_ns"])/1e9
            feature = encode_direct(world.windows["Drone1"],raw,world.phases["Drone1"]+first_input_s)
            first_action = np.clip(pool.predict(model,feature),[2.,-3.,-3.,-2.],[20.,3.,3.,2.])
            raw,collision,sampled = advance(world,initial_input_age)
            world.observe("Drone1",raw,collision,sampled)
        elif initial_wait:
            raw,collision,sampled = advance(world,initial_wait)
            world.observe("Drone1",raw,collision,sampled)
        for _ in range(1000):
            elapsed = (raw.timestamp_ns-world.initial["Drone1"]["sim_ns"])/1e9
            if elapsed >= 3. or world.failures["Drone1"]:
                break
            if time.perf_counter()-began > 60.:
                raise RuntimeError("bounded stepped diagnostic exceeded60s")
            input_s = elapsed if first_action is None else first_input_s
            if first_action is None:
                feature = encode_direct(world.windows["Drone1"],raw,world.phases["Drone1"]+elapsed)
                action = np.clip(pool.predict(model,feature),[2.,-3.,-3.,-2.],[20.,3.,3.,2.])
            else:
                action,first_action = first_action,None
            send_control(world.drones["Drone1"],action,world.calibration,duration=1.)
            world.client.ping()
            raw,collision,sampled = advance(world,min(period,3.-elapsed))
            following = (raw.timestamp_ns-world.initial["Drone1"]["sim_ns"])/1e9
            if following <= elapsed:
                raise RuntimeError("simulation step did not advance physical time")
            commands.append({"simulation_s":elapsed,"next_simulation_s":following,
                "input_simulation_s":input_s,"input_age_s":elapsed-input_s,
                "actual_hold_s":following-elapsed,"control":action.tolist()})
            world.observe("Drone1",raw,collision,sampled)
        physical = world.physical_report(3.,realtime=False)["Drone1"]
        physical["clock_valid"] = None
        report = {"scenario":scenario,"model":model,"requested_hold_s":period,
            "requested_initial_wait_s":initial_wait,
            "requested_initial_input_age_s":initial_input_age,
            "first_actual_input_age_s":commands[0]["input_age_s"] if commands else None,
            "actual_max_hold_s":max((c["actual_hold_s"] for c in commands),default=None),
            "physical":physical,"control_recovery_passed":bool(physical["recovery_passed"] and error is None),
            "simulation_stepped":True,"timing_passed":None,"end_to_end_realtime_claimed":False,
            "not_multi_loop_safety_comparison":True,"error":error}
    except Exception as exc:
        report = {"scenario":scenario,"simulation_stepped":True,"error":f"{type(exc).__name__}: {exc}",
            "control_recovery_passed":False,"timing_passed":None,"end_to_end_realtime_claimed":False}
    atomic_json(directory/"report.json",report)
    atomic_json(directory/"commands.json",commands)
    atomic_json(directory/"traces.json",world.traces)
    return report


def run(placement):
    pool = ControllerPool()
    source = read_json(SOURCE/"setup.json")
    if pool.provenance() != source["provenance"]:
        raise ValueError("controller identity differs")
    model = read_json(SOURCE/"selection.json")["candidate"]["configuration"]["models"]["Drone1"]
    # Restrict to the same states whose fast real-time trials missed a deadline.
    cases = [dict(id=f"i0_sign-1_v{v:g}",critical_index=0,position_y_m=-.05,velocity_y_mps=-v) for v in (1.,1.45)]
    OUTPUT.mkdir(parents=True,exist_ok=False)
    atomic_json(OUTPUT/"setup.json",{"source":str(SOURCE),"provenance":pool.provenance(),
        "runner_sha256":sha256(__file__),"simulation_stepped":True,"cases":cases,
        "periods_s":[.020,.050],"actual_step_lengths_recorded":True,
        "purpose":"controller recoverability ONLY; no timing/FPGA/multi-loop claims","test_read":False})
    case,layout = moderate_case()
    world = FrameworkWorld(layout,case["reference_phase_s"],1,2.,phase_span=2.1)
    reports = []
    try:
        for i,scenario in enumerate(cases):
            pair = {}
            for period in ((.020,.050) if i%2==0 else (.050,.020)):
                key = f"T{1000*period:g}"
                r = trial(world,pool,model,scenario,period,OUTPUT/scenario["id"]/key)
                pair[key] = r
                print(f"STEPPED {scenario['id']} {key} recovery={r['control_recovery_passed']} error={r['error']}",flush=True)
                if r["error"]:
                    raise RuntimeError(r["error"])
            initial = [read_json(OUTPUT/scenario["id"]/k/"initial_state.json")["Drone1"] for k in ("T20","T50")]
            reports.append({"scenario":scenario,"reports":pair,"initial_state_match":all(
                initial[0][k]==initial[1][k] for k in ("position","velocity","quaternion","held_control"))})
            atomic_json(OUTPUT/"summary.json",{"cases":reports,"complete":len(reports)==len(cases),
                "simulation_stepped":True,"end_to_end_realtime_claimed":False})
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
