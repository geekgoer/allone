"""Capture an uncovered state, then test two controllers at matched observable starts."""

from pathlib import Path

import numpy as np

from control_adapter import control_to_airsim_command, throttle_to_thrust, send_control
from .decision_dataset import ROOT, atomic_json, sha256
from .dynamic_probe_common import read_json
from .framework_cpu_affinity import CpuPlacement
from .framework_runtime import ControllerPool, ModeMap
from .framework_runtime_world import FrameworkWorld
from .probe_recurring_input_age import trial, advance_to
from .state_mode_calibration import moderate_case


BASE = ROOT/"controller_architecture_study/results"
OUTPUT = BASE/"unknown_mode_recovery_20261026_v3_post_command"
MODE_MAP = BASE/"context_mode_map_v7/mode_map.json"
SOURCE = BASE/"context_seven_development_pairs_v3/i0_pos_v1.1/adaptive"
MODELS = ("plain_mlp_h9_w512_seed0","plain_mlp_h9_w1024_seed0")
PROFILES = {"fresh":(0.,),"jitter5_30":(.005,.030)}


def capture_prefix(world, scenario, directory):
    directory.mkdir(parents=True,exist_ok=False)
    raw = world.initialize(scenario,directory)["Drone1"]
    _,hit,sampled = world.snapshot("Drone1")
    world.observe("Drone1",raw,hit,sampled)
    source = read_json(SOURCE/"traces.json")["Drone1"]
    requests = [r for r in read_json(SOURCE/"requests.json") if r["vehicle"]=="Drone1" and r.get("status")=="applied"]
    requests.sort(key=lambda r:r["rpc_return_s"])
    times = np.array([r["sample_s"] for r in source])
    simulation = np.array([r["t"] for r in source])
    applied = []
    for row in requests:
        requested = float(np.interp(row["rpc_return_s"],times,simulation))
        while (raw.timestamp_ns-world.initial["Drone1"]["sim_ns"])/1e9<requested:
            now = (raw.timestamp_ns-world.initial["Drone1"]["sim_ns"])/1e9
            raw = advance_to(world,raw,min(requested,now+.003))
            if world.failures["Drone1"]:
                break
        if world.failures["Drone1"]:
            break
        send_control(world.drones["Drone1"],row["control"],world.calibration,duration=1.)
        world.client.ping()
        applied.append({"actuation_simulation_s":(raw.timestamp_ns-world.initial["Drone1"]["sim_ns"])/1e9,
            "requested_actuation_s":requested,"control":row["control"],"source_sample_id":row["sample_id"]})
    while (raw.timestamp_ns-world.initial["Drone1"]["sim_ns"])/1e9<source[-1]["t"] and not world.failures["Drone1"]:
        now = (raw.timestamp_ns-world.initial["Drone1"]["sim_ns"])/1e9
        raw = advance_to(world,raw,min(source[-1]["t"],now+.003))
    report = {"error":None,"control_recovery_passed":None,"prefix_only":True,
        "simulation_stepped":True,"end_to_end_realtime_claimed":False,
        "max_command_timing_error_s":max(abs(c["actuation_simulation_s"]-c["requested_actuation_s"]) for c in applied),
        "source_command_times_interpolated_from_wall_and_simulation_timestamps":True,
        "exact_old_trajectory_reproduction_claimed":False}
    atomic_json(directory/"report.json",report)
    atomic_json(directory/"commands.json",applied)
    atomic_json(directory/"traces.json",world.traces)
    return report


def run(placement):
    pool = ControllerPool()
    modes = ModeMap(MODE_MAP,pool.provenance(),2.)
    reference,layout = moderate_case()
    target = next(a for a in reversed(read_json(SOURCE/"analyses.json"))
        if a["vehicle"]=="Drone1" and a.get("mode_selection",{}).get("status")=="unknown_or_no_admissible_mode")
    OUTPUT.mkdir(parents=True,exist_ok=False)
    atomic_json(OUTPUT/"plan.json",{"purpose":"test observed unknown-state recovery, not a realtime or region certificate",
        "source_report_sha256":sha256(SOURCE/"report.json"),"old_target_risk":target["risk"],
        "capture_scenario":{"critical_index":0,"position_y_m":.05,"velocity_y_mps":1.1},
        "capture_protocol":"replay old applied controls through last observed source time;3ms capture ticks; measure timing error",
        "models":list(MODELS),"profiles":PROFILES,
        "provenance":pool.provenance(),"mode_map_sha256":modes.sha256,"runtime_mode_map_changed":False,
        "target_selection":"nearest unknown trace state to old target risk: scales h=.025, hdot=.1, pressure=.25; no recovery-outcome selection",
        "source_sha256":{n:sha256(Path(__file__).with_name(n)) for n in
            ("probe_unknown_mode_recovery.py","probe_recurring_input_age.py","framework_runtime_world.py","framework_runtime.py")},
        "simulation_stepped":True,"end_to_end_realtime_claimed":False,"test_read":False})
    world = FrameworkWorld(layout,reference["reference_phase_s"],1,2.)
    try:
        capture = capture_prefix(world,{"critical_index":0,"position_y_m":.05,"velocity_y_mps":1.1},OUTPUT/"capture")
        if capture["error"]:
            raise RuntimeError(capture["error"])
        trace = read_json(OUTPUT/"capture/traces.json")["Drone1"]
        unknown = [r for r in trace if 0.<r["t"]<.5 and modes.select(r["risk"],pool.profile)[0] is None]
        if not unknown:
            atomic_json(OUTPUT/"summary.json",{"capture":capture,"complete":True,"target_found":False,
                "note":"fixed-controller capture did not visit an unknown state in the predeclared first0.5s; no target invented"})
            return
        target = min(unknown,key=lambda r:sum(((r["risk"][k]-target["risk"][k])/s)**2
            for k,s in (("h",.025),("hdot",.1),("turn_pressure",.25))))
        commands = read_json(OUTPUT/"capture/commands.json")
        # Command application does not instantaneously change kinematics.
        # Explicitly test the POST-command state/control pair at this same
        # physics tick, separately from the v2 pre-command reconstruction.
        earlier = [c for c in commands if c["actuation_simulation_s"]<=target["t"]+1e-9]
        if earlier:
            previous = earlier[-1]
            age = target["t"]-previous["actuation_simulation_s"]
            mapped = control_to_airsim_command(previous["control"],world.calibration,duration=1.)
            held = [throttle_to_thrust(mapped.throttle,world.calibration),mapped.roll_rate,-mapped.pitch_rate,-mapped.yaw_rate]
        else:
            age,held = target["t"],world.initial["Drone1"]["held_control"]
        if age>=.040:
            raise RuntimeError("captured held-control age exceeds the declared recovery protocol")
        scenario = {"relative_initial":{
            "position_error":(np.array(target["position"])-target["reference_position"]).tolist(),
            "velocity_error":(np.array(target["velocity"])-target["reference_velocity"]).tolist(),
            **{k:target[k] for k in ("quaternion","angular_velocity","acceleration")},"held_control":held}}
        phase = reference["reference_phase_s"]+target["t"]
        atomic_json(OUTPUT/"target.json",{"state":target,"scenario":scenario,"phase_s":phase,"held_control_age_s":age,
            "mode_lookup":modes.select(target["risk"],pool.profile)[1],
            "control_pair_semantics":"post-command at the same simulation tick; kinematics unchanged instantaneously",
            "match_scope":"kinematic position, velocity, quaternion, angular velocity and acceleration; hidden actuator/controller internal state is reset"})
    finally:
        atomic_json(OUTPUT/"cleanup_capture.json",world.close())

    reports = []
    world = FrameworkWorld(layout,phase,1,2.)
    try:
        for i,model in enumerate(MODELS):
            profiles = tuple(PROFILES) if i==0 else tuple(reversed(PROFILES))
            for profile in profiles:
                directory = OUTPUT/f"{model}_{profile}"
                r = trial(world,pool,model,scenario,PROFILES[profile],directory,held_control_age_s=age)
                first = read_json(directory/"traces.json")["Drone1"][0]
                deltas = {k:float(np.max(np.abs(np.array(first[k])-target[k]))) for k in
                    ("quaternion","angular_velocity","acceleration")}
                deltas.update(position_error=float(np.max(np.abs(np.array(first["position"])-first["reference_position"]-scenario["relative_initial"]["position_error"]))),
                    velocity_error=float(np.max(np.abs(np.array(first["velocity"])-first["reference_velocity"]-scenario["relative_initial"]["velocity_error"]))))
                matched = all(v<1e-4 for v in deltas.values())
                reports.append({"model":model,"profile":profile,"report":r,"observable_initial_match":matched,"initial_max_abs_delta":deltas})
                atomic_json(OUTPUT/"summary.json",{"capture":capture,"rows":reports,"complete":len(reports)==4,"target_found":True,
                    "runtime_mode_map_changed":False,"simulation_stepped":True,"end_to_end_realtime_claimed":False,
                    "region_certificate":False,"hidden_internal_state_matched":False})
                print(f"UNKNOWN_RECOVERY {len(reports)}/4 {model} {profile} recovery={r['control_recovery_passed']} bound={r['sampled_update_bound_passed']} matched={matched} error={r['error']}",flush=True)
                if r["error"] or not matched:
                    raise RuntimeError(r["error"] or "observable initial state mismatch; do not label mode admissible")
    finally:
        atomic_json(OUTPUT/"cleanup_recovery.json",world.close())


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
