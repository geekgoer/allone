"""Finite matched-delay recovery profiling with common physical delivery slots.

An EMPIRICAL two-dimensional map, not a neural verification result. Continuous
physics and a shared30ms compute-to-delivery target separate timing validity
from controller recovery capability.
"""

import argparse
from collections import Counter
from pathlib import Path
import time

import numpy as np

from control_adapter import send_control
from .decision_dataset import ROOT, atomic_json, sha256
from .dynamic_probe_common import read_json
from .fast_convex_preprocess import encode, FastReferenceWindow
from .fly_decision_dnn import StoredRoutePlanner
from .framework_runtime import CONTRACT, COLLISION_CONTRACT, CONTROL_LIMIT, ControllerPool, MODELS, state_bin
from .framework_runtime_world import FrameworkWorld
from .probe_airsim_three import load_layout
from .state_mode_calibration import moderate_case


OUTPUT = ROOT/"controller_architecture_study/results/framework_runtime_calibration_v2"


def geometries():
    turn,layout = moderate_case()
    straight,digest = load_layout(48,0)
    planner = StoredRoutePlanner(straight)
    window = FastReferenceWindow(planner.trajectory,planner.config)
    times = np.linspace(1.5,planner.trajectory.duration-3.3,150)
    target = np.asarray(straight["gates"][0]["center"])-[2.,0.,0.]
    phase = min(times,key=lambda t:np.linalg.norm(window.window(float(t),horizon=1)[0].position-target))
    return [("turn",layout,turn["reference_phase_s"],turn["route_sha256"]),
            ("straight",straight,float(phase),digest)]


def rollout(world, pool, model, case, delay, directory, initial_scenario=None, *, enforce_tube=True):
    directory.mkdir(parents=True,exist_ok=False)
    raw = world.initialize({"profile_risk":case} if initial_scenario is None else initial_scenario,directory)["Drone1"]
    raw,collision,sampled = world.snapshot("Drone1")
    world.observe("Drone1",raw,collision,sampled)
    initial_risk = world.traces["Drone1"][0]["risk"]
    commands,error,next_update = [],None,delay
    tolerance,lead = .004,.030
    initial_feature = encode(raw,world.windows["Drone1"].window(world.phases["Drone1"]))
    for _ in range(3):
        pool.predict(model,initial_feature)
    if delay == 0.:
        action = np.clip(pool.predict(model,initial_feature),[2.,-3.,-3.,-2.],[20.,3.,3.,2.])
        send_control(world.drones["Drone1"],action,world.calibration,duration=1.)
        world.client.ping()
        commands.append({"simulation_s":0.,"model":model,"control":action.tolist(),
            "target_s":0.,"send_s":0.,"target_missed":False,"precomputed_during_initial_pause":True})
        next_update = CONTROL_LIMIT
    world.client.simPause(False)
    origin = time.perf_counter()
    world.traces["Drone1"][0]["sample_s"] = origin
    next_observation,pending = 0.,None
    try:
        while True:
            now = time.perf_counter()-origin
            if now > 5.:
                raise RuntimeError("continuous calibration did not cover the fixed simulation window")
            if now >= next_observation or (pending is None and now >= next_update-lead):
                raw,collision,sampled = world.snapshot("Drone1")
                world.observe("Drone1",raw,collision,sampled)
                next_observation = time.perf_counter()-origin+.020
            row = world.traces["Drone1"][-1]
            if world.failures["Drone1"] or (enforce_tube and row["risk"]["h"] < 0.) or row["t"] >= 3.:
                break
            if pending is None and time.perf_counter()-origin >= next_update-lead:
                refs = world.windows["Drone1"].window(world.phases["Drone1"]+row["t"])
                action = pool.predict(model,encode(raw,refs))
                action = np.clip(action,[2.,-3.,-3.,-2.],[20.,3.,3.,2.])
                pending = action,next_update
            if pending is not None and time.perf_counter()-origin >= pending[1]:
                action,target = pending
                sent = time.perf_counter()-origin
                if sent >= target+CONTROL_LIMIT:
                    raise RuntimeError("common calibration delivery deadline expired; stale output not sent")
                send_control(world.drones["Drone1"],action,world.calibration,duration=1.)
                world.client.ping()
                commands.append({"simulation_s":row["t"],"model":model,"control":action.tolist(),
                    "target_s":target,"send_s":sent,"rpc_return_s":time.perf_counter()-origin,
                    "target_missed":bool(sent > target+tolerance)})
                next_update = target+CONTROL_LIMIT
                pending = None
            time.sleep(.0005)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    world.client.simPause(True)
    report = world.physical_report(3.,enforce_tube=enforce_tube,realtime=True)["Drone1"]
    comparable = bool(report["clock_valid"] and all(not c["target_missed"] for c in commands) and error is None)
    report.update(model=model,requested_risk=list(case),initial_risk=initial_risk,state_bin=state_bin(initial_risk),
        initial_delay_target_s=delay,actual_first_update_s=commands[0]["simulation_s"] if commands else None,
        max_update_gap_s=max(np.diff([r["simulation_s"] for r in commands]),default=None),
        recovery_passed=bool(report["recovery_passed"] and error is None),timing_comparable=comparable,error=error,
        simulation_stepped=False,common_delivery_lead_s=lead,delivery_tolerance_s=tolerance,
        real_time_performance_claimed=False,test_read=False,
        safety_predicate="route_tube" if enforce_tube else "gate_task")
    atomic_json(directory/"report.json",report)
    atomic_json(directory/"traces.json",world.traces["Drone1"])
    atomic_json(directory/"commands.json",commands)
    if error and "common calibration delivery deadline" not in error:
        raise RuntimeError(error)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=OUTPUT)
    parser.add_argument("--net-width",type=float,default=2.)
    parser.add_argument("--safety-predicate",choices=("route_tube","gate_task"),default="route_tube")
    args = parser.parse_args()
    root = args.output_dir
    if args.safety_predicate == "gate_task" and root.resolve() == OUTPUT.resolve():
        raise ValueError("gate-task calibration requires a new output directory; retain the old tube map")
    enforce_tube = args.safety_predicate == "route_tube"
    root.mkdir(parents=True,exist_ok=True)
    pool = ControllerPool()
    geometry = geometries()
    cases = [(h,v) for h in (.15,.375) for v in (-1.5,-.5,0.,.5,1.5)]
    plan = {"contract":CONTRACT,"provenance":pool.provenance(),"net_width_m":args.net_width,
        "cases":cases,"geometry":{g:{"phase_s":p,"route_sha256":s} for g,_,p,s in geometry},
        "delays_s":[0.,CONTROL_LIMIT],"period_s":CONTROL_LIMIT,"duration_s":3.,
        "safety_predicate":args.safety_predicate,"collision_observation":COLLISION_CONTRACT,
        "criterion":("no collision/tube exit; " if enforce_tube else "no collision; ")+
            "expected gate crossings with unchanged envelope/clearance; >=80% progress; last250ms ep<=.3m and ev<=.6m/s",
        "source_sha256":{name:sha256(Path(__file__).with_name(name)) for name in
            ("framework_runtime.py","framework_runtime_world.py","framework_runtime_calibrate.py")},
        "domain":"environment48 straight/turn at nominal5m/s; nominal feedforward initial attitude and held thrust/zero rates",
        "coverage":"finite representative states and two delay endpoints; NOT all states, delays or held controls in each bin",
        "is_safety_certificate":False,"simulation_stepped":False,"common_delivery_lead_s":.030,
        "delivery_tolerance_s":.004,"test_read":False}
    # JSON round trip makes tuple/list identity stable on resume.
    import json
    plan = json.loads(json.dumps(plan))
    if (root/"plan.json").exists() and read_json(root/"plan.json") != plan:
        raise ValueError("calibration identity changed; preserve it and use another output directory")
    atomic_json(root/"plan.json",plan)
    reports = []
    total = len(geometry)*len(cases)*len(MODELS)*2
    world = None
    try:
        for g,layout,phase,digest in geometry:
            world = FrameworkWorld(layout,phase,1,args.net_width)
            for ci,case in enumerate(cases):
                # Balanced rotation avoids one model always being first.
                order = MODELS[ci%len(MODELS):]+MODELS[:ci%len(MODELS)]
                for delay in (0.,CONTROL_LIMIT):
                    for model in order:
                        directory = root/f"{g}_c{ci:02d}_delay{int(delay*1000):03d}_{model}"
                        if (directory/"report.json").exists():
                            report = read_json(directory/"report.json")
                        else:
                            if directory.exists():
                                raise ValueError(f"partial calibration retained: {directory}")
                            report = rollout(world,pool,model,case,delay,directory,enforce_tube=enforce_tube)
                        reports.append(report)
                        print(f"CALIBRATION {len(reports)}/{total} {directory.name} recovery={report['recovery_passed']}",flush=True)
                        atomic_json(root/"progress.json",{"completed":len(reports),"total":total,
                            "successes":sum(r["recovery_passed"] for r in reports)})
            world.close()
            world = None
    finally:
        if world is not None:
            atomic_json(root/"cleanup_report.json",world.close())
    bins = {}
    for report in reports:
        key = report["state_bin"]
        if key is None:
            raise RuntimeError("profile initial state did not land in a declared bin")
        bins.setdefault(key,{"modes":{}})
        bins[key]["modes"].setdefault(report["model"],[]).append(report)
    for key,entry in bins.items():
        for model,rows in list(entry["modes"].items()):
            complete = len(rows) == 4
            valid = [r for r in rows if r["timing_comparable"]]
            entry["modes"][model] = {"status":"empirical_admissible" if complete and len(valid)==len(rows) and all(r["recovery_passed"] for r in valid)
                else "observed_failure" if any(not r["recovery_passed"] for r in valid) else "unknown",
                "samples":len(rows),"timing_comparable":len(valid),"successes":sum(r["recovery_passed"] for r in rows)}
    result = {"contract":CONTRACT,"provenance":pool.provenance(),"net_width_m":args.net_width,
        "safety_predicate":args.safety_predicate,"collision_observation":COLLISION_CONTRACT,
        "complete":len(reports)==total,"bins":bins,"analysis_to_next_command_limit_s":CONTROL_LIMIT,
        "calibration_plan_sha256":sha256(root/"plan.json"),"source_plan":str(root/"plan.json"),
        "scope":plan["coverage"],"status_counts":dict(Counter(s["status"] for b in bins.values() for s in b["modes"].values())),
        "is_safety_certificate":False,"test_read":False}
    atomic_json(root/"mode_map.json",result)
    atomic_json(root/"cleanup_report.json",{"status":"CLEANUP_COMPLETE"})
    print(f"MODE_MAP_READY {root/'mode_map.json'} {result['status_counts']}",flush=True)


if __name__ == "__main__":
    main()
