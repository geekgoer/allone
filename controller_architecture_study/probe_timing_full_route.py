"""Real-clock, fixed-controller timing experiment; never abort on a missing map entry."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import math
from pathlib import Path
import shutil
import time

import numpy as np

from control_adapter import send_control
from .decision_dataset import ROOT, atomic_json, sha256
from .dynamic_probe_common import read_json
from .fly_decision_dnn import StoredRoutePlanner
from .framework_cpu_affinity import CpuPlacement
from .framework_runtime import ControllerPool
from .framework_runtime_editor import read_background_throttle
from .framework_runtime_features import encode_direct
from .framework_runtime_wait import DeadlineWaiter
from .framework_runtime_world import FrameworkWorld
from .state_mode_calibration import moderate_case


MODEL = "plain_mlp_h9_w1024_seed0"
PRIOR = ROOT/"controller_architecture_study/results/initial_recovery_window_v1"
SOURCES = ("probe_timing_full_route.py", "framework_runtime.py", "framework_runtime_world.py",
    "framework_runtime_features.py", "framework_runtime_wait.py", "framework_cpu_affinity.py",
    "safety_comparison_flight.py", "convex_teacher.py", "fast_convex_preprocess.py")


def following_slot(index, finished, first_release, period):
    """Discard obsolete releases; do not burst stale catch-up jobs."""
    following = max(index+1, math.ceil((finished-first_release)/period))
    return following, following-index-1


def stats_ms(values):
    a = np.asarray(values, dtype=float)*1000
    return {"count":len(a), "mean":float(a.mean()), "p99":float(np.percentile(a,99)),
        "max":float(a.max())} if len(a) else None


def trial(world, pool, scenario, arm, directory, duration, teacher=None):
    directory.mkdir(parents=True, exist_ok=False)
    raw = world.initialize(scenario,directory)["Drone1"]
    _,collision,sampled = world.snapshot("Drone1")
    world.observe("Drone1",raw,collision,sampled)
    # Warm model/solver outside the live physics interval, on this exact input.
    for _ in range(3):
        if teacher is None:
            pool.predict(MODEL,encode_direct(world.windows["Drone1"],raw,world.phases["Drone1"]))
        else:
            teacher.query(raw,world.windows["Drone1"].window(world.phases["Drone1"]))
    commands, error = [], None
    period = arm["period_ms"]/1000
    response_floor = arm["response_floor_ms"]/1000
    origin = time.perf_counter()
    first_release = origin+period*arm["release_phase_fraction"]
    previous_command, slot = origin, 0
    try:
        world.client.simPause(False)
        with DeadlineWaiter() as timer:
            while True:
                release = first_release+slot*period
                timer.wait([],release)
                raw,collision,sampled = world.snapshot("Drone1")
                row = world.observe("Drone1",raw,collision,sampled)
                elapsed = row["t"]
                if elapsed >= duration:
                    break
                if sampled-origin > 2*duration+3:
                    raise RuntimeError("simulation failed to finish within bounded wall time")
                compute_start = time.perf_counter()
                if teacher is None:
                    feature = encode_direct(world.windows["Drone1"],raw,world.phases["Drone1"]+elapsed)
                    action = pool.predict(MODEL,feature)
                else:
                    action,_ = teacher.query(raw,world.windows["Drone1"].window(world.phases["Drone1"]+elapsed))
                compute_done = time.perf_counter()
                action = np.clip(action,[2.,-3.,-3.,-2.],[20.,3.,3.,2.])
                if not np.isfinite(action).all():
                    raise RuntimeError("nonfinite control")
                # An explicitly injected transport/queue surrogate, NOT measured FPGA contention.
                timer.wait([],sampled+response_floor)
                command_start = time.perf_counter()
                send_control(world.drones["Drone1"],action,world.calibration,duration=1.)
                world.client.ping()
                returned = time.perf_counter()
                next_slot,skipped = following_slot(slot,returned,first_release,period)
                commands.append({"slot":slot, "release_s":release-origin, "sample_s":sampled-origin,
                    "sample_simulation_s":elapsed, "command_return_s":returned-origin,
                    "response_s":returned-sampled, "release_to_return_s":returned-release,
                    "release_jitter_s":sampled-release, "update_gap_return_s":returned-previous_command,
                    "preprocess_and_inference_s":compute_done-compute_start,
                    "command_rpc_s":returned-command_start, "skipped_releases":skipped,
                    "nominal_deadline_missed":returned > release+period,
                    "control":action.tolist()})
                previous_command,slot = returned,next_slot
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        world.hover_all()
    physical = world.physical_report(duration)["Drone1"]
    initial = read_json(directory/"initial_state.json")["Drone1"]
    report = {"arm":arm, "scenario":scenario, "duration_s":duration,
        "model":MODEL if teacher is None else "original_convex_MPC_teacher",
        "physical":physical, "error":error, "commands":len(commands),
        "initial_state_signature":{k:initial[k] for k in ("position","velocity","quaternion","held_control")},
        "full_flight_completed":bool(physical["window_covered"] and error is None),
        "success":bool(physical["recovery_passed"] and physical["clock_valid"] and error is None),
        "collision_observed":any(r["collision"] for r in world.traces["Drone1"]),
        "nominal_deadline_misses":sum(r["nominal_deadline_missed"] for r in commands),
        "skipped_releases":sum(r["skipped_releases"] for r in commands),
        "response_ms":stats_ms([r["response_s"] for r in commands]),
        "update_gap_return_ms":stats_ms([r["update_gap_return_s"] for r in commands]),
        "release_jitter_ms":stats_ms([r["release_jitter_s"] for r in commands]),
        "lookup_used":False, "deadline_expiry_aborts":False, "collision_causes_early_abort":False,
        "actuation_timestamp":"command RPC return proxy; exact physical application time not observed",
        "real_time_unpaused":True, "multi_UAV_comparison":False, "FPGA_verified":False,
        "test_read":False}
    atomic_json(directory/"commands.json",commands)
    atomic_json(directory/"traces.json",world.traces)
    atomic_json(directory/"report.json",report)
    return report


def timing_win(fast, slow):
    return bool(fast["success"] and slow["collision_observed"] and slow["full_flight_completed"]
        and slow["physical"]["clock_valid"] and fast["initial_state_signature"]==slow["initial_state_signature"])


def screening_candidate(fast, slow):
    # Screening can nominate a pair before terminal settling passes; confirmation cannot.
    candidate = {**fast,"success":bool(fast["physical"]["physical_task_passed"]
        and fast["physical"]["clock_valid"] and fast["full_flight_completed"])}
    return timing_win(candidate,slow)


def run(root, stage, source_root=None):
    root.mkdir(parents=True,exist_ok=False)
    frozen = root/"frozen_source"
    frozen.mkdir()
    for name in SOURCES:
        shutil.copy2(Path(__file__).with_name(name),frozen/name)
    editor = read_background_throttle()
    if editor["background_throttle_enabled"]:
        raise RuntimeError("disable editor background throttling before timing trials")
    pool = ControllerPool()
    case,layout = moderate_case()
    end = StoredRoutePlanner(layout).trajectory.duration
    cases = []
    for name,phase,sign in (("i0_positive_v1.1",case["reference_phase_s"],1),
                           ("i6_negative_v1.1",case["reference_phase_s"]-2.1,-1)):
        old = f"i{0 if sign==1 else 6}_sign{sign:+d}_v1.4"
        scenario = deepcopy(read_json(PRIOR/f"{old}_DNN_delay0/report.json")["scenario"])
        scenario["relative_initial"]["velocity_error"] = [0.,sign*1.1,0.]
        cases.append({"id":name,"phase_s":phase,"duration_s":end-phase+1.,"scenario":scenario})
    jobs = []
    if stage == "screen":
        for i,c in enumerate(cases):
            arms = [{"period_ms":p,"response_floor_ms":r,"release_phase_fraction":6/7}
                for r in (0.,20.) for p in (20.,40.,60.,80.)]
            if i%2:
                arms.reverse()
            jobs.extend((c,a,"DNN",0) for a in arms)
    else:
        source = read_json(source_root/"summary.json")
        if not source["complete"] or source["provenance"] != pool.provenance():
            raise ValueError("confirmation requires complete matching screen")
        for c in cases:
            c["duration_s"] += 2.  # Same extra terminal observation for every confirmation arm.
            rows = [r for r in source["rows"] if r["case"]==c["id"]]
            fast = next(r["report"] for r in rows if r["report"]["arm"]["period_ms"]==20.
                and r["report"]["arm"]["response_floor_ms"]==0.)
            slow = next(r["report"] for r in rows if r["report"]["arm"]["period_ms"]==80.
                and r["report"]["arm"]["response_floor_ms"]==0.)
            if screening_candidate(fast,slow):
                for rep in (1,2):
                    for p in ((20.,80.) if rep==1 else (80.,20.)):
                        jobs.append((c,{"period_ms":p,"response_floor_ms":0.,"release_phase_fraction":6/7},"DNN",rep))
                for p in (20.,80.):
                    jobs.append((c,{"period_ms":p,"response_floor_ms":0.,"release_phase_fraction":0.},"DNN",3))
            # Isolate response delay at the same T when the declared screen finds that contrast.
            fast_r = next(r["report"] for r in rows if r["report"]["arm"]["period_ms"]==40.
                and r["report"]["arm"]["response_floor_ms"]==0.)
            slow_r = next(r["report"] for r in rows if r["report"]["arm"]["period_ms"]==40.
                and r["report"]["arm"]["response_floor_ms"]==20.)
            if screening_candidate(fast_r,slow_r):
                for response in (0.,20.):
                    jobs.append((c,{"period_ms":40.,"response_floor_ms":response,"release_phase_fraction":6/7},"DNN",5))
        # Always test the actual teacher at achievable 10/20ms, with measured R, not paused physics.
        for c in cases:
            for p in (10.,20.):
                jobs.append((c,{"period_ms":p,"response_floor_ms":0.,"release_phase_fraction":0.},"MPC",4))
    plan = {"stage":stage,"cases":cases,"jobs":[{"case":c["id"],"arm":a,"controller":t,"repeat":r} for c,a,t,r in jobs],
        "provenance":pool.provenance(),"editor":editor,"net_width_m":2.,"gate_height_m":4.,
        "source_hashes":{n:sha256(frozen/n) for n in SOURCES},
        "source_summary_sha256":sha256(source_root/"summary.json") if source_root else None,
        "terminal_observation_s":3. if stage=="confirm" else 1.,
        "full_remaining_route":True,"paired_state_and_model":True,"lookup_used":False,
        "late_outputs_applied_for_exploration":True,"injected_delay_labelled":True,
        "release_phase":"single-loop surrogate of last staggered client; not shared-accelerator measurement",
        "purpose":"timing-sensitive recoverability only; not full framework or FALCON comparison"}
    atomic_json(root/"plan.json",plan)
    rows,world,current,teacher = [],None,None,None
    try:
        for c,arm,controller,repeat in jobs:
            if current != c["id"]:
                if world is not None:
                    atomic_json(root/f"cleanup_{len(rows):03d}.json",world.close())
                world = FrameworkWorld(layout,c["phase_s"],1,2.,task_duration=c["duration_s"])
                current = c["id"]
            if controller=="MPC" and teacher is None:
                from .convex_teacher import ConvexTeacher
                teacher = ConvexTeacher()
                atomic_json(root/"teacher.json",teacher.signature())
            name = f"{len(rows)+1:03d}_{c['id']}_{controller}_T{arm['period_ms']:g}_R{arm['response_floor_ms']:g}_rep{repeat}"
            print(f"START {len(rows)+1}/{len(jobs)} {name}",flush=True)
            report = trial(world,pool,c["scenario"],arm,root/name,c["duration_s"],teacher if controller=="MPC" else None)
            rows.append({"case":c["id"],"directory":str(root/name),"controller":controller,"repeat":repeat,"report":report})
            atomic_json(root/"summary.json",{"complete":len(rows)==len(jobs),"provenance":plan["provenance"],"rows":rows})
            print(f"  complete={report['full_flight_completed']} success={report['success']} collision={report['collision_observed']} clock={report['physical']['clock_ratio']} R={report['response_ms']} error={report['error']}",flush=True)
            if report["error"]:
                raise RuntimeError(report["error"])
    finally:
        if world is not None:
            atomic_json(root/"cleanup_final.json",world.close())
    print(f"COMPLETE {root}",flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("screen","confirm"),default="screen")
    parser.add_argument("--source-root",type=Path)
    parser.add_argument("--output",type=Path)
    args = parser.parse_args()
    if args.stage=="confirm" and args.source_root is None:
        parser.error("confirm requires --source-root")
    root = args.output or ROOT/"controller_architecture_study/results"/("timing_full_route_"+args.stage+"_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    placement = CpuPlacement()
    try:
        with placement:
            run(root,args.stage,args.source_root)
    finally:
        if root.exists():
            atomic_json(root/"cpu_placement.json",placement.report)


if __name__=="__main__":
    main()
