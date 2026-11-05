"""Continuous-physics neural control with bounded admission and real segments."""

import gc
import time

from .decision_dataset import atomic_json
from .framework_runtime_features import encode_direct
from .framework_runtime import analyze_budget, bounded_plan, account_planning_delay, remaining_stage_cost, input_age_fits, release_deadline
from .framework_runtime_worker import RuntimeServer
from .framework_runtime_io import ProcessCommandIO
from .framework_runtime_wait import DeadlineWaiter


# Explicit predictive costs, not WCET or a Windows hard-real-time guarantee.
# Command return and subsequent state fetch are different timeline events.
# v3 records:286 commands, max12.91ms;25ms is an empirical padded prediction,
# NOT WCET. Full I/O channel occupancy remains40ms, including the fresh state.
DEFAULT_COSTS = {"input_s":.003,"sample_io_s":.020,"output_s":.0015,"handoff_s":.001,"io_s":.040,"command_s":.025,
    "ingress_s":.0015,"egress_s":.0015,"completion_guard_s":.001,"dispatch_guard_s":.020,
    "planner_reserve_s":.001,"is_WCET":False}


class FlightAbort(RuntimeError):
    def __init__(self, kind, vehicle, detail):
        self.kind,self.vehicle,self.detail = kind,vehicle,detail
        super().__init__(f"{kind}: {vehicle}: {detail}")


def analysis_fits_release_gap(now, plans, bound):
    """Feedback analysis must not occupy another loop's promised release."""
    return all(now+bound <= row["planned_release_s"] for row in plans.values())


def command_would_be_blocked(now, work_s, costs, rows):
    finish = now+work_s+costs["handoff_s"]+costs["command_s"]+costs["completion_guard_s"]
    return any(row.get("status") != "applied" and finish > row["deadline_s"] for row in rows)


def feedback_can_run(policy, now, plans, costs, rows):
    if command_would_be_blocked(now,costs["output_s"],costs,rows):
        return False
    # Adaptive releases are prospective decisions, not additional hard
    # deadlines. The global replanner may move them while retaining every D.
    return policy!="static" or analysis_fits_release_gap(now,plans,costs["output_s"])


def admission_progress(pending):
    return tuple((r["sample_id"],r.get("status"),"worker_enqueue_s" in r,
        "io_started_s" in r,"io_response_received_s" in r) for _,r in sorted(pending.items()))


def commit_plan(proposal, plans, waiting):
    if proposal is None:
        return False
    original = {**plans,**waiting}
    if (proposal.keys() != original.keys() or any(proposal[v][k] != r[k]
            for v,r in original.items() for k in ("model","deadline_s","sample_id"))):
        raise RuntimeError("admission changed a fixed context or deadline")
    plans.clear()
    plans.update(proposal)
    waiting.clear()
    return True


def fly(world, pool, configuration, scenario, policy, directory, *, mode_map=None, costs=None, duration=3., cpu_layout=None):
    if policy not in ("static","adaptive","deadline_only"):
        raise ValueError("unknown runtime policy")
    if policy == "adaptive" and mode_map is None:
        raise ValueError("adaptive control requires a provenance-checked empirical mode map")
    directory.mkdir(parents=True,exist_ok=False)
    costs = dict(DEFAULT_COSTS if costs is None else costs)
    states = world.initialize(scenario,directory)
    server = RuntimeServer("static" if policy == "static" else "edf",(cpu_layout or {}).get("inference_cpus"),
        scheduler_cpus=(cpu_layout or {}).get('frontend_cpus'))
    if server.provenance != pool.provenance():
        server.close()
        raise ValueError("runtime worker artifact identity mismatch")
    atomic_json(directory/"worker_profile.json",server.profile)
    names = world.vehicles
    inflight_limit = len(names)
    costs["io_slots"] = len(names)
    try:
        # The helper has already traversed the upstream IPC queue. Its local
        # guard charges only the remaining command RPC, not that queue again.
        io = ProcessCommandIO(len(names),world.calibration,costs.get("actuation_s",costs["command_s"]),(cpu_layout or {}).get("io_cpus"))
    except BaseException:
        server.close()
        raise
    priority = {v:configuration["priority_order"].index(v) for v in names}
    periods = configuration["periods_ms"]
    plans,pending,ready,finished_io,sampled_ready,sequence = {},{},{},{},{},{v:0 for v in names}
    waiting,last_attempt_token = {},None
    requests,decisions,analyses = [],[],[]
    tail_observations = set()
    abort,error,origin = None,None,None
    release_waiter = None
    loop_stalls,previous_tick,previous_stage = [],None,"startup"
    gc_was_enabled = gc.isenabled()
    gc_suspended = False
    gc_events = []
    def gc_event(phase, info):
        gc_events.append({"phase":phase,"generation":info["generation"],"wall_s":time.perf_counter()})
    gc.callbacks.append(gc_event)

    def ordering(v, row):
        return (priority[v] if policy == "static" else row["deadline_s"],row["planned_release_s"],v)

    def collect_results():
        for action,row in server.drain():
            v = row["vehicle"]
            if v not in pending or pending[v]["sample_id"] != row["sample_id"]:
                raise RuntimeError("executor returned an unknown or superseded context")
            if row["model"] != row["frozen_model"] or row["deadline_s"] != row["frozen_deadline_s"]:
                raise RuntimeError("in-flight configuration or deadline changed")
            row["worker_status"] = row["status"]
            row["received_s"] = time.perf_counter()
            ready[v] = action,row

    def collect_samples():
        for packet,row in io.drain_samples():
            row["status"] = "sampled"
            sampled_ready[row["vehicle"]] = packet,row

    def pending_specs(now):
        specs = []
        def remaining(r,key,start):
            value,tail = remaining_stage_cost(costs,key,max(0.,now-r.get(start,now)))
            if tail:
                tail_observations.add((r["sample_id"],key))
            return value
        for v,r in pending.items():
            row = {**r,"stage":"output" if v in ready else "compute"}
            if v in io.samples:
                row.update(stage="sampling",sample_io_remaining_s=remaining(r,"sample_io_s","sample_submitted_s"))
                specs.append(row)
                continue
            if v in sampled_ready:
                row["stage"] = "input_ready"
                specs.append(row)
                continue
            if row["stage"] == "compute":
                row["ingress_remaining_s"] = (0. if "worker_enqueue_s" in r else
                    remaining(r,"ingress_s","enqueue_s"))
            future = io.futures.get(v)
            if v in finished_io or (future is not None and future.done()):
                if future is not None:
                    future.result()  # propagate a communication exception before admitting more work
                if r["status"] != "applied":
                    raise FlightAbort(r["status"],v,"command I/O failed before a new schedule could be committed")
                row.update(stage="analysis",command_completed_s=r["rpc_return_s"])
            elif future is not None:
                row.update(stage="io",io_remaining_s=remaining(r,"io_s","io_submitted_s"),
                    command_remaining_s=remaining(r,"command_s","io_submitted_s"))
                if r.get("status") == "applied":
                    row["command_completed_s"] = r["rpc_return_s"]
            specs.append(row)
        return specs

    def replan(trigger):
        nonlocal last_attempt_token
        began = time.perf_counter()
        began_cpu = time.thread_time()
        # A result may arrive while the CPU handles another loop's observation.
        # Re-check the pipe before forecasting; otherwise completed inference
        # is incorrectly charged as an entirely new full-model computation.
        collect_results()
        collect_samples()
        planning_time = time.perf_counter()
        reserved_until = max(planning_time,began+costs["planner_reserve_s"])
        forecast_costs = {**costs,"frontend_reserved_until_s":reserved_until}
        specs = pending_specs(planning_time)
        candidates = {**plans,**waiting}
        proposal,info = bounded_plan(planning_time,candidates,specs,server.profile,forecast_costs,
            inflight_limit=inflight_limit)
        if proposal is None and info["checked"] < 5:
            # IPC and inference continue during this bounded CPU calculation.
            # Before refusing, refresh once only if observable progress changed.
            collect_results()
            collect_samples()
            planning_time = time.perf_counter()
            reserved_until = max(reserved_until,planning_time)
            forecast_costs["frontend_reserved_until_s"] = reserved_until
            refreshed = pending_specs(planning_time)
            if refreshed != specs:
                first = info
                proposal,info = bounded_plan(planning_time,candidates,refreshed,server.profile,forecast_costs,
                    maximum_candidates=5-first["checked"],inflight_limit=inflight_limit)
                info.update(checked=first["checked"]+info["checked"],candidate_limit=5,
                    events=first["events"]+info["events"],snapshot_refreshed=True,
                    previous_attempt=first)
                specs = refreshed
        end = time.perf_counter()
        info.update(trigger=trigger,started_s=began,finished_s=end,overhead_ms=1000*(end-began),
            cpu_ms=1000*(time.thread_time()-began_cpu),
            pending_jobs=len(pending),future_jobs=len(candidates),waiting_admission=len(waiting))
        decisions.append(info)
        info["planning_time_s"] = planning_time
        info["frontend_reserved_until_s"] = reserved_until
        if proposal is None:
            keys = ("sample_id","vehicle","model","priority","deadline_s","planned_release_s","stage","io_remaining_s","command_remaining_s","sample_io_remaining_s","ingress_remaining_s","command_completed_s")
            info["forecast_future"] = [{k:r[k] for k in keys if k in r} for r in candidates.values()]
            info["forecast_pending"] = [{k:r[k] for k in keys if k in r} for r in specs]
        if proposal is not None:
            proposal = account_planning_delay(proposal,info,specs,reserved_until,end,costs)
            if proposal is None:
                info["reason"] = "planning lag exhausted original completion slack; no deadline extension"
        last_attempt_token = admission_progress(pending)
        info["admission_deferred"] = not commit_plan(proposal,plans,waiting)
        # A failed prediction admits no new work. Existing commitments continue;
        # retry only on observable progress, with all original deadlines intact.
        return not info["admission_deferred"]

    def next_dynamic(v, state, observation, action, actuation):
        analysis = analyze_budget(state,action,observation["sample_s"],actuation,world.gates[v],world.net_width)
        analysis.update(vehicle=v,risk=observation["risk"])
        analyses.append(analysis)
        if not analysis["valid"]:
            raise FlightAbort("no_positive_budget",v,"fresh post-command analysis has no positive remaining horizon")
        if policy == "adaptive":
            model,selection = mode_map.select(observation["risk"],server.profile)
            analysis["mode_selection"] = selection
            if model is None:
                raise FlightAbort("no_admissible_mode",v,selection)
            analysis["deadline_s"] = min(analysis["deadline_s"],observation["sample_s"]+selection["delay_limit_s"],
                actuation+selection["future_update_limit_s"])
            analysis["valid"] = analysis["deadline_s"] > observation["sample_s"]
            if not analysis["valid"]:
                raise FlightAbort("no_positive_budget",v,"mode-specific update bound already expired")
        else:
            model = configuration["models"][v]
        waiting[v] = {"sample_id":f"{v}:{sequence[v]}","vehicle":v,"model":model,
            "priority":priority[v],"planned_release_s":time.perf_counter(),"deadline_s":analysis["deadline_s"],
            "analysis_s":analysis["analysis_s"],"analysis_risk":analysis["risk"],"stage":"input"}
        if policy == "adaptive" and selection.get("input_age_limit_s") is not None:
            waiting[v]["input_age_limit_s"] = selection["input_age_limit_s"]

    def finish_command(v, packet, row):
        row["handled_s"] = time.perf_counter()
        if packet is None or row["status"] != "applied":
            raise FlightAbort(row["status"],v,"both policies reject stale outputs or insufficient delivery budget")
        if row["rpc_return_s"] > row["deadline_s"]:
            raise FlightAbort("command_return_late",v,"actual RPC proxy exceeded the original deadline")
        if not input_age_fits(row,row["rpc_return_s"]):
            raise FlightAbort("input_age_limit_exceeded",v,"command RPC proxy exceeded its input-age limit")
        pending.pop(v)
        raw,collision,sampled = packet
        states[v] = raw
        observation = world.observe(v,raw,collision,sampled)
        row["post_command_observation_s"] = sampled
        if world.failures[v]:
            raise FlightAbort(world.failures[v]["kind"],v,world.failures[v])
        if observation["t"] < duration and time.perf_counter()-origin < duration:
            if policy == "static":
                release = row["planned_release_s"]+periods[v]/1000
                plans[v] = {"sample_id":f"{v}:{sequence[v]}","vehicle":v,"model":configuration["models"][v],
                    "priority":priority[v],"planned_release_s":release,
                    "deadline_s":release+periods[v]/1000,"stage":"input"}
            else:
                next_dynamic(v,raw,observation,row["mapped_control"],row["rpc_return_s"])
                replan(v)
        row["output_finished_s"] = time.perf_counter()

    try:
        release_waiter = DeadlineWaiter()
        # Initial old commands and state were placed under pause; all setup and
        # worker/model warmup are outside the experiment's physical time window.
        initial_observations = {}
        for v in names:
            packet = type("InitialCollision",(),{"has_collided":False,"time_stamp":world.initial[v]["collision_stamp"],"object_name":""})()
            initial_observations[v] = world.observe(v,states[v],packet,0.)
            encode_direct(world.windows[v],states[v],world.phases[v])  # warm full feature path; result is discarded
        # The placed state and held command are fixed under the initial pause.
        # Build BOTH policies' first schedule relative to physics start, not
        # after that start. Steady-state replanning remains fully charged.
        for i,v in enumerate(names):
            observation = initial_observations[v]
            if policy == "static":
                release = i*periods[v]/1000/len(names) if configuration["phase"] == "staggered" else 0.
                plans[v] = {"sample_id":f"{v}:0","vehicle":v,"model":configuration["models"][v],
                    "priority":priority[v],"planned_release_s":release,"deadline_s":release+periods[v]/1000,"stage":"input"}
            else:
                next_dynamic(v,states[v],observation,world.initial[v]["held_control"],0.)
        if policy != "static":
            began = time.perf_counter()
            proposal,info = bounded_plan(0.,waiting,[],server.profile,costs,inflight_limit=inflight_limit)
            info.update(trigger="initial_pause_bootstrap",outside_physics=True,
                overhead_ms=1000*(time.perf_counter()-began))
            decisions.append(info)
            if proposal is None:
                raise FlightAbort("admission_refused","bootstrap",info["reason"])
            commit_plan(proposal,plans,waiting)
        # Retained evidence: a41ms generation2 collection stalled all parent
        # I/O threads and expired otherwise fast jobs. Collect outside physics;
        # reference counting remains active during this bounded mission window.
        gc.collect()
        gc.disable()
        gc_suspended = True
        world.client.simPause(False)
        origin = time.perf_counter()
        for observation in initial_observations.values():
            observation["sample_s"] = origin
        for row in plans.values():
            for key in ("planned_release_s","deadline_s","analysis_s"):
                if key in row:
                    row[key] += origin
        for row in analyses:
            for key in ("analysis_s","actuation_proxy_s","deadline_s"):
                row[key] += origin
        for info in decisions:
            info["finish"] = {k:t+origin for k,t in info.get("finish",{}).items()}
            if "release_base_s" in info:
                info["release_base_s"] += origin

        while time.perf_counter()-origin < duration:
            tick = time.perf_counter()
            if previous_tick is not None and tick-previous_tick > .005:
                loop_stalls.append({"operation":previous_stage,"start_s":previous_tick,"end_s":tick})
            previous_tick,previous_stage = tick,"collect_results_and_samples"
            collect_results()
            collect_samples()
            for packet,row in io.drain():
                finished_io[row["vehicle"]] = packet,row

            # Dispatch an already computed urgent command before spending CPU
            # time deciding a later request's release. I/O itself is parallel.
            if ready:
                previous_stage = "dispatch_command"
                v = min(ready,key=lambda n:ordering(n,ready[n][1]))
                action,row = ready.pop(v)
                if not input_age_fits(row,time.perf_counter(),costs["command_s"]):
                    raise FlightAbort("input_age_budget_exhausted",v,"no fresh-command budget remains; output not applied")
                row["io_handoff_s"] = time.perf_counter()
                io.send(action,row)
                continue

            # An already captured input is committed work. Finish its feature
            # stage before feedback planning or releasing further samples.
            output_would_be_blocked = command_would_be_blocked(time.perf_counter(),costs["input_s"],costs,
                (r for v,r in pending.items() if v not in io.futures and v not in io.samples and v not in sampled_ready))
            if sampled_ready and not output_would_be_blocked:
                previous_stage = "feature_input"
                v = min(sampled_ready,key=lambda n:ordering(n,sampled_ready[n][1]))
                packet,row = sampled_ready.pop(v)
                row["frontend_start_s"] = time.perf_counter()
                row["status"] = "frontend_input"
                raw,collision,sampled = packet
                observation = world.observe(v,raw,collision,sampled)
                states[v] = raw
                if world.failures[v]:
                    raise FlightAbort(world.failures[v]["kind"],v,world.failures[v])
                if observation["t"] >= duration:
                    row["status"] = "cancelled_after_window"
                    pending.pop(v)
                    continue
                feature = encode_direct(world.windows[v],raw,world.phases[v]+observation["t"])
                row.update(sample_s=sampled,sample_sim_ns=int(raw.timestamp_ns),feature_done_s=time.perf_counter(),
                    status="submitted")
                if row["feature_done_s"] >= row["deadline_s"]:
                    row["status"] = "expired_before_submit"
                    raise FlightAbort("deadline_expired_before_submit",v,"frontend exhausted the original deadline")
                server.submit(feature,row)
                continue

            live = (r for v,r in pending.items() if v not in io.futures and v not in io.samples
                and v not in sampled_ready and v not in finished_io)
            if policy!="static" and finished_io and feedback_can_run(policy,time.perf_counter(),plans,costs,live):
                previous_stage = "feedback_before_prospective_release"
                v = min(finished_io,key=lambda n:ordering(n,finished_io[n][1]))
                packet,row = finished_io.pop(v)
                finish_command(v,packet,row)
                continue

            # A promised release is not optional CPU analysis work. Launch its
            # asynchronous capture before handling another loop's observation;
            # the forecast also starts captures at their committed releases.
            available = [v for v,r in plans.items() if r["planned_release_s"] <= time.perf_counter()]
            if available and len(pending) < inflight_limit:
                previous_stage = "dispatch_sample"
                v = min(available,key=lambda n:ordering(n,plans[n]))
                row = plans.pop(v)
                row["job_released_s"] = time.perf_counter()
                row["control_deadline_s"] = row["deadline_s"]
                row["deadline_s"] = release_deadline(row,row["job_released_s"])
                row.update(frozen_model=row["model"],frozen_deadline_s=row["deadline_s"],status="sampling")
                requests.append(row)
                pending[v] = row
                sequence[v] += 1
                io.sample(row)
                continue

            analysis_would_block_output = command_would_be_blocked(time.perf_counter(),costs["output_s"],costs,
                (r for v,r in pending.items() if v not in io.futures and v not in io.samples and v not in sampled_ready and v not in finished_io))
            if (finished_io and not analysis_would_block_output
                    and analysis_fits_release_gap(time.perf_counter(),plans,costs["output_s"])):
                previous_stage = "feedback_analysis_and_planning"
                v = min(finished_io,key=lambda n:ordering(n,finished_io[n][1]))
                packet,row = finished_io.pop(v)
                finish_command(v,packet,row)
                continue

            expired = [v for v,r in {**plans,**waiting}.items() if r["deadline_s"] <= time.perf_counter()]
            if expired:
                raise FlightAbort("deadline_expired_before_release",expired[0],"unreleased or unadmitted promise expired; no deadline extension")
            if waiting and admission_progress(pending) != last_attempt_token:
                previous_stage = "retry_after_progress"
                replan("observed_pipeline_progress")
                continue
            previous_stage = "wait_for_pipeline"
            wake_at = min([origin+duration,
                *(r["planned_release_s"] for r in plans.values()),
                *(r["deadline_s"] for r in waiting.values())])
            release_waiter.wait((server.connection.connection,io.connection.connection),wake_at)
            resumed = time.perf_counter()
            if resumed-wake_at > .005:
                loop_stalls.append({"operation":"event_wait_overrun","start_s":wake_at,"end_s":resumed})

        # Cover the complete fixed window even for a loop whose last job was
        # earlier. Jobs due before the window end must still have been delivered.
        end = origin+duration
        overdue = [r for r in [*pending.values(),*plans.values(),*waiting.values()] if r["deadline_s"] <= end and r.get("status") != "applied"]
        if overdue:
            raise FlightAbort("undelivered_committed_deadline",overdue[0]["vehicle"],"deadline within mission window was not met")
        for v in names:
            raw,collision,sampled = world.snapshot(v)
            # AirSim reports the last physics tick, not the wall-clock instant.
            # Obtain the first tick covering the declared window; do not score
            # missing final data as controller failure or extend job deadlines.
            capture_limit = time.perf_counter()+.100
            while (raw.timestamp_ns-world.initial[v]["sim_ns"])/1e9 < duration and time.perf_counter()<capture_limit:
                time.sleep(.001)
                raw,collision,sampled = world.snapshot(v)
            world.observe(v,raw,collision,sampled)
            if world.failures[v]:
                raise FlightAbort(world.failures[v]["kind"],v,world.failures[v])
    except FlightAbort as exc:
        abort = {"kind":exc.kind,"vehicle":exc.vehicle,"detail":exc.detail,
                 "wall_s":time.perf_counter()-origin if origin is not None else None}
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        gc.callbacks.remove(gc_event)
        if release_waiter is not None:
            release_waiter.close()
        try:
            io.close()
        except Exception as exc:
            error = error or f"command I/O cleanup {type(exc).__name__}: {exc}"
        try:
            server.close()
        except Exception as exc:
            error = error or f"compute cleanup {type(exc).__name__}: {exc}"
        try:
            world.hover_all()
        except Exception as exc:
            error = error or f"cleanup {type(exc).__name__}: {exc}"
        if gc_was_enabled:
            gc.enable()

    vehicles = world.physical_report(duration)
    segment_jobs = [s["sample_id"] for s in server.segments]
    # Count resumptions directly, independent of identical segment timestamps.
    seen,last,resumptions = set(),None,0
    for name in segment_jobs:
        if name != last and name in seen:
            resumptions += 1
        seen.add(name)
        last = name
    cutoff = origin+(abort["wall_s"] if abort is not None else duration) if origin is not None else float("inf")
    timing_abort = abort is not None and abort["kind"] not in ("collision","flight_guard")
    missed_before_stop = [] if origin is None else [r["sample_id"] for r in [*requests,*plans.values(),*waiting.values()]
        if r["deadline_s"] < cutoff and r.get("status") != "applied"]
    timing_ok = bool(not timing_abort and error is None and not missed_before_stop
        and all(r.get("rpc_return_s",0.) <= r["deadline_s"] for r in requests if r.get("status") == "applied"))
    actual_late = [r["sample_id"] for r in requests if r.get("rpc_return_s",0.) > r["deadline_s"]]
    expired_outputs = [r["sample_id"] for r in requests if r.get("status","").startswith("expired")]
    admission_failure = abort is not None and abort["kind"] in (
        "admission_refused","planner_budget_exhausted","no_admissible_mode","no_positive_budget","delivery_budget_exhausted")
    physical = all(r["physical_task_passed"] for r in vehicles.values())
    recovery = all(r["recovery_passed"] for r in vehicles.values())
    clock_ok = all(r["clock_valid"] for r in vehicles.values())
    scores = [r["score"]["J"] for r in vehicles.values() if r["score"]]
    per_vehicle_models = {v:[r["model"] for r in requests if r["vehicle"]==v] for v in names}
    model_switches = sum(sum(a!=b for a,b in zip(ms[:-1],ms[1:])) for ms in per_vehicle_models.values())
    report = {"policy":policy,"scenario":scenario,"configuration":configuration,"vehicles":vehicles,
        "cpu_placement":{"layout":cpu_layout,"inference_process":server.placement,"communication":io.placement},
        "actual_vehicles":len(names),"physical_task_passed":physical,"recovery_passed":recovery,
        "timing_passed":timing_ok,"clock_valid":clock_ok,"accepted":physical and recovery and timing_ok and clock_ok and abort is None,
        "total_J":float(sum(scores)) if physical and len(scores)==len(names) and abort is None and error is None else None,
        "abort":abort,"error":error,"collisions":sum(r["failure"] is not None and r["failure"]["kind"]=="collision" for r in vehicles.values()),
        "requests":len(requests),"applied":sum(r["status"]=="applied" for r in requests),
        "undelivered_deadlines_before_stop":missed_before_stop,
        "late_command_returns":actual_late,"expired_output_jobs":expired_outputs,
        "admission_refused":bool(admission_failure),
        "actual_timing_violation_observed":bool(actual_late or expired_outputs or missed_before_stop),
        "model_switches":model_switches,"segment_resumptions":resumptions,"executed_segments":len(server.segments),
        "max_simultaneous_jobs_limit":inflight_limit,"command_io_channels":len(names),"finite_plan_candidate_limit":5,
        "planner_max_ms":max((r["overhead_ms"] for r in decisions if not r.get("outside_physics")),default=0.),
        "admission_deferrals":sum(r.get("admission_deferred",False) for r in decisions),
        "unadmitted_at_stop":list(waiting.values()),
        "initial_schedule_precomputed_under_pause":origin is not None,"physical_clock_started":origin is not None,
        "input_cost_exceedances":sum(r.get("feature_done_s",0.)-r["frontend_start_s"]>costs["input_s"] for r in requests if "frontend_start_s" in r),
        "sample_io_cost_exceedances":sum(r["sample_io_finished_s"]-r["sample_io_started_s"]>costs["sample_io_s"] for r in requests if "sample_io_finished_s" in r),
        "output_cost_exceedances":sum(r["output_finished_s"]-r["handled_s"]>costs["output_s"] for r in requests if "output_finished_s" in r),
        "io_cost_exceedances":sum(r["io_finished_s"]-r["io_started_s"]>costs["io_s"] for r in requests if "io_finished_s" in r),
        "command_cost_exceedances":sum(r["rpc_return_s"]-r["io_started_s"]>costs["command_s"] for r in requests if "rpc_return_s" in r),
        "costs":costs,"model_map_sha256":mode_map.sha256 if mode_map else None,
        "model_profile_is_WCET":False,"actuation_semantics":"command send + ping return proxy; fresh subsequent state request, NOT hardware acknowledgement",
        "prediction_tail_escalations":[{"sample_id":i,"stage":k} for i,k in sorted(tail_observations)],
        "automatic_gc_suspended_during_bounded_window":gc_suspended,"automatic_gc_restored":gc.isenabled()==gc_was_enabled,
        "same_failure_handling":"both policies stop on stale output, late command, collision or expired promise; hover is not success",
        "is_safety_certificate":False,"FPGA_verified":False,"test_read":False}
    for name,value in (("report",report),("gc_events",gc_events),("requests",requests),("waiting_admission",list(waiting.values())),("traces",world.traces),
            ("segments",server.segments),("decisions",decisions),("analyses",analyses),
            ("loop_stalls",loop_stalls),("transport_stalls",{"frontend_compute":server.connection.stalls,
                "frontend_io":io.connection.stalls,"compute_helper":server.helper_stalls,"io_helper":io.helper_stalls})):
        atomic_json(directory/f"{name}.json",value)
    return report
