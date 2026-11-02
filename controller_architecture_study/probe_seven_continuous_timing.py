"""Seven real loops, one segmented worker; exploratory complete-flight comparison."""

import argparse
from contextlib import contextmanager
import ctypes
from datetime import datetime, timezone
import gc
import math
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

import numpy as np

from .decision_dataset import ROOT, atomic_json, sha256
from .dynamic_probe_common import read_json
from .fly_decision_dnn import StoredRoutePlanner
from .framework_cpu_affinity import CpuPlacement
from .framework_runtime import ControllerPool, analyze_budget, route_risk, fixed_planning_profile
from .framework_runtime_editor import read_background_throttle
from .framework_runtime_features import encode_direct, risk_reference
from .framework_runtime_io import ProcessCommandIO
from .framework_runtime_wait import DeadlineWaiter
from .framework_runtime_worker import RuntimeServer
from .framework_runtime_world import FrameworkWorld
from .probe_timing_full_route import MODEL, following_slot, stats_ms
from .safety_comparison_core import BODY_RADIUS, CLEARANCE


MIN_PERIOD = .020
CONTROL_CAP = .080
RECOVERY_LOOKAHEAD = .5
PERIODS_MS = (40.,20.,60.,30.,80.)
POLICIES = ("static","edf_only","adaptive")

_cycle_api=None


def thread_cycles():
    global _cycle_api
    if _cycle_api is None:
        _cycle_api=ctypes.WinDLL("kernel32",use_last_error=True).QueryThreadCycleTime
        _cycle_api.argtypes=(ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulonglong))
        _cycle_api.restype=ctypes.c_int
    cycles=ctypes.c_ulonglong()
    if not _cycle_api(ctypes.c_void_p(-2),ctypes.byref(cycles)):
        raise ctypes.WinError(ctypes.get_last_error())
    return cycles.value


@contextmanager
def frontend_span(records,stage,job=None):
    if records is None:
        yield
        return
    start,cpu,cycles=time.perf_counter(),time.thread_time(),thread_cycles()
    try:
        yield
    finally:
        end,used=time.perf_counter(),time.thread_time()-cpu
        if end-start>=.001:
            records.append(dict(stage=stage,job=job,start_s=start,end_s=end,thread_cpu_s=used,
                thread_cycles=thread_cycles()-cycles))


def urgency(risk):
    # Continuous empirical urgency, NOT a proved safe period or recovery region.
    return RECOVERY_LOOKAHEAD*max(0.,-risk["hdot"])/max(.025,risk["h"])


def requested_period(nominal, risk):
    return max(MIN_PERIOD,nominal/(1.+urgency(risk)))


def two_sided_period(nominal,risk,net_width):
    if risk["h"]<=0.:
        return MIN_PERIOD
    from .inflight_sensing import boundary_urgency
    radius=(net_width/2 if risk["boundary"]=="lateral" else 2.)-BODY_RADIUS-CLEARANCE
    return max(MIN_PERIOD,nominal/(1.+boundary_urgency(risk,radius)))


def next_feedback_release(previous,sample,target,deadline,reserve,*,calendar=False):
    anchor=previous["planned_release_s" if calendar else "job_released_s"] if previous else sample
    lower=anchor+MIN_PERIOD if previous else sample
    return max(lower,min(anchor+target,deadline-reserve))


def control_deadline(nominal_deadline, budget, actuation, physical_budget=True,*,state_only=False):
    # Disabling the physical bound for an ablation retains the control update cap.
    bound = budget["deadline_s"] if physical_budget else actuation+CONTROL_CAP
    return bound if state_only else min(nominal_deadline,bound)


def matched_initials(left, right):
    return left.keys()==right.keys() and all(left[v][k]==right[v][k] for v in left
        for k in ("position","velocity","quaternion","held_control"))


def sensor_event_plan(old,event,now,last_released,nominal):
    release=max(now,last_released+MIN_PERIOD)
    extra=release+MIN_PERIOD<=old["planned_release_s"]
    return {**old,"planned_release_s":release,"slot":old["slot"]-1 if extra else old["slot"],
        "deadline_s":min(old["deadline_s"],event["deadline_s"],release+nominal),
        "nominal_deadline_s":min(old["nominal_deadline_s"],release+nominal),
        "event_inserted":extra,"sensor_capture_s":event["capture_s"],"sensor_received_s":event["received_s"],
        "event_goal_s":event["deadline_s"],"event_installed_s":now,"event_request":True}


def calibrated_response_reserve(report):
    # Frozen from a separate normal run; empirical guard, not a WCET certificate.
    return max(.001,1.25*report.get("capture_to_command_ms",report["response_ms"])["p99"]/1000+.0005)


def cached_input(record,row,last_command,now,minimum_sim_ns=0):
    if record is None:
        return False
    raw,hit,captured=record["packet"]
    return (captured>=max(row["planned_release_s"],last_command)
        and 0.<=now-captured<=MIN_PERIOD and (not minimum_sim_ns or raw.timestamp_ns>=minimum_sim_ns))


def next_frontend_work(completed,inputs,plans,now,key,frontend_order="priority",*,protect=False,planning=None,eager_completion=False,input_wait_limit_s=None):
    if input_wait_limit_s is not None:
        aged=[v for v,(_,r) in inputs.items() if now-r.get('sample_response_received_s',now)>=input_wait_limit_s]
        if aged:
            v=min(aged,key=lambda v:inputs[v][1]['sample_response_received_s'])
            return 'input',v,inputs[v][1]
    if eager_completion and completed:
        v=min(completed,key=lambda name:completed[name][1].get('rpc_return_s',now))
        return 'completed',v,completed[v][1]
    work=[("completed",v,r) for v,(_,r) in completed.items()]
    work.extend(("input",v,r) for v,(_,r) in inputs.items())
    work.extend(("release",v,r) for v,r in plans.items() if r["planned_release_s"]<=now)
    if planning:work.extend(('planning',v,r) for v,r in planning.items())
    if protect:
        unfinished=[item for item in work if item[0]!='completed']
        if unfinished:
            return min(unfinished,key=lambda item:key(item[2]))
        return min(work,key=lambda item:item[2].get('rpc_return_s',now)) if work else (None,None,None)
    if frontend_order=="category":
        rank={"completed":0,"input":1,"release":2}
        choose=lambda item:(rank[item[0]],*key(item[2]))
    elif frontend_order=="deadline":
        choose=lambda item:(item[2]["deadline_s"],item[2]["planned_release_s"],item[2]["priority"])
    else:
        choose=lambda item:key(item[2])
    return min(work,key=choose) if work else (None,None,None)


def planning_obligation(previous,next_release=None):
    # Acknowledged work has no old control deadline left. Its maintenance is
    # due before the next release, so neither old cleanup nor planning starves.
    due=(previous['planned_release_s']+previous['requested_period_s']
        if next_release is None else next_release)
    return dict(vehicle=previous['vehicle'],sample_id=previous['sample_id'],priority=previous['priority'],
        planned_release_s=previous['planned_release_s'],deadline_s=due)


def input_blocks_command(now,row,pending,costs):
    """Do not occupy the frontend across a live computed job's delivery budget."""
    if not costs:return False
    finish=now+costs['input_s']+costs.get('handoff_s',0.)+costs.get('command_s',0.)+costs.get('completion_guard_s',0.)
    return any(r['vehicle']!=row['vehicle'] and 'enqueue_s' in r and 'rpc_return_s' not in r
        and now<r['deadline_s']<finish for r in pending.values())


class ContactStop(Exception):
    pass


def mode_is_difficult(risk, preview_acceleration, net_width, *, relative=False):
    radius=(net_width/2 if risk["boundary"]=="lateral" else 2.)-BODY_RADIUS-CLEARANCE
    margin_limit=.3*radius if relative else .30
    projected_limit=.1*radius if relative else .05
    return (risk["h"]<margin_limit or risk["h"]+.5*min(0.,risk["hdot"])<projected_limit
        or np.linalg.norm(preview_acceleration)>2.5)


def collision_vehicle_count(traces, contact_stop=None):
    vehicles={v for v,rows in traces.items() if any(r["collision"] for r in rows)}
    if contact_stop:vehicles.add(contact_stop["vehicle"])
    return len(vehicles)


def flight(world, pool, period_ms, scenario, policy, directory, duration, placement, *, sensing=None, response_reserve_s=None, frontend_order="priority", trace_frontend=False, resident_worker=False, model=MODEL, scheduler=None, physical_budget=True, mode_policy=None, stop_on_collision=False, loop_config=None, collision_policy="global", joint_shadow=None):
    is_static = policy in ("static","static_now","critical_fast")
    continuous = policy in ("adaptive","adaptive_now","adaptive_twosided")
    immediate = policy in ("static_now","event_edf","adaptive_now")
    scheduler = scheduler or ("static" if is_static else "edf")
    if scheduler not in ("static","edf") or model not in pool.models:
        raise ValueError("unknown scheduling policy or controller model")
    if frontend_order not in ("priority","category","deadline"):
        raise ValueError("unknown frontend processing order")
    config=loop_config or {}
    io_backend=config.get('io_backend','python')
    if io_backend not in ('python','cpp'):raise ValueError('unknown I/O backend')
    if joint_shadow is not None and io_backend!='cpp':raise ValueError('joint shadow requires native calendar checks')
    recovery_assumptions=None
    if config.get('recovery_budget') is not None:
        from .recovery_budget import RecoveryAssumptions
        recovery_assumptions=RecoveryAssumptions(**config['recovery_budget'])
    frozen_profile=(fixed_planning_profile(pool.profile,config['planning_profile'])
        if config.get('planning_profile') is not None else None)
    calendar=config.get("release_contract")=="committed_calendar_v1"
    release_tape=config.get('release_tape')
    if release_tape is not None and (not calendar or not is_static or set(release_tape)!=set(world.vehicles)):
        raise ValueError('diagnostic tapes require fixed committed arrivals for every vehicle')
    if calendar and not (is_static or continuous):
        raise ValueError("committed calendar supports periodic static and state-triggered adaptive policies")
    deadline_policy=config.get("deadline_policy","observe")
    if deadline_policy not in ("observe","enforce"):raise ValueError("unknown deadline policy")
    enforce=deadline_policy=="enforce"
    eager_completion=config.get('eager_completion',calendar)
    queue_policy=config.get('queue_policy','latest' if calendar and not enforce and release_tape is None else 'all')
    recover_after_miss=config.get('recover_after_miss',calendar and not enforce)
    if enforce and (queue_policy!='all' or recover_after_miss):
        raise ValueError('strict enforcement cannot ignore missed deadlines or coalesce arrivals')
    if eager_completion and not calendar:raise ValueError('eager completion requires a committed calendar')
    pipeline_costs=config.get("pipeline_costs")
    if enforce and not pipeline_costs:raise ValueError("enforcement requires frozen full-pipeline costs")
    global_admission=config.get('global_admission',False)
    joint_adaptation=config.get('joint_adaptation',False)
    frontend_aging=config.get('frontend_aging',joint_adaptation)
    if joint_adaptation and (not global_admission or not config.get('compute_cost_s') or io_backend!='cpp'):
        raise ValueError('joint adaptation requires global admission, frozen compute costs and native calendar batches')
    frontend_protection=config.get('frontend_protection',True)
    if global_admission and (not continuous or not calendar or not pipeline_costs):
        raise ValueError('global admission requires a continuous committed calendar and frozen pipeline costs')
    if pipeline_costs:
        pipeline_costs={**pipeline_costs,'frontend_protection':frontend_protection,'planning_enabled':continuous,
            'eager_completion':eager_completion}
        if eager_completion:
            pipeline_costs['io_slots']=min(config.get('max_inflight',len(world.vehicles)),len(world.vehicles))
        if frontend_aging:pipeline_costs['input_wait_limit_s']=MIN_PERIOD/2
    planner_costs=({**pipeline_costs,'input_s':pipeline_costs['output_s']+pipeline_costs.get('planner_reserve_s',0.)}
        if pipeline_costs else None)
    models={v:config.get("models",{}).get(v,model) for v in world.vehicles}
    periods={v:config.get("periods_ms",{}).get(v,period_ms)/1000 for v in world.vehicles}
    policies={v:config.get("mode_policy",{}).get(v,mode_policy) for v in world.vehicles}
    priority=config.get("priority_order",world.vehicles)
    if set(priority)!=set(world.vehicles) or len(priority)!=len(world.vehicles):
        raise ValueError("priority order must contain each vehicle exactly once")
    if any(m not in pool.models for m in models.values()) or any(not math.isfinite(p) or p<MIN_PERIOD for p in periods.values()):
        raise ValueError("existing models and finite periods >=20ms required")
    if config.get("max_inflight",len(world.vehicles))<1:
        raise ValueError("at least one in-flight slot required")
    if collision_policy not in ("global","vehicle"):
        raise ValueError("unknown collision policy")
    if any(m and (is_static or any(m[k] not in pool.models for k in ("cruise","recovery"))) for m in policies.values()):
        raise ValueError("mode adaptation requires a dynamic policy and existing models")
    if continuous and response_reserve_s is None:
        raise ValueError("continuous release planning requires a measured full-response reserve")
    reserve = float(response_reserve_s) if response_reserve_s is not None else 0.
    if not math.isfinite(reserve) or reserve<0.:
        raise ValueError("finite nonnegative full-response reserve required")
    directory.mkdir(parents=True,exist_ok=False)
    states = world.initialize(scenario,directory)
    initial_states = dict(states)
    server = RuntimeServer(scheduler,placement["inference_cpus"],allow_late=not enforce,resident_worker=resident_worker,model_names=tuple(pool.models))
    planning_profile=server.profile if frozen_profile is None else frozen_profile
    io = None
    requests,analyses,ready,inputs,pending,plans = [],[],{},{},{},{}
    from .framework_releases import ReleaseQueue,calendar_successor,replan_unreleased,retained_unreleased,post_actuation_observation,commit_dynamic_release,retained_successor,taped_release,release_completed_io,acknowledged_sources
    released=ReleaseQueue(world.vehicles,queue_policy)
    sequence = {v:0 for v in world.vehicles}
    first_release,initial_observations = {},{}
    nominal = period_ms/1000
    origin,error = None,None
    origin_utc_ns=None
    contact_stop=None
    retired,park_states={},{}
    guard_costs=config.get("compute_cost_s")
    desired_periods,committed_periods=dict(periods),dict(periods)
    guard_urgencies={v:0. for v in world.vehicles}
    recovery_memory={v:{} for v in world.vehicles}
    planned_models=dict(models)
    desired_models=dict(models)
    admissions=[]
    admission_needed=True
    initial_taskset_admitted=None
    initial_taskset_forecast=None
    old_gc = gc.isenabled()
    finalizer_errors = []
    sensor_decisions,deferred_events=[],{}
    sensor_armed={v:True for v in world.vehicles}
    sensor_twosided_armed={v:True for v in world.vehicles}
    sensor_stamps={v:0 for v in world.vehicles}
    last_released={v:float("-inf") for v in world.vehicles}
    last_command={v:float("-inf") for v in world.vehicles}
    sensor_latest={}
    completed={}
    feedback={}
    terminal_feedback_cancelled=[]
    frontend_spans=[] if trace_frontend else None

    def park_retired(v):
        if v in retired and not retired[v]["parked"] and (io is None or v not in io.futures):
            world.park_vehicle(v,park_states[v])
            retired[v].update(parked=True,parked_wall_s=time.perf_counter())

    def check_contact(v,raw,hit):
        nonlocal contact_stop
        if stop_on_collision and int(hit.time_stamp)>max(0,world.initial[v]["collision_stamp"]):
            if collision_policy=="vehicle":
                if v not in retired:
                    retired[v]=dict(vehicle=v,kind="collision",object=str(hit.object_name),collision_stamp=int(hit.time_stamp),
                        observed_t_s=(raw.timestamp_ns-world.initial[v]["sim_ns"])/1e9,
                        stopped_wall_s=time.perf_counter(),position=raw.position.tolist(),parked=False,
                        workload_rule="failed loop withdraws; survivors continue; postfailure load is separately reported")
                    park_states[v]=raw
                    plans.pop(v,None);deferred_events.pop(v,None)
                    feedback.pop(v,None)
                    if io_backend=='cpp':io.cancel(v)
                    released.cancel(v,"collision_cancelled_while_queued")
                    print("CONTACT: only vehicle retired",retired[v],flush=True)
                park_retired(v)
                return
            world.client.simPause(True)
            contact_stop=dict(vehicle=v,object=str(hit.object_name),collision_stamp=int(hit.time_stamp),
                observed_t_s=(raw.timestamp_ns-world.initial[v]["sim_ns"])/1e9,paused=True,
                stopped_wall_s=time.perf_counter(),collision_flag=bool(hit.has_collided),
                position=raw.position.tolist())
            print("CONTACT: simulation PAUSED",contact_stop,flush=True)
            raise ContactStop()

    def deadline_stop(v,row,reason):
        nonlocal admission_needed
        if v in retired:
            park_retired(v)
            return
        expired_at_detection=time.perf_counter()>row["deadline_s"]
        raw,hit,_=world.snapshot(v)
        check_contact(v,raw,hit)
        if v in retired:return
        fault_type=("prediction_infeasible" if row.get("status")=="admission_rejected" else
            "actual_response_overrun" if row.get("status")=="applied" else row.get("status","timing_guard"))
        event=dict(vehicle=v,kind="deadline",fault_type=fault_type,reason=reason,sample_id=row["sample_id"],
            actual_deadline_elapsed=expired_at_detection,
            observed_t_s=(raw.timestamp_ns-world.initial[v]["sim_ns"])/1e9,
            stopped_wall_s=time.perf_counter(),position=raw.position.tolist(),parked=False,
            workload_rule="timing failure; this loop withdraws; no safety or mission success inferred")
        retired[v]=event;park_states[v]=raw
        admission_needed=True
        plans.pop(v,None);deferred_events.pop(v,None)
        feedback.pop(v,None)
        if io_backend=='cpp':io.cancel(v)
        released.cancel(v,"deadline_cancelled_while_queued")
        world.failures[v]=dict(kind="deadline",t=event["observed_t_s"],reason=reason)
        print("CONTRACT: only vehicle retired",v,fault_type,reason,flush=True)
        park_retired(v)

    def order(row):
        return (row["priority"] if scheduler=="static" else row["deadline_s"],row["planned_release_s"])

    def make_plan(v, raw, observation, action, actuation, previous=None):
        nonlocal admission_needed
        admission_needed=True
        began = time.perf_counter()
        nominal=periods[v]
        sample = observation["sample_s"]
        budget = analyze_budget(raw,action,sample,actuation,world.gates[v],world.net_width,
            recovery_assumptions=recovery_assumptions)
        base_period=MIN_PERIOD if policy=="critical_fast" and v==scenario.get("fast_vehicle") else nominal
        target = (two_sided_period(nominal,observation["risk"],world.net_width) if policy=="adaptive_twosided" else
            requested_period(nominal,observation["risk"]) if continuous else base_period)
        selected_model=models[v]
        mode=policies[v]
        difficult=False
        anticipation=None
        if mode:
            relative=mode.get("contract") in ("relative_preview_v2","relative_preview_v3")
            if mode.get("contract")=="relative_preview_v3":
                from .heterogeneous_prediction import anticipatory_mode
                difficult,anticipation=anticipatory_mode(world.windows[v],world.phases[v]+observation["t"],
                    observation,world.net_width,recovery_memory[v])
            else:
                preview=risk_reference(world.windows[v],world.phases[v]+observation["t"]+.4)
                difficult=mode_is_difficult(observation["risk"],preview.acceleration,world.net_width,relative=relative)
            selected_model=mode["recovery" if difficult else "cruise"]
            if relative and difficult:
                target=min(target,MIN_PERIOD)
        admission=None
        desired_models[v]=selected_model
        if not joint_adaptation:planned_models[v]=selected_model
        if guard_costs and continuous:
            from .heterogeneous_load_guard import allocate,load
            desired_periods[v]=target if difficult else .060
            guard_urgencies[v]=urgency(observation["risk"])+(1. if difficult else 0.)
            if joint_adaptation:target=desired_periods[v]
            else:
                active_models={name:m for name,m in planned_models.items() if name not in retired}
                allocation,admission=allocate({name:desired_periods[name] for name in active_models},guard_urgencies,active_models,guard_costs)
                candidate={name:p for name,p in committed_periods.items() if name not in retired}
                candidate[v]=allocation[v]
                if load(candidate,active_models,guard_costs)>.95+1e-12 and allocation[v]<committed_periods[v]:
                    target=committed_periods[v]
                    admission["awaiting_other_loop_relaxation"]=True
                else:
                    target=allocation[v]
                committed_periods[v]=target
                admission.update(committed_load=load({name:committed_periods[name] for name in active_models},active_models,guard_costs),
                    scope="shared serialized compute envelope; not full-pipeline response-time certification")
        if global_admission and previous is None and not joint_adaptation:
            target=nominal;selected_model=models[v]
            committed_periods[v]=target;planned_models[v]=selected_model
        index = priority.index(v)
        if previous is None:
            offset = config.get("release_offsets_s",{}).get(v,index*nominal/len(world.vehicles))
            release = sample+offset*(math.exp(-urgency(observation["risk"])) if policy=="adaptive" else 1.)
            if immediate:
                release = sample
            first_release[v] = release if immediate else sample+offset
            slot,skipped = 0,0
            if policy=="event_insert" and observation["risk"]["h"]+RECOVERY_LOOKAHEAD*min(0.,observation["risk"]["hdot"])<=0.:
                release = sample
                slot = -1 if offset>0. else 0
        elif continuous:
            release = previous["planned_release_s"]+target
            slot,skipped = previous["slot"]+1,0
        else:
            earliest = max(time.perf_counter(),previous["job_released_s"]+MIN_PERIOD) if policy in ("event_online","event_twosided") else (
                max(time.perf_counter(),previous["planned_release_s"]+MIN_PERIOD) if policy=="event_insert" else time.perf_counter())
            slot,skipped = following_slot(previous["slot"],earliest,first_release[v],base_period)
            release = first_release[v]+slot*base_period
        nominal_deadline = release+target
        deadline = nominal_deadline if is_static else control_deadline(nominal_deadline,budget,actuation,physical_budget)
        if mode and mode.get("contract")=="relative_preview_v3":
            deadline=control_deadline(nominal_deadline,budget,actuation,physical_budget,state_only=True)
        if policy=="event_edf" and previous is None:
            deadline = min(deadline,sample+requested_period(nominal,observation["risk"]))
        if policy=="event_insert" and previous is None and release==sample:
            deadline = min(deadline,sample+requested_period(nominal,observation["risk"]))
        if continuous:
            earliest = previous["planned_release_s"]+MIN_PERIOD if previous else sample
            release = max(earliest,min(release,deadline-reserve))
            if policy=="adaptive_twosided":
                release=next_feedback_release(previous,sample,target,deadline,reserve,calendar=calendar)
        if mode and mode.get("contract")=="relative_preview_v3":nominal_deadline=release+target
        row = {"sample_id":f"{v}:{sequence[v]}","vehicle":v,"model":selected_model,"priority":index,
            "planned_release_s":release,"deadline_s":deadline,"nominal_deadline_s":nominal_deadline,
            "requested_period_s":target,"slot":slot,"skipped_releases":skipped,"status":"planned",
            "obligation_actuation_s":actuation,"obligation_sample_s":sample}
        row["minimum_release_s"]=(previous["planned_release_s" if calendar else "job_released_s"]+MIN_PERIOD if previous else first_release[v]) if continuous else release
        if calendar:row['serial_vehicle']=True
        row["event_inserted"] = policy=="event_insert" and slot==-1
        analyses.append({**budget,"vehicle":v,"sample_id":row["sample_id"],"risk":observation["risk"],
            "requested_period_s":target,"planned_release_s":release,"response_reserve_s":reserve,
            "job_deadline_s":deadline,"B_binding":physical_budget and not is_static and sample+budget["budget_s"]<=min(nominal_deadline,actuation+CONTROL_CAP),
            "physical_budget_enabled":physical_budget,
            "resource_admission":admission,
            "anticipation":anticipation,
            "prediction_already_late":deadline<=time.perf_counter() if origin is not None else deadline<=sample,
            "analysis_ms":1000*(time.perf_counter()-began),"continuous_rule_is_safety_certificate":False})
        if joint_shadow is not None:
            joint_shadow.note(v,raw,observation,action,actuation,budget,row)
        return row

    def consider_sensor(record):
        from .inflight_sensing import event_edge, projected_margin
        began=time.perf_counter()
        raw,hit,captured=record["packet"]
        v=record["row"]["vehicle"]
        if v in retired:return
        nominal=periods[v]
        sensor_latest[v]=record
        elapsed=(raw.timestamp_ns-world.initial[v]["sim_ns"])/1e9
        if raw.timestamp_ns<=max(sensor_stamps[v],states[v].timestamp_ns) or not 0.<=elapsed<duration:
            return
        sensor_stamps[v]=raw.timestamp_ns
        risk=route_risk(raw,risk_reference(world.windows[v],world.phases[v]+elapsed),world.net_width)
        old_edge,sensor_armed[v]=event_edge(sensor_armed[v],risk)
        radius=(world.net_width/2 if risk["boundary"]=="lateral" else 2.)-BODY_RADIUS-CLEARANCE
        new_edge,sensor_twosided_armed[v]=event_edge(sensor_twosided_armed[v],risk,radius=radius)
        edge=new_edge if policy in ("event_twosided","adaptive_twosided") else old_edge
        goal=two_sided_period(nominal,risk,world.net_width) if policy in ("event_twosided","adaptive_twosided") else requested_period(nominal,risk)
        decision={"vehicle":v,"sim_ns":int(raw.timestamp_ns),"capture_s":captured,
            "received_s":record["received_s"],"risk":risk,"edge":edge,
            "one_sided_edge":old_edge,"two_sided_edge":new_edge,
            "one_sided_margin":projected_margin(risk),"two_sided_margin":projected_margin(risk,radius),
            "deadline_s":captured+goal,"applied":policy in ("event_online","event_twosided","adaptive_twosided") and edge,
            "detection_ms":1000*(time.perf_counter()-began)}
        sensor_decisions.append(decision)
        if decision["applied"]:
            deferred_events[v]=decision

    def install_event(v,event):
        nonlocal admission_needed
        admission_needed=True
        physical_deadline=plans[v]["deadline_s"]
        old=plans[v]
        candidate=sensor_event_plan(old,event,time.perf_counter(),last_released[v],periods[v])
        if policies[v] and policies[v].get("contract")=="relative_preview_v3":
            # A risk-triggered earlier request does not invent a tighter physical D.
            candidate["deadline_s"]=physical_deadline
        plans[v]=protect_change(candidate,{**old,'deadline_s':candidate['deadline_s']}) if global_admission else candidate

    def protect_change(candidate,fallback):
        from .heterogeneous_deadlines import pending_specs,admit_change
        now=time.perf_counter()
        specs=pending_specs(now,pending,ready,inputs,completed,io,pipeline_costs,
            feedback=feedback if eager_completion else None,current_feedback=candidate.get('feedback_from_job'))
        specs.extend({**r,'stage':'input'} for r in released.rows())
        if joint_adaptation:
            from .heterogeneous_deadlines import admit_group
            from .heterogeneous_load_guard import allocate,resource_loads
            base={**plans,candidate['vehicle']:fallback}
            desired={v:desired_models[v] for v in base}
            allocation,rate_info=allocate({v:desired_periods[v] for v in base},guard_urgencies,
                desired,guard_costs,pipeline_costs=pipeline_costs,
                max_periods={v:periods[v] if desired_periods[v]<=MIN_PERIOD else .060 for v in base})
            proposal={}
            for name,row in base.items():
                proposed={**row,'model':desired[name],'requested_period_s':allocation[name]}
                proposal[name]=(replan_unreleased(proposed,last_released[name],
                    int(row['sample_id'].rsplit(':',1)[1]),now,reserve) if math.isfinite(last_released[name]) else proposed)
            unchanged=all(v in plans and all(proposal[v][k]==plans[v][k] for k in ('model','requested_period_s','planned_release_s'))
                and proposal[v]['deadline_s']>=plans[v]['deadline_s'] for v in proposal)
            demands=resource_loads({v:r['requested_period_s'] for v,r in proposal.items()},desired,guard_costs,pipeline_costs)
            if unchanged and max(demands.values())<=.95+1e-12:
                selected_group=proposal
                info=dict(decision='deadline_refresh',feasible=None,future_feasible=None,checked=0,
                    attempts=[],planning_ms=1000*(time.perf_counter()-now),already_expired_ids=[],
                    scope='unchanged release/mode/rate; fresh held-control deadline only, no new timing guarantee')
            else:
                selected_group,info=admit_group(now,base,specs,proposal,planning_profile,pipeline_costs,guard_costs,
                    config.get('max_inflight',len(world.vehicles)),policy=scheduler,recover_after_miss=recover_after_miss)
            info['allocation']=rate_info
            # Publication proceeds independently during the Python forecast.
            # Commit no part of the proposal if any native job ID has advanced.
            if not io.schedule_batch(selected_group,origin+duration):
                info.update(decision='calendar_race_retained',feasible=False,future_feasible=False,
                    reason='a native release raced the forecast; no configuration in the group was installed')
                selected=plans.get(candidate['vehicle'],fallback)
            else:
                plans.update(selected_group)
                for name,row in selected_group.items():
                    committed_periods[name]=row['requested_period_s'];planned_models[name]=row['model']
                selected=selected_group[candidate['vehicle']]
        else:
            selected,info=admit_change(now,plans,specs,candidate,fallback,planning_profile,pipeline_costs,
                config.get('max_inflight',len(world.vehicles)),policy=scheduler,recover_after_miss=recover_after_miss)
        admissions.append(dict(trigger=candidate['sample_id'],started_s=now,**info))
        selected['global_admission']=dict(feasible=info['feasible'],decision=info['decision'],
            future_feasible=info.get('future_feasible',False),already_expired_ids=info.get('already_expired_ids',[]),
            checked=info['checked'],planning_ms=info['planning_ms'])
        return selected

    def check_admission():
        nonlocal admission_needed
        from .heterogeneous_deadlines import pending_specs,admit
        now=time.perf_counter()
        specs=pending_specs(now,pending,ready,inputs,completed,io,pipeline_costs)
        specs.extend({**r,'stage':'input'} for r in released.rows())
        proposal,info=admit(time.perf_counter(),plans,specs,planning_profile,pipeline_costs,
            config.get("max_inflight",len(world.vehicles)),policy=scheduler,allow_earlier=not is_static)
        trigger=min([*plans.values(),*released.rows()],key=order);v=trigger['vehicle']
        own_envelope=sum(pipeline_costs.get(k,0.) for k in
            ("sample_io_s","input_s","ingress_s","egress_s","handoff_s","command_s","completion_guard_s"))
        own_envelope+=sum(planning_profile[trigger["model"]]["segment_estimates_s"])
        admissions.append(dict(trigger=trigger["sample_id"],model=trigger["model"],started_s=now,
            deadline_s=trigger["deadline_s"],minimum_release_s=trigger["minimum_release_s"],
            own_empirical_envelope_s=own_envelope,available_s=trigger["deadline_s"]-max(now,trigger["minimum_release_s"]),
            pending_count=len(specs),**info))
        if proposal is None:
            trigger["status"]="admission_rejected"
            deadline_stop(v,trigger,info["reason"])
            return
        plans.update(proposal)
        for row in plans.values():
            if 'previous_release_s' in row:
                row['selected_release_interval_s']=row['planned_release_s']-row['previous_release_s']
        admission_needed=False

    def register_due(now):
        nonlocal admission_needed
        # A release is a job arrival, not permission to occupy an execution slot.
        if io_backend=='cpp':
            scheduled={}
            for v,row in list(plans.items()):
                if v in retired or row['planned_release_s']>=origin+duration-1e-9:
                    plans.pop(v,None);io.cancel(v)
                elif joint_adaptation:scheduled[v]=row
                else:io.schedule(row,origin+duration,static=is_static)
            if joint_adaptation:io.schedule_batch(scheduled,origin+duration)
            for row in io.drain_releases():
                v=row['vehicle']
                if v in retired:continue
                released.publish(row,row['native_release_s'])
                row['release_parent_received_s']=time.perf_counter()
                admission_needed=True;requests.append(row)
                sequence[v]=max(sequence[v],int(row['sample_id'].rsplit(':',1)[1])+1)
                last_released[v]=row['planned_release_s']
                if v in plans and int(plans[v]['sample_id'].rsplit(':',1)[1])<sequence[v]:
                    following=calendar_successor(row,sequence[v],dynamic=not is_static)
                    plans[v]=following
                    # The native timer already installed this periodic successor.
                    io.scheduled[v]=io.plan_key(following)
            return
        for v in world.vehicles:
            while v in plans and plans[v]['planned_release_s']<=now:
                row=plans.pop(v)
                if v in retired or row['planned_release_s']>=origin+duration-1e-9:
                    continue
                released.publish(row,now)
                admission_needed=True
                requests.append(row)
                sequence[v]+=1
                last_released[v]=row['planned_release_s']
                if release_tape is not None:
                    following=taped_release(release_tape,v,sequence[v],origin)
                    if following is not None:plans[v]=following
                else:
                    plans[v]=calendar_successor(row,sequence[v],dynamic=not is_static)

    def plan_from_acknowledgements(only=None):
        # The helper acknowledges actuation BEFORE its post-command state RPC.
        # Use the newest available independent observation; retain its true age.
        sources=acknowledged_sources(pending,feedback) if eager_completion else pending
        for v,row in sorted(sources.items(),key=lambda item:item[1].get('rpc_return_s',float('inf'))):
            if only is not None and v!=only:continue
            if is_static or row.get('status')!='applied' or row.get('next_plan_created') or v in retired:
                continue
            candidates=[(states[v],None,world.traces[v][-1]['sample_s'])]
            if v in sensor_latest:candidates.append(sensor_latest[v]['packet'])
            if v in completed:candidates.append(completed[v][0])
            if v in feedback:candidates.append(feedback[v][0])
            packet=post_actuation_observation(candidates,row['rpc_return_s'])
            if packet is None:continue
            raw,hit,sampled=packet
            if hit is not None:check_contact(v,raw,hit)
            if v in retired:continue
            if hit is None:
                hit=SimpleNamespace(time_stamp=world.initial[v]['collision_stamp'],has_collided=False,object_name='')
            observation=({**world.traces[v][-1],'sample_s':sampled,'rpc_state_stamp_reused':True}
                if joint_adaptation and raw.timestamp_ns==world.traces[v][-1]['sim_ns']
                else world.observe(v,raw,hit,sampled))
            states[v]=raw
            row['next_plan_created']=True
            row['next_plan_started_s']=time.perf_counter()
            if observation['t']<duration:
                candidate=make_plan(v,raw,observation,row['mapped_control'],row['rpc_return_s'],row)
                candidate['feedback_from_job']=row['sample_id']
                if calendar:
                    candidate=replan_unreleased(candidate,last_released[v],sequence[v],time.perf_counter(),reserve)
                if global_admission:
                    fallback=(retained_unreleased(candidate,plans.get(v,row),last_released[v],sequence[v],time.perf_counter(),reserve)
                        if calendar else retained_successor(candidate,row,time.perf_counter()))
                    candidate=protect_change(candidate,fallback)
                    committed_periods[v]=candidate['requested_period_s'];planned_models[v]=candidate['model']
                    analyses[-1].update(proposed_period_s=analyses[-1]['requested_period_s'],
                        requested_period_s=candidate['requested_period_s'],global_admission=candidate['global_admission'])
                plans[v]=candidate
                if not calendar:commit_dynamic_release(plans[v],row,time.perf_counter())
                plans[v]['planning_state_age_s']=max(0.,time.perf_counter()-sampled)
            row['next_plan_done_s']=time.perf_counter()
            if eager_completion and v in feedback:
                old=feedback.pop(v)[1]
                if old is not row:old['planning_superseded_by']=row['sample_id']
            if frontend_protection:return True
        return False

    def planning_jobs():
        result={}
        if is_static:return result
        sources=acknowledged_sources(pending,feedback) if eager_completion else pending
        for v,row in sources.items():
            if row.get('status')!='applied' or v in retired:continue
            if input_blocks_command(time.perf_counter(),row,pending,planner_costs):continue
            if eager_completion:
                if not row.get('next_plan_created'):
                    candidates=[feedback[v][0]] if v in feedback else []
                    if v in sensor_latest:candidates.append(sensor_latest[v]['packet'])
                    candidates.append((states[v],None,world.traces[v][-1]['sample_s']))
                    if post_actuation_observation(candidates,row['rpc_return_s']) is not None:
                        result[v]=planning_obligation(row)
                continue
            if row.get('next_plan_created'):
                following=plans.get(v) or (released.queues[v][0] if released.queues[v] else None)
                if v in completed and following:
                    result[v]=planning_obligation(row,following['planned_release_s'])
            elif (v in completed or (v in sensor_latest and sensor_latest[v]['packet'][2]>=row['rpc_return_s'])
                    or world.traces[v][-1]['sample_s']>=row['rpc_return_s']):
                result[v]=planning_obligation(row)
        return result

    def shadow_pending(now):
        from .heterogeneous_deadlines import pending_specs
        specs=pending_specs(now,pending,ready,inputs,completed,io,pipeline_costs,
            feedback=feedback if eager_completion else None)
        return specs+[{**r,'stage':'input'} for r in released.rows()]

    try:
        if server.provenance!=pool.provenance():
            raise ValueError("inference worker model identity mismatch")
        io_cpus=placement["io_cpus"][:1] if sensing is not None else placement["io_cpus"]
        io_class=ProcessCommandIO
        io_vehicles=len(world.vehicles)
        if io_backend=='cpp':
            from .framework_native_io import NativeCommandIO
            if not calendar or release_tape is not None:raise ValueError('native runtime requires a live committed calendar')
            io_class=NativeCommandIO;io_vehicles=world.vehicles
        io = io_class(io_vehicles,world.calibration,pipeline_costs["actuation_s"] if enforce else .003,io_cpus,allow_late=not enforce)
        if sensing is not None:
            sensing.prepare()
        for v in world.vehicles:
            hit = SimpleNamespace(has_collided=False,time_stamp=world.initial[v]["collision_stamp"],object_name="")
            initial_observations[v] = world.observe(v,states[v],hit,0.)
            encode_direct(world.windows[v],states[v],world.phases[v])
            plans[v] = make_plan(v,states[v],initial_observations[v],world.initial[v]["held_control"],0.)
            if release_tape is not None:plans[v]=taped_release(release_tape,v,0,0.)
        if joint_adaptation:
            from .heterogeneous_load_guard import allocate
            allocation,rate_info=allocate(desired_periods,guard_urgencies,desired_models,guard_costs,pipeline_costs=pipeline_costs,
                max_periods={v:periods[v] if desired_periods[v]<=MIN_PERIOD else .060 for v in plans})
            for v,row in plans.items():
                row.update(requested_period_s=allocation[v],nominal_deadline_s=row['planned_release_s']+allocation[v])
                committed_periods[v]=allocation[v];planned_models[v]=row['model']
            admissions.append(dict(trigger='initial_sustained_demand',outside_physics=True,**rate_info))
        if enforce or global_admission:
            from .heterogeneous_deadlines import admit
            proposal,info=admit(0.,plans,[],planning_profile,pipeline_costs,
                config.get("max_inflight",len(world.vehicles)),policy=scheduler,allow_earlier=not is_static)
            initial_taskset_forecast=proposal is not None and (not joint_adaptation or rate_info['feasible'])
            admissions.append(dict(trigger="initial_taskset",outside_physics=True,**info))
            if enforce:
                initial_taskset_admitted=initial_taskset_forecast
                plans.clear()
                if proposal:plans.update(proposal)
            elif proposal is not None:
                plans.update(proposal)
            admission_needed=False
            if initial_taskset_admitted is False:print("CONTRACT: initial task set not admitted; no control mission started",flush=True)
            elif initial_taskset_forecast is False:print('FORECAST: initial task set infeasible; retained baseline is observational',flush=True)
        gc.collect()
        gc.disable()
        if trace_frontend:thread_cycles()
        origin = time.perf_counter()
        origin_utc_ns=time.time_ns()
        for v in world.vehicles:
            initial_observations[v]["sample_s"] = origin
            first_release[v] += origin
            last_command[v]=origin
        for row in [*plans.values(),*analyses]:
            for key in ("planned_release_s","minimum_release_s","deadline_s","nominal_deadline_s","analysis_s","actuation_proxy_s","job_deadline_s",
                        'dispatch_not_before_s','delivery_not_before_s','obligation_actuation_s','obligation_sample_s'):
                if key in row:
                    row[key] += origin
        if io_backend=='cpp':register_due(time.perf_counter())
        if joint_shadow is not None:joint_shadow.start(world.initial,origin+duration)
        if initial_taskset_admitted is not False:world.client.simPause(False)
        if sensing is not None:
            sensing.start(origin,duration if initial_taskset_admitted is not False else 0.,world.initial)
        with DeadlineWaiter() as timer:
            while plans or pending or released or (eager_completion and (feedback or completed)):
                now = time.perf_counter()
                if now-origin > 2*duration+3:
                    queues=dict(plans={v:r['sample_id'] for v,r in plans.items()},
                        pending={v:r['status'] for v,r in pending.items()},released=len(released.rows()),
                        inputs=list(inputs),ready=list(ready),completed=list(completed),
                        feedback={v:r.get('next_plan_created',False) for v,(_,r) in feedback.items()})
                    raise RuntimeError("bounded physical window did not complete: "+str(queues))
                if calendar:register_due(now)
                if now>=origin+duration and not (plans or pending or released or inputs or ready or completed):
                    # No control obligation remains. Feedback can only create
                    # successors outside this finite mission; it must not keep
                    # a completed flight waiting for another observation.
                    terminal_feedback_cancelled.extend(r['sample_id'] for _,r in feedback.values())
                    feedback.clear()
                    break
                with frontend_span(frontend_spans,"inference_drain"):
                    outputs=server.drain()
                for action,row in outputs:
                    if action is None:
                        if enforce:
                            pending.pop(row["vehicle"],None)
                            deadline_stop(row["vehicle"],row,"inference deadline expired")
                            continue
                        raise RuntimeError("exploratory executor unexpectedly discarded an output")
                    row['delivery_not_before_s']=max(row.get('delivery_not_before_s',0.),
                        row.get('sample_s',0.)+row.get('observed_send_wait_s',0.))
                    ready[row["vehicle"]] = action,row
                with frontend_span(frontend_spans,"sample_drain"):
                    packets=io.drain_samples()
                for packet,row in packets:
                    inputs[row["vehicle"]] = packet,row
                with frontend_span(frontend_spans,"command_drain"):
                    packets=io.drain()
                for packet,row in packets:
                    if eager_completion:release_completed_io(pending,row,time.perf_counter())
                    if row.get('status')=='applied':last_command[row['vehicle']]=max(last_command[row['vehicle']],row['rpc_return_s'])
                    if joint_shadow is not None and row.get('status')=='applied':
                        joint_shadow.observe_ack(packet,row,plans.get(row['vehicle']))
                    if row['vehicle'] in completed:raise RuntimeError('unprocessed completion overwritten')
                    completed[row["vehicle"]]=packet,row
                if calendar:
                    if not frontend_protection:plan_from_acknowledgements()
                    register_due(time.perf_counter())
                if enforce:
                    now=time.perf_counter()
                    for queued in released.rows():
                        if queued['vehicle'] not in retired and now>=queued['deadline_s']:
                            deadline_stop(queued['vehicle'],queued,"released job expired while queued; release not postponed")
                    for v,row in list(pending.items()):
                        if row.get("status")=="applied":last_command[v]=max(last_command[v],row["rpc_return_s"])
                        if v not in retired and row.get("status")!="applied" and now>=row["deadline_s"]:
                            deadline_stop(v,row,"committed deadline elapsed while a pipeline operation was pending")
                    for v in set(plans)|set(pending):
                        if v not in retired and now>=last_command[v]+CONTROL_CAP and now<origin+duration:
                            deadline_stop(v,dict(sample_id=v+":held_control",deadline_s=last_command[v]+CONTROL_CAP,
                                status="control_update_limit_expired"),"maximum control update interval elapsed")
                # Preserve command priority over feature processing and release decisions.
                deliverable=[v for v,(_,r) in ready.items() if time.perf_counter()>=r.get('delivery_not_before_s',0.)]
                if deliverable:
                    v = min(deliverable,key=lambda v:order(ready[v][1]))
                    action,row = ready.pop(v)
                    if v in retired:
                        row["status"]="collision_discarded_before_command"
                        pending.pop(v,None);park_retired(v)
                        continue
                    if enforce:
                        from .heterogeneous_deadlines import delivery_fits
                        if not delivery_fits(time.perf_counter(),row,pipeline_costs):
                            row["status"]="delivery_budget_exhausted"
                            pending.pop(v,None)
                            deadline_stop(v,row,"remaining command budget cannot meet the unchanged deadline")
                            continue
                    with frontend_span(frontend_spans,"command_submit",row["sample_id"]):
                        io.send(action,row)
                    continue
                if sensing is not None:
                    with frontend_span(frontend_spans,"sensor_drain"):
                        records=sensing.drain(maximum=1)
                    for record in records:
                        check_contact(record["row"]["vehicle"],record["packet"][0],record["packet"][1])
                        with frontend_span(frontend_spans,"sensor_analysis",record["row"]["vehicle"]):
                            consider_sensor(record)
                    for v,event in list(deferred_events.items()):
                        if v in plans and v not in pending:
                            install_event(v,event)
                            deferred_events.pop(v)
                now=time.perf_counter()
                retired_pending=any(v in retired for v in pending)
                if enforce and (plans or released) and admission_needed and not inputs and not completed and not retired_pending:
                    check_admission()
                    continue
                dispatch_plans=(released.dispatchable(pending,config.get("max_inflight",len(world.vehicles)),now) if calendar else
                    plans if not (enforce and retired_pending) and len(pending)<config.get("max_inflight",len(world.vehicles)) else {})
                eligible_inputs={v:r for v,r in inputs.items() if not (frontend_protection
                    and input_blocks_command(now,r[1],pending,pipeline_costs))}
                cleanup=({v:r for v,r in completed.items() if not frontend_protection or r[1].get('next_plan_created')
                    or not input_blocks_command(now,r[1],pending,planner_costs)} if not inputs or eligible_inputs else {})
                if eager_completion:cleanup=dict(completed)
                kind,v,_=next_frontend_work(cleanup,eligible_inputs,dispatch_plans,now,order,frontend_order,
                    protect=frontend_protection,planning=planning_jobs() if calendar and frontend_protection else None,
                    eager_completion=eager_completion,input_wait_limit_s=MIN_PERIOD/2 if frontend_aging else None)
                if kind=='planning':
                    plan_from_acknowledgements(only=v)
                    register_due(time.perf_counter())
                    if v not in completed:continue
                    kind='completed'
                if kind=="completed":
                    if calendar and frontend_protection and not eager_completion:plan_from_acknowledgements(only=v)
                    packet,row=completed.pop(v)
                    row["completion_frontend_start_s"]=time.perf_counter()
                    if packet is None or row["status"]!="applied":
                        if enforce:
                            pending.pop(v,None)
                            deadline_stop(v,row,"command deadline guard rejected delivery")
                            continue
                        raise RuntimeError("exploratory command unexpectedly discarded")
                    if not eager_completion:pending.pop(v)
                    if enforce and row["rpc_return_s"]>row["deadline_s"]:
                        deadline_stop(v,row,"actual command RPC returned beyond the committed deadline")
                    raw,hit,sampled=packet
                    check_contact(v,raw,hit)
                    if v in retired:
                        park_retired(v)
                        row["completion_frontend_done_s"]=time.perf_counter()
                        continue
                    last_command[v]=max(last_command[v],row["rpc_return_s"])
                    with frontend_span(frontend_spans,"postcommand_observation",row["sample_id"]):
                        if not calendar or raw.timestamp_ns>=world.traces[v][-1]['sim_ns']:
                            observation=world.observe(v,raw,hit,sampled)
                            states[v]=raw
                        else:observation=world.traces[v][-1]
                    if observation["t"]<duration and not calendar:
                        with frontend_span(frontend_spans,"next_plan",row["sample_id"]):
                            plans[v]=make_plan(v,raw,observation,row["mapped_control"],row["rpc_return_s"],row)
                    if eager_completion and not is_static and not row.get('next_plan_created'):
                        if v in feedback:feedback[v][1]['planning_superseded_by']=row['sample_id']
                        feedback[v]=packet,row
                    row["completion_frontend_done_s"]=time.perf_counter()
                    continue
                if kind=="input":
                    packet,row = inputs.pop(v)
                    raw,hit,sampled = packet
                    check_contact(v,raw,hit)
                    if v in retired:
                        pending.pop(v,None)
                        row["status"]="collision_discarded_before_inference"
                        park_retired(v)
                        continue
                    row["frontend_state_start_s"]=time.perf_counter()
                    with frontend_span(frontend_spans,"input_observation",row["sample_id"]):
                        record_input=raw.timestamp_ns>=world.traces[v][-1]['sim_ns']
                        observation = world.observe(v,raw,hit,sampled,record=record_input)
                        row['input_observation_recorded']=record_input
                    if sensing is not None and raw.timestamp_ns>=states[v].timestamp_ns:
                        states[v]=raw
                    if observation["t"] >= duration:
                        pending.pop(v)
                        row["status"] = "after_window_not_computed"
                        continue
                    feature_start = time.perf_counter()
                    with frontend_span(frontend_spans,"input_encode",row["sample_id"]):
                        feature = encode_direct(world.windows[v],raw,world.phases[v]+observation["t"])
                    row.update(sample_s=sampled,feature_start_s=feature_start,feature_done_s=time.perf_counter(),status="submitted")
                    with frontend_span(frontend_spans,"inference_submit",row["sample_id"]):
                        server.submit(feature,row)
                    continue
                if kind=="release":
                    row = released.take(v,time.perf_counter()) if calendar else plans.pop(v)
                    if enforce and time.perf_counter()>=row["deadline_s"]:
                        row["status"]="expired_before_release"
                        deadline_stop(v,row,"committed deadline elapsed before request dispatch")
                        continue
                    if not calendar:
                        row["job_released_s"] = time.perf_counter()
                        last_released[v]=row["job_released_s"]
                        requests.append(row)
                        sequence[v] += 1
                    pending[v] = row
                    record=sensor_latest.get(v)
                    if cached_input(record,row,last_command[v],time.perf_counter(),states[v].timestamp_ns):
                        row.update(input_source="sensor_cache",sample_submitted_s=time.perf_counter(),
                            sample_response_received_s=time.perf_counter(),cached_sensor_received_s=record["received_s"])
                        inputs[v]=record["packet"],row
                    else:
                        row["input_source"]="rpc"
                        with frontend_span(frontend_spans,"sample_submit",row["sample_id"]):
                            io.sample(row)
                    continue
                # Advice is background work: already-ready control delivery,
                # input preparation and releases take precedence.
                if joint_shadow is not None:
                    joint_shadow.tick(plans,shadow_pending,io,last_released,last_command,
                        config.get('max_inflight',len(world.vehicles)))
                    if getattr(joint_shadow,'apply_rates',False):
                        for group in joint_shadow.drain_committed():
                            for name,row in group.items():
                                committed_periods[name]=row['requested_period_s'];planned_models[name]=row['model']
                    now=time.perf_counter()
                if plans or pending or released or (eager_completion and (feedback or completed)):
                    wake = min([now+.100,*[r["planned_release_s"] for r in (plans if calendar else dispatch_plans).values()]])
                    future_deliveries=[r['delivery_not_before_s'] for _,r in ready.values() if r.get('delivery_not_before_s',0.)>now]
                    if future_deliveries:wake=min(wake,*future_deliveries)
                    future_dispatches=[r['dispatch_not_before_s'] for r in released.rows() if r.get('dispatch_not_before_s',0.)>now]
                    if future_dispatches:wake=min(wake,*future_dispatches)
                    if frontend_protection:
                        delivery_limits=[r['deadline_s'] for r in pending.values()
                            if 'enqueue_s' in r and 'rpc_return_s' not in r and r['deadline_s']>now]
                        if delivery_limits:wake=min(wake,*delivery_limits)
                    if enforce:
                        limits=[r["deadline_s"] for r in [*pending.values(),*released.rows()] if r['vehicle'] not in retired and r.get("status")!="applied"]
                        limits += [last_command[v]+CONTROL_CAP for v in set(plans)|set(pending) if v not in retired and now<origin+duration]
                        if limits:wake=min(wake,*limits)
                    pipes=[server.connection.connection,io.connection.connection]
                    if sensing is not None:
                        pipes.append(sensing.connection.connection)
                    with frontend_span(frontend_spans,"wait",wake):
                        timer.wait(pipes,wake)
        # Capture all loops at/after the common physical task end, with no reset.
        for v in world.vehicles:
            raw,hit,sampled = world.snapshot(v)
            world.observe(v,raw,hit,sampled)
        if sensing is not None:
            until=time.perf_counter()+.150
            with DeadlineWaiter() as timer:
                while not sensing.covers(world.initial,duration) and time.perf_counter()<until:
                    sensing.drain()
                    timer.wait([sensing.connection.connection],time.perf_counter()+.002)
    except ContactStop:
        pass
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        for resource in (joint_shadow,sensing,io,server):
            if resource is not None:
                try:
                    resource.close()
                except Exception as exc:
                    finalizer_errors.append(str(exc))
        if contact_stop is None:
            try:
                world.hover_all()
            except Exception as exc:
                finalizer_errors.append(str(exc))
        if old_gc:
            gc.enable()
    control_traces=world.traces
    control_vehicles=world.physical_report(duration)
    if sensing is not None:
        world.traces={v:[] for v in world.vehicles}
        world.failures={v:None for v in world.vehicles}
        for v in world.vehicles:
            hit=SimpleNamespace(has_collided=False,time_stamp=world.initial[v]["collision_stamp"],object_name="")
            world.observe(v,initial_states[v],hit,origin or 0.)
            records=sorted((r for r in sensing.records if r["row"]["vehicle"]==v),key=lambda r:r["packet"][0].timestamp_ns)
            for record in records:
                raw,hit,sampled=record["packet"]
                if raw.timestamp_ns>=world.initial[v]["sim_ns"]:
                    world.observe(v,raw,hit,sampled)
                    if (raw.timestamp_ns-world.initial[v]["sim_ns"])/1e9>=duration:
                        break
    for v,event in retired.items():
        if event.get("kind")=="deadline":
            world.failures[v]=dict(kind="deadline",t=event["observed_t_s"],reason=event["reason"])
    vehicles = world.physical_report(duration)
    applied = [r for r in requests if r.get("status")=="applied"]
    end = contact_stop["stopped_wall_s"] if contact_stop else (origin+duration if origin is not None else float("inf"))
    cutoff=end if contact_stop else float("inf")
    def timing_cutoff(row):
        return min(cutoff,retired[row["vehicle"]]["stopped_wall_s"]) if row["vehicle"] in retired else cutoff
    late = [r["sample_id"] for r in applied if r["deadline_s"]<r["rpc_return_s"]<=timing_cutoff(r)]
    nominal_late = [r["sample_id"] for r in applied if r["nominal_deadline_s"]<r["rpc_return_s"]<=timing_cutoff(r)]
    undelivered = [r["sample_id"] for r in requests if r.get("status")!="applied" and r["deadline_s"]<=min(end,timing_cutoff(r))]
    nominal_undelivered = [r["sample_id"] for r in requests if r.get("status")!="applied" and r["nominal_deadline_s"]<=min(end,timing_cutoff(r))]
    ids = [s["sample_id"] for s in server.segments]
    seen,last,resumptions = set(),None,0
    for ident in ids:
        resumptions += ident!=last and ident in seen
        seen.add(ident)
        last = ident
    physical = all(p["physical_task_passed"] for p in vehicles.values())
    success = all(p["recovery_passed"] and p["clock_valid"] for p in vehicles.values()) and error is None and not finalizer_errors
    report = {"policy":policy,"frontend_order":frontend_order,"scenario":scenario,"period_ms":period_ms,"model":model,"vehicles":vehicles,
        "scheduler":scheduler,"physical_budget_enabled":physical_budget,
        "duration_s":duration,"wall_origin_s":origin,"clock_anchor":dict(perf_counter_s=origin,utc_ns=origin_utc_ns),"full_flight_completed":all(p["window_covered"] for p in control_vehicles.values()) and error is None,
        "physical_task_passed":physical,"success":success,"error":error,"cleanup_errors":finalizer_errors,
        "collisions":len({v for v,t in world.traces.items() if any(r["collision"] for r in t)}|{v for v,e in retired.items() if e.get("kind","collision")=="collision"}|({contact_stop["vehicle"]} if contact_stop else set())),"timing_evaluation_end_s":end,
        "requests":len(requests),"applied":len(applied),"late_jobs":late,"nominal_late_jobs":nominal_late,
        "observed_strict_timing_passed":initial_taskset_admitted is not False and not late and not undelivered and not any(e.get("kind")=="deadline" for e in retired.values()) and error is None,
        "nominal_strict_timing_passed":initial_taskset_admitted is not False and not nominal_late and not nominal_undelivered and not any(e.get("kind")=="deadline" for e in retired.values()) and error is None,
        "undelivered_deadlines":undelivered,"nominal_undelivered_deadlines":nominal_undelivered,
        "skipped_releases":sum(r["skipped_releases"] for r in requests),
        "extra_corrective_requests":sum(r.get("event_inserted",False) for r in requests),
        "total_J":sum(p["score"]["J"] for p in vehicles.values()) if physical and all(p["score"] for p in vehicles.values()) else None,
        "response_ms":stats_ms([r["rpc_return_s"]-r["job_released_s" if calendar else "sample_submitted_s"] for r in applied]),
        "release_contract":"independent_native_calendar_v2" if io_backend=='cpp' else "independent_calendar_v2" if calendar else "completion_coupled_legacy",
        "io_backend":io_backend,"dynamic_release_waits_for_ack":False if calendar else None,
        "eager_completion":eager_completion,"release_queue_policy":queue_policy,"recover_after_miss":recover_after_miss,
        "terminal_feedback_cancelled":terminal_feedback_cancelled,
        "effective_pipeline_costs":pipeline_costs,
        "superseded_before_dispatch":len(released.superseded),
        "superseded_job_ids":[r['sample_id'] for r in released.superseded],
        "io_return_to_slot_release_ms":stats_ms([r['pipeline_released_s']-r['io_response_received_s'] for r in requests
            if 'pipeline_released_s' in r]),
        "diagnostic_release_tape":release_tape is not None,
        "dispatch_wait_ms":stats_ms([r['dispatch_s']-r['job_released_s'] for r in requests if 'dispatch_s' in r]),
        "dynamic_planning_lateness_ms":stats_ms([r['planning_lateness_s'] for r in requests if 'planning_lateness_s' in r]),
        "selected_release_interval_ms":{v:stats_ms([r['selected_release_interval_s'] for r in requests
            if r['vehicle']==v and 'selected_release_interval_s' in r]) for v in world.vehicles},
        "release_to_sample_ms":stats_ms([r['sample_s']-r['job_released_s'] for r in applied]),
        "actual_release_gap_ms":{v:stats_ms(np.diff([r['job_released_s'] for r in requests if r['vehicle']==v])) for v in world.vehicles},
        "planned_release_gap_ms":{v:stats_ms(np.diff([r['planned_release_s'] for r in requests if r['vehicle']==v])) for v in world.vehicles},
        "planned_release_to_command_ms":stats_ms([r["rpc_return_s"]-r["planned_release_s"] for r in applied]),
        "release_jitter_ms":stats_ms([r["job_released_s"]-r["planned_release_s"] for r in applied]),
        "capture_to_command_ms":stats_ms([r["rpc_return_s"]-r["sample_s"] for r in applied]),
        "cached_inputs":sum(r.get("input_source")=="sensor_cache" for r in requests),
        "response_reserve_s":reserve,"response_reserve_calibrated":response_reserve_s is not None,
        "selection_ms":stats_ms([a["analysis_ms"]/1000 for a in analyses]),
        "B_binding_count":sum(a["B_binding"] for a in analyses),"analyses":len(analyses),"segment_resumptions":resumptions,
        "per_vehicle_timing":{v:{"first_command_ms":next((1000*(r["rpc_return_s"]-origin) for r in applied if r["vehicle"]==v),None),
            "response_ms":stats_ms([r["rpc_return_s"]-r["sample_submitted_s"] for r in applied if r["vehicle"]==v]),
            "update_gap_ms":stats_ms(np.diff([r["rpc_return_s"] for r in applied if r["vehicle"]==v]))} for v in world.vehicles},
        "worker_profile":server.profile,"planning_profile":planning_profile,"planning_profile_frozen":frozen_profile is not None,
        "worker_placement":server.placement,"io_placement":io.placement if io else None,
        "lookup_used":False,"late_outputs_applied_for_exploration":not enforce,"collision_causes_early_abort":stop_on_collision and collision_policy=="global",
        "collision_retires_vehicle":stop_on_collision and collision_policy=="vehicle",
        "contact_stop":contact_stop,"mode_policy":mode_policy,
        "loop_configuration":config,"collision_policy":collision_policy,"retired_vehicles":retired,
        "active_vehicle_count_final":len(world.vehicles)-len(retired),
        "deadline_policy":deadline_policy,"deadline_failures":{v:e for v,e in retired.items() if e.get("kind")=="deadline"},
        "actual_deadline_failure_ids":[e["sample_id"] for e in retired.values() if e.get("kind")=="deadline" and e.get("actual_deadline_elapsed")],
        "deadline_contract":"state_budget_and_control_update_cap" if any(m and m.get("contract")=="relative_preview_v3" for m in policies.values()) else "nominal_deadline_intersected_with_state_budget",
        "admission_checks":len(admissions),"admission_rejections":sum(a['feasible'] is False for a in admissions),
        "initial_taskset_admitted":initial_taskset_admitted,
        "initial_taskset_forecast_passed":initial_taskset_forecast,
        "global_admission_enabled":global_admission,"frontend_protection_enabled":frontend_protection,
        "joint_adaptation":joint_adaptation,"frontend_input_wait_limit_s":MIN_PERIOD/2 if frontend_aging else None,
        "joint_decision_shadow":joint_shadow is not None and not getattr(joint_shadow,'apply_rates',False),
        "shadow_recommendations_applied":bool(getattr(joint_shadow,'installed_groups',[])),
        "joint_rate_application_enabled":bool(getattr(joint_shadow,'apply_rates',False)),
        "frontend_input_wait_ms":stats_ms([r['frontend_state_start_s']-r['sample_response_received_s'] for r in requests
            if 'frontend_state_start_s' in r and 'sample_response_received_s' in r]),
        "global_admission_decisions":{key:sum(a.get('decision')==key for a in admissions)
            for key in ('accepted','retained_previous','infeasible_retained','recovery_accepted','recovery_retained_previous',
                'joint_accepted','deadline_refresh','calendar_race_retained')},
        "all_reconfigurations_admitted":global_admission and initial_taskset_forecast is not False
            and all(a['feasible'] for a in admissions if a.get('decision')!='deadline_refresh'),
        "admission_ms":stats_ms([a['planning_ms']/1000 for a in admissions if not a.get('outside_physics')]),
        "hard_realtime_certified":False,
        "model_frozen_within_job":True,"preemption_boundary":"two complete FC layers, never mid-segment",
        "scope":"fixed-c continuous-timing ablation; empirical B and recovery urgency; not full certified framework/FALCON/FPGA",
        "response_semantics":("job arrival through command RPC return, including dispatch queue; exact physical actuation unobserved" if calendar else
            "parent sample request through command RPC return; exact physical actuation unobserved"),
        "test_read":False}
    if io is not None:
        atomic_json(directory/"io_transport.json",{"deliveries":io.delivery_records,
            "helper_stalls":io.helper_stalls,"parent_stalls":io.connection.stalls})
    atomic_json(directory/"frontend_spans.json",frontend_spans or [])
    atomic_json(directory/"admissions.json",admissions)
    atomic_json(directory/"transport_stalls.json",dict(inference_parent=server.connection.stalls,
        inference_worker=server.helper_stalls,sensor_parent=sensing.connection.stalls if sensing is not None else []))
    if sensing is not None:
        report.update(sensor_summary=sensing.summary,sensor_primary_scoring=True,
            sensor_window_covered=all(p["window_covered"] for p in vehicles.values()),
            sensor_period_s=.020,sensor_placement=sensing.placement,
            detector_edge_count=sum(d["edge"] for d in sensor_decisions),
            detector_ms=stats_ms([d["detection_ms"]/1000 for d in sensor_decisions]),
            sensor_delivery_age_ms=stats_ms([r["received_s"]-r["packet"][2] for r in sensing.records]),
            event_response_ms=stats_ms([r["rpc_return_s"]-r["sensor_capture_s"] for r in applied if "sensor_capture_s" in r]),
            control_trace_metrics=control_vehicles,
            shared_observation_rule="same20ms sensor service and detector on both sides; injection log unavailable until close")
        samples=[]
        for record in sensing.records:
            raw,hit,sampled=record["packet"]
            samples.append({"vehicle":record["row"]["vehicle"],"sim_ns":int(raw.timestamp_ns),
                "position":raw.position.tolist(),"velocity":raw.velocity.tolist(),"quaternion":raw.quaternion.tolist(),
                "captured_s":sampled,"received_s":record["received_s"],"collision_stamp":int(hit.time_stamp),
                "collision_object":str(hit.object_name),"io":record["row"]})
        atomic_json(directory/"sensor_samples.json",samples)
        atomic_json(directory/"sensor_decisions.json",sensor_decisions)
        atomic_json(directory/"control_traces.json",control_traces)
    for name,value in (("report",report),("requests",requests),("analyses",analyses),("traces",world.traces),("segments",server.segments)):
        atomic_json(directory/f"{name}.json",value)
    return report


def run(root, placement):
    root.mkdir(parents=True,exist_ok=False)
    sys.setswitchinterval(.0005)
    editor = read_background_throttle()
    if editor["background_throttle_enabled"]:
        raise RuntimeError("editor background throttling must stay disabled")
    pool = ControllerPool()
    from .state_mode_calibration import moderate_case
    case,layout = moderate_case()
    phase = case["reference_phase_s"]
    duration = StoredRoutePlanner(layout).trajectory.duration-(phase-2.1)+3.
    source_names = ("probe_seven_continuous_timing.py","probe_timing_full_route.py","framework_runtime_editor.py","framework_runtime_worker.py",
        "framework_runtime_io.py","framework_runtime_wire.py","framework_runtime_io_wire.py","framework_runtime.py",
        "framework_runtime_world.py","framework_runtime_wait.py","framework_runtime_features.py","framework_cpu_affinity.py",
        "safety_comparison_flight.py","safety_comparison_core.py")
    frozen = root/"frozen_source"
    frozen.mkdir()
    for name in source_names:
        shutil.copy2(Path(__file__).with_name(name),frozen/name)
    plan = {"vehicles":7,"net_width_m":2.,"phase_span_s":2.1,"duration_s":duration,"case_source":case,"layout":layout,
        "model":MODEL,"provenance":pool.provenance(),"cpu_layout":placement.layout,"editor":editor,
        "normal_search_periods_ms":list(PERIODS_MS),"normal_selection":"highest total J among normal complete, recovered, clock-valid, zero nominal deadline misses",
        "challenge_rule":"two previously confirmed v1.1 cases; only if no win, one predeclared v1.2 escalation; repeat winning cases once",
        "minimum_release_interval_s":MIN_PERIOD,"control_update_cap_s":CONTROL_CAP,"continuous_recovery_lookahead_s":RECOVERY_LOOKAHEAD,
        "first_release_compression":"staggered offset multiplied by exp(-lookahead*closing_rate/margin)",
        "recovery_period_rule":"max(20ms, nominal/(1+lookahead*closing_rate/margin)); margin floor25mm",
        "rule_is_safety_certificate":False,"baseline_fixed_c_ablation":True,"training_performed":False,
        "source_sha256":{n:sha256(frozen/n) for n in source_names}}
    atomic_json(root/"plan.json",plan)
    world = FrameworkWorld(layout,phase,7,2.,phase_span=2.1,task_duration=duration)
    rows=[]
    reserve=None
    nominal = {"id":"normal","critical_index":-1,"position_y_m":0.,"velocity_y_mps":0.}

    def execute(name,period,scenario,policy):
        print(f"FLIGHT {name} policy={policy} T={period:g}",flush=True)
        r = flight(world,pool,period,scenario,policy,root/name,duration,placement.layout,response_reserve_s=reserve)
        rows.append({"name":name,"directory":str(root/name),"report":r})
        atomic_json(root/"progress.json",{"rows":rows,"complete":False})
        print(f"  full={r['full_flight_completed']} success={r['success']} collisions={r['collisions']} nominal_late={len(r['nominal_late_jobs'])} J={r['total_J']} B={r['B_binding_count']}/{r['analyses']} R={r['response_ms']} error={r['error']}",flush=True)
        if r["error"] or r["cleanup_errors"]:
            raise RuntimeError("technical failure retained; no automatic retry")
        return r

    try:
        candidates=[]
        for p in PERIODS_MS:
            r=execute(f"normal_static_T{p:g}",p,nominal,"static")
            candidates.append({"period_ms":p,"report":r})
        eligible=[c for c in candidates if c["report"]["success"] and c["report"]["nominal_strict_timing_passed"]]
        if not eligible:
            atomic_json(root/"summary.json",{"complete":True,"status":"NO_OBSERVED_FEASIBLE_STATIC_BASELINE","rows":rows})
            return
        selected=max(eligible,key=lambda c:c["report"]["total_J"])
        reserve=calibrated_response_reserve(selected["report"])
        p=selected["period_ms"]
        atomic_json(root/"selection.json",{"selected":selected,"frozen_before_challenges":True,"not_global_FALCON_optimum":True,"candidate_count":len(candidates)})
        for policy in ("edf_only","adaptive"):
            execute(f"normal_{policy}",p,nominal,policy)
        pairs=[]
        for speed in (1.1,1.2):
            batch=[]
            for index,sign in ((0,1),(6,-1)):
                scenario={"id":f"i{index}_sign{sign:+d}_v{speed:g}","critical_index":index,"position_y_m":sign*.05,"velocity_y_mps":sign*speed}
                pair={"scenario":scenario,"reports":{},"directories":{}}
                for policy in (POLICIES if index==0 else tuple(reversed(POLICIES))):
                    name=f"{scenario['id']}_r1_{policy}"
                    pair["reports"][policy]=execute(name,p,scenario,policy)
                    pair["directories"][policy]=str(root/name)
                initials=[read_json(Path(pair["directories"][q])/"initial_state.json") for q in POLICIES]
                pair["initial_states_match"]=all(matched_initials(initials[0],x) for x in initials[1:])
                pair["observed_safety_win"]=pair["initial_states_match"] and pair["reports"]["static"]["collisions"]>0 and pair["reports"]["adaptive"]["success"]
                pairs.append(pair)
                batch.append(pair)
                atomic_json(root/"pairs.json",pairs)
            wins=[pair for pair in batch if pair["observed_safety_win"]]
            if wins:
                for pair in wins:
                    repeat={"scenario":pair["scenario"],"reports":{},"directories":{},"repeat":2}
                    for policy in reversed(POLICIES):
                        name=f"{pair['scenario']['id']}_r2_{policy}"
                        repeat["reports"][policy]=execute(name,p,pair["scenario"],policy)
                        repeat["directories"][policy]=str(root/name)
                    initials=[read_json(Path(repeat["directories"][q])/"initial_state.json") for q in POLICIES]
                    repeat["initial_states_match"]=all(matched_initials(initials[0],x) for x in initials[1:])
                    repeat["observed_safety_win"]=repeat["initial_states_match"] and repeat["reports"]["static"]["collisions"]>0 and repeat["reports"]["adaptive"]["success"]
                    pairs.append(repeat)
                break
        atomic_json(root/"summary.json",{"complete":True,"rows":rows,"pairs":pairs,
            "observed_safety_wins":sum(x["observed_safety_win"] for x in pairs),"selected_period_ms":p,
            "development_only":True,"independent_confirmation":False,"FPGA_verified":False})
    finally:
        atomic_json(root/"cleanup_report.json",world.close())
    print(f"COMPLETE {root}",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    root=args.output or ROOT/"controller_architecture_study/results"/("seven_continuous_timing_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    placement=CpuPlacement()
    try:
        with placement:
            run(root,placement)
    finally:
        if root.exists():
            atomic_json(root/"cpu_placement.json",placement.report)


if __name__=="__main__":
    main()
