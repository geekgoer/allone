"""CPU implementation of the framework contracts; no physical safety certificate."""

import json
import math
import time

import numpy as np

from .dynamic_probe_common import DATA, PLAIN
from .decision_dataset import sha256
from .eval_offline import load_checkpoint, load_normalization
from .safety_comparison_core import BODY_RADIUS, CLEARANCE, LayerJob, safe_horizon


MODELS = tuple(f"plain_mlp_h{d}_w{w}_seed0" for w in (512, 1024) for d in (9, 21, 33))
CONTRACT = "framework_237d_segment_edf_v1"
COLLISION_CONTRACT = "simGetCollisionInfo_changed_nonzero_stamp_v1"
TIMING_CONTRACT = "parent_visible_state_and_direct_collision_v3"
CONTROL_LIMIT = .080
H_EDGES = (0., .25, .50)
HDOT_EDGES = (-2.5, -.8, -.2, .2, .8, 2.5)


class ControllerPool:
    """Six existing FALCON-sized models, immutable two-FC segments."""
    def __init__(self,model_names=None):
        import torch
        torch.set_num_threads(1)
        if not torch.set_flush_denormal(True):
            raise RuntimeError("CPU flush-denormal mode unavailable")
        self.torch, self.norm = torch, load_normalization(DATA)
        names=MODELS if model_names is None else tuple(model_names)
        if not names or len(set(names))!=len(names) or any(not m.startswith('plain_mlp_') for m in names):
            raise ValueError('distinct existing plain237D controller names required')
        self.models = {m: load_checkpoint(PLAIN/m/"best_model.pt", device="cpu") for m in names}
        self.layers = {m: tuple(net.children()) for m,net in self.models.items()}
        self.chunks = {m: tuple(torch.jit.script(torch.nn.Sequential(*layers[i:i+4]))
            for i in range(0,len(layers),4)) for m,layers in self.layers.items()}
        self.profile = {}
        for name in self.models:
            measured = []
            for k in range(25):
                job, segments = self.start(name,self.norm["x_mean"]), []
                while not job.done:
                    began = time.perf_counter()
                    job.step()
                    segments.append(time.perf_counter()-began)
                if k >= 5:
                    measured.append(segments)
            a = np.asarray(measured)
            # Empirical estimates with explicit margins, NOT certified WCET.
            self.profile[name] = {"segment_estimates_s": (1.5*a.max(axis=0)+.00005).tolist(),
                "measured_compute_p95_s": float(np.percentile(a.sum(axis=1),95)),
                "measured_segment_max_s": float(a.max()), "samples":len(a), "is_WCET":False}

    def start(self, name, feature):
        return LayerJob(self,name,feature)

    def predict(self, name, feature):
        job = self.start(name,feature)
        while not job.done:
            job.step()
        return job.output

    def warm_worker(self):
        self.torch.set_num_threads(1)
        self.torch.set_flush_denormal(True)
        for name in self.models:
            for _ in range(3):
                self.predict(name,self.norm["x_mean"])

    def provenance(self):
        return {"contract":CONTRACT,"device":"cpu","torch":str(self.torch.__version__),
            "normalization_sha256":sha256(DATA/"normalization.json"),
            "models":{m:sha256(PLAIN/m/"best_model.pt") for m in self.models},
            "segment":"up to two FC layers with their original ReLUs; frozen model per job"}


def fixed_planning_profile(measured, frozen):
    """Use the same empirical segment costs in paired trials, never as WCET."""
    if set(frozen)!=set(measured):
        raise ValueError('frozen planning profile must contain the same controller pool')
    for name,row in frozen.items():
        values=row.get('segment_estimates_s',[])
        if (len(values)!=len(measured[name]['segment_estimates_s'])
                or not all(math.isfinite(x) and x>0. for x in values)):
            raise ValueError('frozen planning costs must match finite positive model segments')
    return {name:{**row,'segment_estimates_s':list(row['segment_estimates_s'])} for name,row in frozen.items()}


def route_risk(state, reference, net_width):
    """Two-dimensional moving-route-tube feature, distinct from mesh clearance."""
    from .state_mode_probe import normal_frame
    if np.linalg.norm(reference.velocity[:2]) >= .5:
        normal, derivative = normal_frame(reference)
    else:
        # Terminal hover has no velocity tangent; retain the reference yaw frame.
        w,x,y,z = reference.orientation
        yaw = math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
        normal, derivative = np.array([-math.sin(yaw),math.cos(yaw),0.]),np.zeros(3)
    e, dv = state.position-reference.position, state.velocity-reference.velocity
    lateral = float(e@normal)
    lateral_rate = float(dv@normal+e@derivative)
    margins = [net_width/2-BODY_RADIUS-CLEARANCE-abs(lateral),
               2.-BODY_RADIUS-CLEARANCE-abs(e[2])]
    rates = [-np.sign(lateral)*lateral_rate if lateral else -abs(lateral_rate),
             -np.sign(e[2])*dv[2] if e[2] else -abs(dv[2])]
    active = int(np.argmin(margins))
    return {"h":float(margins[active]),"hdot":float(rates[active]),
            "boundary":"lateral" if active == 0 else "vertical",
            "turn_pressure":float(np.sign(lateral)*(reference.acceleration@normal))}


def state_bin(risk):
    values = (risk["h"],risk["hdot"])
    if not np.isfinite(values).all():
        return None
    indices = []
    for value,edges in zip(values,(H_EDGES,HDOT_EDGES)):
        if not edges[0] <= value <= edges[-1]:
            return None
        indices.append(min(len(edges)-2,int(np.searchsorted(edges,value,side="right")-1)))
    return f"h{indices[0]}_v{indices[1]}"


def context_grid_key(risk, steps):
    return ":".join(str(math.floor(risk[k]/steps[k]+.5)) for k in ("h","hdot","turn_pressure"))


def input_age_fits(row, now, remaining_command_s=0.):
    limit = row.get("input_age_limit_s")
    return limit is None or now+remaining_command_s <= row["sample_s"]+limit


def release_deadline(row, released_at):
    age = row.get("input_age_limit_s")
    deadline = row["deadline_s"]
    if row.get("fixed_period_slot"):
        deadline = min(deadline, released_at+row["requested_period_s"])
    return deadline if age is None else min(deadline,released_at+age)


class ModeMap:
    def __init__(self, path, provenance, net_width):
        self.data = json.loads(path.read_text(encoding="utf-8"))
        if self.data.get("collision_observation") != COLLISION_CONTRACT:
            raise ValueError("mode map used an unverified collision source; recollect with the direct collision API")
        if self.data.get("safety_predicate") != "gate_task":
            raise ValueError("mode map must use the same gate-task safety predicate as online acceptance")
        if (self.data["provenance"] != provenance or self.data["net_width_m"] != net_width
                or self.data["contract"] != CONTRACT or not self.data["complete"]):
            raise ValueError("mode map artifact/domain mismatch or incomplete calibration")
        if self.data.get("context_feature") not in (None,"turn_pressure"):
            raise ValueError("unsupported mode-map context feature")
        if self.data.get("context_feature") and any("turn_pressure_range" not in c
                for c in self.data.get("local_cells",[])):
            raise ValueError("context-conditioned cells require explicit turn-pressure bounds")
        if "context_grid_steps" in self.data and (self.data.get("context_feature")!="turn_pressure" or
                any(not math.isfinite(self.data["context_grid_steps"].get(k,0.)) or
                    self.data["context_grid_steps"].get(k,0.)<=0. for k in ("h","hdot","turn_pressure"))):
            raise ValueError("finite positive context grid steps required")
        self.sha256 = sha256(path)

    def select(self, risk, profile):
        key = state_bin(risk)
        coarse_key = key
        contextual = self.data.get("context_feature") == "turn_pressure"
        if contextual and not math.isfinite(risk.get("turn_pressure",float("nan"))):
            return None,{"state_bin":key,"status":"missing_or_nonfinite_context","eligible":[]}
        # A contextual map must not fall back to an unconditioned positive bin.
        row = {} if contextual else self.data["bins"].get(key,{})
        grid = None
        if "context_grid_steps" in self.data and np.isfinite([risk["h"],risk["hdot"],risk["turn_pressure"]]).all():
            grid_key = context_grid_key(risk,self.data["context_grid_steps"])
            grid = self.data.get("context_grid",{}).get(grid_key)
            if grid is not None:
                row,key = grid,"grid:"+grid_key
        veto = grid is not None and (not grid.get("modes") or "first_failed_wait_s" in grid)
        for cell in (() if veto else self.data.get("local_cells",[])):
            if grid is not None and cell.get("evidence_kind")!="direct_recovery_calibration":
                continue
            if (cell["h_range"][0] <= risk["h"] <= cell["h_range"][1]
                    and cell["hdot_range"][0] <= risk["hdot"] <= cell["hdot_range"][1]
                    and cell.get("state_bin",coarse_key)==coarse_key
                    and (not contextual or cell["turn_pressure_range"][0] <= risk["turn_pressure"]
                        <= cell["turn_pressure_range"][1])):
                row,key = cell,cell["id"]
                break
        eligible = [m for m,s in row.get("modes",{}).items() if s["status"] == "empirical_admissible"]
        if not eligible:
            return None,{"state_bin":key,"status":"unknown_or_no_admissible_mode","eligible":[]}
        model = min(eligible,key=lambda m:sum(profile[m]["segment_estimates_s"]))
        limit = float(row["modes"][model].get("delay_limit_s",self.data["analysis_to_next_command_limit_s"]))
        if not math.isfinite(limit) or not 0. < limit <= self.data["analysis_to_next_command_limit_s"]:
            raise ValueError("invalid empirically validated mode wait limit")
        update = float(row["modes"][model].get("future_update_limit_s",CONTROL_LIMIT))
        if not math.isfinite(update) or not 0. < update <= CONTROL_LIMIT:
            raise ValueError("invalid empirically validated future-update limit")
        age = row["modes"][model].get("input_age_limit_s")
        if age is not None and (not math.isfinite(age) or not 0. < age <= CONTROL_LIMIT):
            raise ValueError("invalid input-age limit")
        return model,{"state_bin":key,"status":"empirical_admissible","eligible":eligible,
            "selected":model,"delay_limit_s":limit,"future_update_limit_s":update,"input_age_limit_s":age}


def analyze_budget(state, held_control, sample_s, actuation_s, gates, net_width,
                   maximum_update_s=CONTROL_LIMIT, *, recovery_assumptions=None, held_analyzer=None):
    """Analyze a fresh post-command state; never restart an old promise's clock."""
    control = np.asarray(held_control,dtype=float)
    if control.shape != (4,) or not np.isfinite(control).all() or sample_s < actuation_s:
        raise ValueError("finite actually sent control and post-command observation required")
    if held_analyzer is not None:
        from .held_control_safety import safe_deadline
        result = held_analyzer.analyze(state, control)
        timing = safe_deadline(sample_s, sample_s, actuation_s, result, maximum_update_s)
        return {**result, **timing, 'valid':timing['feasible'], 'analysis_s':sample_s, 'actuation_proxy_s':actuation_s,
                'held_control':control.tolist(), 'constraints':[],
                'actuation_timestamp_certified':False, 'is_safety_certificate':False}
    budget, constraints = safe_horizon(state.position,state.velocity,gates,net_width)
    recovery=None
    if recovery_assumptions is not None:
        from dataclasses import asdict
        from .recovery_budget import recovery_horizon
        horizon,details=recovery_horizon(state.position,state.velocity,gates,net_width,recovery_assumptions)
        recovery=dict(budget_s=horizon,geometric_budget_s=budget,assumptions=asdict(recovery_assumptions),
            constraints=details,binding=horizon<budget,
            outside_local_recovery_set=any(r['status']=='outside_local_recovery_set' for r in details),
            is_safety_certificate=False)
        budget=min(budget,horizon)
    deadline = min(sample_s+budget,actuation_s+maximum_update_s)
    result={"analysis_s":sample_s,"actuation_proxy_s":actuation_s,"held_control":control.tolist(),
        "budget_s":budget,"deadline_s":deadline,"valid":bool(deadline > sample_s),
        "constraints":constraints,"method":"gate-slab/rate prototype v2; not control-aware dynamics",
        "uses_control_in_bound":False,"is_safety_certificate":False}
    if recovery is not None:
        result.update(recovery=recovery,method="gate-approach recovery assumptions v1; not certified UAV dynamics")
    return result


def remaining_stage_cost(costs, key, elapsed):
    """Escalate observed nominal overruns to the SAME development tail estimate.

    Never restart the operation clock or its physical deadline. An exhausted
    tail remains negative so admission can explicitly reject that cost context.
    """
    nominal = costs[key]
    tail = costs.get("tail_costs",{}).get(key,nominal)
    escalated = elapsed >= nominal and tail > nominal
    return (tail if escalated else nominal)-elapsed,escalated


def simulate_pipeline(now, future, pending, profile, costs, *, policy="edf", inflight_limit=3,missed_deadline_ids=()):
    """Finite next-job forecast: serialized frontend, separate segmented backend.

    Pending compute is charged its FULL cost plus one segment blocking guard.
    This is conservative about progress, but empirical timings are not WCET.
    Future periods beyond the next committed job are intentionally not certified.
    """
    jobs = {}
    for prospective,row in [*[(True,r) for r in future],*[(False,r) for r in pending]]:
        name = row["sample_id"]
        if name in jobs:
            raise ValueError("duplicate pipeline job")
        jobs[name] = {**row,"stage":row.get("stage","input"),"cursor":0,
            "segments":profile[row["model"]]["segment_estimates_s"],
            "compute_release_s":now+row.get("ingress_remaining_s",costs.get("ingress_s",0.)),
            "output_release_s":now}
        if prospective or row.get('fixed_period_slot'):
            jobs[name]["deadline_s"] = release_deadline(row,row["planned_release_s"])
    if not jobs:
        return True,{},0
    missed=set(missed_deadline_ids)
    if any(name not in jobs or jobs[name]['deadline_s']>now for name in missed):
        raise ValueError('only already-missed deadlines may be excluded from future feasibility')
    periodic_sampling = costs.get("periodic_sampling",False)
    for row in jobs.values():
        if not math.isfinite(row["deadline_s"]) or not math.isfinite(row["planned_release_s"]):
            raise ValueError("finite scheduling times required")
        following=min((r for r in jobs.values() if r is not row
            and r.get('vehicle')==row.get('vehicle') and r['planned_release_s']>row['planned_release_s']),
            key=lambda r:r['planned_release_s'],default=None)
        release=following['planned_release_s'] if following else row['planned_release_s']+row.get('requested_period_s',0.)
        row['maintenance_due_s']=release
        if periodic_sampling and 'completion_frontend_done_s' not in row:
            row['maintenance_due_s']+= (following or row).get('requested_period_s',0.)
    def key(name):
        row = jobs[name]
        if costs.get('frontend_protection') and row['stage']=='analysis':
            return (row['maintenance_due_s'],row['planned_release_s'],name)
        return (row["priority"] if policy == "static" else row["deadline_s"],row["planned_release_s"],name)
    def frontend_deadline(name):
        row=jobs[name]
        due=row['maintenance_due_s'] if row['stage']=='analysis' else row['deadline_s']
        return due,row['planned_release_s'],row['priority'],name
    def analysis_cost(name):
        if 'analysis_remaining_s' in jobs[name]:return jobs[name]['analysis_remaining_s']
        return costs['output_s']+(costs.get('planner_reserve_s',0.)
            if costs.get('frontend_protection') and costs.get('planning_enabled',True) and not jobs[name].get('next_plan_created') else 0.)
    def dispatch_allowed(name):
        row=jobs[name]
        if not row.get('serial_vehicle'):return True
        same=[n for n,r in jobs.items() if r.get('vehicle')==row['vehicle'] and n!=name and r['stage']!='done'
            and (not r.get('nonblocking_maintenance') or
                (periodic_sampling and 'completion_frontend_done_s' not in r))]
        waiting_stages = ('input','input_ready','sampling_busy') if periodic_sampling else ('input',)
        return not any(jobs[n]['stage'] not in waiting_stages or
            (jobs[n]['planned_release_s'],n)<(row['planned_release_s'],name) for n in same)
    clock, frontend, backend = now,None,None
    completed = {r['sample_id']:r['command_completed_s'] for r in pending if 'command_completed_s' in r}
    frontend_available = max(now,costs.get("frontend_reserved_until_s",now))
    management_period=costs.get('management_period_s',0.)
    management_budget=costs.get('management_budget_s',0.)
    management_next=costs.get('management_next_s',now+management_period)
    acknowledged = {r["sample_id"] for r in pending if "command_completed_s" in r}
    io_s, handoff_s = costs.get("io_s",0.),costs.get("handoff_s",0.)
    command_s = costs.get("command_s",io_s)
    sample_io_s = costs.get("sample_io_s",0.)
    release_dispatch = costs.get('period_release_dispatch_s',0.) if periodic_sampling else 0.
    previous_completion = False
    io_slots, transfers, captures, command_returns = costs.get("io_slots",len(jobs)),[],[],[]
    for name,row in jobs.items():
        if row['stage']=='input_ready':row['input_ready_s']=row.get('sample_response_received_s',now)
        if row["stage"] == "io":
            remaining = row.get("io_remaining_s",io_s)
            if remaining <= 0.:
                return False,{},0
            row["stage"] = "io_busy"
            transfers.append((name,now+remaining))
            if "command_completed_s" in row:
                completed[name] = row["command_completed_s"]
            else:
                command_remaining = row.get("command_remaining_s",min(remaining,command_s))
                if command_remaining <= 0.:
                    return False,{},0
                command_returns.append((name,now+command_remaining))
        elif row["stage"] == "analysis" and io_s:
            completed[name] = row.get("command_completed_s",now)
            if costs.get('eager_completion'):row['nonblocking_maintenance']=True
        elif row["stage"] == "sampling":
            remaining = row["sample_io_remaining_s"]
            if remaining <= 0.:
                return False,{},0
            row["stage"] = "sampling_busy"
            captures.append((name,now+remaining))
    blocking = max((max(r["segments"]) for r in jobs.values() if r["stage"] == "compute"),default=0.)
    backend_available = now+blocking
    limit = 20+8*sum(4+len(r["segments"]) for r in jobs.values())
    for events in range(limit):
        if frontend is not None and frontend[2] <= clock+1e-12:
            name,stage,end = frontend
            jobs[name]["stage"] = "compute" if stage == "input" else "io_busy" if stage == "handoff" else "done"
            if stage == "input":
                jobs[name]["compute_release_s"] = end+costs.get("ingress_s",0.)
            if stage == "handoff":
                transfers.append((name,end+io_s))
                command_returns.append((name,end+command_s))
            if stage == "output":
                if not io_s and name not in acknowledged:
                    completed[name] = end
            frontend = None
        unfinished_transfers = []
        for name,end in transfers:
            if end <= clock+1e-12:
                jobs[name]["stage"] = "analysis"
                if costs.get('eager_completion'):jobs[name]['nonblocking_maintenance']=True
                if io_s:
                    completed.setdefault(name,end)
            else:
                unfinished_transfers.append((name,end))
        transfers = unfinished_transfers
        waiting_returns = []
        for name,end in command_returns:
            if end <= clock+1e-12:
                completed[name] = end
            else:
                waiting_returns.append((name,end))
        command_returns = waiting_returns
        waiting_captures = []
        for name,end in captures:
            if end <= clock+1e-12:
                jobs[name]["stage"] = "input_ready"
                jobs[name]['input_ready_s']=end
            else:
                waiting_captures.append((name,end))
        captures = waiting_captures
        if backend is not None and backend[1] <= clock+1e-12:
            name,end = backend
            jobs[name]["cursor"] += 1
            jobs[name]["stage"] = "output" if jobs[name]["cursor"] == len(jobs[name]["segments"]) else "compute"
            if jobs[name]["stage"] == "output":
                jobs[name]["output_release_s"] = end+costs.get("egress_s",0.)
            backend = None
        if any(end+costs.get("completion_guard_s",0.) > jobs[name]["deadline_s"]+1e-10
                    for name,end in completed.items() if name not in acknowledged and name not in missed):
            return False,completed,events+1
        if all(r["stage"] == "done" for r in jobs.values()):
            return True,completed,events+1
        if any(n not in completed and n not in missed and r["stage"] != "done" and r["deadline_s"] < clock-1e-10 for n,r in jobs.items()):
            return False,completed,events+1
        ready_analysis = [n for n,r in jobs.items() if r["stage"] == "analysis"]
        to_send = [n for n,r in jobs.items() if r["stage"] == "output" and r["output_release_s"] <= clock+1e-12]
        ready_input = any(r["stage"]=="input_ready" for r in jobs.values())
        cleanup_cost=max((analysis_cost(n) for n in ready_analysis),default=costs['output_s'])
        analysis_blocks_output = any(clock+cleanup_cost+handoff_s+command_s+costs.get("completion_guard_s",0.) > r["deadline_s"]
            for r in jobs.values() if r["stage"] in ("compute","backend_busy","output") and r['sample_id'] not in missed)
        protected_frontend=costs.get('frontend_protection',False)
        feedback_first = not protected_frontend and policy!="static" and io_s and ready_analysis and not ready_input and not analysis_blocks_output
        # Reserve future CPU management opportunities as well as the present
        # review. The XPU may continue while this host CPU budget is occupied.
        if management_budget and frontend is None and clock>=max(frontend_available,management_next)-1e-12:
            frontend_available=clock+management_budget
            management_next=clock+management_period
        # Capture dispatch also requires the serialized frontend. Match the
        # live feedback-first policy instead of capturing through CPU work.
        if periodic_sampling and frontend is None and clock >= frontend_available-1e-12:
            busy = {jobs[n]['vehicle'] for n,_ in captures}
            for name in sorted(jobs,key=key):
                row = jobs[name]
                if row['stage']=='input' and row['planned_release_s']+release_dispatch<=clock+1e-12 and row['vehicle'] not in busy:
                    row['stage']='sampling_busy'
                    captures.append((name,clock+sample_io_s))
                    busy.add(row['vehicle'])
        elif sample_io_s and frontend is None and not ready_input and not feedback_first and not to_send and clock >= frontend_available-1e-12:
            inflight = sum(r["stage"] not in ("input","done") and not r.get('nonblocking_maintenance') for r in jobs.values())
            waiting = sorted((n for n,r in jobs.items() if r["stage"] == "input"
                and r["planned_release_s"] <= clock+1e-12 and dispatch_allowed(n)),key=key)
            for name in waiting[:max(0,min(io_slots-len(transfers)-len(captures),inflight_limit-inflight))]:
                jobs[name]["stage"] = "sampling_busy"
                captures.append((name,clock+sample_io_s))
        if frontend is None and clock >= frontend_available-1e-12:
            waiting_stages = ("input","frontend_input","done","input_ready","sampling_busy") if periodic_sampling else ("input","frontend_input","done")
            inflight = sum(r["stage"] not in waiting_stages and not r.get('nonblocking_maintenance') for r in jobs.values())
            input_candidates=[n for n,r in jobs.items() if r['stage']==('input_ready' if sample_io_s else 'input')
                and r['planned_release_s']<=clock+1e-12 and
                ((sample_io_s and not periodic_sampling) or (inflight<inflight_limit and dispatch_allowed(n)))]
            aged_inputs=[n for n in input_candidates if costs.get('input_wait_limit_s') is not None
                and clock-jobs[n].get('input_ready_s',jobs[n]['planned_release_s'])>=costs['input_wait_limit_s']]
            maintenance_first=protected_frontend and ready_analysis and not analysis_blocks_output and (
                not input_candidates or (not aged_inputs and key(min(ready_analysis,key=key))<=key(min(input_candidates,key=key))))
            if to_send and len(transfers)+(0 if periodic_sampling else len(captures)) < io_slots:
                name = min(to_send,key=key)
                jobs[name]["stage"] = "frontend_handoff"
                frontend = (name,"handoff",clock+handoff_s)
            elif periodic_sampling:
                candidates=input_candidates+([] if previous_completion and input_candidates else ready_analysis)
                if candidates:
                    name=min(candidates,key=frontend_deadline)
                    previous_completion=jobs[name]['stage']=='analysis'
                    jobs[name]['stage']='frontend_output' if previous_completion else 'frontend_input'
                    frontend=(name,'output',clock+analysis_cost(name)) if previous_completion else (
                        name,'input',clock+costs['input_s'])
            elif ready_analysis and (((not io_s or feedback_first) and not protected_frontend) or maintenance_first):
                name = min(ready_analysis,key=key)
                jobs[name]["stage"] = "frontend_output"
                frontend = (name,"output",clock+analysis_cost(name))
            elif sample_io_s or inflight < inflight_limit:
                # Do not start a non-preemptive input RPC/feature stage if it
                # would block an imminent urgent command beyond its deadline.
                delivery_cost = handoff_s+command_s+costs.get("completion_guard_s",0.)+(costs["output_s"] if not io_s else 0.)
                blocking_output = not periodic_sampling and any(clock+costs["input_s"]+delivery_cost > r["deadline_s"]
                    for r in jobs.values() if r["stage"] in ("compute","backend_busy","output") and r['sample_id'] not in missed)
                ready = [] if blocking_output else [n for n,r in jobs.items()
                    if r["stage"] == ("input_ready" if sample_io_s else "input") and r["planned_release_s"] <= clock+1e-12
                    and ((sample_io_s and not periodic_sampling) or (inflight<inflight_limit and dispatch_allowed(n)))]
                if ready:
                    aged=[n for n in aged_inputs if n in ready]
                    name = min(aged,key=lambda n:jobs[n].get('input_ready_s',jobs[n]['planned_release_s'])) if aged else min(ready,key=key)
                    jobs[name]["stage"] = "frontend_input"
                    frontend = (name,"input",clock+costs["input_s"])
            if not periodic_sampling and frontend is None and ready_analysis and not analysis_blocks_output and (not protected_frontend or not ready_input):
                # Acknowledged jobs are complete. Their next-job planning
                # cannot block the input stage of an already committed job.
                name = min(ready_analysis,key=key)
                jobs[name]["stage"] = "frontend_output"
                frontend = (name,"output",clock+analysis_cost(name))
        if backend is None and clock >= backend_available-1e-12:
            ready = [n for n,r in jobs.items() if r["stage"] == "compute" and r["compute_release_s"] <= clock+1e-12]
            if ready:
                name = min(ready,key=key)
                row = jobs[name]
                row["stage"] = "backend_busy"
                backend = (name,clock+row["segments"][row["cursor"]])
        times = [v for v in (frontend[2] if frontend else None,backend[1] if backend else None,
                 backend_available if backend_available > clock+1e-12 else None,
                 frontend_available if frontend_available > clock+1e-12 else None) if v is not None]
        times += [end for _,end in transfers]
        times += [end for _,end in command_returns]
        times += [end for _,end in captures]
        if backend is None:
            times += [r["compute_release_s"] for r in jobs.values() if r["stage"] == "compute" and r["compute_release_s"] > clock+1e-12]
        if frontend is None:
            if management_budget and management_next>clock+1e-12 and any(
                    r['stage'] in ('input_ready','output','analysis') or
                    (r['stage']=='input' and r['planned_release_s']<=clock) for r in jobs.values()):
                times.append(management_next)
            times += [r["planned_release_s"]+release_dispatch for r in jobs.values()
                if r["stage"] == "input" and r["planned_release_s"]+release_dispatch > clock+1e-12]
            times += [r["output_release_s"] for r in jobs.values() if r["stage"] == "output" and r["output_release_s"] > clock+1e-12]
        if not times:
            return False,completed,events+1
        clock = min(times)
    raise RuntimeError("bounded pipeline simulation event limit exceeded")


def bounded_plan(now, future, pending, profile, costs, *, maximum_candidates=5, inflight_limit=3):
    """Commit only a feasible forecast; never change deadlines or running modes."""
    invalid = []
    for row in pending:
        keys = (("sample_io_remaining_s",) if row.get("stage") == "sampling" else
            ("io_remaining_s",) if "command_completed_s" in row else ("io_remaining_s","command_remaining_s"))
        for key in keys:
            if key in row and row[key] <= 0.:
                invalid.append({"sample_id":row["sample_id"],"stage":row["stage"],key:row[key]})
        if row.get("ingress_remaining_s",0.) < 0.:
            invalid.append({"sample_id":row["sample_id"],"stage":row["stage"],
                "ingress_remaining_s":row["ingress_remaining_s"]})
    if invalid:
        return None,{"feasible":False,"checked":0,"candidate_limit":maximum_candidates,"events":0,
            "invalid_cost_contexts":invalid,
            "reason":"pending operation has no positive residual under the supplied empirical cost model; not a proof of physical infeasibility"}
    if not future:
        feasible,finish,events = simulate_pipeline(now,[],pending,profile,costs,inflight_limit=inflight_limit)
        return ({} if feasible else None),{"feasible":feasible,"checked":1,"events":events,"finish":finish}
    ordered = sorted(future.values(),key=lambda r:(r["deadline_s"],r["sample_id"]))
    full_frontend = any(r.get("input_age_limit_s") is not None for r in future.values())
    offsets,packed_offsets,paired_offsets = {},{},{}
    offset,packed_offset,paired_offset = 0.,0.,0.
    for index,row in enumerate(ordered):
        offsets[row["vehicle"]] = offset
        packed_offsets[row["vehicle"]] = packed_offset
        paired_offsets[row["vehicle"]] = paired_offset
        offset += max(costs["input_s"]+costs.get("handoff_s",0.)+(costs["output_s"] if full_frontend else 0.),
            sum(profile[row["model"]]["segment_estimates_s"]))
        packed_offset += max(costs["input_s"]+costs.get("handoff_s",0.),
            sum(profile[row["model"]]["segment_estimates_s"]))
        if index:
            pace = max(costs["input_s"]+costs.get("handoff_s",0.),sum(profile[row["model"]]["segment_estimates_s"]))
            quantum = costs.get("handoff_s",0.)
            paired_offset += math.ceil(pace/quantum)*quantum if quantum>0. else pace
    checked,events = 0,0
    best,best_info = None,None
    limit = min(5,maximum_candidates)

    def check(pattern, release):
        nonlocal checked,events,best,best_info
        spacing = paired_offsets if pattern=="paired_ingress" else packed_offsets if pattern=="staggered" else offsets
        proposal = {v:{**row,"planned_release_s":release+(spacing[v] if pattern!="common" else 0.)}
            for v,row in future.items()}
        feasible,finish,count = simulate_pipeline(now,list(proposal.values()),pending,profile,costs,inflight_limit=inflight_limit)
        checked += 1
        events += count
        if feasible:
            best,best_info = proposal,{"feasible":True,"finish":finish,"release_pattern":pattern,"release_base_s":release,
                "release_offsets_s":{v:r["planned_release_s"]-release for v,r in proposal.items()}}
        return feasible

    # Reuse one feasible forecast's slack for a SINGLE checked right shift.
    # Repeated bisection consumed frontend time that the control jobs need.
    # A full-frontend spacing can miss a tight group of physical deadlines;
    # common release can instead exhaust input freshness. Try the original
    # compact spacing too, but accept it only under the same complete forecast.
    # Prime two independent state RPCs together, then pace the rest in handoff
    # quanta. This is only a release candidate; all actual costs are still
    # checked by the same complete forecast, including feedback and input age.
    patterns = (("pipeline_staggered","paired_ingress","staggered","common") if len(future)>1
        else ("pipeline_staggered",)) if full_frontend else ("staggered","common")
    for pattern in patterns:
        if checked >= limit:
            break
        earliest = max(now,costs.get("frontend_reserved_until_s",now))
        if not check(pattern,earliest):
            continue
        slack = min(r["deadline_s"]-best_info["finish"][r["sample_id"]]
            -costs.get("completion_guard_s",0.)-costs["dispatch_guard_s"] for r in ordered)
        if slack > 1e-9 and checked < limit:
            check(pattern,earliest+slack)
        break
    if best is not None:
        return best,{**best_info,"checked":checked,"candidate_limit":limit,"events":events,
            "search":"bounded templates plus one forecast-slack right-shift check; not an exact latest-release solver"}
    return None,{"feasible":False,"checked":checked,"candidate_limit":maximum_candidates,
        "events":events,"reason":"bounded policy found no feasible schedule; not proof of theoretical infeasibility"}


def account_planning_delay(proposal, info, pending, planned_at, finished_at, costs):
    """Shift a forecast conservatively if planning outlasted its reservation.

    All unacknowledged completions and future releases move by the whole lag;
    no deadline moves. Reject if the original slack cannot pay for that lag.
    """
    lag = max(0.,finished_at-planned_at)
    if not lag:
        return proposal
    rows = {r["sample_id"]:r for r in [*proposal.values(),*pending]}
    acknowledged = {r["sample_id"] for r in pending if "command_completed_s" in r}
    finish = {name:end+(0. if name in acknowledged else lag) for name,end in info["finish"].items()}
    if any(end+(0. if name in acknowledged else costs.get("completion_guard_s",0.))
           > rows[name]["deadline_s"] for name,end in finish.items()):
        return None
    info.update(planning_lag_s=lag,finish=finish)
    if "release_base_s" in info:
        info["release_base_s"] += lag
    return {v:{**r,"planned_release_s":r["planned_release_s"]+lag} for v,r in proposal.items()}
