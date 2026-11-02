"""Separate-process joint advice, with opt-in atomic rate application.

One nonblocking latest-value mailbox bounds backlog. Ordinary changes merge
for 50ms; a newly exhausted response margin bypasses that merge interval.
Default is shadow-only; application requires fresh whole-group review.
"""

from collections import Counter
import ctypes
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np

from .decision_dataset import atomic_json
from .joint_contract_optimizer import optimize
from .joint_contract_probe import load_priors
from .joint_contract_recovery import prepare_recovery
from .joint_commit_guard import review as review_commit,rebase_slots,repair_current_rates


def tag_rate_change(row,previous,generation):
    # Ignore sub-microsecond subtraction noise in absolute release clocks.
    period=abs(row['requested_period_s']-previous['requested_period_s'])>1e-6
    release=abs(row['planned_release_s']-previous['planned_release_s'])>1e-6
    row.update(joint_commit_generation=generation,joint_commit_s=time.perf_counter(),
        joint_rate_changed=period or release,joint_period_changed=period,joint_release_changed=release,
        joint_deadline_changed=abs(row['deadline_s']-previous['deadline_s'])>1e-6)


def effect_options_for(snapshot,profile,costs,ready,ranker,features):
    """Bounded action grid; delay includes WAIT from observed state to action.

    The cadence is a proposed future nominal cadence, not a promise of equal
    actuation gaps. Shared-queue forecast still checks the whole task group.
    """
    service=sum(costs.get(k,0.) for k in ('sample_io_s','input_s','ingress_s','egress_s','handoff_s','command_s'))
    queued=sum(sum(profile[r['model']]['segment_estimates_s']) for r in snapshot['pending']
        if 'command_completed_s' not in r)
    options={}
    for v,row in snapshot['future'].items():
        actions=[];state=snapshot['states'][v]
        if v not in features:options[v]=[];continue
        for model in ((row['model'],) if snapshot.get('rates_only') else ranker.models):
            if model not in profile:continue
            compute=sum(profile[model]['segment_estimates_s'])
            earliest=max(ready,row['minimum_release_s'],row['previous_release_s']+.020)
            latest=min(row['deadline_s']-service-compute,row['previous_release_s']+.080)
            releases={row['previous_release_s']+t for t in (.020,.035,.050,.065,.080)}|{earliest,latest,row['planned_release_s']}
            releases=np.array(sorted(releases));periods=releases-row['previous_release_s']
            delays=releases-state['sample_s']+service+compute+queued
            valid=(releases>=earliest)&(releases<=latest)&ranker.supported_many(model,periods,delays)
            for release,period,delay in zip(releases[valid],periods[valid],delays[valid]):
                actions.append(dict(id=f'{model}/T{period*1000:.6f}',model=model,release_s=release,
                    initial_hold_delay_estimate_s=delay,period_s=period,
                    source='state_conditioned_advantage_ranker',quality_scope='ADVISORY proxy ranking; NOT recovery qualified or FALCON J'))
        if actions:
            values=ranker.scores(features[v],[a['model'] for a in actions],
                [a['period_s'] for a in actions],[a['initial_hold_delay_estimate_s'] for a in actions])
            for action,value in zip(actions,values):action['value']=float(value)
        options[v]=actions
    return options


class Trigger:
    def __init__(self, interval=.050):
        self.interval, self.last = interval, -float('inf')
        self.dirty, self.urgent, self.levels = set(), set(), {}
        self.first_notice = None

    def notice(self, vehicle, now, slack=None):
        self.dirty.add(vehicle)
        if self.first_notice is None:self.first_notice=now
        if slack is None:return
        level = slack <= 0.
        if level and not self.levels.get(vehicle,False):self.urgent.add(vehicle)
        self.levels[vehicle] = level

    def due(self, now):
        return bool(self.urgent or (self.dirty and now-self.last>=self.interval))

    def consumed(self, now):
        info=dict(reason='response_margin_edge' if self.urgent else 'merged_updates',
                  dirty=sorted(self.dirty),urgent=sorted(self.urgent),notice_s=self.first_notice)
        self.last=now;self.dirty.clear();self.urgent.clear();self.first_notice=None
        return info


FIELDS = ('vehicle','sample_id','model','priority','planned_release_s','deadline_s',
    'minimum_release_s','requested_period_s','previous_release_s','serial_vehicle','input_age_limit_s',
    'stage','sample_io_remaining_s','sample_response_received_s','ingress_remaining_s',
    'io_remaining_s','command_remaining_s','command_completed_s','analysis_remaining_s',
    'nonblocking_maintenance','next_plan_created','obligation_actuation_s','obligation_sample_s')


def compact(row):
    return {k:row[k] for k in FIELDS if k in row}


def options_for(snapshot, priors, profile, costs, planning_ready_s=None):
    options={}
    service=sum(costs.get(k,0.) for k in
        ('sample_io_s','input_s','ingress_s','egress_s','handoff_s','command_s','completion_guard_s'))
    for v,row in snapshot['future'].items():
        actions=[]
        for model,knots in priors.get(v,{}).items():
            if snapshot.get('rates_only') and model!=row['model']:continue
            periods,values=zip(*knots)
            latest=row['deadline_s']-service-sum(profile[model]['segment_estimates_s'])-row['previous_release_s']
            candidate=list(periods)
            current=row['planned_release_s']-row['previous_release_s']
            derived=[latest,current]
            if planning_ready_s is not None:derived.append(planning_ready_s-row['previous_release_s'])
            candidate.extend(t for t in derived if periods[0]<=t<=periods[-1])
            for period in sorted(set(candidate)):
                actions.append(dict(id=f'{model}/T{period*1000:.6f}',model=model,
                    release_s=row['previous_release_s']+period,value=float(np.interp(period,periods,values)),
                    source='bracketed_nominal_short_J_prior',quality_scope='NOT state-conditioned recovery qualification'))
        options[v]=actions
    return options


def timing_options_for(snapshot,profile,costs,planning_ready_s):
    """Probe timing independently of missing J measurements; never invent J."""
    service=sum(costs.get(k,0.) for k in
        ('sample_io_s','input_s','ingress_s','egress_s','handoff_s','command_s','completion_guard_s'))
    options={}
    models=sorted(profile,key=lambda m:sum(profile[m]['segment_estimates_s']))
    for v,row in snapshot['future'].items():
        actions=[]
        earliest=max(planning_ready_s,row['minimum_release_s'],row['previous_release_s']+.020)
        for model in ((row['model'],) if snapshot.get('rates_only') else models):
            latest=min(row['deadline_s']-service-sum(profile[model]['segment_estimates_s']),
                       row['previous_release_s']+.080)
            if latest<earliest:continue
            for release in sorted({earliest,latest}):
                actions.append(dict(id=f'{model}/r{release:.9f}',model=model,release_s=release,value=0.,
                    source='timing_feasibility_probe',quality_scope='UNKNOWN; zero is a dummy objective, NOT J'))
        options[v]=actions
    return options


def freeze_near_releases(snapshot, now, profile, costs, reserve=.006):
    """Preserve near/published IDs and charge their sustained demand as well."""
    future={};fixed=[]
    for row in snapshot['future'].values():
        if row['planned_release_s']<=now+reserve:fixed.append({**row,'stage':'input'})
        else:future[row['vehicle']]=row
    frontend=sum(costs.get(k,0.) for k in ('input_s','output_s','planner_reserve_s','handoff_s'))
    locked_load=np.sum([[sum(profile[r['model']]['segment_estimates_s'])/r['requested_period_s'],
                        frontend/r['requested_period_s']] for r in fixed],axis=0) if fixed else np.zeros(2)
    return future,snapshot['pending']+fixed,.95-locked_load,fixed


def decide_snapshot(snapshot, certificates, now, profile, costs, priors, *, recover_after_miss=False,
                    feasibility_only=False, restore_without_J=False, effect=None):
    """Identical finite decision path for archived snapshots and live shadow."""
    began=time.perf_counter();future={v:dict(r) for v,r in snapshot['future'].items()}
    # Optional live review needs 3ms plus a 1ms return/dispatch allowance,
    # beyond the bounded 6ms planner. This time is charged, never hidden in R.
    ready_s=now+.006+(.004 if effect is not None else .001)
    recovery=dict(renewals=[],deferred=[],previous_misses=[],historical_deadlines_repaired=False)
    if recover_after_miss:
        future,recovery=prepare_recovery(now,future,snapshot['pending'],certificates)
    else:
        for v,cert in certificates.items():
            if cert.get('available'):future[v]['deadline_s']=min(future[v]['deadline_s'],cert['deadline_s'])
    movable,pending,caps,fixed=freeze_near_releases({**snapshot,'future':future},now,profile,costs,
        reserve=ready_s-now-.001)
    effect_used=effect is not None and not feasibility_only
    if effect_used:
        options=effect_options_for({**snapshot,'future':movable},profile,costs,ready_s,*effect)
    else:
        options=(timing_options_for({**snapshot,'future':movable},profile,costs,ready_s) if feasibility_only else
            options_for({**snapshot,'future':movable},priors,profile,costs,ready_s))
    preparation=time.perf_counter()-began
    if not movable or min(caps)<=0.:
        proposal=None;report=dict(accepted=False,reason='no_movable_capacity_inside_planning_window')
    elif preparation>=.006:
        proposal=None;report=dict(accepted=False,reason='preparation_exhausted_planning_budget')
    else:
        proposal,report=optimize(now+preparation,movable,pending,options,profile,costs,
            capacities=tuple(caps),inflight_limit=snapshot['inflight_limit'],
            recover_after_miss=recover_after_miss,planning_budget_s=.006-preparation)
    timing_objective=feasibility_only
    if restore_without_J and report.get('reason')=='no_individually_time_feasible_contract':
        ranking_rejection=report['reason']
        options=timing_options_for({**snapshot,'future':movable},profile,costs,ready_s)
        elapsed=time.perf_counter()-began
        if elapsed<.006:
            proposal,report=optimize(now+elapsed,movable,pending,options,profile,costs,
                capacities=tuple(caps),inflight_limit=snapshot['inflight_limit'],
                recover_after_miss=recover_after_miss,planning_budget_s=.006-elapsed)
            report['ranking_rejection']=ranking_rejection
            timing_objective=True
    report.update(recovery=recovery,decision_total_ms=1000*(time.perf_counter()-began),
        selection_objective=('timing_feasibility_only' if timing_objective else
            'state_conditioned_recovery_tracking_advantage' if effect_used else 'nominal_short_J_prior'),
        quality_ranked=not timing_objective,
        quality_qualified=False,
        all_loop_future_contracts_covered=not recovery['deferred'])
    if recovery['previous_misses']:report['strict_timing_passed']=False
    return proposal,report,movable,caps,fixed


def worker(memory, length, generation, lock, changed, stop, ready, error, profile, costs, priors_path, cpu, path,
           recover_after_miss, geometry, effect_path, reply):
    from types import SimpleNamespace
    from threadpoolctl import threadpool_limits
    from .framework_cpu_affinity import set_process_cpus
    from .held_control_safety import GateHoldAnalyzer, GateHoldBatch, HoldBounds, safe_deadline
    from environment import Gate
    try:
        placement=set_process_cpus([cpu]);priors=load_priors(Path(priors_path));seen=0
        from .controller_effect import EffectRanker,reference_from_spec,live_features,contextual_live_features
        ranker=EffectRanker(effect_path) if effect_path else None
        windows={v:reference_from_spec(g['reference']) for v,g in geometry.items()} if ranker else {}
        analyzers={};iteration_ms=[];cpu_ms=[]
        bounds=HoldBounds(2.,.01,.05,.15,10.,'Development sensitivity; bounds NOT plant-certified')
        for v,g in geometry.items():
            gates=[Gate(r['index'],np.array(r['center']),r['orientation']) for r in g['gates']]
            analyzers[v]=GateHoldAnalyzer(gates,g['net_width'],bounds)
        batch=GateHoldBatch(analyzers)
        with threadpool_limits(limits=1), Path(path).open('w',encoding='utf-8') as handle:
            ready.set()
            while not stop.is_set() or generation.value!=seen:
                changed.wait(.050);changed.clear()
                payload=None
                with lock:
                    if generation.value!=seen:
                        seen=generation.value;payload=ctypes.string_at(ctypes.addressof(memory),length.value)
                if payload is None:
                    continue
                begin=time.perf_counter();cpu_begin=time.thread_time()
                snapshot=json.loads(payload);future=snapshot['future'];barriers={}
                original_future={v:dict(r) for v,r in future.items()}
                paired={v:SimpleNamespace(**{k:np.array(s[k]) for k in ('position','velocity','quaternion')})
                    for v,s in snapshot['states'].items() if s['held_pair_current']}
                analyses=batch.analyze(paired,{v:snapshot['states'][v]['held_control'] for v in paired})
                for v,state in snapshot['states'].items():
                    if not state['held_pair_current']:
                        barriers[v]=dict(available=False,reason='newer acknowledged command lacks paired state')
                        continue
                    analysis=analyses[v]
                    timing=safe_deadline(state['sample_s'],begin,state['actuation_s'],analysis,.080)
                    barriers[v]={'available':True,**analysis,**timing,
                        'sample_s':state['sample_s'],'actuation_s':state['actuation_s']}
                effect_features={}
                if ranker:
                    for v,s in snapshot['states'].items():
                        if s['held_pair_current']:
                            elapsed=geometry[v]['reference']['phase']+s['elapsed_s']
                            state={k:np.asarray(s[k]) for k in ('position','velocity','quaternion','held_control')}
                            effect_features[v]=(contextual_live_features(windows[v],elapsed,state,geometry[v]['net_width'],
                                analyses[v]['mesh_margin_m']) if ranker.context else live_features(windows[v],elapsed,state))
                solve_s=time.perf_counter()
                proposal,report,movable,caps,fixed=decide_snapshot(snapshot,barriers,solve_s,
                    profile,costs,priors,recover_after_miss=recover_after_miss,
                    restore_without_J=recover_after_miss,effect=(ranker,effect_features) if ranker else None)
                if proposal and ranker and report['quality_ranked']:
                    # The optimizer's shared queue can differ from the cheap
                    # initial delay estimate. Do not call out-of-support delay ranked.
                    finish=next(c['finish'] for c in report['checks'] if c['passed'])
                    supported=all(ranker.supported(r['model'],r['requested_period_s'],
                        finish[r['sample_id']]-snapshot['states'][v]['sample_s']) for v,r in proposal.items())
                    report['final_forecast_inside_ranker_support']=supported
                    if not supported:report['quality_ranked']=False
                end=time.perf_counter()
                if reply is not None and proposal and report.get('all_loop_future_contracts_covered'):
                    memory_out,length_out,generation_out,lock_out=reply
                    model_advantage={}
                    if ranker:
                        finish=next(c['finish'] for c in report['checks'] if c['passed'])
                        for v,r in proposal.items():
                            old_model=original_future[v]['model'];delay=finish[r['sample_id']]-snapshot['states'][v]['sample_s']
                            if v not in effect_features or r['model']==old_model or not all(ranker.supported(m,r['requested_period_s'],delay) for m in (r['model'],old_model)):
                                model_advantage[v]=False;continue
                            scores=ranker.raw_scores(effect_features[v],[r['model'],old_model],
                                [r['requested_period_s']]*2,[delay]*2)
                            model_advantage[v]=bool(scores[0]-scores[1]>max(0.,ranker.override_margin or 0.))
                    recommendation=dict(generation=seen,proposal={**{r['vehicle']:r for r in fixed},**proposal},
                        source_future=original_future,native_basis=snapshot['native_basis'],
                        control_tokens={v:s['control_token'] for v,s in snapshot['states'].items()},
                        certificates=barriers,quality_ranked=report['quality_ranked'],model_advantage_eligible=model_advantage)
                    encoded=json.dumps(recommendation,allow_nan=False).encode()
                    if len(encoded)>len(memory_out):raise RuntimeError('recommendation exceeds return mailbox')
                    if lock_out.acquire(False):
                        try:
                            ctypes.memmove(ctypes.addressof(memory_out),encoded,len(encoded))
                            length_out.value=len(encoded);generation_out.value=seen
                        finally:lock_out.release()
                end=time.perf_counter()
                record=dict(generation=seen,trigger=snapshot['trigger'],snapshot_s=snapshot['captured_s'],
                    worker_begin_s=begin,worker_end_s=end,mailbox_wait_ms=1000*(begin-snapshot['captured_s']),
                    total_worker_ms=1000*(end-begin),worker_cpu_ms=1000*(time.thread_time()-cpu_begin),
                    snapshot_to_result_ms=1000*(end-snapshot['captured_s']),barriers=barriers,
                    original_future=original_future,pending=snapshot['pending'],
                    frozen_near_release_ids=[r['sample_id'] for r in fixed],
                    movable_loops=list(movable),remaining_capacities=caps.tolist(),
                    generations_can_advance_after_read=True,proposal=proposal,report=report,
                    prior_scope='nominal short-J; future flight J NOT measured',recommendations_applied=False,
                    all_original_releases_still_future=bool(movable) and end<min(r['planned_release_s'] for r in movable.values()),
                    placement=placement)
                handle.write(json.dumps(record,allow_nan=False)+'\n');handle.flush()
                iteration_ms.append(1000*(time.perf_counter()-begin))
                cpu_ms.append(1000*(time.thread_time()-cpu_begin))
            atomic_json(Path(path).with_name('worker_totals.json'),dict(iteration_ms=iteration_ms,
                cpu_total_ms=sum(cpu_ms),includes_result_serialization_and_log_flush=True))
    except Exception:
        import traceback
        Path(str(path)+'.error.txt').write_text(traceback.format_exc(),encoding='utf-8');error.set();ready.set()


class JointDecisionShadow:
    def __init__(self, profile, costs, priors_path, cpu, directory, *, geometry, recover_after_miss=False,effect_path=None,
                 fresh_review=False,apply_rates=False):
        if apply_rates and (not fresh_review or not effect_path):
            raise ValueError('rate application requires fresh review and a frozen effect ranker')
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=False)
        ctx=mp.get_context('spawn')
        self.memory=ctx.RawArray(ctypes.c_ubyte,262144)
        self.length,self.generation=ctx.RawValue('i',0),ctx.RawValue('q',0)
        self.lock,self.changed,self.stop,self.ready,self.error=ctx.Lock(),ctx.Event(),ctx.Event(),ctx.Event(),ctx.Event()
        self.profile,self.costs,self.trigger=profile,costs,Trigger()
        self.effect_path=effect_path;self.initial={};self.end_s=0.;self.review_rows=[];self.review_ms=[];self.reply_seen=0
        self.fresh_review=fresh_review
        self.apply_rates=apply_rates;self.installed_groups=[];self.committed_updates=[]
        if fresh_review:
            from .held_control_safety import GateHoldAnalyzer,GateHoldBatch,HoldBounds
            from .controller_effect import EffectRanker,reference_from_spec
            from environment import Gate
            bounds=HoldBounds(2.,.01,.05,.15,10.,'Development sensitivity; bounds NOT plant-certified')
            self.review_batch=GateHoldBatch({v:GateHoldAnalyzer([Gate(r['index'],np.array(r['center']),r['orientation'])
                for r in g['gates']],g['net_width'],bounds) for v,g in geometry.items()})
            self.review_geometry=geometry
            self.review_windows={v:reference_from_spec(g['reference']) for v,g in geometry.items()}
            self.review_ranker=EffectRanker(effect_path)
        self.reply=(ctx.RawArray(ctypes.c_ubyte,131072),ctx.RawValue('i',0),ctx.RawValue('q',0),ctx.Lock()) if effect_path else None
        self.recovery_enabled=recover_after_miss;self.ack_updates=0
        self.states={};self.note_ms=[];self.capture_ms=[];self.skips=Counter();self.offers=0;self.notes=0
        self.last_attempt=-float('inf');self.started=False;self.closed=False
        self.path=self.directory/'decisions.jsonl'
        self.process=ctx.Process(target=worker,args=(self.memory,self.length,self.generation,self.lock,self.changed,
            self.stop,self.ready,self.error,profile,costs,str(priors_path),cpu,str(self.path),recover_after_miss,geometry,
            str(effect_path) if effect_path else None,self.reply),daemon=True)
        self.process.start()
        if not self.ready.wait(30.) or self.error.is_set():
            self.stop.set();self.process.join(5.)
            raise RuntimeError('shadow worker did not initialize; inspect its error file')

    def start(self,initial=None,end_s=None):
        self.initial=initial or {};self.end_s=end_s or 0.
        self.started=True

    def note(self, v, raw, observation, action, actuation, budget, row):
        if not self.started:return
        begin=time.perf_counter();self.notes+=1
        previous=self.states.get(v)
        if previous and (actuation<previous['actuation_s'] or
            (actuation==previous['actuation_s'] and observation['sample_s']<previous['observation']['sample_s'])):
            self.note_ms.append(1000*(time.perf_counter()-begin));return
        token=row.get('control_ack_token')
        if token is None and previous and previous['actuation_s']==actuation:token=previous.get('control_token')
        self.states[v]=dict(raw=raw,observation=observation,held_control=action,
            actuation_s=actuation,control_token=token)
        slack=None
        if budget is not None:
            cost=sum(self.profile[row['model']]['segment_estimates_s'])+sum(self.costs.get(k,0.) for k in
                ('sample_io_s','input_s','ingress_s','egress_s','handoff_s','command_s','completion_guard_s'))+.006
            slack=budget['deadline_s']-max(begin,row['planned_release_s'])-cost
        self.trigger.notice(v,begin,slack)
        self.note_ms.append(1000*(time.perf_counter()-begin))

    def observe_ack(self, packet, row, plan):
        """Record the paired post-command RPC now, not after legacy planning."""
        if not self.started or packet is None or plan is None:return
        raw,_,sampled=packet
        if sampled<row['rpc_return_s']:return
        self.ack_updates+=1
        self.note(row['vehicle'],raw,{'sample_s':sampled},row['mapped_control'],row['rpc_return_s'],
            None,{**plan,'control_ack_token':row.get('control_ack_token')})

    def review_latest(self,plans,pending_builder,io,inflight_limit,last_released,last_command):
        if self.reply is None:return
        memory,length,generation,lock=self.reply
        if generation.value==self.reply_seen or not lock.acquire(False):return
        began=time.perf_counter()
        try:
            self.reply_seen=generation.value
            recommendation=json.loads(ctypes.string_at(ctypes.addressof(memory),length.value))
        finally:lock.release()
        pending=[compact(r) for r in pending_builder(time.perf_counter())]
        rebase_info={}
        if self.fresh_review:
            from .held_control_safety import safe_deadline
            from .controller_effect import contextual_live_features,live_features
            valid=all(v in self.states and self.states[v].get('control_token') is not None and
                abs(self.states[v]['actuation_s']-last_command[v])<1e-9 for v in plans)
            if not valid:
                self.review_ms.append(1000*(time.perf_counter()-began))
                self.review_rows.append(dict(generation=self.reply_seen,passed=False,installed=False,
                    quality_ranked=False,reason='fresh_paired_state_unavailable'));return
            batch=self.review_batch.analyze({v:self.states[v]['raw'] for v in plans},
                {v:self.states[v]['held_control'] for v in plans})
            now=time.perf_counter();certs={}
            for v in plans:
                s=self.states[v]
                certs[v]={'available':True,**batch[v],**safe_deadline(s['observation']['sample_s'],now,
                    s['actuation_s'],batch[v],.080),'sample_s':s['observation']['sample_s'],'actuation_s':s['actuation_s']}
            rebuilt,rebase_info=rebase_slots(recommendation,plans,pending,certs,last_released,now)
            if rebuilt is None:
                self.review_ms.append(1000*(time.perf_counter()-began))
                self.review_rows.append(dict(generation=self.reply_seen,passed=False,installed=False,
                    quality_ranked=False,**rebase_info));return
            # Model changes need a fresh comparison against the CURRENT mode.
            # Without sufficient evidence keep that mode; rate advice remains
            # explicitly unqualified until the fresh timing/safety checks pass.
            model_deferred=[]
            for v,row in rebuilt['proposal'].items():
                if row['model']==plans[v]['model']:continue
                if self.apply_rates:
                    row['model']=plans[v]['model'];model_deferred.append(v);continue
                if not recommendation.get('model_advantage_eligible',{}).get(v,False):
                    row['model']=plans[v]['model'];model_deferred.append(v);continue
                s=self.states[v];raw=s['raw'];g=self.review_geometry[v]
                state=dict(position=raw.position,velocity=raw.velocity,quaternion=raw.quaternion,held_control=s['held_control'])
                elapsed=g['reference']['phase']+(raw.timestamp_ns-self.initial[v]['sim_ns'])/1e9
                ranker=self.review_ranker
                feature=(contextual_live_features(self.review_windows[v],elapsed,state,g['net_width'],batch[v]['mesh_margin_m'])
                    if ranker.context else live_features(self.review_windows[v],elapsed,state))
                candidates=[row,plans[v]];delays=[]
                for c in candidates:
                    delays.append(c['planned_release_s']-s['observation']['sample_s']+
                        sum(self.profile[c['model']]['segment_estimates_s'])+
                        sum(self.costs.get(k,0.) for k in ('sample_io_s','input_s','ingress_s','egress_s','handoff_s','command_s')))
                supported=all(ranker.supported(c['model'],c['requested_period_s'],d) for c,d in zip(candidates,delays))
                score=ranker.raw_scores(feature,[c['model'] for c in candidates],
                    [c['requested_period_s'] for c in candidates],delays) if supported else [0.,0.]
                if not supported or score[0]-score[1]<=max(0.,ranker.override_margin or 0.):
                    row['model']=plans[v]['model'];model_deferred.append(v)
            recommendation={**recommendation,**rebuilt,'native_basis':io.calendar_basis(plans),
                'control_tokens':{v:self.states[v]['control_token'] for v in plans},'quality_ranked':False}
            rebase_info['model_changes_without_fresh_advantage_deferred']=model_deferred
        if self.apply_rates:
            for v,row in recommendation['proposal'].items():
                tag_rate_change(row,plans[v],self.reply_seen)
        remaining=.003-(time.perf_counter()-began)
        if remaining<=0.:
            info=dict(passed=False,installed=False,reason='fresh_context_exhausted_review_budget')
        else:
            info=review_commit(recommendation,plans,pending,io,self.profile,self.costs,end_s=self.end_s,
                inflight_limit=inflight_limit,budget_s=remaining,commit=self.apply_rates,rates_only=self.apply_rates)
        remaining=.003-(time.perf_counter()-began)
        if self.fresh_review and not info['passed'] and info['reason'] in (
                'fresh_pipeline_infeasible','insufficient_review_or_release_time','fresh_periodic_load_exceeds_capacity') and remaining>.001:
            basis=io.calendar_basis(plans)
            now=time.perf_counter()
            candidate,repair=repair_current_rates(plans,pending,recommendation['certificates'],last_released,now,
                self.profile,self.costs,inflight_limit=inflight_limit,budget_s=remaining-.0005)
            rebase_info['fresh_rate_repair']=repair
            remaining=.003-(time.perf_counter()-began)
            if candidate and remaining>0.:
                if self.apply_rates:
                    for v,row in candidate.items():
                        tag_rate_change(row,plans[v],self.reply_seen)
                recommendation={**recommendation,'proposal':candidate,'source_future':plans,'native_basis':basis,'quality_ranked':False}
                info=review_commit(recommendation,plans,pending,io,self.profile,self.costs,end_s=self.end_s,
                    inflight_limit=inflight_limit,budget_s=remaining,commit=self.apply_rates,rates_only=self.apply_rates)
        if self.fresh_review:
            rebase_info['actual_period_changes']={v:dict(before=plans[v]['requested_period_s'],after=r['requested_period_s'])
                for v,r in recommendation['proposal'].items() if r['requested_period_s']!=plans[v]['requested_period_s']}
            rebase_info['actual_model_changes']={v:dict(before=plans[v]['model'],after=r['model'])
                for v,r in recommendation['proposal'].items() if r['model']!=plans[v]['model']}
        if info.get('installed'):
            installed=info.pop('installed_proposal')
            plans.update(installed);self.committed_updates.append(installed)
            self.installed_groups.append(dict(generation=self.reply_seen,installed_s=time.perf_counter(),
                active_loops=list(installed),period_changes=rebase_info.get('actual_period_changes',{}),
                model_changes=rebase_info.get('actual_model_changes',{}),proposal=installed))
        self.review_ms.append(1000*(time.perf_counter()-began))
        self.review_rows.append(dict(generation=self.reply_seen,quality_ranked=recommendation['quality_ranked'],
            active_loops=list(plans),rebased=self.fresh_review,rebase=rebase_info,**info))

    def drain_committed(self):
        updates=self.committed_updates;self.committed_updates=[]
        return updates

    def tick(self, plans, pending_builder, io, last_released, last_command, inflight_limit):
        self.review_latest(plans,pending_builder,io,inflight_limit,last_released,last_command)
        now=time.perf_counter()
        if not self.trigger.due(now) or now-self.last_attempt<.001:return
        self.last_attempt=now;begin=now
        try:
            if self.error.is_set():self.skips['worker_error']+=1;return
            if not plans or any(v not in self.states or not np.isfinite(last_released[v]) for v in plans):
                self.skips['missing_initial_feedback']+=1;return
            # Native calendar successors default minimum_release to the OLD
            # periodic slot. A not-yet-published job may legally move earlier:
            # the real minimum is the previous publication +20ms.
            future={v:{**compact(row),'previous_release_s':last_released[v],
                       'minimum_release_s':last_released[v]+.020} for v,row in plans.items()}
            specs=[compact(r) for r in pending_builder(now)]
            if len({r['sample_id'] for r in specs})!=len(specs):
                self.skips['duplicate_pending_id']+=1;return
            states={}
            for v in future:
                state=self.states[v];raw=state['raw'];observation=state['observation']
                states[v]=dict(position=raw.position.tolist(),velocity=raw.velocity.tolist(),
                    quaternion=raw.quaternion.tolist(),held_control=list(map(float,state['held_control'])),
                    sample_s=observation['sample_s'],actuation_s=state['actuation_s'],
                    held_pair_current=abs(state['actuation_s']-last_command[v])<1e-9)
                if self.effect_path:
                    if state.get('control_token') is None:self.skips['missing_control_token']+=1;return
                    states[v].update(control_token=state['control_token'],
                        elapsed_s=(raw.timestamp_ns-self.initial[v]['sim_ns'])/1e9)
            captured=time.perf_counter()
            snap=dict(captured_s=captured,future=future,rates_only=self.apply_rates,
                pending=specs,states=states,inflight_limit=inflight_limit,
                trigger=dict(reason='response_margin_edge' if self.trigger.urgent else 'merged_updates',
                    dirty=sorted(self.trigger.dirty),urgent=sorted(self.trigger.urgent),notice_s=self.trigger.first_notice))
            if self.effect_path:snap['native_basis']=io.calendar_basis(plans)
            payload=json.dumps(snap,allow_nan=False,separators=(',',':')).encode()
            if len(payload)>len(self.memory):raise RuntimeError('shadow snapshot exceeds mailbox capacity')
            # Single Python owner freezes frontend rows; native mutex confirms
            # no calendar publication was omitted during the copy/encoding.
            if not io.calendar_matches(plans):self.skips['native_publication_race']+=1;return
            if not self.lock.acquire(False):self.skips['mailbox_busy']+=1;return
            try:
                ctypes.memmove(ctypes.addressof(self.memory),payload,len(payload))
                self.length.value=len(payload);self.generation.value+=1
            finally:self.lock.release()
            self.changed.set()
            self.trigger.consumed(captured);self.offers+=1
        finally:
            self.capture_ms.append(1000*(time.perf_counter()-begin))

    def close(self):
        if self.closed:return
        self.stop.set();self.changed.set();self.process.join(10.)
        if self.process.is_alive():self.process.terminate();self.process.join(5.);self.skips['worker_forced_stop']+=1
        self.closed=True
        rows=[json.loads(line) for line in self.path.read_text(encoding='utf-8').splitlines()] if self.path.exists() else []
        def stats(a):
            return dict(count=len(a),mean=float(np.mean(a)),p99=float(np.percentile(a,99)),max=float(max(a))) if a else None
        accepted=[r for r in rows if r['report']['accepted']]
        totals_path=self.directory/'worker_totals.json'
        totals=json.loads(totals_path.read_text()) if totals_path.exists() else {}
        report=dict(recommendations_applied=bool(self.installed_groups),physical_safety_certified=False,notes=self.notes,
            application_enabled=self.apply_rates,installed_groups=len(self.installed_groups),
            fresh_rebase_enabled=self.fresh_review,
            effect_ranker_enabled=bool(self.effect_path),atomic_review_ms=stats(self.review_ms),
            atomic_reviews=len(self.review_rows),atomic_dry_run_passed=sum(r['passed'] for r in self.review_rows),
            atomic_review_reasons=dict(Counter(r['reason'] for r in self.review_rows)),
            recovery_enabled=self.recovery_enabled,paired_ack_updates=self.ack_updates,
            offers=self.offers,processed=len(rows),
            coalesced_before_processing=sum(max(0,r['generation']-previous-1)
                for previous,r in zip([0]+[r['generation'] for r in rows[:-1]],rows)),
            trigger_reasons=dict(Counter(r['trigger']['reason'] for r in rows)),skips=dict(self.skips),
            note_ms=stats(self.note_ms),capture_ms=stats(self.capture_ms),
            frontend_total_ms=sum(self.note_ms)+sum(self.capture_ms)+sum(self.review_ms),
            worker_total_ms=stats([r['total_worker_ms'] for r in rows]),
            worker_cpu_total_ms=totals.get('cpu_total_ms'),worker_iteration_ms=stats(totals.get('iteration_ms',[])),
            worker_cost_includes_logging=bool(totals),
            snapshot_to_result_ms=stats([r['snapshot_to_result_ms'] for r in rows]),
            accepted=len(accepted),accepted_original_ids_still_unreleased=sum(r['all_original_releases_still_future'] for r in accepted),
            future_group_feasible=sum(r['report'].get('all_loop_future_contracts_covered',False) for r in accepted),
            quality_ranked_accepted=sum(r['report'].get('quality_ranked',False) for r in accepted),
            timing_only_accepted=sum(r['report'].get('selection_objective')=='timing_feasibility_only' for r in accepted),
            ranked_but_forecast_outside_support=sum(r['report'].get('selection_objective')==
                'state_conditioned_recovery_tracking_advantage' and not r['report'].get('quality_ranked') for r in accepted),
            strict_timing_passed=sum(r['report'].get('strict_timing_passed',False) for r in accepted),
            renewed_obligations=sum(len(r['report'].get('recovery',{}).get('renewals',[])) for r in rows),
            rejection_reasons=dict(Counter(r['report'].get('reason') for r in rows if not r['report']['accepted'])),
            worker_error=self.error.is_set() or self.process.exitcode!=0,
            scope=('guarded rate assistance to legacy control; no recovery/safety qualification' if self.apply_rates else
                'real-time shadow; optional fresh whole-group forecast and native atomic dry-run; NO takeover/qualification'))
        atomic_json(self.directory/'atomic_reviews.json',self.review_rows)
        atomic_json(self.directory/'installed_groups.json',self.installed_groups)
        atomic_json(self.directory/'report.json',report)
        return report
