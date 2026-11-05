"""Finite pipeline admission and honest empirical timing contracts."""
import time
import numpy as np

from .decision_dataset import atomic_json,sha256
from .dynamic_probe_common import read_json
from .framework_runtime import release_deadline,simulate_pipeline,remaining_stage_cost


def freeze_costs(source,output,statistic='maximum',*,native=False):
    if statistic not in ('maximum','p95'):raise ValueError('unknown timing statistic')
    rows=read_json(source/'requests.json')
    fields={'sample_io_s':('sample_submitted_s','sample_response_received_s'),
        'input_s':('frontend_state_start_s','feature_done_s'),
        'output_s':('completion_frontend_start_s','completion_frontend_done_s'),
        'io_s':('io_submitted_s','io_response_received_s'),
        'command_s':('io_submitted_s','rpc_return_s'),
        'actuation_s':('io_started_s','rpc_return_s'),
        'ingress_s':('enqueue_s','worker_accept_received_s'),
        'egress_s':('ready_s','io_submitted_s')}
    if native:fields['ingress_s']=('enqueue_s','worker_enqueue_s')
    costs={};measurements={};tails={}
    for name,(start,end) in fields.items():
        a=np.asarray([r[end]-r[start] for r in rows if r.get('status')=='applied' and start in r and end in r])
        if not len(a) or np.min(a)<0.:raise ValueError('missing or negative pipeline timing: '+name)
        costs[name]=float(1.25*(np.max(a) if statistic=='maximum' else np.percentile(a,95))+.0002)
        tails[name]=float(1.25*np.max(a)+.0002)
        measurements[name]=dict(count=len(a),max_s=float(np.max(a)),p99_s=float(np.percentile(a,99)))
    costs.update(handoff_s=.0001,completion_guard_s=.0005,dispatch_guard_s=0.,planner_reserve_s=.001,io_slots=3,is_WCET=False)
    if native:
        jitter=[max(0.,r['release_parent_received_s']-r['planned_release_s']) for r in rows
            if 'release_parent_received_s' in r and 'planned_release_s' in r]
        if jitter:
            costs['period_release_dispatch_s']=float(1.25*np.percentile(jitter,95)+.0002)
            tails['period_release_dispatch_s']=float(1.25*max(jitter)+.0002)
        analyses=read_json(source/'analyses.json')
        a=np.asarray([r['analysis_ms']/1000 for r in analyses])
        if not len(a):raise ValueError('native calibration requires recorded budget/mode analysis')
        # Reconfiguration simulations charge their measured time in admit();
        # ordinary feedback only refreshes B/mode, not another global search.
        costs['planner_reserve_s']=float(1.25*np.percentile(a,95)+.0005)
        measurements['planner_reserve_s']=dict(count=len(a),p95_s=float(np.percentile(a,95)),
            max_s=float(np.max(a)),scope='budget/mode analysis +0.5ms observation allowance; transition checks charged separately')
    costs['tail_costs']=tails
    payload=dict(costs=costs,measurements=measurements,source=str(source),
        source_sha256=sha256(source/'requests.json'),method=f'1.25*observed {statistic} +0.2ms; maxima retained as pending-operation tails',
        statistic=statistic,native_calibration=native,soft_prediction_only=True,
        scope='empirical admission envelope; not a Windows hard-real-time proof')
    atomic_json(output,payload)
    return payload


def pending_specs(now,pending,ready,inputs,completed,io,costs,*,feedback=None,current_feedback=None):
    result=[]
    def remaining(row,name,start):
        return remaining_stage_cost(costs,name,max(0.,now-row.get(start,now)))[0]
    for v,row in pending.items():
        spec={**row,'stage':'output' if v in ready else 'compute'}
        if v in io.samples:
            spec.update(stage='sampling',sample_io_remaining_s=remaining(row,'sample_io_s','sample_submitted_s'))
        elif v in inputs:spec['stage']='input_ready'
        elif v in completed:
            spec.update(stage='analysis',command_completed_s=row.get('rpc_return_s',now))
        elif v in io.futures:
            spec.update(stage='io',io_remaining_s=remaining(row,'io_s','io_submitted_s'),
                command_remaining_s=remaining(row,'command_s','io_submitted_s'))
            if row.get('status')=='applied':spec['command_completed_s']=row['rpc_return_s']
        elif spec['stage']=='compute':
            spec['ingress_remaining_s']=0. if 'worker_enqueue_s' in row else remaining(row,'ingress_s','enqueue_s')
        result.append(spec)
    present={r['sample_id'] for r in result}
    maintenance={**(feedback or {}),**completed} if feedback is not None else {}
    for packet,row in maintenance.values():
        cleanup_left='completion_frontend_done_s' not in row
        planning_left=not row.get('next_plan_created')
        if row['sample_id'] not in present and row['sample_id']!=current_feedback and (cleanup_left or planning_left):
            result.append({**row,'stage':'analysis','command_completed_s':row['rpc_return_s'],
                'nonblocking_maintenance':True,'analysis_remaining_s':
                    (costs.get('planner_reserve_s',0.) if planning_left else 0.)+(costs['output_s'] if cleanup_left else 0.)})
    return result


def admit(now,plans,pending,profile,costs,inflight_limit=3,*,policy='edf',allow_earlier=True,planning_started_s=None,missed_deadline_ids=()):
    """Check current releases, then ONE earlier-release alternative; freeze all D/c."""
    began=time.perf_counter() if planning_started_s is None else planning_started_s;attempts=[]
    missed=set(missed_deadline_ids)
    for row in pending:
        if any(row.get(k,1.)<=0. for k in ('sample_io_remaining_s','io_remaining_s','command_remaining_s')) or row.get('ingress_remaining_s',0.)<0.:
            return None,dict(feasible=False,reason='observed operation exceeded the supplied empirical envelope',checked=0,
                sample_id=row['sample_id'],planning_ms=1000*(time.perf_counter()-began))
    candidates=[plans,{v:{**r,'planned_release_s':max(now,r.get('minimum_release_s',now))} for v,r in plans.items()}]
    if not allow_earlier:candidates=candidates[:1]
    for index,proposal in enumerate(candidates):
        options=dict(policy=policy,inflight_limit=inflight_limit)
        if missed:options['missed_deadline_ids']=missed
        good,finish,events=simulate_pipeline(now,list(proposal.values()),pending,profile,costs,**options)
        lag=time.perf_counter()-began
        deadlines={r['sample_id']:release_deadline(r,r['planned_release_s']) for r in [*proposal.values(),*pending]}
        acknowledged={r['sample_id'] for r in pending if 'command_completed_s' in r}
        adjusted={k:t+(0. if k in acknowledged else lag) for k,t in finish.items()}
        accepted=good and all(t+costs.get('completion_guard_s',0.)<=deadlines[k] for k,t in adjusted.items()
            if k not in acknowledged and k not in missed)
        unmet=[k for k,d in deadlines.items() if k not in acknowledged and k not in missed
            and (k not in adjusted or adjusted[k]+costs.get('completion_guard_s',0.)>d)]
        attempts.append(dict(feasible=good,accepted=accepted,unmet_deadline_ids=unmet,
            events=events,finish=adjusted,planning_lag_s=lag,
            response_forecast_s={r['sample_id']:adjusted[r['sample_id']]-r['planned_release_s']+
                costs.get('completion_guard_s',0.) for r in proposal.values() if r['sample_id'] in adjusted}))
        if accepted:
            return {v:{**r,'planned_release_s':max(r['planned_release_s'],now+lag) if allow_earlier else r['planned_release_s']} for v,r in proposal.items()},dict(
                feasible=True,checked=index+1,attempts=attempts,planning_ms=1000*lag,
                deadline_or_model_changed=False,is_WCET=False)
    return None,dict(feasible=False,checked=len(candidates),attempts=attempts,planning_ms=1000*(time.perf_counter()-began),
        reason='the bounded release candidate set could not meet every committed deadline under the empirical pipeline model',is_WCET=False)


def admit_change(now,plans,pending,candidate,fallback,profile,costs,inflight_limit=3,*,policy='edf',recover_after_miss=False):
    """Check every known obligation; reject a change before touching other plans."""
    v=candidate['vehicle']
    if (fallback['vehicle']!=v or fallback['sample_id']!=candidate['sample_id']
            or fallback['deadline_s']!=candidate['deadline_s']):
        raise ValueError('fallback must retain the same physical obligation')
    began=time.perf_counter();attempts=[]
    obligations=[*{**plans,v:candidate}.values(),*pending]
    expired=[r['sample_id'] for r in obligations if 'command_completed_s' not in r and r['deadline_s']<=now]
    if expired and not recover_after_miss:
        return dict(fallback),dict(feasible=False,decision='infeasible_retained',checked=0,attempts=[],
            covered_ids=[r['sample_id'] for r in obligations],already_expired_ids=expired,
            planning_ms=1000*(time.perf_counter()-began),other_plans_changed=False,is_WCET=False,
            reason='an unfinished obligation has already expired; a new configuration cannot repair its past deadline')
    for decision,row in (('accepted',candidate),('retained_previous',fallback)):
        future={**plans,v:row}
        remaining={**costs,'frontend_reserved_until_s':now+costs.get('planner_reserve_s',0.)}
        proposal,info=admit(now,future,pending,profile,remaining,inflight_limit,policy=policy,
            allow_earlier=False,planning_started_s=began,missed_deadline_ids=expired if recover_after_miss else ())
        attempts.append(dict(decision=decision,**info))
        if proposal is not None:
            return proposal[v],dict(feasible=not expired,future_feasible=True,
                decision=('recovery_'+decision if expired else decision),checked=len(attempts),attempts=attempts,
                already_expired_ids=expired,historical_deadlines_repaired=False,
                protected_future_ids=[r['sample_id'] for r in [*future.values(),*pending] if r['sample_id'] not in expired],
                covered_ids=[r['sample_id'] for r in [*future.values(),*pending]],
                planning_ms=1000*(time.perf_counter()-began),other_plans_changed=False,is_WCET=False)
        if all(candidate.get(k)==fallback.get(k) for k in ('model','planned_release_s','requested_period_s')):break
    return dict(fallback),dict(feasible=False,decision='infeasible_retained',checked=len(attempts),attempts=attempts,
        already_expired_ids=expired,future_feasible=False,
        covered_ids=[r['sample_id'] for r in [*[r for name,r in plans.items() if name!=v],*pending,candidate]],
        planning_ms=1000*(time.perf_counter()-began),other_plans_changed=False,is_WCET=False,
        reason='neither candidate nor retained configuration passed; continuation is observational, not admitted')


def delivery_fits(now,row,costs):
    return now+costs['actuation_s']+costs.get('handoff_s',0.)<=row['deadline_s']


def admit_group(now,base,pending,candidate,profile,costs,compute_costs,inflight_limit=3,*,policy='edf',recover_after_miss=True):
    """Two bounded alternatives; change only unreleased configurations as a group."""
    from .heterogeneous_load_guard import resource_loads
    if base.keys()!=candidate.keys() or any(any(base[v].get(k)!=candidate[v].get(k)
            for k in ('vehicle','sample_id','deadline_s')) for v in base):
        raise ValueError('group must preserve every unreleased job commitment and physical deadline')
    pending_ids={r['sample_id'] for r in pending}
    if pending_ids.intersection(r['sample_id'] for r in candidate.values()):
        raise ValueError('a released or in-flight commitment cannot be changed')
    began=now;attempts=[]
    expired=[r['sample_id'] for r in [*base.values(),*pending]
        if 'command_completed_s' not in r and r['deadline_s']<=now]
    if expired and not recover_after_miss:
        return dict(base),dict(feasible=False,future_feasible=False,decision='infeasible_retained',
            already_expired_ids=expired,checked=0,attempts=[],planning_ms=0.,other_plans_changed=False)
    for name,proposal in [('joint_accepted',candidate),('retained_previous',base)]:
        demands=resource_loads({v:r['requested_period_s'] for v,r in proposal.items()},
            {v:r['model'] for v,r in proposal.items()},compute_costs,costs)
        if max(demands.values())>.95+1e-12:
            attempts.append(dict(decision=name,resource_loads=demands,feasible=False,reason='sustained resource demand exceeds .95'))
            continue
        selected,info=admit(now,proposal,pending,profile,costs,inflight_limit,policy=policy,
            allow_earlier=False,planning_started_s=began,missed_deadline_ids=expired if recover_after_miss else ())
        attempts.append(dict(decision=name,resource_loads=demands,**info))
        if selected is not None:
            return selected,dict(feasible=not expired,future_feasible=True,decision=name,
                checked=len(attempts),attempts=attempts,already_expired_ids=expired,
                historical_deadlines_repaired=False,other_plans_changed=any(selected[v]!=base[v] for v in base),
                planning_ms=1000*(time.perf_counter()-began),is_WCET=False)
    return dict(base),dict(feasible=False,future_feasible=False,decision='infeasible_retained',
        checked=len(attempts),attempts=attempts,already_expired_ids=expired,other_plans_changed=False,
        planning_ms=1000*(time.perf_counter()-began),is_WCET=False,
        reason='neither the joint change nor retained configuration passed rate and transition checks')
