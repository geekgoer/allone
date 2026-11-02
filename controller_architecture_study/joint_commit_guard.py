"""Fresh finite-pipeline admission + native atomic context guard.

Dry-run by default; optional guarded installation. Empirical forecast, not WCET.
"""

import time

from .framework_runtime import simulate_pipeline
from .joint_contract_recovery import prepare_recovery
from .joint_contract_optimizer import optimize


def repair_current_rates(plans,pending,certificates,last_released,now,profile,costs,*,inflight_limit,budget_s):
    """Bounded timing restoration with CURRENT modes and no invented quality.

    Restores prompt updates inside the fresh all-loop deadline/resource
    constraints. Used only after failed admission, not an optimizer of J.
    """
    future={v:{**row,'previous_release_s':last_released[v],
        'minimum_release_s':last_released[v]+.020} for v,row in plans.items()}
    future,recovery=prepare_recovery(now,future,pending,certificates,reserve_s=budget_s)
    if recovery['deferred']:return None,dict(reason='fresh_hold_obligation_unavailable',recovery=recovery)
    service=sum(costs.get(k,0.) for k in ('sample_io_s','input_s','ingress_s','egress_s','handoff_s','command_s','completion_guard_s'))
    options={}
    for v,row in future.items():
        first=max(now+budget_s,row['minimum_release_s'])
        last=min(row['previous_release_s']+.080,row['deadline_s']-service-sum(profile[row['model']]['segment_estimates_s']))
        releases={first,last,row['planned_release_s']}
        options[v]=[dict(id=f"{row['model']}/repair/{r:.9f}",model=row['model'],release_s=r,
            value=-(r-row['previous_release_s'])/.080,
            source='fresh_current_mode_timing_repair',quality_scope='UNKNOWN; prompt recovery updates, not measured flight J')
            for r in sorted(releases) if first<=r<=last]
    proposal,report=optimize(now,future,pending,options,profile,costs,inflight_limit=inflight_limit,
        planning_budget_s=budget_s,recover_after_miss=True,max_checks=2,price_iterations=8)
    report['recovery']=recovery
    return proposal,report


def rebase_slots(recommendation,plans,pending,certificates,last_released,now,*,reserve_s=.0015):
    """Reuse configuration advice, never its old slot IDs or old hold clocks.

    A single frontend owner rebuilds the candidate on its CURRENT calendars.
    Fresh full-group forecast and native CAS are still mandatory afterwards.
    """
    if plans.keys()!=recommendation['source_future'].keys():
        return None,dict(reason='active_loop_group_changed')
    future={v:{**row,'previous_release_s':last_released[v],
        'minimum_release_s':last_released[v]+.020} for v,row in plans.items()}
    current,recovery=prepare_recovery(now,future,pending,certificates,reserve_s=reserve_s)
    if recovery['deferred']:return None,dict(reason='fresh_hold_obligation_unavailable',recovery=recovery)
    proposed={v:dict(row) for v,row in current.items()};deferred=[];changes=[]
    for v,row in proposed.items():
        desired=recommendation['proposal'][v];period=desired['requested_period_s']
        release=last_released[v]+period
        if release<=now+reserve_s or release<row.get('minimum_release_s',-float('inf')):
            deferred.append(v);continue
        if (desired['model'],period)!=(row['model'],row['requested_period_s']):changes.append(v)
        row.update(model=desired['model'],requested_period_s=period,planned_release_s=release,
            nominal_deadline_s=release+period)
    return dict(proposal=proposed,source_future=plans,certificates=certificates),dict(
        reason='rebased_on_current_frontend',configuration_changes=changes,
        advice_already_passed_deferred=deferred,recovery=recovery,
        old_slot_ids_reused=False,old_certificates_reused=False)


def review(recommendation,plans,pending,io,profile,costs,*,end_s,inflight_limit,
           budget_s=.003,commit=False,rates_only=False):
    began=time.perf_counter()
    proposal=recommendation['proposal'];source=recommendation['source_future']
    result=dict(passed=False,installed=False,quality_qualified=False,physical_safety_certified=False)
    def fail(reason):return {**result,'reason':reason,'review_ms':1000*(time.perf_counter()-began)}
    if plans.keys()!=source.keys() or proposal.keys()!=plans.keys():return fail('active_loop_group_changed')
    keys=('sample_id','model','planned_release_s','requested_period_s','deadline_s')
    if any(any(plans[v].get(k)!=source[v].get(k) for k in keys) for v in plans):return fail('frontend_plan_changed')
    certs=recommendation['certificates']
    if any(not certs.get(v,{}).get('available') for v in plans):return fail('missing_paired_hold_certificate')
    if {r['sample_id'] for r in pending}.intersection(r['sample_id'] for r in proposal.values()):
        return fail('proposal_job_already_published')
    frontend=sum(costs.get(k,0.) for k in ('input_s','output_s','planner_reserve_s','handoff_s'))
    load=sum(sum(profile[r['model']]['segment_estimates_s'])/r['requested_period_s'] for r in proposal.values())
    if load>.95+1e-10 or sum(frontend/r['requested_period_s'] for r in proposal.values())>.95+1e-10:
        return fail('fresh_periodic_load_exceeds_capacity')
    checked={v:dict(row) for v,row in proposal.items()}
    for v,row in checked.items():
        if rates_only and row['model']!=plans[v]['model']:return fail('rates_only_rejects_model_change')
        # Subtracting two absolute clocks can represent a legal 20ms interval
        # a few picoseconds below .020. Normalize before native installation.
        if abs(row['requested_period_s']-.020)<=1e-9:row['requested_period_s']=.020
        row['deadline_s']=min(row['deadline_s'],certs[v]['deadline_s'])
        row['nominal_deadline_s']=row['planned_release_s']+row['requested_period_s']
        if row['deadline_s']<=began or row['planned_release_s']<=began+budget_s:return fail('insufficient_review_or_release_time')
    # Already missed pending jobs still consume every remaining stage. They
    # cannot make the new group's forecast look repaired retroactively.
    missed={r['sample_id'] for r in pending if 'command_completed_s' not in r and r['deadline_s']<=began}
    reserved={**costs,'frontend_reserved_until_s':max(began+budget_s,costs.get('frontend_reserved_until_s',began))}
    feasible,finish,_=simulate_pipeline(began,list(checked.values()),pending,profile,reserved,
        policy='edf',inflight_limit=inflight_limit,missed_deadline_ids=missed)
    if time.perf_counter()-began>=budget_s:return fail('review_budget_exhausted')
    if not feasible:return fail('fresh_pipeline_infeasible')
    for row in [*checked.values(),*pending]:
        if 'command_completed_s' in row or row['sample_id'] in missed:continue
        deadline=min(row['deadline_s'],row['planned_release_s']+row.get('input_age_limit_s',float('inf')))
        if finish.get(row['sample_id'],float('inf'))+costs.get('completion_guard_s',0.)>deadline+1e-10:return fail('fresh_completion_exceeds_deadline')
    guard=io.guarded_batch(recommendation['native_basis'],checked,recommendation['control_tokens'],end_s,
        min(began+budget_s,min(r['planned_release_s'] for r in checked.values())),commit=commit)
    return {**result,**guard,'review_ms':1000*(time.perf_counter()-began),
        **({'installed_proposal':checked} if guard.get('installed') else {}),
        'historical_misses':sorted(missed),'historical_deadlines_repaired':False,
        'finite_pipeline_forecast':True,'dry_run':not commit}
