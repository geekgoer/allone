"""Paired real application of reviewed rates, with fixed models and EDF.

Static selected configuration, legacy dynamic control, and legacy + guarded
joint-rate assistance. This is NOT an exact FALCON/SVM reproduction claim.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys

from .decision_dataset import ROOT,atomic_json,sha256
from .dynamic_probe_common import read_json
from .fair_static_dynamic import MODEL_BANK,case_for
from .fly_convex_policy import load_airsim_client
from .framework_cpu_affinity import CpuPlacement
from .framework_runtime import ControllerPool
from .framework_runtime import simulate_pipeline
from .framework_runtime_worker import RuntimeServer
from .framework_runtime_editor import read_background_throttle
from .heterogeneous_motivation import create_world,execute,make_layout
from .joint_contract_shadow import JointDecisionShadow
from .controller_effect import reference_spec
from .probe_seven_continuous_timing import matched_initials
from .run_joint_contract_shadow import worker_core
from .run_native_runtime import timing


BASE=ROOT/'controller_architecture_study/results'


def estimate_periods(config,profile,costs,requested_ms=None):
    """Estimate response from phased and synchronized recurring releases."""
    from .heterogeneous_load_guard import resource_loads
    models=config['models']
    names=config.get('priority_order',list(models))
    compute={m:sum(row['segment_estimates_s']) for m,row in profile.items()}
    periods=[requested_ms] if requested_ms is not None else range(20,81,5)
    assumptions={**costs,'periodic_sampling':True,'eager_completion':True,'frontend_protection':True,
        'management_period_s':.010,'management_budget_s':.003,'io_slots':len(names)}
    limit=config.get('max_inflight',len(names))
    result={}
    for scope, bounds in [('planning',assumptions),('observed_stage_maxima',{**assumptions,**costs.get('tail_costs',{})})]:
        attempts=[]
        for period_ms in periods:
            period=period_ms/1000.
            demands=resource_loads(dict.fromkeys(names,period),models,compute,bounds)
            phases=[]
            if max(demands.values())<=.95:
                for synchronized in (False,True):
                    rows=[]
                    for priority,v in enumerate(names):
                        offset=0. if synchronized else config.get('release_offsets_s',{}).get(v,priority*period/len(names))
                        for cycle in range(3):
                            release=offset+cycle*period
                            rows.append(dict(vehicle=v,sample_id=f'{v}:{cycle}',model=models[v],priority=priority,
                                planned_release_s=release,requested_period_s=period,deadline_s=release+period,
                                fixed_period_slot=True,serial_vehicle=True))
                    passed, finish, _=simulate_pipeline(0.,rows,[],profile,bounds,inflight_limit=limit)
                    guard=bounds.get('completion_guard_s',0.)
                    responses=[1000*(finish[r['sample_id']]-r['planned_release_s']+guard)
                        for r in rows if r['sample_id'] in finish]
                    phases.append(dict(synchronized=synchronized,passed=passed and len(finish)==len(rows),
                        forecast_max_ms=max(responses,default=None),completed=len(finish),jobs=len(rows)))
            passed=bool(phases and phases[0]['passed'])
            attempts.append(dict(period_ms=period_ms,passed=passed,loads=demands,phases=phases,
                synchronized_stress_passed=bool(phases and phases[1]['passed'])))
            if passed:break
        result[scope]=dict(recommended_ms=next((r['period_ms'] for r in attempts if r['passed']),None),
            attempts=attempts)
    return dict(**result,is_WCRT=False,scope='empirical recurring-release forecast, three releases per vehicle',
        response_origin='planned release',fixed_models=models)


def check_static_baseline(report, requests, traces, turn_start_s):
    """Use each vehicle's observed simulation time to delimit the first turn."""
    origin=report['wall_origin_s']
    per_vehicle={}
    for vehicle,samples in traces.items():
        cutoff=next((row['sample_s'] for row in samples if row['t']>=turn_start_s),None)
        if cutoff is None or cutoff<=origin:
            per_vehicle[vehicle]={'passed':False,'reason':'missing_turn_start_observation'}
            continue
        jobs=[r for r in requests if r['vehicle']==vehicle]
        due=[r for r in jobs if r['nominal_deadline_s']<=cutoff]
        misses=[r['sample_id'] for r in due if r.get('status')!='applied'
            or r.get('rpc_return_s',float('inf'))>r['nominal_deadline_s']]
        acks=sorted(r['rpc_return_s'] for r in jobs if r.get('status')=='applied'
            and origin<=r.get('rpc_return_s',float('inf'))<=cutoff)
        points=[origin,*acks,cutoff]
        gap=max(end-start for start,end in zip(points,points[1:]))
        per_vehicle[vehicle]=dict(passed=bool(due and acks and not misses and gap<=.080+1e-9),
            jobs_due=len(due),missed_ids=misses,max_command_gap_ms=1000*gap)
    passed=bool(report['initial_taskset_forecast_passed'] and report['physical_task_passed']
        and report['collisions']==0 and report['period_deadline_metrics']['first_published_periods_verified']
        and per_vehicle and per_vehicle.keys()==report['vehicles'].keys()
        and all(row['passed'] for row in per_vehicle.values()))
    return dict(passed=passed,turn_start_sim_s=turn_start_s,per_vehicle=per_vehicle)


def run(output,boundary_report,seed=6,ranker_path=None,*,period_drop=False,period_ms=None,period_plan_only=False,
        timing_source=None,light_vehicles=(),vehicle_count=None,gate_width_m=None,
        staggered_turns=False,static_only=False,joint_only=False,spread_releases=False,require_static_baseline=False,
        trace_frontend=False):
    boundary=read_json(boundary_report)
    if not boundary['complete']:raise ValueError('finish and audit the boundary collection before application')
    output.mkdir(parents=True,exist_ok=False)
    if not period_plan_only:load_airsim_client()
    sys.setswitchinterval(.0005)
    origin=BASE/'fair_pool_search_20261028_v1'
    config=read_json(origin/'strengthened/selection.json')['configuration'];source=read_json(origin/'plan.json')
    case=case_for(seed)
    if vehicle_count is not None:
        if not 1<=vehicle_count<=case['count']:
            raise ValueError('vehicle count is outside the selected scenario')
        active=set(list(case['layouts'])[:vehicle_count])
        case.update(count=vehicle_count,layouts={v:r for v,r in case['layouts'].items() if v in active})
        for key in ('models','periods_ms','release_offsets_s'):
            if key in config:config[key]={v:r for v,r in config[key].items() if v in active}
        config['priority_order']=[v for v in config['priority_order'] if v in active]
        config['max_inflight']=min(config['max_inflight'],vehicle_count)
    case['virtual_extra_s']=0.
    if gate_width_m is not None:case['net_width_m']=gate_width_m
    turn_windows={}
    if staggered_turns:
        groups=((1,2),(4,),(6,7))
        for i,(vehicle,original) in enumerate(case['layouts'].items()):
            turns=groups[i] if i<len(groups) else ()
            layout=make_layout(i,seed,original['difficulty'],original['nominal_speed'],turn_segments=turns)
            layout['family']=('early','middle','late')[i] if i<len(groups) else 'straight'
            case['layouts'][vehicle]=layout
            turn_windows[vehicle]=[(seg['start_time'],seg['start_time']+seg['duration'])
                for seg in layout['segments'] if any(value!=0. for row in seg['coefficients'][1:] for value in row[1:])]
    if set(light_vehicles)-config['models'].keys():
        raise ValueError('light-model override refers to an unknown vehicle')
    for vehicle in light_vehicles:
        config['models'][vehicle]='plain_mlp_h2_w128_seed0'
    if spread_releases:
        names=config['priority_order']
        config['release_offsets_s']={v:i*period_ms/(1000*len(names)) for i,v in enumerate(names)}
    prior=origin/'strengthened/search_progress.json'
    profile_path=BASE/'joint_contract_shadow_20261030_v3/baseline/report.json'
    profile=read_json(profile_path)['worker_profile']
    costs=read_json(BASE/'joint_runtime_repair_20261029_v3/pipeline_costs.json')['costs']
    if timing_source is not None:
        from .heterogeneous_deadlines import freeze_costs
        costs=freeze_costs(timing_source,output/'recalibrated_pipeline.json','p95',native=True)['costs']
    if period_drop:
        costs.setdefault('period_release_dispatch_s',.001)
        costs['actuation_s']=max(costs['actuation_s'],costs.get('tail_costs',{}).get('actuation_s',0.))
        costs['command_s']=max(costs['command_s'],costs['actuation_s'])
    compute={m:sum(r['segment_estimates_s']) for m,r in profile.items()}
    ranker=ranker_path or BASE/'controller_effect_repair_20261031_v2/ranker/ranker.npz'
    recovery=read_json(Path(__file__).with_name('configs')/'recovery_budget_prototype_v1.json')
    order=['static_selected']*2 if static_only else ['static_selected','legacy_dynamic','reviewed_rates','reviewed_rates','legacy_dynamic','static_selected']
    if joint_only:order=['static_selected','reviewed_rates','reviewed_rates','static_selected']
    if require_static_baseline:order=['static_selected','static_selected','reviewed_rates','reviewed_rates']
    turn_start=min((start for windows in turn_windows.values() for start,_ in windows),default=None)
    files=[Path(__file__).with_name(n) for n in ('run_reviewed_rate_trial.py','joint_contract_shadow.py',
        'joint_commit_guard.py','joint_contract_optimizer.py','joint_contract_recovery.py','held_control_safety.py',
        'framework_native_io.py','framework_releases.py','framework_runtime_worker.py','framework_cpu_xpu.py',
        'framework_runtime_wire.py','controller_effect.py',
        'probe_seven_continuous_timing.py','heterogeneous_motivation.py','framework_runtime.py',
        'heterogeneous_load_guard.py','heterogeneous_deadlines.py')]
    frozen=output/'source';frozen.mkdir()
    for p in files:shutil.copy2(p,frozen/p.name)
    plan=dict(order=order,seed=seed,configuration=deepcopy(config),models_fixed_across_arms=True,
        active_vehicles=list(case['layouts']),net_width_m=case['net_width_m'],staggered_turns=staggered_turns,
        turn_windows_s=turn_windows,spread_releases=spread_releases,joint_rates_exclusive=joint_only,
        require_static_baseline=require_static_baseline,trace_frontend=trace_frontend,
        scheduler='two-FC-segment EDF in all arms; deadlines can differ',
        same_base_costs_in_all_arms=True,extra_management_costs_accounted=True,profile=profile,costs=costs,compute_costs=compute,
        profile_sha256=sha256(profile_path),ranker_sha256=sha256(ranker),boundary_sha256=sha256(boundary_report),
        boundary_counts=boundary['counts_by_split'],source_sha256={str(p):sha256(p) for p in files},
        controller_provenance=source['provenance'],light_vehicles=list(light_vehicles),external_impulse=False,initial_tracking_offset=False,
        virtual_prefix_disabled=True,hard_realtime_certified=False,
        scope='functional repeated paired pilot; reviewed rates supplement the existing dynamic policy; not exact FALCON')
    atomic_json(output/'plan.json',plan)
    placement=CpuPlacement();rows=[];first=None;static_gate_passed=None
    try:
        with placement:
            cpu=worker_core(placement)
            if not period_plan_only and read_background_throttle()['background_throttle_enabled']:
                raise RuntimeError('disable editor background throttle')
            pool=ControllerPool(model_names=MODEL_BANK)
            if pool.provenance()!=source['provenance']:raise RuntimeError('existing DNN source changed')
            # Freeze ONE host/XPU dispatch-aware profile before any flight;
            # all arms use it, instead of timing a different interface as free.
            sampler=RuntimeServer('edf',placement.layout['inference_cpus'],
                scheduler_cpus=placement.layout['frontend_cpus'],resident_worker=True,model_names=MODEL_BANK)
            try:
                if sampler.provenance!=pool.provenance():raise RuntimeError('CPU/XPU model provenance changed')
                profile=deepcopy(sampler.profile)
                plan['cpu_xpu_placement']=sampler.placement
            finally:sampler.close()
            compute={m:sum(r['segment_estimates_s']) for m,r in profile.items()}
            plan.update(profile=profile,compute_costs=compute,
                profile_scope='fresh CPU/XPU segment roundtrip calibration; shared by all arms; not WCET',
                historical_profile_sha256=plan.pop('profile_sha256'),
                cpu_management_service=dict(period_s=.010,budget_s=.003,is_hard_bound=False))
            if period_drop:
                estimate=estimate_periods(config,profile,costs,period_ms)
                atomic_json(output/'period_forecast.json',estimate)
                selected=estimate['observed_stage_maxima']['recommended_ms']
                if selected is None:
                    selected=estimate['planning']['recommended_ms']
                print('PERIOD FORECAST',output/'period_forecast.json','empirical starting period ms',selected,flush=True)
                atomic_json(output/'plan.json',plan)
                if period_plan_only:return
                if selected is None:
                    raise RuntimeError('No initial period passed the empirical forecast. Inspect period_forecast.json.')
                config['periods_ms']=dict.fromkeys(config['models'],selected)
                plan.update(configuration=deepcopy(config),period_policy='period_drop',
                    period_forecast_scope='observed_stage_maxima' if estimate['observed_stage_maxima']['recommended_ms']
                    is not None else 'planning',initial_period_ms=selected)
            atomic_json(output/'plan.json',plan)
            world=create_world(case,True)
            geometry={v:dict(net_width=world.net_width,gates=[dict(index=g.index,center=g.center.tolist(),
                orientation=g.orientation.tolist()) for g in world.gates[v]],
                reference=reference_spec(world.windows[v],world.phases[v])) for v in world.vehicles}
            try:
                for index,arm in enumerate(order):
                    if require_static_baseline and index==2:
                        static_gate_passed=all(row['static_baseline_gate']['passed'] for row in rows)
                        print('STATIC BASELINE GATE', 'PASS' if static_gate_passed else 'FAIL; joint flights withheld',flush=True)
                        if not static_gate_passed:break
                    name=f'{index+1:02d}_{arm}';observer=None
                    if arm=='reviewed_rates':
                        observer=JointDecisionShadow(profile,costs,prior,cpu,output/(name+'_planner'),geometry=geometry,
                            recover_after_miss=True,effect_path=ranker,fresh_review=True,apply_rates=True)
                    try:
                        arm_config={**config,'joint_rates_exclusive':bool(joint_only and observer),
                            'require_initial_forecast':require_static_baseline}
                        result=execute(world,pool,placement,case,arm_config,output,name,dynamic=arm!='static_selected',
                            physical=True,anticipatory=arm!='static_selected',recovery_models=dict(config['models']),
                            guard_costs=compute,pipeline_costs=costs,global_admission=arm!='static_selected',
                            frontend_protection=True,recovery_budget=recovery,planning_profile=profile,io_backend='cpp',
                            runtime_repair=True,joint_adaptation=arm!='static_selected',scheduler='edf',joint_shadow=observer,
                            deadline_policy='period_drop' if period_drop else 'observe',trace_frontend=trace_frontend)
                    finally:
                        if observer is not None:observer.close()
                    directory=output/name;initial=read_json(directory/'initial_state.json')
                    if first is not None and not matched_initials(first,initial):raise RuntimeError('paired initial states differ')
                    first=initial;requests=read_json(directory/'requests.json');flight=read_json(directory/'report.json')
                    tagged=[r for r in requests if 'joint_commit_generation' in r]
                    delivered=[r for r in tagged if r.get('status')=='applied' and 'rpc_return_s' in r]
                    row=dict(arm=arm,name=name,physical_result=result,timing=timing(directory),
                        tagged_requests=len(tagged),tagged_controls_delivered=len(delivered),
                        changed_rate_controls_delivered=sum(r.get('joint_rate_changed',False) for r in delivered),
                        changed_period_controls_delivered=sum(r.get('joint_period_changed',False) for r in delivered),
                        changed_release_controls_delivered=sum(r.get('joint_release_changed',False) for r in delivered),
                        changed_deadline_controls_delivered=sum(r.get('joint_deadline_changed',False) for r in delivered),
                        models_match_frozen=all(r['model']==config['models'][r['vehicle']] for r in requests),
                        conditional_budget_late_count=len(flight['late_jobs']),
                        period_deadline_metrics=flight.get('period_deadline_metrics',{}),
                        initial_taskset_forecast_passed=flight['initial_taskset_forecast_passed'],
                        control_window_covered=flight['full_flight_completed'],sensor_window_covered=flight['sensor_window_covered'],
                        planner=read_json(output/(name+'_planner')/'report.json') if observer else None)
                    stops=flight.get('retired_vehicles',{})
                    first_collision=min((r['stopped_wall_s'] for r in stops.values()),default=float('inf'))
                    row['changed_rate_controls_before_first_collision']=sum(r.get('joint_rate_changed',False)
                        and r['rpc_return_s']<first_collision for r in delivered)
                    row['host_cpu_segment_dispatches']=len(read_json(directory/'cpu_segment_dispatches.json'))
                    if require_static_baseline and arm=='static_selected':
                        row['static_baseline_gate']=check_static_baseline(flight,requests,
                            read_json(directory/'traces.json'),turn_start)
                    if not row['models_match_frozen']:raise RuntimeError('a supposedly rate-only arm changed models')
                    rows.append(row);atomic_json(output/'progress.json',rows)
                    print('APPLICATION',name,'collisions',flight['collisions'],'tagged_delivered',len(delivered),
                        'J_measured',result['short_measured_J'],flush=True)
            finally:atomic_json(output/'cleanup.json',world.close())
        summary=dict(complete=len(rows)==len(order),same_initials=True,rows=rows,
            static_gate_passed=static_gate_passed,joint_stage_started=any(r['arm']=='reviewed_rates' for r in rows),
            total_installed_groups=sum((r['planner'] or {}).get('installed_groups',0) for r in rows),
            tagged_controls_delivered=sum(r['tagged_controls_delivered'] for r in rows),
            changed_rate_controls_delivered=sum(r['changed_rate_controls_delivered'] for r in rows),
            changed_period_controls_delivered=sum(r['changed_period_controls_delivered'] for r in rows),
            changed_release_controls_delivered=sum(r['changed_release_controls_delivered'] for r in rows),
            recommendations_actually_executed=any(r['tagged_controls_delivered'] for r in rows),
            models_fixed=True,scope=plan['scope'],hard_realtime_certified=False)
        atomic_json(output/'summary.json',summary)
        print('Application summary:',output/'summary.json','installed',summary['total_installed_groups'],
            'delivered',summary['tagged_controls_delivered'],flush=True)
    finally:atomic_json(output/'cpu_placement.json',placement.report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--boundary-report',type=Path,required=True)
    p.add_argument('--seed',type=int,default=6);p.add_argument('--ranker',type=Path)
    p.add_argument('--period-drop',action='store_true')
    p.add_argument('--period-plan-only',action='store_true')
    p.add_argument('--period-ms',type=float,metavar='20..80')
    p.add_argument('--light-vehicles',nargs='+',default=(),help='Use the 2x128 controller for these vehicles in every arm')
    p.add_argument('--vehicles',type=int,choices=range(1,11),metavar='1..10')
    p.add_argument('--gate-width-m',type=float)
    p.add_argument('--staggered-turns',action='store_true')
    p.add_argument('--static-only',action='store_true')
    p.add_argument('--joint-only',action='store_true',help='Compare static against reviewed rates without legacy rate updates')
    p.add_argument('--spread-releases',action='store_true',help='Spread first releases uniformly across --period-ms')
    p.add_argument('--require-static-baseline',action='store_true')
    p.add_argument('--trace-frontend',action='store_true')
    p.add_argument('--timing-source',type=Path,help='Archived native flight directory containing requests.json and analyses.json')
    a=p.parse_args()
    if a.period_ms is not None and not (a.period_drop or a.period_plan_only):
        p.error('--period-ms requires --period-drop or --period-plan-only')
    if a.period_ms is not None and not 20<=a.period_ms<=80:
        p.error('--period-ms must be between 20 and 80')
    if a.gate_width_m is not None and not 0<a.gate_width_m<float('inf'):
        p.error('--gate-width-m must be finite and positive')
    if a.joint_only and (a.static_only or not (a.period_drop or a.period_plan_only)):
        p.error('--joint-only requires period-drop and cannot be combined with --static-only')
    if a.spread_releases and a.period_ms is None:
        p.error('--spread-releases requires an explicit --period-ms')
    if a.require_static_baseline and not (a.joint_only and a.staggered_turns):
        p.error('--require-static-baseline requires --joint-only and --staggered-turns')
    run(a.output,a.boundary_report,a.seed,a.ranker,period_drop=a.period_drop or a.period_plan_only,
        period_ms=a.period_ms,period_plan_only=a.period_plan_only,timing_source=a.timing_source,
        light_vehicles=a.light_vehicles,vehicle_count=a.vehicles,gate_width_m=a.gate_width_m,
        staggered_turns=a.staggered_turns,static_only=a.static_only,joint_only=a.joint_only,
        spread_releases=a.spread_releases,require_static_baseline=a.require_static_baseline,
        trace_frontend=a.trace_frontend)
