"""Short heterogeneous plants, finite static search and explicitly cached scores."""

import argparse
from copy import deepcopy
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import sys

import numpy as np

from falcon_paper_fidelity.motion_primitive import QuinticPolynomial
from .decision_dataset import ROOT,atomic_json,sha256
from .dynamic_probe_common import read_json
from .framework_cpu_affinity import CpuPlacement
from .framework_runtime import ControllerPool,MODELS
from .framework_runtime_editor import read_background_throttle
from .framework_runtime_world import FrameworkWorld
from .inflight_sensing import InflightSensing
from .probe_seven_continuous_timing import flight,matched_initials


PERIODS=(20.,30.,40.,50.,60.,80.)


def make_layout(vehicle_index,seed=0,difficulty=1.,speed=5.):
    if not np.isfinite([difficulty,speed]).all() or min(difficulty,speed)<=0.:
        raise ValueError('finite positive difficulty and speed required')
    rng=np.random.default_rng(seed*100+vehicle_index)
    family='continuous' if vehicle_index<3 else 'smooth' if vehicle_index<8 else 'late'
    amplitude=.75*difficulty*(.90+.10*rng.random())
    onset=1 if family=='continuous' else min(7,5+(vehicle_index-8)) if family=='late' else 99
    p=np.array([0.,0.,-4.]);v=np.array([speed,0.,0.]);zero=np.zeros(3)
    segments=[];gates=[];clock=0.
    for k in range(9):
        base_dt=2.1 if k in (0,8) else 1.26
        dt=base_dt*5./speed
        q=p.copy();q[0]+=5.*base_dt*(.5 if k==8 else 1.)
        if k>=onset and k<8:
            q[1]=amplitude*(-1. if (k+vehicle_index)%2 else 1.)
            q[2]=-4.+.12*np.sin(k+vehicle_index)
        end_v=zero if k==8 else np.array([speed,0.,0.])
        primitive=QuinticPolynomial(p,v,zero,q,end_v,zero,dt)
        segments.append(dict(start_time=clock,duration=dt,coefficients=primitive.coefficients.tolist(),heading_orientation=[1.,0.,0.,0.]))
        if k<8:gates.append(dict(index=k+1,center=q.tolist(),orientation=[1.,0.,0.,0.]))
        p,v=q,end_v;clock+=dt
    return dict(segments=segments,gates=gates,duration=clock,family=family,
        seed=seed,nominal_speed=speed,difficulty=difficulty,
        duration_fitting='fixed durations; no hidden slowdown')


def make_case(seed=0,difficulty=1.,speed=5.,count=10):
    layouts={f'Drone{i+1}':make_layout(i,seed,difficulty,speed) for i in range(count)}
    duration=np.ceil((next(iter(layouts.values()))['duration']+1.68)/.42-1e-9)*.42
    onset=min((s['start_time'] for layout in layouts.values() if layout['family']=='late'
        for s in layout['segments'] if abs(np.asarray(s['coefficients'])[1,3:]).max()>1e-9),default=duration+2.)
    cache_end=np.floor(min(4.62*5./speed,onset-2.05)/.42+1e-9)*.42
    cache_start=np.ceil(2.1*5./speed/.42-1e-9)*.42
    if cache_end-cache_start<3*.42-1e-9:raise ValueError('speed leaves too little independent prefix for caching')
    return dict(id=f'heterogeneous_n{count}_s{seed}_d{difficulty:g}_v{speed:g}',
        layouts=layouts,count=count,duration_s=float(duration),net_width_m=1.8,phase_s=0.,phase_span_s=0.,
        cache_interval_s=[float(cache_start),float(cache_end)],virtual_extra_s=round(84.*5./speed/.42)*.42,
        external_impulse=False,initial_tracking_offset=False,
        speed_parameterization='same gates and geometry; scale trajectory time, velocity and acceleration',
        prefix_estimate='repeat representative measured window costs; not a physically flown prefix')


def score_windows(trace,duration):
    if not trace or trace[0]['t']>.025 or trace[-1]['t']<duration:return None
    grid=np.arange(0.,duration+1e-8,.02)
    values=np.interp(grid,[r['t'] for r in trace],[r['state_cost'] for r in trace])
    n=len(values)//21
    return values[:n*21].reshape(n,21).sum(axis=1)


def cached_score(trace,case):
    windows=score_windows(trace,case['duration_s'])
    if windows is None:return None
    first,last=(round(t/.42) for t in case['cache_interval_s'])
    cached=windows[first:last]
    if len(cached)!=last-first:raise ValueError('cache interval not covered')
    extra=round(case['virtual_extra_s']/.42)
    total_cost=float(windows.sum()+extra*cached.mean())
    blocks=[float(a.mean()) for a in np.array_split(cached,3)]
    return dict(J_hat=(len(windows)+extra)/total_cost if total_cost>0 else None,
        short_J=float(len(windows)/windows.sum()) if windows.sum()>0 else None,
        actual_windows=len(windows),virtual_windows=extra,actual_cost_sum=float(windows.sum()),
        cached_mean_cost=float(cached.mean()),cache_block_means=blocks,
        cached_segment_physically_repeated=False,full_mission_safety_inferred=False)


def initial_configuration(vehicles):
    small='plain_mlp_h9_w512_seed0';large='plain_mlp_h9_w1024_seed0'
    return dict(models={v:large if int(v[5:])<=3 or int(v[5:])>=9 else small for v in vehicles},
        periods_ms={v:40. if int(v[5:])<=3 else 60. if int(v[5:])>=9 else 80. for v in vehicles},
        priority_order=list(vehicles),release_offsets_s={v:i*.020/len(vehicles) for i,v in enumerate(vehicles)},max_inflight=min(3,len(vehicles)))


def configuration_key(config):
    return json.dumps(config,sort_keys=True,separators=(',',':'))


def propose(best,step,rng,seen,model_names=MODELS):
    vehicles=list(best['models'])
    for attempt in range(100):
        candidate=deepcopy(best);v=vehicles[int(rng.integers(len(vehicles)))]
        kind=(step+attempt)%3
        if kind==0:
            choices=[p for p in PERIODS if p!=candidate['periods_ms'][v]]
            candidate['periods_ms'][v]=float(rng.choice(choices))
        elif kind==1:
            choices=[m for m in model_names if m!=candidate['models'][v]]
            candidate['models'][v]=str(rng.choice(choices))
        else:
            a,b=rng.choice(len(vehicles),2,replace=False)
            order=candidate['priority_order'];order[a],order[b]=order[b],order[a]
        if configuration_key(candidate) not in seen:return candidate
    raise RuntimeError('could not find a new bounded-search neighbor')


def capacity_estimate(config,pool):
    # Frozen explicit empirical envelopes. A utilization screen is NOT RTA/WCET.
    rate=sum(1000/p for p in config['periods_ms'].values())
    costs={m:sum(p['segment_estimates_s']) for m,p in pool.profile.items()}
    compute=sum(costs[config['models'][v]]*1000/p
        for v,p in config['periods_ms'].items())
    frontend=.0010*rate
    io=.0035*rate/2
    return dict(admitted=compute<=.95,
        compute_load=compute,frontend_load=frontend,io_load=io,is_response_time_guarantee=False,
        frontend_cost_s=.0010,io_cost_s=.0035,compute_cost_s=costs,
        screening='only measured serialized CPU compute envelope; frontend/IO figures are diagnostics, not an exclusion rule')


def strong_starts(vehicles):
    base=initial_configuration(vehicles)
    danger=deepcopy(base)
    for v in vehicles:
        if int(v[5:])<=3 or int(v[5:])>=9:danger['periods_ms'][v]=20.
    uniform=deepcopy(base)
    uniform['models']={v:'plain_mlp_h9_w1024_seed0' for v in vehicles}
    uniform['periods_ms']={v:20. for v in vehicles}
    medium=deepcopy(uniform);medium['periods_ms']={v:30. for v in vehicles}
    return [base,danger,uniform,medium]


def create_world(case,physical):
    return FrameworkWorld(next(iter(case['layouts'].values())),0.,case['count'],case['net_width_m'],
        phase_span=0.,task_duration=case['duration_s'],layouts=case['layouts'],spawn_gates=physical)


def execute(world,pool,placement,case,config,root,name,*,dynamic=False,physical=True,reserve=.020,guard_costs=None,
        anticipatory=False,recovery_model=None,pipeline_costs=None,deadline_policy='observe',recovery_models=None,
        scheduler=None,frontend_order=None,global_admission=False,frontend_protection=True,recovery_budget=None,planning_profile=None,
        io_backend='python',runtime_repair=None,joint_adaptation=False,frontend_aging=None,joint_shadow=None):
    directory=root/name
    sensing=InflightSensing(world.vehicles,world.calibration,placement.layout['io_cpus'][1],None)
    settings=deepcopy(config)
    settings.update(deadline_policy=deadline_policy,pipeline_costs=pipeline_costs,release_contract='committed_calendar_v1',
        global_admission=global_admission,frontend_protection=frontend_protection,io_backend=io_backend,joint_adaptation=joint_adaptation)
    if frontend_aging is not None:settings['frontend_aging']=frontend_aging
    if runtime_repair is not None:
        settings.update(eager_completion=runtime_repair,queue_policy='latest' if runtime_repair else 'all',
            recover_after_miss=runtime_repair)
    if dynamic and recovery_budget is not None:settings['recovery_budget']=deepcopy(recovery_budget)
    if planning_profile is not None:settings['planning_profile']=deepcopy(planning_profile)
    if dynamic:
        # Keep each selected model to isolate rate/scheduling before mode changes.
        settings['mode_policy']={v:dict(cruise=m,recovery=m,contract='relative_preview_v2') for v,m in settings['models'].items()}
        if anticipatory:
            for mode in settings['mode_policy'].values():mode['contract']='relative_preview_v3'
            if recovery_model:settings['mode_policy']['Drone9']['recovery']=recovery_model
            for v,m in (recovery_models or {}).items():settings['mode_policy'][v]['recovery']=m
        if guard_costs:settings['compute_cost_s']=guard_costs
    print('HETEROGENEOUS',name,'physical_gates',physical,'dynamic',dynamic,flush=True)
    report=flight(world,pool,60.,dict(id=name,critical_index=-1),
        'adaptive_twosided' if dynamic else 'static',directory,case['duration_s'],placement.layout,
        sensing=sensing,response_reserve_s=reserve,frontend_order=frontend_order or ('deadline' if dynamic else 'category'),
        scheduler=scheduler,
        resident_worker=True,model=MODELS[0],loop_config=settings,
        stop_on_collision=physical,collision_policy='vehicle',joint_shadow=joint_shadow)
    if report['error'] or report['cleanup_errors']:raise RuntimeError(f"trial failure: {report['error']} {report['cleanup_errors']}")
    traces=read_json(directory/'traces.json')
    metrics={v:cached_score(t,case) if (not physical or not report['vehicles'][v]['failure']) else None for v,t in traces.items()}
    complete=report['sensor_window_covered'] and all(v['clock_valid'] for v in report['vehicles'].values()) and all(metrics.values())
    result=dict(name=name,physical_gates=physical,nominal_profile_without_obstacle_contact=not physical,
        physical_passed=report['success'] if physical else None,collisions=report['collisions'],retired=report['retired_vehicles'],
        score_kind='cached representative-cost estimate',J_hat=sum(m['J_hat'] for m in metrics.values()) if complete else None,
        short_measured_J=sum(m['short_J'] for m in metrics.values()) if complete else None,
        complete=complete,per_vehicle=metrics,nominal_misses=len(report['nominal_late_jobs'])+len(report['nominal_undelivered_deadlines'])+report['skipped_releases'],
        actual_deadline_misses=len(set(report['late_jobs'])|set(report['undelivered_deadlines'])|set(report['actual_deadline_failure_ids'])),response_ms=report['response_ms'],
        active_initial=case['count'],active_final=report['active_vehicle_count_final'],simulation_paused=world.client.simIsPause())
    result['shared_compute_guard_enabled']=bool(dynamic and guard_costs)
    result.update(deadline_policy=deadline_policy,deadline_failures=report['deadline_failures'],
        admission_checks=report['admission_checks'],admission_rejections=report['admission_rejections'],
        hard_realtime_certified=False)
    atomic_json(directory/'fast_score.json',dict(case_id=case['id'],configuration=config,
        configuration_key=configuration_key(config),provenance=pool.provenance(),cpu_layout=placement.layout,
        cache_shared_load_key='entire configuration including all loops, periods, priorities, offsets and pipeline limit',result=result))
    print('RESULT',name,'J_hat',result['J_hat'],'collision',result['collisions'],'physical_passed',result['physical_passed'],
        'deadline_failures',len(result['deadline_failures']),'admission_rejections',result['admission_rejections'],flush=True)
    return result


def run_search(root,case,pool,placement,budget):
    world=create_world(case,False)
    rng=np.random.default_rng(123);seen=set();rows=[]
    best=initial_configuration(world.vehicles);best_score=-float('inf')
    starts=strong_starts(world.vehicles)
    try:
        for step in range(budget):
            if (root/'STOP').exists():break
            candidate=starts[step] if step<len(starts) else propose(best,step,rng,seen)
            seen.add(configuration_key(candidate))
            capacity=capacity_estimate(candidate,pool)
            if capacity['admitted']:
                result=execute(world,pool,placement,case,candidate,root,f'candidate_{step:02d}',physical=False)
                score=result['J_hat'] if result['complete'] else None
            else:result=None;score=None
            improved=score is not None and score>best_score
            if improved:best,best_score=deepcopy(candidate),score
            rows.append(dict(index=step,configuration=candidate,capacity=capacity,result=result,improved=improved))
            atomic_json(root/'search_progress.json',dict(evaluations=len(rows),rows=rows,best_configuration=best,best_J_hat=best_score if np.isfinite(best_score) else None))
        if not np.isfinite(best_score):raise RuntimeError('no valid admitted profile candidate')
        selection=dict(configuration=best,J_hat=best_score,calibration_case=case,rows=rows,evaluations=len(rows),
            compute_cost_s={m:sum(p['segment_estimates_s']) for m,p in pool.profile.items()},
            provenance=pool.provenance(),search='budgeted greedy single-coordinate/priority neighbor search',
            prediction_domain='real shared CPU nominal tracking without gate meshes; costs reweighted with measured prefix windows',
            exact_global_optimum=False,full_FALCON=False,full_long_mission_flown=False)
        atomic_json(root/'selection.json',selection)
        return selection
    finally:atomic_json(root/'profile_cleanup.json',world.close())


def refresh_cached_selection(path,root):
    """Re-score the SAME visited20 candidates; never use validation outcomes."""
    from .heterogeneous_load_guard import recover_profile_costs
    selection=deepcopy(read_json(path))
    case=selection['calibration_case']
    first=next(iter(case['layouts'].values()))
    current=make_case(speed=first['nominal_speed'])['cache_interval_s']
    if case['cache_interval_s']!=current:
        case['cache_interval_s']=current
        for row in selection['rows']:
            if row['result'] is None:continue
            traces=read_json(path.parent/row['result']['name']/'traces.json')
            scores={v:cached_score(t,case) for v,t in traces.items()}
            row['result']['per_vehicle']=scores
            row['result']['J_hat']=sum(s['J_hat'] for s in scores.values()) if all(scores.values()) else None
        candidates=[r for r in selection['rows'] if r['result'] and r['result']['J_hat'] is not None]
        best=max(candidates,key=lambda r:r['result']['J_hat'])
        selection.update(configuration=best['configuration'],J_hat=best['result']['J_hat'])
    if 'compute_cost_s' not in selection:
        selection['compute_cost_s'],selection['cost_reconstruction']=recover_profile_costs(selection['rows'])
    selection.update(original_selection=str(path),original_selection_sha256=sha256(path),
        reweighting='same candidate pool; prefix cache ends before future reference reveals late turns; no validation result used')
    atomic_json(root/'selection.json',selection)
    return selection


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--difficulty',type=float,default=1.)
    parser.add_argument('--speed',type=float,default=5.)
    parser.add_argument('--width',type=float,default=1.8)
    parser.add_argument('--count',type=int,default=10)
    parser.add_argument('--budget',type=int,default=20)
    parser.add_argument('--selection',type=Path)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--static-only',action='store_true')
    parser.add_argument('--smoke-stop',action='store_true')
    parser.add_argument('--no-load-guard',action='store_true')
    parser.add_argument('--interactive',action='store_true')
    parser.add_argument('--anticipatory',action='store_true')
    parser.add_argument('--mode-calibration',type=Path)
    parser.add_argument('--pipeline-costs',type=Path)
    parser.add_argument('--deadline-policy',choices=('observe','enforce'),default='observe')
    args=parser.parse_args()
    if not 2<=args.count<=12 or not 1<=args.budget<=20:raise ValueError('count2..12 and budget1..20 required')
    root=args.output or ROOT/'controller_architecture_study/results'/('heterogeneous_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    root.mkdir(parents=True,exist_ok=False)
    case=make_case(args.seed,args.difficulty,args.speed,2 if args.smoke_stop else args.count)
    case['net_width_m']=args.width
    if args.smoke_stop:
        case['layouts']['Drone2']=make_layout(4,args.seed,args.difficulty,args.speed)
        case['layouts']['Drone1']['gates'][0]['center'][1]=.85
        case.update(duration_s=6.3,mechanism_fixture_only=True,
            fixture='offset first gate forDrone1 only; not a natural-flight motivation result')
    atomic_json(root/'case.json',case)
    if args.prepare_only:return
    mode_rows=read_json(args.mode_calibration/'summary.json') if args.mode_calibration else []
    eligible=[r['model'] for r in mode_rows if r.get('model') and r['success']]
    if args.mode_calibration and not eligible:raise ValueError('calibration contains no passing recovery mode; do not invent an admissible controller')
    names=tuple(dict.fromkeys((*MODELS,*eligible)))
    costs=read_json(args.pipeline_costs)['costs'] if args.pipeline_costs else None
    sys.setswitchinterval(.0005)
    placement=CpuPlacement()
    with placement:
        pool=ControllerPool(model_names=names)
        recovery_model=min(eligible,key=lambda m:sum(pool.profile[m]['segment_estimates_s'])) if eligible else None
        if args.mode_calibration:
            recorded=read_json(args.mode_calibration/'plan.json')
            domain=recorded['case'];route=domain['layouts']['Drone9']
            if (recorded['vehicle']!='Drone9' or not np.isclose(domain['net_width_m'],args.width)
                    or not np.isclose(route['nominal_speed'],args.speed) or not np.isclose(route['difficulty'],args.difficulty)):
                raise ValueError('recovery calibration does not match the saved route family/width/speed/difficulty')
            for m in eligible:
                if recorded['provenance']['models'][m]!=pool.provenance()['models'][m]:raise ValueError('calibrated model changed')
            if recorded['provenance']['normalization_sha256']!=pool.provenance()['normalization_sha256']:
                raise ValueError('calibration normalization changed')
            atomic_json(root/'mode_selection.json',dict(calibration=str(args.mode_calibration),eligible=eligible,
                selected=recovery_model,scope='Drone9 saved late-turn route at the recorded calibration conditions; empirical, not universal admissibility'))
        if read_background_throttle()['background_throttle_enabled']:raise RuntimeError('disable background editor throttling')
        atomic_json(root/'plan.json',dict(case=case,budget=args.budget,source_sha256={n:sha256(Path(__file__).with_name(n)) for n in
            ('heterogeneous_motivation.py','heterogeneous_load_guard.py','heterogeneous_prediction.py','heterogeneous_deadlines.py','framework_releases.py','probe_seven_continuous_timing.py','framework_runtime_worker.py','framework_runtime_world.py','safety_comparison_flight.py','safety_comparison_core.py','framework_runtime.py')},
            profile_gate_meshes=False,validation_gate_meshes=True,provenance=pool.provenance(),cpu_layout=placement.layout,
            anticipatory=args.anticipatory,deadline_policy=args.deadline_policy,
            pipeline_costs_source=str(args.pipeline_costs) if args.pipeline_costs else None,
            validation_seed_rule='calibration seed+1 unless separately selected case requested',FPGA_verified=False))
        frozen=root/'frozen_source';frozen.mkdir()
        for name in read_json(root/'plan.json')['source_sha256']:
            shutil.copy2(Path(__file__).with_name(name),frozen/name)
        if args.smoke_stop:
            world=create_world(case,True)
            try:
                config=initial_configuration(world.vehicles)
                config['models']={v:'plain_mlp_h9_w512_seed0' for v in world.vehicles}
                config['periods_ms']={v:40. for v in world.vehicles}
                result=execute(world,pool,placement,case,config,root,'collision_stop_check')
                traces=read_json(root/'collision_stop_check/traces.json')
                event=result['retired'].get('Drone1')
                continued=bool(event and traces['Drone2'][-1]['t']>event['observed_t_s']+1.
                    and np.linalg.norm(np.asarray(traces['Drone2'][-1]['position'])-traces['Drone2'][0]['position'])>10.)
                passed=bool(event and event['parked'] and 'Drone2' not in result['retired'] and continued and not result['simulation_paused'])
                atomic_json(root/'summary.json',dict(mechanism_fixture_only=True,passed=passed,
                    other_vehicle_continued=continued,result=result))
                if not passed:raise RuntimeError('single-vehicle collision stop check failed')
            finally:atomic_json(root/'validation_cleanup.json',world.close())
            return
        selection=refresh_cached_selection(args.selection,root) if args.selection else run_search(root,case,pool,placement,args.budget)
        actual=pool.provenance();actual['models']={m:actual['models'][m] for m in selection['provenance']['models']}
        if selection['provenance']!=actual:raise RuntimeError('selection models changed')
        guard_costs={**selection['compute_cost_s'],**{m:sum(pool.profile[m]['segment_estimates_s']) for m in eligible}}
        original=selection['calibration_case'];first=next(iter(original['layouts'].values()))
        if not (np.isclose(args.speed,first['nominal_speed']) and np.isclose(args.difficulty,first['difficulty'])
                and np.isclose(args.width,original['net_width_m'])):
            raise ValueError('changed speed/difficulty/width requires a new search; only validation seed may change')
        validation=make_case(args.seed+1,args.difficulty,args.speed,args.count)
        validation['net_width_m']=args.width
        config=selection['configuration']
        if set(config['models'])!=set(validation['layouts']):raise ValueError('selection vehicle set differs')
        world=create_world(validation,True)
        try:
            results={}
            for arm in (('static',) if args.static_only else ('static','dynamic')):
                if args.interactive:input(f'准备好观察{arm}后按Enter；碰撞仅停止对应无人机：')
                results[arm]=execute(world,pool,placement,validation,config,root,'validation_'+arm,dynamic=arm=='dynamic',
                    guard_costs=None if args.no_load_guard else guard_costs,anticipatory=args.anticipatory,
                    recovery_model=recovery_model,pipeline_costs=costs,deadline_policy=args.deadline_policy)
            match=True if args.static_only else matched_initials(read_json(root/'validation_static/initial_state.json'),read_json(root/'validation_dynamic/initial_state.json'))
            contact_split=bool(not args.static_only and results['static']['collisions'] and results['dynamic']['physical_passed'] and match)
            dynamic_clean=bool(not args.static_only and results['dynamic']['actual_deadline_misses']==0
                and (args.anticipatory or results['dynamic']['nominal_misses']==0) and not results['dynamic']['deadline_failures']
                and not results['dynamic']['admission_rejections'])
            atomic_json(root/'summary.json',dict(selection=str(args.selection or root/'selection.json'),validation_case=validation,
                results=results,initial_states_match=match,full_FALCON=False,
                contact_split_observed=contact_split,rescue=contact_split and dynamic_clean,
                strict_timing_passed=all(r['actual_deadline_misses']==0
                    and not r['deadline_failures'] and not r['admission_rejections'] for r in results.values())))
        finally:atomic_json(root/'validation_cleanup.json',world.close())
    atomic_json(root/'cpu_placement.json',placement.report)
    print('Summary:',root/'summary.json',flush=True)


if __name__=='__main__':main()
