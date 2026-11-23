"""Read-only management timing and applied-action audit for PS trial directories."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path


def rows(path):
    if not path.is_file():return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def stats(values):
    values=sorted(values)
    if not values:return {'count':0}
    def q(f):
        pos=f*(len(values)-1);low=int(pos);high=min(low+1,len(values)-1)
        return values[low]+(values[high]-values[low])*(pos-low)
    return dict(count=len(values),mean=sum(values)/len(values),p50=q(.5),p99=q(.99),max=values[-1])


def analyze(folder):
    requests=rows(folder/'requests.jsonl');plans=rows(folder/'plans.jsonl')
    phases=defaultdict(lambda:defaultdict(list));rejections=Counter();revalidated=0
    for row in plans:
        if row.get('event')!='obligation_plan':continue
        inst=row['install'];rejections[inst['status']]+=1
        revalidated+=bool(inst.get('revalidated',False))
        trace=row.get('management_timing',{})
        timed={'snapshot_lock_wait':trace.get('snapshot',{}).get('lock_wait'),
               'snapshot_lock_held':trace.get('snapshot',{}).get('lock_held'),
               'snapshot_build':trace.get('snapshot',{}).get('build'),
               'initial_planning':trace.get('planning'), 'log_object':trace.get('log_object'),
               'revalidation':inst.get('revalidation_timing')}
        install_timing=inst.get('timing',inst.get('first_install',{}).get('timing',{}))
        timed['install_lock_wait']=install_timing.get('lock_wait')
        timed['validate_install']=install_timing.get('validate_install')
        refresh=inst.get('refreshed_context',{}).get('timing',{})
        timed['refresh_lock_wait']=refresh.get('lock_wait')
        timed['refresh_lock_held']=refresh.get('lock_held')
        timed['refresh_proof_wait']=refresh.get('proof_wait')
        timed['refresh_snapshot_build']=refresh.get('build')
        previous=row.get('previous_log_enqueue')
        if previous:timed['previous_log_enqueue']=previous['timing']
        for name,value in timed.items():
            if value is not None:
                for clock in ['wall_ms','cpu_ms']:phases[name][clock].append(value[clock])
    delivered=[r for r in requests if r['status']=='delivered']
    active=[r for r in plans if r.get('event')=='obligation_plan']
    worst=max(active,key=lambda r:r['wall_ms']) if active else None
    stage_cpu=defaultdict(list);stage_wall=defaultdict(list)
    usage=defaultdict(list)
    for row in active:
        for name,stage in row.get('management_timing',{}).get('planning_stages',{}).items():
            stage_cpu[name].append(stage['cpu_ms']);stage_wall[name].append(stage['wall_ms'])
        for name,value in (row.get('management_timing',{}).get('planning_rusage') or {}).items():
            usage[name].append(value)
    return dict(run=folder.name,requests=len(requests),statuses=dict(Counter(r['status'] for r in requests)),
        actual_models=dict(Counter(str(r['model_id']) for r in delivered)),
        response_ms=stats([(r['command_return_us']-r['planned_release_us'])/1000 for r in delivered]),
        observed_command_deadline_misses=sum(r['command_return_us']>min(r['period_deadline_us'],r['actuation_deadline_us']) for r in delivered),
        manager_wall_ms=stats([r['wall_ms'] for r in active]),manager_cpu_ms=stats([r['cpu_ms'] for r in active]),
        install_statuses=dict(rejections),revalidated_installs=revalidated,
        triggers=dict(Counter(r.get('trigger','legacy_periodic') for r in active)),
        planning_stage_cpu_ms={name:stats(values) for name,values in stage_cpu.items()},
        planning_stage_wall_ms={name:stats(values) for name,values in stage_wall.items()},
        planning_rusage={name:stats(values) for name,values in usage.items()},
        skip_candidates=sum(bool(r.get('decision',{}).get('skipped')) for r in active),
        actual_planned_skips=sum(r['status']=='planned_skip' for r in requests),
        stage_timing={k:{clock:stats(values) for clock,values in clocks.items()} for k,clocks in phases.items()},
        worst_manager=None if worst is None else dict(wall_ms=worst['wall_ms'],cpu_ms=worst['cpu_ms'],
            action=worst['decision'].get('action'),checks=worst['decision'].get('checks'),install_status=worst['install']['status'],
            management_timing=worst.get('management_timing')),
        requests_sha256=hashlib.sha256((folder/'requests.jsonl').read_bytes()).hexdigest())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Output must be new')
    result=[analyze(folder) for folder in args.data]
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    for r in result:
        print(r['run'],'requests',r['requests'],'status',r['statuses'],'revalidated',r['revalidated_installs'],
              'skips',r['actual_planned_skips'],'manager',r['manager_wall_ms'])
