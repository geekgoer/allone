"""Summarize serialized offline profiles; no board/RPC access."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path


def stats(values):
    values=sorted(values)
    index=.99*(len(values)-1)
    lower=int(index)
    return dict(count=len(values),mean=sum(values)/len(values),p50=values[len(values)//2],
                p99=values[lower]+(values[min(lower+1,len(values)-1)]-values[lower])*(index-lower),max=values[-1])


def summarize(folder,output):
    if output.exists():
        raise ValueError('Summary directory must be new')
    output.mkdir(parents=True)
    runs=[]
    combined=defaultdict(list)
    for path in sorted(folder.glob('arm_*.json')):
        data=json.loads(path.read_text(encoding='utf-8'))
        if 'records' not in data:
            continue
        variant=path.stem[4:].rsplit('_',1)[0]
        assert data['decision_parity'] and data['cases']==91
        rows=data['records']
        runs.append(dict(file=path.name,variant=variant,wall_ms=stats([r['wall_ms'] for r in rows]),
                         cpu_ms=stats([r['cpu_ms'] for r in rows])))
        combined[variant].extend(rows)
    report=dict(scope='Actual ARM offline replay only; not live PS response or WCET',runs=runs,variants={})
    phase_rows=[]
    for variant,rows in combined.items():
        phases={}
        if variant.endswith('profile'):
            for stage in rows[0]['phases']:
                phases[stage]=dict(mean_wall_ms=sum(r['phases'][stage]['wall_ms'] for r in rows)/len(rows),
                                  mean_cpu_ms=sum(r['phases'][stage]['cpu_ms'] for r in rows)/len(rows),
                                  mean_calls=sum(r['phases'][stage]['calls'] for r in rows)/len(rows))
                phase_rows.append(dict(variant=variant,stage=stage,**phases[stage]))
        groups={}
        for name in sorted({r['name'] for r in rows}):
            selected=[r for r in rows if r['name']==name]
            groups[name]=dict(wall_ms=stats([r['wall_ms'] for r in selected]),cpu_ms=stats([r['cpu_ms'] for r in selected]))
        report['variants'][variant]=dict(records=len(rows),wall_ms=stats([r['wall_ms'] for r in rows]),
            cpu_ms=stats([r['cpu_ms'] for r in rows]),exclusive_phases=phases,per_case=groups)
    assert set(combined)=={'baseline','profile','move_baseline','move_profile'}
    assert sum(len(rows) for rows in combined.values())==7280
    (output/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    with (output/'phases.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(phase_rows[0]))
        writer.writeheader();writer.writerows(phase_rows)
    for variant,data in report['variants'].items():
        print(variant,'wall',data['wall_ms'],'cpu',data['cpu_ms'])
    print('Baseline instrumented exclusive CPU stages:')
    for stage,data in sorted(report['variants']['profile']['exclusive_phases'].items(),key=lambda p:-p[1]['mean_cpu_ms']):
        print(stage,data)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    summarize(args.folder,args.output)
