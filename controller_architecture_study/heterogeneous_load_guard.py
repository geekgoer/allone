"""Small shared-core demand guard; empirical utilization, not a WCET proof."""

import numpy as np


SLOTS=(.020,.030,.040,.050,.060)


def load(periods,models,costs):
    return sum(costs[models[v]]/p for v,p in periods.items())


def resource_loads(periods,models,costs,pipeline_costs=None):
    result={'backend':load(periods,models,costs)}
    if pipeline_costs is not None:
        service=sum(pipeline_costs.get(k,0.) for k in ('input_s','output_s','planner_reserve_s','handoff_s'))
        result['frontend']=sum(service/p for p in periods.values())
    return result


def allocate(desired,urgencies,models,costs,limit=.95,*,pipeline_costs=None,max_periods=None):
    ceilings={v:min(SLOTS[-1],(max_periods or {}).get(v,SLOTS[-1])) for v in desired}
    periods={v:min(ceilings[v],max(SLOTS[0],p)) for v,p in desired.items()}
    relaxed=[]
    for _ in range(len(periods)*len(SLOTS)):
        demands=resource_loads(periods,models,costs,pipeline_costs)
        demand=max(demands.values())
        if demand<=limit+1e-12:
            return periods,dict(feasible=True,load=demand,resource_loads=demands,relaxed=relaxed,limit=limit)
        candidates=[v for v,p in periods.items() if any(p<x<=ceilings[v]+1e-12 for x in SLOTS)]
        if not candidates:break
        v=min(candidates,key=lambda name:(urgencies.get(name,0.),-costs[models[name]]/periods[name],name))
        periods[v]=next(p for p in SLOTS if periods[v]+1e-12<p<=ceilings[v]+1e-12)
        relaxed.append(v)
    demands=resource_loads(periods,models,costs,pipeline_costs)
    return periods,dict(feasible=False,load=max(demands.values()),resource_loads=demands,relaxed=relaxed,limit=limit)


def recover_profile_costs(rows):
    """Recover the frozen costs from logged linear sums, without re-calibration."""
    names=sorted({m for r in rows for m in r['configuration']['models'].values()})
    matrix=[];target=[]
    for row in rows:
        c=row['configuration']
        matrix.append([sum(1000/c['periods_ms'][v] for v,m in c['models'].items() if m==name) for name in names])
        target.append(row['capacity']['compute_load'])
    a=np.asarray(matrix);b=np.asarray(target)
    values,_,rank,_=np.linalg.lstsq(a,b,rcond=None)
    residual=float(np.max(np.abs(a@values-b)))
    if rank!=len(names) or residual>1e-9 or np.min(values)<=0.:
        raise ValueError('logged capacity sums do not identify positive model costs')
    return dict(zip(names,map(float,values))),dict(rank=int(rank),max_residual=residual,
        method='exact linear reconstruction of frozen source capacity sums; no new timing measurements')
