"""Small shared-core demand guard; empirical utilization, not a WCET proof."""

import numpy as np


SLOTS=(.020,.030,.040,.050,.060)


def project_frequencies(desired, weights, lower, upper, demands, capacity):
    """Weighted frequency projection with backend and frontend capacities."""
    desired, weights, lower, upper, demands, capacity = map(
        lambda value: np.asarray(value, dtype=float),
        (desired, weights, lower, upper, demands, capacity))
    if np.any(lower > upper) or np.any(demands @ lower > capacity + 1e-10):
        return None

    def cap(preferred, demand, limit):
        result = np.clip(preferred, lower, upper)
        if demand @ result <= limit + 1e-10:
            return result
        positive = demand > 0
        high = max(0., np.max((preferred[positive]-lower[positive])*weights[positive]/demand[positive]))
        low = 0.
        for _ in range(22):
            price = (low+high)/2
            result = np.clip(preferred-price*demand/weights, lower, upper)
            if demand @ result > limit:
                low = price
            else:
                high = price
        return np.clip(preferred-high*demand/weights, lower, upper)

    backend, frontend = demands
    result = cap(desired, backend, capacity[0])
    if frontend @ result <= capacity[1]+1e-10:
        return result
    low = 0.
    positive = frontend > 0
    high = max(0., np.max((desired[positive]-lower[positive])*weights[positive]/frontend[positive]))
    for _ in range(18):
        price = (low+high)/2
        result = cap(desired-price*frontend/weights, backend, capacity[0])
        if frontend @ result > capacity[1]:
            low = price
        else:
            high = price
    result = cap(desired-high*frontend/weights, backend, capacity[0])
    return result if np.all(demands @ result <= capacity+1e-10) else None


def load(periods,models,costs):
    return sum(costs[models[v]]/p for v,p in periods.items())


def resource_loads(periods,models,costs,pipeline_costs=None):
    result={'backend':load(periods,models,costs)}
    if pipeline_costs is not None:
        service=sum(pipeline_costs.get(k,0.) for k in ('input_s','output_s','planner_reserve_s','handoff_s'))
        management=(pipeline_costs.get('management_budget_s',0.)/pipeline_costs['management_period_s']
            if pipeline_costs.get('management_budget_s') else 0.)
        result['frontend']=sum(service/p for p in periods.values())+management
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
