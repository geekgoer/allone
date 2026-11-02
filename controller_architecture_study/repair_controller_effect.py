"""Train-only grouped CV and an empirical advantage guard; no policy NN."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from .controller_effect import EffectRanker,features241,physical_context,action_features,load_outcomes,evaluation
from .decision_dataset import atomic_json,sha256


def fit(rows,models,penalty):
    groups=sorted({r['group'] for r in rows})
    states={g:np.r_[features241(next(r for r in rows if r['group']==g)['state_feature']),
        physical_context(next(r for r in rows if r['group']==g)['initial_observation'])] for g in groups}
    roots={g:next(r for r in rows if r['group']==g).get('cv_group',g) for g in groups}
    counts={root:sum(v==root for v in roots.values()) for root in set(roots.values())}
    weights=np.array([1./counts[roots[g]] for g in groups]);weights/=weights.sum()
    x=np.array(list(states.values()));mean=np.sum(x*weights[:,None],axis=0)
    std=np.maximum(np.sqrt(np.sum((x-mean)**2*weights[:,None],axis=0)),.05)
    phi=np.array([np.r_[1.,(states[r['group']]-mean)/std] for r in rows])
    action=action_features(models,[r['model'] for r in rows],[r['cadence_s'] for r in rows],[r['actual_delay_s'] for r in rows])
    design=np.einsum('ni,nj->nij',phi,action).reshape(len(phi),-1)
    target=np.array([r['utility'] for r in rows])
    for g in groups:
        ids=[i for i,r in enumerate(rows) if r['group']==g]
        divisor=np.sqrt(len(ids)*counts[roots[g]])
        design[ids]=(design[ids]-design[ids].mean(axis=0))/divisor
        target[ids]=(target[ids]-target[ids].mean())/divisor
    regularization=np.full(design.shape[1],penalty)
    regularization[:action.shape[1]]=1.  # shrink contextual slopes toward common action effects
    theta=np.linalg.solve(design.T@design+np.diag(regularization),design.T@target)
    return mean,std,theta.reshape(phi.shape[1],action.shape[1])


def raw_score(rows,models,parameters):
    mean,std,theta=parameters
    phi=np.array([np.r_[1.,(np.r_[features241(r['state_feature']),physical_context(r['initial_observation'])]-mean)/std] for r in rows])
    action=action_features(models,[r['model'] for r in rows],[r['cadence_s'] for r in rows],[r['actual_delay_s'] for r in rows])
    return np.einsum('ni,ij,nj->n',phi,theta,action)


def route_folds(training):
    roots=sorted({r.get('cv_group',r['group']) for r in training})
    return [set(roots[i::4]) for i in range(4)]


def cross_validate(training,models,costs):
    groups=sorted({r['group'] for r in training});folds=route_folds(training)
    experiments=[]
    for penalty in (1.,10.,100.,1000.):
        predictions=np.empty(len(training))
        for fold in folds:
            fit_rows=[r for r in training if r.get('cv_group',r['group']) not in fold]
            ids=[i for i,r in enumerate(training) if r.get('cv_group',r['group']) in fold]
            predictions[ids]=raw_score([training[i] for i in ids],models,fit(fit_rows,models,penalty))
        regret=[];overstatement=[]
        for g in groups:
            ids=[i for i,r in enumerate(training) if r['group']==g]
            y=np.array([training[i]['utility'] for i in ids]);p=predictions[ids]
            base=min(range(len(ids)),key=lambda j:(costs[training[ids[j]]['model']],training[ids[j]]['cadence_s'],training[ids[j]]['requested_delay_s']))
            regret.append(float(y.max()-y[p.argmax()]))
            overstatement.append(float(np.max((p-p[base])-(y-y[base]))))
        roots={g:next(r for r in training if r['group']==g).get('cv_group',g) for g in groups}
        scene_overstatement=[max(overstatement[i] for i,g in enumerate(groups) if roots[g]==root)
            for root in sorted(set(roots.values()))]
        scene_regret=[np.mean([regret[i] for i,g in enumerate(groups) if roots[g]==root])
            for root in sorted(set(roots.values()))]
        experiments.append(dict(penalty=penalty,regret=float(np.mean(scene_regret)),
            overstatement=scene_overstatement,predictions=predictions))
    best=min(experiments,key=lambda e:(e['regret'],-e['penalty']))
    # Empirical, scene-level out-of-fold error guard. This is NOT a conformal
    # closed-loop safety guarantee and does not replace global admission.
    margin=float(max(0.,np.quantile(best['overstatement'],.90,method='higher')))
    return best['penalty'],margin,[{k:v for k,v in e.items() if k!='predictions'} for e in experiments]


def boundary_outcomes(directory):
    from .controller_effect import outcome_utility
    rows=[]
    report=json.loads((directory/'report.json').read_text())
    if not report['complete']:raise ValueError('complete boundary sampling required')
    for p in map(Path,report['selected_reports']):
        row=json.loads(p.read_text());trace=json.loads((p.parent/'trace.json').read_text())
        row.update(split='train' if row['split']=='discovery' else 'boundary_confirmation',
            cv_group=f"boundary_route_{row['route_seed']}",utility=outcome_utility(row,trace),
            initial_observation={k:trace[0][k] for k in ('risk','box_margin_m')})
        rows.append(row)
    return rows


def train(source,cadence,original,output,boundary=None):
    rows=load_outcomes(source,cadence)
    if boundary is not None:rows.extend(boundary_outcomes(boundary))
    training=[r for r in rows if r['split']=='train']
    frozen=EffectRanker(original);models=frozen.models
    plan=json.loads((source/'plan.json').read_text());costs=plan['model_cost_s']
    with threadpool_limits(limits=1):
        penalty,margin,cv=cross_validate(training,models,costs)
        mean,std,theta=fit(training,models,penalty)
    output.mkdir(parents=True,exist_ok=False)
    support=frozen.support
    if boundary is not None:
        from scipy.spatial import ConvexHull
        support={}
        for model in models:
            points=np.unique([[r['cadence_s'],r['requested_delay_s']] for r in training if r['model']==model],axis=0)
            if len(np.unique(points[:,0]))==1:
                t=points[0,0];lo,hi=points[:,1].min(),points[:,1].max()
                hull=[[1.,0.,-t],[-1.,0.,t],[0.,1.,-hi],[0.,-1.,lo]]
            else:hull=ConvexHull(points).equations.tolist()
            support[model]=dict(points=points.tolist(),hull=hull)
    np.savez(output/'ranker.npz',mean=mean,std=std,theta=theta,models=np.array(models),
        support=json.dumps(support),physical_context=True,override_margin=margin,
        model_costs=np.array([costs[m] for m in models]))
    ranker=EffectRanker(output/'ranker.npz')
    splits=['train','validation','calibration']+(['boundary_confirmation'] if boundary is not None else [])
    metrics={s:evaluation([r for r in rows if r['split']==s],ranker,costs) for s in splits}
    sample=ranker.features_for_row(training[0]);times=[]
    with threadpool_limits(limits=1):
        for _ in range(500):
            began=time.perf_counter();ranker.scores(sample,models*5,[.05]*40,[.03]*40)
            times.append(1000*(time.perf_counter()-began))
    report=dict(method='geometry-conditioned action advantage with train-group CV and empirical override guard',
        parameters=int(theta.size),context_penalty=penalty,override_margin=margin,group_cv=cv,metrics=metrics,
        fit_groups=len({r['group'] for r in training}),fit_independent_routes=len({r.get('cv_group',r['group']) for r in training}),
        hyperparameters_selected_on='four training-route folds only; adjacent boundary states stay together',
        boundary_data=str(boundary) if boundary else None,
        boundary_sha256=sha256(boundary/'report.json') if boundary else None,
        old_validation_is_development_data=True,new_holdout_not_yet_evaluated=True,test_evaluated=False,
        source_sha256=sha256(source/'paired_rows.json'),cadence_sha256=sha256(cadence/'rows.json'),
        model_sha256=sha256(output/'ranker.npz'),source_code_sha256=sha256(Path(__file__)),
        score40_ms=dict(mean=float(np.mean(times)),p99=float(np.percentile(times,99)),max=max(times)),
        qualified=False,scope='proxy ranking only; abstains to cheapest/fastest candidate without sufficient learned advantage')
    atomic_json(output/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('metrics','group_cv')},indent=2),flush=True)
    for s,m in metrics.items():print(s,{k:v for k,v in m.items() if k!='selections'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','cadence','original','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--boundary',type=Path)
    a=p.parse_args();train(a.source,a.cadence,a.original,a.output,a.boundary)
