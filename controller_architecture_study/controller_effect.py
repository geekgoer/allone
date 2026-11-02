"""Small contextual advantage ranker; decisions and safety remain explicit.

Within-scene centering removes common scene difficulty. The learned score is
a relative recovery/tracking proxy, NEVER FALCON J or a safety certificate.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np

from .decision_dataset import atomic_json, sha256
from .fast_convex_preprocess import FastReferenceWindow, VECTOR_FLIP, QUATERNION_FLIP, _attitudes, _normalize


def features241(state):
    z=np.asarray(state,dtype=float)
    if z.shape!=(241,) or not np.isfinite(z).all():raise ValueError('finite paired 237D state + held control required')
    ref=z[7:217].reshape(21,10)
    return reduced_features(ref[0,:3],ref[0,7:10],z[:4],ref[0,3:7],z[4:7],
        ref[[0,5,10,20],7:10],z[237:])


def reduced_features(pe,ve,q,qr,velocity,preview,held):
    angle=2*np.arccos(np.clip(abs(q@qr)/(np.linalg.norm(q)*np.linalg.norm(qr)),0.,1.))/np.pi
    preview_acc=np.diff(preview,axis=0)/np.array([.5,.5,1.])[:,None]
    return np.r_[pe/3.,ve/5.,angle,np.linalg.norm(velocity)/10.,
        np.linalg.norm(preview[-1]-preview[0])/10.,np.linalg.norm(preview_acc,axis=1).max()/10.,
        held[0]/9.80665-1.,np.linalg.norm(held[1:])/3.,pe@ve/15.]


def reference_spec(window,phase):
    return dict(phase=float(phase),gravity=window.config.gravity,
        **{k:getattr(window,k).tolist() for k in ('coefficients','starts','ends','durations','headings')},
        trajectory=dict(duration=window.trajectory.duration,
            **{k:getattr(window.trajectory,k).tolist() for k in ('final_position','final_velocity','final_acceleration')}))


def reference_from_spec(spec):
    window=FastReferenceWindow.__new__(FastReferenceWindow)
    for k in ('coefficients','starts','ends','durations','headings'):setattr(window,k,np.array(spec[k]))
    window.config=SimpleNamespace(gravity=spec['gravity'])
    window.trajectory=SimpleNamespace(**spec['trajectory'])
    return window


def live_features(window,elapsed,state):
    p,v,a,heading=window._evaluate(elapsed+np.array([0.,.5,1.,2.]))
    qr=_normalize(_attitudes(a[:1],heading[:1],window.config.gravity))[0]
    # Current/ref quaternions and vectors must use exactly the frozen ENU contract.
    return reduced_features((p[0]-state['position'])*VECTOR_FLIP,(v[0]-state['velocity'])*VECTOR_FLIP,
        _normalize(state['quaternion'])*QUATERNION_FLIP,qr,state['velocity']*VECTOR_FLIP,
        (v-state['velocity'])*VECTOR_FLIP,np.asarray(state['held_control']))


def action_features(models,names,periods,delays):
    t,r=np.asarray(periods)/.1,np.asarray(delays)/.1
    one=np.array([[float(n==m) for m in models] for n in names])
    return np.c_[one,one*t[:,None],one*r[:,None],t*t,r*r,t*r]


class EffectRanker:
    def __init__(self,path):
        data=np.load(path,allow_pickle=False)
        self.mean,self.std,self.theta=data['mean'],data['std'],data['theta']
        self.models=list(data['models']);self.support=json.loads(str(data['support']))
        self.hulls={m:np.array(s['hull']) for m,s in self.support.items()}
        self.context=bool(data['physical_context']) if 'physical_context' in data else False
        self.override_margin=float(data['override_margin']) if 'override_margin' in data else None
        self.costs=dict(zip(self.models,data['model_costs'])) if 'model_costs' in data else {}

    def raw_scores(self,state,names,periods,delays):
        phi=np.r_[1.,(np.asarray(state)-self.mean)/self.std]
        action=action_features(self.models,names,periods,delays)
        return action@(phi@self.theta)

    def scores(self,state,names,periods,delays):
        score=self.raw_scores(state,names,periods,delays)
        if self.override_margin is not None:
            baseline=min(range(len(names)),key=lambda i:(self.costs[names[i]],periods[i],delays[i]))
            score=score-score[baseline]-self.override_margin;score[baseline]=0.
        return score

    def features_for_row(self,row):
        base=features241(row['state_feature'])
        return np.r_[base,physical_context(row['initial_observation'])] if self.context else base

    def supported(self,model,period,delay):
        return bool(self.supported_many(model,np.array([period]),np.array([delay]))[0])

    def supported_many(self,model,periods,delays):
        hull=self.hulls.get(model)
        if hull is None:return np.zeros(len(periods),dtype=bool)
        return np.all(np.c_[periods,delays]@hull[:,:2].T+hull[:,2]<=1e-7,axis=1)


def outcome_utility(row,trace):
    label=row['label']
    if not label['valid']:raise ValueError('invalid outcome is not a zero-reward trial')
    if label.get('collision') or row['error_kind']=='flight':return 0.
    if trace[-1]['t']<1.999:raise ValueError('noncollision utility requires a complete observed window')
    t=np.array([r['t'] for r in trace]);cost=np.array([r['state_cost'] for r in trace])
    mean=float(np.sum(np.diff(t)*(cost[1:]+cost[:-1])/2)/(t[-1]-t[0]))
    return float(label['qualified'])+.5/(1.+mean)


def load_outcomes(source,cadence):
    original=json.loads((source/'paired_rows.json').read_text())
    extra=json.loads((cadence/'rows.json').read_text())
    if len(extra)!=96 or not all(r['label']['valid'] for r in extra):raise ValueError('complete 96-trial cadence extension required')
    rows=[]
    for root,group in ((source,original),(cadence,extra)):
        for r in group:
            if r['split'] not in ('train','validation','calibration'):continue
            if not r['label']['valid']:raise ValueError('invalid paired row')
            if root==source:
                parent=root/'trials'/f"{r['group']}_d{r['requested_delay_s']*1000:03.0f}_{r['model']}"
            else:
                parent=root/'trials'/f"{r['group']}_T{r['cadence_s']*1000:.0f}_d{r['requested_delay_s']*1000:.0f}_{r['model']}"
            found=None
            for report in sorted(parent.glob('attempt_*/report.json')):
                if json.loads(report.read_text())['label']['valid']:found=report.parent;break
            if found is None:raise ValueError(f'missing valid trace: {parent}')
            trace=json.loads((found/'trace.json').read_text())
            rows.append({**r,'cadence_s':r.get('cadence_s',.050),
                'utility':outcome_utility(r,trace),'trace_sha256':sha256(found/'trace.json'),
                'initial_observation':{k:trace[0][k] for k in ('risk','box_margin_m')}})
    for g in {r['group'] for r in rows}:
        group=[r for r in rows if r['group']==g]
        if len({r['split'] for r in group})!=1 or any(not np.allclose(group[0]['state_feature'],r['state_feature'],atol=2e-4,rtol=0) for r in group):
            raise ValueError('paired scene feature/split mismatch')
    return rows


def evaluation(rows,ranker,costs):
    predicted=[];chosen=[];baseline=[];oracle=[];pairs=[]
    for g in sorted({r['group'] for r in rows}):
        group=[r for r in rows if r['group']==g]
        score=ranker.scores(ranker.features_for_row(group[0]),[r['model'] for r in group],
            [r['cadence_s'] for r in group],[r['actual_delay_s'] for r in group])
        y=np.array([r['utility'] for r in group]);best=int(np.argmax(score))
        chosen.append(dict(group=g,model=group[best]['model'],period_s=group[best]['cadence_s'],
            delay_s=group[best]['actual_delay_s'],utility=float(y[best]),qualified=group[best]['label']['qualified'],
            regret=float(y.max()-y[best])))
        cheap=min(range(len(group)),key=lambda i:(costs[group[i]['model']],group[i]['cadence_s'],group[i]['requested_delay_s']))
        baseline.append(y[cheap]);oracle.append(y.max());predicted.extend(score.tolist())
        a,b=np.triu_indices(len(group),1);distinct=np.abs(y[a]-y[b])>.02
        pairs.extend(((score[a]-score[b])*(y[a]-y[b])>0)[distinct].tolist())
    return dict(groups=len(chosen),rows=len(rows),pairwise_accuracy=float(np.mean(pairs)) if pairs else None,
        selected_mean_utility=float(np.mean([r['utility'] for r in chosen])),
        selected_qualified=sum(r['qualified'] for r in chosen),
        mean_measured_regret=float(np.mean([r['regret'] for r in chosen])),
        fixed_cheapest_fastest_mean_utility=float(np.mean(baseline)),oracle_mean_utility=float(np.mean(oracle)),
        selections=chosen,scope='finite paired action blocks, not global J or a recovery certificate')


def physical_context(observation):
    risk=observation['risk']
    return np.array([risk['h'],risk['hdot']/5.,risk['turn_pressure']/10.,observation['box_margin_m']/5.])


def contextual_live_features(window,elapsed,state,net_width,mesh_margin):
    from .framework_runtime import route_risk
    p,v,a,heading=window._evaluate(elapsed+np.array([0.,.5,1.,2.]))
    qr=_normalize(_attitudes(a[:1],heading[:1],window.config.gravity))[0]
    q=_normalize(qr*QUATERNION_FLIP)
    ref=SimpleNamespace(position=p[0],velocity=v[0],acceleration=a[0],orientation=q)
    raw=SimpleNamespace(**state)
    base=reduced_features((p[0]-state['position'])*VECTOR_FLIP,(v[0]-state['velocity'])*VECTOR_FLIP,
        _normalize(state['quaternion'])*QUATERNION_FLIP,qr,state['velocity']*VECTOR_FLIP,
        (v-state['velocity'])*VECTOR_FLIP,np.asarray(state['held_control']))
    return np.r_[base,physical_context(dict(
        risk=route_risk(raw,ref,net_width),box_margin_m=mesh_margin))]


def train(source,cadence,output):
    from scipy.spatial import ConvexHull
    from threadpoolctl import threadpool_limits
    rows=load_outcomes(source,cadence);training=[r for r in rows if r['split']=='train']
    plan=json.loads((source/'plan.json').read_text());models=plan['models']
    state=np.array([features241(r['state_feature']) for r in training])
    # Scene-equal normalization, independent of how many actions a scene has.
    unique=np.array([state[next(i for i,r in enumerate(training) if r['group']==g)] for g in sorted({r['group'] for r in training})])
    mean,std=unique.mean(axis=0),np.maximum(unique.std(axis=0),.05)
    phi=np.c_[np.ones(len(state)),(state-mean)/std]
    action=action_features(models,[r['model'] for r in training],[r['cadence_s'] for r in training],[r['actual_delay_s'] for r in training])
    design=np.einsum('ni,nj->nij',phi,action).reshape(len(phi),-1)
    y=np.array([r['utility'] for r in training])
    for g in {r['group'] for r in training}:
        ids=np.array([i for i,r in enumerate(training) if r['group']==g])
        design[ids]=(design[ids]-design[ids].mean(axis=0))/np.sqrt(len(ids))
        y[ids]=(y[ids]-y[ids].mean())/np.sqrt(len(ids))
    with threadpool_limits(limits=1):theta=np.linalg.solve(design.T@design+np.eye(design.shape[1]),design.T@y).reshape(phi.shape[1],action.shape[1])
    support={}
    for m in models:
        # Requested protocols define the support, small measured jitter does not extend it.
        points=np.unique([[r['cadence_s'],r['requested_delay_s']] for r in training if r['model']==m],axis=0)
        if len(np.unique(points[:,0]))==1:
            t=points[0,0];lo,hi=points[:,1].min(),points[:,1].max()
            hull=[[1.,0.,-t],[-1.,0.,t],[0.,1.,-hi],[0.,-1.,lo]]
        else:hull=ConvexHull(points).equations.tolist()
        support[m]=dict(points=points.tolist(),hull=hull)
    output.mkdir(parents=True,exist_ok=False)
    np.savez(output/'ranker.npz',mean=mean,std=std,theta=theta,models=np.array(models),support=json.dumps(support))
    ranker=EffectRanker(output/'ranker.npz')
    metrics={s:evaluation([r for r in rows if r['split']==s],ranker,plan['model_cost_s']) for s in ('train','validation','calibration')}
    trial=features241(training[0]['state_feature']);times=[]
    with threadpool_limits(limits=1):
        for _ in range(500):
            begin=time.perf_counter();ranker.scores(trial,models*5,[.05]*(5*len(models)),[.03]*(5*len(models)))
            times.append(1000*(time.perf_counter()-begin))
    report=dict(method='scene-centered bilinear ridge advantage ranker',ridge_lambda=1.,parameters=int(theta.size),
        state_features=int(len(mean)),utility='qualified + 0.5/(1+observed mean tracking cost); collision=0',
        coefficients_sha256=sha256(output/'ranker.npz'),source_sha256=sha256(source/'paired_rows.json'),
        cadence_sha256=sha256(cadence/'rows.json'),metrics=metrics,
        scoring_40_actions_ms=dict(mean=float(np.mean(times)),p99=float(np.percentile(times,99)),max=max(times)),
        test_evaluated=False,physical_safety_certified=False,online_takeover_allowed=False,
        limitations='small independent scene sample; convex-hull interpolation is support, not certification; Q is not FALCON J')
    atomic_json(output/'report.json',report)
    atomic_json(output/'training_manifest.json',[dict(group=r['group'],split=r['split'],model=r['model'],
        cadence_s=r['cadence_s'],delay_s=r['actual_delay_s'],trace_sha256=r['trace_sha256']) for r in rows])
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--cadence',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();train(args.source,args.cadence,args.output)
