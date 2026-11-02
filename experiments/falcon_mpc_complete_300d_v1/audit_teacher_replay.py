"""Replay stored solver inputs, inspect history sensitivity and false convergence."""
from pathlib import Path
import ctypes
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from falcon_paper_fidelity.acado_mpc import AcadoMPCConfig, AcadoRtiMPC


def summary(a):
    a=np.asarray(a,dtype=np.float64)
    return dict(n=len(a), mean=float(a.mean()), p50=float(np.median(a)),
                p95=float(np.quantile(a,.95)), max=float(a.max()))


def main():
    base=Path(__file__).resolve().parent
    output=base/'audit_generalization'
    cfg=json.loads((base/'dataset_config.json').read_text())
    mpc=dict(cfg['mpc']); mpc['library']=str((base/mpc['library']).resolve())
    ctl=AcadoRtiMPC(AcadoMPCConfig(**mpc))
    pointer=ctypes.POINTER(ctypes.c_double)
    def solve(x, steps=16, stop=True):
        current=np.ascontiguousarray(x[:10],dtype=np.float64)
        refs=np.ascontiguousarray(x[10:220],dtype=np.float64)
        uref=np.ascontiguousarray(np.r_[x[220:],x[-4:]],dtype=np.float64)
        ctl._raise_status(ctl._dll.falcon_acado_reset(current.ctypes.data_as(pointer)), 'reset')
        controls=[]; diagnostics=[]
        for k in range(steps):
            u=np.empty(4); diag=np.empty(3)
            status=ctl._dll.falcon_acado_solve(current.ctypes.data_as(pointer),refs.ctypes.data_as(pointer),uref.ctypes.data_as(pointer),u.ctypes.data_as(pointer),diag.ctypes.data_as(pointer))
            ctl._raise_status(status,'replay')
            u[2:]*=-1
            controls.append(u.copy()); diagnostics.append(diag.copy())
            if stop and k>=3 and np.max(np.abs(controls[-1]-controls[-2]))<=.001:
                break
        return np.asarray(controls),np.asarray(diagnostics)
    selected=[]
    rng=np.random.default_rng(1729)
    for split in ('validation','train'):
        with np.load(base/'dnn_dataset_300d_140k_v1_within_environment'/f'{split}.npz') as z:
            features, targets, sample_ids = z['X'], z['U'], z['sample_id']
            with np.load(output/f'{split}_errors.npz') as e:
                indices=np.unique(np.r_[e['worst_indices'][:50],rng.choice(len(features),100,False)])
            for i in indices:
                selected.append((split,int(i),features[i],targets[i],int(sample_ids[i])))
    results=[]
    for n,(split,i,x,target,sample_id) in enumerate(selected):
        controls,diag=solve(x)
        long,ld=solve(x,steps=64,stop=False)
        results.append(dict(split=split,index=i,sample_id=sample_id,
            original=target.tolist(),replayed=controls[-1].tolist(),refined64=long[-1].tolist(),
            replay_max_abs=float(np.max(np.abs(controls[-1]-target))),
            refinement_max_abs=float(np.max(np.abs(long[-1]-controls[-1]))),
            kkt=float(diag[-1,0]),kkt64=float(ld[-1,0]),
            objective=float(diag[-1,1]),objective64=float(ld[-1,1]),
            final_delta64=float(np.max(np.abs(long[-1]-long[-2]))),
            iterations=len(controls)))
        if n%50==0: print('replay',n,'/',len(selected),flush=True)
    # Interleave unrelated queries before repeating the same problems.
    for original_index in rng.permutation(len(selected)):
        _,_,x,_,_=selected[int(original_index)]
        controls,_=solve(x)
        results[int(original_index)]['order_change_max_abs']=float(np.max(np.abs(controls[-1]-results[int(original_index)]['replayed'])))
    report={'summary':{key:summary([r[key] for r in results]) for key in ('replay_max_abs','refinement_max_abs','order_change_max_abs','kkt','kkt64','final_delta64')},'rows':results}
    (output/'teacher_replay.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report['summary'],indent=2))


if __name__=='__main__': main()
