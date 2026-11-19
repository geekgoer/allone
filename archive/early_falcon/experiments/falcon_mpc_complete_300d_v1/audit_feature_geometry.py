"""Input conditioning and local interpolation diagnostic; train/validation only."""
from pathlib import Path
import json
import numpy as np
from threadpoolctl import threadpool_limits
from sklearn.neighbors import NearestNeighbors


def recenter(x):
    z=x.astype(np.float64).copy()
    refs=z[:,10:220].reshape(-1,21,10)
    refs[:,:,:3]-=z[:,None,:3]
    refs[:,:,7:10]-=z[:,None,7:10]
    return z


def main():
    base=Path(__file__).resolve().parent
    out=base/'audit_generalization'
    ds=base/'dnn_dataset_300d_140k_v1_within_environment'
    data={}
    for name in ('train','validation'):
        with np.load(ds/f'{name}.npz') as z:
            data[name]={k:z[k] for k in ('X','U','sample_family','query_group_id','source_kind')}
    tr,va=data['train'],data['validation']
    indices=np.random.default_rng(55).choice(len(va['X']),1000,replace=False)
    ymean=tr['U'].mean(axis=0); ystd=tr['U'].std(axis=0)
    target=(va['U'][indices]-ymean)/ystd
    y=(tr['U']-ymean)/ystd
    report={'validation_queries':len(indices),'recoding':'Lossless affine recoding: keep current state, subtract current p and v from each reference p and v.'}
    for name,transform in [('absolute',lambda x:x.astype(np.float64)),('relative_pv_lossless',recenter)]:
        tx=transform(tr['X']); vx=transform(va['X'][indices])
        mean=tx.mean(axis=0); std=np.maximum(tx.std(axis=0),1e-6)
        tx=((tx-mean)/std).astype(np.float32); vx=((vx-mean)/std).astype(np.float32)
        with threadpool_limits(limits=4):
            nn=NearestNeighbors(n_neighbors=8,algorithm='brute',n_jobs=1).fit(tx)
            distances,neighbors=nn.kneighbors(vx)
        report[name]={}
        for k in (1,8):
            err=np.mean(y[neighbors[:,:k]],axis=1)-target
            report[name][f'knn{k}_normalized_mse']=float(np.mean(err**2))
            report[name][f'knn{k}_per_output']=np.mean(err**2,axis=0).tolist()
        report[name]['distance_quantiles']=np.quantile(distances[:,0],[.5,.9,.99]).tolist()
        print(name,report[name],flush=True)
    x=tr['X'].astype(np.float64)
    refs=x[:,10:220].reshape(-1,21,10)
    report['conditioning']={
        'current_position_std':x[:,:3].std(axis=0).tolist(),
        'reference0_position_std':refs[:,0,:3].std(axis=0).tolist(),
        'position_error_std':(refs[:,0,:3]-x[:,:3]).std(axis=0).tolist(),
        'position_pair_correlation':[float(np.corrcoef(x[:,j],refs[:,0,j])[0,1]) for j in range(3)],
        'velocity_error_std':(refs[:,0,7:10]-x[:,7:10]).std(axis=0).tolist(),
        'parameters_small_model':(300+1)*512+8*(512+1)*512+(512+1)*4,
    }
    (out/'feature_geometry.json').write_text(json.dumps(report,indent=2))
    print(report['conditioning'])


if __name__=='__main__': main()
