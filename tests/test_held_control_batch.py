from types import SimpleNamespace
import numpy as np

from environment import Gate
from controller_architecture_study.held_control_safety import GateHoldAnalyzer,GateHoldBatch,HoldBounds


def test_batched_periods_equal_individual_proofs_with_different_obstacle_counts():
    rng=np.random.default_rng(413)
    bounds=HoldBounds(2.,.01,.05,.15,10.,'test')
    analyzers={f'd{i}':GateHoldAnalyzer([Gate(k,np.array([k*3.,0.,-4.]),np.array([1.,0.,0.,0.]))
        for k in range(1,i+2)],1.7,bounds) for i in range(5)}
    batch=GateHoldBatch(analyzers)
    for _ in range(12):
        states={v:SimpleNamespace(position=rng.uniform([0.,-1.,-5.],[10.,1.,-3.]),velocity=rng.uniform(-4.,4.,3),
            quaternion=np.r_[1.,rng.uniform(-.15,.15,3)]) for v in analyzers}
        controls={v:np.r_[rng.uniform(7.,15.),rng.uniform(-2.,2.,3)] for v in analyzers}
        results=batch.analyze(states,controls)
        for v,a in analyzers.items():
            old=a.analyze(states[v],controls[v])
            assert abs(results[v]['budget_s']-old['budget_s'])<1e-12
            assert results[v]['binding_obstacle']==old['binding_obstacle']
            margin=(a.normals@states[v].position-a.offsets).max(axis=1).min()
            assert abs(margin-results[v]['mesh_margin_m'])<1e-12
