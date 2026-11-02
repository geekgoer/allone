import numpy as np

from controller_architecture_study.fair_static_dynamic import (
    MODEL_BANK,PROTOCOL,best_candidate,candidate_score,search_starts)
from controller_architecture_study.heterogeneous_motivation import configuration_key,propose


def recovery():
    return {f'Drone{i}':'plain_mlp_h2_w128_seed0' if i in (1,3,9) else
        'plain_mlp_h9_w1024_seed0' if i in (2,10) else 'plain_mlp_h9_w512_seed0' for i in range(1,11)}


def test_static_seeds_include_dynamic_models_and_successful_fixed_combination():
    models=recovery();starts=search_starts(list(models),MODEL_BANK,models)
    assert len(starts)<=PROTOCOL['budget']
    assert len({configuration_key(c) for c in starts})==len(starts)
    assert set(m for c in starts for m in c['models'].values())==set(MODEL_BANK)
    assert all(c['max_inflight']==10 for c in starts)
    assert any(c['models']==models for c in starts)
    assert any(set(c['models'].values())=={'plain_mlp_h2_w128_seed0'} for c in starts)
    assert any(set(c['models'].values())=={'plain_mlp_h3_w256_seed0'} for c in starts)


def test_neighbor_search_can_choose_small_models_outside_old_six_model_bank():
    starts=search_starts(list(recovery()),MODEL_BANK,recovery());base=starts[0]
    pool=('plain_mlp_h2_w128_seed0','plain_mlp_h3_w256_seed0')
    result=propose(base,1,np.random.default_rng(7),{configuration_key(base)},model_names=pool)
    assert any(m in pool for m in result['models'].values())


def test_static_selection_is_by_J_not_contact_or_deadline_outcomes():
    safe=dict(index=0,result=dict(complete=True,J_hat=2.,collisions=0,nominal_misses=0))
    high_J=dict(index=1,result=dict(complete=True,J_hat=3.,collisions=2,nominal_misses=12))
    incomplete=dict(index=2,result=dict(complete=False,J_hat=100.,collisions=0))
    assert best_candidate([safe,high_J,incomplete]) is high_J
    assert candidate_score(dict(result=None)) is None
    assert candidate_score(dict(result=dict(complete=True,J_hat=float('nan')))) is None
