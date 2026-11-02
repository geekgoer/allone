import json

import numpy as np
import pytest
import torch

from controller_architecture_study.controller_selector.data import (
    VERSION, PROTOCOL, architecture, recovery_label, paired_groups, simultaneous_calibration)
from controller_architecture_study.controller_selector.model import MarginPredictor
from controller_architecture_study.controller_selector.predict import NumpyPredictor, Selector


def trace(position_error=.1, velocity_error=.2, collision=False):
    return [dict(t=float(t), position=[5*t, 0., -4.], reference_position=[5*t, 0., -4.],
        collision=collision and t >= 1., box_margin_m=.2,
        position_error_m=position_error, velocity_error_mps=velocity_error) for t in np.linspace(0., 2., 201)]


def test_recovery_margin_separates_control_failure_from_missing_timing_data():
    assert recovery_label(trace(), [])['margin'] > 0.
    assert recovery_label(trace(position_error=.4), [])['margin'] < 0.
    collision = recovery_label(trace(collision=True), [], error_kind='flight')
    assert collision['valid'] and collision['margin'] <= -1.
    assert recovery_label(trace(), [], error_kind='timing')['margin'] is None


def test_moving_away_from_reference_is_not_recovery():
    rows = trace()
    for row in rows:
        row['position'][0] *= -.1
    assert recovery_label(rows, [])['components']['progress'] < 0.


def test_incomplete_pair_block_is_not_used_as_a_negative_label():
    names = ['plain_mlp_h2_w128_seed0', 'plain_mlp_h3_w256_seed0']
    row = dict(group='g', split='train', model=names[0], requested_delay_s=.02,
               state_feature=np.zeros(241).tolist(), label=dict(valid=True))
    used, rejected = paired_groups([row], names, [.02])
    assert not used and 'g' in rejected
    other = {**row, 'model':names[1], 'split':'test'}
    with pytest.raises(ValueError, match='multiple'):
        paired_groups([row, other], names, [.02])


def test_group_calibration_accounts_for_the_worst_candidate_before_selection():
    target = np.zeros(40)
    pred = np.tile([.1, .7], 20)
    groups = np.repeat(np.arange(20), 2)
    q = simultaneous_calibration(pred, target, groups, .05)
    assert q['available'] and q['correction'] == pytest.approx(.7)
    assert not simultaneous_calibration(pred[:4], target[:4], groups[:4], .05)['available']


def test_numpy_and_scripted_exports_match_shared_torch_scores(tmp_path):
    torch.manual_seed(3)
    m = MarginPredictor().eval()
    assert sum(p.numel() for p in m.parameters()) == 8673
    x = np.random.default_rng(2).normal(size=241).astype(np.float32)
    c = np.random.default_rng(4).normal(size=(8, 7)).astype(np.float32)
    with torch.inference_mode():
        expected = m.score_candidates(torch.from_numpy(x), torch.from_numpy(c)).numpy()
        scripted = torch.jit.script(m).score_candidates(torch.from_numpy(x), torch.from_numpy(c)).numpy()
    path = tmp_path/'weights_fp32.npz'
    np.savez(path, mean=np.zeros(241, np.float32), std=np.ones(241, np.float32),
             **{k:v.detach().numpy() for k,v in m.state_dict().items()})
    np.testing.assert_allclose(NumpyPredictor(path).predict(x,c), expected, rtol=2e-5, atol=1e-6)
    np.testing.assert_allclose(scripted, expected, rtol=2e-5, atol=1e-6)


def test_selector_respects_time_mask_and_unavailable_calibration(tmp_path):
    model = MarginPredictor()
    for p in model.parameters():
        p.data.zero_()
    model.head[-1].bias.data.fill_(1.)
    np.savez(tmp_path/'weights_fp32.npz', mean=np.zeros(241, np.float32), std=np.ones(241, np.float32),
             **{k:v.detach().numpy() for k,v in model.state_dict().items()})
    names = ['plain_mlp_h2_w128_seed0', 'plain_mlp_h3_w256_seed0']
    spec = dict(models=names, descriptors={m:architecture(m).tolist() for m in names},
                calibration=dict(correction=.1), measured_delay_range_s=[.01,.08], update_bound_s=.06,
                model_cost_s=dict(zip(names,[.001,.002])))
    (tmp_path/'selector_spec.json').write_text(json.dumps(spec))
    chooser = Selector(tmp_path)
    r = chooser.recommend(np.zeros(241), dict.fromkeys(names,.03), .06, dict(zip(names,[.1,.02])), .05,.001)
    assert r['model'] == names[1] and not r['configuration_installed']
    chooser.spec['calibration']['correction'] = None
    assert chooser.recommend(np.zeros(241),dict.fromkeys(names,.03),.06,dict.fromkeys(names,.02),.05,.001)['model'] is None


def test_training_export_on_synthetic_pairs_uses_no_test_labels(tmp_path):
    from controller_architecture_study.controller_selector.train import train
    names = ['plain_mlp_h2_w128_seed0','plain_mlp_h3_w256_seed0']
    plan = dict(version=VERSION, models=names, delays_s=[.02], protocol=PROTOCOL, provenance={},
                descriptors={m:architecture(m).tolist() for m in names}, model_cost_s=dict.fromkeys(names,.001))
    rng = np.random.default_rng(3)
    data = tmp_path/'data'; data.mkdir()
    (data/'plan.json').write_text(json.dumps(plan))
    rows = []
    for split, count in [('train',8),('validation',2),('calibration',20),('test',2)]:
        for i in range(count):
            x = rng.normal(size=241).tolist()
            for mi, name in enumerate(names):
                rows.append(dict(group=f'{split}_{i}', split=split, model=name, requested_delay_s=.02,
                    actual_delay_s=.02, state_feature=x, label=dict(valid=True,margin=.2*mi-.1)))
    (data/'paired_rows.json').write_text(json.dumps(rows))
    train(data,tmp_path/'a',epochs=3)
    for row in rows:
        if row['split'] == 'test':
            row['label']['margin'] = 10000.
            row['actual_delay_s'] = .9
    (data/'paired_rows.json').write_text(json.dumps(rows))
    train(data,tmp_path/'b',epochs=3)
    first = torch.load(tmp_path/'a/best_model.pt',weights_only=False)['model']
    second = torch.load(tmp_path/'b/best_model.pt',weights_only=False)['model']
    assert all(torch.equal(first[k],second[k]) for k in first)
    a = json.loads((tmp_path/'a/selector_spec.json').read_text())
    b = json.loads((tmp_path/'b/selector_spec.json').read_text())
    assert a['measured_delay_range_s'] == b['measured_delay_range_s']
    assert not json.loads((tmp_path/'a/training_report.json').read_text())['test_evaluated']
