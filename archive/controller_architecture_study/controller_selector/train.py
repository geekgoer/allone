"""Train, group-calibrate and export the small controller qualification model."""

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from ..decision_dataset import atomic_json, sha256
from .data import VERSION, PROTOCOL, paired_groups, simultaneous_calibration
from .model import MarginPredictor
from .predict import NumpyPredictor


def arrays(rows, plan, split):
    found = [r for r in rows if r['split'] == split]
    x = np.array([r['state_feature'] for r in found], np.float32).reshape(-1, 241)
    c = np.array([plan['descriptors'][r['model']]+[
        (r['actual_delay_s'] if r['actual_delay_s'] is not None else r['requested_delay_s'])/.1,
        plan['protocol']['update_bound_s']/.1] for r in found], np.float32).reshape(-1, 7)
    y = np.array([r['label']['margin'] for r in found], np.float32)
    return found, x, c, y


def metrics(rows, prediction, *, correction=0., costs):
    y = np.array([r['label']['margin'] for r in rows])
    good = np.array(prediction)-correction >= 0.
    chosen, regret, no_oracle = [], [], 0
    # One deployment query is a scene and its prescribed delay. All candidates
    # in this block had their own independent paired physical rollout.
    groups = sorted({(r['group'], r['requested_delay_s']) for r in rows})
    for group, delay in groups:
        indices = [i for i,r in enumerate(rows) if r['group'] == group and r['requested_delay_s'] == delay]
        actual = [i for i in indices if y[i] >= 0.]
        accepted = [i for i in indices if good[i]]
        if not actual:
            no_oracle += 1
        if accepted:
            index = min(accepted, key=lambda i:costs[rows[i]['model']])
            chosen.append(dict(model=rows[index]['model'], correct=bool(y[index] >= 0.)))
            if actual and y[index] >= 0.:
                regret.append(costs[rows[index]['model']]-min(costs[rows[i]['model']] for i in actual))
    return dict(rows=len(rows), margin_mse=float(np.mean((np.array(prediction)-y)**2)) if len(y) else None,
        candidate_acceptance_fraction=float(good.mean()) if len(y) else None,
        false_qualified_fraction_among_accepted=float(np.mean(y[good] < 0.)) if good.any() else None,
        queries=len(groups), queries_with_selection=len(chosen), queries_without_true_qualified=no_oracle,
        false_selection_fraction=float(np.mean([not r['correct'] for r in chosen])) if chosen else None,
        selected_models=dict(Counter(r['model'] for r in chosen)),
        mean_cost_regret_s=float(np.mean(regret)) if regret else None,
        scope='recorded paired rollout outcomes, not a physical safety certificate')


def train(data, output, epochs=600, alpha=.05, evaluate_test=False, resume=False):
    torch.set_num_threads(1)
    torch.manual_seed(123)
    np.random.seed(123)
    plan = json.loads((data/'plan.json').read_text())
    if plan['version'] != VERSION:
        raise ValueError('selector paired recovery contract mismatch')
    raw = json.loads((data/'paired_rows.json').read_text())
    rows, rejected = paired_groups(raw, plan['models'], plan['delays_s'])
    if rejected:
        raise ValueError('exported paired rows contain an invalid group')
    counts = {s:len({r['group'] for r in rows if r['split'] == s}) for s in
              ('train', 'validation', 'calibration', 'test')}
    if counts['train'] < 8 or counts['validation'] < 2:
        raise ValueError(f'collect at least 8 training and 2 validation scene groups first: {counts}')
    output.mkdir(parents=True, exist_ok=resume)
    identity = dict(data_sha256=sha256(data/'paired_rows.json'), plan_sha256=sha256(data/'plan.json'),
                    epochs=epochs, alpha=alpha, architecture='241-32-16 + 23-16-1', seed=123)
    if resume and (output/'identity.json').exists():
        if json.loads((output/'identity.json').read_text()) != identity:
            raise ValueError('training data/config changed; use a new output directory')
    else:
        atomic_json(output/'identity.json', identity)
    split = {s:arrays(rows, plan, s) for s in ('train', 'validation', 'calibration')}
    _, x, c, y = split['train']
    mean, std = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-3)
    tensors = {s:(torch.from_numpy((xx-mean)/std), torch.from_numpy(cc), torch.from_numpy(yy))
               for s,(_,xx,cc,yy) in split.items()}
    model = MarginPredictor()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    start, best, history = 0, float('inf'), []
    checkpoint = output/'latest_checkpoint.pt'
    if resume and checkpoint.exists():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler']); torch.set_rng_state(saved['rng'])
        start, best, history = saved['epoch'], saved['best'], saved['history']
    loss_fn = nn.HuberLoss(delta=.2)
    def loss(pred, target):
        return loss_fn(pred, target)+.5*torch.mean(torch.relu(pred-target)**2)
    began = time.perf_counter()
    for epoch in range(start+1, epochs+1):
        model.train()
        xx, cc, yy = tensors['train']
        for batch in torch.randperm(len(xx)).split(128):
            optimizer.zero_grad(set_to_none=True)
            value = loss(model(xx[batch], cc[batch]), yy[batch])
            value.backward(); optimizer.step()
        scheduler.step(); model.eval()
        with torch.inference_mode():
            validation = float(loss(model(*tensors['validation'][:2]), tensors['validation'][2]))
        if validation < best:
            best = validation
            torch.save(dict(model=model.state_dict(), epoch=epoch, validation_loss=best), output/'best_model.pt')
        history.append(dict(epoch=epoch, validation_loss=validation, best=best))
        if epoch % 10 == 0 or epoch == epochs:
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                rng=torch.get_rng_state(), epoch=epoch, best=best, history=history), checkpoint)
        if epoch % 50 == 0 or epoch == epochs:
            print(f'selector epoch={epoch}/{epochs} val={validation:.6g} best={best:.6g}', flush=True)
    saved = torch.load(output/'best_model.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(saved['model']); model.eval()
    predicted = {}
    with torch.inference_mode():
        for name, (xx,cc,_) in tensors.items():
            predicted[name] = model(xx,cc).numpy()
    cal_rows = split['calibration'][0]
    correction = simultaneous_calibration(predicted['calibration'], split['calibration'][3],
                                         [r['group'] for r in cal_rows], alpha)
    weights = {k:v.detach().numpy() for k,v in model.state_dict().items()}
    np.savez_compressed(output/'weights_fp32.npz', mean=mean, std=std, **weights)
    torch.jit.script(model).save(str(output/'selector_scripted.pt'))
    numpy_model = NumpyPredictor(output/'weights_fp32.npz')
    differences = []
    for i in range(min(32, len(split['validation'][0]))):
        expected = predicted['validation'][i]
        got = numpy_model.predict(split['validation'][1][i], split['validation'][2][i:i+1])[0]
        differences.append(abs(float(expected-got)))
    if max(differences, default=0.) > 2e-4:
        raise RuntimeError('NumPy export disagrees with the trained model')
    all_delays = [r['actual_delay_s'] for r in split['train'][0] if r['actual_delay_s'] is not None]
    spec = dict(version=VERSION, models=plan['models'], descriptors=plan['descriptors'],
        model_cost_s=plan['model_cost_s'], provenance=plan['provenance'], normalization='train-only mean/std included in NPZ',
        condition='5 architecture descriptors + measured initial_delay/.1 + update_bound/.1',
        update_bound_s=PROTOCOL['update_bound_s'], measured_delay_range_s=[min(all_delays), max(all_delays)],
        calibration=correction, physical_safety_certified=False, production_approved=False,
        measured_delay_points_only=True, global_admission_required=True)
    atomic_json(output/'selector_spec.json', spec)
    correction_value = correction['correction'] if correction['available'] else float('inf')
    evaluation = {s:metrics(split[s][0], predicted[s], correction=correction_value, costs=plan['model_cost_s'])
                  for s in ('validation', 'calibration')}
    if evaluate_test:
        rr, xx, cc, yy = arrays(rows, plan, 'test')
        with torch.inference_mode():
            pp = model(torch.from_numpy((xx-mean)/std), torch.from_numpy(cc)).numpy()
        evaluation['test'] = metrics(rr, pp, correction=correction_value, costs=plan['model_cost_s'])
    report = dict(groups=counts, epoch=saved['epoch'], epochs_completed=epochs, train_seconds=time.perf_counter()-began,
        parameters=sum(p.numel() for p in model.parameters()), calibration=correction,
        numpy_max_abs_error=max(differences, default=0.), evaluation=evaluation,
        test_used_for_model_selection=False, test_evaluated=evaluate_test,
        training_policy='validation-selected; positive overestimation penalized; scene-group splits',
        online_policy_modified=False, physical_safety_certified=False)
    atomic_json(output/'training_report.json', report)
    atomic_json(output/'history.json', history)
    print(json.dumps(report, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=600); p.add_argument('--alpha', type=float, default=.05)
    p.add_argument('--resume', action='store_true'); p.add_argument('--evaluate-test', action='store_true')
    args = p.parse_args()
    if args.epochs < 1:
        p.error('epochs must be positive')
    train(args.data, args.output, args.epochs, args.alpha, args.evaluate_test, args.resume)


if __name__ == '__main__':
    main()
