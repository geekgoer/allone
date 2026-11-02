"""Read-only model-family comparison on the frozen recovery dataset.

Fits only train scenes, reports validation and calibration, leaves test unused.
Original predictor and flight policy are never overwritten.
"""

import argparse
from pathlib import Path
import time

import joblib
import numpy as np
import psutil
import sklearn
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from threadpoolctl import threadpool_limits

from ..decision_dataset import atomic_json, sha256
from ..dynamic_probe_common import read_json
from ..framework_cpu_affinity import choose_layout, cpu_topology, set_process_cpus
from .data import simultaneous_calibration
from .predict import NumpyPredictor
from .train import arrays, metrics


def compare(data, predictor, output, repeats=2000):
    output.mkdir(parents=True, exist_ok=False)
    placement = set_process_cpus(choose_layout(cpu_topology(), psutil.Process().cpu_affinity())['frontend_cpus'])
    plan = read_json(data/'plan.json')
    # Combined file is loaded, but no test row is inspected, transformed,
    # fitted, scored or used to select thresholds/hyperparameters here.
    rows = [r for r in read_json(data/'paired_rows.json') if r['split'] != 'test']
    splits = {s:arrays(rows, plan, s) for s in ('train', 'validation', 'calibration')}
    nn = NumpyPredictor(predictor/'weights_fp32.npz')
    mean, std = nn.weights['mean'], nn.weights['std']
    joined = {s:np.concatenate(((x-mean)/std, c), axis=1) for s,(_,x,c,_) in splits.items()}
    costs = np.array([plan['model_cost_s'][m] for m in plan['models']])
    query_indices = {}
    for i,r in enumerate(splits['validation'][0]):
        query_indices.setdefault((r['group'], r['requested_delay_s']), []).append(i)
    queries = [(splits['validation'][1][ix[0]], splits['validation'][2][ix]) for ix in query_indices.values()]
    algorithms = dict(ridge=Ridge(alpha=1.), tree_depth4=DecisionTreeRegressor(max_depth=4, min_samples_leaf=12, random_state=123),
                      forest32_depth4=RandomForestRegressor(n_estimators=32, max_depth=4,
                          min_samples_leaf=12, n_jobs=1, random_state=123))
    models = {}
    with threadpool_limits(limits=1):
        for name, estimator in algorithms.items():
            start = time.perf_counter()
            estimator.fit(joined['train'], splits['train'][3])
            training_s = time.perf_counter()-start
            joblib.dump(estimator, output/(name+'.joblib'))
            predictions = {s:estimator.predict(joined[s]) for s in ('validation', 'calibration')}
            models[name] = dict(estimator=estimator, training_s=training_s, predictions=predictions)
        models['trained_neural_net'] = dict(training_s=None, predictions={s:np.array([
            nn.predict(x, c[None])[0] for x,c in zip(splits[s][1],splits[s][2])])
            for s in ('validation', 'calibration')})
        results = {}
        for name, item in models.items():
            cal = simultaneous_calibration(item['predictions']['calibration'], splits['calibration'][3],
                [r['group'] for r in splits['calibration'][0]], .05)
            if name == 'trained_neural_net':
                score = nn.predict
            elif name == 'ridge':
                estimator = item['estimator']
                def score(x, c, estimator=estimator):
                    # Share the state dot product across all candidates.
                    return ((x-mean)/std)@estimator.coef_[:241]+c@estimator.coef_[241:]+estimator.intercept_
            else:
                estimator = item['estimator']
                def score(x, c, estimator=estimator):
                    batch = np.concatenate((np.broadcast_to((x-mean)/std, (len(c),241)), c), axis=1)
                    return estimator.predict(batch)
            correction = cal['correction'] if cal['available'] else np.inf
            def predict_and_choose(query):
                x, c = query
                p = score(x, c)-correction
                mask = (p >= 0.) & (c[:, 5]*.1 <= .100)
                return int(np.argmin(np.where(mask,costs,np.inf))) if mask.any() else -1
            for i in range(100):
                predict_and_choose(queries[i % len(queries)])
            latency = []
            for i in range(repeats):
                start = time.perf_counter()
                predict_and_choose(queries[i % len(queries)])
                latency.append(1e6*(time.perf_counter()-start))
            val = splits['validation'][0]
            result = dict(training_s=item['training_s'], calibration=cal,
                validation_uncalibrated=metrics(val,item['predictions']['validation'], costs=plan['model_cost_s']),
                validation_calibrated=metrics(val,item['predictions']['validation'], correction=correction,costs=plan['model_cost_s']),
                latency_us=dict(mean=float(np.mean(latency)), p99=float(np.percentile(latency,99)), max=float(max(latency)), samples=repeats))
            if name.startswith('tree'):
                result['tree_nodes'] = item['estimator'].tree_.node_count
            if name.startswith('forest'):
                result['tree_nodes'] = sum(e.tree_.node_count for e in item['estimator'].estimators_)
            results[name] = result
        pred = models['trained_neural_net']['predictions']['calibration']
        target = splits['calibration'][3]
        worst = int(np.argmax(pred-target))
        row = splits['calibration'][0][worst]
        diagnostic = dict(group=row['group'],model=row['model'],delay_s=row['requested_delay_s'],
            target_margin=float(target[worst]),predicted_margin=float(pred[worst]),
            overestimation=float(pred[worst]-target[worst]), components=row['label'].get('components'),
            positive_target_max=float(max(r['label']['margin'] for r in rows if r['label']['margin'] >= 0.)))
        disagreement = {}
        for s,(rr,_,_,_) in splits.items():
            groups = {}
            for r in rr:
                groups.setdefault((r['group'],r['requested_delay_s']),[]).append(r)
            disagreement[s] = dict(independent_scenes=len({r['group'] for r in rr}), queries=len(groups),
                queries_with_candidate_disagreement=sum(len({r['label']['qualified'] for r in block}) > 1 for block in groups.values()))
    report = dict(scope='fixed hyperparameters, identical train-scene split and group calibration; not deployment or full algorithm ranking',
        cpu_placement=placement, sklearn=sklearn.__version__, numpy=np.__version__, candidates=len(plan['models']),
        features='same 241-state + 7 architecture/delay inputs; shared-state implementations for ridge/NN',
        latency_includes='normalization, model scoring, calibration subtraction, time mask and selection',
        latency_excludes='raw input construction, controller inference, analytic safety and full global admission',
        source_sha256=dict(data=sha256(data/'paired_rows.json'), predictor=sha256(predictor/'weights_fp32.npz')),
        models=results, worst_calibration_overestimation=diagnostic, candidate_disagreement=disagreement,
        test_evaluated=False, original_predictor_changed=False, flight_policy_changed=False)
    atomic_json(output/'report.json',report)
    print('Comparison:',output/'report.json')
    for name,row in results.items():
        print(name,'latency_us',row['latency_us'],'q',row['calibration']['correction'],
              'val_MSE',row['validation_uncalibrated']['margin_mse'],
              'selected_after_calibration',row['validation_calibrated']['queries_with_selection'])
    print('Worst calibration:',diagnostic)
    print('Candidate disagreement:',disagreement)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True); p.add_argument('--predictor',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args(); compare(a.data,a.predictor,a.output)


if __name__ == '__main__':
    main()
