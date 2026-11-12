"""Profile Main18 physical command predictions on frozen validation routes."""
import argparse
from pathlib import Path
import re

import numpy as np

from .decision_dataset import atomic_json, sha256
from .dynamic_probe_common import read_json
from .ip37_shadow import make_client
from .probe_main18_matched_recovery import BUNDLE

PACK = Path('F:/FalconReproduction/237D_data/board_237d_timing_20261106_pair_run001/board_237d_timing_20261106_pair_run001')


def select(profile, budget_ms):
    candidates = [row for row in profile['models'] if row['fit_coverage'] == 1.
                  and row['service_p99_ms'] <= budget_ms]
    if candidates:
        return min(candidates, key=lambda row: (row['fit_nmse'], row['service_p99_ms'], row['id']))
    return None


def run(pack, output):
    validation = BUNDLE/'shared/validation.npz'
    normalization = BUNDLE/'shared/normalization.json'
    u_std = np.asarray(read_json(normalization)['u_std'])
    registry = {r['id']: r for r in read_json(BUNDLE/'model_registry.json')['models'] if r['cohort'] == 'main18'}
    header = pack/'src/pool237_config.h'
    ids = {int(i): name for i, name in re.findall(r'\{(\d+), "(main18_[^"]+)"', header.read_text(encoding='utf-8'))}
    if set(ids) != set(range(1, 19)) or set(ids.values()) != set(registry):
        raise ValueError('unexpected Main18 model mapping')
    order = sorted(ids)
    with np.load(validation, allow_pickle=False) as saved:
        route, env = saved['route_id'], saved['environment_id']
        rng = np.random.default_rng(20261111)
        indices = rng.permutation(np.concatenate([
            rng.choice(np.flatnonzero(route == r), 8, replace=False) for r in np.unique(route)]))
        x, y, rows = saved['X'][indices], saved['U'][indices], saved['row_id'][indices]
        fit = np.isin(env[indices], np.unique(env)[::2])
    output.mkdir(parents=True, exist_ok=False)
    client, manifest = make_client(pack, '192.168.77.2', 5237, 1500)
    try:
        if sha256(normalization) != manifest['normalization_sha256']:
            raise ValueError('validation normalization differs from the board')
        good = np.zeros((len(x), len(order)), dtype=bool)
        response = np.zeros((len(x), len(order), 4))
        service = np.zeros_like(good, dtype=float)
        with (output/'requests.jsonl').open('w', encoding='utf-8') as log:
            import json
            for i, sample in enumerate(x):
                offset = i % len(order)
                raw = sample.tolist()
                for model in order[offset:]+order[:offset]:
                    reply = client.infer(model, raw)
                    j = model-1
                    good[i,j] = reply['status'] == 0 and reply['saturated_input_channels'] == 0
                    response[i,j] = reply['U_raw_physical']
                    service[i,j] = reply['board_service_us']/1000
                    log.write(json.dumps(dict(row_id=int(rows[i]), **reply))+'\n')
                if (i+1) % 100 == 0:
                    print(f'{i+1}/{len(x)} validation inputs', flush=True)
        common = good.all(axis=1)
        error = response-y[:, None, :]
        squared = ((error/u_std)**2).mean(axis=2)
        if not (common & fit).any() or not (common & ~fit).any():
            raise ValueError('no common successful rows for both environment groups')
        models = []
        for model, name in sorted(ids.items()):
            j = model-1
            models.append(dict(id=model, name=name, macs=registry[name]['mac_count'],
                fit_coverage=float(good[fit,j].mean()), eval_coverage=float(good[~fit,j].mean()),
                service_p99_ms=float(np.percentile(service[fit & good[:,j],j], 99)),
                eval_service_p99_ms=float(np.percentile(service[~fit & good[:,j],j], 99)),
                fit_nmse=float(squared[fit & common,j].mean()), eval_nmse=float(squared[~fit & common,j].mean()),
                eval_rmse_physical=np.sqrt((error[~fit & common,j]**2).mean(axis=0)).tolist()))
        profile = dict(models=models, wire_tag=manifest['wire_tag_hex'], validation_sha256=sha256(validation),
            source_sha256=sha256(Path(__file__)), sample_count=len(x), fit_rows=int(fit.sum()),
            eval_rows=int((~fit).sum()), fit_common=int((fit & common).sum()),
            eval_common=int((~fit & common).sum()), fit_environments=np.unique(env[indices][fit]).tolist(),
            eval_environments=np.unique(env[indices][~fit]).tolist())
        profile['choices'] = [dict(budget_ms=b, model=select(profile,b)) for b in (.1,.2,.5,1.,2.,4.,8.,15.)]
        atomic_json(output/'profile.json', profile)
        print(output/'profile.json', flush=True)
    finally:
        client.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pack', type=Path, default=PACK)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.pack, args.output)
