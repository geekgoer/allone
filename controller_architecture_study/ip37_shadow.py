"""IP37 shadow inference on archived AirSim states and live probe inputs."""
import argparse
from collections import Counter
import importlib.util
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from .convex_features import CONTRACT
from .dynamic_probe_common import DATA, PLAIN, describe, read_json, sha256, atomic_json


def make_client(bundle, host, port, timeout_ms):
    files = ('manifest.json', 'windows/pool237_udp.py', 'normalization.json')
    hashes = {name.removeprefix('./'): digest for digest, name in
              (line.split(maxsplit=1) for line in (bundle / 'SHA256SUMS.txt').read_text().splitlines()
               if line.strip())}
    for name in files:
        if sha256(bundle / name) != hashes[name]:
            raise ValueError(f'IP37 bundle hash mismatch: {name}')
    manifest = read_json(bundle / 'manifest.json')
    norm = read_json(bundle / 'normalization.json')
    if (norm['input_contract'] != CONTRACT
            or hashes['normalization.json'] != manifest['normalization_sha256']):
        raise ValueError('IP37 bundle has the wrong input contract')
    spec = importlib.util.spec_from_file_location('ip37_verified_codec', bundle / files[1])
    codec = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(codec)
    # This Windows Python's monotonic() ticks at ~16ms. Use QPC for the
    # unchanged codec's timeout accounting, wire timestamp and local RTT.
    codec.monotonic_us = lambda: perf_counter_ns() // 1000
    port = manifest.get('port', 5237) if port is None else port
    if not 1 <= port <= 65535:
        raise ValueError('Invalid IP37 port')
    client = codec.Client(host, port, int(manifest['wire_tag_hex'], 16), timeout_ms)
    client.input_mean = np.asarray(norm['x_mean'], dtype=np.float32)
    client.input_std = np.asarray(norm['x_std'], dtype=np.float32)
    client.input_scales = {m['numeric_id']: np.float32(m.get('runtime_input_scale') or m['expected_input_scale'])
                           for m in manifest['models']}
    client.rejected_inputs = []
    return client, manifest


def predict(client, model, feature, *, record_input=False):
    raw = feature.tolist()
    response = client.infer(model, raw)
    if response['status'] or response['saturated_input_channels']:
        error = f"IP37 status={response['status']}, saturated channels={response['saturated_input_channels']}"
        if response['status'] == 3:
            scaled = ((feature - client.input_mean) / client.input_std) * client.input_scales[model]
            over = np.flatnonzero((scaled >= 128.) | (scaled <= -129.) | ~np.isfinite(scaled))
            channels = [(int(i), float(feature[i]), float(scaled[i])) for i in over]
            error += f"; local estimate (index, raw, scaled)={channels}"
            client.rejected_inputs.append(dict(model_id=model, sequence=response['sequence'],
                                               channels=channels, raw237=raw))
        raise ValueError(error)
    if record_input:
        response['raw237'] = raw
    return np.asarray(response.pop('U_raw_physical')), response


def network_summary(rows):
    ready = [r for r in rows if r['status'] == 'ready']
    return {
        'requested': len(rows), 'ready': len(ready),
        'errors': sum(r['status'] == 'inference_error' for r in rows),
        'busy_drops': sum(r['status'] == 'dropped_busy' for r in rows),
        **{key: describe([r[field] / 1000 for r in ready])
           for key, field in (('rtt_ms', 'rtt_us'), ('board_service_ms', 'board_service_us'),
                              ('dpu_api_ms', 'run_api_wall_us'))},
        'rtt_minus_board_service_ms': describe(
            [(r['rtt_us'] - r['board_service_us']) / 1000 for r in ready]),
    }


def archived_inputs(archive, per_group):
    """Use raw237 saved by controller_selector.collect, without reconstructing state."""
    features, samples = [], []
    counts = Counter()
    seen = set()
    for path in sorted(archive.glob('calibration_*/attempt_*/report.json')):
        row = read_json(path)
        group = row['group']
        if (row['split'] != 'calibration' or counts[group] >= per_group
                or row.get('error_kind') in ('infrastructure', 'interrupted')):
            continue
        if not row.get('simulation_stepped') or row.get('state_feature') is None:
            continue
        selector_feature = np.asarray(row['state_feature'], dtype=np.float32)
        if selector_feature.shape != (241,) or not np.isfinite(selector_feature).all():
            raise ValueError(f'Expected raw237 plus the 4 held control values: {path}')
        x = selector_feature[:237]  # Same slice used by the original DNN call.
        if x.tobytes() in seen:
            continue
        seen.add(x.tobytes())
        counts[group] += 1
        samples.append(dict(group=group, source=str(path), source_sha256=sha256(path)))
        features.append(x)
    if not features:
        raise ValueError('No valid recorded raw237 calibration inputs')
    return np.stack(features), samples


def fp32_reference(x, manifest):
    import torch
    from .eval_offline import load_checkpoint, load_normalization

    if sha256(DATA / 'normalization.json') != manifest['normalization_sha256']:
        raise ValueError('FP32 normalization does not match deployed models')
    norm = load_normalization(DATA)
    tensor = torch.from_numpy((x - norm['x_mean']) / norm['x_std'])
    reference = {}
    for item in manifest['models']:
        model_id = item['numeric_id']
        name = item['id'].removeprefix('main18_')
        path = PLAIN / name / 'best_model.pt'
        if sha256(path) != item['source_checkpoint_sha256']:
            raise ValueError(f'FP32 checkpoint does not match model {model_id}')
        net = load_checkpoint(path, device='cpu')
        with torch.inference_mode():
            reference[model_id] = net(tensor).numpy() * norm['u_std'] + norm['u_mean']
    return reference


def run(args):
    x, samples = archived_inputs(args.archive, args.per_group)
    client, manifest = make_client(args.bundle, args.board, args.port, args.timeout_ms)
    models = tuple(model['numeric_id'] for model in manifest['models'])
    rows = []
    try:
        reference = fp32_reference(x, manifest)
        args.output.mkdir(parents=True, exist_ok=False)
        for i, (feature, source) in enumerate(zip(x, samples)):
            for model_id in (models if i % 2 == 0 else reversed(models)):
                row = {**source, 'model': model_id}
                try:
                    result, timing = predict(client, model_id, feature)
                    baseline = reference[model_id][i]
                    row.update(timing, status='ready', output=result.tolist(), fp32=baseline.tolist(),
                               error_physical=(result-baseline).tolist())
                except (OSError, ValueError) as exc:
                    row.update(status='inference_error', error=f'{type(exc).__name__}: {exc}')
                rows.append(row)
    finally:
        client.close()
    summary = {}
    for model in models:
        group = [row for row in rows if row['model'] == model]
        summary[model] = network_summary(group)
        errors = np.asarray([r['error_physical'] for r in group if r['status'] == 'ready'])
        if errors.size:
            summary[model].update(
                physical_rmse=np.sqrt(np.mean(errors**2, axis=0)).tolist(),
                physical_abs_p95=np.percentile(abs(errors), 95, axis=0).tolist(),
                physical_abs_max=abs(errors).max(axis=0).tolist(),
            )
    atomic_json(args.output / 'requests.json', rows)
    atomic_json(args.output / 'summary.json', {
        'scope': 'Archived AirSim recovery calibration inputs replayed sequentially on real board',
        'outputs_applied': False, 'unique_states': len(x),
        'per_group': dict(Counter(s['group'] for s in samples)),
        'models': summary,
    })
    print(f'Report: {args.output / "summary.json"}')
    if any(s['errors'] for s in summary.values()):
        raise SystemExit(2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--board', default='192.168.77.2')
    parser.add_argument('--port', type=int)
    parser.add_argument('--timeout-ms', type=float, default=100.)
    parser.add_argument('--per-group', type=int, default=4)
    run(parser.parse_args())
