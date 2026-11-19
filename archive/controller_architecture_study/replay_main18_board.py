"""Read-only IP37 replay on features reconstructed from a matched AirSim run."""
import argparse
from collections import Counter
from pathlib import Path
import re

import numpy as np

from airsim_interface import RawDroneState
from .decision_dataset import atomic_json, sha256
from .dynamic_probe_common import describe, read_json
from .fast_convex_preprocess import FastReferenceWindow, encode
from .fly_decision_dnn import StoredRoutePlanner
from .ip37_shadow import make_client
from .probe_main18_matched_recovery import BUNDLE, CASES, MODELS
from .state_mode_calibration import moderate_case


def samples_from_airsim(airsim):
    setup = read_json(airsim/'setup.json')
    route, layout = moderate_case()
    if setup['route'] != route or setup['backend'] != 'airsim':
        raise ValueError('expected the matched moderate AirSim route')
    planner = StoredRoutePlanner(layout)
    window = FastReferenceWindow(planner.trajectory, planner.config)
    phase = route['reference_phase_s']
    rows = []
    for case in CASES:
        base = airsim/case['id']/MODELS[0]
        trace = {round(r['t'], 8): r for r in read_json(base/'trace.json')}
        commands = read_json(base/'commands.json')
        for idx in sorted(set(range(0, len(commands), 4)) | {len(commands)-1}):
            command = commands[idx]
            state = trace[round(command['sample_s'], 8)]
            refs = window.window(phase+command['sample_s'], include_body_rates=False)
            # Relative position features are invariant to the vehicle's map offset.
            position = np.asarray(state['position'])-state['reference_position']+refs[0].position
            raw = RawDroneState(position, np.asarray(state['quaternion']), np.asarray(state['velocity']),
                                np.asarray(state['acceleration']), np.asarray(state['angular_velocity']))
            rows.append(dict(case=case['id'], command_index=idx, sample_s=command['sample_s'],
                             raw237=encode(raw, refs).tolist()))
    return rows


def replay(airsim, client_bundle, host, port, output):
    samples = samples_from_airsim(airsim)
    client, manifest = make_client(client_bundle, host, port, 1000)
    responses = []
    try:
        if sha256(BUNDLE/'shared/normalization.json') != manifest['normalization_sha256']:
            raise ValueError('AirSim and board normalization differ')
        header = client_bundle/'src/pool237_config.h'
        entries = re.findall(r'\{(\d+), "(main18_[^"]+)"', header.read_text(encoding='utf-8'))
        models = {int(n): name for n, name in entries}
        registry = read_json(BUNDLE/'model_registry.json')['models']
        if set(models) != set(range(1, 19)) or set(models.values()) != {
                row['id'] for row in registry if row['cohort'] == 'main18'}:
            raise ValueError('board model ID mapping does not cover main18')
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(output/'inputs.json', dict(airsim=str(airsim.resolve()), stride=4, samples=samples,
                                              models=models, header_sha256=sha256(header),
                                              wire_tag_hex=manifest['wire_tag_hex']))
        order = sorted(models)
        for sample_id, sample in enumerate(samples):
            offset = sample_id % len(order)
            for model in order[offset:]+order[:offset]:
                for repeat in range(2):
                    reply = client.infer(model, sample['raw237'])
                    responses.append(dict(sample_id=sample_id, case=sample['case'], repeat=repeat, **reply))
            if (sample_id+1) % 10 == 0:
                print(f'{sample_id+1}/{len(samples)} replayed across main18', flush=True)
    finally:
        client.close()
        if (output/'inputs.json').is_file():
            atomic_json(output/'requests.json', responses)
    metrics = {}
    for model, name in models.items():
        rows = [r for r in responses if r['requested_model_id'] == model]
        good = [r for r in rows if r['status'] == 0 and r['saturated_input_channels'] == 0]
        metrics[name] = dict(model_id=model, request_count=len(rows), status_counts=dict(Counter(r['status'] for r in rows)),
            **{key: describe([r[field]/1000 for r in good])
               for key, field in (('board_service_ms', 'board_service_us'),
                                  ('dpu_api_ms', 'run_api_wall_us'), ('rtt_ms', 'rtt_us'))},
            same_model_service_ms=describe([r['board_service_us']/1000 for r in good if not r['switched']]),
            switched_service_ms=describe([r['board_service_us']/1000 for r in good if r['switched']]))
    atomic_json(output/'summary.json', dict(complete=True, sample_count=len(samples), host=host, port=port, models=metrics))
    print(f'{len(responses)} replies from {len(models)} models. Summary {output/"summary.json"}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--airsim', type=Path, required=True)
    parser.add_argument('--client-bundle', type=Path, required=True)
    parser.add_argument('--host', default='192.168.77.2')
    parser.add_argument('--port', type=int, default=5237)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    replay(args.airsim, args.client_bundle, args.host, args.port, args.output)
