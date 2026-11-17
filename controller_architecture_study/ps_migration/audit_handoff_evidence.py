"""Offline evidence consistency audit; never contacts AirSim or a DPU service."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import struct
import sys


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def percentile(values, q):
    values = sorted(values)
    x = (len(values)-1)*q
    lo, hi = math.floor(x), math.ceil(x)
    return values[lo]+(values[hi]-values[lo])*(x-lo)


def audit(root, deployment=None):
    reference = root/'reference_assets/deployment'
    evidence = root/'evidence/release001'
    registry = read(reference/'dpu_model_registry.json')
    payload = read(reference/'payload_manifest.json')
    summary = read(evidence/'results/release001_board480/summary.json')
    norm = read(reference/'normalization.json')
    # Match the original validator and the C++ embedded float32 constants.
    output_std = [struct.unpack('<f', struct.pack('<f', v))[0] for v in norm['u_std']]
    assert registry['wire_tag_hex'] == summary['wire_tag_hex'] == '9253dd1fc87f8d7f'
    assert sha(evidence/'release_001/freeze.json') == registry['freeze_sha256'] == summary['freeze_sha256']
    assert sha(reference/'normalization.json') == registry['normalization_sha256']
    assert sha(evidence/'results/release001_board480/requests.csv') == summary['requests_csv_sha256']
    reports = {m['model_id']: m for m in summary['models']}
    values = {}; golden_rows = 0
    for model in registry['models']:
        mid = model['numeric_id']
        for name in (model['golden_file'], model['references_file']):
            assert sha(reference/name) == payload['files'][name], name
        refs = read(reference/model['references_file'])
        assert refs['model_id'] == mid and refs['model'] == model['id']
        assert len(refs['golden']) == 32 and len(refs['final480']) == 480
        golden = [[float(x) for x in line.split()] for line in (reference/model['golden_file']).read_text().splitlines() if line.strip()]
        assert len(golden) == 32
        for line, row in zip(golden, refs['golden']):
            expected = row['X_raw']+row['U_int8']+row['U_fp32']
            assert len(line) == 245 and all(math.isfinite(v) for v in line)
            assert max(abs(a-b) for a, b in zip(line, expected)) < 2e-6
        golden_rows += len(golden)
        values[mid] = {int(r['row_id']): r for r in refs['final480']}
        log = evidence/'results/release001_board_goldens'/f"{model['id']}.log"
        footprint = log.with_name(f"{model['id']}_footprint_after_load.txt")
        scales = reports[mid]['actual_getter_scales']
        assert sha(log) == scales['log_sha256'] and sha(footprint) == scales['footprint_sha256']
    service = defaultdict(list); errors = defaultdict(float); seen = defaultdict(set); max_int8 = 0.
    with (evidence/'results/release001_board480/requests.csv').open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            mid, rid = int(row['model_id']), int(row['row_id'])
            assert int(row['status']) == 0 and row['outcome'] == 'SUCCESS'
            assert int(row['actual_model_id']) == mid and int(row['saturated_input_channels']) == 0
            assert rid not in seen[mid]; seen[mid].add(rid)
            ref = values[mid][rid]
            assert ref['valid'] and row['valid_reference'] == 'True'
            raw = [float(row[f'raw{i}']) for i in range(237)]
            assert max(abs(a-b) for a, b in zip(raw, ref['X_raw'])) < 1e-12
            y = [float(row[f'board_u{i}']) for i in range(4)]
            assert all(math.isfinite(v) for v in raw+y)
            max_int8 = max(max_int8, max(abs(a-b) for a, b in zip(y, ref['U_int8'])))
            errors[mid] += sum(((y[i]-ref['U_teacher'][i])/output_std[i])**2 for i in range(4))
            cost = float(row['board_service_us'])
            assert cost == int(row['board_end_us'])-int(row['board_start_us'])
            assert float(row['run_api_wall_us']) == int(row['board_run_end_us'])-int(row['board_run_start_us'])
            service[mid].append(cost)
    assert set(service) == set(range(1, 19)) and max_int8 <= 1e-4
    for mid, costs in service.items():
        assert len(costs) == len(seen[mid]) == 480
        assert max(costs) == reports[mid]['board_service_us']['max']
        assert math.isclose(percentile(costs, .99), reports[mid]['board_service_us']['p99'], abs_tol=1e-8)
        assert math.isclose(errors[mid]/(480*4), reports[mid]['board_vs_teacher']['standardized_mse'], abs_tol=1e-12)
    flights = []
    for folder in sorted((root/'evidence/windows_reference').iterdir()):
        result, rows = read(folder/'result.json'), read(folder/'requests.json')
        counts = dict(Counter(r['status'] for r in rows))
        assert len(rows) == result['requests'] and counts == result['outcomes']
        assert result['timing_passed'] == (bool(rows) and all(r['status'] == 'delivered' for r in rows))
        flights.append(dict(name=folder.name, requests=len(rows), outcomes=counts,
                            flight_passed=result['flight_passed'], timing_passed=result['timing_passed']))
    result = dict(scope='stored evidence consistency, not a new hardware experiment', golden_rows=golden_rows,
                  formal_requests=sum(map(len, service.values())), board_int8_max_difference=max_int8,
                  cost_csv_mse_and_getter_hashes_verified=True, flights=flights,
                  external_deployment='NOT_CHECKED' if deployment is None else 'CHECKING')
    if deployment is not None:
        lock = read(root/'EXTERNAL_DEPENDENCIES.json')['deployment']
        for name, digest in lock['files'].items():
            assert sha(deployment/name) == digest, name
        sys.path.insert(0, str(root/'source'))
        from ps_core.profile import read_profile
        modes, identity = read_profile(evidence, deployment=deployment)
        assert len(modes) == 18 and identity['tag'] == registry['wire_tag_hex']
        result['external_deployment'] = 'HASH_AND_PROFILE_PASS'
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--deployment', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.root.resolve(), args.deployment), indent=2))
