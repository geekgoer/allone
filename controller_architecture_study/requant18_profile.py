"""Read the frozen release001 identity and measured service profiles offline."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class ControllerMode:
    numeric_id: int
    name: str
    service_ms: float
    selection_mse: float
    check_mse: float


def read_profile(delivery, *, statistic='max', service_factor=1., deployment=None):
    """Selection uses TRAIN512; final480 is reserved for descriptive reporting.

    Service is the board service wall time. Even its measured maximum is an
    empirical profile, not WCET. Network and PS costs belong to the caller.
    """
    root = Path(delivery)
    package = Path(deployment) if deployment is not None else root/'board_package_requant18_release001'
    manifest = json.loads((package/'payload_manifest.json').read_text(encoding='utf-8'))
    for name, digest in manifest['files'].items():
        path = (package/name).resolve()
        if not path.is_relative_to(package.resolve()):
            raise ValueError('Payload path escapes the deployment package')
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Payload hash mismatch: {name}')
    registry = json.loads((package/'dpu_model_registry.json').read_text(encoding='utf-8'))
    summary = json.loads((root/'results/release001_board480/summary.json').read_text(encoding='utf-8'))
    freeze_hash = hashlib.sha256((root/'release_001/freeze.json').read_bytes()).hexdigest()
    if (registry['wire_tag_hex'] != summary['wire_tag_hex']
            or registry['freeze_sha256'] != summary['freeze_sha256']
            or freeze_hash != registry['freeze_sha256']
            or registry['contract'] != 'decision_convex_local_relative_pv_237d_v1'):
        raise ValueError('Release identity mismatch')
    csv_path = root/'results/release001_board480/requests.csv'
    if hashlib.sha256(csv_path.read_bytes()).hexdigest() != summary['requests_csv_sha256']:
        raise ValueError('Board CSV hash mismatch')
    if statistic not in ('p99', 'max') or not math.isfinite(service_factor) or service_factor < 1.:
        raise ValueError('Use p99 or max and a service multiplier >= 1')
    measured = {item['model_id']: item for item in summary['models']}
    modes = []
    for item in registry['models']:
        row = measured[item['numeric_id']]
        if row['status'] != 'PASS' or row['counts']['success'] != 480:
            raise ValueError(f"Incomplete board evidence for {item['numeric_id']}")
        values = (row['board_service_us'][statistic]/1000*service_factor,
                  item['train_selection']['teacher_std_mse'],
                  row['board_vs_teacher']['standardized_mse'])
        if not all(math.isfinite(x) and x >= 0 for x in values) or values[0] <= 0:
            raise ValueError('Invalid model metrics')
        modes.append(ControllerMode(item['numeric_id'], item['id'], *values))
    if {m.numeric_id for m in modes} != set(range(1, 19)) or len(modes) != 18:
        raise ValueError('Expected exactly 18 numeric identities')
    identity = dict(tag=registry['wire_tag_hex'], port=registry['port'],
                    contract=registry['contract'], freeze_sha256=freeze_hash,
                    statistic=statistic, service_factor=service_factor,
                    scope='empirical serialized board service; TRAIN512 model selection',
                    input_boundary37_retested=False)
    return tuple(sorted(modes, key=lambda m: m.service_ms)), identity


def pareto(modes):
    return tuple(m for m in modes if not any(
        n.service_ms <= m.service_ms and n.selection_mse <= m.selection_mse
        and (n.service_ms < m.service_ms or n.selection_mse < m.selection_mse)
        for n in modes))


def profile_dict(modes):
    return [asdict(mode) for mode in modes]
