#!/usr/bin/env python3
"""Explicit restricted-period condition, keeping all measured costs/models."""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--period-ms', type=float, default=20.)
a = p.parse_args()
config = json.loads(a.source.read_text(encoding='utf-8'))
if config['schema'] != 'ps-obligation-config/v1' or a.period_ms not in config['periods_ms']:
    raise ValueError('Need a period present in the frozen parent configuration')
config.update(period_ms=a.period_ms, periods_ms=[a.period_ms],
              pressure_condition='T{:g}ms/vehicle; allowed period fixed{:g}ms to isolate skip/recovery, no artificial DPU/CPU delay'.format(a.period_ms,a.period_ms),
              pressure_authorization='User explicitly allowed separate pressure configuration on2026-11-16',
              parent_config_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest())
config['enable_skip'] = False  # Runtime requires explicit --enable-skip.
with a.out.open('x', encoding='utf-8') as stream:
    json.dump(config, stream, indent=2)
    stream.write('\n')
print('PRESSURE_CONFIG_FROZEN no_weight_change=1 no_input_clipping=1 no_fake_DPU=1 period_ms={:g}'.format(a.period_ms))
