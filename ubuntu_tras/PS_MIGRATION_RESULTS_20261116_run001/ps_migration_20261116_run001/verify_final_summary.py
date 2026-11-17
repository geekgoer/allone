#!/usr/bin/env python3
"""Cross-check final CSV against the four independently saved stage analyses."""
import csv
import json
from pathlib import Path

root=Path(__file__).resolve().parent
folders=['analysis_fixed_single_formal_002','analysis_normal_dual_v4_formal_002',
         'analysis_pressure20_pair002','analysis_pressure30_formal_002']
runs=[r for d in folders for r in json.loads((root/d/'analysis.json').read_text())['runs']]
rows=list(csv.DictReader((root/'CROSS_STAGE_FINAL_001.csv').open()))
assert len(runs)==len(rows)==17
assert {Path(r['data']).name for r in runs}=={r['run'] for r in rows}
assert sum(r['counts']['recorded_slots'] for r in runs)==18004
assert sum(json.loads(r['statuses']).get('delivered',0) for r in rows)==18004
for row in rows:
    assert row['observed_complete_flight']=='True' and row['audit_passed']=='True'
    assert json.loads(row['actual_models'])=={'11':json.loads(row['statuses'])['delivered']}
    assert int(row['observed_planned_skips'])==0
    assert row['skip_recovery_experiment_validated']=='False'
print('FINAL_CROSS_STAGE_AUDIT_PASS runs=17 delivered=18004 actual_model=11 observed_skips=0')
