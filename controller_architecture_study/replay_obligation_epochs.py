"""Compare old/new unpublished obligation binding; no AirSim, fitting or actuation."""
import argparse
from collections import Counter
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

from .decision_dataset import atomic_json,sha256
from .joint_contract_recovery import prepare_recovery


def replay(source,output):
    frozen=source/'source/joint_contract_recovery.py'
    spec=importlib.util.spec_from_file_location('frozen_obligation_recovery',frozen)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows=[];counts=Counter()
    for path in sorted(source.glob('*_planner/decisions.jsonl')):
        for line in path.read_text().splitlines():
            d=json.loads(line);future,pending=d['original_future'],d['pending']
            original=deepcopy((future,pending))
            # Identical archived clock for both versions, not exact solver entry.
            now=d['worker_begin_s'];certs=d['barriers']
            before,old=module.prepare_recovery(now,future,pending,certs)
            after,new=prepare_recovery(now,future,pending,certs)
            assert (future,pending)==original,'replay mutated published records'
            for v in future:
                for key in ('sample_id','model','planned_release_s','nominal_deadline_s'):
                    assert after[v].get(key)==future[v].get(key),'rebind changed a task identity/release/nominal deadline'
            expired=lambda block:[v for v,r in block.items() if r['deadline_s']<=now]
            counts['snapshots']+=1;counts['old_expired_future']+=len(expired(before));counts['new_expired_future']+=len(expired(after))
            counts['new_epoch_rebindings']+=sum(r['new_control_epoch'] for r in new['renewals'])
            rows.append(dict(source=str(path),generation=d['generation'],clock_s=now,
                before_expired_future=expired(before),after_expired_future=expired(after),
                previous=old,current=new,published_work_unchanged=True))
    if not rows:raise ValueError('no archived decision snapshots found')
    report=dict(status='REPLAY_INVARIANTS_PASS',counts=dict(counts),rows=rows,
        source=str(source),frozen_sha256=sha256(frozen),current_sha256=sha256(Path(__file__).with_name('joint_contract_recovery.py')),
        archived_inputs_sha256={str(p):sha256(p) for p in source.glob('*_planner/decisions.jsonl')},
        clock_proxy='archived worker_begin_s for both versions',
        scope='binding audit only; no full scheduling, current native context, flight or safety acceptance',
        actual_controls_applied=0,historical_deadlines_repaired=False)
    atomic_json(output/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();replay(a.source,a.output)
