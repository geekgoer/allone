"""Matched archived-queue decisions; never a counterfactual flight outcome."""

import argparse
from collections import Counter
import json
from pathlib import Path

import psutil
from threadpoolctl import threadpool_limits

from .decision_dataset import atomic_json,sha256
from .dynamic_probe_common import read_json
from .framework_cpu_affinity import choose_layout,cpu_topology,set_process_cpus
from .joint_contract_probe import load_priors
from .joint_contract_shadow import decide_snapshot


def run(source,output):
    output.mkdir(parents=True,exist_ok=False)
    placement=set_process_cpus(choose_layout(cpu_topology(),psutil.Process().cpu_affinity())['frontend_cpus'])
    plan=read_json(source/'plan.json')
    from .decision_dataset import ROOT
    priors_path=ROOT/'controller_architecture_study/results/fair_pool_search_20261028_v1/strengthened/search_progress.json'
    if sha256(priors_path)!=plan['priors_sha256']:raise RuntimeError('frozen utility priors changed')
    priors=load_priors(priors_path)
    records=[json.loads(line) for line in (source/'shadow_worker/decisions.jsonl').read_text().splitlines()]
    results=[]
    with threadpool_limits(limits=1):
        for row in records:
            snapshot=dict(future=row['original_future'],pending=row['pending'],
                inflight_limit=plan['configuration']['max_inflight'])
            certs={v:dict(c) for v,c in row['barriers'].items()}
            # The old file stored age, not the explicit observation timestamp.
            # Reconstruct only that logged time, never invent a newer state.
            for c in certs.values():
                if c.get('available') and 'sample_s' not in c:
                    c['sample_s']=row['worker_begin_s']-c['state_age_s']
            arms={}
            for mode in ('strict','recovery','timing_probe'):
                proposal,info,_,_,_=decide_snapshot(snapshot,certs,row['worker_begin_s'],
                    plan['shadow_profile'],plan['costs'],priors,recover_after_miss=mode!='strict',
                    feasibility_only=mode=='timing_probe')
                arms[mode]=dict(proposal=proposal,report=info)
            results.append(dict(generation=row['generation'],arms=arms))
    summary={}
    for mode in ('strict','recovery','timing_probe'):
        reports=[r['arms'][mode]['report'] for r in results]
        accepted=[r for r in reports if r['accepted']]
        summary[mode]=dict(accepted=len(accepted),
            all_loop_future_contracts_covered=sum(r['all_loop_future_contracts_covered'] for r in accepted),
            strict_timing_passed=sum(r.get('strict_timing_passed',False) for r in accepted),
            renewed_obligations=sum(len(r['recovery']['renewals']) for r in reports),
            reasons=dict(Counter(r.get('reason') for r in reports if not r['accepted'])),
            deferred=dict(Counter(d['reason'] for r in reports for d in r['recovery']['deferred'])))
    atomic_json(output/'decisions.json',results)
    report=dict(source=str(source),source_sha256=sha256(source/'shadow_worker/decisions.jsonl'),
        samples=len(records),placement=placement,arms=summary,flight_outcome_resimulated=False,
        recommendations_applied=False,physical_safety_certified=False,test_read=False,
        scope='same archived baseline queues, empirical costs and logged certificates; not causal flight benefit')
    atomic_json(output/'report.json',report)
    print(report,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.source,args.output)


if __name__=='__main__':main()
