"""Freeze management outliers and classify skip-candidate validation intervals."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def audit(folder):
    requests=rows(folder/'requests.jsonl')
    plans=[r for r in rows(folder/'plans.jsonl') if r.get('event')=='obligation_plan']
    candidates=[]
    for r in plans:
        if not r['decision'].get('skipped'):continue
        lo,hi=r['search_end_us'],r['end_us']
        paired=[q for q in requests if q.get('command_return_us',hi+1)<=lo
                and hi<q.get('paired_analysis_ready_us',0)]
        pending=[q for q in requests if q.get('command_submit_us',hi+1)<=lo
                 and hi<q.get('command_return_us',0)]
        status=r['install']['status']
        kind=('keep_now_feasible' if status=='revalidation_skip_no_longer_needed' else
              'waiting_new_control_observation_throughout' if paired else
              'command_pending_throughout' if pending else 'other_or_interval_boundary')
        evidence=paired if paired else pending
        candidates.append(dict(start_us=r['start_us'],search_end_us=lo,end_us=hi,status=status,
            classification=kind,matching_requests=[dict(vehicle=q['vehicle'],cycle=q['cycle'],
                command_submit_us=q.get('command_submit_us'),command_return_us=q.get('command_return_us'),
                paired_analysis_ready_us=q.get('paired_analysis_ready_us')) for q in evidence],
            proof_available=r['install'].get('refreshed_context',{}).get('proof_available')))
    return dict(run=folder.name,scope='Observed intervals only; boundary/other cases are not assigned a cause',
        classifications=dict(Counter(r['classification'] for r in candidates)),
        statuses=dict(Counter(r['status'] for r in candidates)),candidates=candidates,
        plans_sha256=hashlib.sha256((folder/'plans.jsonl').read_bytes()).hexdigest())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,nargs='+',required=True)
    p.add_argument('--outlier-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    (a.output/'candidate_context.json').write_text(json.dumps([audit(f) for f in a.data],indent=2),encoding='utf-8')
    plans=[r for r in rows(a.outlier_run/'plans.jsonl') if r.get('event')=='obligation_plan']
    worst=max(plans,key=lambda r:r['wall_ms'])
    ordinary=min((r for r in plans if r['decision'].get('checks')==worst['decision']['checks']),key=lambda r:r['wall_ms'])
    inputs=[]
    for name,row in [('worst',worst),('ordinary_same_checks',ordinary)]:
        (a.output/(name+'_row.json')).write_text(json.dumps(row,indent=2),encoding='utf-8')
        s=row['snapshot'];cfg=s['config'];skip=cfg['enable_skip']
        options=dict(now=0.,decision_ms=cfg['decision_ms'],blocked_until=s['blocked_until_ms'],
            permissions=s['permissions'],max_checks=(6 if skip else 7) if cfg.get('revalidate') else 8,
            max_candidates=12,enable_skip=skip)
        inputs.append(dict(name=name,request=dict(calendar=s['calendar'],modes=s['modes'],options=options),expected=row['decision']))
    (a.output/'replay_inputs.json').write_text(json.dumps(inputs,indent=2),encoding='utf-8')
    print('Frozen',len(inputs),'inputs; candidate interval audit saved')
