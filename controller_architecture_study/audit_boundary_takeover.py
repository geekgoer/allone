"""Read-only evidence audit for boundary sampling and actual rate application."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from .decision_dataset import atomic_json,sha256


def audit(root,application_name='application'):
    read=lambda p:json.loads(p.read_text())
    application_root=root/application_name
    report=dict(boundary_complete=False,ranker_frozen=False,application_complete=False,
        actual_application_observed=False,hard_realtime_certified=False,application_directory=str(application_root))
    if (root/'boundary/report.json').exists():
        boundary=read(root/'boundary/report.json')
        report.update(boundary_complete=boundary['complete'],boundary_counts=boundary['counts_by_split'],
            boundary_sha256=sha256(root/'boundary/report.json'),
            differing_groups=[r for r in boundary['groups'] if r['period_recovery_differs'] or r['model_recovery_differs']])
    if (root/'ranker/report.json').exists():
        ranker=read(root/'ranker/report.json')
        report.update(ranker_frozen=True,ranker_sha256=ranker['model_sha256'],
            fit_state_groups=ranker['fit_groups'],fit_independent_routes=ranker['fit_independent_routes'],
            boundary_confirmation=ranker['metrics'].get('boundary_confirmation'))
    if (application_root/'summary.json').exists():
        application=read(application_root/'summary.json')
        report.update(application_complete=application['complete'],
            actual_application_observed=application['recommendations_actually_executed'],
            installed_groups=application['total_installed_groups'],
            tagged_controls_delivered=application['tagged_controls_delivered'],
            changed_period_controls_delivered=application['changed_period_controls_delivered'],
            changed_release_controls_delivered=application['changed_release_controls_delivered'])
        arms={}
        for arm in ('static_selected','legacy_dynamic','reviewed_rates'):
            rows=[r for r in application['rows'] if r['arm']==arm]
            j=[r['physical_result']['short_measured_J'] for r in rows if r['physical_result']['complete']]
            arms[arm]=dict(trials=len(rows),physical_passes=sum(r['physical_result']['physical_passed'] for r in rows),
                collisions=[r['timing']['collisions'] for r in rows],all_models_frozen=all(r['models_match_frozen'] for r in rows),
                comparable_measured_J=j,mean_J_when_comparable=float(np.mean(j)) if j else None,
                nominal_misses=[r['timing']['nominal_late']+r['timing']['nominal_undelivered'] for r in rows],
                conditional_budget_late=[r['conditional_budget_late_count'] for r in rows],
                control_windows=[r['control_window_covered'] for r in rows],sensor_windows=[r['sensor_window_covered'] for r in rows],
                delivered_joint_controls=sum(r['tagged_controls_delivered'] for r in rows))
        report['arms']=arms
        decisions=[]
        for row in application['rows']:
            path=application_root/(row['name']+'_planner')/'decisions.jsonl'
            if path.exists():decisions.extend(json.loads(line) for line in path.read_text().splitlines())
        expired=[d for d in decisions if d['report'].get('reason')=='existing_unfinished_deadline_expired']
        before_release=[]
        for d in expired:
            ids=set(d['report']['expired_ids'])
            contracts=[*d['original_future'].values(),*d['pending']]
            before_release.append(any(r['sample_id'] in ids and r['deadline_s']<=r['planned_release_s'] for r in contracts))
        report['decision_diagnostics']=dict(decisions=len(decisions),
            background_accepted=sum(d['report']['accepted'] for d in decisions),
            rejection_reasons=dict(Counter(d['report'].get('reason') for d in decisions if not d['report']['accepted'])),
            expired_contract_rejections=len(expired),expired_contract_predates_release_snapshots=sum(before_release),
            scope='archived decision counts; repeated states, not independent trials or a causal overhead attribution')
        report['all_cpus_restored']=read(application_root/'cpu_placement.json')['restoration_passed']
        report['interpretation']='reviewed_rates is guarded assistance to legacy dynamic policy; repeated pilot, not exact FALCON or a statistical safety proof'
    name='stage_summary.json' if application_name=='application' else f'stage_summary_{application_root.name}.json'
    atomic_json(root/name,report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('differing_groups','boundary_confirmation')},indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--application',default='application')
    args=p.parse_args();audit(args.root,args.application)
