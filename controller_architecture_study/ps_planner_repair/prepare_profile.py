"""Build immutable source variants and real-input replay payload, offline only."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


def prepare(delivery, output):
    output.mkdir(parents=True, exist_ok=False)
    source = delivery/'scheduler/scheduler.cpp'
    original = source.read_text(encoding='utf-8')
    profile = original.replace('#include "scheduler.hpp"', '#include "scheduler.hpp"\n#include "profile.hpp"', 1)
    replacements = [
        ('Modes parse_modes(const Json& raw) {', 'Modes parse_modes(const Json& raw) { PS_SCOPE(Preprocess);'),
        ('Choice parse_choice(const Json& raw) {', 'Choice parse_choice(const Json& raw) { PS_SCOPE(Preprocess);'),
        ('Jobs parse_jobs(const Json& raw) {', 'Jobs parse_jobs(const Json& raw) { PS_SCOPE(Preprocess);'),
        ('Json simulate(const std::vector<Demand>& jobs,const Sequence& order,double decision,double block) {',
         'Json simulate(const std::vector<Demand>& jobs,const Sequence& order,double decision,double block) { PS_SCOPE(Simulation);'),
        ('    Json slack=Json::object(),misses=', '    PS_SCOPE(SimulationReport);\n    Json slack=Json::object(),misses='),
        ('double tail(const Jobs&jobs,const Sequence&order,const Choice&choice,const Modes&modes,double now){',
         'double tail(const Jobs&jobs,const Sequence&order,const Choice&choice,const Modes&modes,double now){ PS_SCOPE(Budget);'),
        ('double dispatch(const Jobs&jobs,const Sequence&order,const Choice&choice,const Modes&modes,double now){',
         'double dispatch(const Jobs&jobs,const Sequence&order,const Choice&choice,const Modes&modes,double now){ PS_SCOPE(Budget);'),
        ('Json evaluate_impl(const Jobs&jobs,const Sequence&order,const Choice&choice,const Modes&modes,double now,double decision,double block){',
         'Json evaluate_impl(const Jobs&jobs,const Sequence&order,const Choice&choice,const Modes&modes,double now,double decision,double block){ PS_SCOPE(GlobalCheck);'),
        ('Json diagnose_impl(const Jobs&jobs,const Choice&choice,const Modes&modes,double now,double decision,double block){',
         'Json diagnose_impl(const Jobs&jobs,const Choice&choice,const Modes&modes,double now,double decision,double block){ PS_SCOPE(Diagnosis);'),
        ('void validate(const Json&calendar,const Json&raw_modes){',
         'void validate(const Json&calendar,const Json&raw_modes){ PS_SCOPE(Preprocess);'),
        ('Json plan(const Json&calendar,const Json&raw_modes,const Json&options){',
         'Json plan(const Json&calendar,const Json&raw_modes,const Json&options){ PS_SCOPE(Other);'),
        ('        if(candidates>=max_candidates||checks>=max_checks)return;',
         '        PS_SCOPE(Candidate);\n        if(candidates>=max_candidates||checks>=max_checks)return;'),
    ]
    for before, after in replacements:
        if profile.count(before) != 1:
            raise ValueError('Unknown frozen source at '+before)
        profile = profile.replace(before, after, 1)
    report_return=next(line for line in original.splitlines() if line.startswith('    return Json{{"feasible",misses.empty()}'))
    moved_report='''    Json report{{"feasible",misses.empty()},{"worst",worst.empty()?Json(nullptr):Json(worst)},
        {"switches",jobs.empty()?0:int(jobs.size())-1},{"scope","finite deterministic service envelope; not plant or infinite-horizon certificate"}};
    report["misses"]=std::move(misses); report["finish"]=std::move(finish);
    report["command_finish"]=std::move(command_finish); report["slack"]=std::move(slack);
    report["waits"]=std::move(waits); report["timeline"]=std::move(timeline);
    report["jobs"]=std::move(raw_jobs); return report;'''
    moved=original.replace(report_return,moved_report,1)
    moved_profile=profile.replace(report_return,moved_report,1)
    eval_return=next(line for line in original.splitlines() if line.startswith('    return Json{{"accepted",forecast.at('))
    moved_eval='''    Json checked{{"accepted",forecast.at("feasible").get<bool>()&&overlap.empty()}};
    checked["overlap"]=std::move(overlap);checked["forecast"]=std::move(forecast);return checked;'''
    approved_line=next(line for line in original.splitlines() if line.startswith('        approved=Json{{"accepted",true}'))
    moved_approved=approved_line.replace('{"jobs",updated},','').replace('{"forecast",result.at("forecast")},','')
    moved_approved+='\n        approved["jobs"]=std::move(updated);approved["forecast"]=std::move(result.at("forecast"));'
    for before,after in [(eval_return,moved_eval),(approved_line,moved_approved)]:
        assert moved.count(before)==moved_profile.count(before)==1
        moved=moved.replace(before,after,1)
        moved_profile=moved_profile.replace(before,after,1)
    for name, text in [('baseline', original), ('profile', profile),
                       ('move_baseline',moved), ('move_profile',moved_profile)]:
        variant = output/name
        variant.mkdir()
        (variant/'scheduler.cpp').write_text(text, encoding='utf-8')
        shutil.copyfile(delivery/'scheduler/scheduler.hpp', variant/'scheduler.hpp')
    here=Path(__file__).parent
    for name in ['profile.hpp', 'profile_replay.cpp', 'build_arm.sh']:
        shutil.copyfile(here/name, output/name)
    project=here.parent.parent
    json_header=project/'controller_architecture_study/results/ps_unified_handoff_20261116_004/route_b_unified_20261116/third_party/AirSim/MavLinkCom/common_utils/json.hpp'
    shutil.copyfile(json_header,output/'json.hpp')
    inputs=[]
    for name in ['pressure30_skip_on_001','pressure30_skip_on_002','pressure30_skip_on_003','dual_adaptive_102']:
        rows=[json.loads(line) for line in (delivery/name/'plans.jsonl').open(encoding='utf-8')]
        selected=[(i,row) for i,row in enumerate(rows) if row['decision'].get('accepted') and row['decision'].get('skipped')]
        if name=='dual_adaptive_102':
            selected=[max(enumerate(rows),key=lambda pair:pair[1]['wall_ms'])]
            for action in ['keep_current','no_feasible_checked_plan','obligation_context_not_ready']:
                match=next(((i,row) for i,row in enumerate(rows) if row.get('snapshot') and row['decision'].get('action')==action), None)
                if match and match[0]!=selected[0][0]:selected.append(match)
        for i,row in selected:
            snap=row['snapshot']
            inputs.append(dict(name='{}:{}'.format(name,i),
                request=dict(calendar=snap['calendar'],modes=snap['modes'],options=dict(now=0.,decision_ms=snap['config']['decision_ms'],
                    blocked_until=snap['blocked_until_ms'],permissions=snap['permissions'],max_checks=8,max_candidates=12,
                    enable_skip=snap['config']['enable_skip'])),
                expected=row['decision']))
    (output/'inputs.json').write_text(json.dumps(inputs,indent=2),encoding='utf-8')
    manifest={p.relative_to(output).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
              for p in output.rglob('*') if p.is_file()}
    (output/'manifest.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        cases=len(inputs),files=manifest),indent=2),encoding='utf-8')
    with tarfile.open(str(output)+'.tar.gz','w:gz') as archive:
        for path in output.rglob('*'):
            if path.is_file():archive.add(path,arcname=path.relative_to(output).as_posix())
    print('Prepared',len(inputs),'real inputs at',output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--delivery',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    prepare(args.delivery,args.output)
