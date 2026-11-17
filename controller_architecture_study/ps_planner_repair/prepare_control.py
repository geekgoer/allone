"""Prepare a separate measured-control revision from the frozen PS delivery."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
from instrument_live_scheduler import instrument


def prepare(delivery,output):
    output.mkdir(parents=True,exist_ok=False)
    live=Path(__file__).parent/'live'
    for name in ['adaptive_runtime.cpp','adaptive_live.hpp','local_dpu.cpp','local_dpu.hpp']:
        shutil.copyfile(delivery/name,output/name)
    for directory,names in [('features',['features.cpp','features.hpp']),('rpc',['airsim_rpc.cpp','airsim_rpc.hpp'])]:
        (output/directory).mkdir()
        for name in names:shutil.copyfile(delivery/directory/name,output/directory/name)
    (output/'scheduler').mkdir()
    for p in live.iterdir():
        if p.suffix in ('.cpp','.hpp'):shutil.copyfile(p,output/'scheduler'/p.name)
    shutil.copyfile(Path(__file__).parent/'profile.hpp',output/'scheduler/profile.hpp')
    scheduler=output/'scheduler/scheduler.cpp'
    scheduler.write_text(instrument(scheduler.read_text(encoding='utf-8')),encoding='utf-8')
    path=output/'adaptive_runtime.cpp'
    text=path.read_text(encoding='utf-8')
    text=text.replace('#include "adaptive_live.hpp"','#include "adaptive_live.hpp"\n#include "scheduler/management_timing.hpp"\n#include "scheduler/profile.hpp"\n#include <sys/resource.h>',1)
    before='''                Json snapshot=calendar.snapshot_current();
                Json decision=calendar.plan_snapshot(snapshot);const auto searched=a::now_us();
                Json installed=calendar.install(snapshot,decision);const auto ended=a::now_us();
                if(installed.value("installed",false))executor.notify();
                log.submit(Json{{"event","obligation_plan"},{"start_us",begin},{"search_end_us",searched},{"end_us",ended},
                    {"cpu_ms",thread_cpu_ms()-cpu},{"wall_ms",double(ended-begin)/1000.},{"snapshot",std::move(snapshot)},
                    {"decision",std::move(decision)},{"install",std::move(installed)}});'''
    after='''                Json measured=calendar.snapshot_profiled();
                Json snapshot=std::move(measured["snapshot"]);
                const auto paired_before=measured.at("paired_generation").get<std::uint64_t>();
                rusage usage_before{},usage_after{};
                const bool usage_before_ok=getrusage(RUSAGE_THREAD,&usage_before)==0;
                const auto plan_begin=ps_scheduler::management_stamp();
                ps_profile::begin();
                Json decision=calendar.plan_snapshot(snapshot);const auto searched=a::now_us();
                ps_profile::end();
                const auto plan_end=ps_scheduler::management_stamp();
                const bool usage_after_ok=getrusage(RUSAGE_THREAD,&usage_after)==0;
                Json installed=calendar.install_with_revalidation(snapshot,decision);const auto ended=a::now_us();
                const double used_cpu=thread_cpu_ms()-cpu;
                const std::string install_status=installed.value("status",std::string());
                const bool retry_pair=install_status=="revalidation_context_not_ready"||
                    install_status=="revalidation_delivery_in_flight"||snapshot.is_null();
                if(installed.value("installed",false))executor.notify();
                Json initial_snapshot=nullptr,initial_decision=nullptr;
                if(installed.value("revalidated",false)){
                    initial_snapshot=std::move(snapshot);initial_decision=std::move(decision);
                    snapshot=std::move(installed["refreshed_context"]["snapshot"]);
                    decision=std::move(installed["revalidation"]["decision"]);
                }
                const auto log_begin=ps_scheduler::management_stamp();
                Json stages=Json::object();
                for(int stage=0;stage<ps_profile::Count;++stage)
                    stages[ps_profile::names[stage]]=Json{{"cpu_ms",ps_profile::state.cpu[stage]/1e6},
                        {"wall_ms",ps_profile::state.wall[stage]/1e6},{"calls",ps_profile::state.calls[stage]}};
                Json usage=nullptr;
                if(usage_before_ok&&usage_after_ok)
                    usage=Json{{"minor_faults",usage_after.ru_minflt-usage_before.ru_minflt},
                        {"major_faults",usage_after.ru_majflt-usage_before.ru_majflt},
                        {"voluntary_switches",usage_after.ru_nvcsw-usage_before.ru_nvcsw},
                        {"involuntary_switches",usage_after.ru_nivcsw-usage_before.ru_nivcsw}};
                Json record{{"event","obligation_plan"},{"trigger",trigger},{"paired_generation",paired_before},
                    {"start_us",begin},{"search_end_us",searched},{"end_us",ended},
                    {"cpu_ms",used_cpu},{"wall_ms",double(ended-begin)/1000.},
                    {"management_timing",Json{{"snapshot",measured.at("timing")},
                      {"planning",ps_scheduler::management_interval(plan_begin,plan_end)},
                      {"planning_stages",std::move(stages)},{"planning_rusage",std::move(usage)}}},
                    {"snapshot",std::move(snapshot)},{"decision",std::move(decision)},
                    {"install",std::move(installed)},{"initial_snapshot",std::move(initial_snapshot)},
                    {"initial_decision",std::move(initial_decision)}};
                const auto log_end=ps_scheduler::management_stamp();
                record["management_timing"]["log_object"]=ps_scheduler::management_interval(log_begin,log_end);
                record["previous_log_enqueue"]=previous_enqueue;
                const auto enqueue_begin=ps_scheduler::management_stamp();log.submit(std::move(record));
                const auto enqueue_end=ps_scheduler::management_stamp();
                previous_enqueue=Json{{"source_start_us",begin},{"timing",ps_scheduler::management_interval(enqueue_begin,enqueue_end)}};'''
    assert text.count(before)==1
    text=text.replace(before,after,1)
    before='            double next=origin_ms;'
    assert text.count(before)==1
    text=text.replace(before,'            double next=origin_ms;Json previous_enqueue=nullptr;std::string trigger="periodic";',1)
    before='double origin_ms,double interval_ms):worker_([&,adaptive,origin_ms,interval_ms]{'
    assert text.count(before)==1
    text=text.replace(before,'double origin_ms,double interval_ms,bool paired_trigger):worker_([&,adaptive,origin_ms,interval_ms,paired_trigger]{',1)
    before='''                // Match reference plan_live's post-call 10ms service spacing;
                // no fake catch-up planner bursts following an overlong search.
                next=a::now_ms()+interval_ms;'''
    after='''                // Retry a changed-context proposal when a paired observation
                // and its slot finish. One wait per call, with a 5ms minimum
                // planner spacing and the unchanged periodic fallback.
                const double completed=a::now_ms();next=completed+interval_ms;trigger="periodic";
                if(paired_trigger&&retry_pair&&calendar.wait_for_paired(paired_before,interval_ms)){
                    const double awakened=a::now_ms();
                    next=std::min(next,std::max(completed+5.,awakened));
                    if(next<completed+interval_ms)trigger="paired_observation";
                }'''
    assert text.count(before)==1
    text=text.replace(before,after,1)
    before='Planner planner(calendar,executor,plan_journal,policy=="adaptive",double(origin)/1000.,10.);'
    assert text.count(before)==1
    text=text.replace(before,'Planner planner(calendar,executor,plan_journal,policy=="adaptive",double(origin)/1000.,10.,config.value("paired_trigger",true));',1)
    before='''            // Aliasing shared ownership permits immutable queueing without a'''
    # The calendar signals the management thread from finish() only after the
    # matching observation and actual command return are both recorded.
    assert text.count(before)==1
    path.write_text(text,encoding='utf-8')
    shutil.copyfile(Path(__file__).parent/'build_control.sh',output/'build_control.sh')
    for name in ['config_dual_001.json','config_pressure30_001.json']:
        config=json.loads((delivery/name).read_text(encoding='utf-8'))
        config.update(revalidate=True,revalidation_ms=2.,paired_trigger=True)
        if 'pressure' in name:
            config['revalidation_ms']=3.5
        (output/name.replace('.json','_repair.json')).write_text(json.dumps(config,indent=2),encoding='utf-8')
        if 'pressure' in name:
            config['paired_trigger']=False
            (output/name.replace('.json','_paired_off.json')).write_text(json.dumps(config,indent=2),encoding='utf-8')
    manifest={p.relative_to(output).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
              for p in output.rglob('*') if p.is_file()}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    with tarfile.open(str(output)+'.tar.gz','w:gz') as archive:
        for p in output.rglob('*'):
            if p.is_file():archive.add(p,arcname=p.relative_to(output).as_posix())
    print('Prepared control revision files',len(manifest))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--delivery',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();prepare(args.delivery,args.output)
