"""Generate a separate board source tree with the hold-expiry watchdog enabled."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from .instrument_live_scheduler import instrument
from .prepare_capacity_executor_trace import replace_one


def add_critical_update_probe(code):
    code = replace_one(code, '        const int initial_model=config.at("model_id").get<int>();',
        '''        const auto probe=config.at("critical_update_probe");
        const auto probe_vehicle=probe.at("vehicle").get<std::string>();
        const auto probe_cycle=probe.at("cycle").get<std::uint64_t>();
        const auto probe_count=probe.value("count",1);
        const bool omit_update=probe.at("omit_update").get<bool>();
        if(policy!="fixed"||enable_skip||config.count("initial_periods")||probe_vehicle.empty()||
           !probe.at("cycle").is_number_integer()||probe.at("cycle").get<std::int64_t>()<=0||
           (probe.count("count")&&!probe.at("count").is_number_integer())||probe_count<1||probe_count>3)
            throw std::runtime_error("Critical update probe requires fixed policy, uniform period, positive cycle and 1 to 3 updates");
        const int initial_model=config.at("model_id").get<int>();''')
    code = replace_one(code, '        config["names"]=names;',
        '''        if(!by_name.count(probe_vehicle)||(double(probe_cycle)+probe_count+1.)*config.at("period_ms").get<double>()>=duration*1000.)
            throw std::runtime_error("Critical update probe lies outside the prepared flight");
        config["names"]=names;''')
    return replace_one(code, '                // Registry publication, queued(), selected-model take(), real',
        '''                if(v->name==probe_vehicle&&slot->cycle>=probe_cycle&&slot->cycle<probe_cycle+probe_count){
                    row["critical_update_probe"]=true;
                    if(omit_update){
                        row["status"]="probe_omitted_update";
                        throw std::runtime_error("__drop__");
                    }
                }
                // Registry publication, queued(), selected-model take(), real''')


def add_command_retime_probe(code):
    code = replace_one(code, '        const int initial_model=config.at("model_id").get<int>();',
        '''        const auto& probe=config.at("command_retime_probe");
        const std::string probe_vehicle=probe.at("vehicle");
        const auto probe_cycle=probe.at("cycle").get<std::int64_t>();
        const double probe_delay_ms=probe.at("ack_delay_ms");
        const bool probe_retime=probe.at("retime");
        if(policy!="fixed"||enable_skip||!isolate_command_costs||!single_command_check||
           !config.value("retire_on_hold_expiry",false)||config.count("initial_periods")||
           !probe.at("cycle").is_number_integer()||probe_cycle<=0||!std::isfinite(probe_delay_ms)||
           probe_delay_ms<=config.at("period_ms").get<double>()||probe_delay_ms>=maximum_update_ms)
            throw std::runtime_error("Command-retime probe requires fixed policy with isolated guarded commands and a bounded test delay");
        const int initial_model=config.at("model_id").get<int>();''')
    code = replace_one(code, '        config["names"]=names;',
        '''        if(!by_name.count(probe_vehicle)||(double(probe_cycle)+1.)*config.at("period_ms").get<double>()+maximum_update_ms>=duration*1000.)
            throw std::runtime_error("Command-retime probe lies outside the prepared flight");
        config["names"]=names;''')
    code = replace_one(code, 'v->worker=std::thread([&,v]{for(;;)',
        'v->worker=std::thread([&,v]{std::uint64_t probe_version=0;for(;;)')
    code = replace_one(code, '                row["command_sent"]=true;row["command_return_proxy"]=true;',
        '''                if(ack.ping_ok&&v->name==probe_vehicle&&slot->cycle==std::uint64_t(probe_cycle)){
                    row["probe_real_ping_return_us"]=ack_us;
                    std::this_thread::sleep_for(std::chrono::microseconds(std::uint64_t(probe_delay_ms*1000.)));
                    row["command_return_us"]=ack_us=a::now_us();
                }
                row["command_sent"]=true;row["command_return_proxy"]=true;''')
    code = replace_one(code, '                const double post_sample=double(ack_us-output_ready)/1000.;',
        '''                if(row.count("probe_real_ping_return_us")&&row["status"]=="command_return_late")probe_version=v->version;
                const double post_sample=double(ack_us-output_ready)/1000.;''')
    return replace_one(code, '            raw_journal.submit(std::shared_ptr<const Json>(slot,&slot->log));executor.notify();',
        '''            if(probe_retime&&probe_version&&v->version==probe_version&&row["status"]=="dropped_inference"){
                const auto update=calendar.try_command_retime(v->name,slot->cycle,a::now_ms);
                if(!update.is_null()){row["command_retime"]=update;probe_version=0;}
            }
            raw_journal.submit(std::shared_ptr<const Json>(slot,&slot->log));executor.notify();''')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    probes = parser.add_mutually_exclusive_group()
    probes.add_argument("--critical-update-probe", action="store_true",
                        help="Generate an experiment-only fixed-policy binary that requires an explicit omission config")
    probes.add_argument("--command-retime-probe", action="store_true",
                        help="Generate an experiment-only fixed-policy binary with one controlled ACK observation delay")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    source = here.parent / "ps_board_source/20261117_executor_trace"
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest.items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Frozen source hash differs: {name}")
    args.output.mkdir(parents=True, exist_ok=False)
    for name in manifest:
        target = args.output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
    for name in ("scheduler.cpp", "scheduler.hpp", "scheduler_tests.cpp", "live_calendar.cpp",
                 "live_calendar.hpp", "live_calendar_tests.cpp", "revalidate.cpp", "revalidate.hpp",
                 "revalidate_tests.cpp", "management_timing.hpp", "snapshot_precheck_bench.cpp"):
        shutil.copyfile(here / "live" / name, args.output / "scheduler" / name)
    scheduler = args.output / "scheduler/scheduler.cpp"
    scheduler.write_text(instrument(scheduler.read_text(encoding="utf-8")),
                         encoding="utf-8", newline="\n")
    executor = args.output / "adaptive_live.hpp"
    code = executor.read_text(encoding="utf-8")
    code = replace_one(code, 'std::shared_ptr<Slot> slot;Json dispatch;',
                       'std::shared_ptr<Slot> slot;Json dispatch;\n'
                       '        std::uint64_t take_start=0,take_end=0;double take_lock_wait=0.;')
    code = replace_one(code, '                dispatch=calendar_.take(now_ms());',
                       '                take_start=now_us();\n'
                       '                dispatch=calendar_.take(double(take_start)/1000.,&take_lock_wait,now_ms);\n'
                       '                take_end=now_us();')
    code = replace_one(code, '            slot->log["dpu_dequeue_us"]=now_us();',
                       '            slot->log["dpu_dequeue_us"]=now_us();\n'
                       '            slot->log["calendar_take_start_us"]=take_start;\n'
                       '            slot->log["calendar_take_return_us"]=take_end;\n'
                       '            slot->log["calendar_take_lock_wait_us"]=take_lock_wait;\n           ')
    executor.write_text(code, encoding="utf-8", newline="\n")
    runtime = args.output / "adaptive_runtime.cpp"
    code = runtime.read_text(encoding="utf-8")
    code = replace_one(code, '        const double maximum_update_ms=',
                       '        if(!config.count("retire_on_hold_expiry"))config["retire_on_hold_expiry"]=true;\n'
                       '        const bool single_command_check=config.value("single_command_check",false);\n'
                       '        config["single_command_check"]=single_command_check;\n'
                       '        const bool isolate_command_costs=config.value("isolate_command_costs",false);\n'
                       '        config["isolate_command_costs"]=isolate_command_costs;\n'
                       '        const double maximum_update_ms=')
    for sentence in ('experimental error<0.15m and first6 reference speed-vector change<0.3m/s; separate conditional hold required',
                     'new sampled state paired with actual acknowledged held version'):
        code = replace_one(code, f'"{sentence}");', f'"{sentence}",a::now_ms());')
    code = replace_one(code, '            try{\n                const auto release_row=calendar.row(v->name,slot->cycle);',
                       '            ps_scheduler::CommandAdmission admission;\n'
                       '            try{\n                const auto release_row=calendar.row(v->name,slot->cycle);')
    code = replace_one(code, '''                const auto current=calendar.row(v->name,slot->cycle);row["calendar_after_dpu"]=current;
                due=std::uint64_t(current.at("actuation_deadline_ms").get<double>()*1000.);
                if(current.at("cancelled").get<bool>()||a::now_us()+std::uint64_t(costs.command_reserve()*1000.)>=due){
                    row["status"]="dropped_before_command";throw std::runtime_error("__drop__");}''',
        '''                std::uint64_t row_start=0,row_end=0,check=0,reserve=0,reserve_read=0;
                double row_lock_wait=0.;bool cancelled=false,reject=false;
                const auto log_precheck=[&]{
                    if(single_command_check)return;
                    row["command_precheck"]={{"row_start_us",row_start},{"row_return_us",row_end},
                        {"row_lock_wall_us",row_lock_wait},{"checked_us",check},{"reserve_return_us",reserve_read},
                        {"deadline_us",due},{"reserve_us",reserve},{"cancelled",cancelled},{"accepted",!reject}};
                };
                if(!single_command_check){
                    row_start=a::now_us();
                    const auto current=calendar.row(v->name,slot->cycle,&row_lock_wait);
                    row_end=a::now_us();row["calendar_after_dpu"]=current;
                    due=std::uint64_t(current.at("actuation_deadline_ms").get<double>()*1000.);
                    cancelled=current.at("cancelled").get<bool>();
                    check=a::now_us();
                    reserve=cancelled?0:std::uint64_t(1000.*(isolate_command_costs?
                        calendar.command_reserve(v->name):costs.command_reserve()));
                    reserve_read=a::now_us();
                    reject=cancelled||check+reserve>=due;
                    if(reject){log_precheck();row["status"]="dropped_before_command";throw std::runtime_error("__drop__");}
                }''')
    code = replace_one(code, '''                row["command_guard_accepted"]=calendar.command_begin(v->name,slot->cycle,a::now_ms());
                if(!row.at("command_guard_accepted").get<bool>()){''',
                       '                bool guard_accepted;\n'
                       '                if(single_command_check){\n'
                       '                    admission=calendar.command_begin_current(v->name,slot->cycle);\n'
                       '                    guard_accepted=admission.accepted;\n'
                       '                }else{\n'
                       '                    guard_accepted=calendar.command_begin(v->name,slot->cycle,a::now_ms());\n'
                       '                    row["command_guard_accepted"]=guard_accepted;\n'
                       '                }\n'
                       '                if(!guard_accepted){\n'
                       '                    log_precheck();')
    code = replace_one(code, '                row["command_submit_us"]=std::uint64_t(ack.submit_ns/1000);',
                       '                log_precheck();\n'
                       '                row["command_submit_us"]=std::uint64_t(ack.submit_ns/1000);')
    code = replace_one(code, 'row["status"]=ack_us<due?"delivered":"command_return_late";',
                       'row["status"]=(single_command_check?ack_us/1000.<admission.deadline_ms:ack_us<due)'
                       '?"delivered":"command_return_late";')
    code = replace_one(code, '''                costs.observe(0.,double(ack_us-output_ready)/1000.,0.,double(ack_us-row.at("control_converted_us").get<std::uint64_t>())/1000.);''',
        '''                const double post_sample=double(ack_us-output_ready)/1000.;
                const double reserve_sample=double(ack_us-row.at("control_converted_us").get<std::uint64_t>())/1000.;
                if(isolate_command_costs)calendar.update_command_costs(v->name,post_sample,reserve_sample);
                else costs.observe(0.,post_sample,0.,reserve_sample);''')
    code = replace_one(code, '            const auto finished=a::now_us();row["slot_return_us"]=finished;',
        '''            const auto finished=a::now_us();row["slot_return_us"]=finished;
            if(admission.accepted||admission.rejection){
                row["command_guard_accepted"]=admission.accepted;
                row["command_admission"]={{"checked_us",admission.checked_ms*1000.},
                    {"deadline_us",admission.deadline_ms*1000.},{"reserve_us",admission.reserve_ms*1000.},
                    {"lock_wall_us",admission.lock_wait_us},{"accepted",admission.accepted}};
                if(admission.rejection)row["command_admission"]["rejection"]=admission.rejection;
            }''')
    code = replace_one(code, '        std::string stop_reason;',
        """        Json hold_expiries=Json::array();
        auto record_expiries=[&](const Json& events){
            for(const auto& event:events){
                auto* v=by_name.at(event.at("vehicle").get<std::string>());
                std::lock_guard<std::mutex> lock(v->mutex);
                v->retired=true;v->retirement_reason="hold_expired";
                hold_expiries.push_back(event);
            }
        };
        std::string stop_reason;""")
    code = replace_one(code, '            const auto released=calendar.release_due(double(current)/1000.,double(end)/1000.);',
        """            Json expiry_events;
            const auto released=calendar.release_due(double(current)/1000.,double(end)/1000.,&expiry_events);
            record_expiries(expiry_events);""")
    code = replace_one(code, '        raw_journal.close();plan_journal.close();',
        """        // Drain events already latched by workers before shutdown.
        record_expiries(calendar.expire_holds(std::min(a::now_ms(),double(end)/1000.)));
        writej(out+"/hold_expiries.json",hold_expiries);
        raw_journal.close();plan_journal.close();""")
    if args.critical_update_probe:
        code = add_critical_update_probe(code)
    if args.command_retime_probe:
        code = add_command_retime_probe(code)
    runtime.write_text(code, encoding="utf-8", newline="\n")
    manifest = {p.relative_to(args.output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(args.output.rglob("*")) if p.is_file()}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
