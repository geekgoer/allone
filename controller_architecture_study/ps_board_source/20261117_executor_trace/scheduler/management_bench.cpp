// Actual LiveCalendar methods with synthetic lifecycle; no DPU/RPC or flight.
#include "live_calendar.hpp"
#include "management_timing.hpp"
#include <fstream>
#include <iostream>
using ps_scheduler::Json;
int main(int argc,char**argv){try{
    if(argc!=4)throw std::runtime_error("Usage: management_bench live_config.json repeats output.json");
    std::ifstream input(argv[1]);Json supplied;input>>supplied;
    const int repeats=std::stoi(argv[2]);if(repeats<1||repeats>500)throw std::runtime_error("repeats 1..500 required");
    Json records=Json::array();
    for(int i=0;i<repeats;++i){
        Json config=supplied;config["adaptive"]=true;config["enable_skip"]=false;
        const double origin=ps_scheduler::monotonic_ms()-12.;config["origin_ms"]=origin;
        ps_scheduler::LiveCalendar rt(config,supplied.at("modes"));
        rt.release_due(origin,origin+1000.);
        rt.observe("Drone1",origin,origin+200.,0,true,"synthetic reference-matched hold");
        rt.observe("Drone2",origin,origin+200.,0,true,"synthetic reference-matched hold");
        rt.queued("Drone1",0,origin,origin+1.,origin+200.);rt.take(origin+2.);
        rt.dpu_done("Drone1",0,origin+4.,true);rt.command_begin("Drone1",0,origin+5.);
        rt.command_return("Drone1",0,origin+6.);
        rt.observe("Drone1",origin+7.,origin+200.,1,true,"synthetic acknowledged control");
        rt.finish("Drone1",0,origin+8.,"delivered");
        const auto begin=ps_scheduler::management_stamp();
        Json snapshot=rt.snapshot_profiled();
        if(snapshot.at("snapshot").is_null())throw std::runtime_error("Synthetic management context expired");
        const auto p0=ps_scheduler::management_stamp();
        Json decision=rt.plan_snapshot(snapshot.at("snapshot"));const auto p1=ps_scheduler::management_stamp();
        Json installed=rt.install_profiled(snapshot.at("snapshot"),decision);const auto done=ps_scheduler::management_stamp();
        if(!installed.value("installed",false))throw std::runtime_error("Synthetic manager did not install: "+installed.dump());
        const auto log0=ps_scheduler::management_stamp();
        // Match the original full JSON log-object construction separately from
        // snapshot/planning/commit. Serialization and disk I/O are outside it.
        Json log{{"snapshot",snapshot.at("snapshot")},{"decision",decision},{"install",installed}};
        const auto log1=ps_scheduler::management_stamp();
        Json record{{"iteration",i},{"snapshot",snapshot.at("timing")},{"planning",ps_scheduler::management_interval(p0,p1)},
            {"install",installed.at("timing")},{"pipeline",ps_scheduler::management_interval(begin,done)},
            {"log_object",ps_scheduler::management_interval(log0,log1)},{"checks",decision.at("checks")}};
        records.push_back(record);
    }
    std::ofstream output(argv[3]);output<<Json{{"scope","actual management methods, synthetic two-loop state; no DPU/RPC, no live lock contention"},
        {"records",records}}.dump(2)<<'\n';
    std::cout<<"MANAGEMENT_BENCH_PASS records="<<records.size()<<'\n';return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
