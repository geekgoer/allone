#include "scheduler.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

// Finite offline calendar replay: it NEVER invokes DPU or AirSim. Records the
// actual wall cost of the C++ planner; these observations are not certified WCET.
int main(int argc,char**argv){try{
    using ps_scheduler::Json;
    if(argc!=4)throw std::invalid_argument("Usage: scheduler_bench fixture.json repeats output.csv");
    const int repeats=std::stoi(argv[2]);
    if(repeats<1||repeats>100000)throw std::invalid_argument("Expected repeats 1..100000");
    std::ifstream in(argv[1]);if(!in)throw std::runtime_error("Cannot read fixture");Json fixture;in>>fixture;
    struct Record{int repetition,seed,checks,candidates;double begin,end;bool accepted;std::string action;};
    std::vector<Record> rows;rows.reserve(std::size_t(repeats)*fixture.at("cases").size());
    // Preheat the exact path once per fixture before observations.
    for(const auto&c:fixture.at("cases"))ps_scheduler::plan(c.at("calendar"),fixture.at("modes"),Json{{"now",0.},{"decision_ms",fixture.at("decision_ms")},{"permissions",c.at("permissions")},{"enable_skip",true}});
    for(int r=0;r<repeats;++r)for(const auto&c:fixture.at("cases")){
        const Json options{{"now",0.},{"decision_ms",fixture.at("decision_ms")},{"permissions",c.at("permissions")},{"enable_skip",true}};
        const double start=ps_scheduler::monotonic_ms();
        const Json result=ps_scheduler::plan(c.at("calendar"),fixture.at("modes"),options);
        const double end=ps_scheduler::monotonic_ms();
        rows.push_back({r,c.at("seed").get<int>(),result.at("checks").get<int>(),result.at("candidates").get<int>(),start,end,result.at("accepted").get<bool>(),result.at("action").get<std::string>()});
    }
    std::ofstream out(argv[3]);if(!out)throw std::runtime_error("Cannot write CSV");out.precision(17);
    out<<"repetition,seed,plan_start_monotonic_ms,plan_end_monotonic_ms,elapsed_ms,accepted,action,checks,candidates\n";
    for(const auto&r:rows)out<<r.repetition<<','<<r.seed<<','<<r.begin<<','<<r.end<<','<<r.end-r.begin<<','<<r.accepted<<','<<r.action<<','<<r.checks<<','<<r.candidates<<'\n';
    std::cout<<"SCHEDULER_BENCH_DONE records="<<rows.size()<<" scope=offline_synthetic_calendar_no_DPU_no_AirSim\n";return 0;
}catch(const std::exception&e){std::cerr<<"ERROR "<<e.what()<<'\n';return 1;}}
