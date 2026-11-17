// Reads saved calendars; never links DNNDK or AirSim/RPC code.
#include "scheduler.hpp"
#include "profile.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

namespace ps_profile { thread_local State state; }
using ps_scheduler::Json;
struct Input { std::string name; Json request, expected; };
bool equal_result(const Json& a,const Json& b) {
    if(a.is_number()&&b.is_number()) {
        const double left=a.get<double>(),right=b.get<double>();
        return std::abs(left-right)<=1e-10+1e-12*std::max(std::abs(left),std::abs(right));
    }
    if(a.is_object()&&b.is_object()) {
        if(a.size()!=b.size())return false;
        for(auto it=b.begin();it!=b.end();++it)
            if(!a.count(it.key())||!equal_result(a.at(it.key()),it.value()))return false;
        return true;
    }
    if(a.is_array()&&b.is_array()) {
        if(a.size()!=b.size())return false;
        for(std::size_t i=0;i<a.size();++i)if(!equal_result(a.at(i),b.at(i)))return false;
        return true;
    }
    return a==b;
}
int main(int argc, char** argv) { try {
    if(argc!=4) throw std::runtime_error("Usage: profile_replay inputs.json repeats output.json");
    std::ifstream stream(argv[1]); Json raw; stream>>raw;
    const int repeats=std::stoi(argv[2]);
    if(repeats<1||repeats>100) throw std::runtime_error("Expected repeats 1..100");
    std::vector<Input> inputs;
    for(const auto& item:raw) inputs.push_back({item.at("name"),item.at("request"),item.at("expected")});
    const auto run=[](const Json& request) {
        return ps_scheduler::plan(request.at("calendar"),request.at("modes"),request.at("options"));
    };
    const auto verify=[](const Json& result,const Json& expected) {
        if(result.size()!=expected.size())throw std::runtime_error("Decision field count mismatch");
        for(auto it=expected.begin();it!=expected.end();++it)
            if(!result.count(it.key())||!equal_result(result.at(it.key()),it.value()))
                throw std::runtime_error("Decision mismatch at "+it.key());
    };
    for(const auto& input:inputs) verify(run(input.request),input.expected);
    Json rows=Json::array();
    std::vector<std::size_t> order; for(std::size_t i=0;i<inputs.size();++i) order.push_back(i);
    std::mt19937 random(20261116);
    for(int repetition=0;repetition<repeats;++repetition) {
        std::shuffle(order.begin(),order.end(),random);
        for(auto index:order) {
            const auto& input=inputs[index];
            ps_profile::begin();
            const auto start=ps_profile::stamp();
            Json result=run(input.request);
            const auto end=ps_profile::stamp();
            ps_profile::end();
            verify(result,input.expected);
            Json phases=Json::object();
            for(int stage=0;stage<ps_profile::Count;++stage)
                phases[ps_profile::names[stage]]=Json{{"wall_ms",ps_profile::state.wall[stage]/1e6},
                    {"cpu_ms",ps_profile::state.cpu[stage]/1e6},{"calls",ps_profile::state.calls[stage]}};
            rows.push_back(Json{{"name",input.name},{"repetition",repetition},
                {"wall_ms",(end.wall-start.wall)/1e6},{"cpu_ms",(end.cpu-start.cpu)/1e6},
                {"phases",phases},{"checks",result.at("checks")},{"action",result.at("action")}});
        }
    }
    std::ofstream out(argv[3]); if(!out) throw std::runtime_error("Cannot write output");
    out<<Json{{"scope","Offline deterministic calendar replay; no DPU, RPC, live snapshot or commit"},
#ifdef PS_PROFILE_ENABLED
        {"instrumented",true},
#else
        {"instrumented",false},
#endif
        {"cases",inputs.size()},{"repeats",repeats},{"decision_parity",true},{"records",rows}}.dump(2)<<'\n';
    std::cout<<"PASS inputs="<<inputs.size()<<" repeats="<<repeats<<" records="<<rows.size()<<'\n';
    return 0;
} catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; } }
