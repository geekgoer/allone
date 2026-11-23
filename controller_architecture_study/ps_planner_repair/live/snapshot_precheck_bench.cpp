// Compare both paths on the same twelve-loop calendar without RPC or DPU.
#include "live_calendar.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>

using ps_scheduler::Json;
using ps_scheduler::LiveCalendar;

int main(int argc,char** argv){try{
    const bool off_lock=argc==5&&std::string(argv[4])=="--off-lock";
    if(argc!=4&&!off_lock)throw std::invalid_argument("Usage: snapshot_precheck_bench config.json repeats timings.csv [--off-lock]");
    std::ifstream input(argv[1]);Json config;input>>config;
    const auto modes=config.at("modes");config.erase("modes");
    const int repeats=std::stoi(argv[2]);
    if(repeats<1)throw std::invalid_argument("Positive repeat count required");
    std::ofstream csv(argv[3]);
    if(!csv)throw std::runtime_error("Cannot open timings output");
    const char* feature=off_lock?"snapshot_build_off_lock":"snapshot_precheck";
    csv<<"case,"<<(off_lock?"off_lock":"precheck")<<",repeat,wall_us,held_us\n";
    const double origin=100000.,period=config.at("period_ms"),stamp=origin+.95*period;
    config["origin_ms"]=origin;config["names"]=Json::array();
    for(int i=1;i<=12;++i)config["names"].push_back("Drone"+std::to_string(i));
    config["snapshot_precheck"]=off_lock;
    config["snapshot_build_off_lock"]=false;
    for(const int missing:{1,12,0}){
        config[feature]=false;LiveCalendar baseline(config,modes);
        config[feature]=true;LiveCalendar candidate(config,modes);
        for(auto* calendar:{&baseline,&candidate}){
            calendar->release_due(stamp,stamp+period);
            for(int i=1;i<=12;++i)if(i!=missing)
                calendar->observe("Drone"+std::to_string(i),stamp,stamp+10.*period,0,true);
        }
        auto left=baseline.snapshot(stamp),right=candidate.snapshot(stamp);
        if(!left.is_null())left.at("config").erase(feature);
        if(!right.is_null())right.at("config").erase(feature);
        if(left!=right||left.is_null()!=(missing!=0))throw std::runtime_error("Snapshot equivalence failed");
        if(!left.is_null()&&left.at("calendar").at("jobs").size()!=36)
            throw std::runtime_error("Expected twelve carry jobs and two future jobs per loop");
        for(int j=-100;j<repeats;++j)for(int order:{0,1}){
            const bool enabled=(j+order)%2!=0;
            auto& calendar=enabled?candidate:baseline;
            double held=0.;
            const auto begin=std::chrono::steady_clock::now();
            const auto snapshot=calendar.snapshot(stamp,&held);
            const auto elapsed=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-begin).count();
            if(snapshot.is_null()!=(missing!=0))throw std::runtime_error("Unexpected timed snapshot result");
            if(j>=0)csv<<missing<<','<<enabled<<','<<j<<','<<elapsed<<','<<held<<'\n';
        }
    }
    if(!csv)throw std::runtime_error("Cannot write timings output");
    std::cout<<"SNAPSHOT_PRECHECK_BENCH_PASS comparisons="<<3*repeats<<'\n';return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
