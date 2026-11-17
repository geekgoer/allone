// Latest Route-B obligation-calendar runtime on PS. This is not the obsolete
// ABC runner. Native IO interfaces are adapted to Linux CLOCK_MONOTONIC, threads
// and native AirSim RPC; all planning belongs to the existing LiveCalendar port.
#include "adaptive_live.hpp"
#include "scheduler/management_timing.hpp"
#include "scheduler/profile.hpp"
#include <sys/resource.h>
#include "features/features.hpp"
#include "rpc/airsim_rpc.hpp"
#include <algorithm>
#include <cmath>
#include <csignal>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <sys/stat.h>

using Json=nlohmann::json;
namespace a=psadaptive;
namespace f=psfeatures;
volatile sig_atomic_t continue_run=1;
void interrupted(int){continue_run=0;}
Json readj(const std::string& path){std::ifstream input(path);if(!input)throw std::runtime_error("missing "+path);Json j;input>>j;return j;}
void writej(const std::string& path,const Json& j){std::ofstream output(path);output<<j.dump(2)<<'\n';if(!output)throw std::runtime_error("write failure "+path);}
template<class T,std::size_t N>Json arr(const std::array<T,N>& x){Json j=Json::array();for(const auto& v:x)j.push_back(v);return j;}
f::State state_of(const psairsim::State& x){f::State s;s.position=x.position;s.velocity=x.velocity;s.quaternion=x.orientation_wxyz;
    s.acceleration=x.acceleration;s.angular_velocity=x.angular_velocity;s.timestamp_ns=x.timestamp_ns;return s;}
double norm3(const f::Vec3& x){return std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);}
f::Vec3 minus3(const f::Vec3& x,const f::Vec3& y){return {{x[0]-y[0],x[1]-y[1],x[2]-y[2]}};}
bool stable_phase(double error,const std::array<f::Reference,21>& refs){
    double change=0.;for(std::size_t i=0;i<6;++i)change=std::max(change,norm3(minus3(refs[i].velocity,refs[0].velocity)));
    return error<.15&&change<.3;
}
void flight_guard(const f::State& state,const f::Vec3& offset,double error){
    for(double x:state.position)if(!std::isfinite(x))throw std::runtime_error("flight_guard_nonfinite_position");
    for(double x:state.velocity)if(!std::isfinite(x))throw std::runtime_error("flight_guard_nonfinite_velocity");
    for(double x:state.quaternion)if(!std::isfinite(x))throw std::runtime_error("flight_guard_nonfinite_quaternion");
    if(state.position[2]+offset[2]>-.3||!std::isfinite(error)||error>8.||norm3(state.velocity)>15.)
        throw std::runtime_error("flight_guard");
}
double thread_cpu_ms(){timespec t{};if(clock_gettime(CLOCK_THREAD_CPUTIME_ID,&t))return -1.;return t.tv_sec*1000.+t.tv_nsec/1e6;}

struct Costs {
    ps_scheduler::LiveCalendar& calendar;std::mutex mutex;
    double prep,post,rearm,reserve;
    Costs(ps_scheduler::LiveCalendar& c,const Json& config):calendar(c),prep(config.at("prep_ms").get<double>()),
        post(config.at("post_ms").get<double>()),rearm(config.at("rearm_ms").get<double>()),reserve(config.at("command_reserve_ms").get<double>()){}
    void observe(double p,double q,double r,double c){std::lock_guard<std::mutex> lock(mutex);
        prep=std::max(prep,p);post=std::max(post,q);rearm=std::max(rearm,r);reserve=std::max(reserve,c);
        calendar.update_costs(prep,post,rearm,reserve);
    }
    double command_reserve(){std::lock_guard<std::mutex> lock(mutex);return reserve;}
};
struct Vehicle {
    std::string name;f::Trajectory trajectory;std::vector<f::Gate> gates;f::Vec3 offset{};
    std::unique_ptr<f::GateHoldAnalyzer> analyzer;std::unique_ptr<psairsim::Client> client;
    std::mutex mutex;std::condition_variable cv;bool stop=false,retired=false;
    std::string retirement_reason;std::shared_ptr<a::Slot> pending;std::thread worker;
    f::State initial;f::Control held{{9.81,0,0,0}};
    std::uint64_t actuation_us=0,version=0,collision_stamp=0;
    std::vector<std::shared_ptr<a::Slot>> rows;
    ~Vehicle(){{std::lock_guard<std::mutex> lock(mutex);stop=true;}cv.notify_all();if(worker.joinable())worker.join();}
};
class Planner {
    std::atomic<bool> stop_{false},failed_{false};std::thread worker_;
public:
    Planner(ps_scheduler::LiveCalendar& calendar,a::Executor& executor,a::Journal& log,bool adaptive,
            double origin_ms,double interval_ms,bool paired_trigger):worker_([&,adaptive,origin_ms,interval_ms,paired_trigger]{
        if(!adaptive)return;
        try{
            double next=origin_ms;Json previous_enqueue=nullptr;std::string trigger="periodic";
            while(!stop_&&!log.failed()&&continue_run){
                // Bound wake latency and do not make planner service depend on
                // the foreground loop becoming idle. Search remains 8/12 bounded.
                const auto now=a::now_ms();if(now<next){std::this_thread::sleep_for(std::chrono::microseconds(
                    std::uint64_t(std::min(1.,next-now)*1000.)));continue;}
                const auto begin=a::now_us();const double cpu=thread_cpu_ms();
                Json measured=calendar.snapshot_profiled();
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
                previous_enqueue=Json{{"source_start_us",begin},{"timing",ps_scheduler::management_interval(enqueue_begin,enqueue_end)}};
                // Retry a changed-context proposal when a paired observation
                // and its slot finish. One wait per call, with a 5ms minimum
                // planner spacing and the unchanged periodic fallback.
                const double completed=a::now_ms();next=completed+interval_ms;trigger="periodic";
                if(paired_trigger&&retry_pair&&calendar.wait_for_paired(paired_before,interval_ms)){
                    const double awakened=a::now_ms();
                    next=std::min(next,std::max(completed+5.,awakened));
                    if(next<completed+interval_ms)trigger="paired_observation";
                }
            }
        }catch(const std::exception& e){failed_=true;log.submit(Json{{"event","planner_failure"},{"time_us",a::now_us()},{"error",e.what()}});}
    }){}
    bool failed()const{return failed_.load();}
    void close(){stop_=true;if(worker_.joinable())worker_.join();}
    ~Planner(){close();}
};

// Preserve unchanged frozen golden format: RAW237, selected INT8 physical4,
// followed by an optional same-source FP32 reference. Only first241 are read.
Json validate_and_warm(psdpu::LocalDpu& dpu,const std::string& package,const std::vector<int>& ids){
    const auto registry=readj(package+"/dpu_model_registry.json");Json results=Json::array();
    for(int id:ids){std::string name;
        for(const auto& m:registry.at("models"))if(m.at("numeric_id").get<int>()==id)name=m.at("id").get<std::string>();
        if(name.empty())throw std::runtime_error("registry missing model "+std::to_string(id));
        std::ifstream input(package+"/golden/"+name+".txt");if(!input)throw std::runtime_error("own model golden missing "+name);
        std::vector<psdpu::Input> inputs;std::string line;double error=0.;
        while(std::getline(input,line)){if(line.empty()||line[0]=='#')continue;std::istringstream row(line);psdpu::Input x;std::array<float,4> y;
            for(auto& v:x)if(!(row>>v))throw std::runtime_error("bad golden input "+name);
            for(auto& v:y)if(!(row>>v))throw std::runtime_error("bad golden output "+name);
            const auto r=dpu.infer(id,x);if(r.status)throw std::runtime_error("own golden rejected "+name);
            for(std::size_t c=0;c<4;++c)error=std::max(error,std::abs(double(r.output[c])-y[c]));
            inputs.push_back(x);
        }
        if(inputs.size()!=32||error>1e-4)throw std::runtime_error("own golden mismatch "+name);
        for(int k=0;k<100;++k)if(dpu.infer(id,inputs[k%inputs.size()]).status)throw std::runtime_error("warmup failed "+name);
        results.push_back({{"model_id",id},{"name",name},{"golden_rows",inputs.size()},{"warmup",100},{"max_abs",error},{"status","PASS"}});
    }
    return results;
}

int main(int argc,char** argv){
    std::string prepared,out,config_path,policy="fixed",host="192.168.77.1",package="/home/root/board_package_requant18_release001";
    double duration=20.;bool enable_skip=false,touched=false;std::unique_ptr<psairsim::Client> manager;
    try{
        for(int i=1;i<argc;++i){std::string option=argv[i];
            if(option=="--enable-skip"){enable_skip=true;continue;}
            if(++i>=argc)throw std::runtime_error("missing option value "+option);
            std::string value=argv[i];
            if(option=="--prepared")prepared=value;else if(option=="--out")out=value;else if(option=="--config")config_path=value;
            else if(option=="--policy")policy=value;else if(option=="--host")host=value;else if(option=="--package")package=value;
            else if(option=="--duration-s")duration=std::stod(value);else throw std::runtime_error("unknown option "+option);
        }
        if(prepared.empty()||out.empty()||config_path.empty()||duration<20.||!std::isfinite(duration)||(policy!="fixed"&&policy!="adaptive"))
            throw std::runtime_error("Requires --prepared JSON --config JSON --out NEWDIR --policy fixed|adaptive and >=20s");
        if(enable_skip&&policy!="adaptive")throw std::runtime_error("Skip opt-in requires adaptive policy; default is off");
        if(mkdir(out.c_str(),0775))throw std::runtime_error("output exists or cannot be created");
        std::signal(SIGINT,interrupted);std::signal(SIGTERM,interrupted);
        const auto setup=readj(prepared);Json config=readj(config_path);
        writej(out+"/prepared_source.json",setup);writej(out+"/config_source.json",config);
        if(setup.value("schema",std::string())!="ps-prepared-case/v1"||setup.value("status",std::string())!="PREPARED_PAUSED")
            throw std::runtime_error("prepared scene schema/status mismatch");
        if(config.value("schema",std::string())!="ps-obligation-config/v1"||!config.count("modes")||!config.count("provenance"))
            throw std::runtime_error("Explicit measured ps-obligation-config/v1 modes and provenance required");
        // No silent legacy Windows network_ms=3, no guessed local RPC budget.
        for(const char* k:{"prep_ms","post_ms","rearm_ms","command_reserve_ms","decision_ms"})
            if(!config.count(k)||!std::isfinite(config.at(k).get<double>())||config.at(k).get<double>()<=0.)
                throw std::runtime_error(std::string("Positive empirical cost required: ")+k);
        if(!config.count("period_ms"))config["period_ms"]=50.;
        if(!config.count("periods_ms"))config["periods_ms"]=Json::array({20.,30.,40.,50.,60.,80.});
        if(!config.count("model_id"))config["model_id"]=11;
        if(!config.count("candidate_ids"))config["candidate_ids"]=Json::array({5,10,13,8,11});
        if(!config.count("fresh_ms"))config["fresh_ms"]=80.;
        if(!config.count("maximum_update_ms"))config["maximum_update_ms"]=160.;
        if(config.value("planner_interval_ms",10.)!=10.)throw std::runtime_error("Reference planner service interval is 10ms after each call");
        const double maximum_update_ms=config.at("maximum_update_ms").get<double>();
        if(maximum_update_ms<=0.||!std::isfinite(maximum_update_ms))throw std::runtime_error("Invalid maximum update cap");
        for(double period:config.at("periods_ms").get<std::vector<double>>())
            if(period<20.||period>=maximum_update_ms)throw std::runtime_error("Native Route-B periods must be >=20ms and below maximum update cap");
        const int initial_model=config.at("model_id").get<int>();
        auto ids=config.at("candidate_ids").get<std::vector<int>>();
        if(std::find(ids.begin(),ids.end(),initial_model)==ids.end())throw std::runtime_error("Default model must be in candidate_ids");
        for(int id:ids)if(id<1||id>18)throw std::runtime_error("Unknown frozen model ID");
        config["adaptive"]=policy=="adaptive";config["enable_skip"]=enable_skip;
        const auto calibration=f::Calibration::from_json(setup.at("calibration"));
        const double reference_start=setup.value("reference_start_s",1.5);
        manager.reset(new psairsim::Client(host,41451,2000));if(!manager->is_paused())throw std::runtime_error("prepare must leave AirSim paused");
        psdpu::LocalDpu dpu;const auto identity=dpu.identity();writej(out+"/actual_dpu_identity.json",identity);
        const auto registry=readj(package+"/dpu_model_registry.json");
        for(const char* field:{"wire_tag_hex","normalization_sha256"})
            if(!config.count(field)||config.at(field)!=identity.at(field)||config.at(field)!=registry.at(field))
                throw std::runtime_error(std::string("Actual frozen model identity differs from cost config: ")+field);
        if(!config.count("freeze_sha256")||config.at("freeze_sha256")!=registry.at("freeze_sha256"))
            throw std::runtime_error("Frozen selection identity differs from config");
        for(const auto& mode:config.at("modes")){
            bool found=false;
            for(const auto& item:registry.at("models"))if(item.at("numeric_id")==mode.at("numeric_id")){
                found=true;
                if(item.at("id")!=mode.at("name")||std::abs(item.at("train_selection").at("teacher_std_mse").get<double>()-mode.at("selection_mse").get<double>())>1e-15)
                    throw std::runtime_error("Model name/TRAIN512 score changed in local mode profile");
            }
            if(!found)throw std::runtime_error("Unknown profile model identity");
        }
        const auto checked=validate_and_warm(dpu,package,ids);writej(out+"/golden.json",checked);
        a::Journal raw_journal(out+"/events.jsonl"),plan_journal(out+"/plans.jsonl");
        std::mutex ps_mutex;std::vector<std::unique_ptr<Vehicle>> vehicles;std::map<std::string,Vehicle*> by_name;
        Json initial_states=Json::array(),names=Json::array();
        const auto& case_source=setup.at("original_case");
        const double clear_width=case_source.value("net_width_m",
            case_source.at("gate_center_width_m").get<double>()-.15);
        const double clear_height=case_source.at("gate_center_height_m").get<double>()-.15;
        if(clear_width<=0.||clear_height<=0.||
           std::abs(clear_width+.15-case_source.at("gate_center_width_m").get<double>())>1e-6)
            throw std::runtime_error("Case aperture and physical gate geometry disagree");
        for(auto item=setup.at("layouts").begin();item!=setup.at("layouts").end();++item){
            auto v=std::unique_ptr<Vehicle>(new Vehicle);v->name=item.key();v->trajectory=f::Trajectory::from_layout(item.value());
            for(const auto& gate:item.value().at("gates"))v->gates.push_back(f::Gate::from_json(gate));
            for(int c=0;c<3;++c)v->offset[c]=setup.at("offsets").at(v->name).at(c).get<double>();
            v->analyzer.reset(new f::GateHoldAnalyzer(v->gates,clear_width,f::HoldBounds(),clear_height));
            v->client.reset(new psairsim::Client(host,41451,2000));const auto raw=v->client->get_state(v->name);v->initial=state_of(raw);
            const auto expected=v->trajectory.evaluate(reference_start);double dot=0.;for(int c=0;c<4;++c)dot+=raw.orientation_wxyz[c]*expected.orientation[c];
            flight_guard(v->initial,v->offset,norm3(minus3(raw.position,expected.position)));
            if(norm3(minus3(raw.position,expected.position))>.05||norm3(minus3(raw.velocity,expected.velocity))>.2||std::abs(dot)<.999)
                throw std::runtime_error("prepared initial state mismatch "+v->name);
            v->collision_stamp=v->client->get_collision(v->name).timestamp_ns;
            const auto hover=f::prepare_live_command(v->held,calibration);const auto wire=f::native_rpc_command(hover);
            const auto ack=v->client->move_rates_throttle(v->name,wire.roll_rate,wire.pitch_rate,wire.yaw_rate,wire.throttle,wire.duration);
            if(!ack.ping_ok)throw std::runtime_error("hover prime ping failed "+v->name);
            v->held=hover.effective;
            initial_states.push_back({{"vehicle",v->name},{"position",arr(raw.position)},{"velocity",arr(raw.velocity)},
                {"quaternion",arr(raw.orientation_wxyz)},{"sim_timestamp_ns",raw.timestamp_ns},{"observed_us",a::now_us()},
                {"effective_held",arr(v->held)},{"held_version",0}});
            v->rows.reserve(std::size_t(duration*1000./20.)+2);names.push_back(v->name);by_name[v->name]=v.get();vehicles.push_back(std::move(v));
        }
        if(vehicles.empty()||vehicles.size()>12)throw std::runtime_error("Capacity study accepts 1..12 prepared vehicles");
        writej(out+"/initial_states.json",initial_states);
        // All model loads, parity tests, warmup, reset and priming are outside
        // the formal measurement. Starting-clock construction is charged below.
        touched=true;manager->pause(false);const auto origin=a::now_us();const auto end=origin+std::uint64_t(duration*1e6);
        config["names"]=names;config["origin_ms"]=double(origin)/1000.;config["measurement_stop_ms"]=double(end)/1000.;
        const Json modes=config.at("modes");config.erase("modes");
        ps_scheduler::LiveCalendar calendar(config,modes);Costs costs(calendar,config);
        // Declaration order matters: workers are explicitly joined below;
        // exceptions must also destroy vehicles BEFORE executor/calendar.
        a::Executor executor(calendar,[&](int id,const psdpu::Input& raw){return dpu.infer(id,raw);});
        // Vehicles were constructed before executor for paused setup. On an
        // exceptional exit a scope guard below joins them before executor dies.
        struct JoinVehicles {std::vector<std::unique_ptr<Vehicle>>& vehicles;~JoinVehicles(){
            for(auto& v:vehicles){{std::lock_guard<std::mutex> l(v->mutex);v->stop=true;}v->cv.notify_all();}
            for(auto& v:vehicles)if(v->worker.joinable())v->worker.join();
        }} join_vehicles{vehicles};
        // The prepared state was read while paused, not at the return of the
        // unpause RPC. Do not stamp it as a fresh live observation. The planner
        // waits until each first worker publishes a real sampled pair/version0.
        for(auto& item:vehicles)item->actuation_us=origin;
        for(auto& item:vehicles){auto* v=item.get();v->worker=std::thread([&,v]{for(;;){std::shared_ptr<a::Slot> slot;
            {std::unique_lock<std::mutex> lock(v->mutex);v->cv.wait(lock,[&]{return v->stop||bool(v->pending);});
                if(!v->pending&&v->stop)return;
                slot=v->pending;v->pending.reset();}
            auto& row=slot->log;std::uint64_t sample=0,output_ready=0,ack_us=0;
            try{
                const auto release_row=calendar.row(v->name,slot->cycle);
                std::uint64_t due=std::uint64_t(release_row.at("deadline_ms").get<double>()*1000.);
                row["sample_start_us"]=sample=a::now_us();
                if(sample>=due||release_row.at("cancelled").get<bool>()){row["status"]="dropped_before_sample";throw std::runtime_error("__drop__");}
                const auto raw=v->client->get_state(v->name);row["state_ready_us"]=a::now_us();const auto state=state_of(raw);
                row["sim_timestamp_ns"]=state.timestamp_ns;row["position"]=arr(state.position);row["velocity"]=arr(state.velocity);
                row["quaternion"]=arr(state.quaternion);row["acceleration"]=arr(state.acceleration);row["angular_velocity"]=arr(state.angular_velocity);
                const auto hit=v->client->get_collision(v->name);row["collision_ready_us"]=a::now_us();
                if(hit.has_collided&&hit.timestamp_ns!=v->collision_stamp){row["collision_object"]=hit.object_name;throw std::runtime_error("collision");}
                double hold_ms=0.;row["ps_lock_requested_us"]=a::now_us();
                {std::lock_guard<std::mutex> lock(ps_mutex);row["ps_lock_acquired_us"]=a::now_us();
                    const auto refs=v->trajectory.window(double(sample-origin)/1e6+reference_start);row["reference_ready_us"]=a::now_us();
                    const auto error=norm3(minus3(state.position,refs[0].position));row["tracking_error_m"]=error;
                    row["reference_position"]=arr(refs[0].position);row["reference_velocity"]=arr(refs[0].velocity);
                    flight_guard(state,v->offset,error);
                    const auto analysis=v->analyzer->analyze(state,v->held);row["analysis_ready_us"]=a::now_us();
                    const auto deadline=f::safe_deadline(sample/1e6,a::now_us()/1e6,v->actuation_us/1e6,analysis,maximum_update_ms/1000.);
                    hold_ms=deadline.deadline_s*1000.;row["hold_deadline_us"]=std::uint64_t(std::max(0.,hold_ms)*1000.);
                    row["hold_budget_s"]=analysis.budget_s;row["held_control"]=arr(v->held);row["held_version"]=v->version;
                    row["fresh_deadline_us"]=sample+std::uint64_t(config.at("fresh_ms").get<double>()*1000.);
                    row["phase_stable"]=stable_phase(error,refs);
                    row["observe_accepted"]=calendar.observe(v->name,sample/1000.,hold_ms,v->version,row.at("phase_stable").get<bool>(),
                        "experimental error<0.15m and first6 reference speed-vector change<0.3m/s; separate conditional hold required");
                    slot->raw=f::encode(state,refs);row["raw237"]=arr(slot->raw);row["feature_ready_us"]=a::now_us();
                }
                const auto ready=row.at("feature_ready_us").get<std::uint64_t>();costs.observe(double(ready-sample)/1000.,0.,0.,0.);
                // Registry publication, queued(), selected-model take(), real
                // infer, dpu_done(), and future completion are one causal chain.
                auto future=executor.enqueue(slot,double(sample)/1000.,double(ready)/1000.,hold_ms);
                const auto reply=future.get();const auto result=reply.result;
                row["output_ready_us"]=output_ready=a::now_us();row["dpu_status"]=result.status;row["inference_dispatched"]=reply.dispatched;
                if(!reply.dispatched){row["status"]="dropped_inference";throw std::runtime_error("__drop__");}
                row["input_prep_start_us"]=result.dispatch_us;row["input_prep_end_us"]=result.prep_end_us;
                row["dpu_submit_us"]=result.run_start_us;row["dpu_result_readable_us"]=result.run_end_us;row["dpu_output_ready_us"]=result.end_us;
                row["saturation"]=result.saturation;row["switched"]=result.switched;
                if(result.status){row["rejected_channels"]=result.rejected_channels;row["status"]=result.status==3?"input_range_rejected":"dpu_error";
                    throw std::runtime_error("inference_status");}
                row["physical_output"]=arr(result.output);
                // A later absolute release can cancel an active job. Copies
                // returned by queued/take are NOT authoritative after waiting.
                const auto current=calendar.row(v->name,slot->cycle);row["calendar_after_dpu"]=current;
                due=std::uint64_t(current.at("actuation_deadline_ms").get<double>()*1000.);
                if(current.at("cancelled").get<bool>()||a::now_us()+std::uint64_t(costs.command_reserve()*1000.)>=due){
                    row["status"]="dropped_before_command";throw std::runtime_error("__drop__");}
                f::Control control;for(int c=0;c<4;++c)control[c]=result.output[c];const auto command=f::prepare_live_command(control,calibration);
                const auto wire=f::native_rpc_command(command);row["control_converted_us"]=a::now_us();
                row["native_rpc_control"]=Json::array({wire.roll_rate,wire.pitch_rate,wire.yaw_rate,wire.throttle});
                row["command_guard_start_us"]=a::now_us();
                row["command_guard_accepted"]=calendar.command_begin(v->name,slot->cycle,a::now_ms());
                if(!row.at("command_guard_accepted").get<bool>()){
                    row["status"]="dropped_before_command";throw std::runtime_error("__drop__");}
                row["command_attempt_start_us"]=a::now_us();row["command_may_have_been_sent"]=true;row["command_sent"]=nullptr;
                const auto ack=v->client->move_rates_throttle(v->name,wire.roll_rate,wire.pitch_rate,wire.yaw_rate,wire.throttle,wire.duration);
                row["command_submit_us"]=std::uint64_t(ack.submit_ns/1000);row["command_enqueued_us"]=std::uint64_t(ack.enqueued_ns/1000);
                row["command_ping_start_us"]=std::uint64_t(ack.ping_start_ns/1000);row["command_return_us"]=ack_us=std::uint64_t(ack.ping_return_ns/1000);
                row["command_sent"]=true;row["command_return_proxy"]=true;row["effective_control"]=arr(command.effective);
                if(!ack.ping_ok)throw std::runtime_error("command_ping_failed_uncertain_hold");
                // A real acknowledged command changed the held controller,
                // even when late. Never keep an old certificate after this.
                v->version=calendar.command_return(v->name,slot->cycle,ack_us/1000.);v->held=command.effective;v->actuation_us=ack_us;
                row["new_held_version"]=v->version;row["status"]=ack_us<due?"delivered":"command_return_late";
                costs.observe(0.,double(ack_us-output_ready)/1000.,0.,double(ack_us-row.at("control_converted_us").get<std::uint64_t>())/1000.);
                const auto paired_sample=a::now_us();row["paired_sample_start_us"]=paired_sample;
                const auto paired=v->client->get_state(v->name);row["paired_state_ready_us"]=a::now_us();
                const auto paired_hit=v->client->get_collision(v->name);row["paired_collision_ready_us"]=a::now_us();
                if(paired_hit.has_collided&&paired_hit.timestamp_ns!=v->collision_stamp){row["collision_object"]=paired_hit.object_name;throw std::runtime_error("collision_after_command");}
                {std::lock_guard<std::mutex> lock(ps_mutex);const auto paired_state=state_of(paired);
                    const auto refs=v->trajectory.window(double(paired_sample-origin)/1e6+reference_start);
                    flight_guard(paired_state,v->offset,norm3(minus3(paired_state.position,refs[0].position)));
                    const auto analysis=v->analyzer->analyze(paired_state,v->held);
                    const auto deadline=f::safe_deadline(paired_sample/1e6,a::now_us()/1e6,v->actuation_us/1e6,analysis,maximum_update_ms/1000.);
                    row["paired_analysis_ready_us"]=a::now_us();row["paired_hold_budget_s"]=analysis.budget_s;
                    row["paired_position"]=arr(paired_state.position);row["paired_sim_timestamp_ns"]=paired_state.timestamp_ns;
                    row["paired_observe_accepted"]=calendar.observe(v->name,paired_sample/1000.,deadline.deadline_s*1000.,v->version,
                        stable_phase(norm3(minus3(paired_state.position,refs[0].position)),refs),"new sampled state paired with actual acknowledged held version");
                }
            }catch(const std::exception& e){const std::string error=e.what();if(error!="__drop__"){
                row["error"]=error;const auto prior=row.value("status",std::string("pending"));
                if(prior=="pending")row["status"]="error";
                {std::lock_guard<std::mutex> lock(v->mutex);v->retired=true;v->retirement_reason=error;}
                try{calendar.retire(v->name);}catch(const std::exception& nested){row["retire_error"]=nested.what();}
            }}
            const auto finished=a::now_us();row["slot_return_us"]=finished;
            if(ack_us)costs.observe(0.,0.,double(finished-ack_us)/1000.,0.);
            try{row["calendar_finished"]=calendar.finish(v->name,slot->cycle,double(finished)/1000.,row.at("status").get<std::string>());}
            catch(const std::exception& e){row["finish_error"]=e.what();row["status"]="lifecycle_error";calendar.retire(v->name);
                std::lock_guard<std::mutex> lock(v->mutex);v->retired=true;v->retirement_reason=e.what();}
            // Aliasing shared ownership permits immutable queueing without a
            // large RAW237/JSON copy in the command worker.
            raw_journal.submit(std::shared_ptr<const Json>(slot,&slot->log));executor.notify();
        }});}
        Planner planner(calendar,executor,plan_journal,policy=="adaptive",double(origin)/1000.,10.,config.value("paired_trigger",true));
        writej(out+"/execution_identity.json",{{"backend","PS native IO -> latest obligation calendar -> local DNNDK -> native AirSim RPC"},
            {"stage","obligation_live"},{"policy",policy},{"enable_skip",enable_skip},{"model_id",initial_model},
            {"period_ms",config.at("period_ms")},{"duration_s",duration},{"reference_start_s",reference_start},{"origin_us",origin},
            {"first_publisher_ready_us",a::now_us()},{"clock","PS CLOCK_MONOTONIC us in raw logs; calendar ms; simulation ns separate"},
            {"dpu_scope","DPU call wall-clock, not pure hardware"},{"command_ack_scope","native command+ping return proxy, not physical actuator timestamp"},
            {"raw237_normalization","once inside unchanged frozen Pool"},{"input_clipping",false},{"planner_interval_ms",10.},
            {"planner_bounds",{{"max_checks",8},{"max_candidates",12}}},{"costs_are_empirical_not_WCET",true},
            {"live_config",config},{"initial_modes",modes},{"shared_ps_convention","serialized numerical feature/hold work; full RPC/frontend wall time conservatively charged as prep/post/rearm"},
            {"linux_priority","SCHED_OTHER default, no RT guarantee"},{"native_io_port","Linux absolute release publisher/per-vehicle native RPC ownership; calendar controls release edits"}});
        std::string stop_reason;
        while(continue_run&&!raw_journal.failed()&&!plan_journal.failed()&&!executor.failed()&&!planner.failed()){
            const auto current=a::now_us();
            // Always consume the cutoff-admitted overdue releases, even when a
            // late publisher wakes just after end. Future releases are excluded.
            const auto released=calendar.release_due(double(current)/1000.,double(end)/1000.);
            for(const auto& event:released){const auto name=event.at("vehicle").get<std::string>();auto* v=by_name.at(name);
                auto slot=std::make_shared<a::Slot>();slot->vehicle=name;slot->cycle=event.at("cycle").get<std::uint64_t>();
                slot->log={{"vehicle",name},{"cycle",slot->cycle},{"model_id",event.at("model_id")},{"status",event.at("status")},
                    {"period_ms",event.at("period_ms")},{"planned_release_us",std::uint64_t(event.at("release_ms").get<double>()*1000.)},
                    {"actual_publish_us",a::now_us()},{"period_deadline_us",std::uint64_t(event.at("deadline_ms").get<double>()*1000.)},
                    {"command_sent",false},{"calendar_release",event}};
                if(event.count("recovery_due_ms"))slot->log["recovery_due_us"]=std::uint64_t(event.at("recovery_due_ms").get<double>()*1000.);
                v->rows.push_back(slot);
                if(event.at("status")!="sampling"){
                    slot->log["slot_return_us"]=a::now_us();slot->log["calendar_finished"]=calendar.finish(name,slot->cycle,a::now_ms(),event.at("status").get<std::string>());
                    raw_journal.submit(std::shared_ptr<const Json>(slot,&slot->log));continue;
                }
                slot->log["status"]="pending";
                {std::lock_guard<std::mutex> lock(v->mutex);
                    if(v->pending)throw std::runtime_error("calendar allowed duplicate pending vehicle worker");
                    v->pending=slot;}
                v->cv.notify_one();
            }
            if(current>=end)break;
            bool all_retired=true;for(const auto& v:vehicles){std::lock_guard<std::mutex> lock(v->mutex);all_retired=all_retired&&v->retired;}
            if(all_retired){stop_reason="all_vehicles_retired";break;}
            const double next=calendar.next_wakeup_ms();
            // A planner can move next_ms earlier. Poll at <=250us rather than
            // sleeping until an obsolete pre-commit next_wakeup timestamp.
            const double delay=std::min(.25,std::max(0.,std::min(next,double(end)/1000.)-a::now_ms()));
            if(delay>0.)std::this_thread::sleep_for(std::chrono::microseconds(std::uint64_t(delay*1000.)));
        }
        planner.close();
        if(stop_reason.empty()&&!continue_run)stop_reason="interrupted";
        if(stop_reason.empty()&&(raw_journal.failed()||plan_journal.failed()))stop_reason="journal_failure";
        if(stop_reason.empty()&&executor.failed())stop_reason="executor_failure";
        if(stop_reason.empty()&&planner.failed())stop_reason="planner_failure";
        for(auto& v:vehicles){{std::lock_guard<std::mutex> lock(v->mutex);v->stop=true;}v->cv.notify_all();}
        for(auto& v:vehicles)if(v->worker.joinable())v->worker.join();
        executor.close();const auto pause_start=a::now_us();manager->pause(true);const auto pause_return=a::now_us();
        raw_journal.close();plan_journal.close();writej(out+"/calendar_final.json",calendar.state());
        Json final_states=Json::array();for(auto& v:vehicles)try{
            const auto start=a::now_us();const auto raw=v->client->get_state(v->name);const auto returned=a::now_us();const auto collision=v->client->get_collision(v->name);
            final_states.push_back({{"vehicle",v->name},{"position",arr(raw.position)},{"velocity",arr(raw.velocity)},
                {"quaternion",arr(raw.orientation_wxyz)},{"sim_timestamp_ns",raw.timestamp_ns},{"observed_us",a::now_us()},
                {"state_rpc_start_us",start},{"state_rpc_return_us",returned},{"pause_start_us",pause_start},{"pause_return_us",pause_return},
                {"collision_new",collision.has_collided&&collision.timestamp_ns!=v->collision_stamp},{"collision_object",collision.object_name}});
        }catch(const std::exception& e){final_states.push_back({{"vehicle",v->name},{"error",e.what()},{"observed_us",a::now_us()}});}
        writej(out+"/final_states.json",final_states);
        Json summary={{"scope","latest obligation calendar + real DNNDK; process completion is not flight/strategy acceptance"},
            {"policy",policy},{"enable_skip",enable_skip},{"duration_s",duration},{"early_stop",stop_reason},{"pause_start_us",pause_start},
            {"pause_return_us",pause_return},{"interrupted",!bool(continue_run)},{"journal_failed",raw_journal.failed()||plan_journal.failed()},
            {"raw_journal_rows",raw_journal.written()},{"planner_log_rows",plan_journal.written()},{"executor_failed",executor.failed()},{"executor_failure_reason",executor.failure_reason()},
            {"planner_failed",planner.failed()},{"vehicles",Json::array()}};
        std::ofstream jsonl(out+"/requests.jsonl"),csv(out+"/requests.csv");
        const std::vector<std::string> keys={"vehicle","cycle","model_id","status","period_ms","planned_release_us","actual_publish_us",
            "sample_start_us","state_ready_us","collision_ready_us","reference_ready_us","analysis_ready_us","feature_ready_us","dpu_enqueue_us",
            "dpu_dequeue_us","input_prep_start_us","input_prep_end_us","dpu_submit_us","dpu_result_readable_us","output_ready_us",
            "command_submit_us","command_return_us","slot_return_us","actuation_deadline_us","recovery_due_us","sim_timestamp_ns"};
        for(std::size_t i=0;i<keys.size();++i){if(i)csv<<',';csv<<keys[i];}csv<<'\n';
        for(const auto& v:vehicles){Json counts=Json::object(),model_counts=Json::object();
            for(const auto& slot:v->rows){const auto status=slot->log.at("status").get<std::string>();const int count=counts.value(status,0);counts[status]=count+1;
                if(slot->log.value("inference_dispatched",false)){const auto id=std::to_string(slot->log.at("model_id").get<int>());const int n=model_counts.value(id,0);model_counts[id]=n+1;}
                jsonl<<slot->log.dump()<<'\n';for(std::size_t i=0;i<keys.size();++i){if(i)csv<<',';if(slot->log.count(keys[i])){
                    const auto& value=slot->log.at(keys[i]);if(value.is_string())csv<<value.get<std::string>();else csv<<value.dump();}}csv<<'\n';
            }
            summary["vehicles"].push_back({{"name",v->name},{"planned_slots",v->rows.size()},{"counts",counts},{"inferred_models",model_counts},
                {"retired",v->retired},{"reason",v->retirement_reason}});
        }
        jsonl.flush();csv.flush();if(!jsonl||!csv||raw_journal.failed()||plan_journal.failed())throw std::runtime_error("raw record write failure invalidates trial");
        writej(out+"/summary.json",summary);std::cout<<summary.dump(2)<<std::endl;
        return executor.failed()||planner.failed()?1:0;
    }catch(const std::exception& e){
        if(touched&&manager)try{manager->pause(true);}catch(...){}
        if(!out.empty())try{writej(out+"/failure.json",{{"error",e.what()},{"finished_us",a::now_us()}});}catch(...){}
        std::cerr<<"ERROR "<<e.what()<<std::endl;return 1;
    }
}
