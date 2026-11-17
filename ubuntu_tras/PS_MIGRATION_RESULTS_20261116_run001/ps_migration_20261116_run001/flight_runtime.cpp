// Stage 1: PS absolute release + native AirSim IO + original RAW237 + local DPU.
// No Python, loopback inference RPC, model replacement, or input clipping.
#include "local_dpu.hpp"
#include "features/features.hpp"
#include "rpc/airsim_rpc.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <deque>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <thread>
#include <sys/stat.h>
using Json=nlohmann::json;
using Clock=std::chrono::steady_clock;
namespace f=psfeatures;
volatile sig_atomic_t continue_run=1;
void interrupted(int){continue_run=0;}
uint64_t now_us(){return std::chrono::duration_cast<std::chrono::microseconds>(Clock::now().time_since_epoch()).count();}
Json readj(const std::string& p){std::ifstream s(p);if(!s)throw std::runtime_error("missing "+p);Json j;s>>j;return j;}
void writej(const std::string& p,const Json& j){std::ofstream s(p);s<<j.dump(2)<<'\n';if(!s)throw std::runtime_error("write failure "+p);}
template<class T,size_t N> Json arr(const std::array<T,N>& a){Json j=Json::array();for(auto x:a)j.push_back(x);return j;}
f::State state_of(const psairsim::State& x){f::State r;r.position=x.position;r.velocity=x.velocity;r.acceleration=x.acceleration;
  r.angular_velocity=x.angular_velocity;r.quaternion=x.orientation_wxyz;r.timestamp_ns=x.timestamp_ns;return r;}
double norm3(const f::Vec3& a){return std::sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2]);}
f::Vec3 minus3(const f::Vec3& a,const f::Vec3& b){return {{a[0]-b[0],a[1]-b[1],a[2]-b[2]}};}

struct Slot {Json log;psdpu::Input raw{};int model=11;uint64_t due=0;std::promise<psdpu::Result> result;};
// Dedicated writer: publication and command paths enqueue immutable pointers,
// never format/write JSON. This also preserves completed slots on exceptions.
class SlotJournal {
  std::mutex mutex_;std::condition_variable cv_;std::deque<std::shared_ptr<Slot>> queue_;
  std::ofstream file_;bool stop_=false;std::atomic<bool> failed_{false};std::thread writer_;
 public:
  explicit SlotJournal(const std::string& path):file_(path){
    if(!file_)throw std::runtime_error("cannot create raw journal");
    writer_=std::thread([this]{for(;;){std::shared_ptr<Slot> s;{
      std::unique_lock<std::mutex> l(mutex_);cv_.wait(l,[&]{return stop_||!queue_.empty();});
      if(stop_&&queue_.empty())return;
      s=queue_.front();queue_.pop_front();}
      try{file_<<s->log.dump()<<'\n';file_.flush();if(!file_)failed_=true;}
      catch(...){failed_=true;}
    }});
  }
  void submit(const std::shared_ptr<Slot>& s){
    {std::lock_guard<std::mutex> l(mutex_);
      if(queue_.size()>=32768){failed_=true;return;}
      queue_.push_back(s);}
    cv_.notify_one();
  }
  bool failed()const{return failed_.load();}
  void close(){
    {std::lock_guard<std::mutex> l(mutex_);stop_=true;}
    cv_.notify_one();if(writer_.joinable())writer_.join();
  }
  ~SlotJournal(){close();}
};
class DpuExecutor {
  psdpu::LocalDpu& dpu_;std::mutex mutex_;std::condition_variable cv_;std::deque<std::shared_ptr<Slot>> queue_;
  bool finish_=false;std::thread thread_;
 public:
  explicit DpuExecutor(psdpu::LocalDpu& d):dpu_(d),thread_([this]{run();}){}
  ~DpuExecutor(){{std::lock_guard<std::mutex> l(mutex_);finish_=true;}cv_.notify_one();if(thread_.joinable())thread_.join();}
  std::future<psdpu::Result> enqueue(std::shared_ptr<Slot> s){auto future=s->result.get_future();{
    std::lock_guard<std::mutex> l(mutex_);if(queue_.size()>=12)throw std::runtime_error("bounded DPU queue full");
    s->log["dpu_enqueue_us"]=now_us();queue_.push_back(s);}cv_.notify_one();return future;}
 private:
  void run(){for(;;){std::shared_ptr<Slot> s;{
    std::unique_lock<std::mutex> l(mutex_);cv_.wait(l,[&]{return finish_||!queue_.empty();});
    if(queue_.empty()&&finish_)return;
    auto next=std::min_element(queue_.begin(),queue_.end(),[](const auto& a,const auto& b){return a->due<b->due;});
    s=*next;queue_.erase(next);}
    try{s->log["dpu_dequeue_us"]=now_us();
      if(now_us()>=s->due){psdpu::Result r;r.status=5;s->result.set_value(r);continue;}
      auto r=dpu_.infer(s->model,s->raw);s->log["dpu_local_return_us"]=now_us();s->result.set_value(r);
    }catch(...){s->result.set_exception(std::current_exception());}
  }}
};
struct Vehicle {
  std::string name;f::Trajectory trajectory;std::vector<f::Gate> gates;f::Vec3 offset{};
  std::unique_ptr<f::GateHoldAnalyzer> analyzer;std::unique_ptr<psairsim::Client> client;
  std::mutex mutex;std::condition_variable cv;bool busy=false,stop=false,retired=false;
  std::shared_ptr<Slot> pending;std::thread worker;
  f::Control held{{9.81,0,0,0}};uint64_t actuation=0,version=0,collision_stamp=0;
  std::string retirement_reason;std::vector<std::shared_ptr<Slot>> rows;
  ~Vehicle(){
    {std::lock_guard<std::mutex> l(mutex);stop=true;}
    cv.notify_all();
    if(worker.joinable())worker.join();
  }
};
int main(int argc,char** argv){
  std::string prepared,out,host="192.168.77.1",package="/home/root/board_package_requant18_release001";
  int model=11;double period_ms=50.,fresh_ms=80.,maximum_update_ms=160.,duration=20.;
  double reference_start=1.5,post_reserve_ms=2.;bool touched=false;
  std::unique_ptr<psairsim::Client> manager;
  try{
    for(int i=1;i<argc;++i){std::string k=argv[i];if(++i==argc)throw std::runtime_error("missing option value");std::string v=argv[i];
      if(k=="--prepared")prepared=v;else if(k=="--out")out=v;else if(k=="--host")host=v;
      else if(k=="--package")package=v;else if(k=="--model")model=std::stoi(v);else if(k=="--period-ms")period_ms=std::stod(v);
      else if(k=="--duration-s")duration=std::stod(v);else throw std::runtime_error("unknown option "+k);}
    if(prepared.empty()||out.empty()||period_ms!=50.||duration<20.||model<1||model>18)
      throw std::runtime_error("Stage1 requires --prepared JSON --out NEWDIR [--model 11] fixed T50ms and >=20s");
    if(mkdir(out.c_str(),0775)!=0)throw std::runtime_error("output exists or cannot be created");
    std::signal(SIGINT,interrupted);std::signal(SIGTERM,interrupted);
    Json setup=readj(prepared);writej(out+"/prepared_source.json",setup);
    if(setup.value("schema",std::string())!="ps-prepared-case/v1"||setup.value("status",std::string())!="PREPARED_PAUSED")
      throw std::runtime_error("prepared scene schema/status mismatch");
    reference_start=setup.value("reference_start_s",1.5);
    auto calibration=f::Calibration::from_json(setup.at("calibration"));
    manager.reset(new psairsim::Client(host,41451,2000));
    if(!manager->is_paused())throw std::runtime_error("prepare must leave AirSim paused");
    psdpu::LocalDpu dpu;writej(out+"/actual_dpu_identity.json",dpu.identity());
    // Exact own golden before unpausing; this binary links all frozen 18 ELF.
    std::string model_name;auto registry=readj(package+"/dpu_model_registry.json");
    for(const auto& m:registry["models"])if(m["numeric_id"].get<int>()==model)model_name=m["id"].get<std::string>();
    std::ifstream g(package+"/golden/"+model_name+".txt");if(!g)throw std::runtime_error("own model golden missing");
    std::vector<psdpu::Input> goldens;std::string line;double max_error=0.;
    while(std::getline(g,line)){if(line.empty()||line[0]=='#')continue;std::istringstream s(line);psdpu::Input x;std::array<float,4> y;
      for(auto& v:x)if(!(s>>v))throw std::runtime_error("bad golden input");
      for(auto& v:y)if(!(s>>v))throw std::runtime_error("bad golden output");
      auto r=dpu.infer(model,x);if(r.status)throw std::runtime_error("local golden rejected");
      for(size_t c=0;c<4;++c)max_error=std::max(max_error,std::abs(double(r.output[c])-y[c]));
      goldens.push_back(x);}
    if(goldens.size()!=32||max_error>1e-4)throw std::runtime_error("local golden failed");
    for(int i=0;i<100;++i)if(dpu.infer(model,goldens[i%goldens.size()]).status)throw std::runtime_error("warmup failed");
    writej(out+"/golden.json",{{"rows",goldens.size()},{"max_abs",max_error},{"status","PASS"}});
    SlotJournal journal(out+"/events.jsonl");
    uint64_t origin=0;std::mutex ps_mutex;DpuExecutor executor(dpu);
    std::vector<std::unique_ptr<Vehicle>> vehicles;Json initial_states=Json::array();
    for(auto it=setup.at("layouts").begin();it!=setup.at("layouts").end();++it){
      auto v=std::unique_ptr<Vehicle>(new Vehicle);v->name=it.key();v->trajectory=f::Trajectory::from_layout(it.value());
      for(const auto& gate:it.value()["gates"])v->gates.push_back(f::Gate::from_json(gate));
      for(int c=0;c<3;++c)v->offset[c]=setup["offsets"][v->name][c].get<double>();
      v->analyzer.reset(new f::GateHoldAnalyzer(v->gates,3.85,f::HoldBounds(),3.85));
      v->client.reset(new psairsim::Client(host,41451,2000));auto raw=v->client->get_state(v->name);
      initial_states.push_back({{"vehicle",v->name},{"position",arr(raw.position)},{"velocity",arr(raw.velocity)},
        {"quaternion",arr(raw.orientation_wxyz)},{"sim_timestamp_ns",raw.timestamp_ns},{"observed_us",now_us()}});
      auto expected=v->trajectory.evaluate(reference_start);double dot=0;for(int c=0;c<4;++c)dot+=raw.orientation_wxyz[c]*expected.orientation[c];
      if(norm3(minus3(raw.position,expected.position))>.05||norm3(minus3(raw.velocity,expected.velocity))>.2||std::abs(dot)<.999)
        throw std::runtime_error("prepared initial state mismatch "+v->name);
      v->collision_stamp=v->client->get_collision(v->name).timestamp_ns;
      auto hover=f::prepare_live_command(v->held,calibration);const auto c=f::native_rpc_command(hover);
      auto ack=v->client->move_rates_throttle(v->name,c.roll_rate,c.pitch_rate,c.yaw_rate,c.throttle,c.duration);
      if(!ack.ping_ok)throw std::runtime_error("hover prime failed");
      v->held=hover.effective;
      v->rows.reserve(size_t(std::ceil(duration*1000/period_ms))+1);vehicles.push_back(std::move(v));
    }
    writej(out+"/initial_states.json",initial_states);
    if(vehicles.empty()||vehicles.size()>2)throw std::runtime_error("stage1 supports one or two fixed vehicles only");
    for(auto& p:vehicles){Vehicle* v=p.get();v->worker=std::thread([&,v]{for(;;){std::shared_ptr<Slot> slot;{
      std::unique_lock<std::mutex> lock(v->mutex);v->cv.wait(lock,[&]{return v->stop||bool(v->pending);});
      if(!v->pending&&v->stop)break;
      slot=v->pending;v->pending.reset();}
      auto& row=slot->log;
      try{
        row["sample_start_us"]=now_us();if(now_us()>=slot->due){row["status"]="dropped_before_sample";throw std::runtime_error("__drop__");}
        auto raw=v->client->get_state(v->name);row["state_ready_us"]=now_us();row["sim_timestamp_ns"]=raw.timestamp_ns;
        auto state=state_of(raw);row["position"]=arr(state.position);row["velocity"]=arr(state.velocity);row["quaternion"]=arr(state.quaternion);
        auto hit=v->client->get_collision(v->name);row["collision_ready_us"]=now_us();
        if(hit.has_collided&&hit.timestamp_ns!=v->collision_stamp){row["collision_object"]=hit.object_name;throw std::runtime_error("collision");}
        uint64_t sample=row["sample_start_us"].get<uint64_t>();row["ps_lock_requested_us"]=now_us();
        {std::lock_guard<std::mutex> ps_lock(ps_mutex);row["ps_lock_acquired_us"]=now_us();
          auto refs=v->trajectory.window((sample-origin)/1e6+reference_start);row["reference_ready_us"]=now_us();
          row["reference_position"]=arr(refs[0].position);row["reference_velocity"]=arr(refs[0].velocity);
          double error=norm3(minus3(state.position,refs[0].position));row["tracking_error_m"]=error;
          if(state.position[2]+v->offset[2]>-.3||!std::isfinite(error)||error>8.||norm3(state.velocity)>15.)throw std::runtime_error("flight_guard");
          auto analysis=v->analyzer->analyze(state,v->held);row["analysis_ready_us"]=now_us();
          auto deadline=f::safe_deadline(sample/1e6,now_us()/1e6,v->actuation/1e6,analysis,maximum_update_ms/1000.);
          row["hold_deadline_us"]=uint64_t(std::max(0.,deadline.deadline_s)*1e6);row["hold_budget_s"]=analysis.budget_s;
          row["fresh_deadline_us"]=sample+uint64_t(fresh_ms*1000);row["held_control"]=arr(v->held);row["held_version"]=v->version;
          slot->due=std::min({slot->due,row["hold_deadline_us"].get<uint64_t>(),row["fresh_deadline_us"].get<uint64_t>()});
          row["actuation_deadline_us"]=slot->due;slot->raw=f::encode(state,refs);row["raw237"]=arr(slot->raw);row["feature_ready_us"]=now_us();
        }
        if(now_us()>=slot->due){row["status"]="dropped_preprocess";throw std::runtime_error("__drop__");}
        auto future=executor.enqueue(slot);auto result=future.get();
        row["input_prep_start_us"]=result.dispatch_us;row["input_prep_end_us"]=result.prep_end_us;
        row["dpu_submit_us"]=result.run_start_us;row["dpu_result_readable_us"]=result.run_end_us;row["dpu_output_ready_us"]=result.end_us;
        row["output_ready_us"]=now_us();row["dpu_status"]=result.status;row["saturation"]=result.saturation;row["switched"]=result.switched;
        if(result.status==5){row["status"]="dropped_queue";throw std::runtime_error("__drop__");}
        if(result.status){row["rejected_channels"]=result.rejected_channels;
          row["status"]=result.status==3?"input_range_rejected":"dpu_error";throw std::runtime_error("inference_status");}
        row["physical_output"]=arr(result.output);
        if(now_us()+uint64_t(post_reserve_ms*1000)>=slot->due){row["status"]="dropped_before_command";throw std::runtime_error("__drop__");}
        f::Control u;for(int c=0;c<4;++c)u[c]=result.output[c];auto command=f::prepare_live_command(u,calibration);
        row["control_converted_us"]=now_us();const auto c=f::native_rpc_command(command);
        row["native_rpc_control"]=Json::array({c.roll_rate,c.pitch_rate,c.yaw_rate,c.throttle});
        row["command_attempt_start_us"]=now_us();row["command_may_have_been_sent"]=true;
        row["command_sent"]=nullptr; // Unknown if RPC throws after enqueueing; do not label it unsent.
        auto ack=v->client->move_rates_throttle(v->name,c.roll_rate,c.pitch_rate,c.yaw_rate,c.throttle,c.duration);
        row["command_submit_us"]=uint64_t(ack.submit_ns/1000);row["command_enqueued_us"]=uint64_t(ack.enqueued_ns/1000);
        row["command_ping_start_us"]=uint64_t(ack.ping_start_ns/1000);row["command_return_us"]=uint64_t(ack.ping_return_ns/1000);
        row["command_sent"]=true;row["command_return_proxy"]=true;row["effective_control"]=arr(command.effective);
        v->held=command.effective;v->actuation=uint64_t(ack.ping_return_ns/1000);++v->version;
        row["status"]=ack.ping_ok?(v->actuation<slot->due?"delivered":"command_return_late"):"command_return_error";
        // Reference implementation pairs a fresh state with the new held control.
        row["paired_sample_start_us"]=now_us();auto paired=v->client->get_state(v->name);row["paired_state_ready_us"]=now_us();
        auto paired_hit=v->client->get_collision(v->name);row["paired_collision_ready_us"]=now_us();
        if(paired_hit.has_collided&&paired_hit.timestamp_ns!=v->collision_stamp)throw std::runtime_error("collision_after_command");
        {std::lock_guard<std::mutex> ps_lock(ps_mutex);auto paired_analysis=v->analyzer->analyze(state_of(paired),v->held);
          row["paired_analysis_ready_us"]=now_us();row["paired_hold_budget_s"]=paired_analysis.budget_s;}
      }catch(const std::exception& e){std::string error=e.what();if(error!="__drop__"){
        row["error"]=error;if(row.value("status",std::string("pending"))=="pending")row["status"]="error";
        {std::lock_guard<std::mutex> l(v->mutex);v->retired=true;v->retirement_reason=error;}
      }}
      row["slot_return_us"]=now_us();{std::lock_guard<std::mutex> l(v->mutex);v->busy=false;}v->cv.notify_all();
      journal.submit(slot);
    }});}
    touched=true;manager->pause(false);origin=now_us();for(auto& v:vehicles)v->actuation=origin;
    writej(out+"/execution_identity.json",{{"backend","native PS threads -> local DNNDK -> AirSim RPC"},{"stage","fixed_configuration"},
      {"model_id",model},{"period_ms",period_ms},{"duration_s",duration},{"reference_start_s",reference_start},{"origin_us",origin},
      {"clock","PS CLOCK_MONOTONIC; simulation timestamp separate"},{"dpu_scope","DPU call wall-clock, not pure hardware"},
      {"command_ack_scope","RPC ping return proxy, not physical actuation timestamp"},{"paired_refresh","inline, original Windows default"},
      {"ps_affinity","not set"},{"linux_priority","SCHED_OTHER default; not PREEMPT_RT claimed"},{"input_clipping",false}});
    size_t cycles=size_t(std::ceil(duration*1000/period_ms));std::string early_stop;
    for(size_t k=0;k<cycles&&continue_run&&!journal.failed();++k){uint64_t release=origin+uint64_t(k*period_ms*1000);
      std::this_thread::sleep_until(Clock::time_point(std::chrono::microseconds(release)));
      for(auto& v:vehicles){auto slot=std::make_shared<Slot>();slot->model=model;slot->due=release+uint64_t(period_ms*1000);
        slot->log={{"vehicle",v->name},{"cycle",k},{"model_id",model},{"planned_release_us",release},{"actual_publish_us",now_us()},
          {"period_ms",period_ms},{"period_deadline_us",slot->due},{"status","pending"},{"command_sent",false}};
        std::lock_guard<std::mutex> l(v->mutex);v->rows.push_back(slot);
        if(v->retired){slot->log["status"]="not_released_vehicle_retired";slot->log["retirement_reason"]=v->retirement_reason;journal.submit(slot);}
        else if(now_us()>=slot->due){slot->log["status"]="dropped_release_expired";journal.submit(slot);}
        else if(v->busy){slot->log["status"]="dropped_busy_io";journal.submit(slot);}
        else{v->busy=true;v->pending=slot;v->cv.notify_one();}
      }
      bool all_retired=true;for(auto& v:vehicles){std::lock_guard<std::mutex> l(v->mutex);all_retired=all_retired&&v->retired;}
      if(all_retired){early_stop="all_vehicles_retired";break;}
    }
    if(continue_run&&!journal.failed()&&early_stop.empty())
      std::this_thread::sleep_until(Clock::time_point(std::chrono::microseconds(origin+uint64_t(duration*1e6))));
    for(auto& v:vehicles){{std::lock_guard<std::mutex> l(v->mutex);v->stop=true;}v->cv.notify_one();}
    for(auto& v:vehicles)v->worker.join();
    const auto pause_start_us=now_us();manager->pause(true);const auto pause_return_us=now_us();
    if(early_stop.empty()&&!continue_run)early_stop="interrupted";
    if(early_stop.empty()&&journal.failed())early_stop="journal_failed";
    // Preserve every originally planned slot, without inventing future releases.
    for(auto& v:vehicles)for(size_t k=v->rows.size();k<cycles;++k){auto s=std::make_shared<Slot>();
      s->log={{"vehicle",v->name},{"cycle",k},{"model_id",model},{"planned_release_us",origin+uint64_t(k*period_ms*1000)},
        {"actual_publish_us",nullptr},{"status","not_released_trial_stopped"},{"stop_reason",early_stop},
        {"command_sent",false},{"observed_us",pause_return_us},{"period_ms",period_ms}};
      v->rows.push_back(s);journal.submit(s);
    }
    journal.close();
    Json final_states=Json::array();
    for(auto& v:vehicles){try{const auto state_rpc_start_us=now_us();auto raw=v->client->get_state(v->name);
      const auto state_rpc_return_us=now_us();auto hit=v->client->get_collision(v->name);
      final_states.push_back({{"vehicle",v->name},{"position",arr(raw.position)},{"velocity",arr(raw.velocity)},
        {"quaternion",arr(raw.orientation_wxyz)},{"sim_timestamp_ns",raw.timestamp_ns},{"observed_us",now_us()},
        {"state_rpc_start_us",state_rpc_start_us},{"state_rpc_return_us",state_rpc_return_us},
        {"pause_start_us",pause_start_us},{"pause_return_us",pause_return_us},
        {"collision_new",hit.has_collided&&hit.timestamp_ns!=v->collision_stamp},{"collision_object",hit.object_name}});
    }catch(const std::exception&e){final_states.push_back({{"vehicle",v->name},{"error",e.what()},{"observed_us",now_us()}});}}
    writej(out+"/final_states.json",final_states);
    Json summary={{"scope","fixed PS chain; control/flight/timing are separate"},{"interrupted",!bool(continue_run)},
      {"duration_s",duration},{"early_stop",early_stop},{"pause_start_us",pause_start_us},{"pause_return_us",pause_return_us},
      {"journal_failed",journal.failed()},{"vehicles",Json::array()}};
    std::ofstream jsonl(out+"/requests.jsonl");
    std::ofstream csv(out+"/requests.csv");csv<<"vehicle,cycle,model_id,status,planned_release_us,actual_publish_us,sample_start_us,state_ready_us,collision_ready_us,reference_ready_us,analysis_ready_us,feature_ready_us,dpu_enqueue_us,dpu_dequeue_us,dpu_submit_us,dpu_result_readable_us,output_ready_us,command_submit_us,command_return_us,slot_return_us,actuation_deadline_us,sim_timestamp_ns\n";
    std::vector<std::string> keys={"vehicle","cycle","model_id","status","planned_release_us","actual_publish_us","sample_start_us","state_ready_us","collision_ready_us","reference_ready_us","analysis_ready_us","feature_ready_us","dpu_enqueue_us","dpu_dequeue_us","dpu_submit_us","dpu_result_readable_us","output_ready_us","command_submit_us","command_return_us","slot_return_us","actuation_deadline_us","sim_timestamp_ns"};
    for(auto& v:vehicles){Json counts=Json::object();for(auto& s:v->rows){auto status=s->log["status"].get<std::string>();
        // C++14 does not order assignment operand evaluation. Read BEFORE []
        // can insert null in this legacy JSON library.
        const int count=counts.value(status,0);counts[status]=count+1;
        jsonl<<s->log.dump()<<'\n';for(size_t i=0;i<keys.size();++i){if(i)csv<<',';if(s->log.count(keys[i])){auto value=s->log[keys[i]];if(value.is_string())csv<<value.get<std::string>();else csv<<value.dump();}}csv<<'\n';}
      summary["vehicles"].push_back({{"name",v->name},{"planned_slots",v->rows.size()},{"counts",counts},{"retired",v->retired},{"reason",v->retirement_reason}});
    }
    jsonl.flush();csv.flush();
    if(!jsonl||!csv||journal.failed())throw std::runtime_error("raw output journal/write failure; trial invalid");
    writej(out+"/summary.json",summary);std::cout<<summary.dump(2)<<std::endl;
    return 0; // Process completion is not flight/strategy acceptance; inspect raw statuses.
  }catch(const std::exception& e){
    if(touched&&manager)try{manager->pause(true);}catch(...){}
    if(!out.empty())try{writej(out+"/failure.json",{{"error",e.what()},{"finished_us",now_us()}});}catch(...){}
    std::cerr<<"ERROR "<<e.what()<<std::endl;return 1;
  }
}
