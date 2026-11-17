#pragma once
// Application plumbing, not an alternative scheduler. LiveCalendar remains the
// sole owner of release, model choice, ordering, cancellation and held versions.
#include "local_dpu.hpp"
#include "scheduler/live_calendar.hpp"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace psadaptive {
using Json=nlohmann::json;
inline std::uint64_t now_us() {
    timespec t{};
    if(clock_gettime(CLOCK_MONOTONIC,&t))throw std::runtime_error("CLOCK_MONOTONIC unavailable");
    return std::uint64_t(t.tv_sec)*1000000+std::uint64_t(t.tv_nsec/1000);
}
inline double now_ms(){return double(now_us())/1000.;}
inline std::string key(const std::string& vehicle,std::uint64_t cycle){return vehicle+":"+std::to_string(cycle);}

// Non-blocking producers; a bounded queue failure is fatal, never silent loss.
// Serialization/flush happen only on the dedicated writer thread.
class Journal {
    std::mutex mutex_;std::condition_variable cv_;
    std::deque<std::shared_ptr<const Json>> queue_;
    std::ofstream output_;std::size_t capacity_;bool stopped_=false;
    std::atomic<bool> failed_{false};std::atomic<std::uint64_t> written_{0};
    std::thread writer_;
public:
    explicit Journal(const std::string& path,std::size_t capacity=8192):output_(path),capacity_(capacity){
        if(!output_||!capacity_)throw std::runtime_error("Cannot create bounded journal "+path);
        writer_=std::thread([this]{for(;;){std::shared_ptr<const Json> item;{
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock,[&]{return stopped_||!queue_.empty();});
            if(stopped_&&queue_.empty())return;
            item=queue_.front();queue_.pop_front();
        }
        try{output_<<item->dump()<<'\n';output_.flush();if(!output_)failed_=true;else ++written_;}
        catch(...){failed_=true;}
        }});
    }
    bool submit(std::shared_ptr<const Json> item){
        {std::lock_guard<std::mutex> lock(mutex_);
            if(stopped_||queue_.size()>=capacity_){failed_=true;return false;}
            queue_.push_back(std::move(item));
        }
        cv_.notify_one();return true;
    }
    bool submit(Json item){return submit(std::make_shared<const Json>(std::move(item)));}
    bool failed()const{return failed_.load();}
    std::uint64_t written()const{return written_.load();}
    void close(){
        {std::lock_guard<std::mutex> lock(mutex_);stopped_=true;}
        cv_.notify_all();if(writer_.joinable())writer_.join();
    }
    ~Journal(){close();}
};

struct Reply {psdpu::Result result;Json dispatch;bool dispatched=false;};
struct Slot {
    std::string vehicle;std::uint64_t cycle=0;
    psdpu::Input raw{};Json log;std::promise<Reply> promise;
};

// Only this thread may call the real frozen DNNDK Pool. Infer is injected solely
// to allow CPU-only lifecycle plumbing tests; production binds LocalDpu::infer.
class Executor {
    ps_scheduler::LiveCalendar& calendar_;
    std::function<psdpu::Result(int,const psdpu::Input&)> infer_;
    std::mutex mutex_;std::condition_variable cv_;
    std::map<std::string,std::shared_ptr<Slot>> payloads_;
    bool stop_=false;std::atomic<bool> failed_{false};std::thread worker_;
public:
    Executor(ps_scheduler::LiveCalendar& calendar,
             std::function<psdpu::Result(int,const psdpu::Input&)> infer):calendar_(calendar),infer_(std::move(infer)),worker_([this]{run();}){}
    std::future<Reply> enqueue(const std::shared_ptr<Slot>& slot,double sample_ms,double ready_ms,double hold_ms){
        auto future=slot->promise.get_future();
        // Register payload BEFORE queued() exposes it. take()+lookup uses this
        // same mutex, so neither a fast executor nor a plan commit can race it.
        std::lock_guard<std::mutex> lock(mutex_);
        const auto name=key(slot->vehicle,slot->cycle);
        if(stop_||failed_.load()||payloads_.size()>=12||payloads_.count(name))throw std::runtime_error("DPU payload registry full/closed/failed/duplicate");
        payloads_.emplace(name,slot);
        Json queued;
        try{queued=calendar_.queued(slot->vehicle,slot->cycle,sample_ms,ready_ms,hold_ms);}
        catch(...){payloads_.erase(name);throw;}
        slot->log["calendar_queued"]=queued;
        slot->log["actuation_deadline_us"]=std::uint64_t(queued.at("actuation_deadline_ms").get<double>()*1000.);
        slot->log["dpu_enqueue_us"]=now_us();
        if(queued.at("cancelled").get<bool>()){
            payloads_.erase(name);Reply reply;reply.dispatch=queued;reply.result.status=5;
            slot->promise.set_value(std::move(reply));
        }
        cv_.notify_all();return future;
    }
    void notify(){cv_.notify_all();}
    bool failed()const{return failed_.load();}
    void close(){
        {std::lock_guard<std::mutex> lock(mutex_);stop_=true;}
        cv_.notify_all();if(worker_.joinable())worker_.join();
    }
    ~Executor(){close();}
private:
    void run(){for(;;){std::shared_ptr<Slot> slot;Json dispatch;
        try{
            {std::unique_lock<std::mutex> lock(mutex_);
                if(stop_&&payloads_.empty())return;
                dispatch=calendar_.take(now_ms());
                if(dispatch.is_null()){
                    cv_.wait_for(lock,std::chrono::microseconds(250));continue;
                }
                const auto name=key(dispatch.at("vehicle").get<std::string>(),dispatch.at("cycle").get<std::uint64_t>());
                auto it=payloads_.find(name);
                if(it==payloads_.end())throw std::runtime_error("calendar dispatch without registered payload "+name);
                slot=it->second;payloads_.erase(it);
            }
            Reply reply;reply.dispatch=dispatch;reply.dispatched=dispatch.at("dispatch").get<bool>();
            slot->log["dpu_dequeue_us"]=now_us();slot->log["calendar_dispatch"]=dispatch;
            if(!reply.dispatched){reply.result.status=5;slot->promise.set_value(std::move(reply));continue;}
            const int selected=dispatch.at("configuration_id").get<int>();
            slot->log["model_id"]=selected;
            try{
                reply.result=infer_(selected,slot->raw);
                const auto returned=now_us();slot->log["dpu_local_return_us"]=returned;
                calendar_.dpu_done(slot->vehicle,slot->cycle,double(returned)/1000.,reply.result.status==0);
            }catch(...){
                // Real inference has returned/thrown. Only now release the
                // nonpreemptive calendar resource, never on a future timeout.
                calendar_.dpu_done(slot->vehicle,slot->cycle,now_ms(),false);
                throw;
            }
            slot->promise.set_value(std::move(reply));
        }catch(...){
            failed_=true;
            if(slot)try{slot->promise.set_exception(std::current_exception());}catch(...){}
            // A structural failure is observable. Resolve any waiting payloads
            // rather than leaving per-vehicle workers blocked during shutdown.
            std::lock_guard<std::mutex> lock(mutex_);stop_=true;
            for(auto& item:payloads_)try{item.second->promise.set_exception(std::current_exception());}catch(...){}
            payloads_.clear();return;
        }
    }}
};
} // namespace psadaptive
