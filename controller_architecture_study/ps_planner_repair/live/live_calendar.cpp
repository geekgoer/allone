#include "live_calendar.hpp"
#include "management_timing.hpp"
#include "revalidate.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>
#include <chrono>

namespace ps_scheduler {
namespace {
double n(const Json&j,const char*k){return j.at(k).get<double>();}
std::string s(const Json&j,const char*k){return j.at(k).get<std::string>();}
bool finite(double x){return std::isfinite(x);}
void finite_time(double value,const char* stage){
    if(!finite(value))throw std::invalid_argument(std::string("Finite monotonic time required for ")+stage);
}
Json make_job(const std::string&name,const std::string&vehicle,double release,double ready,
              double prep,double post,double slot,double hold,double fresh,
              const Json&allowed,int model,bool recovery){
    return Json{{"name",name},{"vehicle",vehicle},{"release",release},{"prep_ready",ready},
        {"prep_ms",prep},{"post_ms",post},{"transport_ms",0.},{"slot_deadline",slot},
        {"hold_deadline",hold},{"fresh_deadline",fresh},{"allowed_modes",allowed},
        {"current_mode",model},{"release_options",Json::array()},{"recovery",recovery},
        {"mode_locked",false},{"periodic_edit",false}};
}
Json rejected(const char*reason){return Json{{"installed",false},{"status",reason}};}
bool observation_usable(const Json& loop,double now){
    const auto& ob=loop.at("observation");
    return !loop.at("command_pending").get<bool>()&&!ob.is_null()&&
        n(ob,"sample_ms")<=now&&n(ob,"deadline_ms")>now&&n(loop,"next_ms")>now;
}
}
std::string LiveCalendar::key(const std::string&v,std::uint64_t cycle){return v+":"+std::to_string(cycle);}
LiveCalendar::LiveCalendar(Json c,Json modes):config_(std::move(c)),modes_(Json::object()),
    isolate_command_costs_(config_.value("isolate_command_costs",false)){
    if(modes.is_array())for(const auto&m:modes)modes_[std::to_string(m.at("numeric_id").get<int>())]=m;
    else modes_=std::move(modes);
    if(!config_.count("adaptive"))config_["adaptive"]=true;
    if(!config_.count("enable_skip"))config_["enable_skip"]=false;
    if(config_.value("retire_on_hold_expiry",false)&&
       (!config_.count("maximum_update_ms")||!finite(n(config_,"maximum_update_ms"))||
        n(config_,"maximum_update_ms")<=0.))throw std::invalid_argument("Positive finite maximum update required for hold expiry");
    if(!config_.count("fresh_ms"))config_["fresh_ms"]=80.;
    if(!config_.count("command_reserve_ms"))config_["command_reserve_ms"]=config_.at("post_ms");
    if(!config_.count("rearm_ms"))config_["rearm_ms"]=config_.at("post_ms");
    const auto names=config_.at("names").get<std::vector<std::string>>();
    if(names.empty()||std::set<std::string>(names.begin(),names.end()).size()!=names.size())throw std::invalid_argument("Unique live vehicle names required");
    for(const char*k:{"origin_ms","period_ms","prep_ms","post_ms","decision_ms","fresh_ms","command_reserve_ms","rearm_ms"})
        if(!finite(n(config_,k))||((std::string(k)!="origin_ms")&&n(config_,k)<0))throw std::invalid_argument("Invalid live timing configuration");
    if(n(config_,"period_ms")<=0||n(config_,"fresh_ms")<=0)throw std::invalid_argument("Positive period/freshness required");
    auto periods=config_.at("periods_ms").get<std::vector<double>>();
    if(periods.empty())throw std::invalid_argument("Supported periods required");
    for(double p:periods)if(!finite(p)||p<=0)throw std::invalid_argument("Invalid supported period");
    std::sort(periods.begin(),periods.end());periods.erase(std::unique(periods.begin(),periods.end()),periods.end());config_["periods_ms"]=periods;
    const auto ids=config_.at("candidate_ids").get<std::vector<int>>();
    if(ids.empty()||std::set<int>(ids.begin(),ids.end()).size()!=ids.size())throw std::invalid_argument("Unique candidate IDs required");
    for(int id:ids){auto idkey=std::to_string(id);if(!modes_.count(idkey)||!finite(n(modes_[idkey],"service_ms"))||n(modes_[idkey],"service_ms")<=0||!finite(n(modes_[idkey],"selection_mse"))||n(modes_[idkey],"selection_mse")<0)throw std::invalid_argument("Invalid local model profile");}
    const int default_model=config_.at("model_id").get<int>();if(std::find(ids.begin(),ids.end(),default_model)==ids.end())throw std::invalid_argument("Default outside candidate models");
    for(std::size_t i=0;i<names.size();++i){double period=n(config_,"period_ms");if(config_.count("initial_periods"))period=config_.at("initial_periods").at(names[i]).get<double>();if(std::find(periods.begin(),periods.end(),period)==periods.end())throw std::invalid_argument("Unsupported initial period");double release=n(config_,"origin_ms")+double(i)*period/double(names.size());loops_[names[i]]=Json{{"period_ms",period},{"previous_ms",release-period},{"next_ms",release},{"cycle",0},{"busy",nullptr},{"observation",nullptr},{"control_version",0},{"command_pending",false},{"retired",false}};}
}
Json LiveCalendar::state()const{std::lock_guard<std::mutex>lock(mutex_);return Json{{"revision",revision_},{"loops",loops_},{"future_modes",future_modes_},{"pending_recovery",pending_},{"planned_skip",planned_skip_},{"active",active_.empty()?Json(nullptr):Json(active_)},{"dispatch_order",dispatch_order_},{"dispatch_expires_ms",dispatch_expires_ms_},{"config",config_},{"modes",modes_}};}
double LiveCalendar::next_wakeup_ms()const{std::lock_guard<std::mutex>lock(mutex_);double next=std::numeric_limits<double>::infinity();for(const auto&l:loops_)if(!l.at("retired").get<bool>())next=std::min(next,n(l,"next_ms"));return next;}
double LiveCalendar::command_reserve_unlocked(const Json&job)const{
    const double shared=n(config_,"command_reserve_ms");
    if(!isolate_command_costs_)return shared;
    return std::max(shared,loops_.at(job.at("vehicle").get_ref<const std::string&>()).value("command_reserve_ms",shared));
}
double LiveCalendar::command_reserve(const std::string&v)const{
    std::lock_guard<std::mutex>lock(mutex_);
    const double shared=n(config_,"command_reserve_ms");
    return std::max(shared,loops_.at(v).value("command_reserve_ms",shared));
}
Json LiveCalendar::row(const std::string&v,std::uint64_t cycle,double* lock_wait_us)const{
    using Clock=std::chrono::steady_clock;
    const auto start=lock_wait_us?Clock::now():Clock::time_point{};
    std::lock_guard<std::mutex>lock(mutex_);
    if(lock_wait_us)*lock_wait_us=std::chrono::duration<double,std::micro>(Clock::now()-start).count();
    return rows_.at(key(v,cycle));
}
std::uint64_t LiveCalendar::control_version(const std::string&v)const{std::lock_guard<std::mutex>lock(mutex_);return loops_.at(v).at("control_version").get<std::uint64_t>();}
Json LiveCalendar::release_due(double now,double stop,Json* expiries){
    finite_time(now,"release_due");
    if(std::isnan(stop)||stop==-std::numeric_limits<double>::infinity())throw std::invalid_argument("Invalid release cutoff");
    std::lock_guard<std::mutex>lock(mutex_);Json out=Json::array();
    for(const auto&v:config_.at("names").get<std::vector<std::string>>()){
        auto&l=loops_.at(v);expire_hold_unlocked(v,l,now);if(l.at("retired").get<bool>())continue;
        while(n(l,"next_ms")<=now&&n(l,"next_ms")<stop-1e-6){
            const double r=n(l,"next_ms"),period=n(l,"period_ms");const auto cycle=l.at("cycle").get<std::uint64_t>();const auto name=key(v,cycle);
            int model=config_.at("model_id").get<int>();if(future_modes_.count(name)){model=future_modes_.at(name).get<int>();future_modes_.erase(name);}
            Json item{{"vehicle",v},{"cycle",cycle},{"release_ms",r},{"published_ms",now},{"period_ms",period},{"deadline_ms",r+period},{"model_id",model},{"status","sampling"},{"phase","sampling"},{"cancelled",false},{"control_version",l.at("control_version")}};
            l["previous_ms"]=r;l["next_ms"]=r+period;l["cycle"]=cycle+1;
            if(planned_skip_.count(v)&&planned_skip_.at(v).get<std::uint64_t>()==cycle){item["status"]="planned_skip";planned_skip_.erase(v);}
            else{
                if(!l.at("busy").is_null()){auto&old=rows_.at(l.at("busy").get<std::string>());if(n(old,"deadline_ms")<=now){old["expired"]=true;old["cancelled"]=true;}}
                if(r+period<=now)item["status"]="dropped_release_late";
                else if(!l.at("busy").is_null())item["status"]="dropped_busy_io";
                else if(l.at("command_pending").get<bool>())item["status"]="dropped_uncertain_command";
                else l["busy"]=name;
                if(pending_.count(v)&&pending_[v].at("cycle").get<std::uint64_t>()==cycle)item["recovery_due_ms"]=pending_[v].at("deadline_ms");
            }
            rows_[name]=item;out.push_back(item);++revision_;
        }
    }
    if(expiries){*expiries=Json::array();expiries->swap(hold_expiries_);}
    return out;
}
bool LiveCalendar::observe(const std::string&v,double sample,double hold,std::uint64_t version,bool stable,const std::string&evidence,double available){
    if(!finite(sample)||!finite(hold)||!finite(available))throw std::invalid_argument("Finite observed times required");
    std::lock_guard<std::mutex>lock(mutex_);auto&l=loops_.at(v);
    const double now=std::max(sample,available);
    expire_hold_unlocked(v,l,now);
    if(l.at("retired").get<bool>()||l.at("command_pending").get<bool>()||version!=l.at("control_version").get<std::uint64_t>()||(!l.at("observation").is_null()&&sample<=n(l.at("observation"),"sample_ms")))return false;
    if(config_.value("retire_on_hold_expiry",false)){
        if(l.count("hold_cap_ms"))hold=std::min(hold,n(l,"hold_cap_ms"));
        l["watch_hold_ms"]=hold;
        expire_hold_unlocked(v,l,now);
        if(l.at("retired").get<bool>())return false;
    }
    if(!l.at("observation").is_null()&&hold<n(l.at("observation"),"deadline_ms"))dispatch_order_=Json::array();
    l["observation"]=Json{{"sample_ms",sample},{"deadline_ms",hold},{"control_version",version},{"stable",stable},{"evidence",evidence}};++revision_;proof_ready_.notify_all();return true;
}
Json LiveCalendar::queued(const std::string&v,std::uint64_t cycle,double sample,double ready,double hold){
    finite_time(sample,"queued sample");finite_time(ready,"queued ready");finite_time(hold,"queued hold");
    if(ready<sample)throw std::invalid_argument("Input cannot be ready before sampling begins");
    std::lock_guard<std::mutex>lock(mutex_);auto&l=loops_.at(v);auto name=key(v,cycle);auto&r=rows_.at(name);
    if(l.at("busy")!=name)throw std::invalid_argument("Input ready for a non-owned row");
    double deadline=std::min(n(r,"deadline_ms"),std::min(hold,sample+n(config_,"fresh_ms")));if(r.count("recovery_due_ms"))deadline=std::min(deadline,n(r,"recovery_due_ms"));
    r["sample_start_ms"]=sample;r["queue_ms"]=ready;r["actuation_deadline_ms"]=deadline;r["phase"]="queued";
    if(deadline<=ready||r.at("cancelled").get<bool>()){
        // The enqueue caller resolves this rejection and removes the payload.
        // take() must not return the same job while its worker is finishing.
        r["cancelled"]=true;
        r["dispatch_checked"]=true;
        r["dispatch"]=false;
        r["status"]="dropped_preprocess";
    }
    ++revision_;return r;
}
Json LiveCalendar::take(double now,double* lock_wait_us,double (*refresh_ms)()){
    finite_time(now,"take");
    using Clock=std::chrono::steady_clock;
    const auto wait_start=lock_wait_us?Clock::now():Clock::time_point{};
    std::lock_guard<std::mutex>lock(mutex_);
    if(lock_wait_us)*lock_wait_us=std::chrono::duration<double,std::micro>(Clock::now()-wait_start).count();
    if(!active_.empty())return nullptr;
    const bool separated=config_.value("separate_dispatch_service",false);
    if(separated){
        if(!refresh_ms)throw std::invalid_argument("Separated service requires a monotonic dispatch clock");
        now=std::max(now,refresh_ms());
    }
    std::vector<std::string> live;
    // Historical rows remain for audit, but never scan them on every 250us
    // executor wake. queued() requires busy ownership, so these are exactly
    // the possible live rows. Sorting preserves the former map traversal order.
    std::vector<std::string> owned;
    for(const auto&loop:loops_)if(!loop.at("busy").is_null())owned.push_back(loop.at("busy").get<std::string>());
    std::sort(owned.begin(),owned.end());
    for(const auto&name:owned){auto&r=rows_.at(name);if(s(r,"phase")!="queued"||r.value("dispatch_checked",false))continue;double due=r.value("actuation_deadline_ms",n(r,"deadline_ms"));if(r.at("cancelled").get<bool>()||due<=now){r["cancelled"]=true;r["dispatch_checked"]=true;r["dispatch"]=false;r["status"]="dropped_inference";++revision_;return r;}live.push_back(name);}
    if(live.empty())return nullptr;
    if(now>dispatch_expires_ms_)dispatch_order_=Json::array();
    auto earliest=[&](){return *std::min_element(live.begin(),live.end(),[&](const std::string&a,const std::string&b){const auto&x=rows_.at(a);const auto&y=rows_.at(b);return x.value("actuation_deadline_ms",n(x,"deadline_ms"))<y.value("actuation_deadline_ms",n(y,"deadline_ms"));});};
    std::string selected;
    bool planned_head=false;
    if(config_.at("adaptive").get<bool>()&&!dispatch_order_.empty()){
        const auto head=dispatch_order_.at(0).get<std::string>();
        if(std::find(live.begin(),live.end(),head)!=live.end()){selected=head;planned_head=true;}
        else{
            auto split=head.rfind(':');std::string v=head.substr(0,split);auto cycle=std::stoull(head.substr(split+1));const auto&l=loops_.at(v);bool exists=!l.at("retired").get<bool>()&&(l.at("cycle").get<std::uint64_t>()<=cycle||(!l.at("busy").is_null()&&rows_.at(l.at("busy").get<std::string>()).at("cycle").get<std::uint64_t>()==cycle));
            bool all_room=true;for(const auto&name:live){const auto&r=rows_.at(name);double service=n(modes_.at(std::to_string(r.at("model_id").get<int>())),"service_ms");if(r.value("actuation_deadline_ms",n(r,"deadline_ms"))<=now+service)all_room=false;}
            if(exists&&all_room)return nullptr;
            dispatch_order_=Json::array();selected=earliest();
        }
    }else selected=earliest();
    auto&r=rows_.at(selected);Json tail=Json::array();for(const auto&name:dispatch_order_)if(name!=selected)tail.push_back(name);dispatch_order_=tail;
    if(future_modes_.count(selected)){r["model_id"]=future_modes_.at(selected);future_modes_.erase(selected);}
    const double reserve=command_reserve_unlocked(r);
    double budget=r.value("actuation_deadline_ms",n(r,"deadline_ms"))-now-reserve;std::vector<int>affordable;
    for(int id:config_.at("candidate_ids").get<std::vector<int>>())if(n(modes_.at(std::to_string(id)),"service_ms")<=budget)affordable.push_back(id);
    int selected_model=r.at("model_id").get<int>();
    const auto mode_cost=[&](int id){return n(modes_.at(std::to_string(id)),"service_ms");};
    auto reject_budget=[&](int required_model,const char* reason){
        r["dispatch_rejection_evidence"]={{"checked_ms",now},{"budget_ms",budget},
            {"command_reserve_ms",reserve},
            {"required_model_id",required_model},{"required_service_ms",mode_cost(required_model)}};
        r["cancelled"]=true;r["dispatch_checked"]=true;r["dispatch"]=false;r["status"]="dropped_inference";
        r["dispatch_rejection"]=reason;++revision_;return r;
    };
    if(affordable.empty()||(!config_.at("adaptive").get<bool>()&&std::find(affordable.begin(),affordable.end(),selected_model)==affordable.end())){
        int required_model=selected_model;
        if(config_.at("adaptive").get<bool>()){
            for(int id:config_.at("candidate_ids"))if(mode_cost(id)<mode_cost(required_model))required_model=id;
        }
        return reject_budget(required_model,"no_model_fits_remaining_delivery_budget");
    }
    if(config_.at("adaptive").get<bool>()){
        auto quality_less=[&](int a,int b){return n(modes_.at(std::to_string(a)),"selection_mse")<n(modes_.at(std::to_string(b)),"selection_mse");};
        if(!planned_head&&live.size()>1){
            // Screen the current ready queue, assuming later jobs may use their
            // fastest candidate. This does not certify future jobs or recovery.
            std::sort(live.begin(),live.end(),[&](const std::string&a,const std::string&b){
                const auto&x=rows_.at(a);const auto&y=rows_.at(b);
                return std::make_pair(x.value("actuation_deadline_ms",n(x,"deadline_ms")),a)<
                       std::make_pair(y.value("actuation_deadline_ms",n(y,"deadline_ms")),b);
            });
            const auto ids=config_.at("candidate_ids").get<std::vector<int>>();
            const int fastest=*std::min_element(ids.begin(),ids.end(),[&](int a,int b){
                return n(modes_.at(std::to_string(a)),"service_ms")<
                       n(modes_.at(std::to_string(b)),"service_ms");
            });
            auto fits=[&](int head_mode){
                double finish=now;
                for(const auto&name:live){
                    const auto&job=rows_.at(name);
                    const int mode=name==selected?head_mode:fastest;
                    finish+=n(modes_.at(std::to_string(mode)),"service_ms");
                    if(finish+command_reserve_unlocked(job)>=
                       job.value("actuation_deadline_ms",n(job,"deadline_ms"))-epsilon)return false;
                }
                return true;
            };
            std::vector<int> protecting;for(int id:affordable)if(fits(id))protecting.push_back(id);
            r["ready_queue_fast_tail_fits"]=!protecting.empty();
            r["ready_queue_count"]=live.size();
            if(!protecting.empty()){
                if(std::find(protecting.begin(),protecting.end(),selected_model)==protecting.end())
                    selected_model=*std::min_element(protecting.begin(),protecting.end(),quality_less);
            }else selected_model=*std::min_element(affordable.begin(),affordable.end(),[&](int a,int b){
                return std::make_pair(n(modes_.at(std::to_string(a)),"service_ms"),n(modes_.at(std::to_string(a)),"selection_mse"))<
                       std::make_pair(n(modes_.at(std::to_string(b)),"service_ms"),n(modes_.at(std::to_string(b)),"selection_mse"));
            });
        }else if(n(modes_.at(std::to_string(selected_model)),"service_ms")>budget)
            selected_model=*std::min_element(affordable.begin(),affordable.end(),quality_less);
        r["model_id"]=selected_model;
    }
    if(separated){
        const double refreshed=std::max(now,refresh_ms());
        budget-=refreshed-now;
        now=refreshed;
        if(mode_cost(selected_model)>budget){
            if(!config_.at("adaptive").get<bool>())
                return reject_budget(selected_model,"selected_model_no_longer_fits_at_dispatch");
            const int fastest=*std::min_element(affordable.begin(),affordable.end(),
                [&](int a,int b){return mode_cost(a)<mode_cost(b);});
            const double handoff=std::max(now,refresh_ms());
            budget-=handoff-now;
            now=handoff;
            if(mode_cost(fastest)>budget)return reject_budget(fastest,"no_model_fits_at_dispatch_handoff");
            r["dispatch_handoff_downgraded_from"]=selected_model;
            selected_model=fastest;
            r["model_id"]=selected_model;
            if(r.count("ready_queue_fast_tail_fits")){
                double end=now;
                bool tail_fits=true;
                for(const auto& name:live){
                    const auto& job=rows_.at(name);
                    end+=mode_cost(fastest);
                    if(end+command_reserve_unlocked(job)>=job.value("actuation_deadline_ms",n(job,"deadline_ms"))-epsilon){tail_fits=false;break;}
                }
                r["ready_queue_fast_tail_fits"]=tail_fits;
            }
        }
    }
    r["board_start_ms"]=now;r["phase"]="board";r["configuration_id"]=selected_model;r["dispatch_checked"]=true;r["dispatch"]=true;active_=selected;++revision_;return r;
}
void LiveCalendar::dpu_done(const std::string&v,std::uint64_t cycle,double now,bool success){
    finite_time(now,"dpu_done");
    std::lock_guard<std::mutex>lock(mutex_);auto name=key(v,cycle);if(active_!=name)throw std::invalid_argument("DPU completion identity does not match active nonpreemptive job");auto&r=rows_.at(name);if(now<n(r,"board_start_ms"))throw std::invalid_argument("DPU completion precedes dispatch");r["board_done_ms"]=now;r["phase"]="post";r["board_success"]=success;auto&m=modes_.at(std::to_string(r.at("model_id").get<int>()));m["service_ms"]=std::max(n(m,"service_ms"),now-n(r,"board_start_ms"));active_.clear();++revision_;
}
bool LiveCalendar::command_begin(const std::string&v,std::uint64_t cycle,double now){
    std::lock_guard<std::mutex>lock(mutex_);
    return command_begin_unlocked(v,cycle,now).accepted;
}
CommandAdmission LiveCalendar::command_begin_current(const std::string&v,std::uint64_t cycle,double (*clock_ms)()){
    using Clock=std::chrono::steady_clock;
    const auto before=Clock::now();
    std::lock_guard<std::mutex>lock(mutex_);
    const auto acquired=Clock::now();
    auto result=command_begin_unlocked(v,cycle,clock_ms?clock_ms():monotonic_ms());
    result.lock_wait_us=std::chrono::duration<double,std::micro>(acquired-before).count();
    return result;
}
CommandAdmission LiveCalendar::command_begin_unlocked(const std::string&v,std::uint64_t cycle,double now){
    finite_time(now,"command_begin");
    const auto name=key(v,cycle);auto&r=rows_.at(name);auto&l=loops_.at(v);
    if(l.at("busy")!=name)throw std::invalid_argument("Command attempt no longer owns worker");
    if(s(r,"phase")!="post"||!r.value("board_success",false)||!r.count("board_done_ms")||now<n(r,"board_done_ms"))throw std::invalid_argument("Command attempt requires a completed successful DPU job");
    if(l.at("command_pending").get<bool>()||r.count("command_begin_ms")||r.count("command_return_ms"))throw std::invalid_argument("Command attempt already recorded");
    expire_hold_unlocked(v,l,now);
    const double due=std::min(r.value("actuation_deadline_ms",n(r,"deadline_ms")),l.value("watch_hold_ms",std::numeric_limits<double>::infinity()));
    CommandAdmission result;
    result.checked_ms=now;result.deadline_ms=due;result.reserve_ms=command_reserve_unlocked(r);
    if(l.at("retired").get<bool>())result.rejection="retired";
    else if(r.at("cancelled").get<bool>())result.rejection="cancelled";
    else if(r.at("control_version")!=l.at("control_version"))result.rejection="control_version_changed";
    else if(now+result.reserve_ms>=due)result.rejection="insufficient_reserve";
    if(result.rejection){
        r["cancelled"]=true;r["status"]="dropped_before_command";r["command_begin_rejected_ms"]=now;++revision_;return result;
    }
    r["command_begin_ms"]=now;r["command_pending"]=true;l["command_pending"]=true;
    l["observation"]=nullptr;dispatch_order_=Json::array();++revision_;result.accepted=true;return result;
}
std::uint64_t LiveCalendar::command_return(const std::string&v,std::uint64_t cycle,double now){
    finite_time(now,"command_return");
    std::lock_guard<std::mutex>lock(mutex_);auto&r=rows_.at(key(v,cycle));auto&l=loops_.at(v);if(l.at("busy")!=key(v,cycle))throw std::invalid_argument("Command return no longer owns worker");if(s(r,"phase")!="post"||!r.value("board_success",false)||!r.count("board_done_ms")||now<n(r,"board_done_ms"))throw std::invalid_argument("Command return requires a completed successful DPU job");if(r.count("command_return_ms"))throw std::invalid_argument("Command return already recorded");if(!l.at("command_pending").get<bool>()||!r.value("command_pending",false)||!r.count("command_begin_ms")||now<n(r,"command_begin_ms"))throw std::invalid_argument("Command return requires an actual pending command attempt");
    expire_hold_unlocked(v,l,now);
    r["command_return_ms"]=now;r["command_pending"]=false;l["command_pending"]=false;auto version=l.at("control_version").get<std::uint64_t>()+1;l["control_version"]=version;l["observation"]=nullptr;dispatch_order_=Json::array();++revision_;
    if(config_.value("retire_on_hold_expiry",false)&&!l.at("retired").get<bool>()){
        const double cap=now+n(config_,"maximum_update_ms");
        l["hold_cap_ms"]=cap;l["watch_hold_ms"]=cap;
    }
    return version;
}
Json LiveCalendar::finish(const std::string&v,std::uint64_t cycle,double when,const std::string&status){
    finite_time(when,"finish");
    std::lock_guard<std::mutex>lock(mutex_);auto name=key(v,cycle);if(active_==name)throw std::invalid_argument("Cannot release a DPU job before nonpreemptive execution returns");auto&r=rows_.at(name);auto&l=loops_.at(v);r["finished_ms"]=when;r["status"]=status;r["phase"]="finished";
    if(pending_.count(v)&&pending_[v].at("cycle").get<std::uint64_t>()==cycle&&r.count("command_return_ms")){const double deadline=std::min(n(pending_[v],"deadline_ms"),r.value("actuation_deadline_ms",n(r,"deadline_ms")));bool ontime=n(r,"command_return_ms")<=deadline;r["recovery_on_time"]=ontime;if(ontime)pending_.erase(v);}
    if(l.at("busy")==name)l["busy"]=nullptr;
    future_modes_.erase(name); // A completed carry job cannot be dispatched again.
    ++revision_;
    if(r.count("command_return_ms")&&!l.at("observation").is_null()&&
       l.at("observation").at("control_version")==l.at("control_version")){
        ++paired_generation_;
    }
    proof_ready_.notify_all();
    return r;
}
void LiveCalendar::retire_unlocked(const std::string&v,Json&l){
    if(l.at("retired").get<bool>())return;
    l["retired"]=true;
    if(!l.at("busy").is_null())rows_.at(l.at("busy").get<std::string>())["cancelled"]=true;
    pending_.erase(v);planned_skip_.erase(v);dispatch_order_=Json::array();++revision_;
}
void LiveCalendar::retire(const std::string&v){std::lock_guard<std::mutex>lock(mutex_);retire_unlocked(v,loops_.at(v));}
void LiveCalendar::expire_hold_unlocked(const std::string&v,Json&l,double now){
    if(l.at("retired").get<bool>()||!l.count("watch_hold_ms")||now<n(l,"watch_hold_ms"))return;
    hold_expiries_.push_back({{"vehicle",v},{"control_version",l.at("control_version")},
        {"hold_deadline_ms",l.at("watch_hold_ms")},{"detected_ms",now},
        {"command_in_flight",l.at("command_pending")},{"reason","hold_expired"}});
    retire_unlocked(v,l);
}
Json LiveCalendar::expire_holds(double now){
    finite_time(now,"expire_holds");
    std::lock_guard<std::mutex>lock(mutex_);
    for(auto it=loops_.begin();it!=loops_.end();++it)expire_hold_unlocked(it.key(),it.value(),now);
    Json events=Json::array();events.swap(hold_expiries_);return events;
}
void LiveCalendar::update_costs(double prep,double post,double rearm,double reserve){const std::vector<std::pair<std::string,double>> observations{{"prep_ms",prep},{"post_ms",post},{"rearm_ms",rearm},{"command_reserve_ms",reserve}};for(const auto&p:observations)if(!finite(p.second)||p.second<0)throw std::invalid_argument("Invalid cost observation");std::lock_guard<std::mutex>lock(mutex_);for(const auto&p:observations)if(p.second>config_.at(p.first).get<double>()){config_[p.first]=p.second;++revision_;}}
void LiveCalendar::update_command_costs(const std::string&v,double post,double reserve){
    if(!finite(post)||post<0||!finite(reserve)||reserve<0)throw std::invalid_argument("Invalid command cost observation");
    std::lock_guard<std::mutex>lock(mutex_);
    if(!isolate_command_costs_)throw std::invalid_argument("Vehicle command costs require isolation mode");
    auto& loop=loops_.at(v);
    if(loop.at("retired").get<bool>())return;
    const double old_post=loop.value("post_ms",n(config_,"post_ms"));
    const double old_reserve=loop.value("command_reserve_ms",n(config_,"command_reserve_ms"));
    if(post<=old_post&&reserve<=old_reserve)return;
    loop["post_ms"]=std::max(old_post,post);loop["command_reserve_ms"]=std::max(old_reserve,reserve);
    dispatch_order_=Json::array();++revision_;
}

Json LiveCalendar::snapshot(double now,double* lock_held_us)const{
    finite_time(now,"snapshot");std::unique_lock<std::mutex>lock(mutex_);
    if(!lock_held_us)return capture_snapshot(lock,now);
    const auto acquired=management_stamp();ManagementStamp released;
    auto snap=capture_snapshot(lock,now,&released);
    *lock_held_us=(released.wall_ms-acquired.wall_ms)*1000.;return snap;
}
Json LiveCalendar::snapshot_current()const{std::unique_lock<std::mutex>lock(mutex_);return capture_snapshot(lock,monotonic_ms());}
Json LiveCalendar::snapshot_profiled()const{
    const auto begin=management_stamp();std::unique_lock<std::mutex>lock(mutex_);
    const auto acquired=management_stamp();const auto generation=paired_generation_;
    ManagementStamp released;
    Json snap=capture_snapshot(lock,monotonic_ms(),&released);
    const auto built=management_stamp();
    Json result{{"timing",Json{{"lock_wait",management_interval(begin,acquired)},
                               {"lock_held",management_interval(acquired,released)},
                               {"build",management_interval(acquired,built)}}},
                {"paired_generation",generation}};
    result["snapshot"]=std::move(snap);return result;
}
Json LiveCalendar::snapshot_wait_profiled(double wait_ms)const{
    if(!std::isfinite(wait_ms)||wait_ms<0||wait_ms>2.)throw std::invalid_argument("Proof wait must be in [0,2] ms");
    const auto begin=management_stamp();std::unique_lock<std::mutex>lock(mutex_);
    const auto acquired=management_stamp();
    const auto ready=[&](){
        for(const auto& l:loops_){
            if(l.at("retired").get<bool>())continue;
            if(l.at("command_pending").get<bool>()||l.at("observation").is_null())return false;
            if(!l.at("busy").is_null()){
                const auto phase=s(rows_.at(l.at("busy").get<std::string>()),"phase");
                if(phase=="board"||phase=="post")return false;
            }
        }
        return true;
    };
    const bool was_ready=ready();
    if(!was_ready&&wait_ms>0)proof_ready_.wait_for(lock,std::chrono::duration<double,std::milli>(wait_ms),ready);
    const auto waited=management_stamp();const bool proof_available=ready();
    ManagementStamp released;
    Json snap=capture_snapshot(lock,monotonic_ms(),&released);const auto built=management_stamp();
    Json result{{"proof_available",proof_available},{"wait_requested_ms",wait_ms},{"was_ready",was_ready},
        {"timing",Json{{"lock_wait",management_interval(begin,acquired)},
            {"proof_wait",management_interval(acquired,waited)},{"lock_held",management_interval(waited,released)},
            {"build",management_interval(waited,built)}}}};
    result["snapshot"]=std::move(snap);return result;
}
std::uint64_t LiveCalendar::paired_generation()const{
    std::lock_guard<std::mutex>lock(mutex_);return paired_generation_;
}
bool LiveCalendar::wait_for_paired(std::uint64_t generation,double wait_ms)const{
    if(!std::isfinite(wait_ms)||wait_ms<0||wait_ms>10.)throw std::invalid_argument("Planner wait must be in [0,10] ms");
    std::unique_lock<std::mutex>lock(mutex_);
    if(paired_generation_!=generation)return true;
    if(wait_ms>0)proof_ready_.wait_for(lock,std::chrono::duration<double,std::milli>(wait_ms),
                                        [&]{return paired_generation_!=generation;});
    return paired_generation_!=generation;
}
bool LiveCalendar::can_capture_snapshot_unlocked(double now)const{
    if(now<n(config_,"origin_ms"))return false;
    if(!active_.empty()){const auto&r=rows_.at(active_);if(now-n(r,"board_start_ms")>n(modes_.at(std::to_string(r.at("model_id").get<int>())),"service_ms"))return false;}
    // An installed skip is already a committed pending calendar operation;
    // wait for that absolute release to be consumed, not plan to skip it twice.
    if(!planned_skip_.empty())return false;
    if(config_.value("snapshot_precheck",false)){
        // Reject invalid all-loop context before allocating any job JSON.
        for(const auto& loop:loops_)
            if(!loop.at("retired").get<bool>()&&!observation_usable(loop,now))return false;
    }
    return true;
}
Json LiveCalendar::capture_snapshot(std::unique_lock<std::mutex>& lock,double now,ManagementStamp* released)const{
    const auto unlock=[&](){if(released)*released=management_stamp();lock.unlock();};
    if(!can_capture_snapshot_unlocked(now)){unlock();return nullptr;}
    if(!config_.value("snapshot_build_off_lock",false)){
        auto snapshot=build_snapshot(now,config_,modes_,loops_,rows_,future_modes_,pending_,revision_);
        unlock();return snapshot;
    }
    // Capture exactly the data the builder reads. Historical completed rows
    // are kept by the calendar but are never copied into planning input.
    const Json config=config_,modes=modes_,loops=loops_,pending=pending_;
    const auto revision=revision_;
    Json rows=Json::object(),future=Json::object();
    for(auto it=loops.begin();it!=loops.end();++it){
        const auto& loop=it.value();if(loop.at("retired").get<bool>())continue;
        if(!loop.at("busy").is_null()){
            const auto& name=loop.at("busy").get_ref<const std::string&>();rows[name]=rows_.at(name);
        }
        const auto next=key(it.key(),loop.at("cycle").get<std::uint64_t>());
        if(future_modes_.count(next))future[next]=future_modes_.at(next);
    }
    unlock();
    // The stamped time and revision still identify the captured state. The
    // normal install guard checks current revision, versions and elapsed time.
    return build_snapshot(now,config,modes,loops,rows,future,pending,revision);
}
Json LiveCalendar::build_snapshot(double now,const Json& config,const Json& modes,const Json& loops,
                                  const Json& rows,const Json& future_modes,const Json& pending,
                                  std::uint64_t revision){
    const bool precheck=config.value("snapshot_precheck",false),isolate_command_costs=config.value("isolate_command_costs",false);
    Json jobs=Json::array(),permissions=Json::array(),versions=Json::object(),pending_keys=Json::array(),delivering=Json::array();double blocked=0.;
    for(const auto&v:config.at("names").get<std::vector<std::string>>()){
        const auto&l=loops.at(v);if(l.at("retired").get<bool>())continue;if(!precheck&&!observation_usable(l,now))return nullptr;const auto&ob=l.at("observation");const double hold=n(ob,"deadline_ms"),next=n(l,"next_ms");
        double post=n(config,"post_ms");
        if(isolate_command_costs)post=std::max({post,l.value("post_ms",0.),l.value("command_reserve_ms",0.),n(config,"command_reserve_ms")});
        versions[v]=l.at("control_version");
        if(!l.at("busy").is_null()){
            const auto&r=rows.at(l.at("busy").get<std::string>());const auto stage=s(r,"phase");
            if(n(r,"release_ms")>now)return nullptr; // Reject stale explicit snapshot time; never invent negative preparation.
            if(stage=="board"||stage=="post"){
                delivering.push_back(key(v,r.at("cycle").get<std::uint64_t>()));
                double remaining=stage=="board"?std::max(0.,n(modes.at(std::to_string(r.at("model_id").get<int>())),"service_ms")-(now-n(r,"board_start_ms"))):0.;
                if(!r.count("command_return_ms")&&now+remaining+post>std::min(next,std::min(hold,n(r,"deadline_ms"))))return nullptr;
                if(r.count("command_return_ms")&&now+n(config,"rearm_ms")>next)return nullptr;
                blocked=std::max(blocked,remaining);
            }else if(!r.count("command_return_ms")&&stage!="finished"){
                double prep=stage=="queued"?0.:n(config,"prep_ms");double fresh=r.value("sample_start_ms",n(r,"release_ms"))+n(config,"fresh_ms");bool recovery=pending.count(v)&&pending.at(v).at("cycle")==r.at("cycle");
                jobs.push_back(make_job(key(v,r.at("cycle").get<std::uint64_t>()),v,n(r,"release_ms")-now,0.,prep,post,n(r,"deadline_ms")-now,std::min(hold,r.value("recovery_due_ms",hold))-now,fresh-now,config.at("candidate_ids"),r.at("model_id").get<int>(),recovery));
            }
        }
        auto cycle=l.at("cycle").get<std::uint64_t>();auto name=key(v,cycle);bool recovery=pending.count(v)&&pending.at(v).at("cycle").get<std::uint64_t>()==cycle;double period=n(l,"period_ms");int model=future_modes.count(name)?future_modes.at(name).get<int>():config.at("model_id").get<int>();
        Json first=make_job(name,v,next-now,next-now,n(config,"prep_ms"),post,next+period-now,std::min(hold,recovery?n(pending.at(v),"deadline_ms"):hold)-now,next-now+n(config,"fresh_ms"),config.at("candidate_ids"),model,recovery);
        if(!recovery){first["periodic_edit"]=true;for(double t:config.at("periods_ms").get<std::vector<double>>())if(t!=period&&n(l,"previous_ms")+t>now+n(config,"decision_ms"))first["release_options"].push_back(Json::array({n(l,"previous_ms")+t-now,n(l,"previous_ms")+2*t-now}));}
        jobs.push_back(first);Json successor=first;successor["name"]=key(v,cycle+1);successor["release"]=n(first,"release")+period;successor["prep_ready"]=n(first,"release")+period;successor["slot_deadline"]=n(first,"slot_deadline")+period;successor["hold_deadline"]=n(first,"slot_deadline")+period;successor["fresh_deadline"]=n(first,"fresh_deadline")+period;successor["release_options"]=Json::array();successor["recovery"]=false;successor["periodic_edit"]=false;jobs.push_back(successor);
        // A finite flight must not approve a skip whose mandatory EXISTING
        // successor lies outside its release window. Do not clear debt at exit.
        const bool successor_in_window=!config.count("measurement_stop_ms")||next+period<n(config,"measurement_stop_ms")-epsilon;
        bool eligible=config.at("enable_skip").get<bool>()&&successor_in_window&&!pending.count(v)&&l.at("busy").is_null()&&ob.value("stable",false)&&now-n(ob,"sample_ms")<=n(config,"fresh_ms")&&ob.at("control_version")==l.at("control_version");permissions.push_back(Json{{"job",name},{"successor",key(v,cycle+1)},{"hold_until",hold-now},{"valid_until",n(first,"release")},{"eligible",eligible},{"evidence",ob.value("evidence",std::string("unknown phase"))}});
    }
    if(jobs.empty())return nullptr;
    for(auto p=pending.begin();p!=pending.end();++p){auto name=key(p.key(),p.value().at("cycle").get<std::uint64_t>());for(const auto&j:jobs)if(j.at("name")==name)pending_keys.push_back(Json::array({p.key(),name}));}
    return Json{{"snapshot_ms",now},{"calendar",Json{{"jobs",jobs},{"revision",revision},{"pending",pending_keys}}},{"modes",modes},{"permissions",permissions},{"blocked_until_ms",blocked},{"control_versions",versions},{"config",config},{"delivering",delivering}};
}
Json LiveCalendar::plan_snapshot(const Json&snap)const{
    if(snap.is_null())return Json{{"accepted",false},{"action","obligation_context_not_ready"}};
    const bool revalidate=snap.at("config").value("revalidate",false);
    const bool skip=snap.at("config").at("enable_skip").get<bool>();
    return ps_scheduler::plan(snap.at("calendar"),snap.at("modes"),Json{{"now",0.},{"decision_ms",n(snap.at("config"),"decision_ms")},{"blocked_until",snap.at("blocked_until_ms")},{"permissions",snap.at("permissions")},{"max_checks",revalidate?(skip?6:7):8},{"max_candidates",12},{"enable_skip",skip}});
}
Json LiveCalendar::install(const Json&snap,const Json&decision){std::lock_guard<std::mutex>lock(mutex_);double now=monotonic_ms();return install_unlocked(snap,decision,now);}
Json LiveCalendar::install_profiled(const Json&snap,const Json&decision){
    const auto begin=management_stamp();std::unique_lock<std::mutex>lock(mutex_);
    const auto acquired=management_stamp();Json result=install_unlocked(snap,decision,monotonic_ms());
    const auto done=management_stamp();const auto revision=revision_;lock.unlock();
    result["timing"]=Json{{"lock_wait",management_interval(begin,acquired)},
                         {"validate_install",management_interval(acquired,done)}};
    result["current_revision"]=revision;return result;
}
Json LiveCalendar::install_with_revalidation(const Json&snap,const Json&decision){
    Json first=install_profiled(snap,decision);
    if(first.value("installed",false)||first.at("status")!="context_changed")return first;
    if(snap.is_null()||!snap.at("config").value("revalidate",false))return first;
    // Only one bounded wait, then one candidate validation. The final CAS and
    // original absolute edit window still account for all actual elapsed time.
    const double remaining=n(snap,"snapshot_ms")+n(decision,"commit_by")-monotonic_ms();
    const double reserve=snap.at("config").value("revalidation_ms",2.);
    if(remaining<reserve){first["status"]="revalidation_no_commit_window";return first;}
    const double wait=std::min(snap.at("config").value("proof_wait_ms",1.),std::max(0.,remaining-reserve));
    Json refreshed=snapshot_wait_profiled(wait);const auto begin=management_stamp();
    Json checked=revalidate_candidate(snap,decision,refreshed.at("snapshot"),
                                     reserve);
    const auto end=management_stamp();
    Json result;
    if(checked.at("accepted").get<bool>()){
        result=install_profiled(refreshed.at("snapshot"),checked.at("decision"));
        result["revalidated"]=result.value("installed",false);
    }else result=Json{{"installed",false},{"status",checked.at("status")},{"revalidated",false}};
    result["revalidation_timing"]=management_interval(begin,end);
    result["first_install"]=std::move(first);result["refreshed_context"]=std::move(refreshed);
    result["revalidation"]=std::move(checked);return result;
}
Json LiveCalendar::install_at_for_test(const Json&snap,const Json&decision,double now){std::lock_guard<std::mutex>lock(mutex_);return install_unlocked(snap,decision,now);}
Json LiveCalendar::install_unlocked(const Json&snap,const Json&decision,double now){
    finite_time(now,"install");
    if(!config_.at("adaptive").get<bool>())return rejected("fixed_configuration_no_planning");
    if(snap.is_null()||!decision.value("accepted",false))return rejected("no_feasible_checked_plan");
    if(revision_!=snap.at("calendar").at("revision").get<std::uint64_t>())return rejected("context_changed");
    const double elapsed=now-n(snap,"snapshot_ms");if(elapsed<0||elapsed>n(decision,"commit_by")+epsilon)return rejected("decision_budget_exceeded");
    for(auto v=snap.at("control_versions").begin();v!=snap.at("control_versions").end();++v)if(loops_.at(v.key()).at("control_version")!=v.value())return rejected("control_version_changed");
    Json installed;try{installed=ps_scheduler::commit(snap.at("calendar"),decision,elapsed);}catch(const std::exception&e){return Json{{"installed",false},{"status","calendar_commit_rejected"},{"reason",e.what()}};}
    // Validate into private copies; no operation below can partially affect live
    // releases, queues or control state on a failed guard.
    Json loops=loops_,future=future_modes_,pending=pending_,skips=planned_skip_,updates=Json::array();
    for(const auto&j:installed.at("jobs")){
        const std::string name=s(j,"name"),v=s(j,"vehicle");auto&l=loops.at(v);const auto cycle=l.at("cycle").get<std::uint64_t>();future[name]=j.at("current_mode");
        if(name==key(v,cycle)){
            const double next=n(snap,"snapshot_ms")+n(j,"release"),period=n(j,"slot_deadline")-n(j,"release");
            if(next<=now+epsilon||l.at("retired").get<bool>())return rejected("release_already_due");
            if(next!=n(l,"next_ms")||period!=n(l,"period_ms")){updates.push_back(Json{{"vehicle",v},{"cycle",cycle},{"old_release_ms",l.at("next_ms")},{"next_release_ms",next},{"old_period_ms",l.at("period_ms")},{"period_ms",period}});l["next_ms"]=next;l["period_ms"]=period;}
        }
    }
    if(!decision.at("skipped").is_null()){
        const auto&p=decision.at("permission");const std::string target=s(p,"job"),successor=s(p,"successor");auto colon=target.rfind(':');std::string v=target.substr(0,colon);auto cycle=std::stoull(target.substr(colon+1));const auto&l=loops_.at(v);
        if(pending.count(v)||!l.at("busy").is_null()||l.at("cycle").get<std::uint64_t>()!=cycle||n(l,"next_ms")<=now+epsilon)return rejected("skip_no_longer_retractable");
        const auto sc=successor.rfind(':');skips[v]=cycle;pending[v]=Json{{"cycle",std::stoull(successor.substr(sc+1))},{"deadline_ms",n(snap,"snapshot_ms")+n(p,"hold_until")}};
    }
    double min_slack=std::numeric_limits<double>::infinity();for(const auto&v:decision.at("forecast").at("slack"))min_slack=std::min(min_slack,v.get<double>());
    loops_=std::move(loops);future_modes_=std::move(future);pending_=std::move(pending);planned_skip_=std::move(skips);dispatch_order_=decision.at("order");dispatch_expires_ms_=now+std::max(0.,min_slack);++revision_;
    return Json{{"installed",true},{"status","installed"},{"revision",revision_},{"updates",updates},{"commit_ms",now},{"elapsed_ms",elapsed},{"action",decision.at("action")}};
}
Json LiveCalendar::plan_once(){Json snap=snapshot_current();if(snap.is_null())return rejected("obligation_context_not_ready");Json decision=plan_snapshot(snap);Json result=install(snap,decision);result["snapshot_ms"]=snap.at("snapshot_ms");result["decision"]=decision;return result;}
} // namespace ps_scheduler
