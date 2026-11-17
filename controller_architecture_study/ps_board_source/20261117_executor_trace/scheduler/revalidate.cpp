#include "revalidate.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <tuple>
#include <vector>

namespace ps_scheduler { namespace {
double num(const Json& j,const char* key){return j.at(key).get<double>();}
std::string name(const Json& j){return j.at("name").get<std::string>();}
double due(const Json& j){return std::min(num(j,"slot_deadline"),std::min(num(j,"hold_deadline"),num(j,"fresh_deadline")));}
bool close(double a,double b){return std::abs(a-b)<=1e-6;}
Json reject(const char* reason){return Json{{"accepted",false},{"status",reason}};}
std::map<std::string,Json> index(const Json& jobs){
    std::map<std::string,Json> out;
    for(const auto& j:jobs)if(!out.emplace(name(j),j).second)throw std::invalid_argument("Duplicate revalidation job");
    return out;
}
}
Json revalidate_candidate(const Json& old,const Json& proposal,const Json& latest,double reserve){
    if(old.is_null()||latest.is_null())return reject("revalidation_context_not_ready");
    if(!proposal.value("accepted",false))return reject("revalidation_no_candidate");
    if(!std::isfinite(reserve)||reserve<=0)throw std::invalid_argument("Positive revalidation reserve required");
    const int checks=proposal.at("checks").get<int>();
    const int extra_checks=proposal.at("skipped").is_null()?1:2;
    if(checks+extra_checks>8)return reject("revalidation_check_budget_exhausted");
    const double start=num(old,"snapshot_ms"),now=num(latest,"snapshot_ms");
    const double remaining=start+num(proposal,"commit_by")-now;
    if(now<start||remaining<reserve-epsilon)return reject("revalidation_no_commit_window");
    // The current snapshot's board/post carry is guarded locally, but not an
    // explicit PS obligation. Do not approve a stale delta across that omission.
    if(!latest.count("delivering")||!latest.at("delivering").empty())
        return reject("revalidation_delivery_in_flight");
    auto original=index(old.at("calendar").at("jobs"));
    auto wanted=index(proposal.at("jobs"));
    auto current=index(latest.at("calendar").at("jobs"));
    if(current.size()>64)return reject("revalidation_window_limit");
    bool changed=false;
    for(const auto& item:wanted){
        const auto& id=item.first;const auto& desired=item.second;
        if(!original.count(id))return reject("revalidation_unknown_edit");
        const auto& before=original.at(id);
        const bool retimed=!close(num(desired,"release"),num(before,"release"))||
                           !close(num(desired,"slot_deadline"),num(before,"slot_deadline"));
        const bool remoded=desired.at("current_mode")!=before.at("current_mode");
        if(!retimed&&!remoded)continue;
        changed=true;
        if(!current.count(id))return reject("revalidation_edited_job_no_longer_present");
        auto& job=current.at(id);
        if(num(job,"release")<=reserve+epsilon)return reject("revalidation_edit_no_longer_future");
        if(!close(start+num(before,"release"),now+num(job,"release")))
            return reject("revalidation_conflicting_release");
        if(job.at("current_mode")!=before.at("current_mode")&&job.at("current_mode")!=desired.at("current_mode"))
            return reject("revalidation_conflicting_model");
        if(retimed){
            if(job.value("recovery",false))return reject("revalidation_recovery_retime_forbidden");
            const double release=start+num(desired,"release")-now;
            if(release<=reserve+epsilon)return reject("revalidation_release_already_due");
            job["prep_ready"]=release+(num(job,"prep_ready")-num(job,"release"));
            job["fresh_deadline"]=release+(num(job,"fresh_deadline")-num(job,"release"));
            job["release"]=release;job["slot_deadline"]=start+num(desired,"slot_deadline")-now;
        }
        if(remoded){
            const int selected=desired.at("current_mode").get<int>();
            const auto allowed=job.at("allowed_modes").get<std::vector<int>>();
            if(job.value("mode_locked",false)||std::find(allowed.begin(),allowed.end(),selected)==allowed.end())
                return reject("revalidation_model_no_longer_allowed");
            job["current_mode"]=selected;
        }
    }
    Json permission=nullptr,pending=latest.at("calendar").value("pending",Json::array());
    const Json skipped=proposal.at("skipped");
    if(!skipped.is_null()){
        changed=true;const auto& previous=proposal.at("permission");
        const std::string target=skipped.get<std::string>(),successor=previous.at("successor");
        if(!original.count(target)||!current.count(target)||!current.count(successor))
            return reject("revalidation_skip_identity_missing");
        const std::string vehicle=current.at(target).at("vehicle");
        if(old.at("control_versions").at(vehicle)!=latest.at("control_versions").at(vehicle))
            return reject("revalidation_donor_control_changed");
        for(const auto& p:pending)if(p.at(0)==vehicle)return reject("revalidation_recovery_already_owed");
        for(const auto& p:latest.at("permissions"))if(p.at("job")==target&&p.at("successor")==successor)permission=p;
        if(permission.is_null()||!permission.value("eligible",false)||num(permission,"valid_until")<=reserve+epsilon||
           num(current.at(target),"release")<=reserve+epsilon)
            return reject("revalidation_skip_not_eligible");
        // The initial planner reserved an earlier management interval. After
        // progress, KEEP may already fit. Do not spend a feedback update merely
        // because its old proposal still fits: quality-based skipping is off.
        Json keep_jobs=Json::array(),keep_choice=Json::object(),keep_order=Json::array();
        for(const auto& item:current){keep_jobs.push_back(item.second);keep_choice[item.first]=item.second.at("current_mode");}
        std::vector<std::string> keep_names;for(const auto& item:current)keep_names.push_back(item.first);
        std::sort(keep_names.begin(),keep_names.end(),[&](const std::string& a,const std::string& b){
            return std::make_tuple(due(current.at(a)),num(current.at(a),"release"),a)<
                   std::make_tuple(due(current.at(b)),num(current.at(b),"release"),b);});
        for(const auto& id:keep_names)keep_order.push_back(id);
        Json keep=evaluate(keep_jobs,keep_order,keep_choice,latest.at("modes"),0.,reserve,num(latest,"blocked_until_ms"));
        if(keep.at("accepted").get<bool>())return reject("revalidation_skip_no_longer_needed");
        const double hold=std::min(start+num(previous,"hold_until"),now+num(permission,"hold_until"))-now;
        permission["hold_until"]=hold;
        permission["valid_until"]=std::min(start+num(previous,"valid_until")-now,num(permission,"valid_until"));
        auto& next=current.at(successor);next["recovery"]=true;
        next["hold_deadline"]=std::min(num(next,"hold_deadline"),hold);
        pending.push_back(Json::array({vehicle,successor}));current.erase(target);
    }
    // Retain the proposed relative order for surviving jobs. Insert newly
    // released work by deadline; approve only if the resulting full order fits.
    Json order=Json::array();std::set<std::string> seen;
    for(const auto& key:proposal.at("order")){
        const std::string id=key;if(current.count(id)&&seen.insert(id).second)order.push_back(id);
    }
    std::vector<std::string> missing;
    for(const auto& item:current)if(!seen.count(item.first))missing.push_back(item.first);
    std::sort(missing.begin(),missing.end(),[&](const std::string& a,const std::string& b){
        return std::make_pair(due(current.at(a)),a)<std::make_pair(due(current.at(b)),b);});
    for(const auto& id:missing){
        Json merged=Json::array();bool added=false;
        for(const auto& existing:order){
            if(!added&&due(current.at(id))<due(current.at(existing.get<std::string>()))){merged.push_back(id);added=true;}
            merged.push_back(existing);
        }
        if(!added)merged.push_back(id);
        order=std::move(merged);
    }
    if(!changed){
        std::vector<std::string> edf;for(const auto& item:original)edf.push_back(item.first);
        std::sort(edf.begin(),edf.end(),[&](const std::string& a,const std::string& b){
            return std::make_tuple(due(original.at(a)),num(original.at(a),"release"),a)<
                   std::make_tuple(due(original.at(b)),num(original.at(b),"release"),b);});
        if(proposal.at("order")==Json(edf))return reject("context_advanced_no_configuration_change");
    }
    Json jobs=Json::array(),choice=Json::object();
    for(const auto& item:current){jobs.push_back(item.second);choice[item.first]=item.second.at("current_mode");}
    Json calendar{{"jobs",jobs},{"revision",latest.at("calendar").at("revision")},
                  {"pending",latest.at("calendar").value("pending",Json::array())}};
    validate(calendar,latest.at("modes"));
    Json checked=evaluate(jobs,order,choice,latest.at("modes"),0.,reserve,num(latest,"blocked_until_ms"));
    if(!checked.at("accepted").get<bool>())
        return Json{{"accepted",false},{"status","revalidation_pipeline_infeasible"},{"forecast",checked.at("forecast")}};
    double commit_by=std::min(remaining,reserve);
    if(!permission.is_null())commit_by=std::min(commit_by,num(permission,"valid_until"));
    Json decision{{"accepted",true},{"revision",calendar.at("revision")},{"jobs",jobs},{"choice",choice},
        {"order",order},{"pending",pending},{"skipped",skipped},{"permission",permission},{"commit_by",commit_by},
        {"action",proposal.at("action")},{"checks",checks+extra_checks},{"candidates",proposal.at("candidates")},
        {"revalidation_checks",extra_checks},{"scope","one latest-state candidate, bounded full checks; empirical costs"}};
    decision["forecast"]=std::move(checked.at("forecast"));
    return Json{{"accepted",true},{"status","revalidated_candidate"},{"decision",std::move(decision)}};
}
}
