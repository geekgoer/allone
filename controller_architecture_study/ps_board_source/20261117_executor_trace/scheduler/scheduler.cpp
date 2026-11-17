#include "scheduler.hpp"
#include "profile.hpp"
namespace ps_profile {thread_local State state;}

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <stdexcept>
#include <tuple>
#include <vector>
#include <time.h>

namespace ps_scheduler {
namespace {
using Jobs = std::vector<Json>;
using Modes = std::map<int, Json>;
using Choice = std::map<std::string, int>;
using Sequence = std::vector<std::string>;
const double inf = std::numeric_limits<double>::infinity();

double number(const Json& value, const char* key) { return value.at(key).get<double>(); }
std::string name_of(const Json& job) { return job.at("name").get<std::string>(); }
double due(const Json& job) {
    return std::min(number(job,"slot_deadline"), std::min(number(job,"hold_deadline"),number(job,"fresh_deadline")));
}
Modes parse_modes(const Json& raw) { PS_SCOPE(Preprocess);
    Modes out;
    if(raw.is_array()) for(const auto& m:raw) out[m.at("numeric_id").get<int>()]=m;
    else if(raw.is_object()) for(auto i=raw.begin();i!=raw.end();++i) out[std::stoi(i.key())]=i.value();
    else throw std::invalid_argument("Expected model array or numeric-keyed object");
    return out;
}
Choice parse_choice(const Json& raw) { PS_SCOPE(Preprocess);
    Choice out;
    for(auto i=raw.begin();i!=raw.end();++i)out[i.key()]=i.value().get<int>();
    return out;
}
Json choice_json(const Choice& choice) {
    Json out=Json::object(); for(const auto& item:choice)out[item.first]=item.second;return out;
}
Jobs parse_jobs(const Json& raw) { PS_SCOPE(Preprocess); return raw.get<Jobs>(); }
const Json& find_job(const Jobs& jobs,const std::string& name) {
    for(const auto& j:jobs)if(name_of(j)==name)return j;
    throw std::invalid_argument("Unknown job: "+name);
}
double service(const Modes& modes,int id) {return number(modes.at(id),"service_ms");}
double quality(const Modes& modes,int id) {return number(modes.at(id),"selection_mse");}
bool edf_less(const Json& a,const Json& b) {
    return std::make_tuple(due(a),number(a,"release"),name_of(a)) < std::make_tuple(due(b),number(b,"release"),name_of(b));
}
Sequence sequence(const Jobs& jobs) {Sequence s;for(const auto&j:jobs)s.push_back(name_of(j));return s;}
Sequence head_first(const Sequence& tail,const std::string& head){Sequence out{head};for(const auto&n:tail)if(n!=head)out.push_back(n);return out;}
int fastest(const Json& job,const Modes& modes) {
    std::vector<int> ids=job.value("mode_locked",false)?std::vector<int>{job.at("current_mode").get<int>()}:job.at("allowed_modes").get<std::vector<int>>();
    return *std::min_element(ids.begin(),ids.end(),[&](int a,int b){return std::make_tuple(service(modes,a),quality(modes,a),a)<std::make_tuple(service(modes,b),quality(modes,b),b);});
}
Json nullable(double value){return std::isfinite(value)?Json(value):Json(nullptr);}
struct Operation {std::string resource;double cost;};
struct Demand {std::string name,loop,mode;double release,deadline;std::vector<Operation> ops;};
std::vector<Demand> demands(const Jobs& jobs,const Choice& choice,const Modes& modes,double now) {
    std::vector<Demand> out;
    for(const auto&j:jobs){const auto n=name_of(j);int mode=choice.at(n);out.push_back({n,j.at("vehicle").get<std::string>(),std::to_string(mode),std::max(0.,number(j,"prep_ready")-now),due(j)-now,
        {{"ps",number(j,"prep_ms")},{"pl",service(modes,mode)},{"ps",number(j,"post_ms")},{"delay",number(j,"transport_ms")}}});}
    return out;
}
Json demand_json(const Demand& d){Json ops=Json::array();for(const auto&o:d.ops)ops.push_back(Json::array({o.resource,o.cost}));return Json{{"name",d.name},{"loop",d.loop},{"release",d.release},{"deadline",d.deadline},{"segments",Json::array()},{"mode",d.mode},{"remaining_operations",ops},{"command_completed",false}};}

// Exact event/tie ordering of Python's (t, serial, resource, name) heap and
// (priority, release, name, ready, duration) PS/PL ready queues.
struct Event{double t;std::uint64_t serial;std::string resource,name;bool named;};
struct EventLater{bool operator()(const Event&a,const Event&b)const{return std::make_pair(a.t,a.serial)>std::make_pair(b.t,b.serial);}};
struct Queued{double priority,release;std::string name;double ready,cost;};
struct QueueLater{bool operator()(const Queued&a,const Queued&b)const{return std::tie(a.priority,a.release,a.name,a.ready,a.cost)>std::tie(b.priority,b.release,b.name,b.ready,b.cost);}};
Json simulate(const std::vector<Demand>& jobs,const Sequence& order,double decision,double block) { PS_SCOPE(Simulation);
    if(order.size()!=jobs.size())throw std::invalid_argument("PL order must contain every job exactly once");
    std::map<std::string,int> rank,progress;
    std::map<std::string,Demand> by_name;
    for(std::size_t k=0;k<order.size();++k)if(!rank.emplace(order[k],int(k)).second)throw std::invalid_argument("Duplicate PL order job");
    for(const auto&j:jobs){if(!rank.count(j.name)||!by_name.emplace(j.name,j).second)throw std::invalid_argument("Invalid job/order identity");progress[j.name]=0;}
    std::priority_queue<Event,std::vector<Event>,EventLater> events;
    std::map<std::string,std::priority_queue<Queued,std::vector<Queued>,QueueLater>> queues;
    std::map<std::string,bool> busy{{"ps",true},{"pl",true}};
    std::uint64_t serial=0;std::size_t position=0;std::string last_pl;
    Json finish=Json::object(),command_finish=Json::object(),waits=Json::object(),timeline=Json::array();
    for(const auto&j:jobs)waits[j.name]=Json{{"ps",0.},{"pl",0.}};
    auto event=[&](double t,const std::string&r,const std::string&n,bool named){events.push({t,serial++,r,n,named});};
    auto advance=[&](const std::string& n,double t){
        const auto&j=by_name.at(n);int k=progress.at(n);
        if(k==int(j.ops.size())){command_finish[n]=t;finish[n]=t;return;}
        const auto&o=j.ops[k];if(o.cost<0)throw std::invalid_argument("Negative execution demand");
        if(o.resource=="delay"){++progress[n];event(t+o.cost,"delay",n,true);}
        else queues[o.resource].push({o.resource=="pl"?double(rank.at(n)):j.deadline,j.release,n,t,o.cost});
    };
    for(const auto&j:jobs)event(std::max(0.,j.release),"release",j.name,true);
    event(decision,"ps","",false);event(block,"pl","",false);
    while(!events.empty()){
        double now=events.top().t;
        while(!events.empty()&&events.top().t<=now+1e-10){Event e=events.top();events.pop();if(busy.count(e.resource))busy[e.resource]=false;if(e.named)advance(e.name,now);}
        for(const std::string r:std::vector<std::string>{"ps","pl"}){
            if(busy[r]||queues[r].empty())continue;
            if(r=="pl"&&(position>=order.size()||queues[r].top().name!=order[position]))continue;
            Queued q=queues[r].top();queues[r].pop();
            waits[q.name][r]=waits[q.name][r].get<double>()+now-q.ready;
            if(r=="pl"){++position;last_pl=q.name;}
            busy[r]=true;++progress[q.name];timeline.push_back(Json{{"resource",r},{"job",q.name},{"ready",q.ready},{"start",now},{"end",now+q.cost}});
            event(now+q.cost,r,q.name,true);
        }
    }
    if(finish.size()!=jobs.size())throw std::runtime_error("Dispatch order failed to drain");
    PS_SCOPE(SimulationReport);
    Json slack=Json::object(),misses=Json::array(),raw_jobs=Json::array();std::string worst;double worst_slack=inf;
    for(const auto&j:jobs){double s=j.deadline-command_finish.at(j.name).get<double>();slack[j.name]=s;if(s < -epsilon)misses.push_back(j.name);if(s<worst_slack){worst_slack=s;worst=j.name;}raw_jobs.push_back(demand_json(j));}
    Json report{{"feasible",misses.empty()},{"worst",worst.empty()?Json(nullptr):Json(worst)},
        {"switches",jobs.empty()?0:int(jobs.size())-1},{"scope","finite deterministic service envelope; not plant or infinite-horizon certificate"}};
    report["misses"]=std::move(misses); report["finish"]=std::move(finish);
    report["command_finish"]=std::move(command_finish); report["slack"]=std::move(slack);
    report["waits"]=std::move(waits); report["timeline"]=std::move(timeline);
    report["jobs"]=std::move(raw_jobs); return report;
}
double tail(const Jobs&jobs,const Sequence&order,const Choice&choice,const Modes&modes,double now){ PS_SCOPE(Budget);
    double latest=inf;
    for(auto i=order.rbegin();i!=order.rend();++i){const auto&j=find_job(jobs,*i);latest=std::min(due(j)-number(j,"post_ms")-number(j,"transport_ms"),latest)-service(modes,choice.at(*i));if(std::max(now,number(j,"prep_ready"))+number(j,"prep_ms")>latest+epsilon)return -inf;}
    return latest-now;
}
double dispatch(const Jobs&jobs,const Sequence&order,const Choice&choice,const Modes&modes,double now){ PS_SCOPE(Budget);
    if(order.empty())throw std::invalid_argument("Missing dispatch head");
    const auto&head=find_job(jobs,order.front());Sequence rest(order.begin()+1,order.end());double tb=tail(jobs,rest,choice,modes,now);if(tb==-inf)return -inf;
    double end=std::min(due(head)-number(head,"post_ms")-number(head,"transport_ms"),now+tb);return end-(std::max(now,number(head,"prep_ready"))+number(head,"prep_ms"));
}
Json evaluate_impl(const Jobs&jobs,const Sequence&order,const Choice&choice,const Modes&modes,double now,double decision,double block){ PS_SCOPE(GlobalCheck);
    Json forecast=simulate(demands(jobs,choice,modes,now),order,decision,std::max(0.,block-now));Json overlap=Json::array();std::set<std::string> vehicles;
    for(const auto&j:jobs)vehicles.insert(j.at("vehicle").get<std::string>());
    for(const auto&v:vehicles){Jobs loop;for(const auto&j:jobs)if(j.at("vehicle")==v)loop.push_back(j);std::stable_sort(loop.begin(),loop.end(),[](const Json&a,const Json&b){return number(a,"release")<number(b,"release");});for(std::size_t i=1;i<loop.size();++i)if(now+forecast.at("finish").at(name_of(loop[i-1])).get<double>()>number(loop[i],"release")+epsilon)overlap.push_back(name_of(loop[i-1]));}
    Json checked{{"accepted",forecast.at("feasible").get<bool>()&&overlap.empty()}};
    checked["overlap"]=std::move(overlap);checked["forecast"]=std::move(forecast);return checked;
}
Json diagnose_impl(const Jobs&jobs,const Choice&choice,const Modes&modes,double now,double decision,double block){ PS_SCOPE(Diagnosis);
    const auto all=demands(jobs,choice,modes,now);Json worst{{"deficit",0.},{"kind","none"},{"start",0.},{"end",0.},{"jobs",Json::array()}};
    for(const auto&j:all){double sum=0;for(const auto&o:j.ops)sum+=o.cost;double earliest=std::max(j.release,decision)+sum;if(earliest-j.deadline>number(worst,"deficit"))worst=Json{{"deficit",earliest-j.deadline},{"kind","path"},{"start",j.release},{"end",j.deadline},{"jobs",Json::array({j.name})}};}
    std::vector<Demand> sorted=all;std::stable_sort(sorted.begin(),sorted.end(),[](const Demand&a,const Demand&b){return a.deadline<b.deadline;});std::set<double> starts{0.};for(const auto&j:all)starts.insert(std::max(0.,j.release));
    for(double a:starts){Json enclosed=Json::array();double ps=0,pl=0;
        for(const auto&j:sorted){if(j.release<a-1e-9)continue;enclosed.push_back(j.name);for(const auto&o:j.ops){if(o.resource=="ps")ps+=o.cost;if(o.resource=="pl")pl+=o.cost;}double b=j.deadline;if(b<=a)continue;
            for(const auto& r:std::vector<std::pair<std::string,double>>{{"ps",decision},{"pl",std::max(0.,block-now)}}){double blocked=std::max(0.,std::min(b,r.second)-a);double deficit=(r.first=="ps"?ps:pl)-(b-a-blocked);if(deficit>number(worst,"deficit")+1e-9)worst=Json{{"deficit",deficit},{"kind",r.first},{"start",a},{"end",b},{"jobs",enclosed}};}
        }
    }
    worst["type"]=worst.at("kind")=="path"?"A":number(worst,"deficit")>epsilon?"B":"C";return worst;
}
void finite_number(double x,const char*label){if(!std::isfinite(x))throw std::invalid_argument(std::string("Non-finite ")+label);}
} // namespace

void validate(const Json&calendar,const Json&raw_modes){ PS_SCOPE(Preprocess);
    auto jobs=parse_jobs(calendar.at("jobs"));auto modes=parse_modes(raw_modes);std::set<std::string> names;
    if(jobs.empty())throw std::invalid_argument("Expected nonempty uniquely named obligations");
    for(const auto&j:jobs){std::string n=name_of(j);if(!names.insert(n).second)throw std::invalid_argument("Duplicate obligation "+n);
        for(const char*k:{"release","prep_ready","prep_ms","post_ms","transport_ms","slot_deadline","hold_deadline","fresh_deadline"})finite_number(number(j,k),k);
        if(std::min(number(j,"prep_ms"),std::min(number(j,"post_ms"),number(j,"transport_ms")))<0||number(j,"prep_ready")<number(j,"release"))throw std::invalid_argument("Invalid obligation "+n);
        auto allowed=j.at("allowed_modes").get<std::vector<int>>();if(allowed.empty()||std::find(allowed.begin(),allowed.end(),j.at("current_mode").get<int>())==allowed.end())throw std::invalid_argument("Invalid allowed/current mode "+n);
        for(int id:allowed)if(!modes.count(id))throw std::invalid_argument("Unknown allowed model");
        for(const auto&p:j.value("release_options",Json::array())){if(!p.is_array()||p.size()!=2)throw std::invalid_argument("Invalid release option");double r=p.at(0).get<double>(),d=p.at(1).get<double>();if(!std::isfinite(r)||!std::isfinite(d)||d<r)throw std::invalid_argument("Invalid release option");}
    }
    for(const auto&m:modes){double s=service(modes,m.first),q=quality(modes,m.first);if(!std::isfinite(s)||s<=0||!std::isfinite(q)||q<0)throw std::invalid_argument("Invalid model profile");}
    for(const auto&p:calendar.value("pending",Json::array())){bool found=false;for(const auto&j:jobs)if(j.at("name")==p.at(1)&&j.at("vehicle")==p.at(0)&&j.value("recovery",false))found=true;if(!found)throw std::invalid_argument("Missing recovery obligation");}
}
Json evaluate(const Json&jobs,const Json&order,const Json&choice,const Json&modes,double now,double decision,double blocked){return evaluate_impl(parse_jobs(jobs),order.get<Sequence>(),parse_choice(choice),parse_modes(modes),now,decision,blocked);}
Json diagnosis(const Json&jobs,const Json&choice,const Json&modes,double now,double decision,double blocked){return diagnose_impl(parse_jobs(jobs),parse_choice(choice),parse_modes(modes),now,decision,blocked);}
Json tail_budget(const Json&jobs,const Json&order,const Json&choice,const Json&modes,double now){return nullable(tail(parse_jobs(jobs),order.get<Sequence>(),parse_choice(choice),parse_modes(modes),now));}
Json dispatch_budget(const Json&jobs,const Json&order,const Json&choice,const Json&modes,double now){return nullable(dispatch(parse_jobs(jobs),order.get<Sequence>(),parse_choice(choice),parse_modes(modes),now));}

Json plan(const Json&calendar,const Json&raw_modes,const Json&options){ PS_SCOPE(Other);
    validate(calendar,raw_modes);const auto modes=parse_modes(raw_modes);double now=number(options,"now"),decision=options.value("decision_ms",.5),blocked=options.value("blocked_until",0.);int max_checks=options.value("max_checks",8),max_candidates=options.value("max_candidates",12);
    finite_number(now,"now");finite_number(decision,"decision_ms");finite_number(blocked,"blocked_until");if(max_checks<1||max_candidates<1||decision<0)throw std::invalid_argument("Invalid decision budget");
    bool enable_modes=options.value("enable_modes",true),enable_reorder=options.value("enable_reorder",true),enable_retime=options.value("enable_retime",true),enable_skip=options.value("enable_skip",true);
    const Jobs original=parse_jobs(calendar.at("jobs"));Jobs initial=original;std::stable_sort(initial.begin(),initial.end(),edf_less);Sequence order=sequence(initial);Choice base,fast;
    for(const auto&j:initial){auto n=name_of(j);base[n]=j.at("current_mode").get<int>();fast[n]=enable_modes?fastest(j,modes):base[n];}
    int checks=0,candidates=0;Json trace=Json::array(),approved=nullptr;double commit_time=now+decision;const Json initial_diagnosis=diagnose_impl(initial,base,modes,now,decision,blocked);
    auto attempt=[&](const Jobs&jobs,const Sequence&seq,const Choice&choice,const std::string&action,const Json&permission){
        PS_SCOPE(Candidate);
        if(candidates>=max_candidates||checks>=max_checks)return;
        ++candidates;
        double screen=dispatch(jobs,seq,choice,modes,std::max(commit_time,blocked));Json delta=nullptr;
        if(!permission.is_null()){Sequence orig=head_first(order,seq.front());Choice orig_choice=fast;for(const auto&c:choice)orig_choice[c.first]=c.second;double before=dispatch(initial,orig,orig_choice,modes,std::max(commit_time,blocked));if(before!=-inf&&screen!=-inf&&std::min(before,screen)>=0)delta=screen-before;}
        ++checks;Json result=evaluate_impl(jobs,seq,choice,modes,now,decision,blocked);trace.push_back(Json{{"action",action},{"accepted",result.at("accepted")},{"optimistic_head_budget_ms",nullable(screen)},{"skip_net_head_budget_ms",delta},{"misses",result.at("forecast").at("misses")},{"overlap",result.at("overlap")}});
        if(!result.at("accepted").get<bool>())return;
        Json pending=calendar.value("pending",Json::array());
        if(!permission.is_null()){const auto&target=find_job(original,permission.at("job").get<std::string>());pending.push_back(Json::array({target.at("vehicle"),permission.at("successor")}));}
        Json updated=Json::array();double commit_by=commit_time;
        if(!permission.is_null())commit_by=std::min(commit_by,number(permission,"valid_until"));
        for(auto j:jobs){j["current_mode"]=choice.at(name_of(j));updated.push_back(j);for(const auto&old:original)if(name_of(old)==name_of(j)&&number(old,"release")!=number(j,"release"))commit_by=std::min(commit_by,number(j,"release"));}
        approved=Json{{"accepted",true},{"revision",calendar.value("revision",0)},{"order",seq},{"choice",choice_json(choice)},{"pending",pending},{"skipped",permission.is_null()?Json(nullptr):permission.at("job")},{"permission",permission},{"commit_by",commit_by},{"action",action}};
        approved["jobs"]=std::move(updated);approved["forecast"]=std::move(result.at("forecast"));
    };
    attempt(initial,order,base,"keep_current",nullptr);
    if(approved.is_null()&&fast!=base)attempt(initial,order,fast,"faster_modes",nullptr);
    if(approved.is_null()&&enable_reorder&&initial_diagnosis.at("type")=="C"){
        Jobs heads=initial;std::stable_sort(heads.begin(),heads.end(),[&](const Json&a,const Json&b){return std::make_pair(std::max(commit_time,number(a,"prep_ready"))+number(a,"prep_ms"),name_of(a))<std::make_pair(std::max(commit_time,number(b,"prep_ready"))+number(b,"prep_ms"),name_of(b));});
        for(std::size_t i=0;i<std::min(std::size_t(2),heads.size());++i){auto seq=head_first(order,name_of(heads[i]));if(seq!=order)attempt(initial,seq,fast,"reorder",nullptr);if(!approved.is_null())break;}
    }
    if(approved.is_null()&&enable_retime){
        for(const auto&target:initial){if(number(target,"release")<=commit_time+epsilon||target.value("recovery",false))continue;
            for(const auto&pair:target.value("release_options",Json::array())){
                double release=pair.at(0).get<double>(),slot=pair.at(1).get<double>();if(release<=commit_time+epsilon)continue;
                Jobs jobs=initial;for(auto&j:jobs)if(name_of(j)==name_of(target)){j["release"]=release;j["slot_deadline"]=slot;j["prep_ready"]=release+(number(target,"prep_ready")-number(target,"release"));j["fresh_deadline"]=release+(number(target,"fresh_deadline")-number(target,"release"));}
                if(target.value("periodic_edit",false)){Jobs following;for(const auto&j:initial)if(j.at("vehicle")==target.at("vehicle")&&number(j,"release")>number(target,"release"))following.push_back(j);std::stable_sort(following.begin(),following.end(),[](const Json&a,const Json&b){return number(a,"release")<number(b,"release");});
                    for(std::size_t k=0;k<following.size();++k){const auto&old=following[k];double r=release+double(k+1)*(slot-release);for(auto&j:jobs)if(name_of(j)==name_of(old)){j["release"]=r;j["prep_ready"]=r+(number(old,"prep_ready")-number(old,"release"));j["slot_deadline"]=r+(slot-release);j["fresh_deadline"]=r+(number(old,"fresh_deadline")-number(old,"release"));}}
                }
                Jobs sorted=jobs;std::stable_sort(sorted.begin(),sorted.end(),edf_less);attempt(jobs,sequence(sorted),fast,"release_edit",nullptr);if(!approved.is_null()||checks>=max_checks||candidates>=max_candidates)break;
            }
            if(!approved.is_null()||checks>=max_checks||candidates>=max_candidates)break;
        }
    }
    if(approved.is_null()&&enable_skip){
        std::vector<Json> eligible;
        for(const auto&p:options.value("permissions",Json::array())){
            bool found_target=false,found_successor=false;Json target,successor;for(const auto&j:initial){if(j.at("name")==p.at("job")){target=j;found_target=true;}if(j.at("name")==p.at("successor")){successor=j;found_successor=true;}}
            if(!p.value("eligible",false)||p.value("evidence",std::string()).empty()||!found_target||!found_successor||!std::isfinite(number(p,"hold_until"))||!std::isfinite(number(p,"valid_until"))||target.value("recovery",false)||number(target,"release")<=commit_time+epsilon||number(p,"valid_until")<commit_time||target.at("vehicle")!=successor.at("vehicle")||number(successor,"release")<=number(target,"release"))continue;
            bool blocked_skip=false;for(const auto&j:initial)if(j.at("vehicle")==target.at("vehicle")&&number(target,"release")<number(j,"release")&&number(j,"release")<number(successor,"release"))blocked_skip=true;
            for(const auto&pending:calendar.value("pending",Json::array()))if(pending.at(0)==target.at("vehicle"))blocked_skip=true;
            if(!blocked_skip)eligible.push_back(p);
        }
        std::stable_sort(eligible.begin(),eligible.end(),[&](const Json&a,const Json&b){return std::make_pair(-service(modes,fast.at(a.at("job").get<std::string>())),a.at("job").get<std::string>())<std::make_pair(-service(modes,fast.at(b.at("job").get<std::string>())),b.at("job").get<std::string>());});
        for(const auto&p:eligible){Jobs jobs;Choice choice;for(auto j:initial)if(j.at("name")!=p.at("job")){if(j.at("name")==p.at("successor")){j["recovery"]=true;j["hold_deadline"]=std::min(number(j,"hold_deadline"),number(p,"hold_until"));}choice[name_of(j)]=fast.at(name_of(j));jobs.push_back(j);}Jobs sorted=jobs;std::stable_sort(sorted.begin(),sorted.end(),edf_less);Sequence seq=sequence(sorted);attempt(jobs,seq,choice,"skip_recover",p);
            if(approved.is_null()&&enable_reorder){Jobs heads=jobs;std::stable_sort(heads.begin(),heads.end(),[&](const Json&a,const Json&b){return std::make_pair(std::max(commit_time,number(a,"prep_ready"))+number(a,"prep_ms"),name_of(a))<std::make_pair(std::max(commit_time,number(b,"prep_ready"))+number(b,"prep_ms"),name_of(b));});for(std::size_t k=0;k<std::min(std::size_t(2),heads.size());++k){Sequence alt=head_first(seq,name_of(heads[k]));if(alt!=seq)attempt(jobs,alt,choice,"skip_recover_reorder",p);if(!approved.is_null()||checks>=max_checks||candidates>=max_candidates)break;}}
            if(!approved.is_null()||checks>=max_checks||candidates>=max_candidates)break;
        }
    }
    if(!approved.is_null()&&enable_modes&&checks<max_checks&&candidates<max_candidates){Json saved=approved;auto jobs=parse_jobs(saved.at("jobs"));auto seq=saved.at("order").get<Sequence>();auto choice=parse_choice(saved.at("choice"));const auto&head=find_job(jobs,seq.front());if(!head.value("mode_locked",false)){double budget=dispatch(jobs,seq,choice,modes,std::max(commit_time,blocked));int best=choice.at(name_of(head));bool any=false;for(int id:head.at("allowed_modes").get<std::vector<int>>())if(budget!=-inf&&service(modes,id)<=budget+epsilon){if(!any||std::make_tuple(quality(modes,id),service(modes,id),id)<std::make_tuple(quality(modes,best),service(modes,best),best)){best=id;any=true;}}if(quality(modes,best)<quality(modes,choice.at(name_of(head)))){choice[name_of(head)]=best;attempt(jobs,seq,choice,"quality_upgrade",saved.at("permission"));if(!trace.back().at("accepted").get<bool>())approved=saved;}}}
    if(approved.is_null())approved=Json{{"accepted",false},{"revision",calendar.value("revision",0)},{"action","no_feasible_checked_plan"}};
    approved["checks"]=checks;approved["candidates"]=candidates;approved["trace"]=trace;approved["diagnosis"]=initial_diagnosis;approved["scope"]="finite PS/PL replay under supplied service and control qualifications";return approved;
}

Json commit(const Json&calendar,const Json&decision,double now){
    finite_number(now,"commit time");if(!decision.at("accepted").get<bool>()||decision.at("revision").get<int>()!=calendar.value("revision",0))throw std::invalid_argument("Rejected or stale decision");if(now>number(decision,"commit_by")+epsilon)throw std::invalid_argument("Decision expired");
    Jobs before=parse_jobs(calendar.at("jobs")),jobs=parse_jobs(decision.at("jobs"));
    if(!decision.at("skipped").is_null()){const auto&p=decision.at("permission");const auto&target=find_job(before,p.at("job").get<std::string>());if(number(target,"release")<=now+epsilon||now>number(p,"valid_until"))throw std::invalid_argument("Skip is no longer retractable");}
    for(const auto&old:before)for(const auto&j:jobs)if(name_of(old)==name_of(j)&&number(old,"release")!=number(j,"release")&&number(old,"release")<=now+epsilon)throw std::invalid_argument("Cannot retime a published job");
    return Json{{"jobs",jobs},{"revision",calendar.value("revision",0)+1},{"pending",decision.at("pending")}};
}
Json delivered(const Json&calendar,const std::string&name,double when){
    const auto jobs=parse_jobs(calendar.at("jobs"));const auto&job=find_job(jobs,name);if(!std::isfinite(when)||!(number(job,"release")<=when&&when<=due(job)+epsilon))throw std::invalid_argument("Late delivery cannot discharge a control obligation");Json pending=Json::array(),remaining=Json::array();for(const auto&p:calendar.value("pending",Json::array()))if(p.at(1)!=name)pending.push_back(p);for(const auto&j:jobs)if(name_of(j)!=name)remaining.push_back(j);return Json{{"jobs",remaining},{"revision",calendar.value("revision",0)+1},{"pending",pending}};
}
Json tighten_hold(const Json&calendar,const std::string&vehicle,double deadline){
    finite_number(deadline,"hold bound");bool found=false;Json jobs=calendar.at("jobs");for(auto&j:jobs)if(j.at("vehicle")==vehicle){found=true;j["hold_deadline"]=std::min(number(j,"hold_deadline"),deadline);}if(!found)throw std::invalid_argument("Expected existing vehicle");return Json{{"jobs",jobs},{"revision",calendar.value("revision",0)+1},{"pending",calendar.value("pending",Json::array())}};
}
double monotonic_ms(){timespec t{};if(clock_gettime(CLOCK_MONOTONIC,&t)!=0)throw std::runtime_error("clock_gettime failed");return double(t.tv_sec)*1000.+double(t.tv_nsec)/1e6;}
CalendarStore::CalendarStore(Json c):calendar_(std::move(c)){}
Json CalendarStore::snapshot()const{std::lock_guard<std::mutex> lock(mutex_);return calendar_;}
Json CalendarStore::commit_at_monotonic_ms(const Json&d,double origin){std::lock_guard<std::mutex> lock(mutex_);const double now=monotonic_ms()-origin;calendar_=commit(calendar_,d,now);return calendar_;}
Json CalendarStore::delivered_at(const std::string&name,double t){std::lock_guard<std::mutex> lock(mutex_);calendar_=delivered(calendar_,name,t);return calendar_;}
Json CalendarStore::tighten(const std::string&v,double d){std::lock_guard<std::mutex> lock(mutex_);calendar_=tighten_hold(calendar_,v,d);return calendar_;}
} // namespace ps_scheduler
