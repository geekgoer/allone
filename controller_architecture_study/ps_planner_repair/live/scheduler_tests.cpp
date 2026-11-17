#include "scheduler.hpp"
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
using ps_scheduler::Json;
namespace {
int tests=0;
void require(bool pass,const std::string& message){++tests;if(!pass)throw std::runtime_error(message);}
template<typename F>void throws(F fn,const std::string&message){bool caught=false;try{fn();}catch(const std::exception&){caught=true;}require(caught,message);}
bool near(double a,double b){return std::abs(a-b)<1e-9;}
Json job(const char*n,const char*v,double r,double d,Json allowed=Json::array({1}),int current=1){return Json{{"name",n},{"vehicle",v},{"release",r},{"prep_ready",r},{"prep_ms",0.},{"post_ms",0.},{"transport_ms",0.},{"slot_deadline",d},{"hold_deadline",d},{"fresh_deadline",d},{"allowed_modes",allowed},{"current_mode",current},{"release_options",Json::array()},{"recovery",false},{"mode_locked",false},{"periodic_edit",false}};}
Json calendar(const Json& jobs){return Json{{"jobs",jobs},{"revision",0},{"pending",Json::array()}};}
Json modes=Json::array({Json{{"numeric_id",1},{"service_ms",2.},{"selection_mse",.02}},Json{{"numeric_id",2},{"service_ms",3.2},{"selection_mse",.01}},Json{{"numeric_id",3},{"service_ms",1.},{"selection_mse",.04}}});
Json plan(const Json&c,Json options=Json::object()){options["now"]=0.;options["decision_ms"]=0.;return ps_scheduler::plan(c,modes,options);}
Json example(){return calendar(Json::array({job("a","a",0.,10.,Json::array({2}),2),job("b","b",2.5,5.5),job("c","c",4.,7.),job("c_next","c",7.,8.5,Json::array({3}),3)}));}
Json permission(){return Json{{"job","c"},{"successor","c_next"},{"hold_until",8.5},{"valid_until",3.9},{"eligible",true},{"evidence","synthetic test"}};}
}
int main(){try{
    auto j=job("a","a",0.,10.);j["fresh_deadline"]=2.5;j["post_ms"]=1.;require(!plan(calendar(Json::array({j})))["accepted"].get<bool>(),"three deadlines/post enforced");j["post_ms"]=.4;auto d=plan(calendar(Json::array({j})));require(d["accepted"].get<bool>()&&near(d["forecast"]["command_finish"]["a"].get<double>(),2.4),"post finish 2.4");
    auto a=job("a","a",0.,6.),b=job("b","b",0.,6.);a["prep_ms"]=3.;b["prep_ms"]=3.;require(!plan(calendar(Json::array({a,b})))["accepted"].get<bool>(),"shared PS conflict");
    auto c=example();auto p=permission();d=plan(c,Json{{"permissions",Json::array({p})}});require(d["accepted"].get<bool>()&&d["skipped"]=="c","qualified skip");auto installed=ps_scheduler::commit(c,d,0.);require(installed["jobs"].size()==3&&installed["pending"]==Json::array({Json::array({"c","c_next"})}),"existing successor owed");throws([&]{ps_scheduler::delivered(installed,"c_next",9.);},"late recovery cannot clear");require(ps_scheduler::delivered(installed,"c_next",8.)["pending"].empty(),"timely recovery clears");
    for(const auto&patch:Json::array({Json{{"eligible",false}},Json{{"hold_until",7.5}},Json{{"valid_until",-1.}}})){auto q=p;for(auto it=patch.begin();it!=patch.end();++it)q[it.key()]=it.value();require(!plan(example(),Json{{"permissions",Json::array({q})}})["accepted"].get<bool>(),"unqualified/recovery bound rejected");}
    c=example();c["jobs"][2]["release"]=0.;require(!plan(c,Json{{"permissions",Json::array({p})}})["accepted"].get<bool>(),"published skip denied");
    c=example();c["jobs"][3]["recovery"]=true;c["pending"]=Json::array({Json::array({"c","c_next"})});require(!plan(c,Json{{"permissions",Json::array({p})}})["accepted"].get<bool>(),"second skip denied while owed");
    c=calendar(Json::array({job("a","a",0.,10.)}));d=plan(c);auto stale=c;stale["revision"]=1;throws([&]{ps_scheduler::commit(stale,d,0.);},"stale version");throws([&]{ps_scheduler::commit(c,d,.1);},"expired commit");auto tightened=ps_scheduler::tighten_hold(c,"a",1.);throws([&]{ps_scheduler::commit(tightened,d,0.);},"hold revision invalidates");require(!plan(tightened)["accepted"].get<bool>(),"tight hold rechecked");
    require(!plan(calendar(Json::array({job("a","a",0.,3.)})),Json{{"blocked_until",2.}})["accepted"].get<bool>(),"active PL blocking reserved");
    c=calendar(Json::array({job("a","one",0.,5.),job("b","one",1.,6.)}));d=plan(c);require(!d["accepted"].get<bool>(),"same worker reuse");bool overlap=false;for(const auto&t:d["trace"])if(!t["overlap"].empty())overlap=true;require(overlap,"worker overlap evidence");
    c=calendar(Json::array({job("a","a",0.,10.,Json::array({1,2}),1),job("b","b",2.5,5.5),job("c","c",4.,7.)}));Json choice{{"a",1},{"b",1},{"c",1}};require(near(ps_scheduler::dispatch_budget(c["jobs"],Json::array({"a","b","c"}),choice,modes,0.).get<double>(),3.),"tail budget 3");d=plan(c,Json{{"enable_reorder",false}});require(d["accepted"].get<bool>()&&d["choice"]["a"]==1,"upgrade respects tail");
    j=job("a","a",5.,6.);j["release_options"]=Json::array({Json::array({2.,6.})});j["fresh_deadline"]=10.;d=plan(calendar(Json::array({j})));require(d["accepted"].get<bool>()&&d["action"]=="release_edit"&&d["jobs"][0]["release"]==2.&&d["jobs"][0]["hold_deadline"]==6.,"retime preserves hold");j["fresh_deadline"]=6.;require(!plan(calendar(Json::array({j})))["accepted"].get<bool>(),"earlier sample lifetime not extended");
    c=calendar(Json::array({job("a","a",0.,10.),job("b","b",3.,5.)}));d=ps_scheduler::evaluate(c["jobs"],Json::array({"b","a"}),Json{{"a",1},{"b",1}},modes,0.,0.);require(near(d["forecast"]["command_finish"]["b"].get<double>(),5.)&&near(d["forecast"]["command_finish"]["a"].get<double>(),7.),"explicit PL order waits for readiness");
    for(int budget=1;budget<=8;++budget){d=plan(example(),Json{{"permissions",Json::array({p})},{"max_checks",budget}});require(d["checks"].get<int>()<=budget,"bounded checks");if(d["accepted"].get<bool>())require(d["forecast"]["feasible"].get<bool>(),"only verified approval");}
    j=job("a","a",0.,10.,Json::array({1,2}),2);j["mode_locked"]=true;d=plan(calendar(Json::array({j})));require(d["choice"]["a"]==2,"locked model kept");
    c=calendar(Json::array({job("a","a",0.,10.)}));d=plan(c);d["jobs"][0]["release"]=2.;throws([&]{ps_scheduler::commit(c,d,0.);},"published retime not commit");
    j=job("a","a",5.,6.);j["fresh_deadline"]=10.;j["periodic_edit"]=true;j["release_options"]=Json::array({Json::array({2.,6.})});auto next=job("n","a",10.,16.);next["hold_deadline"]=20.;next["fresh_deadline"]=20.;d=plan(calendar(Json::array({j,next})));require(d["accepted"].get<bool>(),"periodic retime feasible");bool next_moved=false;for(const auto&x:d["jobs"])if(x["name"]=="n")next_moved=x["release"]==6.&&x["slot_deadline"]==10.;require(next_moved,"periodic successor synchronously retimed");
    c=example();require(!plan(c,Json{{"enable_skip",false},{"permissions",Json::array({p})}})["accepted"].get<bool>(),"production skip off honored");
    c=calendar(Json::array({job("a","a",0.,10.)}));c["pending"]=Json::array({Json::array({"a","missing"})});throws([&]{plan(c);},"invalid pending rejected");
    std::cout<<"SCHEDULER_REGRESSIONS_PASS checks="<<tests<<"\n";return 0;
}catch(const std::exception&e){std::cerr<<"SCHEDULER_REGRESSIONS_FAIL after="<<tests<<" "<<e.what()<<"\n";return 1;}}
