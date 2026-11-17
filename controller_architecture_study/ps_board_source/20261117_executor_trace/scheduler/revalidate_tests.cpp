#include "live_calendar.hpp"
#include "revalidate.hpp"
#include <iostream>
#include <stdexcept>
#include <thread>
using ps_scheduler::Json;
int checks=0;
void check(bool ok,const std::string& message){++checks;if(!ok)throw std::runtime_error(message);}
Json modes(){return Json::array({Json{{"numeric_id",1},{"service_ms",1.},{"selection_mse",.02}},
    Json{{"numeric_id",2},{"service_ms",5.},{"selection_mse",.01}}});}
Json config(){return Json{{"names",Json::array({"A","B"})},{"origin_ms",1000.},{"period_ms",50.},
    {"periods_ms",Json::array({50.,80.})},{"model_id",2},{"candidate_ids",Json::array({1,2})},
    {"prep_ms",.5},{"post_ms",.5},{"decision_ms",10.},{"revalidate",true},{"revalidation_ms",2.}};}
void complete(ps_scheduler::LiveCalendar& rt,const std::string& v,std::uint64_t cycle,double t){
    rt.queued(v,cycle,t,t+1.,t+200.);auto row=rt.take(t+2.);
    check(row.at("vehicle")==v,"expected lifecycle owner dispatched");
    rt.dpu_done(v,cycle,t+7.,true);rt.command_begin(v,cycle,t+8.);rt.command_return(v,cycle,t+9.);
    rt.observe(v,t+10.,t+200.,rt.control_version(v),true,"synthetic bounded hold");
    rt.finish(v,cycle,t+11.,"delivered");
}
Json proposal(const Json& snap,const Json& raw_modes){
    Json cal=snap.at("calendar");
    for(auto& j:cal["jobs"])if(j.at("name")=="A:1")j["allowed_modes"]=Json::array({1}),j["current_mode"]=1;
    return ps_scheduler::plan(cal,raw_modes,Json{{"now",0.},{"decision_ms",10.},{"max_checks",7},
        {"permissions",snap.at("permissions")},{"enable_skip",false}});
}
int main(){try{
    ps_scheduler::LiveCalendar rt(config(),modes());
    rt.release_due(1000.,1500.);rt.observe("A",1000.,1200.,0);rt.observe("B",1000.,1200.,0);
    complete(rt,"A",0,1000.);
    auto old=rt.snapshot(1012.);auto decision=proposal(old,modes());
    check(decision.at("accepted").get<bool>(),"synthetic future model candidate feasible");
    check(decision.at("choice").at("A:1")==1,"future delta changes only a not-yet-released task");
    rt.observe("B",1013.,1200.,0,true,"fresh synthetic state");
    check(rt.install_at_for_test(old,decision,1014.).at("status")=="context_changed","old global CAS remains strict");
    auto latest=rt.snapshot(1014.);auto checked=ps_scheduler::revalidate_candidate(old,decision,latest,2.);
    check(checked.at("accepted").get<bool>(),"fresh context can approve still-feasible future delta");
    check(checked.at("decision").at("revision")==latest.at("calendar").at("revision"),"revalidation binds latest revision");
    check(checked.at("decision").at("checks").get<int>()==decision.at("checks").get<int>()+1,"one full check added");
    check(checked.at("decision").at("commit_by")==2.,"new check reserve never resets old total budget");
    auto short_budget=checked.at("decision");short_budget["commit_by"]=.1;
    const auto before_late=rt.state();
    check(rt.install_at_for_test(latest,short_budget,1014.2).at("status")=="decision_budget_exceeded","late validation cannot bypass final actual-time guard");
    check(rt.state()==before_late,"expired revalidation makes no partial changes");
    check(rt.install_at_for_test(latest,checked.at("decision"),1014.5).at("installed").get<bool>(),"revalidated future edit commits");
    rt.release_due(1025.,1500.);rt.observe("B",1025.,1200.,0);complete(rt,"B",0,1025.);
    auto released=rt.release_due(1050.,1500.);
    check(released.at(0).at("model_id")==1,"actual absolute release receives revalidated model");

    auto bad=decision;bad["checks"]=8;
    check(ps_scheduler::revalidate_candidate(old,bad,latest,2.).at("status")=="revalidation_check_budget_exhausted","total checks remain bounded by eight");
    bad=decision;bad["commit_by"]=2.5;
    check(ps_scheduler::revalidate_candidate(old,bad,latest,2.).at("status")=="revalidation_no_commit_window","old expiry cannot be refreshed away");
    auto changed=latest;changed["delivering"]=Json::array({"B:0"});
    check(ps_scheduler::revalidate_candidate(old,decision,changed,2.).at("status")=="revalidation_delivery_in_flight","unknown residual post demand cannot be ignored");
    changed=latest;for(auto& j:changed["calendar"]["jobs"])j["hold_deadline"]=.1;
    auto rejected=ps_scheduler::revalidate_candidate(old,decision,changed,2.);
    check(!rejected.at("accepted").get<bool>(),"latest shorter holds block formerly feasible candidate");
    changed=latest;for(auto& m:changed["modes"])m["service_ms"]=200.;
    // Snapshot modes are keyed objects, update values instead of model IDs.
    check(!ps_scheduler::revalidate_candidate(old,decision,changed,2.).at("accepted").get<bool>(),"latest larger service demand is rechecked");
    changed=latest;for(auto& j:changed["calendar"]["jobs"])if(j.at("name")=="A:1")j["release"]=-1.;
    check(ps_scheduler::revalidate_candidate(old,decision,changed,2.).at("status")=="revalidation_edit_no_longer_future","released model edit rejected in first repair version");
    changed=latest;for(auto& j:changed["calendar"]["jobs"])if(j.at("name")=="A:1")j["release"]=40.;
    check(ps_scheduler::revalidate_candidate(old,decision,changed,2.).at("status")=="revalidation_conflicting_release","concurrent retime never silently overwritten");
    changed=latest;for(auto& j:changed["calendar"]["jobs"])if(j.at("name")=="A:1")j["mode_locked"]=true;
    check(ps_scheduler::revalidate_candidate(old,decision,changed,2.).at("status")=="revalidation_model_no_longer_allowed","mode locks preserved");
    check(ps_scheduler::revalidate_candidate(old,decision,nullptr,2.).at("status")=="revalidation_context_not_ready","pending or missing control context is not approved");

    ps_scheduler::LiveCalendar other(config(),modes());other.release_due(1000.,1500.);
    other.observe("A",1000.,1200.,0);other.observe("B",1000.,1200.,0);complete(other,"A",0,1000.);
    auto s0=other.snapshot(1012.);auto p0=proposal(s0,modes());
    other.release_due(1025.,1500.);other.observe("B",1025.,1200.,0);
    auto s1=other.snapshot(1026.);auto p1=ps_scheduler::revalidate_candidate(s0,p0,s1,2.);
    // The original 10ms edit window has actually elapsed at this point.
    check(p1.at("status")=="revalidation_no_commit_window","progress does not excuse elapsed original budget");
    p0["commit_by"]=30.;p1=ps_scheduler::revalidate_candidate(s0,p0,s1,2.);
    check(p1.at("accepted").get<bool>(),"newly released sampling carry can be included by a valid wider test window");
    bool carry=false;for(const auto& j:p1.at("decision").at("jobs"))if(j.at("name")=="B:0")carry=true;
    check(carry,"new other-UAV carry retained in complete validation");
    auto state_before=other.state();other.observe("B",1026.1,1200.,0);
    check(other.install_at_for_test(s1,p1.at("decision"),1026.2).at("status")=="context_changed","a second race is still rejected by final CAS");
    check(other.state().at("future_modes")==state_before.at("future_modes"),"failed final CAS installs no partial future edit");

    auto sc=config();sc["origin_ms"]=1000.;sc["period_ms"]=10.;sc["periods_ms"]=Json::array({10.});
    sc["candidate_ids"]=Json::array({2});sc["prep_ms"]=.1;sc["post_ms"]=.1;
    sc["decision_ms"]=.1;sc["enable_skip"]=true;auto sm=modes();sm.at(1)["service_ms"]=6.;
    ps_scheduler::LiveCalendar skipping(sc,sm);skipping.release_due(1000.,1100.);
    skipping.observe("A",1000.1,1040.,0,true,"synthetic qualified hold");
    skipping.observe("B",1000.1,1040.,0,true,"synthetic qualified hold");
    auto ss0=skipping.snapshot(1001.);auto sp0=skipping.plan_snapshot(ss0);
    check(!sp0.at("skipped").is_null(),"independent overload fixture proposes one skip");
    skipping.observe("A",1001.01,1040.,0,true,"new other-UAV observation");
    auto ss1=skipping.snapshot(1001.02);auto sp1=ps_scheduler::revalidate_candidate(ss0,sp0,ss1,.05);
    check(sp1.at("accepted").get<bool>(),"qualified skip survives unrelated observation after full latest check");
    check(sp1.at("decision").at("skipped")==sp0.at("skipped"),"revalidation never retargets skipped cycle");
    check(sp1.at("decision").at("checks").get<int>()==sp0.at("checks").get<int>()+2,"skip keep/skip checks both count toward eight");
    // Pure validator branch test, not a measured cost or a flight condition.
    changed=ss1;for(auto& m:changed["modes"])m["service_ms"]=1.;
    check(ps_scheduler::revalidate_candidate(ss0,sp0,changed,.05).at("status")=="revalidation_skip_no_longer_needed","a currently feasible keep plan prevents unnecessary feedback loss");
    changed=ss1;changed["control_versions"]["B"]=1;
    check(ps_scheduler::revalidate_candidate(ss0,sp0,changed,.05).at("status")=="revalidation_donor_control_changed","changed donor control invalidates former skip evidence");
    changed=ss1;for(auto& p:changed["permissions"])if(p.at("job")==sp0.at("skipped"))p["eligible"]=false;
    check(ps_scheduler::revalidate_candidate(ss0,sp0,changed,.05).at("status")=="revalidation_skip_not_eligible","fresh eligibility is mandatory");
    changed=ss1;changed["calendar"]["pending"]=Json::array({Json::array({"B","B:1"})});
    check(ps_scheduler::revalidate_candidate(ss0,sp0,changed,.05).at("status")=="revalidation_recovery_already_owed","pending recovery forbids another skip");
    changed=ss1;for(auto& p:changed["permissions"])if(p.at("job")==sp0.at("skipped"))p["hold_until"]=2.;
    check(!ps_scheduler::revalidate_candidate(ss0,sp0,changed,.05).at("accepted").get<bool>(),"shorter actual hold invalidates deferred successor");
    changed=ss1;for(auto& p:changed["permissions"])if(p.at("job")==sp0.at("skipped"))p["hold_until"]=100.;
    auto extension=ps_scheduler::revalidate_candidate(ss0,sp0,changed,.05);
    check(extension.at("accepted").get<bool>(),"fresh larger hold does not itself invalidate existing commitment");
    check(std::abs(extension.at("decision").at("permission").at("hold_until").get<double>()+1001.02-1040.)<1e-6,"old promised recovery bound is never extended");
    check(skipping.install_at_for_test(ss1,sp1.at("decision"),1001.03).at("installed").get<bool>(),"latest-state skip and successor install atomically");
    check(skipping.state().at("pending_recovery").size()==1,"successful revalidation creates exactly one recovery debt");
    auto skipped_event=skipping.release_due(1005.,1100.);
    check(skipped_event.size()==1&&skipped_event.at(0).at("status")=="planned_skip","installed delta consumes actual existing release");
    check(skipping.state().at("loops").at("B").at("next_ms")==1015.,"existing immediate recovery slot retained after revalidation");
    auto wc=config();const double real_origin=ps_scheduler::monotonic_ms()-12.;wc["origin_ms"]=real_origin;
    ps_scheduler::LiveCalendar wake(wc,modes());wake.release_due(real_origin,real_origin+1000.);
    wake.observe("A",real_origin,real_origin+200.,0);complete(wake,"A",0,real_origin);
    auto unavailable=wake.snapshot_wait_profiled(0.);
    check(!unavailable.at("proof_available").get<bool>()&&unavailable.at("snapshot").is_null(),"missing proof remains explicit when no waiting budget");
    std::thread observation([&](){const double t=ps_scheduler::monotonic_ms();wake.observe("B",t,t+200.,0,true,"actual-clock synthetic notification");});
    auto signalled=wake.snapshot_wait_profiled(2.);observation.join();
    check(signalled.at("proof_available").get<bool>()&&!signalled.at("snapshot").is_null(),"proof notification releases calendar lock and yields a fresh coherent snapshot");
    check(signalled.at("wait_requested_ms")==2.,"proof wait has one explicit bounded request");
    std::cout<<"REVALIDATION_REGRESSIONS_PASS checks="<<checks<<"\n";return 0;
}catch(const std::exception& e){std::cerr<<"REVALIDATION_REGRESSIONS_FAIL checks="<<checks<<" "<<e.what()<<"\n";return 1;}}
