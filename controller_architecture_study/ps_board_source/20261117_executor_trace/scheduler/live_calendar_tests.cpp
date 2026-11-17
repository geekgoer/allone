#include "live_calendar.hpp"
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
using ps_scheduler::Json;
namespace{
int count=0;
void check(bool pass,const std::string&msg){++count;if(!pass)throw std::runtime_error(msg);}
template<typename F>void throws(F fn,const std::string&msg){bool caught=false;try{fn();}catch(const std::exception&){caught=true;}check(caught,msg);}
Json config(){return Json{{"names",Json::array({"Drone1"})},{"origin_ms",100000.},{"period_ms",50.},{"periods_ms",Json::array({20.,50.,80.})},{"model_id",2},{"candidate_ids",Json::array({1,2})},{"prep_ms",.5},{"post_ms",.5},{"decision_ms",3.}};}
Json modes(){return Json::array({Json{{"numeric_id",1},{"service_ms",1.},{"selection_mse",.02}},Json{{"numeric_id",2},{"service_ms",5.},{"selection_mse",.01}}});}
}
int main(){try{
    ps_scheduler::LiveCalendar rt(config(),modes());
    check(rt.snapshot(99999.).is_null(),"before origin has no planner snapshot");
    auto released=rt.release_due(100000.,100500.);
    check(released.size()==1&&released.at(0).at("status")=="sampling","first absolute release admitted");
    check(rt.next_wakeup_ms()==100050.,"calendar advances from release, not completion");
    check(!rt.observe("Drone1",100000.,100200.,1),"wrong held control version rejected");
    check(rt.observe("Drone1",100000.,100200.,0),"matching held control observed");
    check(!rt.observe("Drone1",100000.,100200.,0),"nonfresh observation rejected");
    auto snap=rt.snapshot(100001.);check(!snap.is_null(),"ready context produces finite calendar");
    check(snap.at("calendar").at("jobs").size()==3,"carry + two future obligations retained");
    auto decision=rt.plan_snapshot(snap);
    check(decision.at("accepted").get<bool>(),"default live finite plan accepted");
    check(rt.install_at_for_test(snap,decision,100002.).at("installed").get<bool>(),"fresh plan atomic install");
    check(!rt.install_at_for_test(snap,decision,100002.).at("installed").get<bool>(),"same plan cannot commit twice");
    auto q=rt.queued("Drone1",0,100000.,100002.,100200.);
    check(q.at("actuation_deadline_ms")==100050.,"period deadline frozen at release");
    throws([&]{rt.command_return("Drone1",0,100002.5);},"cannot acknowledge a command before inference returns");
    auto take=rt.take(100003.);check(take.at("dispatch").get<bool>()&&take.at("model_id")==2,"DPU selects current fitting configuration");
    check(rt.take(100003.1).is_null(),"active nonpreemptive job owns DPU");
    throws([&]{rt.finish("Drone1",0,100004.,"dropped");},"cannot software-release active hardware job");
    check(rt.snapshot(100009.).is_null(),"exceeded empirical active envelope invalidates prediction");
    rt.dpu_done("Drone1",0,100008.,true);
    throws([&]{rt.command_return("Drone1",0,100008.1);},"cannot acknowledge a command that was never attempted");
    auto before_command=rt.snapshot(100008.2);auto before_command_plan=rt.plan_snapshot(before_command);
    check(rt.command_begin("Drone1",0,100008.5),"owned successful job starts real command under calendar guard");
    auto paired_epoch=rt.paired_generation();
    check(!rt.wait_for_paired(paired_epoch,0.),"command attempt alone never wakes paired-state planning");
    check(rt.control_version("Drone1")==0,"attempt does not fabricate acknowledged held-control version");
    check(!rt.observe("Drone1",100008.6,100200.,0),"old-control observer rejected while native RPC is pending");
    check(rt.snapshot(100008.6).is_null(),"pending native RPC blocks stale held-state planning context");
    check(rt.install_at_for_test(before_command,before_command_plan,100008.6).at("status")=="context_changed","pre-command snapshot cannot commit during native command RPC");
    throws([&]{rt.command_begin("Drone1",0,100008.7);},"one command attempt per completed inference");
    auto version=rt.command_return("Drone1",0,100009.);
    check(version==1,"actual command return increments held control version");
    throws([&]{rt.command_return("Drone1",0,100009.1);},"duplicate return cannot increment held-control version twice");
    check(rt.snapshot(100009.1).is_null(),"new command invalidates old state/control certificate");
    check(rt.observe("Drone1",100010.,100200.,1),"new applied-control pair accepted");
    check(!rt.wait_for_paired(paired_epoch,0.),"paired observation before slot completion is not a ready event");
    rt.finish("Drone1",0,100011.,"delivered");
    check(rt.wait_for_paired(paired_epoch,0.)&&rt.paired_generation()==paired_epoch+1,
          "completed matching observation advances planner wake generation exactly once");
    check(rt.snapshot_profiled().at("paired_generation")==paired_epoch+1,
          "snapshot and paired wake generation are captured under the same lock");
    check(!rt.wait_for_paired(rt.paired_generation(),0.),"consumed event does not cause a busy retry");
    check(rt.row("Drone1",0).at("status")=="delivered","completed raw row retained");
    auto next=rt.release_due(100050.,100500.);check(next.at(0).at("cycle")==1,"next slot unshifted");
    auto busy=rt.release_due(100100.,100500.);check(busy.at(0).at("status")=="dropped_busy_io","busy release preserved as explicit drop");
    check(rt.row("Drone1",1).at("cancelled").get<bool>(),"expired old unfinished row cancelled");
    rt.finish("Drone1",1,100101.,"dropped_preprocess");
    auto late=rt.release_due(100250.,100500.);check(late.size()==3,"catchup records every planned release");
    check(late.at(0).at("status")=="dropped_release_late"&&late.at(1).at("status")=="dropped_release_late","late releases do not execute old inference");
    check(late.at(2).at("status")=="sampling","latest unexpired release admitted");
    rt.retire("Drone1");check(rt.release_due(100300.,100500.).empty(),"retired loop stops releases");

    auto fixed=config();fixed["adaptive"]=false;ps_scheduler::LiveCalendar fixed_rt(fixed,modes());
    fixed_rt.release_due(100000.,100500.);fixed_rt.observe("Drone1",100000.,100020.,0);
    fixed_rt.queued("Drone1",0,100000.,100001.,100004.);auto rejected=fixed_rt.take(100002.);
    check(!rejected.at("dispatch").get<bool>(),"fixed accurate model refused if only another model fits");
    check(rejected.at("dispatch_rejection")=="no_model_fits_remaining_delivery_budget","guard reason retained");
    fixed_rt.finish("Drone1",0,100002.1,"dropped_inference");
    auto fixed_snap=fixed_rt.snapshot(100003.);auto fixed_decision=fixed_rt.plan_snapshot(fixed_snap);
    check(fixed_rt.install_at_for_test(fixed_snap,fixed_decision,100003.1).at("status")=="fixed_configuration_no_planning","fixed baseline cannot accidentally install adaptation");
    ps_scheduler::LiveCalendar stale_rt(config(),modes());stale_rt.release_due(100000.,100500.);stale_rt.observe("Drone1",100000.,100200.,0);
    auto stale_snap=stale_rt.snapshot(100001.);auto stale_decision=stale_rt.plan_snapshot(stale_snap);
    stale_rt.observe("Drone1",100001.1,100030.,0);
    check(stale_rt.install_at_for_test(stale_snap,stale_decision,100002.).at("status")=="context_changed","hold tightening invalidates plan revision");
    auto fresh_snap=stale_rt.snapshot(100002.);auto fresh_decision=stale_rt.plan_snapshot(fresh_snap);
    auto before=stale_rt.state();auto refused=stale_rt.install_at_for_test(fresh_snap,fresh_decision,100006.);
    check(!refused.at("installed").get<bool>(),"elapsed commit rejected");check(stale_rt.state()==before,"failed group commit has no partial state effects");

    auto sc=config();sc["names"]=Json::array({"A","B"});sc["origin_ms"]=1000.;sc["period_ms"]=10.;sc["periods_ms"]=Json::array({10.});sc["candidate_ids"]=Json::array({2});sc["prep_ms"]=.1;sc["post_ms"]=.1;sc["decision_ms"]=.1;sc["enable_skip"]=true;auto sm=modes();sm.at(1)["service_ms"]=6.;
    ps_scheduler::LiveCalendar skip_rt(sc,sm);skip_rt.release_due(1000.,1100.);skip_rt.observe("A",1000.1,1040.,0,true,"synthetic qualified hold");skip_rt.observe("B",1000.1,1040.,0,true,"synthetic qualified hold");
    auto skip_snap=skip_rt.snapshot(1001.);auto skip_plan=skip_rt.plan_snapshot(skip_snap);
    check(skip_plan.at("accepted").get<bool>()&&!skip_plan.at("skipped").is_null(),"finite overload example requires one qualified skip");
    auto installed=skip_rt.install_at_for_test(skip_snap,skip_plan,1001.05);check(installed.at("installed").get<bool>(),"skip + existing recovery commits atomically");
    check(skip_rt.state().at("pending_recovery").size()==1,"recovery debt recorded");
    check(skip_rt.snapshot(1001.06).is_null(),"already installed skip cannot be replanned away");
    auto events=skip_rt.release_due(1005.,1100.);check(events.size()==1&&events.at(0).at("status")=="planned_skip","skip consumes exact absolute release without period change");
    check(skip_rt.state().at("loops").at("B").at("next_ms")==1015.,"recovery is existing immediate successor, not extra synthetic request");
    check(skip_rt.state().at("config").at("enable_skip").get<bool>(),"skip only explicit opt-in");
    auto recovery_events=skip_rt.release_due(1015.,1100.);Json recovery=nullptr;
    for(const auto&event:recovery_events)if(event.at("vehicle")=="B")recovery=event;
    check(!recovery.is_null()&&recovery.at("cycle")==1&&recovery.count("recovery_due_ms"),"existing successor carries explicit recovery deadline");
    auto recovery_q=skip_rt.queued("B",1,1015.,1015.1,1040.);
    check(recovery_q.at("actuation_deadline_ms")==1025.,"recovery deadline still intersects slot and freshness");
    auto recovery_taken=skip_rt.take(1015.2);check(recovery_taken.at("vehicle")=="B"&&recovery_taken.at("dispatch").get<bool>(),"recovery runs through the real one-DPU dispatch gate");
    skip_rt.dpu_done("B",1,1021.2,true);check(skip_rt.command_begin("B",1,1021.3),"recovery acknowledgement follows a real guarded command attempt");skip_rt.command_return("B",1,1021.4);
    auto recovery_finished=skip_rt.finish("B",1,1021.5,"delivered");
    check(recovery_finished.at("recovery_on_time").get<bool>()&&skip_rt.state().at("pending_recovery").empty(),"debt clears only after actual on-time command acknowledgement");

    auto no_skip_cfg=config();ps_scheduler::LiveCalendar no_skip(no_skip_cfg,modes());
    no_skip.release_due(100000.,100500.);no_skip.observe("Drone1",100000.,100200.,0,true,"stable but no opt-in");
    auto ns=no_skip.snapshot(100001.);check(!ns.at("permissions").at(0).at("eligible").get<bool>(),"stable reference alone never enables skips");
    throws([&]{no_skip.queued("Drone1",0,100002.,100001.,100100.);},"causally inverted input stamps rejected");
    throws([&]{no_skip.take(std::numeric_limits<double>::quiet_NaN());},"NaN never bypasses dispatch budget guard");
    auto before_costs=no_skip.state();throws([&]{no_skip.update_costs(100.,-1.,1.,1.);},"invalid multi-field cost observation rejected");
    check(no_skip.state()==before_costs,"invalid cost update is atomic");

    ps_scheduler::LiveCalendar uncertain(config(),modes());uncertain.release_due(100000.,100500.);
    uncertain.observe("Drone1",100000.,100200.,0);uncertain.queued("Drone1",0,100000.,100001.,100200.);
    uncertain.take(100002.);uncertain.dpu_done("Drone1",0,100007.,true);
    check(uncertain.command_begin("Drone1",0,100008.),"uncertain RPC first had a real attempted submission");
    uncertain.finish("Drone1",0,100009.,"error_no_ack");
    check(uncertain.control_version("Drone1")==0,"unacknowledged attempt never claims a new known held control");
    check(uncertain.state().at("loops").at("Drone1").at("command_pending").get<bool>(),"uncertain delivery remains explicit after worker cleanup");
    check(!uncertain.observe("Drone1",100010.,100200.,0),"cleanup cannot re-enable stale held-control observation after uncertain command");
    check(uncertain.release_due(100050.,100500.).at(0).at("status")=="dropped_uncertain_command","unknown held control cannot start another inference even if caller forgot retirement");

    ps_scheduler::LiveCalendar late_ack(config(),modes());late_ack.release_due(100000.,100500.);
    late_ack.observe("Drone1",100000.,100200.,0);late_ack.queued("Drone1",0,100000.,100001.,100200.);
    late_ack.take(100002.);late_ack.dpu_done("Drone1",0,100007.,true);
    check(late_ack.command_begin("Drone1",0,100008.),"late acknowledgement started within its real command budget");
    check(late_ack.command_return("Drone1",0,100051.)==1,"late actual acknowledgement still updates held control");
    check(!late_ack.state().at("loops").at("Drone1").at("command_pending").get<bool>(),"actual acknowledgement resolves native pending flag even when late");
    late_ack.finish("Drone1",0,100052.,"command_return_late");

    ps_scheduler::LiveCalendar begin_expired(config(),modes());begin_expired.release_due(100000.,100500.);
    begin_expired.queued("Drone1",0,100000.,100001.,100200.);begin_expired.take(100002.);begin_expired.dpu_done("Drone1",0,100007.,true);
    check(!begin_expired.command_begin("Drone1",0,100049.5),"atomic command guard reserves post cost before submission");
    check(!begin_expired.state().at("loops").at("Drone1").at("command_pending").get<bool>(),"refused unsent command creates no uncertain pending state");
    // Reproduce the real early-stop race: caller captured 100049 before waiting
    // for the lock, but a release at 100050 already owns the row under that lock.
    ps_scheduler::LiveCalendar stale(config(),modes());stale.release_due(100000.,100500.);
    stale.observe("Drone1",100000.,100200.,0);stale.queued("Drone1",0,100000.,100001.,100200.);
    stale.take(100002.);stale.dpu_done("Drone1",0,100007.,true);stale.command_begin("Drone1",0,100008.);
    stale.command_return("Drone1",0,100009.);stale.observe("Drone1",100010.,100200.,1);stale.finish("Drone1",0,100011.,"delivered");
    stale.release_due(100050.,100500.);
    check(stale.snapshot(100049.).is_null(),"stale pre-lock time never creates prep_ready before a future carry release");
    auto valid_after_release=stale.snapshot(100050.1);
    check(!valid_after_release.is_null(),"post-lock coherent time still admits valid carry");
    check(stale.plan_snapshot(valid_after_release).at("accepted").get<bool>(),"coherent carry retains original planning result");
    stale.observe("Drone1",100051.,100200.,1);
    check(stale.snapshot(100050.9).is_null(),"stale clock may not observe a future sample");

    ps_scheduler::LiveCalendar history(config(),modes());
    for(int k=0;k<1000;++k){double t=100000.+50.*k;
        history.release_due(t,t+500.);history.observe("Drone1",t,t+200.,k);
        history.queued("Drone1",k,t,t+1.,t+200.);
        auto actual=history.take(t+2.);if(actual.at("cycle").get<int>()!=k)throw std::runtime_error("historical row redispatched");
        history.dpu_done("Drone1",k,t+7.,true);history.command_begin("Drone1",k,t+8.);
        history.command_return("Drone1",k,t+9.);history.finish("Drone1",k,t+10.,"delivered");
    }
    check(history.state().at("loops").at("Drone1").at("control_version")==1000,"1000 real lifecycle seams retain exact completion chain");
    check(history.take(150000.).is_null(),"completed history cannot become dispatch candidates");
    check(history.state().at("future_modes").empty(),"completed carry-mode overrides are not retained");
    auto finite_config=config();finite_config["enable_skip"]=true;finite_config["measurement_stop_ms"]=100075.;
    ps_scheduler::LiveCalendar finite_window(finite_config,modes());finite_window.release_due(100000.,100075.);
    finite_window.observe("Drone1",100000.,100200.,0);finite_window.queued("Drone1",0,100000.,100001.,100200.);
    finite_window.take(100002.);finite_window.dpu_done("Drone1",0,100007.,true);finite_window.command_begin("Drone1",0,100008.);
    finite_window.command_return("Drone1",0,100009.);finite_window.observe("Drone1",100010.,100200.,1,true,"synthetic stable condition");
    finite_window.finish("Drone1",0,100011.,"delivered");
    auto end_snapshot=finite_window.snapshot(100012.);
    check(!end_snapshot.at("permissions").at(0).at("eligible").get<bool>(),"never skip if mandatory existing successor lies outside finite flight window");
    std::cout<<"LIVE_CALENDAR_REGRESSIONS_PASS checks="<<count<<"\n";return 0;
}catch(const std::exception&e){std::cerr<<"LIVE_CALENDAR_REGRESSIONS_FAIL checks="<<count<<" error="<<e.what()<<"\n";return 1;}}
