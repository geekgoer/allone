#include "live_calendar.hpp"
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
using ps_scheduler::Json;
namespace{
int count=0;
double fake_dispatch_ms=0.,fake_dispatch_step_ms=0.;
double dispatch_clock(){const auto current=fake_dispatch_ms;fake_dispatch_ms+=fake_dispatch_step_ms;return current;}
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
    double take_lock_wait=-1.;
    auto take=rt.take(100003.,&take_lock_wait);check(take.at("dispatch").get<bool>()&&take.at("model_id")==2,"DPU selects current fitting configuration");
    check(std::isfinite(take_lock_wait)&&take_lock_wait>=0.,"calendar lock wait is reported for an instrumented take");
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

    auto paired_config=config();
    paired_config["names"]=Json::array({"Drone1","Drone2"});
    auto watchdog=paired_config;
    watchdog["retire_on_hold_expiry"]=true;watchdog["maximum_update_ms"]=160.;
    ps_scheduler::LiveCalendar automatic(watchdog,modes());
    automatic.release_due(100025.,100500.);
    automatic.observe("Drone1",100025.,100035.,0);
    automatic.observe("Drone2",100025.,100300.,0);
    automatic.queued("Drone1",0,100025.,100025.1,100035.);
    automatic.queued("Drone2",0,100025.,100025.2,100300.);
    check(automatic.take(100026.).at("vehicle")=="Drone1","watchdog target owns active DPU");
    Json expiries;
    automatic.release_due(100035.,100500.,&expiries);
    check(expiries.size()==1&&expiries[0]["vehicle"]=="Drone1"&&expiries[0]["control_version"]==0,
          "publisher emits expiry atomically before releasing ordinary work");
    check(automatic.take(100035.1).is_null(),"expiry does not release active hardware");
    automatic.dpu_done("Drone1",0,100035.2,true);
    check(!automatic.command_begin("Drone1",0,100035.3),"expired completion cannot issue a command");
    automatic.finish("Drone1",0,100035.4,"dropped_before_command");
    check(automatic.take(100035.5).at("vehicle")=="Drone2","other vehicle still dispatches");
    check(automatic.expire_holds(100036.).empty(),"expiry and its event are one-shot");
    check(automatic.release_due(100075.,100500.).size()==1,"only survivor continues releasing");

    auto one_watchdog=config();
    one_watchdog["retire_on_hold_expiry"]=true;one_watchdog["maximum_update_ms"]=160.;
    ps_scheduler::LiveCalendar pending_watch(one_watchdog,modes());
    pending_watch.release_due(100000.,100500.);
    pending_watch.observe("Drone1",100000.,100020.,0);
    pending_watch.queued("Drone1",0,100000.,100001.,100020.);
    pending_watch.take(100002.);
    pending_watch.dpu_done("Drone1",0,100007.,true);
    pending_watch.command_begin("Drone1",0,100008.);
    check(pending_watch.command_return("Drone1",0,100020.)==1,"ACK at expiry is still recorded");
    const auto pending_events=pending_watch.expire_holds(100021.);
    check(pending_events.size()==1&&pending_events[0]["command_in_flight"]==true&&
          pending_events[0]["control_version"]==0,"late ACK cannot hide the original generation expiry");

    ps_scheduler::LiveCalendar cap_watch(one_watchdog,modes());
    cap_watch.release_due(100000.,100500.);
    cap_watch.queued("Drone1",0,100000.,100001.,100040.);
    cap_watch.take(100002.);
    cap_watch.dpu_done("Drone1",0,100007.,true);
    cap_watch.command_begin("Drone1",0,100008.);
    cap_watch.command_return("Drone1",0,100009.);
    check(cap_watch.observe("Drone1",100100.,100500.,1),"fresh observation is compatible with the ACK cap");
    check(cap_watch.expire_holds(100168.9).empty(),"ACK cap is an absolute frontier");
    check(!cap_watch.observe("Drone1",100168.,100500.,1,false,"late analysis",100169.1),
          "sample processed after expiry cannot refresh the old version");
    const auto cap_events=cap_watch.expire_holds(100170.);
    check(cap_events.size()==1&&cap_events[0]["hold_deadline_ms"]==100169.,"observation cannot extend the ACK cap");

    ps_scheduler::LiveCalendar expired_loop(paired_config,modes());
    expired_loop.release_due(100025.,100150.);
    expired_loop.queued("Drone1",0,100025.,100025.1,100100.);
    expired_loop.queued("Drone2",0,100025.,100025.2,100100.);
    check(expired_loop.take(100026.).at("vehicle")=="Drone1","expiring loop can already own the DPU");
    expired_loop.retire("Drone1");
    check(expired_loop.take(100040.).is_null(),"retirement does not release a running DPU job");
    expired_loop.dpu_done("Drone1",0,100041.,true);
    check(!expired_loop.command_begin("Drone1",0,100041.2),"retired loop cannot submit its finished output even within the old job deadline");
    expired_loop.finish("Drone1",0,100041.3,"dropped_before_command");
    check(expired_loop.take(100041.4).at("vehicle")=="Drone2","another vehicle keeps its place in the DPU queue");
    check(expired_loop.release_due(100050.,100150.).empty(),"retired loop stops generating new control jobs");

    ps_scheduler::LiveCalendar late_after_retire(config(),modes());
    late_after_retire.release_due(100000.,100150.);
    late_after_retire.queued("Drone1",0,100000.,100001.,100100.);
    late_after_retire.take(100002.);
    late_after_retire.dpu_done("Drone1",0,100007.,true);
    check(late_after_retire.command_begin("Drone1",0,100008.),"an RPC can already be outstanding when expiry is detected");
    late_after_retire.retire("Drone1");
    check(late_after_retire.command_return("Drone1",0,100051.)==1,"retirement still records the actual late command ACK");
    check(!late_after_retire.observe("Drone1",100052.,100100.,1)&&
          late_after_retire.release_due(100100.,100150.).empty(),"late ACK cannot reopen a retired control loop");

    auto fixed=config();fixed["adaptive"]=false;ps_scheduler::LiveCalendar fixed_rt(fixed,modes());
    fixed_rt.release_due(100000.,100500.);fixed_rt.observe("Drone1",100000.,100020.,0);
    fixed_rt.queued("Drone1",0,100000.,100001.,100004.);auto rejected=fixed_rt.take(100002.);
    check(!rejected.at("dispatch").get<bool>(),"fixed accurate model refused if only another model fits");
    check(rejected.at("dispatch_rejection")=="no_model_fits_remaining_delivery_budget","guard reason retained");
    const auto fixed_evidence=rejected.at("dispatch_rejection_evidence");
    check(fixed_evidence.at("required_model_id")==2&&fixed_evidence.at("required_service_ms")==5.&&
          fixed_evidence.at("budget_ms")==1.5&&fixed_evidence.at("command_reserve_ms")==.5&&
          fixed_evidence.at("checked_ms")==100002.,"fixed rejection records the actual checked model and budget");
    fixed_rt.finish("Drone1",0,100002.1,"dropped_inference");
    auto fixed_snap=fixed_rt.snapshot(100003.);auto fixed_decision=fixed_rt.plan_snapshot(fixed_snap);
    check(fixed_rt.install_at_for_test(fixed_snap,fixed_decision,100003.1).at("status")=="fixed_configuration_no_planning","fixed baseline cannot accidentally install adaptation");
    auto spike_config=config();spike_config["candidate_ids"]=Json::array({2});
    ps_scheduler::LiveCalendar spike(spike_config,modes());
    spike.release_due(100000.,100150.);
    spike.queued("Drone1",0,100000.,100001.,100100.);
    spike.take(100002.);spike.dpu_done("Drone1",0,100014.,true);
    spike.finish("Drone1",0,100015.,"dropped_before_command");
    spike.release_due(100050.,100150.);
    spike.queued("Drone1",1,100050.,100051.,100062.);
    auto spike_rejected=spike.take(100052.).at("dispatch_rejection_evidence");
    check(spike_rejected.at("required_service_ms")==12.&&spike_rejected.at("budget_ms")==9.5,
          "adaptive rejection freezes the raised service estimate and remaining delivery budget");
    auto separated_config=config();separated_config["separate_dispatch_service"]=true;
    ps_scheduler::LiveCalendar separated(separated_config,modes());
    separated.release_due(100000.,100150.);
    separated.queued("Drone1",0,100000.,100001.,100100.);
    fake_dispatch_ms=100009.;fake_dispatch_step_ms=1.;
    auto selected=separated.take(100002.,nullptr,dispatch_clock);
    separated.dpu_done("Drone1",0,100014.,true);
    check(selected.at("board_start_ms")==100010.&&separated.state().at("modes").at("2").at("service_ms")==5.,
          "wait and selection are excluded from the learned model cost");
    separated.finish("Drone1",0,100015.,"dropped_before_command");
    separated.release_due(100050.,100150.);
    separated.queued("Drone1",1,100050.,100051.,100064.);
    fake_dispatch_ms=100058.;fake_dispatch_step_ms=2.;
    auto late_dispatch=separated.take(100052.,nullptr,dispatch_clock);
    check(late_dispatch.at("model_id")==1&&late_dispatch.at("dispatch_handoff_downgraded_from")==2&&
          late_dispatch.at("board_start_ms")==100062.,"stale mode falls back once to a cheaper model that still fits");
    separated.dpu_done("Drone1",1,100063.,true);
    separated.finish("Drone1",1,100064.,"dropped_before_command");
    separated.release_due(100100.,100150.);
    separated.queued("Drone1",2,100100.,100101.,100113.);
    fake_dispatch_ms=100107.;fake_dispatch_step_ms=3.;
    auto no_room=separated.take(100102.,nullptr,dispatch_clock);
    check(no_room.at("dispatch_rejection")=="no_model_fits_at_dispatch_handoff"&&
          no_room.at("dispatch_rejection_evidence").at("budget_ms")==-.5&&separated.state().at("active").is_null(),
          "without time for even the cheapest alternative the DPU remains free");
    ps_scheduler::LiveCalendar abandoned(config(),modes());
    abandoned.release_due(100000.,100200.);
    auto expired_input=abandoned.queued("Drone1",0,100000.,100005.,100004.);
    check(expired_input.at("cancelled").get<bool>(),"executor resolves the expired input during enqueue");
    check(abandoned.take(100005.1).is_null(),"expiry resolved by enqueue cannot be dispatched before worker finish");
    abandoned.finish("Drone1",0,100006.,"dropped_inference");
    abandoned.release_due(100050.,100200.);
    auto valid_input=abandoned.queued("Drone1",1,100050.,100051.,100054.);
    check(!valid_input.at("cancelled").get<bool>(),"an input ready before its deadline keeps its payload registered");
    auto expired_in_queue=abandoned.take(100054.);
    check(expired_in_queue.at("cycle")==1&&!expired_in_queue.at("dispatch").get<bool>(),
          "dispatch still notifies the worker when an accepted input expires in the queue");
    auto queue_cfg=config();queue_cfg["names"]=Json::array({"A","B","C"});queue_cfg["origin_ms"]=1000.;queue_cfg["period_ms"]=30.;
    queue_cfg["periods_ms"]=Json::array({30.});queue_cfg["prep_ms"]=.1;queue_cfg["post_ms"]=.5;queue_cfg["decision_ms"]=.1;
    auto queue_modes=modes();queue_modes.at(0)["service_ms"]=4.;queue_modes.at(1)["service_ms"]=8.;
    ps_scheduler::LiveCalendar queue_rt(queue_cfg,queue_modes);
    check(queue_rt.release_due(1020.,1100.).size()==3,"three staggered releases enter the same DPU queue");
    queue_rt.queued("A",0,1000.,1020.5,1040.);
    queue_rt.queued("B",0,1010.,1020.6,1034.);
    queue_rt.queued("C",0,1020.,1020.7,1036.);
    check(1021.+8.+4.+4.+.5>1036.,"old local-only head choice leaves the third job late even after two downgrades");
    auto first_queue=queue_rt.take(1021.);
    check(first_queue.at("vehicle")=="A"&&first_queue.at("model_id")==1&&
          first_queue.at("ready_queue_fast_tail_fits").get<bool>(),
          "head is shortened only when even the fastest downstream modes cannot save the old head");
    queue_rt.dpu_done("A",0,1025.,true);
    auto second_queue=queue_rt.take(1025.);
    check(second_queue.at("vehicle")=="B"&&second_queue.at("model_id")==1,"second dispatch leaves time for the last queued job");
    queue_rt.dpu_done("B",0,1029.,true);
    auto third_queue=queue_rt.take(1029.);
    check(third_queue.at("vehicle")=="C"&&third_queue.at("model_id")==1&&1029.+4.+.5<1036.,
          "all three estimated DPU services fit the frozen queue deadlines");
    auto roomy_cfg=queue_cfg;roomy_cfg["names"]=Json::array({"A","B"});
    ps_scheduler::LiveCalendar roomy(roomy_cfg,queue_modes);
    roomy.release_due(1015.,1100.);roomy.queued("A",0,1000.,1015.1,1040.);roomy.queued("B",0,1015.,1015.2,1040.);
    auto roomy_head=roomy.take(1016.);
    check(roomy_head.at("model_id")==2&&roomy_head.at("ready_queue_fast_tail_fits").get<bool>(),
          "when the queued deadlines have room, retain the more accurate committed model");
    ps_scheduler::LiveCalendar tail_repair(roomy_cfg,queue_modes);
    tail_repair.release_due(1015.,1100.);tail_repair.queued("A",0,1000.,1015.1,1025.);tail_repair.queued("B",0,1015.,1015.2,1030.);
    auto tail_head=tail_repair.take(1016.);
    check(tail_head.at("model_id")==2&&tail_head.at("ready_queue_fast_tail_fits").get<bool>(),
          "head keeps its model when only a later queued job needs to downshift");
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
