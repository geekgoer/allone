// CPU-only lifecycle plumbing tests. Fake infer is an explicit injected seam;
// these tests are NOT a substitute for real DNNDK/AirSim flight evidence.
#include "adaptive_live.hpp"
#include <iostream>
#include <cstdio>
#include <unistd.h>
using psadaptive::Json;
namespace a=psadaptive;
int checks=0;
void check(bool pass,const std::string& what){++checks;if(!pass)throw std::runtime_error(what);}
template<class F>void throws(F f,const std::string& what){bool caught=false;try{f();}catch(...){caught=true;}check(caught,what);}
Json config(double origin,double period=50.,bool adaptive=true){return Json{{"names",Json::array({"A"})},{"origin_ms",origin},
    {"period_ms",period},{"periods_ms",Json::array({period})},{"model_id",2},{"candidate_ids",Json::array({1,2})},
    {"prep_ms",.1},{"post_ms",.1},{"rearm_ms",.1},{"command_reserve_ms",.1},{"decision_ms",.1},{"adaptive",adaptive}};}
Json modes(double small=1.,double large=5.){return Json::array({Json{{"numeric_id",1},{"service_ms",small},{"selection_mse",.02}},
    Json{{"numeric_id",2},{"service_ms",large},{"selection_mse",.01}}});}
std::shared_ptr<a::Slot> slot(std::uint64_t cycle=0){auto s=std::make_shared<a::Slot>();s->vehicle="A";s->cycle=cycle;s->log=Json::object();return s;}
int main(){try{
    {
        const auto start=a::now_ms();ps_scheduler::LiveCalendar calendar(config(start),modes());calendar.release_due(start,start+500.);
        std::atomic<int> calls{0};a::Executor executor(calendar,[&](int id,const psdpu::Input&){++calls;check(id==2,"unchanged fitting model reaches infer");return psdpu::Result();});
        auto s=slot();auto future=executor.enqueue(s,start,a::now_ms(),start+500.);
        check(future.wait_for(std::chrono::seconds(1))==std::future_status::ready,"registered queue always resolves");
        const auto reply=future.get();check(reply.dispatched&&calls==1,"exactly one real-path callback per dispatch");
        const auto row=calendar.row("A",0);check(row.at("phase")=="post"&&row.count("board_done_ms"),"dpu_done precedes future readiness");
        check(calendar.state().at("active").is_null(),"real completion cleared active owner");
        calendar.finish("A",0,a::now_ms(),"test_done");executor.close();
    }
    {
        const auto start=a::now_ms();ps_scheduler::LiveCalendar calendar(config(start,20.),modes(1.,100.));calendar.release_due(start,start+500.);
        std::atomic<int> selected{0};a::Executor executor(calendar,[&](int id,const psdpu::Input&){selected=id;return psdpu::Result();});
        auto s=slot();auto future=executor.enqueue(s,start,a::now_ms(),start+500.);const auto reply=future.get();
        check(reply.dispatched&&selected==1,"affordability replacement selected by take reaches actual infer callback");
        check(s->log.at("model_id")==1&&reply.dispatch.at("configuration_id")==1,"actual model retained in raw row");
        calendar.finish("A",0,a::now_ms(),"test_done");
    }
    {
        const auto start=a::now_ms();ps_scheduler::LiveCalendar calendar(config(start,20.,false),modes(1.,100.));calendar.release_due(start,start+500.);
        std::atomic<int> calls{0};a::Executor executor(calendar,[&](int,const psdpu::Input&){++calls;return psdpu::Result();});
        auto s=slot();auto future=executor.enqueue(s,start,a::now_ms(),start+500.);const auto reply=future.get();
        check(!reply.dispatched&&calls==0,"fixed-mode unaffordable dispatch resolves without inference");
        check(reply.result.status==5,"cancelled result explicit status");calendar.finish("A",0,a::now_ms(),"dropped_inference");
    }
    {
        const auto start=a::now_ms();ps_scheduler::LiveCalendar calendar(config(start),modes());calendar.release_due(start,start+500.);
        std::atomic<int> calls{0};a::Executor executor(calendar,[&](int,const psdpu::Input&){++calls;return psdpu::Result();});
        auto s=slot();auto future=executor.enqueue(s,start,a::now_ms(),start-1.);const auto reply=future.get();
        check(!reply.dispatched&&calls==0,"already expired queued row never invokes inference");
        calendar.finish("A",0,a::now_ms(),"dropped_preprocess");
    }
    {
        const auto start=a::now_ms();ps_scheduler::LiveCalendar calendar(config(start,20.),modes(.1,.1));calendar.release_due(start,start+500.);
        std::promise<void> entered,release;auto gate=release.get_future().share();
        a::Executor executor(calendar,[&](int,const psdpu::Input&){entered.set_value();gate.wait_for(std::chrono::milliseconds(100));return psdpu::Result();});
        auto s=slot();auto future=executor.enqueue(s,start,a::now_ms(),start+500.);
        entered.get_future().wait();
        check(future.wait_for(std::chrono::milliseconds(0))!=std::future_status::ready,"active infer not software-completed early");
        throws([&]{calendar.finish("A",0,a::now_ms(),"illegal timeout");},"cannot finish active nonpreemptive infer");
        // Synthetic lifecycle clock jump tests cancellation, not fake DPU timing.
        calendar.release_due(start+20.1,start+500.);
        check(calendar.row("A",0).at("cancelled").get<bool>(),"new release cancels expired active request");
        release.set_value();future.get();
        check(calendar.row("A",0).at("cancelled").get<bool>(),"post-inference authoritative row preserves cancellation");
        calendar.finish("A",0,a::now_ms(),"dropped_before_command");
    }
    {
        const auto start=a::now_ms();ps_scheduler::LiveCalendar calendar(config(start),modes());calendar.release_due(start,start+500.);
        a::Executor executor(calendar,[](int,const psdpu::Input&){return psdpu::Result();});
        auto s=slot();auto future=executor.enqueue(s,start,a::now_ms(),start+500.);future.get();
        check(calendar.command_begin("A",0,a::now_ms()),"on-time command begin invalidates prior held certificate");
        const auto version=calendar.command_return("A",0,start+51.);
        check(version==1,"late actual ack still changes held version");
        check(!calendar.observe("A",start+52.,start+150.,0),"old-control certificate rejected after ack");
        check(calendar.snapshot(start+52.).is_null(),"planner waits for a fresh same-control state pair");
        check(calendar.observe("A",start+52.,start+150.,version),"new state/current held certificate accepted");
        calendar.finish("A",0,start+53.,"command_return_late");
    }
    {
        const auto start=a::now_ms();ps_scheduler::LiveCalendar calendar(config(start),modes());calendar.release_due(start,start+500.);
        a::Executor executor(calendar,[](int,const psdpu::Input&)->psdpu::Result{throw std::runtime_error("injected inference failure");});
        auto s=slot();auto future=executor.enqueue(s,start,a::now_ms(),start+500.);
        throws([&]{future.get();},"infer exception reaches waiting future");
        check(executor.failed(),"executor failure observable");
        throws([&]{executor.enqueue(slot(1),start,a::now_ms(),start+500.);},"enqueue after executor failure refuses rather than hanging future");
        check(calendar.state().at("active").is_null(),"infer throw releases resource only after callback returns");
        calendar.finish("A",0,a::now_ms(),"error");
    }
    {
        const auto path=std::string("/tmp/ps_adaptive_journal_test_")+std::to_string(getpid())+".jsonl";
        {a::Journal log(path,64);for(int i=0;i<20;++i)check(log.submit(Json{{"n",i}}),"journal enqueue accepted");
            log.close();check(!log.failed()&&log.written()==20,"journal drains all accepted rows");
            check(!log.submit(Json{{"late",true}})&&log.failed(),"closed journal refusal becomes visible failure");}
        std::ifstream input(path);std::string line;int n=0;while(std::getline(input,line)){const auto j=Json::parse(line);check(j.at("n")==n,"journal row order preserved");++n;}
        check(n==20,"all raw journal rows readable");std::remove(path.c_str());
    }
    check(std::abs(a::now_ms()-ps_scheduler::monotonic_ms())<10.,"executor and calendar share CLOCK_MONOTONIC epoch");
    std::cout<<"ADAPTIVE_PLUMBING_PASS checks="<<checks<<" scope=CPU_lifecycle_seam_NOT_DPU_evidence\n";return 0;
}catch(const std::exception& e){std::cerr<<"ADAPTIVE_PLUMBING_FAIL checks="<<checks<<" error="<<e.what()<<"\n";return 1;}}
