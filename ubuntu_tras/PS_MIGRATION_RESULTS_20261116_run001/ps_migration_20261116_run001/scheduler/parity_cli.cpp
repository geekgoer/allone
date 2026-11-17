#include "scheduler.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>

int main(int argc,char**argv){
    try{
        using ps_scheduler::Json;
        if(argc!=2&&argc!=3)throw std::invalid_argument("Usage: scheduler_parity fixture.json [output.json]; or scheduler_parity --request request.json");
        bool request=argc==3&&std::string(argv[1])=="--request";
        std::ifstream stream(request?argv[2]:argv[1]);if(!stream)throw std::runtime_error("Cannot open input");Json data;stream>>data;
        if(request){const auto command=data.value("command",std::string("plan"));Json result;
            if(command=="plan")result=ps_scheduler::plan(data.at("calendar"),data.at("modes"),data.at("options"));
            else if(command=="commit")result=ps_scheduler::commit(data.at("calendar"),data.at("decision"),data.at("now").get<double>());
            else if(command=="delivered")result=ps_scheduler::delivered(data.at("calendar"),data.at("name").get<std::string>(),data.at("actuation_time").get<double>());
            else if(command=="tighten_hold")result=ps_scheduler::tighten_hold(data.at("calendar"),data.at("vehicle").get<std::string>(),data.at("deadline").get<double>());
            else throw std::invalid_argument("Unknown command");
            std::cout<<result.dump(2)<<'\n';return 0;
        }
        Json rows=Json::array();bool passed=true;
        for(const auto&row:data.at("cases")){
            Json options{{"now",0.},{"decision_ms",data.at("decision_ms")},{"permissions",row.at("permissions")},{"enable_skip",true}};
            Json actual=ps_scheduler::plan(row.at("calendar"),data.at("modes"),options),selected=Json::object();
            for(auto it=row.at("expected").begin();it!=row.at("expected").end();++it)selected[it.key()]=actual.count(it.key())?actual.at(it.key()):Json(nullptr);
            bool pass=selected==row.at("expected");passed=passed&&pass;rows.push_back(Json{{"seed",row.at("seed")},{"pass",pass},{"expected",row.at("expected")},{"actual",selected},{"full_decision",actual}});
            std::cerr<<"SCHEDULER_FIXTURE seed="<<row.at("seed")<<" pass="<<pass<<" action="<<actual.at("action")<<" checks="<<actual.at("checks")<<'\n';
        }
        Json report{{"status",passed?"PASS":"FAIL"},{"cases",rows},{"scope","numerical frozen-fixture parity, not live flight or WCET"}};
        if(argc==3){std::ofstream out(argv[2]);if(!out)throw std::runtime_error("Cannot write output");out<<report.dump(2)<<'\n';}
        else std::cout<<report.dump(2)<<'\n';
        return passed?0:2;
    }catch(const std::exception&e){std::cerr<<"ERROR "<<e.what()<<'\n';return 1;}
}
