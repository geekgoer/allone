#include <rpc/server.h>
#include <rpc/msgpack.hpp>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <mutex>
#include <regex>
#include <string>
#include <stdexcept>
#include <vector>

// Test-only wire server. No physics, neural inference or real flight.
namespace {
struct Vec { float x_val=0,y_val=0,z_val=0; MSGPACK_DEFINE_MAP(x_val,y_val,z_val); };
struct Quat { float w_val=1,x_val=0,y_val=0,z_val=0; MSGPACK_DEFINE_MAP(w_val,x_val,y_val,z_val); };
struct Pose { Vec position; Quat orientation; MSGPACK_DEFINE_MAP(position,orientation); };
struct Kin { Vec position,linear_velocity,angular_velocity,linear_acceleration,angular_acceleration;Quat orientation;
MSGPACK_DEFINE_MAP(position,orientation,linear_velocity,angular_velocity,linear_acceleration,angular_acceleration);};
struct Collision {bool has_collided=false;MSGPACK_DEFINE_MAP(has_collided);};
struct State {Kin kinematics_estimated;Collision collision;std::uint64_t timestamp=99;int landed_state=1;
MSGPACK_DEFINE_MAP(kinematics_estimated,collision,timestamp,landed_state);};
}
int main(int argc,char** argv){
    try{
        const int port=argc>1?std::stoi(argv[1]):43464;const bool foreign=argc>2&&std::string(argv[2])=="foreign";
        rpc::server server("127.0.0.1",static_cast<std::uint16_t>(port));
        std::mutex lock;bool paused=false;std::map<std::string,State> states;
        std::map<std::string,Pose> actors;
        actors["PSMIG20261116_previous_owned_actor"]=Pose{};
        states["Drone1"]=State{};states["Drone2"]=State{};
        server.bind("ping",[](){return true;});
        server.bind("listVehicles",[](){return std::vector<std::string>{"Drone1","Drone2"};});
        server.bind("simListAssets",[](){return std::vector<std::string>{"EditorCube"};});
        server.bind("simListSceneObjects",[&](const std::string& pattern){
            std::lock_guard<std::mutex> hold(lock);std::vector<std::string> result;
            const std::regex regex(pattern);
            if(foreign&&std::regex_match("FalconUserGate_1_left",regex))result.push_back("FalconUserGate_1_left");
            for(const auto& item:actors)if(std::regex_match(item.first,regex))result.push_back(item.first);return result;});
        server.bind("simDestroyObject",[&](const std::string& name){
            std::lock_guard<std::mutex> hold(lock);std::cout<<"DESTROY "<<name<<'\n'<<std::flush;return actors.erase(name)>0;});
        server.bind("reset",[&](){std::lock_guard<std::mutex> hold(lock);for(auto& item:states)item.second=State{};std::cout<<"RESET\n"<<std::flush;});
        server.bind("simPause",[&](bool p){std::lock_guard<std::mutex> hold(lock);paused=p;});
        server.bind("simIsPaused",[&](){std::lock_guard<std::mutex> hold(lock);return paused;});
        server.bind("simGetVehiclePose",[&](const std::string& name){
            std::lock_guard<std::mutex> hold(lock);const auto& state=states.at(name);Pose p;p.position=state.kinematics_estimated.position;p.orientation=state.kinematics_estimated.orientation;return p;});
        server.bind("simGetObjectPose",[&](const std::string& name){
            std::lock_guard<std::mutex> hold(lock);const auto& state=states.at(name);Pose p;p.position=state.kinematics_estimated.position;p.orientation=state.kinematics_estimated.orientation;
            if(name=="Drone2")p.position.y_val+=20.f;return p;});
        server.bind("enableApiControl",[](bool,const std::string&){});
        server.bind("armDisarm",[](bool,const std::string&){return true;});
        server.bind("simSetKinematics",[&](Kin k,bool ignore,const std::string& name){
            std::lock_guard<std::mutex> hold(lock);if(!paused||!ignore)throw std::runtime_error("Init must be paused and explicitly ignore collisions");states.at(name).kinematics_estimated=k;});
        server.bind("getMultirotorState",[&](const std::string& name){std::lock_guard<std::mutex> hold(lock);return states.at(name);});
        server.bind("moveByAngleRatesThrottle",[&](float x,float y,float z,float throttle,float duration,const std::string&){
            if(x!=0||y!=0||z!=0||std::abs(throttle-.6f)>1e-6||duration!=1.f)throw std::runtime_error("Incorrect prime command");return true;});
        server.bind("simSpawnObject",[&](const std::string& name,const std::string& asset,Pose pose,Vec scale,bool physics,bool blueprint){
            std::lock_guard<std::mutex> hold(lock);if(!paused||asset!="EditorCube"||physics||blueprint||scale.x_val<=0)throw std::runtime_error("Incorrect gate spawn contract");actors[name]=pose;return name;});
        server.async_run(4);std::cout<<"MOCK_PREPARE_READY\n"<<std::flush;
        std::string line;while(std::getline(std::cin,line)&&line!="quit"){}
        server.stop();std::cout<<"MOCK_PREPARE_STOP actors="<<actors.size()<<'\n';return 0;
    }catch(const std::exception& error){std::cerr<<"MOCK_PREPARE_FAILED "<<error.what()<<'\n';return 1;}
}
