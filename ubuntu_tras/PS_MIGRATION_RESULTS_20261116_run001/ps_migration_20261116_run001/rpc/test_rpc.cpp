#include "airsim_rpc.hpp"
#include <rpc/server.h>
#include <rpc/msgpack.hpp>
#include <array>
#include <atomic>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <thread>

// Local protocol-only mock. Not AirSim flight or PS execution evidence.
namespace mock {
struct Vec { float x_val=0,y_val=0,z_val=0; MSGPACK_DEFINE_MAP(x_val,y_val,z_val); };
struct Quat { float w_val=1,x_val=0,y_val=0,z_val=0; MSGPACK_DEFINE_MAP(w_val,x_val,y_val,z_val); };
struct Pose { Vec position; Quat orientation; MSGPACK_DEFINE_MAP(position,orientation); };
struct Kin {
    Vec position,linear_velocity,angular_velocity,linear_acceleration,angular_acceleration;
    Quat orientation;
    MSGPACK_DEFINE_MAP(position,orientation,linear_velocity,angular_velocity,linear_acceleration,angular_acceleration);
};
struct Collision {
    bool has_collided=false; float penetration_depth=0;
    std::uint64_t time_stamp=0; std::string object_name; int object_id=-1;
    MSGPACK_DEFINE_MAP(has_collided,penetration_depth,time_stamp,object_name,object_id);
};
struct State {
    Kin kinematics_estimated; Collision collision;
    std::uint64_t timestamp=1234567890123ULL; int landed_state=1;
    // Real replies contain additional maps. Omitted fields must be harmless.
    std::string extra_ignored="ignored";
    MSGPACK_DEFINE_MAP(kinematics_estimated,collision,timestamp,landed_state,extra_ignored);
};
void check(bool good,const char* what) { if(!good) throw std::runtime_error(what); }
}
int main() {
    try {
        const std::uint16_t port=43463;
        rpc::server server("127.0.0.1",port);
        mock::State state; state.kinematics_estimated.position.x_val=1.25f;
        state.kinematics_estimated.linear_velocity.z_val=-2.f;
        state.collision.has_collided=true; state.collision.object_name="mock_gate";
        std::atomic<bool> paused{false},api{false},armed{false},command_seen{false};
        std::atomic<int> completed{0};
        std::array<float,5> command{{0,0,0,0,0}};
        server.bind("ping",[](){return true;});
        server.bind("listVehicles",[](){return std::vector<std::string>{"Drone1","Drone2"};});
        server.bind("getMultirotorState",[&](const std::string& v){mock::check(v=="Drone1","vehicle wire argument");return state;});
        server.bind("simGetCollisionInfo",[&](const std::string&){return state.collision;});
        server.bind("moveByAngleRatesThrottle",[&](float a,float b,float c,float d,float e,const std::string& v){
            mock::check(v=="Drone1","command vehicle");command={{a,b,c,d,e}};command_seen=true;
            std::this_thread::sleep_for(std::chrono::milliseconds(200));completed++;return true;});
        server.bind("simPause",[&](bool p){paused=p;});
        server.bind("simIsPaused",[&](){return paused.load();});
        server.bind("enableApiControl",[&](bool e,const std::string&){api=e;});
        server.bind("armDisarm",[&](bool a,const std::string&){armed=a;return true;});
        server.bind("simSetKinematics",[&](mock::Kin k,bool,const std::string&){state.kinematics_estimated=k;});
        server.bind("simGetGroundTruthKinematics",[&](const std::string&){return state.kinematics_estimated;});
        server.bind("simGetObjectPose",[&](const std::string& name){
            mock::check(name=="Drone1","object pose argument");
            mock::Pose p;p.position.x_val=10.f;return p;});
        server.bind("simListSceneObjects",[](const std::string& regex){
            mock::check(regex=="PSMIG_.*","scene regex argument");
            return std::vector<std::string>{"PSMIG_mock_gate"};});
        server.bind("simSpawnObject",[](const std::string& name,const std::string& asset,
                                         mock::Pose p,mock::Vec scale,bool physics,bool blueprint){
            mock::check(name=="PSMIG_mock_gate"&&asset=="EditorCube"&&p.position.z_val==-2&&
                        scale.x_val==0.5f&&!physics&&!blueprint,"spawn argument contract");return name;});
        server.bind("simDestroyObject",[](const std::string& name){
            mock::check(name=="PSMIG_mock_gate","destroy owned prefix");return true;});
        server.async_run(3);
        psairsim::Client client("127.0.0.1",port,1000);
        mock::check(client.ping(),"ping");
        mock::check(client.list_vehicles().size()==2,"list vehicles");
        auto s=client.get_state("Drone1");
        mock::check(s.position[0]==1.25&&s.velocity[2]==-2,"NED state decode");
        mock::check(s.orientation_wxyz[0]==1&&s.timestamp_ns==1234567890123ULL,"quaternion/time decode");
        auto collision=client.get_collision("Drone1");
        mock::check(collision.has_collided&&collision.object_name=="mock_gate","collision decode");
        client.pause(true); mock::check(client.is_paused(),"pause");
        client.enable_api("Drone1",true);client.arm("Drone1",true);
        mock::check(api&&armed,"enable/arm argument order");
        s.position={{4,5,-6}};s.orientation_wxyz={{0,0,0,1}};
        client.set_kinematics("Drone1",s,false);
        auto truth=client.get_ground_truth_kinematics("Drone1");
        mock::check(truth.position[2]==-6&&truth.orientation_wxyz[3]==1,"kinematics map");
        mock::check(client.get_object_pose("Drone1").position[0]==10,"object pose decode");
        mock::check(client.list_scene_objects("PSMIG_.*").size()==1,"scene list");
        psairsim::Pose scene_pose;scene_pose.position={{0,1,-2}};
        mock::check(client.spawn_object("PSMIG_mock_gate","EditorCube",scene_pose,
                                       {{0.5,1,2}},false)=="PSMIG_mock_gate","spawn response");
        mock::check(client.destroy_object("PSMIG_mock_gate"),"destroy response");
        bool foreign_rejected=false;
        try { client.destroy_object("UserGate"); }
        catch(const std::invalid_argument&) { foreign_rejected=true; }
        mock::check(foreign_rejected,"foreign scene object must not be destroyed");
        const auto p=client.move_rates_throttle("Drone1",0.1f,-0.2f,0.3f,0.5f,0.2f);
        mock::check(p.ping_ok&&p.submit_ns<=p.enqueued_ns&&p.enqueued_ns<=p.ping_start_ns&&p.ping_start_ns<=p.ping_return_ns,"proxy monotonic ordering");
        // A ping return is NOT completion: deliberately delayed command remains active.
        mock::check(completed==0,"command proxy must not join move duration");
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        mock::check(command_seen&&command[1]==-0.2f&&completed==1,"rates passed without hidden signs");
        bool rejected=false;
        try{client.move_rates_throttle("Drone1",0,0,0,2,0.2f);}catch(const std::invalid_argument&){rejected=true;}
        mock::check(rejected,"out-of-range throttle not silently clipped");
        server.stop();
        std::cout<<"RPC_MOCK_PASS readonly,state,collision,management,command_wire,nonblocking_proxy,guards\n";
        return 0;
    }catch(const std::exception& error){std::cerr<<"RPC_MOCK_FAILED "<<error.what()<<'\n';return 1;}
}
