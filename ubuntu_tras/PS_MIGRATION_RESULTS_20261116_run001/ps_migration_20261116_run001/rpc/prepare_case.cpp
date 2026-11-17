#include "airsim_rpc.hpp"
#include "features.hpp"
#include <json.hpp>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <sys/stat.h>
#include <thread>

using nlohmann::json;
using Vec3 = std::array<double,3>;
using Quat = std::array<double,4>;
namespace {
json read_json(const std::string& path) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("Cannot read JSON: " + path);
    json value; stream >> value; return value;
}
void save(const std::string& path,const json& value) {
    std::ofstream stream(path);
    if (!stream) throw std::runtime_error("Cannot write JSON: " + path);
    stream << value.dump(2) << '\n';
    if (!stream) throw std::runtime_error("JSON write failed: " + path);
}
template<std::size_t N> std::array<double,N> array(const json& row) {
    if (!row.is_array() || row.size()!=N) throw std::runtime_error("Array dimension mismatch");
    std::array<double,N> a;
    for(std::size_t i=0;i<N;++i){a[i]=row.at(i).get<double>();
        if(!std::isfinite(a[i])) throw std::runtime_error("Nonfinite geometry value");}
    return a;
}
std::vector<std::string> split(const std::string& value) {
    std::vector<std::string> result;std::set<std::string> unique;
    std::stringstream input(value);std::string name;
    while(std::getline(input,name,',')){
        if(name.empty() || !std::all_of(name.begin(),name.end(),[](char c){
            return (c>='A'&&c<='Z')||(c>='a'&&c<='z')||(c>='0'&&c<='9')||c=='_';
        }) || !unique.insert(name).second) throw std::runtime_error("Invalid/duplicate vehicle name");
        result.push_back(name);
    }
    if(result.empty())throw std::runtime_error("No selected vehicles");return result;
}
void event(json& setup,const std::string& stage) {
    setup["events"].push_back({{"stage",stage},{"ps_monotonic_ns",psairsim::monotonic_ns()}});
    setup["stage"]=stage;
}
json state_json(const psairsim::State& s) {
    return {{"position",s.position},{"velocity",s.velocity},
            {"acceleration",s.acceleration},{"orientation_wxyz",s.orientation_wxyz},
            {"quaternion",s.orientation_wxyz},{"angular_velocity",s.angular_velocity},
            {"timestamp_ns",s.timestamp_ns},{"landed_state",s.landed_state},
            {"has_collided",s.collision.has_collided}};
}
json build_geometry(const json& source,const std::vector<std::string>& vehicles,
                    const json& offsets,double reference_start,
                    const std::string& asset,double multiplier,
                    const std::string& prefix) {
    if(source.at("version")!="ip37_natural_turn_gates_v1" ||
       std::abs(source.at("gate_center_height_m").get<double>()-4.)>1e-12)
        throw std::runtime_error("Unsupported frozen case contract");
    const double duration=source.at("duration_s").get<double>();
    const double width=source.at("gate_center_width_m").get<double>();
    if(duration<=0 || width<=.15 || reference_start<0)
        throw std::runtime_error("Invalid duration/gate/reference start");
    std::vector<double> means;
    double spacing=12.;
    for(const auto& vehicle:vehicles){
        const auto& layout=source.at("layouts").at(vehicle);
        const auto& first=layout.at("segments").at(0);
        if(reference_start>=first.at("start_time").get<double>()+first.at("duration").get<double>())
            throw std::runtime_error("reference-start-s must precede first gate");
        const auto trajectory=psfeatures::Trajectory::from_layout(layout);
        double sum=0,min_y=std::numeric_limits<double>::infinity(),max_y=-min_y;
        for(int j=0;j<=200;++j){
            const auto ref=trajectory.evaluate(j*(duration/200.));
            const double y=ref.position[1];sum+=y;min_y=std::min(min_y,y);max_y=std::max(max_y,y);
        }
        means.push_back(sum/201.);spacing=std::max(spacing,max_y-min_y+10.);
    }
    json out={{"layouts",json::object()},{"translated_layouts",json::object()},
              {"offsets",offsets},{"shifts",json::object()},
              {"extents_mean_y",means},{"spacing_m",spacing},
              {"duration_s",duration},{"gate_center_width_m",width},
              {"gate_center_height_m",4.},{"gate_thickness_m",.15},
              {"reference_start_s",reference_start},
              {"initial_states",json::object()},{"gate_objects",json::array()},
              {"asset",asset},{"asset_scale_multiplier",multiplier}};
    for(std::size_t i=0;i<vehicles.size();++i){
        const auto& vehicle=vehicles[i];const auto offset=array<3>(offsets.at(vehicle));
        const Vec3 shift{{-offset[0],(double(i)-(double(vehicles.size())-1.)/2.)*spacing-means[i]-offset[1],-offset[2]}};
        auto layout=source.at("layouts").at(vehicle);
        for(auto& segment:layout.at("segments"))for(int axis=0;axis<3;++axis)
            segment["coefficients"][axis][0]=segment.at("coefficients").at(axis).at(0).get<double>()+shift[axis];
        for(auto& gate:layout.at("gates"))for(int axis=0;axis<3;++axis)
            gate["center"][axis]=gate.at("center").at(axis).get<double>()+shift[axis];
        out["layouts"][vehicle]=layout;out["translated_layouts"][vehicle]=layout;
        out["shifts"][vehicle]=shift;
        const auto trajectory=psfeatures::Trajectory::from_layout(layout);
        const auto first=trajectory.evaluate(reference_start);
        psairsim::State state;state.position=first.position;
        state.orientation_wxyz=first.orientation;
        if(reference_start>0)state.velocity=first.velocity;
        out["initial_states"][vehicle]=state_json(state);
        for(const auto& gate:layout.at("gates")){
            const int index=gate.at("index").get<int>();
            const auto center=array<3>(gate.at("center"));
            const auto q=psfeatures::normalize_quaternion(array<4>(gate.at("orientation")));
            const std::array<const char*,4> labels{{"left","right","top","bottom"}};
            const std::array<Vec3,4> deltas{{Vec3{{0,-width/2,0}},Vec3{{0,width/2,0}},Vec3{{0,0,-2}},Vec3{{0,0,2}}}};
            const std::array<Vec3,4> sizes{{Vec3{{.15,.15,4}},Vec3{{.15,.15,4}},Vec3{{.15,width,.15}},Vec3{{.15,width,.15}}}};
            for(int part=0;part<4;++part){
                auto delta=psfeatures::rotate_vector(q,deltas[part]);Vec3 position,scale;
                for(int axis=0;axis<3;++axis){position[axis]=center[axis]+offset[axis]+delta[axis];scale[axis]=sizes[part][axis]*multiplier;}
                out["gate_objects"].push_back({{"name",prefix+vehicle+"_"+std::to_string(index)+"_"+labels[part]},
                    {"vehicle",vehicle},{"gate_index",index},{"part",labels[part]},
                    {"position",position},{"orientation_wxyz",q},{"scale",scale},
                    {"physics_enabled",false},{"is_blueprint",false}});
            }
        }
    }
    return out;
}
}

int main(int argc,char** argv){
    const std::string bundle="/home/cwc/dpu_work/windows_framework/route_b_ps_handoff_20261116";
    std::string host,vehicle_names="Drone1",out,offset_file;
    std::string case_path=bundle+"/project/controller_architecture_study/results/release001_live_case_20261116/case.json";
    std::string calibration_path=bundle+"/project/configs/thrust_calibration.json";
    std::string prefix="PSMIG20261116_";
    int port=41451,timeout_ms=3000;double reference_start=1.5;bool offline=false,created=false,touched=false;
    json setup={{"schema","ps-prepared-case/v1"},{"events",json::array()},
                {"created_actors",json::array()},{"status","STARTING"}};
    std::unique_ptr<psairsim::Client> client;
    try{
        for(int i=1;i<argc;++i){const std::string key=argv[i];
            if(key=="--offline"){offline=true;continue;}
            if(i+1==argc)throw std::runtime_error("Missing argument value: "+key);
            const std::string value=argv[++i];
            if(key=="--host")host=value;else if(key=="--vehicles")vehicle_names=value;
            else if(key=="--out")out=value;else if(key=="--case")case_path=value;
            else if(key=="--calibration")calibration_path=value;
            else if(key=="--port")port=std::stoi(value);else if(key=="--timeout-ms")timeout_ms=std::stoi(value);
            else if(key=="--reference-start-s")reference_start=std::stod(value);
            else if(key=="--offsets")offset_file=value;else if(key=="--prefix")prefix=value;
            else throw std::runtime_error("Unknown option: "+key);
        }
        const std::string required_prefix="PSMIG20261116_";
        const bool valid_prefix=prefix.compare(0,required_prefix.size(),required_prefix)==0 &&
            std::all_of(prefix.begin(),prefix.end(),[](char c){
                return (c>='A'&&c<='Z')||(c>='a'&&c<='z')||(c>='0'&&c<='9')||c=='_';
            });
        if(out.empty()||(!offline&&host.empty())||port<=0||port>65535||timeout_ms<=0||
           !std::isfinite(reference_start)||!valid_prefix)
            throw std::runtime_error("Require --host IP --out NEWDIR --vehicles Drone1[,Drone2] [--case FILE --calibration FILE --reference-start-s 1.5]");
        if(::mkdir(out.c_str(),0755)!=0)throw std::runtime_error("Output must be a NEW directory with existing parent: "+out+" errno="+std::to_string(errno));
        created=true;event(setup,"read_frozen_inputs");
        const auto source=read_json(case_path),calibration_json=read_json(calibration_path);
        const auto calibration=psfeatures::Calibration::from_json(calibration_json);
        const auto vehicles=split(vehicle_names);
        setup["host"]=host;setup["port"]=port;setup["vehicles"]=vehicles;
        setup["case_source_path"]=case_path;setup["calibration_source_path"]=calibration_path;
        setup["original_case"]=source;setup["calibration"]=calibration_json;
        setup["scene_prefix"]=prefix;setup["simulation_touched"]=false;
        setup["reference_start_s"]=reference_start;
        setup["initial_state_scope"]=reference_start>0?"reference-matched flying diagnostic; NOT stationary takeoff acceptance":"route start at rest";
        save(out+"/setup.json",setup);
        // Validate complete geometry and reference-start BEFORE any scene mutation.
        json zero_offsets=json::object();for(const auto& vehicle:vehicles)zero_offsets[vehicle]=Vec3{{0,0,0}};
        build_geometry(source,vehicles,zero_offsets,reference_start,"1M_Cube",1.,prefix);
        json offsets=json::object();std::string asset="1M_Cube";double multiplier=1.;
        if(offline){
            if(offset_file.empty())throw std::runtime_error("--offline requires --offsets FILE");
            offsets=read_json(offset_file);
            event(setup,"offline_geometry_only");
        }else{
            client.reset(new psairsim::Client(host,static_cast<std::uint16_t>(port),timeout_ms));
            if(!client->ping())throw std::runtime_error("AirSim ping returned false");
            const auto available=client->list_vehicles();
            for(const auto& vehicle:vehicles)if(std::find(available.begin(),available.end(),vehicle)==available.end())throw std::runtime_error("Vehicle missing in Play: "+vehicle);
            const auto foreign=client->list_scene_objects("Falcon.*Gate.*");
            setup["foreign_generated_gates"]=foreign;
            save(out+"/setup.json",setup);
            if(!foreign.empty())throw std::runtime_error("Foreign Falcon.*Gate.* actors exist; not deleting them. Restart Play or remove via their owner.");
            const auto own=client->list_scene_objects(prefix+".*");
            setup["previous_owned_actors"]=own;setup["cleanup_results"]=json::object();
            if(!own.empty()){touched=true;setup["simulation_touched"]=true;save(out+"/setup.json",setup);}
            for(const auto& name:own){setup["cleanup_results"][name]=client->destroy_object(name,prefix);save(out+"/setup.json",setup);}
            std::vector<std::string> remaining;
            for(int retry=0;retry<10;++retry){remaining=client->list_scene_objects(prefix+".*");if(remaining.empty())break;std::this_thread::sleep_for(std::chrono::milliseconds(20));}
            setup["cleanup_remaining"]=remaining;
            if(!remaining.empty())throw std::runtime_error("Owned actor deletion not confirmed; scene not reset");
            event(setup,"reset_and_pause");touched=true;setup["simulation_touched"]=true;
            save(out+"/setup.json",setup);client->reset();client->pause(true);
            setup["physics_paused"]=client->is_paused();
            if(!setup.at("physics_paused").get<bool>())throw std::runtime_error("Physics pause not confirmed");
            for(const auto& vehicle:vehicles){
                const auto local=client->get_vehicle_pose(vehicle),world=client->get_object_pose(vehicle);Vec3 offset;
                for(int axis=0;axis<3;++axis)offset[axis]=world.position[axis]-local.position[axis];
                offsets[vehicle]=offset;
            }
            const auto assets=client->list_assets();setup["available_assets"]=assets;
            if(std::find(assets.begin(),assets.end(),asset)==assets.end()){
                if(std::find(assets.begin(),assets.end(),"EditorCube")==assets.end())throw std::runtime_error("No registered 1M_Cube/EditorCube gate mesh");
                asset="EditorCube";multiplier=.5;
            }
        }
        const auto geometry=build_geometry(source,vehicles,offsets,reference_start,asset,multiplier,prefix);
        for(auto item=geometry.begin();item!=geometry.end();++item)setup[item.key()]=item.value();
        const psfeatures::Control hover{{calibration.hover_thrust_accel,0,0,0}};
        const auto prime=psfeatures::prepare_live_command(hover,calibration,1.);
        // Native RPC receives physical wx/wy/wz; the Python-client sign flips
        // must not be copied here (the hover values happen to be all zero).
        const auto native_prime=psfeatures::native_rpc_command(prime);
        setup["effective_control"]=prime.effective;
        setup["prime_command"]={{"roll_rate",native_prime.roll_rate},{"pitch_rate",native_prime.pitch_rate},
            {"yaw_rate",native_prime.yaw_rate},{"throttle",native_prime.throttle},{"duration",native_prime.duration}};
        setup["initialization_ignore_collision"]=true;
        setup["priming_scope"]="Direct calibrated physical hover+zero rates while paused; no wait for duration; ping is RPC-return proxy only";
        save(out+"/setup.json",setup);
        if(!offline){
            event(setup,"initialize_reference_matched_states");
            for(const auto& vehicle:vehicles){
                client->enable_api(vehicle,true);if(!client->arm(vehicle,true))throw std::runtime_error("Arm failed: "+vehicle);
                const auto& initial=setup.at("initial_states").at(vehicle);psairsim::State state;
                state.position=array<3>(initial.at("position"));state.velocity=array<3>(initial.at("velocity"));state.orientation_wxyz=array<4>(initial.at("orientation_wxyz"));
                client->set_kinematics(vehicle,state,true);
            }
            event(setup,"spawn_physical_gates");save(out+"/setup.json",setup);
            for(auto& actor:setup.at("gate_objects")){
                psairsim::Pose pose;pose.position=array<3>(actor.at("position"));pose.orientation_wxyz=array<4>(actor.at("orientation_wxyz"));
                const auto actual=client->spawn_object(actor.at("name").get<std::string>(),asset,pose,array<3>(actor.at("scale")),false,false);
                if(actual.empty())throw std::runtime_error("Spawn returned empty actor name");
                if(actual.compare(0,prefix.size(),prefix)!=0)throw std::runtime_error("Spawn returned unowned actor name: "+actual);
                actor["actual_name"]=actual;setup["created_actors"].push_back(actual);save(out+"/setup.json",setup);
            }
            event(setup,"prime_physical_hover_while_paused");setup["initial_readbacks"]=json::object();setup["prime_proxies"]=json::object();
            for(const auto& vehicle:vehicles){
                const auto readback=client->get_state(vehicle);setup["initial_readbacks"][vehicle]=state_json(readback);
                const auto& initial=setup.at("initial_states").at(vehicle);const auto p=array<3>(initial.at("position")),v=array<3>(initial.at("velocity"));const auto q=array<4>(initial.at("orientation_wxyz"));
                double pe=0,ve=0,qd=0;for(int j=0;j<3;++j){pe+=std::pow(readback.position[j]-p[j],2);ve+=std::pow(readback.velocity[j]-v[j],2);}for(int j=0;j<4;++j)qd+=readback.orientation_wxyz[j]*q[j];
                if(std::sqrt(pe)>.05||std::sqrt(ve)>.2||std::abs(qd)<.999)throw std::runtime_error("Paused initial state mismatch: "+vehicle);
                const auto proxy=client->move_rates_throttle(vehicle,float(native_prime.roll_rate),float(native_prime.pitch_rate),float(native_prime.yaw_rate),float(native_prime.throttle),float(native_prime.duration));
                setup["prime_proxies"][vehicle]={{"submit_ns",proxy.submit_ns},{"enqueued_ns",proxy.enqueued_ns},{"ping_start_ns",proxy.ping_start_ns},{"ping_return_ns",proxy.ping_return_ns},{"ping_ok",proxy.ping_ok}};
                if(!proxy.ping_ok)throw std::runtime_error("Prime ping failed: "+vehicle);
                save(out+"/setup.json",setup);
            }
            setup["physics_paused"]=client->is_paused();if(!setup.at("physics_paused").get<bool>())throw std::runtime_error("Physics unexpectedly running at preparation end");
        }
        event(setup,"finished");setup["status"]=offline?"OFFLINE_LAYOUT_ONLY":"PREPARED_PAUSED";
        save(out+"/setup.json",setup);if(!offline)save(out+"/prepared_case.json",setup);
        std::cout<<setup.at("status").get<std::string>()<<" vehicles="<<vehicles.size()<<" gate_objects="<<setup.at("gate_objects").size()<<" out="<<out<<'\n';
        return 0;
    }catch(const std::exception& error){
        setup["status"]="FAILED";setup["error"]=error.what();
        setup["failed_monotonic_ns"]=psairsim::monotonic_ns();
        if(touched&&client){try{client->pause(true);setup["physics_paused_after_error"]=client->is_paused();}catch(const std::exception& e){setup["pause_after_error_failed"]=e.what();}}
        if(created){try{save(out+"/setup.json",setup);}catch(...){} }
        std::cerr<<"PREPARE_CASE_FAILED "<<error.what()<<'\n';return 1;
    }
}
