#include "airsim_rpc.hpp"
#include <json.hpp>
#include <fstream>
#include <iostream>
#include <stdexcept>

using nlohmann::json;
template<std::size_t N> std::array<double, N> json_array(const json& value) {
    if (!value.is_array() || value.size() != N)
        throw std::invalid_argument("State JSON array length mismatch");
    std::array<double, N> out;
    for (std::size_t i=0;i<N;++i) out[i]=value.at(i).get<double>();
    return out;
}
json state_json(const psairsim::State& s) {
    return json{{"position", s.position}, {"velocity", s.velocity},
                {"acceleration", s.acceleration},
                {"orientation_wxyz", s.orientation_wxyz},
                {"angular_velocity", s.angular_velocity},
                {"angular_acceleration", s.angular_acceleration},
                {"timestamp_ns", s.timestamp_ns},
                {"landed_state", s.landed_state},
                {"has_collided", s.collision.has_collided}};
}
int main(int argc, char** argv) {
    json log;
    try {
        std::string host, vehicle = "Drone1", mode = "probe", input;
        int port = 41451, timeout_ms = 1000;
        bool allow_management = false;
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            if (a == "--allow-management") { allow_management = true; continue; }
            if (i + 1 == argc) throw std::invalid_argument("Missing argument value");
            const std::string value = argv[++i];
            if (a == "--host") host = value;
            else if (a == "--port") port = std::stoi(value);
            else if (a == "--timeout-ms") timeout_ms = std::stoi(value);
            else if (a == "--vehicle") vehicle = value;
            else if (a == "--mode") mode = value;
            else if (a == "--input") input = value;
            else throw std::invalid_argument("Unknown argument: " + a);
        }
        if (host.empty() || port < 1 || port > 65535 || timeout_ms <= 0)
            throw std::invalid_argument("Require --host ADDRESS [--port 41451 --timeout-ms 1000]");
        log = {{"host", host}, {"port", port}, {"vehicle", vehicle},
               {"mode", mode}, {"clock", "PS_CLOCK_MONOTONIC"},
               {"source", "native rpclib 2.3.0 frozen AirSim wire adaptors"}};
        const auto start = psairsim::monotonic_ns();
        psairsim::Client client(host, static_cast<std::uint16_t>(port), timeout_ms);
        if (mode == "probe" || mode == "state") {
            log["ping"] = client.ping();
            log["vehicles"] = client.list_vehicles();
            log["is_paused"] = client.is_paused();
            log["state_start_ns"] = psairsim::monotonic_ns();
            log["state"] = state_json(client.get_state(vehicle));
            log["state_return_ns"] = psairsim::monotonic_ns();
            const auto collision = client.get_collision(vehicle);
            log["collision"] = {{"has_collided", collision.has_collided},
                {"timestamp_ns", collision.timestamp_ns},
                {"object_name", collision.object_name},
                {"object_id", collision.object_id},
                {"penetration_depth", collision.penetration_depth}};
            log["collision_return_ns"] = psairsim::monotonic_ns();
        } else {
            if (!allow_management)
                throw std::invalid_argument("Management requires --allow-management");
            if (mode == "pause") client.pause(true);
            else if (mode == "unpause") client.pause(false);
            else if (mode == "enable") { client.enable_api(vehicle, true); log["armed"] = client.arm(vehicle, true); }
            else if (mode == "disable") { log["disarmed"] = client.arm(vehicle, false); client.enable_api(vehicle, false); }
            else if (mode == "set-kinematics") {
                if (input.empty()) throw std::invalid_argument("Require --input STATE.json");
                std::ifstream stream(input);
                if (!stream) throw std::runtime_error("Cannot open state JSON");
                json in; stream >> in;
                psairsim::State s;
                s.position = json_array<3>(in.at("position"));
                s.velocity = json_array<3>(in.at("velocity"));
                s.orientation_wxyz = json_array<4>(in.at("orientation_wxyz"));
                if (in.count("acceleration")) s.acceleration = json_array<3>(in.at("acceleration"));
                if (in.count("angular_velocity")) s.angular_velocity = json_array<3>(in.at("angular_velocity"));
                client.set_kinematics(vehicle, s, false);
                log["readback"] = state_json(client.get_ground_truth_kinematics(vehicle));
            } else throw std::invalid_argument("Unknown mode: " + mode);
        }
        log["started_ns"] = start;
        log["finished_ns"] = psairsim::monotonic_ns();
        log["status"] = "PASS";
        std::cout << log.dump(2) << '\n';
        return 0;
    } catch (const std::exception& error) {
        log["status"] = "FAILED"; log["error"] = error.what();
        log["finished_ns"] = psairsim::monotonic_ns();
        std::cout << log.dump(2) << '\n';
        return 1;
    }
}
