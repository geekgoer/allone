// Native PS pilot for optional DPU corrections to the onboard AirSim loop.
#include "features/features.hpp"
#include "local_dpu.hpp"
#include "rpc/airsim_rpc.hpp"
#include <rpc/client.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <tuple>

namespace {
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
namespace f = psfeatures;
constexpr int slot_ms = 180;
constexpr int max_age_ms = 55;
constexpr double speed = 1.65;

f::Raw237 make_input(const f::State& state, double amplitude) {
    std::array<f::Reference, 21> refs;
    for (std::size_t i = 0; i < refs.size(); ++i) {
        const double x = std::min(12., (std::max(0., state.position[0])/speed + i*.1)*speed);
        const double v = x < 12. ? speed : 0.;
        const double y = amplitude*std::sin(.6*x);
        const double ay = -.36*v*v*y;
        const double half_roll = .5*std::atan2(ay, 9.8066);
        refs[i].position = {{x, y, -3.}};
        refs[i].velocity = {{v, .6*v*amplitude*std::cos(.6*x), 0.}};
        refs[i].acceleration = {{0., ay, 0.}};
        refs[i].orientation = {{std::cos(half_roll), std::sin(half_roll), 0., 0.}};
    }
    return f::encode(state, refs);
}

void require(bool ok, const char* reason) {
    if (!ok) throw std::runtime_error(reason);
}

Json verify_models(psdpu::LocalDpu& dpu, const std::string& package) {
    std::ifstream registry_file(package + "/dpu_model_registry.json");
    Json registry;
    registry_file >> registry;
    require(registry.at("wire_tag_hex") == "9253dd1fc87f8d7f", "wrong deployed 237D registry");
    Json checks = Json::object();
    for (const auto& model : registry.at("models")) {
        const int id = model.at("numeric_id");
        if (id != 5 && id != 11) continue;
        std::ifstream golden(package + "/" + model.at("golden_file").get<std::string>());
        double error = 0.;
        std::vector<f::Raw237> inputs;
        for (std::string line; std::getline(golden, line);) {
            if (line.empty() || line.front() == '#') continue;
            // The optional FP32 reference at the end of each line is separate.
            std::istringstream row(line);
            f::Raw237 input;
            std::array<float, 4> expected;
            for (auto& x : input) require(bool(row >> x), "missing golden input");
            for (auto& y : expected) require(bool(row >> y), "missing golden output");
            const auto result = dpu.infer(id, input);
            require(result.status == 0, "golden rejected");
            for (std::size_t j = 0; j < expected.size(); ++j)
                error = std::max(error, double(std::abs(expected[j]-result.output[j])));
            inputs.push_back(input);
        }
        require(inputs.size() == 32 && error <= .0001, "golden mismatch");
        for (std::size_t step = 0; step < 100; ++step)
            require(dpu.infer(id, inputs[step % inputs.size()]).status == 0, "warmup failed");
        checks[std::to_string(id)] = error;
    }
    require(checks.size() == 2, "both candidate models required");
    return checks;
}

struct Vehicle {
    std::string name;
    double amplitude, gate2_early, gate3_early, gate3_near;
    int last_slot = 0;
    std::uint64_t sample = 0;
    psairsim::State state;
};

// Critical opportunities are the historically measured upstream bands.
double window_end(const Vehicle& v) {
    for (const double center : {v.gate2_early, v.gate3_early, v.gate3_near})
        if (std::abs(v.state.position[0]-center) <= .25) return center+.25;
    return std::numeric_limits<double>::infinity();
}

double duration_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now()-start).count();
}

Json run(const Json& spec, const std::string& host, const std::string& package) {
    const std::string policy = spec.at("policy");
    const bool adaptive = policy == "adaptive";
    const bool probe = policy == "probe";
    require(adaptive || probe || policy == "fixed", "invalid policy");
    const int fixed_period = spec.at("fixed_period_slots");
    const int fixed_model = spec.at("fixed_model");
    const int cap = spec.at("request_budget");
    std::vector<Vehicle> fleet;
    for (const auto& row : spec.at("fleet")) {
        const auto& probes = spec.at("probes");
        fleet.push_back({row.at("vehicle"), row.at("amplitude"), probes.at("g2_early"),
                         probes.at("g3_early"), probes.at("g3_near"), 0, 0, {}});
    }
    require(!fleet.empty() && fleet.size() <= 6 && cap > 0 && cap <= 46 &&
            fixed_period >= 1 && (fixed_model == 5 || fixed_model == 11), "invalid experiment bounds");
    const int port = spec.at("port");
    const auto calibration = f::Calibration::from_json(spec.at("calibration"));
    psairsim::Client state_rpc(host, port, 100);
    rpc::client pilot_rpc(host, port);
    pilot_rpc.set_timeout(100);
    psdpu::LocalDpu dpu("main18_plain_mlp_h2_w128_seed0,main18_plain_mlp_h4_w1024_seed0");
    const auto checks = verify_models(dpu, package);
    std::cout << Json{{"ready", true}, {"goldens", checks}}.dump() << std::endl;
    std::string start;
    require(bool(std::getline(std::cin, start)) && start == "GO", "start signal missing");
    const auto origin = Clock::now() + std::chrono::milliseconds(450);
    Json requests = Json::array(), slots = Json::array();
    int attempts = 0;
    for (const int slot : spec.at("slots")) {
        const auto release = origin + std::chrono::milliseconds((slot-1)*slot_ms);
        std::this_thread::sleep_until(release);
        if (attempts == cap) break;
        const auto begin = Clock::now();
        Json decision = {{"slot",slot}, {"start_lateness_ms",std::chrono::duration<double,std::milli>(begin-release).count()}};
        try {
            Vehicle* selected = nullptr;
            auto best = std::make_tuple(std::numeric_limits<double>::infinity(), 0, std::string());
            Json candidates = Json::array();
            for (auto& v : fleet) {
                v.sample = pilot_rpc.call("falconTraceClock").as<std::uint64_t>();
                v.state = state_rpc.get_state(v.name);
                const double end = window_end(v);
                const bool critical = std::isfinite(end);
                const int period = adaptive ? (critical ? 1 : 4) : fixed_period;
                const bool eligible = (!probe || &v == &fleet.front()) &&
                    (!v.last_slot || slot-v.last_slot >= period) && v.state.position[0] < 9.95;
                candidates.push_back({{"vehicle",v.name}, {"x_m",v.state.position[0]}, {"T_ms",period*slot_ms},
                                      {"critical",critical}, {"eligible",eligible}});
                if (!eligible) continue;
                const auto key = std::make_tuple(adaptive ? (end-v.state.position[0])/std::max(.2,v.state.velocity[0])
                                                          : std::numeric_limits<double>::infinity(),
                                                 v.last_slot, v.name);
                if (!selected || key < best) { selected = &v; best = key; }
            }
            decision["candidates"] = candidates;
            const auto selected_at = Clock::now();
            decision["sample_and_select_ms"] = duration_ms(begin);
            if (selected) {
                auto& v = *selected;
                const bool critical = std::isfinite(window_end(v));
                const int model = adaptive ? (critical ? 11 : 5) : fixed_model;
                v.last_slot = slot;
                ++attempts;
                Json row = {{"slot",slot}, {"sample_ns",v.sample}, {"vehicle",v.name}, {"model_id",model},
                            {"critical",critical}, {"x_m",v.state.position[0]}, {"status","error"}, {"accepted",false}};
                try {
                    f::State state;
                    state.position=v.state.position; state.velocity=v.state.velocity;
                    state.quaternion=v.state.orientation_wxyz;
                    state.acceleration=v.state.acceleration; state.angular_velocity=v.state.angular_velocity;
                    auto input = make_input(state, v.amplitude);
                    row["encode_ms"] = duration_ms(selected_at);
                    const auto result = dpu.infer(model,input);
                    row["dpu_run_us"] = result.run_end_us-result.run_start_us;
                    row["board_service_us"] = result.end_us-result.start_us;
                    row["dpu_status"] = result.status;
                    if (result.status == 0 && !result.saturation) {
                        f::Control control;
                        std::copy(result.output.begin(), result.output.end(), control.begin());
                        const auto wire = f::native_rpc_command(f::prepare_live_command(control, calibration));
                        const auto submit_at = Clock::now();
                        // AirSim validates its own sample stamp at submission and adoption.
                        if (submit_at < release + std::chrono::milliseconds(max_age_ms)) {
                            row["accepted"] = pilot_rpc.call("falconPilotRates", v.name, std::uint64_t(slot), v.sample,
                                                             wire.roll_rate, wire.pitch_rate, wire.yaw_rate, wire.throttle).as<bool>();
                            row["status"] = row.at("accepted").get<bool>() ? "submitted" : "rejected_at_airsim";
                            row["submit_rpc_ms"] = duration_ms(submit_at);
                        } else {
                            row["status"] = "expired_before_submit";
                        }
                    } else {
                        row["status"] = "inference_rejected";
                    }
                } catch (const std::exception& e) { row["error"] = e.what(); }
                row["release_to_return_ms"] = duration_ms(release);
                requests.push_back(std::move(row));
            }
        } catch (const std::exception& e) { decision["error"] = e.what(); }
        decision["elapsed_ms"] = duration_ms(begin);
        slots.push_back(std::move(decision));
    }
    return {{"requests", requests}, {"slots", slots}};
}
}

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--encode") {
            Json input;
            std::cin >> input;
            auto state = f::State::from_json(input.at("state"));
            std::cout << Json(make_input(state, input.at("amplitude"))).dump() << '\n';
            return 0;
        }
        require(argc == 3, "usage: ps_onboard WINDOWS_HOST MODEL_PACKAGE");
        std::string line;
        require(bool(std::getline(std::cin, line)), "trial spec missing");
        std::cout << run(Json::parse(line), argv[1], argv[2]).dump() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
