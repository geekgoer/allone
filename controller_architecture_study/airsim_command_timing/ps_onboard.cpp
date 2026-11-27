// Native PS pilot for optional DPU corrections to the onboard AirSim loop.
#include "features/features.hpp"
#include "local_dpu.hpp"
#include "rpc/airsim_rpc.hpp"
#include <rpc/client.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <mutex>
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

Json verify_models(psdpu::LocalDpu& dpu, const std::string& package, std::vector<f::Raw237>& background_inputs) {
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
        if (id == 11) background_inputs = std::move(inputs);
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

auto clock_us(Clock::time_point time) {
    return std::chrono::duration_cast<std::chrono::microseconds>(time.time_since_epoch()).count();
}

Json run_background(psdpu::LocalDpu& dpu, std::mutex& device, const std::vector<f::Raw237>& inputs,
                    Clock::time_point origin, int period_us) {
    using Us = std::chrono::microseconds;
    const int horizon_us = 46*slot_ms*1000;
    Json jobs = Json::array();
    for (int offset = 0, index = 0; offset < horizon_us; offset += period_us, ++index) {
        const auto release = origin + Us(offset);
        std::this_thread::sleep_until(release);
        const auto requested = Clock::now();
        std::unique_lock<std::mutex> lock(device);
        const auto started = Clock::now();
        const auto result = dpu.infer(11, inputs[index % inputs.size()]);
        lock.unlock();
        const auto to_us = [](Clock::duration d) { return std::chrono::duration_cast<Us>(d).count(); };
        jobs.push_back({offset, to_us(requested-origin), to_us(started-requested),
                        result.run_start_us, result.run_end_us, result.status});
    }
    return {{"model_id",11}, {"period_us",period_us}, {"origin_us",clock_us(origin)}, {"horizon_us",horizon_us},
            {"fields",{"release_offset_us","requested_offset_us","wait_us","run_start_us","run_end_us","status"}},
            {"jobs",std::move(jobs)}};
}

Json state_record(const Vehicle& v) {
    return {{"position",v.state.position}, {"velocity",v.state.velocity},
            {"quaternion",v.state.orientation_wxyz}, {"sample_ns",v.sample}};
}

double projected_error(const Vehicle& v, double gate_x) {
    return std::abs(v.state.position[1] + .25*v.state.velocity[1] - v.amplitude*std::sin(.6*gate_x));
}

void append_background(psdpu::LocalDpu& dpu, const std::vector<f::Raw237>& inputs, int count, Json& intervals,
                       int offset = 0) {
    for (int i = offset; i < offset + count; ++i) {
        const auto r = dpu.infer(11, inputs[i % inputs.size()]);
        require(r.status == 0 && r.run_end_us > r.run_start_us, "background inference failed");
        intervals.push_back({r.run_start_us, r.run_end_us});
    }
}

constexpr double deadline_budget_ms = 30.;

Json deadline_update(const Vehicle& v, int slot, rpc::client& pilot_rpc, psdpu::LocalDpu& dpu,
                     const f::Calibration& calibration, Clock::time_point start, Clock::time_point window_start,
                     double reserve_ms, bool probe_late) {
    const double before_ms = duration_ms(start);
    const double slack = deadline_budget_ms-duration_ms(window_start);
    Json row = {{"vehicle",v.name}, {"slot",slot}, {"sample_ns",v.sample}, {"before_ms",before_ms},
                {"slack_ms",slack}, {"accepted",false}};
    if (!probe_late && reserve_ms > slack) {
        row["status"] = "insufficient_time";
        return row;
    }
    f::State state;
    state.position=v.state.position; state.velocity=v.state.velocity; state.quaternion=v.state.orientation_wxyz;
    state.acceleration=v.state.acceleration; state.angular_velocity=v.state.angular_velocity;
    const auto input = make_input(state, v.amplitude);
    row["encode_ms"] = duration_ms(start)-before_ms;
    const auto r = dpu.infer(11, input);
    row["dpu_run_us"] = r.run_end_us-r.run_start_us;
    row["dpu_interval_us"] = {r.run_start_us, r.run_end_us};
    row["status"] = "dpu_rejected";
    if (r.status == 0 && !r.saturation) {
        f::Control control;
        std::copy(r.output.begin(), r.output.end(), control.begin());
        const auto wire = f::native_rpc_command(f::prepare_live_command(control, calibration));
        row["submit_ms"] = duration_ms(start);
        if (!probe_late && duration_ms(window_start) >= deadline_budget_ms) {
            row["status"] = "expired_before_submit";
        } else {
            const bool accepted = pilot_rpc.call("falconPilotRates", v.name, std::uint64_t(slot), v.sample,
                wire.roll_rate, wire.pitch_rate, wire.yaw_rate, wire.throttle).as<bool>();
            row["accepted"] = accepted;
            row["status"] = accepted ? "submitted" : "rejected";
        }
    }
    row["return_ms"] = duration_ms(start);
    return row;
}

Json run_deadline_pair(const Json& spec, std::vector<Vehicle>& fleet, psairsim::Client& state_rpc,
                       rpc::client& pilot_rpc, psdpu::LocalDpu& dpu, const std::vector<f::Raw237>& inputs,
                       const f::Calibration& calibration) {
    const auto policy = spec.at("pair_policy").get<std::string>();
    const auto fixed = spec.at("pair_static").get<std::string>();
    const auto gate_x = spec.at("gate_x_m").get<double>();
    const int calls = spec.at("pair_background_calls");
    const double reserve_ms = spec.at("pair_reserve_ms");
    const bool overlap = spec.value("pair_overlap_sample", false);
    const int burst = spec.value("pair_recovery_commands", 1);
    require(fleet.size() == 2 && calls >= 0 && calls <= 25 && reserve_ms >= 0. && reserve_ms < deadline_budget_ms,
            "invalid deadline-pair config");
    require(policy == "local" || policy == "static" || policy == "online" || policy == "timing" ||
            policy == "recovery",
            "invalid deadline-pair policy");
    require(burst >= 1 && burst <= 3 && (policy == "recovery" || burst == 1), "invalid recovery burst");
    Json states = Json::object(), scores = Json::object(), background = Json::array(), requests = Json::array();
    std::array<double, 2> risks{};
    const auto start = Clock::now();
    double background_start_ms = 0., background_end_ms = 0.;
    auto batch = [&] {
        background_start_ms = duration_ms(start);
        append_background(dpu, inputs, calls, background);
        background_end_ms = duration_ms(start);
    };
    std::future<void> pending;
    for (std::size_t i = 0; i < fleet.size(); ++i) {
        auto& v = fleet[i];
        v.sample = pilot_rpc.call("falconTraceClock").as<std::uint64_t>();
        if (i == 0 && overlap) pending = std::async(std::launch::async, batch);
        v.state = state_rpc.get_state(v.name);
        states[v.name] = state_record(v);
        risks[i] = projected_error(v, gate_x);
        scores[v.name] = risks[i];
    }
    const auto sample_done_ms = duration_ms(start);
    // Finish the entire background batch before the foreground uses this DPU.
    if (pending.valid()) pending.get();
    else batch();
    const auto ready_ms = duration_ms(start);
    const bool second_first = policy == "online" ? risks[1] > risks[0] : fleet[1].name == fixed;
    const bool recovery = policy == "recovery";
    const std::size_t count = recovery ? burst : fleet.size();
    for (std::size_t i = 0; i < count && policy != "local"; ++i) {
        auto& v = fleet[recovery ? (second_first ? 1 : 0) : (second_first ? 1-i : i)];
        auto window_start = start;
        if (recovery && i > 0) {
            std::this_thread::sleep_until(start + std::chrono::milliseconds(75*i));
            window_start = Clock::now();
            v.sample = pilot_rpc.call("falconTraceClock").as<std::uint64_t>();
            v.state = state_rpc.get_state(v.name);
        }
        auto row = deadline_update(v, i+1, pilot_rpc, dpu, calibration, start, window_start, reserve_ms, policy == "timing");
        const bool exhausted = row.at("status") == "insufficient_time";
        requests.push_back(std::move(row));
        if (exhausted) break;
    }
    const auto start_ns = states.at(fleet.front().name).at("sample_ns");
    return {{"states",std::move(states)}, {"risk",std::move(scores)}, {"background",std::move(background)},
            {"requests",std::move(requests)}, {"background_done_ms",ready_ms}, {"budget_ms",deadline_budget_ms},
            {"sample_done_ms",sample_done_ms}, {"background_start_ms",background_start_ms},
            {"background_end_ms",background_end_ms}, {"overlap_sample",overlap},
            {"reserve_ms",reserve_ms}, {"start_ns",start_ns}};
}

Json run_recovery_schedule(const Json& settings, std::vector<Vehicle>& fleet, psairsim::Client& state_rpc,
                           rpc::client& pilot_rpc, psdpu::LocalDpu& dpu, const std::vector<f::Raw237>& inputs,
                           const f::Calibration& calibration) {
    const int windows = settings.at("windows");
    const int calls = settings.at("background_calls");
    const double gate_x = settings.at("gate_x_m");
    const double reserve_ms = settings.at("reserve_ms");
    const std::string policy = settings.at("policy");
    const auto healthy = settings.at("healthy_error_m");
    const auto recipients = settings.value("fixed_recipients", std::vector<std::string>{});
    const int spacing_ms = settings.value("spacing_ms", 75);
    const auto offsets = settings.value("window_offsets_ms", std::vector<int>{});
    const bool shared_windows = !offsets.empty();
    const bool yield = policy == "yield";
    const bool deadline = policy == "deadline";
    const int second_batch = settings.value("second_background_calls", calls);
    const bool protect = settings.value("protect_followup", false);
    require(fleet.size() == 2 && windows >= 2 && windows <= 3 && calls == 17 &&
            reserve_ms >= 0. && reserve_ms < deadline_budget_ms, "invalid schedule settings");
    require(policy == "observe" || policy == "queue" || policy == "risk" || policy == "reserve" ||
            policy == "fixed" || yield || deadline,
            "invalid schedule policy");
    if (policy == "fixed") {
        require(recipients.size() == std::size_t(windows), "wrong number of fixed windows");
        require(spacing_ms >= 75 && spacing_ms <= 800, "invalid fixed window spacing");
        for (const auto& recipient : recipients)
            require(recipient.empty() || recipient == fleet[0].name || recipient == fleet[1].name,
                    "unknown fixed recipient");
    }
    require(policy == "fixed" || spacing_ms == 75, "custom spacing requires fixed dispatch");
    if (shared_windows)
        require(windows == 3 && offsets == std::vector<int>{0, 75, 300} &&
                fleet[0].name == "Drone1" && fleet[1].name == "Drone2" &&
                (policy == "fixed" || policy == "risk" || yield || deadline), "invalid shared windows");
    require(!(yield || deadline) || shared_windows, "policy requires shared windows");
    if (second_batch != calls || protect)
        require(policy == "fixed" && windows == 2 && spacing_ms == 75 && second_batch == 19 &&
                recipients == std::vector<std::string>{"Drone2", "Drone2"} && fleet[1].name == "Drone2",
                "invalid follow-up capacity test");
    const double peer_trigger_m = (yield || deadline) ? settings.at("peer_trigger_m").get<double>() : 0.;

    std::array<int, 2> last_sent{{-1, -1}};
    int reserved = -1;
    bool followup_owed = false;
    Json all_states, requests = Json::array(), phases = Json::array(), background = Json::array();
    const auto start = Clock::now();
    for (int w = 0; w < windows; ++w) {
        std::this_thread::sleep_until(start + std::chrono::milliseconds(shared_windows ? offsets[w] : spacing_ms*w));
        const auto window_start = Clock::now();
        Json batch = Json::array(), states = Json::object(), risks = Json::object();
        std::array<double, 2> score{};
        std::future<void> pending;
        const int deferred = protect && w == 1 && last_sent[1] == 0 ? second_batch - calls : 0;
        const int prefix = (w == 1 ? second_batch : calls) - deferred;
        for (int i = 0; i < 2; ++i) {
            auto& v = fleet[i];
            v.sample = pilot_rpc.call("falconTraceClock").as<std::uint64_t>();
            if (i == 0) pending = std::async(std::launch::async, [&] { append_background(dpu, inputs, prefix, batch); });
            v.state = state_rpc.get_state(v.name);
            states[v.name] = state_record(v);
            score[i] = projected_error(v, gate_x);
            risks[v.name] = score[i];
        }
        const auto sample_ms = duration_ms(window_start);
        pending.get();
        const auto ready_ms = duration_ms(window_start);
        background.insert(background.end(), batch.begin(), batch.end());

        bool released = false;
        const int by_risk = score[1] > score[0] ? 1 : 0;
        const int by_wait = last_sent[1] < last_sent[0] ? 1 : 0;
        if (reserved >= 0 && score[reserved] <= healthy.at(w).at(fleet[reserved].name).get<double>()) {
            released = true;
            reserved = -1;
        }
        int first = w == 0 || policy == "risk" ? by_risk : (reserved >= 0 ? reserved : by_wait);
        if (policy == "fixed")
            first = recipients[w].empty() ? -1 : (recipients[w] == fleet[0].name ? 0 : 1);
        else if (shared_windows) {
            if (w == 0) first = 1;
            if (yield && w == 1) {
                followup_owed = last_sent[1] == 0 && score[0] > peer_trigger_m;
                first = followup_owed ? 0 : 1;
            }
            if (yield && w == 2) first = followup_owed ? 1 : 0;
            if (deadline && w > 0) {
                // A job that would finish past its deadline leaves the mandatory queue.
                // Use the other aircraft for a best-effort update if this queue is empty.
                const int finish_ms = offsets[w] + static_cast<int>(deadline_budget_ms);
                const int peer_deadline_ms = 105;
                const int followup_deadline_ms = 330;
                const bool peer_due = score[0] > peer_trigger_m && last_sent[0] < 0 &&
                                      finish_ms <= peer_deadline_ms;
                const bool followup_due = last_sent[1] == 0 && finish_ms <= followup_deadline_ms;
                first = peer_due && (!followup_due || peer_deadline_ms < followup_deadline_ms) ? 0 :
                        (followup_due ? 1 : (last_sent[1] != 0 ? 0 : by_risk));
            }
        }
        if (w == 0) all_states = states;
        phases.push_back({{"window",w}, {"start_ns",fleet.front().sample}, {"states",std::move(states)},
            {"risk",std::move(risks)}, {"sample_ms",sample_ms}, {"ready_ms",ready_ms},
            {"first_candidate",first < 0 ? "" : fleet[first].name}, {"released",released},
            {"background",std::move(batch)}});
        if (yield) phases.back()["followup_owed"] = followup_owed;
        if (policy == "observe" || first < 0) continue;
        const int count = policy == "fixed" || shared_windows ? 1 : 2;
        for (int option = 0; option < count; ++option) {
            const int i = option == 0 ? first : 1-first;
            auto row = deadline_update(fleet[i], int(requests.size())+1, pilot_rpc, dpu, calibration,
                                       start, window_start, reserve_ms, false);
            row["window"] = w;
            row["window_start_ns"] = fleet.front().sample;
            if (row.at("accepted").get<bool>()) {
                last_sent[i] = w;
                if (w == 0 && i == first && policy == "reserve") reserved = i;
            }
            const bool exhausted = row.at("status") == "insufficient_time";
            requests.push_back(std::move(row));
            if (exhausted) break;
        }
        if (deferred) {
            Json tail = Json::array();
            append_background(dpu, inputs, deferred, tail, prefix);
            background.insert(background.end(), tail.begin(), tail.end());
            auto& phase_background = phases.back().at("background");
            phase_background.insert(phase_background.end(), tail.begin(), tail.end());
        }
        if (second_batch != calls) phases.back()["deferred_background_calls"] = deferred;
    }
    const auto start_ns = all_states.at(fleet.front().name).at("sample_ns");
    return {{"states",std::move(all_states)}, {"requests",std::move(requests)}, {"windows",std::move(phases)},
            {"background",std::move(background)}, {"start_ns",start_ns}};
}

Json run(const Json& spec, const std::string& host, const std::string& package) {
    const std::string policy = spec.at("policy");
    const bool use_table = policy == "route_table";
    const bool adaptive = policy == "adaptive" || policy == "lock_T" ||
                          policy == "lock_c" || policy == "lock_order";
    const bool adapt_period = adaptive && policy != "lock_T";
    const bool adapt_model = adaptive && policy != "lock_c";
    const bool adapt_order = adaptive && policy != "lock_order";
    const bool probe = policy == "probe";
    require(adaptive || use_table || probe || policy == "fixed" || policy == "deadline_pair", "invalid policy");
    const auto table = spec.value("route_table", Json::object());
    const int fixed_period = spec.at("fixed_period_slots");
    const int fixed_model = spec.at("fixed_model");
    const int cap = spec.at("request_budget");
    const int background_period_us = spec.value("background_period_us", 0);
    require(background_period_us == 0 || background_period_us >= 2000, "background period is too short");
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
    std::vector<f::Raw237> background_inputs;
    const auto checks = verify_models(dpu, package, background_inputs);
    std::cout << Json{{"ready", true}, {"goldens", checks}}.dump() << std::endl;
    std::string start;
    require(bool(std::getline(std::cin, start)) && start == "GO", "start signal missing");
    if (spec.count("recovery_schedule"))
        return run_recovery_schedule(spec.at("recovery_schedule"), fleet, state_rpc, pilot_rpc, dpu,
                                     background_inputs, calibration);
    if (policy == "deadline_pair")
        return run_deadline_pair(spec, fleet, state_rpc, pilot_rpc, dpu, background_inputs, calibration);
    const auto origin = Clock::now() + std::chrono::milliseconds(450);
    std::mutex device;
    std::future<Json> background;
    if (background_period_us)
        background = std::async(std::launch::async, run_background, std::ref(dpu), std::ref(device),
                                std::cref(background_inputs), origin, background_period_us);
    Json requests = Json::array(), slots = Json::array();
    int attempts = 0;
    for (const int slot : spec.at("slots")) {
        const auto release = origin + std::chrono::milliseconds((slot-1)*slot_ms);
        std::this_thread::sleep_until(release);
        if (attempts == cap) break;
        const auto begin = Clock::now();
        Json decision = {{"slot",slot}, {"start_lateness_ms",std::chrono::duration<double,std::milli>(begin-release).count()}};
        try {
            const auto planned = table.find(std::to_string(slot));
            const std::string recipient = planned != table.end() ? planned->at("vehicle").get<std::string>() : "";
            Vehicle* selected = nullptr;
            auto best = std::make_tuple(std::numeric_limits<double>::infinity(), 0, std::string());
            Json candidates = Json::array();
            for (auto& v : fleet) {
                if (use_table && v.name != recipient) continue;
                v.sample = pilot_rpc.call("falconTraceClock").as<std::uint64_t>();
                v.state = state_rpc.get_state(v.name);
                const double end = window_end(v);
                const bool critical = std::isfinite(end);
                const int period = use_table ? planned->at("period_slots").get<int>()
                                             : (adapt_period ? (critical ? 1 : 4) : fixed_period);
                const bool eligible = (!probe || &v == &fleet.front()) &&
                    (!v.last_slot || slot-v.last_slot >= period) && v.state.position[0] < 9.95;
                candidates.push_back({{"vehicle",v.name}, {"x_m",v.state.position[0]}, {"T_ms",period*slot_ms},
                                      {"critical",critical}, {"eligible",eligible}});
                if (!eligible) continue;
                const auto key = std::make_tuple(adapt_order ? (end-v.state.position[0])/std::max(.2,v.state.velocity[0])
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
                const int model = use_table ? planned->at("model").get<int>()
                                            : (adapt_model ? (critical ? 11 : 5) : fixed_model);
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
                    const auto queued_at = Clock::now();
                    row["device_requested_us"] = clock_us(queued_at);
                    std::unique_lock<std::mutex> lock(device);
                    const auto lock_at = Clock::now();
                    row["device_acquired_us"] = clock_us(lock_at);
                    row["device_wait_ms"] = std::chrono::duration<double,std::milli>(lock_at-queued_at).count();
                    const auto result = dpu.infer(model,input);
                    lock.unlock();
                    row["dpu_run_start_us"] = result.run_start_us;
                    row["dpu_run_end_us"] = result.run_end_us;
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
    Json report = {{"requests", std::move(requests)}, {"slots", std::move(slots)}};
    if (background.valid()) report["background"] = background.get();
    return report;
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
