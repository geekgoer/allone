#pragma once

#include "common/tagged_trace.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace falcon_pilot {
using falcon_trace::Stamp;
using Position = std::array<double, 3>;
constexpr Stamp period_ns = 20000000, fresh_ns = 55000000, telemetry_ns = 8000000;
constexpr Stamp rates_lease_ns = 20000000;

struct Command {
    unsigned source = 0; // 1 local, 2 central proxy, 3 DPU
    Stamp id = 0, sample = 0;
    std::array<double, 4> goal{};
    bool rates = false;
};

struct Flight {
    unsigned mode; // 1 local, 2 central, 3 central with local fallback
    double amplitude, altitude;
    Stamp next = 0, local_seq = 0, last_record = 0, remote_applied = 0, adopted = 0;
    Command remote, applied;
    std::vector<std::vector<double>> trace;
};

inline std::mutex& mutex() { static std::mutex m; return m; }
inline std::map<std::string, Flight>& flights() { static std::map<std::string, Flight> f; return f; }

inline Command steer(const Flight& f, const Position& pos, unsigned source, Stamp id, Stamp stamp)
{
    constexpr double wave = .6;
    const double vx = std::max(0., std::min(1.65, 1.6 * (12. - pos[0])));
    const double error = f.amplitude * std::sin(wave * pos[0]) - pos[1];
    const double vy = std::max(-2.5, std::min(2.5, f.amplitude * wave * std::cos(wave * pos[0]) * vx + 1.3 * error));
    return {source, id, stamp, {{vy, vx, 0., f.altitude}}};
}

inline void start(const std::string& vehicle, unsigned mode, double amplitude, double altitude)
{
    if (mode < 1 || mode > 3 || !std::isfinite(amplitude) || std::abs(amplitude) > 1.5 ||
        !std::isfinite(altitude) || altitude > -1. || altitude < -5.)
        throw std::invalid_argument("Invalid onboard pilot scenario");
    std::lock_guard<std::mutex> lock(mutex());
    if (!flights().emplace(vehicle, Flight{mode, amplitude, altitude}).second)
        throw std::invalid_argument("Stop the previous pilot before starting another");
}

inline bool may_submit(const Flight& f, Stamp id, Stamp sampled, Stamp now)
{
    return f.mode != 1 && sampled <= now && now - sampled < fresh_ns &&
        id > f.remote.id && sampled > f.remote.sample;
}

inline bool submit(const std::string& vehicle, Stamp id, Stamp sampled, double x, double y, double z)
{
    std::lock_guard<std::mutex> lock(mutex());
    auto it = flights().find(vehicle);
    const auto now = falcon_trace::now_ns();
    if (it == flights().end() || !may_submit(it->second, id, sampled, now) ||
        !std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
        return false;
    auto& f = it->second;
    f.remote = steer(f, {x, y, z}, 2, id, sampled);
    return true;
}

inline bool submit_rates(const std::string& vehicle, Stamp id, Stamp sampled,
                         double roll, double pitch, double yaw, double throttle)
{
    std::lock_guard<std::mutex> lock(mutex());
    auto it = flights().find(vehicle);
    if (it == flights().end() || !may_submit(it->second, id, sampled, falcon_trace::now_ns()) ||
        !std::isfinite(roll) || !std::isfinite(pitch) || !std::isfinite(yaw) || !std::isfinite(throttle) ||
        std::abs(roll) > 3. || std::abs(pitch) > 3. || std::abs(yaw) > 2. || throttle < 0. || throttle > 1.)
        return false;
    it->second.remote = {3, id, sampled, {{roll, pitch, yaw, throttle}}, true};
    return true;
}

inline bool active(const std::string& vehicle)
{
    std::lock_guard<std::mutex> lock(mutex());
    return flights().count(vehicle) != 0;
}

inline Command choose(const std::string& vehicle, const Position& pos, Stamp now)
{
    std::lock_guard<std::mutex> lock(mutex());
    auto it = flights().find(vehicle);
    if (it == flights().end()) return {};
    auto& f = it->second;
    const auto& remote = f.remote;
    if (remote.id && remote.sample <= now && now - remote.sample < fresh_ns) {
        if (f.remote_applied != remote.id) return remote;
        if (!remote.rates || (f.applied.source == remote.source && now - f.adopted < rates_lease_ns)) return {};
    }
    if (f.mode == 2 || now < f.next) return {};
    f.next = now + period_ns;
    return steer(f, pos, 1, ++f.local_seq, now);
}

inline void record(const std::string& vehicle, const Position& pos, Stamp finished, const Command& used)
{
    std::lock_guard<std::mutex> lock(mutex());
    auto it = flights().find(vehicle);
    if (it == flights().end()) return;
    auto& f = it->second;
    if (used.id) {
        f.applied = used;
        f.adopted = finished;
        if (used.source != 1) f.remote_applied = used.id;
    }
    if (!used.id && finished - f.last_record < telemetry_ns) return;
    if (f.trace.size() >= 4000) throw std::runtime_error("Onboard trial exceeded telemetry capacity");
    f.last_record = finished;
    f.trace.push_back({double(finished), pos[0], pos[1], pos[2], double(used.source), double(used.id), double(used.sample)});
}

inline std::vector<std::vector<double>> stop(const std::string& vehicle)
{
    std::lock_guard<std::mutex> lock(mutex());
    auto it = flights().find(vehicle);
    if (it == flights().end()) throw std::invalid_argument("No active onboard trial");
    auto trace = std::move(it->second.trace);
    flights().erase(it);
    return trace;
}
} // namespace falcon_pilot
