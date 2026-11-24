#pragma once

#include "json.hpp"
#include <cstdint>
#include <mutex>
#include <string>

// Port of the frozen obligation_scheduler.py and the whole-job PS/PL path of
// bottleneck_joint.py. All times are caller-supplied milliseconds on ONE clock.
// A successful finite replay is conditional on supplied costs/qualifications.
namespace ps_scheduler {
using Json = nlohmann::json;
constexpr double epsilon = 1e-8;

// modes: array of {numeric_id, service_ms, selection_mse, ...}, or object keyed
// by numeric id. options: now(required), decision_ms=.5, blocked_until=0,
// permissions=[], max_checks=8, max_candidates=12, and enable_* flags.
// Algorithm defaults match Python, including enable_skip=true. The production
// migration caller MUST explicitly pass enable_skip=false until stage 3.
Json plan(const Json& calendar, const Json& modes, const Json& options);
Json evaluate(const Json& jobs, const Json& order, const Json& choice,
              const Json& modes, double now, double decision_ms,
              double blocked_until = 0.0);
Json evaluate_ready_edf(const Json& jobs, const Json& modes, double decision_ms, double rearm_ms);
Json diagnosis(const Json& jobs, const Json& choice, const Json& modes,
               double now, double decision_ms, double blocked_until = 0.0);
Json tail_budget(const Json& jobs, const Json& order, const Json& choice,
                 const Json& modes, double now);
Json dispatch_budget(const Json& jobs, const Json& order, const Json& choice,
                     const Json& modes, double now);
void validate(const Json& calendar, const Json& modes);

// Immutable offline state operations. These throw invalid_argument on stale,
// expired or illegal changes; the caller must hold its complete runtime lock.
Json commit(const Json& calendar, const Json& decision, double now);
Json delivered(const Json& calendar, const std::string& name, double actuation_time);
Json tighten_hold(const Json& calendar, const std::string& vehicle, double deadline);

// Optional narrow atomic wrapper for standalone calendars. The caller's live
// release/control-version state is not owned here. Its own runtime lock/version
// must guard snapshot and install; don't mistake this class for that integration.
class CalendarStore {
public:
    explicit CalendarStore(Json calendar);
    Json snapshot() const;
    Json commit_at_monotonic_ms(const Json& decision, double clock_origin_ms = 0.0);
    Json delivered_at(const std::string& name, double actuation_time);
    Json tighten(const std::string& vehicle, double deadline);
private:
    mutable std::mutex mutex_;
    Json calendar_;
};
double monotonic_ms();
} // namespace ps_scheduler
