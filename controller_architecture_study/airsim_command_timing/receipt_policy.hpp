#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <vector>

namespace receipt_policy {

using Time = std::int64_t;
using ClockSample = std::array<Time, 3>; // PS send, simulator stamp, PS receive

struct Decision {
    Time cost_end_ns;
    bool trigger;
    const char* evidence;
    Time adoption_lower_ns = 0;
    Time adoption_upper_ns = 0;
};

inline Decision determine(bool enabled, const std::vector<ClockSample>& calibration,
                          const ClockSample& exchange, std::uint64_t expected_id,
                          const std::vector<std::uint64_t>& receipt, Time sent, Time fallback,
                          Time deadline, Time reserve_limit) {
    Decision decision{fallback, fallback >= deadline && fallback > reserve_limit, "no_matching_adoption"};
    if (receipt.size() != 4 || receipt[1] != expected_id || expected_id == 0 || !receipt[2] ||
        receipt[2] > receipt[3] || receipt[3] > receipt[0])
        return decision;

    constexpr Time max_drift_ppm = 200; // Assumption for this controlled HIL comparison.
    const Time pivot = exchange[2];
    Time offset_low = std::numeric_limits<Time>::lowest();
    Time offset_high = std::numeric_limits<Time>::max();
    const auto add_sample = [&](const ClockSample& sample) {
        // Offset can drift between the recorded send/receive window and either
        // event being classified. Include the command's complete send-to-ping span.
        const Time distance = std::max({std::abs(pivot - sample[0]), std::abs(pivot - sample[2]),
                                        std::abs(sent - sample[0]), std::abs(sent - sample[2])});
        const Time allowance = (distance * max_drift_ppm + 999999) / 1000000;
        offset_low = std::max(offset_low, sample[1] - sample[2] - allowance);
        offset_high = std::min(offset_high, sample[1] - sample[0] + allowance);
    };
    for (const auto& sample : calibration)
        add_sample(sample);
    add_sample(exchange);
    if (calibration.empty() || offset_low > offset_high) {
        decision.evidence = "clock_unresolved";
        return decision;
    }
    const auto lower = std::max(sent, static_cast<Time>(receipt[2]) - offset_high);
    const auto upper = std::min(pivot, static_cast<Time>(receipt[3]) - offset_low);
    if (lower > upper) {
        decision.evidence = "clock_unresolved";
        return decision;
    }
    decision.adoption_lower_ns = lower;
    decision.adoption_upper_ns = upper;
    const bool too_late = lower >= deadline && lower > reserve_limit;
    const bool in_time = upper < deadline || upper <= reserve_limit;
    decision.evidence = too_late ? "adoption_late" : in_time ? "adoption_below_trigger" : "adoption_uncertain";
    if (enabled && (too_late || in_time)) {
        decision.trigger = too_late;
        decision.cost_end_ns = upper;
    }
    return decision;
}

} // namespace receipt_policy
