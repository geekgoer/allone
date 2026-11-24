#include "receipt_policy.hpp"
#include <cassert>

int main() {
    using namespace receipt_policy;
    constexpr Time ms = 1000000;
    const std::vector<ClockSample> calibration{{{100 * ms, 105 * ms, 100 * ms}}};
    const ClockSample slow{{111 * ms, 160 * ms, 170 * ms}};
    const std::vector<std::uint64_t> early{160 * ms, 42, 122 * ms, 123 * ms};
    const std::vector<std::uint64_t> late{160 * ms, 42, 142 * ms, 143 * ms};
    const auto decide = [&](bool enabled, const std::vector<std::uint64_t>& receipt) {
        return determine(enabled, calibration, slow, 42, receipt, 110 * ms, 170 * ms, 130 * ms, 129 * ms);
    };

    const auto baseline = decide(false, early);
    assert(baseline.trigger && baseline.cost_end_ns == 170 * ms);
    const auto verified = decide(true, early);
    assert(!verified.trigger && verified.cost_end_ns > 117 * ms && verified.cost_end_ns < 119 * ms);
    const auto real_late = decide(true, late);
    assert(real_late.trigger && real_late.cost_end_ns > 137 * ms && real_late.cost_end_ns < 139 * ms);
    const auto missing = decide(true, {});
    assert(missing.trigger && missing.cost_end_ns == baseline.cost_end_ns);
    const auto mismatched = decide(true, {160 * ms, 43, 122 * ms, 123 * ms});
    assert(mismatched.trigger && mismatched.cost_end_ns == baseline.cost_end_ns);
    const auto crossing = decide(true, {160 * ms, 42, 134 * ms, 136 * ms});
    assert(crossing.trigger && crossing.cost_end_ns == baseline.cost_end_ns);
    const auto invalid = decide(true, {160 * ms, 42, 159 * ms, 161 * ms});
    assert(invalid.trigger && invalid.cost_end_ns == baseline.cost_end_ns);
    const auto inconsistent = determine(true, calibration, {111 * ms, 206 * ms, 170 * ms},
                                        42, early, 110 * ms, 170 * ms, 130 * ms, 129 * ms);
    assert(inconsistent.trigger && inconsistent.cost_end_ns == baseline.cost_end_ns);
}
