#include "adoption_inbox.hpp"
#include "tagged_trace.hpp"
#include <cassert>

int main() {
    using namespace adoption_events;
    using receipt_policy::ClockSample;
    constexpr std::int64_t ms = 1000000;
    const std::vector<ClockSample> clock{{{100 * ms, 105 * ms, 100 * ms}}};
    const Observation early{42, 1, 7, 110 * ms, 109 * ms, 108 * ms, 150 * ms, 130 * ms, 129 * ms};
    const Event early_event{42, 118 * ms, 119 * ms, 125 * ms};
    const auto use_early = resolve(true, clock, early, early_event);
    assert(!use_early.trigger && use_early.cost_end_ns < 115 * ms);
    assert(resolve(false, clock, early, early_event).trigger);
    assert(resolve(true, clock, early, {}).cost_end_ns == early.ack);
    assert(resolve(true, clock, early, {43, 118 * ms, 119 * ms, 125 * ms}).trigger);

    auto late = early;
    late.ack = 111 * ms;
    const Event late_event{42, 140 * ms, 141 * ms, 142 * ms};
    assert(!resolve(true, clock, late, {}).trigger);
    assert(!resolve(false, clock, late, late_event).trigger);
    assert(resolve(true, clock, late, late_event).trigger);

    const auto cursor = falcon_trace::event_cursor();
    {
        falcon_trace::Command command(7);
        falcon_trace::mark_goal(7);
        assert(falcon_trace::event_cursor() == cursor);
        const auto time = falcon_trace::now_ns();
        falcon_trace::mark_firmware(7, time, time + 1);
        falcon_trace::mark_firmware(7, time + 2, time + 3);
    }
    {
        falcon_trace::Command command(4); // Command IDs need not arrive in numeric order.
        falcon_trace::mark_goal(4);
        const auto time = falcon_trace::now_ns();
        falcon_trace::mark_firmware(4, time, time + 1);
    }
    const auto messages = falcon_trace::events_after(cursor);
    assert(messages.size() == 3 && messages[1][0] == 7 && messages[2][0] == 4);
    const auto empty = falcon_trace::events_after(messages[0][0]);
    assert(empty.size() == 1 && empty[0][0] == messages[0][0]);
}
