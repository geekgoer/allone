#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace falcon_trace {

using Stamp = std::uint64_t;
// ID, RPC handler entry, goal written, start and end of first firmware tick.
using Record = std::array<Stamp, 5>;

inline Stamp now_ns() {
    using namespace std::chrono;
    return duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
}

inline Stamp& current_id() {
    static thread_local Stamp id = 0;
    return id;
}

inline std::array<Record, 32768>& history() {
    static std::array<Record, 32768> records{};
    return records;
}

inline std::mutex& history_mutex() {
    static std::mutex mutex;
    return mutex;
}

struct EventStream {
    std::array<Record, 4096> entries{};
    Stamp published = 0;
    std::condition_variable changed;
};

inline EventStream& event_stream() {
    static EventStream stream;
    return stream;
}

inline Stamp event_cursor() {
    std::lock_guard<std::mutex> lock(history_mutex());
    return event_stream().published;
}

inline std::vector<std::vector<Stamp>> events_after(Stamp cursor) {
    auto& stream = event_stream();
    std::unique_lock<std::mutex> lock(history_mutex());
    if (cursor > stream.published)
        throw std::invalid_argument("Invalid adoption event cursor");
    stream.changed.wait_for(lock, std::chrono::milliseconds(4), [&] { return cursor < stream.published; });
    const Stamp oldest = stream.published > stream.entries.size() ? stream.published - stream.entries.size() : 0;
    if (cursor < oldest)
        throw std::runtime_error("Adoption event history overrun");
    const auto next = std::min(cursor + 64, stream.published);
    std::vector<std::vector<Stamp>> batch{{next}};
    for (; cursor < next; ++cursor) {
        const auto& record = stream.entries[cursor % stream.entries.size()];
        batch.emplace_back(record.begin(), record.end());
    }
    return batch;
}

class Command {
public:
    explicit Command(Stamp id) {
        std::lock_guard<std::mutex> lock(history_mutex());
        if (id == 0 || current_id() != 0 || history()[id % history().size()][0] == id)
            throw std::invalid_argument("Trace IDs must be unique, nonzero and unnested");
        history()[id % history().size()] = {{id, now_ns(), 0, 0, 0}};
        current_id() = id;
    }
    ~Command() { current_id() = 0; }
    Command(const Command&) = delete;
    Command& operator=(const Command&) = delete;
};

inline void mark_goal(Stamp id) {
    std::lock_guard<std::mutex> lock(history_mutex());
    auto& record = history()[id % history().size()];
    if (record[0] == id && record[2] == 0)
        record[2] = now_ns();
}

inline void mark_firmware(Stamp id, Stamp started, Stamp ended) {
    std::lock_guard<std::mutex> lock(history_mutex());
    auto& record = history()[id % history().size()];
    if (record[0] == id && record[2] != 0 && record[3] == 0) {
        record[3] = started;
        record[4] = ended;
        auto& stream = event_stream();
        stream.entries[stream.published % stream.entries.size()] = record;
        ++stream.published;
        stream.changed.notify_one();
    }
}

inline std::vector<std::vector<Stamp>> read(const std::vector<Stamp>& ids) {
    if (ids.size() > 512)
        throw std::invalid_argument("Read at most 512 trace IDs per request");
    std::vector<std::vector<Stamp>> result;
    std::lock_guard<std::mutex> lock(history_mutex());
    for (const auto id : ids) {
        const auto& record = history()[id % history().size()];
        if (id != 0 && record[0] == id)
            result.emplace_back(record.begin(), record.end());
    }
    return result;
}

inline std::vector<Stamp> snapshot(Stamp id) {
    std::lock_guard<std::mutex> lock(history_mutex());
    const auto& record = history()[id % history().size()];
    if (id == 0 || record[0] != id)
        return {now_ns(), 0, 0, 0};
    return {now_ns(), id, record[3], record[4]};
}

} // namespace falcon_trace
