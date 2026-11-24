#pragma once

#include "receipt_policy.hpp"
#include "rpc/airsim_rpc.hpp"
#include <atomic>
#include <mutex>
#include <thread>

namespace adoption_events {

struct Event {
    std::uint64_t id = 0;
    receipt_policy::Time start_ns = 0, end_ns = 0, received_ns = 0;
};

class Inbox {
public:
    Inbox(const std::string& host, std::uint16_t port, std::uint64_t base)
        : base_(base), entries_(32768), client_(host, port, 250),
          cursor_(client_.adoption_cursor()), worker_(&Inbox::read, this) {}

    ~Inbox() { close(); }
    void close() {
        stopped_ = true;
        if (worker_.joinable())
            worker_.join();
    }

    Event lookup(std::uint64_t id) const {
        if (!contains(id))
            return {};
        std::lock_guard<std::mutex> lock(mutex_);
        return entries_[id - base_];
    }

    std::vector<Event> captured() const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<Event> events;
        for (const auto& event : entries_)
            if (event.id != 0)
                events.push_back(event);
        return events;
    }

    bool failed() const { return failed_; }

private:
    bool contains(std::uint64_t id) const { return id >= base_ && id - base_ < entries_.size(); }

    void read() noexcept {
        try {
            while (!stopped_) {
                const auto batch = client_.adoption_events(cursor_);
                const auto now = psairsim::monotonic_ns();
                if (batch.empty() || batch[0].size() != 1 || batch[0][0] < cursor_ ||
                    batch[0][0] - cursor_ != batch.size() - 1)
                    throw std::runtime_error("Malformed adoption stream cursor");
                cursor_ = batch[0][0];
                if (batch.size() == 1)
                    continue;
                std::lock_guard<std::mutex> lock(mutex_);
                for (std::size_t i = 1; i < batch.size(); ++i) {
                    const auto& record = batch[i];
                    if (record.size() != 5 || record[2] == 0 || record[2] > record[3] || record[3] > record[4])
                        throw std::runtime_error("Malformed adoption event");
                    if (!contains(record[0]))
                        continue;
                    auto& event = entries_[record[0] - base_];
                    if (!event.id)
                        event = {record[0], static_cast<receipt_policy::Time>(record[3]),
                                 static_cast<receipt_policy::Time>(record[4]), now};
                }
            }
        } catch (...) {
            failed_ = true;
        }
    }

    const std::uint64_t base_;
    mutable std::mutex mutex_;
    std::vector<Event> entries_;
    psairsim::Client client_;
    std::uint64_t cursor_;
    std::atomic<bool> stopped_{false}, failed_{false};
    std::thread worker_;
};

struct Observation {
    std::uint64_t id = 0, version = 0, cycle = 0;
    receipt_policy::Time sent = 0, converted = 0, output = 0, ack = 0, deadline = 0, reserve_limit = 0;
};

inline receipt_policy::Decision resolve(bool enabled, const std::vector<receipt_policy::ClockSample>& calibration,
                                        const Observation& cmd, const Event& event) {
    const std::vector<std::uint64_t> proof{static_cast<std::uint64_t>(event.end_ns), event.id,
        static_cast<std::uint64_t>(event.start_ns), static_cast<std::uint64_t>(event.end_ns)};
    // Matching ID puts the adoption between PS send and the independent event receipt.
    return receipt_policy::determine(enabled, calibration, {cmd.sent, event.end_ns, event.received_ns},
                                      cmd.id, proof, cmd.sent, cmd.ack, cmd.deadline, cmd.reserve_limit);
}

} // namespace adoption_events
