#pragma once
#include "scheduler.hpp"
#include <mutex>
#include <condition_variable>

namespace ps_scheduler {
// Live lifecycle adapter. EVERY absolute time is CLOCK_MONOTONIC milliseconds.
// config: names[], origin_ms, period_ms, periods_ms[], model_id, candidate_ids[],
// prep_ms, post_ms, decision_ms, fresh_ms=80, command_reserve_ms=post_ms,
// rearm_ms=post_ms, adaptive=true, enable_skip=false, initial_periods={}.
// Costs supplied by the caller must include a documented conservative envelope
// for observation/network/PS preparation and command-return work; no implicit
// historical Windows 3ms network addition is made here.
class LiveCalendar {
public:
    LiveCalendar(Json config, Json modes);
    // All returns include drop/planned_skip rows, not just admitted rows. Only
    // status==sampling rows may be handed to a per-vehicle worker.
    Json release_due(double now_ms, double stop_before_ms, Json* expiries=nullptr);
    double next_wakeup_ms() const;
    Json state() const;
    Json row(const std::string& vehicle, std::uint64_t cycle, double* lock_wait_us=nullptr) const;
    std::uint64_t control_version(const std::string& vehicle) const;
    bool observe(const std::string& vehicle, double sample_ms, double hold_deadline_ms,
                 std::uint64_t held_version, bool stable=false,
                 const std::string& evidence="unknown phase", double available_ms=-1.);
    // The optional policy retires ordinary control, with no fallback RPC.
    // Returns each new expiry once, including expiries latched by worker calls.
    Json expire_holds(double now_ms);
    // Mark sampled RAW237 ready, freezing min(slot, hold, sample+F,recovery).
    // Returns the current row, cancelled=true when already expired/not owned.
    Json queued(const std::string& vehicle, std::uint64_t cycle,
                double sample_ms, double input_ready_ms, double hold_deadline_ms);
    // Atomic single-DPU acquisition. Returns null if idle/no work or waiting for
    // a promised head; returns dispatch=false on a rejected queue row so its
    // waiting worker can finish/record it. Successful row has dispatch=true.
    // With separate_dispatch_service, refresh_ms shares the caller's timebase.
    Json take(double now_ms, double* lock_wait_us=nullptr, double (*refresh_ms)()=nullptr);
    void dpu_done(const std::string& vehicle, std::uint64_t cycle,
                  double result_ready_ms, bool success);
    // Native-I/O command_pending equivalent. Immediately before any command
    // RPC: atomically guard remaining budget and invalidate the old held-state
    // certificate. False means cancelled/insufficient budget: do NOT send.
    // Pending persists on uncertain/no acknowledgement; retire that vehicle.
    bool command_begin(const std::string& vehicle, std::uint64_t cycle,
                       double begin_ms);
    // Call whenever a real command was sent and acknowledged, even if LATE.
    // Invalidates held-control certificates, then caller must observe new pair.
    std::uint64_t command_return(const std::string& vehicle, std::uint64_t cycle,
                                 double command_return_ms);
    Json finish(const std::string& vehicle, std::uint64_t cycle,
                double finish_ms, const std::string& status);
    void retire(const std::string& vehicle);
    void update_costs(double prep_ms,double post_ms,double rearm_ms,
                      double command_reserve_ms);
    Json snapshot(double now_ms) const;
    // Production snapshot clock is sampled AFTER acquiring the calendar lock.
    // Explicit-time snapshot above remains for deterministic reference tests.
    Json snapshot_current() const;
    Json snapshot_profiled() const;
    Json snapshot_wait_profiled(double wait_ms) const;
    // A completed command with a matching paired observation wakes the manager.
    // Periodic planning still provides a fallback; no planner work runs under
    // the calendar lock.
    std::uint64_t paired_generation() const;
    bool wait_for_paired(std::uint64_t generation,double wait_ms) const;
    Json plan_snapshot(const Json& snapshot) const;
    // Acquires calendar lock FIRST and only then reads clock. Full atomic CAS
    // and release/control-version guard; no partial install on any rejection.
    Json install(const Json& snapshot,const Json& decision);
    Json install_profiled(const Json& snapshot,const Json& decision);
    // Strict install first. On a changed context, at most one latest-state
    // validation and one final CAS. Never spin or hold the lock during checking.
    Json install_with_revalidation(const Json& snapshot,const Json& decision);
    Json plan_once();
    // Explicit test seam; no real-time production call should use fake time.
    Json install_at_for_test(const Json& snapshot,const Json& decision,double now_ms);
private:
    std::string key(const std::string&v,std::uint64_t cycle) const;
    void expire_hold_unlocked(const std::string& vehicle, Json& loop, double now);
    void retire_unlocked(const std::string& vehicle, Json& loop);
    Json snapshot_unlocked(double now_ms) const;
    Json install_unlocked(const Json& snapshot,const Json& decision,double now_ms);
    mutable std::mutex mutex_;
    mutable std::condition_variable proof_ready_;
    Json config_, modes_, loops_=Json::object(), rows_=Json::object();
    Json future_modes_=Json::object(),pending_=Json::object(),planned_skip_=Json::object();
    Json dispatch_order_=Json::array();
    Json hold_expiries_=Json::array();
    std::string active_;
    double dispatch_expires_ms_=0.;
    std::uint64_t revision_=0;
    std::uint64_t paired_generation_=0;
};
} // namespace ps_scheduler
