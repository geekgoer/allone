#pragma once
// Offline instrumentation only: exclusive (non-double-counted) stages.
#include <array>
#include <cstdint>
#include <stdexcept>
#include <time.h>

namespace ps_profile {
enum Stage { Other, Preprocess, Diagnosis, Candidate, Budget, GlobalCheck,
             Simulation, SimulationReport, Count };
static const char* const names[] = {"plan_other", "preprocess", "diagnosis", "candidate",
    "backward_budget", "global_check", "simulation", "simulation_report"};
struct Stamp { std::int64_t wall, cpu; };
inline Stamp stamp() {
    timespec w{}, c{};
    if(clock_gettime(CLOCK_MONOTONIC, &w) || clock_gettime(CLOCK_THREAD_CPUTIME_ID, &c))
        throw std::runtime_error("Profiling clock unavailable");
    return {std::int64_t(w.tv_sec)*1000000000+w.tv_nsec,
            std::int64_t(c.tv_sec)*1000000000+c.tv_nsec};
}
struct State {
    bool enabled=false;
    Stage active=Other;
    Stamp previous{};
    std::array<std::int64_t, Count> wall{}, cpu{};
    std::array<unsigned, Count> calls{};
    void tick(Stamp now) {
        wall[active]+=now.wall-previous.wall;
        cpu[active]+=now.cpu-previous.cpu;
        previous=now;
    }
};
extern thread_local State state;
class Scope {
    bool enabled_;
    Stage parent_;
public:
    explicit Scope(Stage stage):enabled_(state.enabled),parent_(state.active) {
        if(enabled_) { state.tick(stamp()); state.active=stage; ++state.calls[stage]; }
    }
    ~Scope() {
        if(enabled_) { state.tick(stamp()); state.active=parent_; }
    }
};
inline void begin() { state=State{}; state.previous=stamp(); state.enabled=true; }
inline void end() { state.tick(stamp()); state.enabled=false; }
}
#define PS_PROFILE_JOIN_(a,b) a##b
#define PS_PROFILE_JOIN(a,b) PS_PROFILE_JOIN_(a,b)
#ifdef PS_PROFILE_ENABLED
#define PS_SCOPE(stage) ps_profile::Scope PS_PROFILE_JOIN(profile_scope_,__LINE__)(ps_profile::stage)
#else
#define PS_SCOPE(stage)
#endif
