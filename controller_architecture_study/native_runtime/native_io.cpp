// Native RPC ownership and an independent absolute release calendar.
// Python retains the existing model executor and B/admission algorithms.
#include "vehicles/multirotor/api/MultirotorRpcLibClient.hpp"
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#undef min
#undef max
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using Client = msr::airlib::MultirotorRpcLibClient;
static double clock_s() {
    LARGE_INTEGER ticks, frequency;
    QueryPerformanceCounter(&ticks);
    QueryPerformanceFrequency(&frequency);
    return double(ticks.QuadPart) / double(frequency.QuadPart);
}
template<size_t N> static void copy_text(char (&dst)[N], const std::string& s) {
    std::memcpy(dst, s.data(), std::min(N - 1, s.size()));
}
extern "C" {
struct Event {
    int32_t kind, vehicle; // 1 sample, 2 acknowledgement, 3 command done, 4 release, 5 error
    uint64_t id, version;
    double submitted, started, state_end, collision_end, finished, planned, period;
    uint64_t sim_ns, collision_ns;
    double state[16], control[4], command[4];
    int32_t collided, status; // 0 applied; 1 expired; 2 insufficient delivery budget; 3 cancelled
    char object[128], error[256];
};
}
struct Task {
    bool command = false, cancelled = false;
    uint64_t id = 0;
    double submitted = 0, deadline = 0;
    std::array<double, 4> control{};
};
struct Loop {
    std::string name;
    std::mutex mutex;
    std::condition_variable wake;
    std::deque<Task> tasks;
    bool busy = false;
    std::thread worker;
};
struct Calendar {
    bool active = false;
    uint64_t id = 0, version = 0;
    uint64_t control_token = 0;
    bool command_pending = false;
    double next = 0, period = 0, end = 0, last = -std::numeric_limits<double>::infinity();
};
class Runtime {
public:
    std::string host;
    int port;
    uint64_t mask;
    double calibration[5], command_bound;
    bool allow_late, mock;
    double mock_delay;
    std::vector<std::unique_ptr<Loop>> loops;
    std::vector<Calendar> calendars;
    std::mutex events_mutex, calendar_mutex, init_mutex;
    std::condition_variable initialized;
    size_t ready = 0;
    std::string init_error;
    std::deque<Event> events;
    bool overflow = false;
    std::atomic<bool> stopping{false};
    HANDLE completion = nullptr, changed = nullptr, timer = nullptr;
    std::thread releaser;

    Runtime(const char* ip, int server_port, const char* names, uint64_t cpus,
            const double* cal, double bound, bool late, bool fake, double delay)
        : host(ip), port(server_port), mask(cpus), command_bound(bound), allow_late(late), mock(fake), mock_delay(delay) {
        std::copy(cal, cal + 5, calibration);
        completion = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        changed = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        timer = CreateWaitableTimerExW(nullptr, nullptr, 2, TIMER_ALL_ACCESS);
        if (!completion || !changed || !timer) { close_handles(); throw std::runtime_error("native wait handle creation failed"); }
        std::istringstream lines(names);
        std::string name;
        while (std::getline(lines, name)) if (!name.empty()) {
            auto loop = std::make_unique<Loop>(); loop->name = name; loops.push_back(std::move(loop));
        }
        if (loops.empty()) { close_handles(); throw std::invalid_argument("no vehicles"); }
        calendars.resize(loops.size());
        try {
            for (size_t i = 0; i < loops.size(); ++i) loops[i]->worker = std::thread([this, i] { run_loop(i); });
            std::unique_lock<std::mutex> lock(init_mutex);
            initialized.wait(lock, [&] { return ready == loops.size(); });
            if (!init_error.empty()) throw std::runtime_error(init_error);
            lock.unlock();
            releaser = std::thread([this] { release_loop(); });
        } catch (...) { stop(); close_handles(); throw; }
    }
    ~Runtime() { stop(); close_handles(); }
    void close_handles() {
        for (HANDLE h : {completion, changed, timer}) if (h) CloseHandle(h);
        completion = changed = timer = nullptr;
    }
    void pin() {
        if (mask && !SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR(mask)))
            throw std::runtime_error("native thread affinity failed");
    }
    void push(const Event& event) {
        std::lock_guard<std::mutex> lock(events_mutex);
        if (events.size() >= 32768) overflow = true;
        else events.push_back(event);
        SetEvent(completion);
    }
    int poll(Event& event) {
        std::lock_guard<std::mutex> lock(events_mutex);
        if (overflow) throw std::runtime_error("native event queue overflow; no result may be silently lost");
        if (events.empty()) { ResetEvent(completion); return 0; }
        event = events.front(); events.pop_front();
        if (events.empty()) ResetEvent(completion);
        return 1;
    }
    void snapshot(Client* client, size_t i, Event& event) {
        double began = clock_s();
        event.planned = began; // State RPC start, not the command's acknowledgement.
        if (mock) {
            std::this_thread::sleep_for(std::chrono::duration<double>(mock_delay));
            event.state_end = event.collision_end = clock_s();
            event.sim_ns = uint64_t(began * 1e9); event.state[3] = 1.; return;
        }
        const auto state = client->getMultirotorState(loops[i]->name);
        event.state_end = clock_s();
        const auto collision = client->simGetCollisionInfo(loops[i]->name);
        event.collision_end = clock_s();
        const auto& k = state.kinematics_estimated;
        const double values[] = {k.pose.position.x(), k.pose.position.y(), k.pose.position.z(),
            k.pose.orientation.w(), k.pose.orientation.x(), k.pose.orientation.y(), k.pose.orientation.z(),
            k.twist.linear.x(), k.twist.linear.y(), k.twist.linear.z(),
            k.accelerations.linear.x(), k.accelerations.linear.y(), k.accelerations.linear.z(),
            k.twist.angular.x(), k.twist.angular.y(), k.twist.angular.z()};
        std::copy(values, values + 16, event.state);
        event.sim_ns = state.timestamp; event.collision_ns = collision.time_stamp;
        event.collided = collision.has_collided; copy_text(event.object, collision.object_name);
    }
    void run_loop(size_t i) {
        std::unique_ptr<Client> client;
        bool connected = false;
        try {
            pin();
            if (!mock) {
                client = std::make_unique<Client>(host, uint16_t(port), 3.f);
                if (!client->ping()) throw std::runtime_error("AirSim ping failed");
                client->getMultirotorState(loops[i]->name);
            }
            connected = true;
        } catch (const std::exception& e) {
            std::lock_guard<std::mutex> lock(init_mutex); init_error = e.what();
        }
        { std::lock_guard<std::mutex> lock(init_mutex); ++ready; initialized.notify_one(); }
        if (!connected) return;
        auto& loop = *loops[i];
        for (;;) {
            Task task;
            { std::unique_lock<std::mutex> lock(loop.mutex);
                loop.wake.wait(lock, [&] { return stopping || !loop.tasks.empty(); });
                if (loop.tasks.empty() && stopping) break;
                task = loop.tasks.front(); loop.tasks.pop_front();
            }
            Event event{}; event.kind = task.command ? 3 : 1; event.vehicle = int(i);
            event.id = task.id; event.submitted = task.submitted; event.started = clock_s();
            try {
                if (task.command) {
                    std::copy(task.control.begin(), task.control.end(), event.control);
                    event.status = stopping || task.cancelled ? 3 : !allow_late && event.started >= task.deadline ? 1 :
                        !allow_late && event.started + command_bound > task.deadline ? 2 : 0;
                    if (event.status == 0) {
                        double throttle = std::clamp(calibration[0] + (task.control[0] - calibration[1]) / calibration[2],
                            calibration[3], calibration[4]);
                        // Python control_adapter flips pitch/yaw, then AirSim's
                        // Python client flips them again at the RPC boundary.
                        // AirLib's C++ client has NO such second conversion.
                        event.command[0]=task.control[1];event.command[1]=task.control[2];
                        event.command[2]=task.control[3];event.command[3]=throttle;
                        if (!mock) {
                            client->moveByAngleRatesThrottleAsync(float(task.control[1]), float(task.control[2]),
                                float(task.control[3]), float(throttle), 1.f, loop.name);
                            client->ping(); // Same RPC acknowledgement proxy as the Python baseline.
                        }
                        { std::lock_guard<std::mutex> lock(calendar_mutex);
                            calendars[i].control_token = task.id; calendars[i].command_pending = false; }
                        Event ack = event; ack.kind = 2; ack.finished = clock_s(); push(ack);
                    }
                    else { std::lock_guard<std::mutex> lock(calendar_mutex); calendars[i].command_pending = false; }
                }
                if (!task.command || event.status == 0) snapshot(client.get(), i, event);
            } catch (const std::exception& e) { event.kind = 5; copy_text(event.error, e.what()); }
            event.finished = clock_s();
            { std::lock_guard<std::mutex> lock(loop.mutex); loop.busy = false; }
            push(event);
        }
    }
    void submit(int index, Task task) {
        auto& loop = *loops.at(index);
        std::lock_guard<std::mutex> lock(loop.mutex);
        if (stopping || loop.busy) throw std::runtime_error("one native I/O operation per vehicle required");
        if (task.command) { std::lock_guard<std::mutex> guard(calendar_mutex); calendars.at(index).command_pending = true; }
        loop.busy = true; task.submitted = clock_s(); loop.tasks.push_back(task); loop.wake.notify_one();
    }
    Calendar schedule(int index, uint64_t id, uint64_t version, double next, double period, double end) {
        if (!std::isfinite(next) || !std::isfinite(period) || period < .020 || !std::isfinite(end))
            throw std::invalid_argument("finite release calendar and period >=20ms required");
        std::lock_guard<std::mutex> lock(calendar_mutex);
        auto& c = calendars.at(index);
        // Changes affect only future releases. Already queued events stay frozen.
        c.id = std::max(c.id, id); c.version = version;
        c.next = std::max(next, c.last + .020);
        c.period = period; c.end = end; c.active = c.next+1e-9 < end;
        SetEvent(changed); return c;
    }
    bool schedule_batch(Event* changes, int count) {
        if (count < 0 || size_t(count) > calendars.size()) throw std::invalid_argument("invalid calendar batch size");
        std::lock_guard<std::mutex> lock(calendar_mutex);
        std::vector<bool> seen(calendars.size(), false);
        for (int i = 0; i < count; ++i) {
            const auto& e = changes[i];
            if (e.vehicle < 0 || size_t(e.vehicle) >= calendars.size() || seen[e.vehicle] ||
                !std::isfinite(e.planned) || !std::isfinite(e.period) || e.period < .020 || !std::isfinite(e.finished))
                throw std::invalid_argument("invalid calendar batch entry");
            seen[e.vehicle] = true;
            const auto& c = calendars[e.vehicle];
            if (c.id != e.id || e.planned + 1e-9 < c.last + .020) return false;
        }
        // The timer cannot publish a partially installed configuration group.
        // A publication racing the forecast rejects the WHOLE group above.
        for (int i = 0; i < count; ++i) {
            const auto& e = changes[i]; auto& c = calendars[e.vehicle];
            c.version = e.version; c.next = e.planned; c.period = e.period;
            c.end = e.finished; c.active = c.next+1e-9 < c.end;
        }
        SetEvent(changed); return true;
    }
    bool calendar_matches(Event* expected, int count) {
        if (count < 0 || size_t(count) > calendars.size()) throw std::invalid_argument("invalid calendar read size");
        std::lock_guard<std::mutex> lock(calendar_mutex);
        for (int i = 0; i < count; ++i) {
            const auto& e = expected[i]; const auto& c = calendars.at(e.vehicle);
            if (!c.active || c.id != e.id || c.version != e.version ||
                std::abs(c.next-e.planned) > 1e-9 || std::abs(c.period-e.period) > 1e-9) return false;
        }
        return true;
    }
    int guarded_batch(Event* rows, int count, bool commit) {
        if (count <= 0 || size_t(count) > calendars.size()) throw std::invalid_argument("invalid guarded group size");
        std::lock_guard<std::mutex> lock(calendar_mutex);
        std::vector<bool> seen(calendars.size(),false);
        const double now=clock_s();
        for (int i=0;i<count;++i) {
            const auto& e=rows[i];
            if (e.vehicle<0 || size_t(e.vehicle)>=calendars.size() || seen[e.vehicle] ||
                !std::isfinite(e.planned) || !std::isfinite(e.period) || e.period<.020 ||
                !std::isfinite(e.finished) || !std::isfinite(e.state_end))
                throw std::invalid_argument("invalid guarded group entry");
            seen[e.vehicle]=true;
            const auto& c=calendars[e.vehicle];
            // submitted/started carry the OLD release/period; version is the
            // expected old version. sim_ns carries the proposed new version.
            if (!c.active || c.id!=e.id || c.version!=e.version ||
                std::abs(c.next-e.submitted)>1e-9 || std::abs(c.period-e.started)>1e-9) return 1;
            if (c.command_pending || c.control_token!=e.collision_ns) return 2;
            if (now>=e.state_end || e.planned<=now || e.planned+1e-9<c.last+.020 || e.planned>=e.finished) return 3;
        }
        if (commit) {
            for (int i=0;i<count;++i) {
                const auto& e=rows[i];auto& c=calendars[e.vehicle];
                c.version=e.sim_ns;c.next=e.planned;c.period=e.period;c.end=e.finished;
                c.active=c.next+1e-9<c.end;
            }
            SetEvent(changed);
        }
        return 0;
    }
    void cancel(int index) {
        std::lock_guard<std::mutex> lock(calendar_mutex); calendars.at(index).active = false; SetEvent(changed);
    }
    void release_loop() {
        try {
            pin();
            while (!stopping) {
                double nearest = std::numeric_limits<double>::infinity();
                { std::lock_guard<std::mutex> lock(calendar_mutex);
                    const double now = clock_s();
                    for (size_t i = 0; i < calendars.size(); ++i) {
                        auto& c = calendars[i];
                        while (c.active && c.next <= now) {
                            Event event{}; event.kind = 4; event.vehicle = int(i); event.id = c.id++;
                            event.version = c.version; event.planned = c.next; event.finished = clock_s(); event.period = c.period;
                            push(event); c.last = c.next; c.next += c.period; c.active = c.next+1e-9 < c.end;
                        }
                        if (c.active) nearest = std::min(nearest, c.next);
                    }
                }
                if (!std::isfinite(nearest)) WaitForSingleObject(changed, INFINITE);
                else {
                    LARGE_INTEGER due; due.QuadPart = -std::max<LONGLONG>(1, LONGLONG((nearest - clock_s()) * 1e7));
                    if (!SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE)) throw std::runtime_error("release timer failed");
                    HANDLE handles[] = {changed, timer}; WaitForMultipleObjects(2, handles, FALSE, INFINITE);
                }
            }
        } catch (const std::exception& e) { Event error{}; error.kind = 5; copy_text(error.error, e.what()); push(error); }
    }
    void stop() {
        stopping = true; if (changed) SetEvent(changed);
        for (auto& loop : loops) loop->wake.notify_one();
        if (releaser.joinable()) releaser.join();
        for (auto& loop : loops) if (loop->worker.joinable()) loop->worker.join();
    }
};

static thread_local std::string last_error;
#define API extern "C" __declspec(dllexport)
#define TRY try
#define FAIL catch (const std::exception& e) { last_error = e.what(); return -1; }
API const char* falcon_error() { return last_error.c_str(); }
API double falcon_clock() { return clock_s(); }
API size_t falcon_event_size() { return sizeof(Event); }
API void* falcon_create(const char* host, int port, const char* names, uint64_t mask,
    const double* cal, double bound, int late, int mock, double delay) {
    try { return new Runtime(host, port, names, mask, cal, bound, late != 0, mock != 0, delay); }
    catch (const std::exception& e) { last_error = e.what(); return nullptr; }
}
API uintptr_t falcon_handle(Runtime* runtime) { return reinterpret_cast<uintptr_t>(runtime->completion); }
API int falcon_poll(Runtime* runtime, Event* event) { TRY { return runtime->poll(*event); } FAIL }
API int falcon_sample(Runtime* runtime, int vehicle, uint64_t id) {
    TRY { Task task; task.id = id; runtime->submit(vehicle, task); return 0; } FAIL
}
API int falcon_send(Runtime* runtime, int vehicle, uint64_t id, const double* control, double deadline) {
    TRY { Task task; task.command = true; task.id = id; task.deadline = deadline;
        const double lower[] = {2., -3., -3., -2.}, upper[] = {20., 3., 3., 2.};
        for (size_t i = 0; i < 4; ++i) { if (!std::isfinite(control[i])) throw std::invalid_argument("nonfinite action");
            task.control[i] = std::clamp(control[i], lower[i], upper[i]); }
        runtime->submit(vehicle, task); return 0; } FAIL
}
API int falcon_schedule(Runtime* runtime, int vehicle, uint64_t id, uint64_t version, double next, double period, double end, Event* installed) {
    TRY { const auto c=runtime->schedule(vehicle, id, version, next, period, end);
        *installed=Event{};installed->id=c.id;installed->planned=c.next;return 0; } FAIL
}
API int falcon_schedule_batch(Runtime* runtime, Event* changes, int count) {
    TRY { return runtime->schedule_batch(changes, count) ? 0 : 1; } FAIL
}
API int falcon_calendar_matches(Runtime* runtime, Event* expected, int count) {
    TRY { return runtime->calendar_matches(expected, count) ? 0 : 1; } FAIL
}
API int falcon_guarded_batch(Runtime* runtime, Event* expected, int count, int commit) {
    TRY { return runtime->guarded_batch(expected,count,commit!=0); } FAIL
}
API int falcon_cancel(Runtime* runtime, int vehicle) { TRY { runtime->cancel(vehicle); return 0; } FAIL }
API void falcon_stop(Runtime* runtime) { runtime->stop(); }
API void falcon_destroy(Runtime* runtime) { delete runtime; }
