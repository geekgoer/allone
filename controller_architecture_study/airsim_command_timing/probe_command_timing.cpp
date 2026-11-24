#include <array>
#include <chrono>
#include <cmath>
#include <future>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>

#include "api/ApiProvider.hpp"
#include "vehicles/multirotor/api/MultirotorRpcLibClient.hpp"
#include "vehicles/multirotor/api/MultirotorRpcLibServer.hpp"
#include "vehicles/multirotor/firmwares/simple_flight/SimpleFlightQuadXParams.hpp"

using namespace std::chrono_literals;
namespace air = msr::airlib;
using Clock = std::chrono::steady_clock;

struct CommandTrace {
    float throttle = 0;
    std::chrono::milliseconds delay_before_goal{0};
    Clock::time_point first_callback, goal_set, first_motor_update;
    std::array<float, 4> motors{};
};

// The bench has one command in flight. Each command uses its own throttle so
// matching requires both the callback payload and the simulated motor outputs.
class MeasuredSimpleFlight : public air::SimpleFlightApi {
public:
    using air::SimpleFlightApi::SimpleFlightApi;

    void expect(float throttle, std::chrono::milliseconds delay) {
        std::lock_guard<std::mutex> lock(mutex_);
        trace_ = CommandTrace{};
        trace_.throttle = throttle;
        trace_.delay_before_goal = delay;
    }

    CommandTrace trace() {
        std::lock_guard<std::mutex> lock(mutex_);
        return trace_;
    }

    void update() override {
        std::lock_guard<std::mutex> lock(mutex_);
        air::SimpleFlightApi::update();
        if (trace_.goal_set == Clock::time_point{} || trace_.first_motor_update != Clock::time_point{})
            return;
        for (unsigned int rotor = 0; rotor < trace_.motors.size(); ++rotor) {
            trace_.motors[rotor] = getActuation(rotor);
            if (std::abs(trace_.motors[rotor] - trace_.throttle) > 1e-5f)
                return;
        }
        trace_.first_motor_update = Clock::now();
    }

protected:
    void commandAngleRatesThrottle(float roll, float pitch, float yaw, float throttle) override {
        std::unique_lock<std::mutex> lock(mutex_);
        if (throttle != trace_.throttle || roll != 0 || pitch != 0 || yaw != 0)
            throw std::runtime_error("Unexpected command payload");
        if (trace_.first_callback == Clock::time_point{}) {
            trace_.first_callback = Clock::now();
            const auto delay = trace_.delay_before_goal;
            lock.unlock();
            if (delay > 0ms)
                std::this_thread::sleep_for(delay);
            lock.lock();
        }
        air::SimpleFlightApi::commandAngleRatesThrottle(roll, pitch, yaw, throttle);
        if (trace_.goal_set == Clock::time_point{})
            trace_.goal_set = Clock::now();
    }

private:
    std::mutex mutex_;
    CommandTrace trace_;
};

double elapsed_ms(Clock::time_point start, Clock::time_point event) {
    return std::chrono::duration<double, std::milli>(event - start).count();
}

void collect(MeasuredSimpleFlight& api, std::uint16_t port) {
    air::MultirotorRpcLibClient client("127.0.0.1", port, 2.0f);
    if (!client.ping())
        throw std::runtime_error("Initial ping failed");

    struct Scenario { const char* name; std::chrono::milliseconds before_goal, after_ping; };
    const std::array<Scenario, 3> scenarios{{
        {"normal", 0ms, 0ms}, {"delay_before_goal", 40ms, 0ms}, {"delay_after_ping", 0ms, 40ms},
    }};
    std::cout << "repeat,scenario,throttle,callback_ms,goal_set_ms,motor_update_ms,ping_return_ms,"
                 "reported_ack_ms,motor_0,motor_1,motor_2,motor_3\n";
    std::cout << std::fixed << std::setprecision(6);
    int sequence = 0;
    for (int repeat = 0; repeat < 12; ++repeat) {
        for (std::size_t i = 0; i < scenarios.size(); ++i) {
            const auto& scenario = scenarios[(i + repeat) % scenarios.size()];
            const float throttle = 0.40f + 0.005f * ++sequence;
            api.expect(throttle, scenario.before_goal);
            const auto submit = Clock::now();
            client.moveByAngleRatesThrottleAsync(0, 0, 0, throttle, 0.06f, "Probe");
            if (!client.ping())
                throw std::runtime_error("Command ping failed");
            const auto ping = Clock::now();
            if (scenario.after_ping > 0ms)
                std::this_thread::sleep_for(scenario.after_ping);
            const auto reported = Clock::now();
            bool completed = false;
            client.waitOnLastTask(&completed, 1.0f);
            if (!completed)
                throw std::runtime_error("Movement RPC did not complete");
            const auto limit = Clock::now() + 150ms;
            auto result = api.trace();
            while (result.first_motor_update == Clock::time_point{} && Clock::now() < limit) {
                std::this_thread::sleep_for(1ms);
                result = api.trace();
            }
            if (result.first_callback < submit || result.goal_set < result.first_callback ||
                result.first_motor_update < result.goal_set)
                throw std::runtime_error("Missing or out-of-order SimpleFlight events");
            std::cout << repeat + 1 << ',' << scenario.name << ',' << throttle << ','
                      << elapsed_ms(submit, result.first_callback) << ',' << elapsed_ms(submit, result.goal_set) << ','
                      << elapsed_ms(submit, result.first_motor_update) << ',' << elapsed_ms(submit, ping) << ','
                      << elapsed_ms(submit, reported);
            for (const auto motor : result.motors)
                std::cout << ',' << motor;
            std::cout << '\n';
        }
    }
}

int main(int argc, char** argv) try {
    const int port = argc == 2 ? std::stoi(argv[1]) : 41590;
    if (argc > 2 || port < 1024 || port > 65535 || port == 41451)
        throw std::invalid_argument("Specify one local diagnostic port other than 41451");

    air::ClockFactory::get();
    air::AirSimSettings::VehicleSetting settings("Probe", "SimpleFlight");
    air::SimpleFlightQuadXParams parameters(&settings, std::make_shared<air::SensorFactory>());
    parameters.initialize(&settings);
    MeasuredSimpleFlight api(&parameters, &settings);
    auto kinematics = air::Kinematics::State::zero();
    air::Environment environment(air::Environment::State{});
    api.setSimulatedGroundTruth(&kinematics, &environment);
    api.reset();
    api.enableApiControl(true);
    if (!api.isApiControlEnabled() || !api.armDisarm(true))
        throw std::runtime_error("Could not enable headless SimpleFlight");

    air::ApiProvider provider(nullptr);
    provider.insert_or_assign("Probe", &api, nullptr);
    air::MultirotorRpcLibServer server(&provider, "127.0.0.1", static_cast<std::uint16_t>(port));
    server.start(false, 4);
    auto results = std::async(std::launch::async, [&] { collect(api, static_cast<std::uint16_t>(port)); });
    while (results.wait_for(0ms) != std::future_status::ready) {
        api.update();
        std::this_thread::sleep_for(2ms);
    }
    results.get();
    server.stop();
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
