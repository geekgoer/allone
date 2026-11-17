#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Small native MessagePack-RPC binding for the frozen AirSim wire contract.
// No inference, feature encoding, scheduling, scene policy, or clipping here.
namespace psairsim {

std::int64_t monotonic_ns();

struct Collision {
    bool has_collided = false;
    std::uint64_t timestamp_ns = 0;
    std::string object_name;
    int object_id = -1;
    double penetration_depth = 0;
};

struct State {
    // Per-vehicle local NED; quaternion is w,x,y,z. Rates are body-frame.
    std::array<double, 3> position{{0, 0, 0}};
    std::array<double, 3> velocity{{0, 0, 0}};
    std::array<double, 3> acceleration{{0, 0, 0}};
    std::array<double, 3> angular_velocity{{0, 0, 0}};
    std::array<double, 3> angular_acceleration{{0, 0, 0}};
    std::array<double, 4> orientation_wxyz{{1, 0, 0, 0}};
    std::uint64_t timestamp_ns = 0;  // Simulation clock, never subtract PS clock.
    int landed_state = 0;
    Collision collision;
};

struct CommandProxy {
    std::int64_t submit_ns = 0;
    std::int64_t enqueued_ns = 0;
    std::int64_t ping_start_ns = 0;
    std::int64_t ping_return_ns = 0;
    bool ping_ok = false;
};

struct Pose {
    std::array<double, 3> position{{0, 0, 0}};
    std::array<double, 4> orientation_wxyz{{1, 0, 0, 0}};
};

class Client {
public:
    Client(const std::string& host, std::uint16_t port = 41451,
           std::int64_t timeout_ms = 1000);
    ~Client();
    Client(const Client&) = delete;
    Client& operator=(const Client&) = delete;
    bool ping();
    std::vector<std::string> list_vehicles();
    State get_state(const std::string& vehicle);
    Collision get_collision(const std::string& vehicle);
    CommandProxy move_rates_throttle(const std::string& vehicle, float wx,
                                    float wy, float wz, float throttle,
                                    float duration);
    // Setup/teardown only. Never invoke scene-management calls in the hot path.
    void pause(bool paused);
    bool is_paused();
    void enable_api(const std::string& vehicle, bool enabled);
    bool arm(const std::string& vehicle, bool armed);
    void set_pose(const std::string& vehicle,
                  const std::array<double, 3>& position,
                  const std::array<double, 4>& orientation_wxyz,
                  bool ignore_collision = false);
    void set_kinematics(const std::string& vehicle, const State& state,
                        bool ignore_collision = false);
    State get_ground_truth_kinematics(const std::string& vehicle);
    Pose get_object_pose(const std::string& object);
    Pose get_vehicle_pose(const std::string& vehicle);
    std::vector<std::string> list_scene_objects(const std::string& regex);
    std::vector<std::string> list_assets();
    std::string spawn_object(const std::string& name, const std::string& asset,
                             const Pose& pose,
                             const std::array<double, 3>& scale,
                             bool physics_enabled = false,
                             bool is_blueprint = false);
    // A nonempty experiment-owned prefix is required before deleting actors.
    bool destroy_object(const std::string& name,
                        const std::string& owned_prefix = "PSMIG_");
    void reset();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace psairsim
