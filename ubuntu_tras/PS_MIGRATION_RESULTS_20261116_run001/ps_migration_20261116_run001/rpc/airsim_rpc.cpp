#include "airsim_rpc.hpp"

#include <rpc/client.h>
#include <rpc/msgpack.hpp>
#include <chrono>
#include <cmath>
#include <future>
#include <stdexcept>
#include <time.h>

namespace psairsim {
namespace wire {
// Map names and RPC arguments match the immutable AirSim snapshot:
// AirLib/include/api/RpcLibAdaptorsBase.hpp and MultirotorRpcLibAdaptors.hpp.
struct Vec {
    float x_val = 0, y_val = 0, z_val = 0;
    MSGPACK_DEFINE_MAP(x_val, y_val, z_val);
};
struct Quat {
    float w_val = 1, x_val = 0, y_val = 0, z_val = 0;
    MSGPACK_DEFINE_MAP(w_val, x_val, y_val, z_val);
};
struct Pose {
    Vec position;
    Quat orientation;
    MSGPACK_DEFINE_MAP(position, orientation);
};
struct Kinematics {
    Vec position, linear_velocity, angular_velocity;
    Vec linear_acceleration, angular_acceleration;
    Quat orientation;
    MSGPACK_DEFINE_MAP(position, orientation, linear_velocity, angular_velocity,
                       linear_acceleration, angular_acceleration);
};
struct CollisionInfo {
    bool has_collided = false;
    float penetration_depth = 0;
    std::uint64_t time_stamp = 0;
    Vec normal, impact_point, position;
    std::string object_name;
    int object_id = -1;
    MSGPACK_DEFINE_MAP(has_collided, penetration_depth, time_stamp, normal,
                       impact_point, position, object_name, object_id);
};
struct Multirotor {
    Kinematics kinematics_estimated;
    CollisionInfo collision;
    std::uint64_t timestamp = 0;
    int landed_state = 0;
    MSGPACK_DEFINE_MAP(kinematics_estimated, collision, timestamp, landed_state);
};

std::array<double, 3> array(const Vec& v) {
    return {{v.x_val, v.y_val, v.z_val}};
}
Vec vec(const std::array<double, 3>& v) {
    Vec r;
    r.x_val = static_cast<float>(v[0]);
    r.y_val = static_cast<float>(v[1]);
    r.z_val = static_cast<float>(v[2]);
    return r;
}
Quat quat(const std::array<double, 4>& q) {
    Quat r;
    r.w_val = static_cast<float>(q[0]); r.x_val = static_cast<float>(q[1]);
    r.y_val = static_cast<float>(q[2]); r.z_val = static_cast<float>(q[3]);
    return r;
}
Collision collision(const CollisionInfo& c) {
    Collision r;
    r.has_collided = c.has_collided; r.timestamp_ns = c.time_stamp;
    r.object_name = c.object_name; r.object_id = c.object_id;
    r.penetration_depth = c.penetration_depth;
    return r;
}
State state(const Kinematics& k) {
    State r;
    r.position = array(k.position); r.velocity = array(k.linear_velocity);
    r.acceleration = array(k.linear_acceleration);
    r.angular_velocity = array(k.angular_velocity);
    r.angular_acceleration = array(k.angular_acceleration);
    r.orientation_wxyz = {{k.orientation.w_val, k.orientation.x_val,
                          k.orientation.y_val, k.orientation.z_val}};
    return r;
}
Kinematics kinematics(const State& s) {
    Kinematics k;
    k.position = vec(s.position); k.linear_velocity = vec(s.velocity);
    k.linear_acceleration = vec(s.acceleration);
    k.angular_velocity = vec(s.angular_velocity);
    k.angular_acceleration = vec(s.angular_acceleration);
    k.orientation = quat(s.orientation_wxyz);
    return k;
}
void require(const RPCLIB_MSGPACK::object& obj, const char* key) {
    if (obj.type != RPCLIB_MSGPACK::type::MAP)
        throw std::runtime_error("AirSim RPC response is not a map");
    for (std::uint32_t i = 0; i < obj.via.map.size; ++i) {
        const auto& field = obj.via.map.ptr[i];
        if (field.key.type == RPCLIB_MSGPACK::type::STR &&
            field.key.as<std::string>() == key) return;
    }
    throw std::runtime_error(std::string("AirSim RPC response missing ") + key);
}
void finite_state(const State& s) {
    const auto check = [](const auto& values) {
        for (const auto value : values)
            if (!std::isfinite(value))
                throw std::runtime_error("Nonfinite AirSim state value");
    };
    check(s.position); check(s.velocity); check(s.acceleration);
    check(s.angular_velocity); check(s.angular_acceleration);
    check(s.orientation_wxyz);
    for (const auto& arr : {s.position, s.velocity, s.acceleration,
                           s.angular_velocity, s.angular_acceleration})
        for (double v : arr)
            if (!std::isfinite(v)) throw std::runtime_error("Non-finite AirSim state");
    double norm = 0;
    for (double v : s.orientation_wxyz) {
        if (!std::isfinite(v)) throw std::runtime_error("Non-finite AirSim quaternion");
        norm += v * v;
    }
    // Validate, never silently renormalize wire data.
    if (norm < 0.5 || norm > 1.5)
        throw std::runtime_error("Invalid AirSim quaternion norm");
}
}  // namespace wire

std::int64_t monotonic_ns() {
    timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts))
        throw std::runtime_error("clock_gettime(CLOCK_MONOTONIC) failed");
    return std::int64_t(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
}

struct Client::Impl {
    rpc::client rpc;
    std::future<RPCLIB_MSGPACK::object_handle> last_move;
    Impl(const std::string& host, std::uint16_t port, std::int64_t timeout_ms)
        : rpc(host, port) { rpc.set_timeout(timeout_ms); }
};
Client::Client(const std::string& host, std::uint16_t port, std::int64_t timeout_ms)
    : impl_(new Impl(host, port, timeout_ms)) {}
Client::~Client() = default;
bool Client::ping() { return impl_->rpc.call("ping").as<bool>(); }
std::vector<std::string> Client::list_vehicles() {
    return impl_->rpc.call("listVehicles").as<std::vector<std::string>>();
}
State Client::get_state(const std::string& vehicle) {
    auto response = impl_->rpc.call("getMultirotorState", vehicle);
    wire::require(response.get(), "kinematics_estimated");
    wire::require(response.get(), "timestamp");
    const auto w = response.as<wire::Multirotor>();
    auto s = wire::state(w.kinematics_estimated);
    s.timestamp_ns = w.timestamp; s.landed_state = w.landed_state;
    s.collision = wire::collision(w.collision);
    wire::finite_state(s);
    return s;
}
Collision Client::get_collision(const std::string& vehicle) {
    auto response = impl_->rpc.call("simGetCollisionInfo", vehicle);
    wire::require(response.get(), "has_collided");
    return wire::collision(response.as<wire::CollisionInfo>());
}
CommandProxy Client::move_rates_throttle(const std::string& vehicle, float wx,
                                        float wy, float wz, float throttle,
                                        float duration) {
    if (!std::isfinite(wx) || !std::isfinite(wy) || !std::isfinite(wz) ||
        !std::isfinite(throttle) || !std::isfinite(duration) ||
        throttle < 0 || throttle > 1 || duration <= 0)
        throw std::invalid_argument("Invalid rates/throttle/duration; caller must adapt controls");
    CommandProxy p;
    // Mirror official AirLib/Python async semantics: do NOT join move duration.
    // Sign changes, if any, belong in the frozen control adapter, not this layer.
    p.submit_ns = monotonic_ns();
    impl_->last_move = impl_->rpc.async_call("moveByAngleRatesThrottle", wx, wy,
                                            wz, throttle, duration, vehicle);
    p.enqueued_ns = monotonic_ns();
    p.ping_start_ns = monotonic_ns();
    p.ping_ok = ping();
    p.ping_return_ns = monotonic_ns();
    // This is only an RPC-return proxy, NOT a physical actuation timestamp.
    return p;
}
void Client::pause(bool paused) { impl_->rpc.call("simPause", paused); }
bool Client::is_paused() { return impl_->rpc.call("simIsPaused").as<bool>(); }
void Client::enable_api(const std::string& vehicle, bool enabled) {
    impl_->rpc.call("enableApiControl", enabled, vehicle);
}
bool Client::arm(const std::string& vehicle, bool armed) {
    return impl_->rpc.call("armDisarm", armed, vehicle).as<bool>();
}
void Client::set_pose(const std::string& vehicle,
                     const std::array<double, 3>& position,
                     const std::array<double, 4>& orientation,
                     bool ignore_collision) {
    wire::Pose p;
    p.position = wire::vec(position); p.orientation = wire::quat(orientation);
    impl_->rpc.call("simSetVehiclePose", p, ignore_collision, vehicle);
}
void Client::set_kinematics(const std::string& vehicle, const State& state,
                           bool ignore_collision) {
    wire::finite_state(state);
    impl_->rpc.call("simSetKinematics", wire::kinematics(state), ignore_collision,
                    vehicle);
}
State Client::get_ground_truth_kinematics(const std::string& vehicle) {
    auto response = impl_->rpc.call("simGetGroundTruthKinematics", vehicle);
    wire::require(response.get(), "position");
    auto state = wire::state(response.as<wire::Kinematics>());
    wire::finite_state(state);
    return state;
}
void Client::reset() { impl_->rpc.call("reset"); }

Pose Client::get_object_pose(const std::string& object) {
    auto response = impl_->rpc.call("simGetObjectPose", object);
    wire::require(response.get(), "position");
    wire::require(response.get(), "orientation");
    const auto pose = response.as<wire::Pose>();
    Pose r;
    r.position = wire::array(pose.position);
    r.orientation_wxyz = {{pose.orientation.w_val, pose.orientation.x_val,
                          pose.orientation.y_val, pose.orientation.z_val}};
    // An unknown actor may return NaNs. Preserve that explicit failure.
    for (auto value : r.position)
        if (!std::isfinite(value)) throw std::runtime_error("Object pose is nonfinite: " + object);
    for (auto value : r.orientation_wxyz)
        if (!std::isfinite(value)) throw std::runtime_error("Object quaternion is nonfinite: " + object);
    return r;
}
std::vector<std::string> Client::list_scene_objects(const std::string& regex) {
    return impl_->rpc.call("simListSceneObjects", regex).as<std::vector<std::string>>();
}
Pose Client::get_vehicle_pose(const std::string& vehicle) {
    auto response = impl_->rpc.call("simGetVehiclePose", vehicle);
    wire::require(response.get(), "position");
    wire::require(response.get(), "orientation");
    const auto p = response.as<wire::Pose>();
    Pose r;
    r.position = wire::array(p.position);
    r.orientation_wxyz = {{p.orientation.w_val, p.orientation.x_val,
                          p.orientation.y_val, p.orientation.z_val}};
    for (auto value : r.position)
        if (!std::isfinite(value)) throw std::runtime_error("Vehicle pose is nonfinite: " + vehicle);
    return r;
}
std::vector<std::string> Client::list_assets() {
    return impl_->rpc.call("simListAssets").as<std::vector<std::string>>();
}
std::string Client::spawn_object(const std::string& name, const std::string& asset,
                                  const Pose& pose,
                                  const std::array<double, 3>& scale,
                                  bool physics_enabled, bool is_blueprint) {
    wire::Pose p;
    p.position = wire::vec(pose.position);
    p.orientation = wire::quat(pose.orientation_wxyz);
    for (auto value : scale)
        if (!std::isfinite(value) || value <= 0)
            throw std::invalid_argument("Spawn scale must be finite and positive");
    return impl_->rpc.call("simSpawnObject", name, asset, p, wire::vec(scale),
                           physics_enabled, is_blueprint).as<std::string>();
}
bool Client::destroy_object(const std::string& name, const std::string& owned_prefix) {
    if (owned_prefix.empty() || name.compare(0, owned_prefix.size(), owned_prefix) != 0)
        throw std::invalid_argument("Refusing deletion outside explicit owned actor prefix");
    return impl_->rpc.call("simDestroyObject", name).as<bool>();
}
}  // namespace psairsim
