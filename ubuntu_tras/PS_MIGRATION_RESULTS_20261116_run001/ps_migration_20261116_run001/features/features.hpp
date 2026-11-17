#pragma once

// Native port of the frozen Windows numerical contract, not a new controller.
#include <array>
#include <cstdint>
#include <string>
#include <vector>
#include "json.hpp"

namespace psfeatures {
using Vec3 = std::array<double, 3>;
using Quat = std::array<double, 4>;  // w, x, y, z in local AirSim NED.
using Control = std::array<double, 4>;  // physical F, wx, wy, wz.
using Raw237 = std::array<float, 237>;

struct State {
    Vec3 position{{0.,0.,0.}}, velocity{{0.,0.,0.}};
    Quat quaternion{{1.,0.,0.,0.}};
    Vec3 acceleration{{0.,0.,0.}}, angular_velocity{{0.,0.,0.}};
    std::uint64_t timestamp_ns = 0;
    static State from_json(const nlohmann::json&);
};
struct Reference {
    Vec3 position{{0.,0.,0.}}, velocity{{0.,0.,0.}}, acceleration{{0.,0.,0.}};
    Quat orientation{{1.,0.,0.,0.}};
    Vec3 body_rates{{0.,0.,0.}};
    static Reference from_json(const nlohmann::json&);
};
Quat normalize_quaternion(const Quat&);
Vec3 rotate_vector(const Quat&, const Vec3&);
struct Segment {
    double start_time = 0., duration = 0.;
    std::array<std::array<double,6>,3> coefficients;
    Quat heading_orientation;
};
class Trajectory {
public:
    static Trajectory from_layout(const nlohmann::json&);
    Reference evaluate(double elapsed) const;
    std::array<Reference,21> window(double elapsed, double dt = .1,
                                    bool include_body_rates = false) const;
    double duration() const { return duration_; }
private:
    std::vector<Segment> segments_;
    double duration_ = 0.;
};

Raw237 encode(const State&, const std::array<Reference,21>&);
// Variable reference sequence implements Python truncation/repeat-last behavior.
Raw237 encode(const State&, const std::vector<Reference>&);
struct Calibration {
    // Fallback matches config.py, but production MUST load the supplied file
    // (this frozen package has hover_throttle=0.6, not this fallback 0.5).
    double hover_throttle = .5, hover_thrust_accel = 9.81;
    double thrust_accel_per_throttle = 19.62;
    double min_throttle = 0., max_throttle = 1.;
    bool calibration_complete = false;
    void validate() const;
    static Calibration from_json(const nlohmann::json&);
};
struct Command {
    double thrust_accel, roll_rate, pitch_rate, yaw_rate, throttle, duration;
};
struct PreparedCommand {
    Control requested, limited, effective;
    Command command;
};
double thrust_to_throttle(double, const Calibration&);
double throttle_to_thrust(double, const Calibration&);
Command convert_command(const Control&, const Calibration&, double duration = 1.);
// Original fly_ip37_joint limits physical commands before converting to AirSim.
// The caller may commit effective as held control ONLY after actual submission.
PreparedCommand prepare_live_command(const Control&, const Calibration&, double duration = 1.);
// C++ AirLib/native RPC does NOT do Python client's second pitch/yaw flip.
// Keep convert_command for Python-layer fixture parity; use this at native wire.
Command native_rpc_command(const PreparedCommand&);

struct HoldBounds {
    double acceleration_error_mps2 = 2., position_error_m = .01;
    double velocity_error_mps = .05, maximum_hold_s = .15, barrier_rate = 10.;
    std::string evidence = "Route-B development assumptions; not plant-certified";
    bool plant_bounds_verified = false;
    double gravity = 9.80665;
    void validate() const;
};
struct Gate {
    int index = 0;
    Vec3 center;
    Quat orientation;
    static Gate from_json(const nlohmann::json&);
};
struct HoldAnalysis {
    double budget_s = 0.;
    bool valid = false, conditional_barrier_certificate = false;
    bool plant_bounds_verified = false, is_safety_certificate = false;
    bool uses_control_in_bound = true;
    std::string binding_obstacle;
    Vec3 nominal_acceleration;
    std::vector<double> per_obstacle_period_s;
    std::vector<int> separating_face;
    // Same diagnostic mesh quantities as GateHoldBatch.analyze.
    double mesh_margin_m = 0., mesh_rate_mps = 0.;
};
Vec3 nominal_acceleration(const Quat&, const Control&, double gravity = 9.80665);
double barrier_period(double distance, double speed, double acceleration,
                      double jerk, const HoldBounds&);
class GateHoldAnalyzer {
public:
    // net_width/net_height are CLEAR aperture sizes; callers of the original
    // live chain use case center width/height minus thickness (.15 m).
    GateHoldAnalyzer(const std::vector<Gate>&, double net_width, const HoldBounds&,
                     double net_height = 4., double thickness = .15,
                     double body_radius = .50, double clearance = .025);
    HoldAnalysis analyze(const State&, const Control& effective_held) const;
private:
    struct Box { std::array<Vec3,6> normals; std::array<double,6> offsets; std::string label; };
    std::vector<Box> boxes_;
    HoldBounds bounds_;
};
struct SafeDeadline {
    double deadline_s, remaining_s, physical_deadline_s, state_age_s;
    bool feasible, is_safety_certificate;
};
SafeDeadline safe_deadline(double sample_s, double now_s, double actuation_s,
                           const HoldAnalysis&, double maximum_update_s,
                           double actuation_time_error_s = 0.);
}  // namespace psfeatures
