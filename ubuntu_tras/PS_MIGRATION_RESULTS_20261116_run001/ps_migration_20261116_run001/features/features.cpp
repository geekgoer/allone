#include "features.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace psfeatures {
Command native_rpc_command(const PreparedCommand& prepared) {
    Command c=prepared.command;
    c.roll_rate=prepared.effective[1];
    c.pitch_rate=prepared.effective[2];
    c.yaw_rate=prepared.effective[3];
    return c;
}
namespace {
constexpr double kFeatureGravity = 9.8066;
constexpr double kReferenceGravity = 9.8066;
constexpr double kBodyRateDifferenceDt = .001;
void require(bool ok, const char* reason) { if (!ok) throw std::invalid_argument(reason); }
template<std::size_t N> bool finite(const std::array<double,N>& a) {
    for (double x : a) if (!std::isfinite(x)) return false;
    return true;
}
template<std::size_t N> double norm(const std::array<double,N>& a) {
    double r = 0.; for (double x : a) r += x*x; return std::sqrt(r);
}
template<std::size_t N> double dot(const std::array<double,N>& a, const std::array<double,N>& b) {
    double r = 0.; for (std::size_t i=0;i<N;++i) r += a[i]*b[i]; return r;
}
template<std::size_t N> std::array<double,N> from_array(const nlohmann::json& j) {
    require(j.is_array() && j.size() == N,"numeric array shape mismatch");
    std::array<double,N> a;
    for (std::size_t i=0;i<N;++i) a[i]=j.at(i).get<double>();
    require(finite(a),"nonfinite numeric array"); return a;
}
Vec3 cross(const Vec3& a,const Vec3& b) {
    return {{a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]}};
}
Vec3 add(const Vec3& a,const Vec3& b) { return {{a[0]+b[0],a[1]+b[1],a[2]+b[2]}}; }
Vec3 sub(const Vec3& a,const Vec3& b) { return {{a[0]-b[0],a[1]-b[1],a[2]-b[2]}}; }
Vec3 scale(const Vec3& a,double s) { return {{a[0]*s,a[1]*s,a[2]*s}}; }
Vec3 flip(const Vec3& a) { return {{a[0],-a[1],-a[2]}}; }
Quat flip(const Quat& a) { return {{a[0],a[1],-a[2],-a[3]}}; }
Vec3 unit(const Vec3& a) { double n=norm(a); require(n>=1e-12,"degenerate attitude axes"); return scale(a,1./n); }
Quat neg(Quat a) { for(double& v:a)v=-v; return a; }
Quat multiply(const Quat& a,const Quat& b) {
    return {{a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3],
             a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2],
             a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1],
             a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0]}};
}
Quat matrix_quaternion(const std::array<Vec3,3>& m) {
    Quat q;
    double tr=m[0][0]+m[1][1]+m[2][2];
    if(tr>0.) {
        double s=2.*std::sqrt(tr+1.);
        q={{.25*s,(m[2][1]-m[1][2])/s,(m[0][2]-m[2][0])/s,(m[1][0]-m[0][1])/s}};
    } else if(m[0][0]>=m[1][1] && m[0][0]>=m[2][2]) {
        double s=2.*std::sqrt(std::max(1.+m[0][0]-m[1][1]-m[2][2],0.));
        q={{(m[2][1]-m[1][2])/s,.25*s,(m[0][1]+m[1][0])/s,(m[0][2]+m[2][0])/s}};
    } else if(m[1][1]>=m[2][2]) {
        double s=2.*std::sqrt(std::max(1.+m[1][1]-m[0][0]-m[2][2],0.));
        q={{(m[0][2]-m[2][0])/s,(m[0][1]+m[1][0])/s,.25*s,(m[1][2]+m[2][1])/s}};
    } else {
        double s=2.*std::sqrt(std::max(1.+m[2][2]-m[0][0]-m[1][1],0.));
        q={{(m[1][0]-m[0][1])/s,(m[0][2]+m[2][0])/s,(m[1][2]+m[2][1])/s,.25*s}};
    }
    return normalize_quaternion(q);
}
Quat attitude(const Vec3& acceleration, const Quat& heading) {
    Quat q=normalize_quaternion(flip(heading));
    Vec3 forward={{1.-2.*(q[2]*q[2]+q[3]*q[3]),2.*(q[1]*q[2]+q[3]*q[0]),0.}};
    if(norm(forward)<1e-9)forward={{1.,0.,0.}};
    forward=unit(forward);
    Vec3 bz=add(flip(acceleration),{{0.,0.,kReferenceGravity}});
    bz=scale(bz,1./std::max(norm(bz),1e-12));
    Vec3 by=cross(bz,forward);
    if(norm(by)<1e-9)by=cross(bz,{{0.,1.,0.}});
    by=unit(by); Vec3 bx=unit(cross(by,bz));
    std::array<Vec3,3> matrix;
    for(std::size_t i=0;i<3;++i)matrix[i]={{bx[i],by[i],bz[i]}};
    return normalize_quaternion(flip(matrix_quaternion(matrix)));
}
Vec3 body_rates(const Quat& start,const Quat& following,double dt) {
    Quat q0=normalize_quaternion(start),q1=normalize_quaternion(following);
    if(dot(q0,q1)<0.)q1=neg(q1);
    Quat inv={{q0[0],-q0[1],-q0[2],-q0[3]}};
    Quat delta=normalize_quaternion(multiply(inv,q1));
    if(delta[0]<0.)delta=neg(delta);
    Vec3 v={{delta[1],delta[2],delta[3]}};double len=norm(v);
    if(len<1e-12)return {{0.,0.,0.}};
    double angle=2.*std::atan2(len,std::max(-1.,std::min(1.,delta[0])));
    return scale(v,angle/(len*dt));
}
double clip(double x,double lo,double hi) { return std::max(lo,std::min(hi,x)); }
struct Barrier { std::array<double,4> c; bool valid; };
Barrier coefficients(double d,double v,double a,double j,const HoldBounds& b) {
    require(std::isfinite(d)&&std::isfinite(v)&&std::isfinite(a)&&std::isfinite(j)&&j>=0.,
            "finite barrier data and nonnegative jerk required");
    double h=d-b.position_error_m,w=v-b.velocity_error_mps,z=a-b.acceleration_error_mps2,k=b.barrier_rate;
    Barrier q; q.c={{z+2.*k*w+k*k*h-1e-10,-j+2.*k*z+k*k*w,-k*j+.5*k*k*z,-k*k*j/6.}};
    for(std::size_t i=1;i<4;++i)q.c[i]=std::min(q.c[i],0.);
    q.valid=h>0. && w+k*h>=0. && q.c[0]>0.; return q;
}
double polynomial(const Barrier& q,double t) {return ((q.c[3]*t+q.c[2])*t+q.c[1])*t+q.c[0];}
double bisect_period(const Barrier& q,double cap) {
    if(!q.valid)return 0.;
    double lo=0.,hi=cap;
    for(int i=0;i<24;++i){double t=(lo+hi)/2.;if(polynomial(q,t)>=0.)lo=t;else hi=t;}
    return lo;
}
}  // namespace

Quat normalize_quaternion(const Quat& input) {
    Quat q=input;
    double n=norm(q);require(finite(q)&&n>=1e-12,"expected finite nonzero quaternion");
    for(double& x:q)x/=n;
    return q;
}
State State::from_json(const nlohmann::json& j) {
    State s;s.position=from_array<3>(j.at("position"));s.velocity=from_array<3>(j.at("velocity"));
    s.quaternion=from_array<4>(j.at("quaternion"));
    if(j.count("acceleration"))s.acceleration=from_array<3>(j.at("acceleration"));
    if(j.count("angular_velocity"))s.angular_velocity=from_array<3>(j.at("angular_velocity"));
    if(j.count("timestamp_ns")&&!j.at("timestamp_ns").is_null())s.timestamp_ns=j.at("timestamp_ns").get<std::uint64_t>();
    return s;
}
Reference Reference::from_json(const nlohmann::json& j) {
    Reference r;r.position=from_array<3>(j.at("position"));r.velocity=from_array<3>(j.at("velocity"));
    r.acceleration=from_array<3>(j.at("acceleration"));r.orientation=from_array<4>(j.at("orientation"));
    if(j.count("body_rates"))r.body_rates=from_array<3>(j.at("body_rates"));
    return r;
}
Trajectory Trajectory::from_layout(const nlohmann::json& j) {
    Trajectory t;double expected=0.;
    for(const auto& row:j.at("segments")){
        Segment s;s.start_time=row.at("start_time").get<double>();s.duration=row.at("duration").get<double>();
        require(std::isfinite(s.start_time)&&std::isfinite(s.duration)&&s.duration>0.,"invalid saved segment time");
        require(std::abs(s.start_time-expected)<=1e-10+1e-5*std::abs(expected),"noncontiguous saved segments");
        require(row.at("coefficients").size()==3,"polynomial needs three axes");
        for(std::size_t i=0;i<3;++i)s.coefficients[i]=from_array<6>(row.at("coefficients").at(i));
        s.heading_orientation=normalize_quaternion(from_array<4>(row.at("heading_orientation")));
        expected=s.start_time+s.duration;t.segments_.push_back(s);
    }
    require(!t.segments_.empty(),"trajectory needs at least one segment");t.duration_=expected;
    double stored=j.at("duration").get<double>();
    require(std::isfinite(stored)&&std::abs(stored-expected)<=1e-8+1e-5*std::abs(expected),"saved duration mismatch");
    return t;
}
Reference Trajectory::evaluate(double elapsed) const {
    require(std::isfinite(elapsed)&&elapsed>=0.&&!segments_.empty(),"invalid trajectory elapsed time");
    const Segment* s=&segments_.back();
    for(const Segment& candidate:segments_)if(elapsed<candidate.start_time+candidate.duration){s=&candidate;break;}
    double t=clip(elapsed-s->start_time,0.,s->duration);
    double t2=t*t,t3=t2*t,t4=t2*t2,t5=t4*t;
    Reference r;
    for(std::size_t axis=0;axis<3;++axis){const auto& c=s->coefficients[axis];
        r.position[axis]=c[0]+c[1]*t+c[2]*t2+c[3]*t3+c[4]*t4+c[5]*t5;
        r.velocity[axis]=c[1]+2.*c[2]*t+3.*c[3]*t2+4.*c[4]*t3+5.*c[5]*t4;
        r.acceleration[axis]=2.*c[2]+6.*c[3]*t+12.*c[4]*t2+20.*c[5]*t3;
    }
    r.orientation=attitude(r.acceleration,s->heading_orientation);
    require(finite(r.position)&&finite(r.velocity)&&finite(r.acceleration),"nonfinite trajectory reference");return r;
}
std::array<Reference,21> Trajectory::window(double elapsed,double dt,bool include_rates) const {
    require(std::isfinite(elapsed)&&elapsed>=0.&&std::isfinite(dt)&&dt>0.,"invalid reference window time");
    std::array<Reference,21> r;
    for(std::size_t i=0;i<r.size();++i){double t=elapsed+i*dt;r[i]=evaluate(t);
        // Body-rate feed-forward is unused by RAW237 but kept for parity tools.
        // evaluate returns NED attitude; flip back to planner z-up before delta.
        if(include_rates)r[i].body_rates=flip(body_rates(flip(r[i].orientation),
            flip(evaluate(t+kBodyRateDifferenceDt).orientation),kBodyRateDifferenceDt));
    }
    return r;
}
Raw237 encode(const State& s,const std::array<Reference,21>& refs) {
    require(finite(s.position)&&finite(s.velocity),"nonfinite measured state");
    std::array<Quat,21> q;
    for(std::size_t i=0;i<q.size();++i){
        require(finite(refs[i].position)&&finite(refs[i].velocity)&&finite(refs[i].acceleration),"nonfinite reference");
        q[i]=normalize_quaternion(refs[i].orientation);
        if((i==0&&q[i][0]<0.)||(i>0&&dot(q[i],q[i-1])<0.))q[i]=neg(q[i]);
    }
    Quat current=normalize_quaternion(s.quaternion);if(dot(current,q[0])<0.)current=neg(current);
    current=normalize_quaternion(flip(current));Raw237 x;std::size_t at=0;
    auto put=[&](double v){require(std::isfinite(v),"nonfinite RAW237");x[at++]=static_cast<float>(v);
        require(std::isfinite(x[at-1]),"RAW237 overflows float32");};
    for(double v:current)put(v);
    for(double v:flip(s.velocity))put(v);
    for(std::size_t i=0;i<refs.size();++i){
        for(double v:flip(sub(refs[i].position,s.position)))put(v);
        for(double v:normalize_quaternion(flip(q[i])))put(v);
        for(double v:flip(sub(refs[i].velocity,s.velocity)))put(v);
    }
    for(std::size_t i=0;i<20;++i)put(norm(add(flip(refs[i].acceleration),{{0.,0.,kFeatureGravity}})));
    require(at==x.size(),"RAW237 dimension mismatch");return x;
}
Raw237 encode(const State& s,const std::vector<Reference>& values) {
    require(!values.empty(),"reference sequence cannot be empty");
    std::array<Reference,21> refs;
    for(std::size_t i=0;i<refs.size();++i)refs[i]=values[std::min(i,values.size()-1)];
    return encode(s,refs);
}
void Calibration::validate() const {
    require(std::isfinite(hover_throttle)&&std::isfinite(hover_thrust_accel)&&
        std::isfinite(thrust_accel_per_throttle)&&std::isfinite(min_throttle)&&std::isfinite(max_throttle),
        "nonfinite throttle calibration");
    require(0.<=min_throttle&&min_throttle<max_throttle&&max_throttle<=1.&&
        min_throttle<=hover_throttle&&hover_throttle<=max_throttle&&
        hover_thrust_accel>0.&&thrust_accel_per_throttle>0.,"invalid throttle calibration");
}
Calibration Calibration::from_json(const nlohmann::json& j) {
    Calibration c;c.hover_throttle=j.at("hover_throttle").get<double>();
    c.hover_thrust_accel=j.at("hover_thrust_accel").get<double>();
    c.thrust_accel_per_throttle=j.at("thrust_accel_per_throttle").get<double>();
    c.min_throttle=j.at("min_throttle").get<double>();c.max_throttle=j.at("max_throttle").get<double>();
    if(j.count("calibration_complete"))c.calibration_complete=j.at("calibration_complete").get<bool>();
    c.validate();return c;
}
double thrust_to_throttle(double thrust,const Calibration& c) {
    c.validate();require(std::isfinite(thrust),"nonfinite physical thrust");
    return clip(c.hover_throttle+(thrust-c.hover_thrust_accel)/c.thrust_accel_per_throttle,c.min_throttle,c.max_throttle);
}
double throttle_to_thrust(double throttle,const Calibration& c) {
    c.validate();require(std::isfinite(throttle),"nonfinite throttle");
    return c.hover_thrust_accel+(clip(throttle,c.min_throttle,c.max_throttle)-c.hover_throttle)*c.thrust_accel_per_throttle;
}
Command convert_command(const Control& u,const Calibration& c,double duration) {
    require(finite(u)&&std::isfinite(duration)&&duration>0.,"invalid physical control or command duration");
    Command result;result.thrust_accel=u[0];result.roll_rate=u[1];result.pitch_rate=-u[2];result.yaw_rate=-u[3];
    result.throttle=thrust_to_throttle(u[0],c);result.duration=duration;return result;
}
PreparedCommand prepare_live_command(const Control& u,const Calibration& c,double duration) {
    require(finite(u),"nonfinite physical output");PreparedCommand p;p.requested=u;
    const Control lo={{2.,-3.,-3.,-2.}},hi={{20.,3.,3.,2.}};
    for(std::size_t i=0;i<4;++i)p.limited[i]=clip(u[i],lo[i],hi[i]);
    p.command=convert_command(p.limited,c,duration);p.effective=p.limited;
    p.effective[0]=throttle_to_thrust(p.command.throttle,c);
    return p;  // caller installs effective only after confirmed command submission.
}
void HoldBounds::validate() const {
    std::array<double,6> v={{acceleration_error_mps2,position_error_m,velocity_error_mps,
        maximum_hold_s,barrier_rate,gravity}};
    require(finite(v)&&v[0]>=0.&&v[1]>=0.&&v[2]>=0.&&v[3]>0.&&v[4]>0.&&v[5]>0.&&!evidence.empty(),
        "explicit finite hold bounds and evidence required");
}
Gate Gate::from_json(const nlohmann::json& j) {
    Gate g;g.index=j.at("index").get<int>();g.center=from_array<3>(j.at("center"));
    g.orientation=normalize_quaternion(from_array<4>(j.at("orientation")));return g;
}
Vec3 rotate_vector(const Quat& value,const Vec3& v) {
    require(finite(v),"nonfinite rotation vector");Quat q=normalize_quaternion(value);
    Quat pure={{0.,v[0],v[1],v[2]}},inv={{q[0],-q[1],-q[2],-q[3]}};
    Quat result=multiply(multiply(q,pure),inv);return {{result[1],result[2],result[3]}};
}
Vec3 nominal_acceleration(const Quat& q,const Control& u,double gravity) {
    require(finite(u)&&u[0]>=0.&&std::isfinite(gravity)&&gravity>0.,"nonnegative effective physical thrust required");
    return sub({{0.,0.,gravity}},scale(rotate_vector(q,{{0.,0.,1.}}),u[0]));
}
double barrier_period(double d,double v,double a,double j,const HoldBounds& b) {
    b.validate();return bisect_period(coefficients(d,v,a,j,b),b.maximum_hold_s);
}
GateHoldAnalyzer::GateHoldAnalyzer(const std::vector<Gate>& gates,double width,const HoldBounds& b,
                                 double height,double thickness,double radius,double clearance):bounds_(b) {
    b.validate();require(!gates.empty()&&std::isfinite(width)&&std::isfinite(height)&&std::isfinite(thickness)&&
        std::isfinite(radius)&&std::isfinite(clearance)&&width>0.&&height>0.&&thickness>0.&&radius>=0.&&clearance>=0.,
        "valid finite gate geometry required");
    // Exactly _gate_parts(gate,width+thickness,height+thickness,thickness).
    width+=thickness;height+=thickness;
    std::array<Vec3,4> local={{{{0.,-width/2.,0.}},{{0.,width/2.,0.}},{{0.,0.,-height/2.}},{{0.,0.,height/2.}}}};
    std::array<Vec3,4> dims={{{{thickness,thickness,height}},{{thickness,thickness,height}},
                             {{thickness,width,thickness}},{{thickness,width,thickness}}}};
    const char* parts[]={"left","right","top","bottom"};
    for(const Gate& gate:gates){
        require(finite(gate.center),"nonfinite gate center");
        std::array<Vec3,3> axes={{rotate_vector(gate.orientation,{{1.,0.,0.}}),
            rotate_vector(gate.orientation,{{0.,1.,0.}}),rotate_vector(gate.orientation,{{0.,0.,1.}})}};
        for(std::size_t part=0;part<4;++part){Box box;Vec3 center=add(gate.center,rotate_vector(gate.orientation,local[part]));
            box.label=std::to_string(gate.index)+":"+parts[part];
            for(std::size_t face=0;face<6;++face){std::size_t axis=face%3;
                box.normals[face]=scale(axes[axis],face<3?-1.:1.);
                box.offsets[face]=dot(box.normals[face],center)+dims[part][axis]/2.+radius+clearance;
            }boxes_.push_back(box);
        }
    }
}
HoldAnalysis GateHoldAnalyzer::analyze(const State& s,const Control& u) const {
    require(finite(s.position)&&finite(s.velocity),"nonfinite measured state for holding");
    HoldAnalysis r;r.nominal_acceleration=nominal_acceleration(s.quaternion,u,bounds_.gravity);
    Vec3 omega={{u[1],u[2],u[3]}};double jerk=u[0]*norm(omega),cap=bounds_.maximum_hold_s;
    double minimum=std::numeric_limits<double>::infinity(),nearest=std::numeric_limits<double>::infinity();
    for(const Box& box:boxes_){double best=0.,box_margin=-std::numeric_limits<double>::infinity(),nearest_rate=0.;
        int face_best=0,first_full=-1;std::array<Barrier,6> q;
        for(std::size_t face=0;face<6;++face){double d=dot(box.normals[face],s.position)-box.offsets[face];
            double v=dot(box.normals[face],s.velocity),a=dot(box.normals[face],r.nominal_acceleration);
            q[face]=coefficients(d,v,a,jerk,bounds_);
            if(q[face].valid&&polynomial(q[face],cap)>=0.&&first_full<0)first_full=static_cast<int>(face);
            if(d>box_margin){box_margin=d;nearest_rate=v;}
        }
        if(first_full>=0){best=cap;face_best=first_full;}
        else for(std::size_t face=0;face<6;++face){double v=bisect_period(q[face],cap);
            if(v>best){best=v;face_best=static_cast<int>(face);}}
        r.per_obstacle_period_s.push_back(best);r.separating_face.push_back(face_best);
        if(best<minimum){minimum=best;r.binding_obstacle=box.label;}
        if(box_margin<nearest){nearest=box_margin;r.mesh_margin_m=box_margin;r.mesh_rate_mps=nearest_rate;}
    }
    r.budget_s=minimum;r.valid=minimum>0.;r.conditional_barrier_certificate=r.valid;
    r.plant_bounds_verified=bounds_.plant_bounds_verified;r.is_safety_certificate=r.valid&&r.plant_bounds_verified;
    return r;
}
SafeDeadline safe_deadline(double sample,double now,double actuation,const HoldAnalysis& a,double maximum,double uncertainty) {
    require(std::isfinite(sample)&&std::isfinite(now)&&std::isfinite(actuation)&&std::isfinite(maximum)&&
        std::isfinite(uncertainty)&&actuation<=sample&&sample<=now&&maximum>0.&&uncertainty>=0.,
        "ordered sample/actuation/analysis clocks and finite uncertainty required");
    SafeDeadline d;d.physical_deadline_s=sample+a.budget_s;
    d.deadline_s=std::min(d.physical_deadline_s,actuation+maximum-uncertainty);
    d.remaining_s=std::max(0.,d.deadline_s-now);d.state_age_s=now-sample;
    d.feasible=d.deadline_s>now;d.is_safety_certificate=a.is_safety_certificate;return d;
}
}  // namespace psfeatures
