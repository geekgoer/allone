#include "features.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

using nlohmann::json;
using namespace psfeatures;
namespace {
json load(const std::string& path) {
    std::ifstream in(path);if(!in)throw std::runtime_error("cannot read "+path);
    json j;in>>j;return j;
}
void check(bool ok,const std::string& reason) { if(!ok)throw std::runtime_error(reason); }
void close(double actual,double expected,double atol,double rtol,const std::string& reason) {
    if(std::abs(actual-expected)>atol+rtol*std::abs(expected))
        throw std::runtime_error(reason+" actual="+std::to_string(actual)+" expected="+std::to_string(expected));
}
template<std::size_t N> double array_error(const std::array<double,N>& a,const json& b,
                                          double atol,double rtol,const std::string& label) {
    check(b.size()==N,label+" shape");double maximum=0.;
    for(std::size_t i=0;i<N;++i){double v=b.at(i).get<double>();
        close(a[i],v,atol,rtol,label+"["+std::to_string(i)+"]");maximum=std::max(maximum,std::abs(a[i]-v));}
    return maximum;
}
json analysis_json(const HoldAnalysis& a) {
    return json{{"budget_s",a.budget_s},{"valid",a.valid},{"binding_obstacle",a.binding_obstacle},
        {"nominal_acceleration",a.nominal_acceleration},{"per_obstacle_period_s",a.per_obstacle_period_s},
        {"separating_face",a.separating_face},{"mesh_margin_m",a.mesh_margin_m},{"mesh_rate_mps",a.mesh_rate_mps},
        {"plant_bounds_verified",a.plant_bounds_verified},{"is_safety_certificate",a.is_safety_certificate}};
}
void holding_self_tests() {
    HoldBounds b;b.acceleration_error_mps2=0.;b.position_error_m=0.;b.velocity_error_mps=0.;
    b.evidence="synthetic exact model";
    check(barrier_period(.2,-1.,8.,0.,b)>barrier_period(.2,-1.,-8.,0.,b),"direction must affect hold budget");
    check(barrier_period(.2,-1.,-8.,0.,b)==0.,"invalid HOCBF must not certify");
    Gate g;g.index=1;g.center={{10.,0.,-4.}};g.orientation={{1.,0.,0.,0.}};
    GateHoldAnalyzer a({g},1.7,b);State s;s.position={{0.,2.,-4.}};s.velocity={{5.,0.,0.}};
    auto clear=a.analyze(s,{{9.80665,0.,0.,0.}});
    check(clear.budget_s>.149&&!clear.is_safety_certificate,"finite gate aperture conditional only");
    s.position={{10.,.925,-4.}};s.velocity={{0.,0.,0.}};
    check(a.analyze(s,{{9.80665,0.,0.,0.}}).budget_s==0.,"inside inflated bar must not certify");
    HoldAnalysis h;h.budget_s=.05;
    auto deadline=safe_deadline(10.01,10.04,10.,h,.08,.002);
    close(deadline.deadline_s,10.06,1e-12,0.,"remaining hold deadline");
    close(deadline.remaining_s,.02,1e-12,0.,"state age charged once");
    check(!safe_deadline(10.01,10.07,10.,h,.08).feasible,"expired cannot be feasible");
    bool rejected=false;try{nominal_acceleration({{0.,0.,0.,0.}},{{9.8,0.,0.,0.}});}catch(const std::invalid_argument&){rejected=true;}
    check(rejected,"zero quaternion rejected");
    rejected=false;try{safe_deadline(10.,10.01,10.02,h,.08);}catch(const std::invalid_argument&){rejected=true;}
    check(rejected,"unordered actuation/sample must reject");
}
}  // namespace

int main(int argc,char** argv) {
    try {
        if(argc<2||argc>3)throw std::runtime_error("usage: test_features HANDOFF_ROOT [REPORT_JSON]");
        std::string root=argv[1];json fixture=load(root+"/fixtures/feature_command_parity.json");
        json scene=load(root+"/project/controller_architecture_study/results/release001_live_case_20261116/case.json");
        double atol=fixture.at("feature_atol").get<double>(),rtol=fixture.at("feature_rtol").get<double>();
        json report={{"scope","offline numerical parity, not flight/PS execution proof"},
                     {"feature_cases",json::array()},{"commands",json::array()}};
        std::size_t total=0;double feature_max=0.,window_max=0.;
        for(const auto& row:fixture.at("cases")){
            State state=State::from_json(row.at("state"));std::array<Reference,21> refs;
            check(row.at("references").size()==21,"fixture horizon must be 21");
            for(std::size_t i=0;i<refs.size();++i)refs[i]=Reference::from_json(row.at("references").at(i));
            Raw237 x=encode(state,refs);double max_error=0.;
            for(std::size_t i=0;i<x.size();++i){double expected=row.at("expected_raw237").at(i).get<double>();
                close(x[i],expected,atol,rtol,"RAW237 index "+std::to_string(i));
                max_error=std::max(max_error,std::abs(static_cast<double>(x[i])-expected));++total;}
            std::string vehicle=row.at("vehicle").get<std::string>();
            auto trajectory=Trajectory::from_layout(scene.at("layouts").at(vehicle));
            auto actual_refs=trajectory.window(row.at("reference_time_s").get<double>());
            double reference_max=0.;
            for(std::size_t i=0;i<actual_refs.size();++i){const json& expected=row.at("references").at(i);
                reference_max=std::max(reference_max,array_error(actual_refs[i].position,expected.at("position"),2e-9,2e-9,"reference position"));
                reference_max=std::max(reference_max,array_error(actual_refs[i].velocity,expected.at("velocity"),2e-9,2e-9,"reference velocity"));
                reference_max=std::max(reference_max,array_error(actual_refs[i].acceleration,expected.at("acceleration"),2e-9,2e-9,"reference acceleration"));
                reference_max=std::max(reference_max,array_error(actual_refs[i].orientation,expected.at("orientation"),2e-9,2e-9,"reference attitude"));
            }
            Raw237 chained=encode(state,actual_refs);double chained_max=0.;
            for(std::size_t i=0;i<x.size();++i){double expected=row.at("expected_raw237").at(i).get<double>();
                close(chained[i],expected,atol,rtol,"polynomial→RAW237 index "+std::to_string(i));
                chained_max=std::max(chained_max,std::abs(static_cast<double>(chained[i])-expected));}
            State sign=state;for(double& q:sign.quaternion)q=-q;
            check(encode(sign,refs)==x,"current quaternion sign invariance");
            auto negative_refs=refs;for(auto& ref:negative_refs)for(double& q:ref.orientation)q=-q;
            check(encode(state,negative_refs)==x,"reference quaternion sign invariance");
            State unused=state;unused.acceleration={{1e9,-1e9,1e9}};unused.angular_velocity={{1e8,1e8,-1e8}};
            check(encode(unused,refs)==x,"measured acceleration/rates excluded from 237D contract");
            report["feature_cases"].push_back({{"vehicle",vehicle},{"reference_time_s",row.at("reference_time_s")},
                {"raw237_max_abs",max_error},{"reference_max_abs",reference_max},{"polynomial_raw237_max_abs",chained_max}});
            feature_max=std::max(feature_max,std::max(max_error,chained_max));window_max=std::max(window_max,reference_max);
        }
        Calibration calibration=Calibration::from_json(fixture.at("calibration"));
        for(const auto& row:fixture.at("commands")){
            Control u;for(std::size_t i=0;i<u.size();++i)u[i]=row.at("physical_control").at(i).get<double>();
            Command c=convert_command(u,calibration,1.);
            json actual={{"thrust_accel",c.thrust_accel},{"roll_rate",c.roll_rate},{"pitch_rate",c.pitch_rate},
                {"yaw_rate",c.yaw_rate},{"throttle",c.throttle},{"duration",c.duration}};
            for(auto it=actual.begin();it!=actual.end();++it)
                close(it.value().get<double>(),row.at("expected_command").at(it.key()).get<double>(),1e-12,0.,it.key());
            auto effective=prepare_live_command(u,calibration,1.);
            report["commands"].push_back({{"physical_control",u},{"command",actual},{"effective_physical",effective.effective}});
        }
        auto saturated=prepare_live_command({{100.,4.,-4.,3.}},calibration,1.);
        close(saturated.effective[0],17.658,1e-12,0.,"feedback must reflect post-throttle-clipping thrust");
        check(saturated.effective[1]==3.&&saturated.effective[2]==-3.&&saturated.effective[3]==2.,"live rate limits");
        holding_self_tests();
        // Emit frozen geometry/observations for independent Python comparison.
        report["hold_cases"]=json::array();HoldBounds bounds;
        for(const auto& row:fixture.at("cases")){
            std::string vehicle=row.at("vehicle").get<std::string>();std::vector<Gate> gates;
            for(const auto& g:scene.at("layouts").at(vehicle).at("gates"))gates.push_back(Gate::from_json(g));
            GateHoldAnalyzer analyzer(gates,scene.at("gate_center_width_m").get<double>()-.15,bounds,4.-.15);
            State state=State::from_json(row.at("state"));
            for(const Control& u:std::array<Control,3>{{{{9.81,0.,0.,0.}},{{10.2,.3,-.2,.1}},{{17.658,3.,-3.,2.}}}})
                report["hold_cases"].push_back({{"vehicle",vehicle},{"state",row.at("state")},
                    {"effective_control",u},{"actual",analysis_json(analyzer.analyze(state,u))}});
        }
        report["status"]="PASS";report["feature_values_checked"]=total;
        report["raw237_max_abs"]=feature_max;report["reference_max_abs"]=window_max;
        report["holding_self_tests"]="PASS";
        if(argc==3){std::ofstream out(argv[2]);check(bool(out),"cannot create report");out<<std::setw(2)<<report<<'\n';}
        std::cout<<"NUMERICAL_PARITY_PASS feature_cases="<<fixture.at("cases").size()<<" values="<<total
            <<" commands="<<fixture.at("commands").size()<<" raw237_max_abs="<<feature_max
            <<" reference_max_abs="<<window_max<<" hold_self_tests=PASS\n";
        return 0;
    }catch(const std::exception& e){std::cerr<<"NUMERICAL_PARITY_FAIL "<<e.what()<<'\n';return 1;}
}
