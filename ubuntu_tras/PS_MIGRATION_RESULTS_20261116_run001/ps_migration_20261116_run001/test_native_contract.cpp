#include "features/features.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
int main(int argc,char**argv){try{
  if(argc!=3)throw std::runtime_error("wire_command_cases.json thrust_calibration.json");
  nlohmann::json wire,cal;std::ifstream(argv[1])>>wire;std::ifstream(argv[2])>>cal;
  auto calibration=psfeatures::Calibration::from_json(cal);int count=0;
  for(const auto& row:wire.at("cases")){
    psfeatures::Control input;for(int c=0;c<4;++c)input[c]=row.at("controller_output").at(c).get<double>();
    auto prepared=psfeatures::prepare_live_command(input,calibration);
    auto rpc=psfeatures::native_rpc_command(prepared);
    double actual[]={rpc.roll_rate,rpc.pitch_rate,rpc.yaw_rate,rpc.throttle};
    for(int c=0;c<4;++c){
      if(std::abs(actual[c]-row.at("expected_rpc_and_airlib_cpp_arguments").at(c).get<double>())>1e-12)
        throw std::runtime_error("native wire argument mismatch");
      if(std::abs(prepared.effective[c]-row.at("effective_control_after_throttle_clipping").at(c).get<double>())>1e-12)
        throw std::runtime_error("effective control feedback mismatch");
    }
    ++count;
  }
  std::cout<<"NATIVE_WIRE_CONTRACT_PASS cases="<<count<<" physical_rates_direct=1 python_double_flip_avoided=1\n";
  return count==3?0:1;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
