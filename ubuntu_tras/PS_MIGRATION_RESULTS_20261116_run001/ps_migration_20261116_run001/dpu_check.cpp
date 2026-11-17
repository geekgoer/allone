#include "local_dpu.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <sys/stat.h>
#include <vector>
using nlohmann::json;
json read_json(const std::string& p){std::ifstream f(p);if(!f)throw std::runtime_error("missing "+p);json j;f>>j;return j;}
void write_json(const std::string& p,const json& j){std::ofstream f(p);f<<j.dump(2)<<'\n';if(!f)throw std::runtime_error("write failed "+p);}
json stats(std::vector<double> v){
  if(v.empty())return nullptr;
  std::sort(v.begin(),v.end());double sum=0;for(double x:v)sum+=x;
  auto q=[&](double p){double x=p*(v.size()-1);size_t lo=size_t(x),hi=std::min(lo+1,v.size()-1);return v[lo]+(v[hi]-v[lo])*(x-lo);};
  return {{"count",v.size()},{"mean",sum/v.size()},{"p50",q(.5)},{"p95",q(.95)},{"p99",q(.99)},{"max",v.back()}};
}
int main(int argc,char** argv){try{
  std::string package,out,problem;int repeats=100,warmup=100;
  for(int i=1;i<argc;++i){std::string k=argv[i];if(++i==argc)throw std::runtime_error("missing option value");std::string v=argv[i];
    if(k=="--package")package=v;else if(k=="--out")out=v;else if(k=="--problem-inputs")problem=v;
    else if(k=="--repeats")repeats=std::stoi(v);else if(k=="--warmup")warmup=std::stoi(v);else throw std::runtime_error("unknown option "+k);}
  if(package.empty()||out.empty()||repeats<1||warmup<0)throw std::runtime_error("--package DIR --out NEWDIR [--problem-inputs JSON] [--repeats 100] [--warmup 100]");
  if(mkdir(out.c_str(),0775)!=0)throw std::runtime_error("output exists or cannot be created: "+out);
  auto registry=read_json(package+"/dpu_model_registry.json");psdpu::LocalDpu dpu;
  write_json(out+"/actual_identity.json",dpu.identity());
  json golden=json::array(),profile=json::array();std::ofstream raw(out+"/local_dpu_requests.csv");
  raw<<"group,model_id,round,sample,status,saturation,start_us,dispatch_us,prep_end_us,dpu_submit_us,dpu_readable_us,output_ready_us,u0,u1,u2,u3,golden_max_abs\n"<<std::setprecision(12);
  auto emit=[&](const std::string& group,int id,int round,int sample,const psdpu::Result& r,double e){
    raw<<group<<','<<id<<','<<round<<','<<sample<<','<<r.status<<','<<r.saturation<<','<<r.start_us<<','<<r.dispatch_us<<','<<r.prep_end_us<<','<<r.run_start_us<<','<<r.run_end_us<<','<<r.end_us;
    for(float y:r.output)raw<<','<<y;
    raw<<','<<e<<'\n';};
  for(const auto& m:registry["models"]){int id=m["numeric_id"].get<int>();std::string name=m["id"].get<std::string>();
    std::ifstream f(package+"/golden/"+name+".txt");if(!f)throw std::runtime_error("missing golden "+name);
    std::vector<psdpu::Input> xs;std::string line;double max_error=0.;
    while(std::getline(f,line)){if(line.empty()||line[0]=='#')continue;
      std::istringstream s(line);psdpu::Input x{};std::array<float,4> ref{};
      for(auto& v:x)if(!(s>>v))throw std::runtime_error("bad golden input");
      for(auto& v:ref)if(!(s>>v))throw std::runtime_error("bad golden output");
      auto r=dpu.infer(id,x);double e=0.;for(int c=0;c<4;++c)e=std::max(e,std::abs(double(r.output[c])-ref[c]));
      emit("golden",id,0,int(xs.size()),r,e);max_error=std::max(max_error,e);xs.push_back(x);
      if(r.status||e>1e-4)throw std::runtime_error("local golden mismatch model "+name);}
    if(xs.size()!=32)throw std::runtime_error("expected 32 golden rows");
    golden.push_back({{"model_id",id},{"rows",xs.size()},{"max_abs",max_error},{"status","PASS"}});
    std::cout<<"LOCAL_GOLDEN_PASS id="<<id<<" rows="<<xs.size()<<" max_abs="<<max_error<<std::endl;
    for(int i=0;i<warmup;++i)if(dpu.infer(id,xs[i%xs.size()]).status)throw std::runtime_error("warmup failure");
    std::vector<double> call,service,prep,post;
    for(int round=1;round<=3;++round)for(int i=0;i<repeats;++i){auto r=dpu.infer(id,xs[i%xs.size()]);emit("profile",id,round,i,r,0.);
      if(r.status)throw std::runtime_error("profile failure");
      call.push_back(r.run_end_us-r.run_start_us);service.push_back(r.end_us-r.start_us);
      prep.push_back(r.prep_end_us-r.dispatch_us);post.push_back(r.end_us-r.run_end_us);}
    profile.push_back({{"model_id",id},{"input_preparation_us",stats(prep)},{"dpu_call_us",stats(call)},
                       {"output_processing_us",stats(post)},{"service_us",stats(service)}});
    write_json(out+"/golden_summary.json",golden);write_json(out+"/cost_profile.json",profile);
  }
  json rejected=json::array();json problems;
  if(!problem.empty())problems=read_json(problem);
  if(!problem.empty())for(const auto& row:problems.at("inputs")){
    auto v=row["raw237"].get<std::vector<float>>();if(v.size()!=237)throw std::runtime_error("wrong problem input size");
    psdpu::Input x;std::copy(v.begin(),v.end(),x.begin());int id=row["model_id"].get<int>();auto r=dpu.infer(id,x);
    rejected.push_back({{"source",row},{"status",r.status},{"saturation",r.saturation},{"rejected_channels",r.rejected_channels},{"output_valid",r.status==0}});
    emit("problem_reproduction",id,0,int(rejected.size()-1),r,0.);}
  raw.close();write_json(out+"/golden_summary.json",golden);write_json(out+"/cost_profile.json",profile);write_json(out+"/problem_reproduction.json",rejected);
  write_json(out+"/completion.json",{{"models",golden.size()},{"rounds",3},{"repeats",repeats},{"status","COMPLETE"}});
  return 0;
}catch(const std::exception& e){std::cerr<<"ERROR "<<e.what()<<std::endl;return 1;}}
