// Standalone model-11 comparison. This process never connects to AirSim.
#include <dnndk/dnndk.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
using Clock=std::chrono::steady_clock;
using Input=std::array<int8_t,240>;
using Activation=std::array<int8_t,1024>;
using Output=std::array<int8_t,4>;

void check(int status,const char* operation){
    if(status!=0)throw std::runtime_error(std::string(operation)+": "+std::to_string(status));
}
double us(Clock::time_point start,Clock::time_point end){
    return std::chrono::duration<double,std::micro>(end-start).count();
}
struct Device {
    Device(){check(dpuOpen(),"dpuOpen");}
    ~Device(){dpuClose();}
};

class Kernel {
public:
    Kernel(const char* name,const char* input,const char* output,int in_size,int out_size):
        in_(input),out_(output),kernel_(dpuLoadKernel(name),dpuDestroyKernel),
        task_(kernel_?dpuCreateTask(kernel_.get(),T_MODE_NORMAL):nullptr,dpuDestroyTask){
        if(!task_)throw std::runtime_error(std::string("Could not load ")+name);
        if(dpuGetInputTensorSize(task_.get(),in_)!=in_size||
           dpuGetOutputTensorSize(task_.get(),out_)!=out_size)
            throw std::runtime_error(std::string("Unexpected tensor size for ")+name);
        ins=dpuGetInputTensorScale(task_.get(),in_);
        outs=dpuGetOutputTensorScale(task_.get(),out_);
        std::cerr<<name<<" input_multiplier="<<ins<<" output_multiplier="<<outs<<"\n";
    }
    template<std::size_t N,std::size_t M>
    void run(const std::array<int8_t,N>& input,std::array<int8_t,M>& output){
        auto payload=input;
        check(dpuSetInputTensorInHWCInt8(task_.get(),in_,payload.data(),N),"input");
        check(dpuRunTask(task_.get()),"run");
        check(dpuGetOutputTensorInHWCInt8(task_.get(),out_,output.data(),M),"output");
    }
    float ins=0,outs=0;
private:
    const char* in_;
    const char* out_;
    std::unique_ptr<DPUKernel,decltype(&dpuDestroyKernel)> kernel_;
    std::unique_ptr<DPUTask,decltype(&dpuDestroyTask)> task_;
};

std::vector<Input> read_inputs(const char* filename){
    std::ifstream stream(filename);
    if(!stream)throw std::runtime_error("input file open failed");
    std::vector<Input> rows;
    std::string line;
    while(std::getline(stream,line)){
        std::istringstream entry(line);
        Input input;
        for(auto& value:input){
            int parsed;
            if(!(entry>>parsed)||parsed<-128||parsed>127)throw std::runtime_error("Invalid INT8 row");
            value=static_cast<int8_t>(parsed);
        }
        std::string extra;
        if(entry>>extra||input[237]||input[238]||input[239])throw std::runtime_error("Invalid input padding");
        rows.push_back(input);
    }
    if(rows.size()<2)throw std::runtime_error("At least two rows are required");
    return rows;
}

void compare_dispatch(const std::vector<Input>& rows,const char* filename,
                      Kernel& whole,Kernel& prefix,Kernel& suffix){
    std::ofstream csv(filename);
    if(!csv)throw std::runtime_error("contention file open failed");
    csv<<"repeat,offset_us,policy,urgent_due_us,donor_due_us,urgent_done_us,donor_done_us,yielded\n";
    const std::array<const char*,4> policies{{"whole","boundary_only","forecast_guard","whole_deferred"}};
    for(int offset:{300,900})for(std::size_t j=0;j<80;++j)for(std::size_t order=0;order<policies.size();++order){
        const auto policy=(j+order)%policies.size();
        const int urgent_due=offset+1650,donor_due=3300;
        // This bench has a scripted release and measured, rounded service estimates.
        const int whole_cost=1320,prefix_cost=590,suffix_cost=760;
        const int urgent_if_split=std::max(offset,prefix_cost)+whole_cost;
        const bool protect=policy==2&&std::max(offset,whole_cost)+whole_cost>urgent_due&&
            urgent_if_split<=urgent_due&&urgent_if_split+suffix_cost<=donor_due;
        const bool split=policy==1||protect;
        Activation hidden;
        Output donor{},urgent{};
        const auto start=Clock::now();
        const auto release=start+std::chrono::microseconds(offset);
        if(policy==3){
            // Compare giving priority to the future request without a segment boundary.
            std::this_thread::sleep_until(release);
            whole.run(rows[(j+1)%rows.size()],urgent);
            const auto urgent_done=Clock::now();
            whole.run(rows[j%rows.size()],donor);
            csv<<j<<','<<offset<<','<<policies[policy]<<','<<urgent_due<<','<<donor_due<<','
               <<us(start,urgent_done)<<','<<us(start,Clock::now())<<",0\n";
            continue;
        }
        bool yielded=false;
        if(split){
            prefix.run(rows[j%rows.size()],hidden);
            yielded=protect||Clock::now()>=release;
            if(!yielded)suffix.run(hidden,donor);
        }else whole.run(rows[j%rows.size()],donor);
        auto donor_done=Clock::now();
        if(donor_done<release)std::this_thread::sleep_until(release);
        whole.run(rows[(j+1)%rows.size()],urgent);
        const auto urgent_done=Clock::now();
        if(yielded){suffix.run(hidden,donor);donor_done=Clock::now();}
        csv<<j<<','<<offset<<','<<policies[policy]<<','<<urgent_due<<','<<donor_due<<','
           <<us(start,urgent_done)<<','<<us(start,donor_done)<<','<<yielded<<'\n';
    }
    if(!csv)throw std::runtime_error("contention file write failed");
}
} // namespace

int main(int argc,char** argv){
    try{
        if(argc<2||argc>3)throw std::runtime_error("usage: benchmark_segments <normalized_input_int8.txt> [contention.csv]");
        const auto rows=read_inputs(argv[1]);
        Device device;
        Kernel whole("rq_73c051a35e68","conv_00","conv_04",240,4);
        Kernel prefix("rb11_seg_a","conv_00","conv_01",240,1024);
        Kernel suffix("rb11_seg_b","conv_02","conv_04",1024,4);
        if(whole.ins!=16.f||prefix.ins!=whole.ins||prefix.outs!=1.f/32.f||suffix.ins!=32.f||
           suffix.outs!=whole.outs||whole.outs!=0.0625f)
            throw std::runtime_error("Scale mismatch: raw passthrough requires matching fixed-point payloads");
        std::cout<<"row,whole_us,prefix_us,suffix_us,seq_total_us,max_abs_output_lsb\n";
        int max_difference=0;
        for(std::size_t j=0;j<100+rows.size()*10;++j){
            const auto& input=rows[j%rows.size()];
            Activation hidden;
            Output ref{},actual{};
            const auto w0=Clock::now();whole.run(input,ref);const auto w1=Clock::now();
            prefix.run(input,hidden);const auto p1=Clock::now();
            suffix.run(hidden,actual);const auto s1=Clock::now();
            int difference=0;
            for(std::size_t c=0;c<ref.size();++c)
                difference=std::max(difference,std::abs(int(ref[c])-int(actual[c])));
            if(j>=100){
                max_difference=std::max(max_difference,difference);
                std::cout<<j%rows.size()<<','<<us(w0,w1)<<','<<us(w1,p1)<<','<<us(p1,s1)<<','
                         <<us(w1,s1)<<','<<difference<<'\n';
            }
        }
        // Preserve A's activation through a second invocation of the same prefix.
        Activation a,b;
        Output a_then,b_then,a_ref,b_ref;
        prefix.run(rows[0],a);prefix.run(rows[1],b);
        suffix.run(b,b_then);suffix.run(a,a_then);
        whole.run(rows[0],a_ref);whole.run(rows[1],b_ref);
        const bool interleaved=a_then==a_ref&&b_then==b_ref;
        std::cerr<<"max_int8_difference="<<max_difference<<" interleaved_exact="<<interleaved<<'\n';
        if(max_difference!=0||!interleaved)return 2;
        if(argc==3)compare_dispatch(rows,argv[2],whole,prefix,suffix);
        return 0;
    }catch(const std::exception& exception){
        std::cerr<<exception.what()<<'\n';
        return 1;
    }
}
