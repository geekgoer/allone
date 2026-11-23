// Optional LD_PRELOAD diagnostic for the single-DPU Route B runtime.
// The trace is buffered in memory and flushed when the controller exits.
#include <dnndk/dnndk.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>

namespace {
using Int = long long;
struct Event {
    Int begin, end, cpu, device, runqueue, voluntary, involuntary, tid;
    int status;
};
constexpr unsigned capacity=50000;
Event events[capacity];
std::atomic<unsigned> used{0};

Int stamp(clockid_t clock){
    timespec tv{};
    if(clock_gettime(clock,&tv)!=0)return -1;
    return Int(tv.tv_sec)*1000000000LL+tv.tv_nsec;
}

Int runqueue_ns(){
    static thread_local const int fd=open("/proc/thread-self/schedstat",O_RDONLY|O_CLOEXEC);
    if(fd<0)return -1;
    char text[128]{};
    if(pread(fd,text,sizeof(text)-1,0)<=0)return -1;
    Int running=0,waiting=0;
    return std::sscanf(text,"%lld %lld",&running,&waiting)==2?waiting:-1;
}

__attribute__((destructor)) void dump(){
    const char* path=std::getenv("ROUTE_B_DNNDK_TRACE");
    if(!path)return;
    FILE* file=std::fopen(path,"wx");
    if(!file){std::perror(path);return;}
    std::fprintf(file,"begin_ns,end_ns,thread_cpu_ns,driver_profile_us,runqueue_ns,voluntary_cs,involuntary_cs,tid,status\n");
    const unsigned count=used.load();
    for(unsigned i=0;i<count&&i<capacity;++i){
        const auto& e=events[i];
        std::fprintf(file,"%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%d\n",
                     e.begin,e.end,e.cpu,e.device,e.runqueue,e.voluntary,e.involuntary,e.tid,e.status);
    }
    if(count>capacity)std::fprintf(stderr,"DNNDK trace dropped %u events\n",count-capacity);
    if(std::fclose(file)!=0)std::perror(path);
}
}

int dpuRunTask(DPUTask* task){
    // Export and look up the ABI symbols used by the deployed DNNDK C++ header.
    static const auto run=reinterpret_cast<int(*)(DPUTask*)>(dlsym(RTLD_NEXT,"_Z10dpuRunTaskP8dpu_task"));
    static const auto profile=reinterpret_cast<Int(*)(DPUTask*)>(dlsym(RTLD_NEXT,"_Z17dpuGetTaskProfileP8dpu_task"));
    static const bool enabled=std::getenv("ROUTE_B_DNNDK_TRACE")!=nullptr;
    if(!run)return -1;
    if(!enabled)return run(task);

    rusage before{},after{};
    const auto tid=Int(syscall(SYS_gettid));
    const bool have_before=getrusage(RUSAGE_THREAD,&before)==0;
    const auto wait_before=runqueue_ns();
    const auto cpu_before=stamp(CLOCK_THREAD_CPUTIME_ID);
    const auto begin=stamp(CLOCK_MONOTONIC);
    const auto status=run(task);
    const auto saved_errno=errno;
    const auto end=stamp(CLOCK_MONOTONIC);
    const auto cpu_after=stamp(CLOCK_THREAD_CPUTIME_ID);
    const auto wait_after=runqueue_ns();
    const bool have_switches=getrusage(RUSAGE_THREAD,&after)==0&&have_before;
    const auto index=used.fetch_add(1,std::memory_order_relaxed);
    if(index<capacity)events[index]={begin,end,cpu_before<0||cpu_after<0?-1:cpu_after-cpu_before,
        status==0&&profile?profile(task):-1,
        wait_before<0||wait_after<wait_before?-1:wait_after-wait_before,
        have_switches?after.ru_nvcsw-before.ru_nvcsw:-1,
        have_switches?after.ru_nivcsw-before.ru_nivcsw:-1,tid,status};
    errno=saved_errno;
    return status;
}
