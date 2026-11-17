#pragma once
#include "scheduler.hpp"
#include <time.h>
#include <stdexcept>
namespace ps_scheduler {
struct ManagementStamp { double wall_ms,cpu_ms; };
inline ManagementStamp management_stamp() {
    timespec wall{},cpu{};
    if(clock_gettime(CLOCK_MONOTONIC,&wall)||clock_gettime(CLOCK_THREAD_CPUTIME_ID,&cpu))
        throw std::runtime_error("Management timing clock unavailable");
    return {wall.tv_sec*1000.+wall.tv_nsec/1e6,cpu.tv_sec*1000.+cpu.tv_nsec/1e6};
}
inline Json management_interval(ManagementStamp start,ManagementStamp end) {
    return Json{{"wall_ms",end.wall_ms-start.wall_ms},{"cpu_ms",end.cpu_ms-start.cpu_ms}};
}
}
