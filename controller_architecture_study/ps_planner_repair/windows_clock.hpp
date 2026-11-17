#pragma once
// Host-only fallback. Performance conclusions must use the actual ARM clocks.
#define NOMINMAX
#include <Windows.h>
#include <chrono>
#include <ctime>
#define CLOCK_MONOTONIC 1
#define CLOCK_THREAD_CPUTIME_ID 2
inline int profile_windows_clock(int clock, timespec* result) {
    long long nanoseconds=0;
    if(clock==CLOCK_THREAD_CPUTIME_ID) {
        FILETIME created,exited,kernel,user;
        if(!GetThreadTimes(GetCurrentThread(),&created,&exited,&kernel,&user)) return -1;
        const auto value=[](FILETIME f) {return (static_cast<unsigned long long>(f.dwHighDateTime)<<32)|f.dwLowDateTime;};
        nanoseconds=static_cast<long long>((value(kernel)+value(user))*100);
    } else nanoseconds=std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    result->tv_sec=static_cast<time_t>(nanoseconds/1000000000);
    result->tv_nsec=static_cast<long>(nanoseconds%1000000000);
    return 0;
}
#define clock_gettime profile_windows_clock
