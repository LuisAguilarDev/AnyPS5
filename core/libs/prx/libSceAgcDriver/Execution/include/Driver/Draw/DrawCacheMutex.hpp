#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAW_DRAWCACHEMUTEX_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAW_DRAWCACHEMUTEX_HPP

#include <shared_mutex>

namespace AgcDriver::DriverDetail {

// The draw cache's lock. Lookups from the prep-pool jobs (APS5_NO_POOL_LOOKUP unset) take it shared and
// bump their counters through std::atomic_ref; inserts, evictions, the LRU touch and the reports take
// it exclusive. Under APS5_PROFILE_DRAW each thread times its waits and holds and prints a [lock] line
// every 10 s, next to the GPU mutex's matrix (APS5_NO_LOCK_PROFILE=1 turns both off).
class DrawCacheMutex {
public:
    DrawCacheMutex() = default;
    DrawCacheMutex(const DrawCacheMutex&) = delete;
    DrawCacheMutex& operator=(const DrawCacheMutex&) = delete;

    void lock();
    bool try_lock();
    void unlock();
    void lock_shared();
    bool try_lock_shared();
    void unlock_shared();

private:
    std::shared_mutex mutex;
};

}

#endif
