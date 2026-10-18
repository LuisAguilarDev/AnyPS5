#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawCacheMutex.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPrepPool.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace AgcDriver::DriverDetail {

namespace {

bool profiled() {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr && std::getenv("APS5_NO_LOCK_PROFILE") == nullptr;
    return profile;
}

struct Side {
    std::uint64_t acquisitions = 0, waits = 0, waitNs = 0, holdNs = 0, maxHoldNs = 0;
};

struct Stats {
    Side exclusive, shared;
    std::chrono::steady_clock::time_point heldSince{};
    std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();
};

Stats& stats() {
    thread_local Stats owned;
    return owned;
}

std::uint64_t nanoseconds(std::chrono::steady_clock::duration duration) {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count());
}

template <typename TTry, typename TLock>
void acquire(Side& side, TTry&& tryLock, TLock&& lock) {
    auto& owned = stats();
    ++side.acquisitions;
    if (!tryLock()) {
        const auto start = std::chrono::steady_clock::now();
        lock();
        owned.heldSince = std::chrono::steady_clock::now();
        ++side.waits;
        side.waitNs += nanoseconds(owned.heldSince - start);
        return;
    }
    owned.heldSince = std::chrono::steady_clock::now();
}

void report(Stats& owned, std::chrono::steady_clock::time_point now) {
    if (now - owned.lastReport < std::chrono::seconds(10)) return;
    const auto interval = std::chrono::duration<double>(now - owned.lastReport).count();
    owned.lastReport = now;
    char thread[48];
    if (DrawPrepPool::InPoolThread()) std::snprintf(thread, sizeof(thread), "a prep-pool thread");
    else if (GuestMemory::GpuLockThreadTag() == 0xffffffffu) std::snprintf(thread, sizeof(thread), "an untagged thread");
    else std::snprintf(thread, sizeof(thread), "queue 0x%x", GuestMemory::GpuLockThreadTag());
    const auto& e = owned.exclusive;
    const auto& s = owned.shared;
    std::fprintf(stderr, "[lock] draw cache mutex on %s (%.0f s): exclusive %llu/%llu waited %.1f ms, held %.1f ms (max %.3f); shared %llu/%llu waited %.1f ms, held %.1f ms (max %.3f)\n", thread, interval, static_cast<unsigned long long>(e.acquisitions), static_cast<unsigned long long>(e.waits), e.waitNs / 1e6, e.holdNs / 1e6, e.maxHoldNs / 1e6, static_cast<unsigned long long>(s.acquisitions), static_cast<unsigned long long>(s.waits), s.waitNs / 1e6, s.holdNs / 1e6, s.maxHoldNs / 1e6);
    owned.exclusive = {};
    owned.shared = {};
}

// The report is printed after the unlock, so no thread waits behind the line.
std::chrono::steady_clock::time_point release(Side& side) {
    auto& owned = stats();
    const auto now = std::chrono::steady_clock::now();
    const auto held = nanoseconds(now - owned.heldSince);
    side.holdNs += held;
    side.maxHoldNs = std::max(side.maxHoldNs, held);
    return now;
}

}

void DrawCacheMutex::lock() {
    if (!profiled()) return mutex.lock();
    acquire(stats().exclusive, [&] { return mutex.try_lock(); }, [&] { mutex.lock(); });
}

bool DrawCacheMutex::try_lock() {
    if (!mutex.try_lock()) return false;
    if (profiled()) {
        ++stats().exclusive.acquisitions;
        stats().heldSince = std::chrono::steady_clock::now();
    }
    return true;
}

void DrawCacheMutex::unlock() {
    if (!profiled()) return mutex.unlock();
    const auto now = release(stats().exclusive);
    mutex.unlock();
    report(stats(), now);
}

void DrawCacheMutex::lock_shared() {
    if (!profiled()) return mutex.lock_shared();
    acquire(stats().shared, [&] { return mutex.try_lock_shared(); }, [&] { mutex.lock_shared(); });
}

bool DrawCacheMutex::try_lock_shared() {
    if (!mutex.try_lock_shared()) return false;
    if (profiled()) {
        ++stats().shared.acquisitions;
        stats().heldSince = std::chrono::steady_clock::now();
    }
    return true;
}

void DrawCacheMutex::unlock_shared() {
    if (!profiled()) return mutex.unlock_shared();
    const auto now = release(stats().shared);
    mutex.unlock_shared();
    report(stats(), now);
}

}
