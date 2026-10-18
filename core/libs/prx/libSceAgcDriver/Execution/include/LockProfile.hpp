#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_LOCKPROFILE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_LOCKPROFILE_HPP

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <utility>

namespace AgcDriver {

// The [lock] lines of the mutexes beside GpuMutex that the draw threads share (APS5_PROFILE_DRAW):
// per mutex and per role of the acquiring thread, the acquisitions, those that found the mutex
// held, the time they waited and the time the mutex was held, printed every 10 s by the first
// release after the window ends. A thread names its role once (TagLockRole); untagged threads
// count as "other".
enum class LockRole : std::uint8_t { Other, Packet, Prep, Spine, Recorder, Count };
void TagLockRole(LockRole role);
LockRole ThreadLockRole();
bool LockProfileEnabled();

class MutexProfile {
public:
    explicit MutexProfile(const char* name);
    void Acquired(bool waited, std::uint64_t waitNs);
    void Released(std::uint64_t holdNs);

private:
    struct Counters {
        std::atomic<std::uint64_t> acquisitions{0};
        std::atomic<std::uint64_t> waits{0};
        std::atomic<std::uint64_t> waitNs{0};
        std::atomic<std::uint64_t> holdNs{0};
    };
    void report(std::int64_t now);
    const char* name;
    std::array<Counters, static_cast<std::size_t>(LockRole::Count)> roles;
    std::atomic<std::int64_t> lastReport;
};

std::int64_t LockProfileNow();

// A std::unique_lock in its uses here (lock, unlock, owns_lock) that reports to a MutexProfile when
// the profile is on: the wait of every lock() (a try first, so an uncontended acquisition counts no
// wait) and the hold until unlock().
template<typename Mutex>
class ProfiledLock {
public:
    ProfiledLock(Mutex& mutex, MutexProfile& profile) : held(mutex, std::defer_lock), profile(profile) { lock(); }
    ~ProfiledLock() {
        if (held.owns_lock()) unlock();
    }
    ProfiledLock(ProfiledLock&& other) noexcept : held(std::move(other.held)), profile(other.profile), acquired(other.acquired), waitedLast(other.waitedLast) {}
    ProfiledLock(const ProfiledLock&) = delete;
    ProfiledLock& operator=(const ProfiledLock&) = delete;

    void lock() {
        if (!LockProfileEnabled()) {
            held.lock();
            return;
        }
        const auto start = LockProfileNow();
        waitedLast = !held.try_lock();
        if (waitedLast) held.lock();
        acquired = LockProfileNow();
        profile.Acquired(waitedLast, static_cast<std::uint64_t>(acquired - start));
    }
    void unlock() {
        held.unlock();
        if (LockProfileEnabled()) profile.Released(static_cast<std::uint64_t>(LockProfileNow() - acquired));
    }
    bool owns_lock() const { return held.owns_lock(); }
    // Whether the last lock() found the mutex held (false while the profile is off).
    bool waited() const { return waitedLast; }

private:
    std::unique_lock<Mutex> held;
    MutexProfile& profile;
    std::int64_t acquired = 0;
    bool waitedLast = false;
};

}

#endif
