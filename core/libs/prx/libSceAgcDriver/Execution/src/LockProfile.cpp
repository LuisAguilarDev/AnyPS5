#include "prx/libSceAgcDriver/Execution/include/LockProfile.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>

namespace AgcDriver {

namespace {

thread_local LockRole threadRole = LockRole::Other;
constexpr std::array<const char*, static_cast<std::size_t>(LockRole::Count)> LockRoleNames{"other", "packet", "prep", "spine", "recorder"};
constexpr std::int64_t ReportWindowNs = 10'000'000'000;

}

void TagLockRole(LockRole role) {
    threadRole = role;
}

LockRole ThreadLockRole() {
    return threadRole;
}

bool LockProfileEnabled() {
    static const bool enabled = std::getenv("APS5_PROFILE_DRAW") != nullptr && std::getenv("APS5_NO_LOCK_PROFILE") == nullptr;
    return enabled;
}

std::int64_t LockProfileNow() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

MutexProfile::MutexProfile(const char* name) : name(name), lastReport(LockProfileNow()) {}

void MutexProfile::Acquired(bool waited, std::uint64_t waitNs) {
    auto& counters = roles[static_cast<std::size_t>(threadRole)];
    counters.acquisitions.fetch_add(1, std::memory_order_relaxed);
    if (!waited) return;
    counters.waits.fetch_add(1, std::memory_order_relaxed);
    counters.waitNs.fetch_add(waitNs, std::memory_order_relaxed);
}

void MutexProfile::Released(std::uint64_t holdNs) {
    roles[static_cast<std::size_t>(threadRole)].holdNs.fetch_add(holdNs, std::memory_order_relaxed);
    const auto now = LockProfileNow();
    auto last = lastReport.load(std::memory_order_relaxed);
    if (now - last >= ReportWindowNs && lastReport.compare_exchange_strong(last, now, std::memory_order_relaxed)) report(now);
}

void MutexProfile::report(std::int64_t) {
    std::string byRole;
    for (std::size_t role = 0; role < roles.size(); ++role) {
        auto& counters = roles[role];
        const auto acquisitions = counters.acquisitions.exchange(0, std::memory_order_relaxed);
        const auto waits = counters.waits.exchange(0, std::memory_order_relaxed);
        const auto waitMs = static_cast<double>(counters.waitNs.exchange(0, std::memory_order_relaxed)) / 1e6;
        const auto holdMs = static_cast<double>(counters.holdNs.exchange(0, std::memory_order_relaxed)) / 1e6;
        if (acquisitions == 0) continue;
        char part[160];
        std::snprintf(part, sizeof(part), " %s %llu/%llu, waited %.1f ms, held %.1f ms;", LockRoleNames[role], static_cast<unsigned long long>(acquisitions), static_cast<unsigned long long>(waits), waitMs, holdMs);
        byRole += part;
    }
    if (!byRole.empty()) ProfilePrint_nid_no_patch("[lock] mutex %s over 10 s, by thread role (acquisitions/waits):%s\n", name, byRole.c_str());
}

}
