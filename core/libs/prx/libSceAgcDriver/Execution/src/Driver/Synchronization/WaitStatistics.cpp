#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include <atomic>
#include <cstdlib>

namespace AgcDriver::DriverDetail {

PollStats& Driver::pollStats() {
    static thread_local PollStats stats;
    return stats;
}

WaitOutcomes& Driver::waitOutcomes() {
    static thread_local WaitOutcomes outcomes;
    return outcomes;
}

Graphics::Recorder::LateStatistics& Driver::lateCountsSeen() {
    static thread_local Graphics::Recorder::LateStatistics seen{};
    return seen;
}

EpochBumps& Driver::epochBumps() {
    static thread_local EpochBumps bumps;
    return bumps;
}

std::chrono::steady_clock::time_point& Driver::lastEpochBump() {
    static thread_local std::chrono::steady_clock::time_point at{};
    return at;
}

namespace {

std::atomic<std::uint64_t> reapBumps{0};
std::atomic<std::uint64_t> reapCommitterAdvances{0};

bool reapEpochKeep() {
    static const bool keep = std::getenv("APS5_NO_REAP_EPOCH_KEEP") == nullptr;
    return keep;
}

}

void Driver::bumpEpoch(std::uint64_t EpochBumps::*counter) {
    lastEpochBump() = std::chrono::steady_clock::now();
    GuestMemory::BumpCollectEpoch();
    const bool reap = counter == &EpochBumps::reaps;
    if (reap) reapBumps.fetch_add(1, std::memory_order_relaxed);
    if (DrawPipeline::Active() && !(reap && reapEpochKeep())) {
        DrawPipeline::EpochToken().fetch_add(1, std::memory_order_relaxed);
        if (reap) reapCommitterAdvances.fetch_add(1, std::memory_order_relaxed);
    }
    ++(epochBumps().*counter);
}

Driver::ReapEpochCounts Driver::ReapEpochs() {
    return {reapBumps.load(std::memory_order_relaxed), reapCommitterAdvances.load(std::memory_order_relaxed)};
}

bool Driver::packetEpoch() {
    static const bool packet = std::getenv("APS5_PACKET_EPOCH") != nullptr;
    return packet;
}

}
