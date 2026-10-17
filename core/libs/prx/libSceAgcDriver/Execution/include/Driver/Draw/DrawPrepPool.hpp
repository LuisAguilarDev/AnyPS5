#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAW_DRAWPREPPOOL_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAW_DRAWPREPPOOL_HPP

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <span>
#include <thread>
#include <utility>
#include <vector>

namespace AgcDriver::DriverDetail {

class DrawPrepPool {
public:
    using Range = std::pair<std::uint64_t, std::uint64_t>;
    using Job = std::function<void(std::uint64_t seq)>;

    static DrawPrepPool& Get();
    static std::size_t Threads();
    static bool InPoolThread();
    static void ExemptThisThread();

    bool Pending() const { return submitted.load(std::memory_order_acquire) != turn.load(std::memory_order_acquire); }
    std::uint64_t Enqueued() const { return turn.load(std::memory_order_acquire); }
    void Submit(Job job);
    void WaitTurn(std::uint64_t seq);
    void FinishTurn(std::uint64_t seq, std::span<const Range> writes);
    bool WrittenBetween(std::uint64_t from, std::uint64_t to, std::span<const Range> reads);
    void WaitAll();
    void NoteRedo() { redos.fetch_add(1, std::memory_order_relaxed); }
    void NoteFrontReadConflict() { frontReadConflicts.fetch_add(1, std::memory_order_relaxed); }
    void NoteSkip() { skips.fetch_add(1, std::memory_order_relaxed); }

private:
    DrawPrepPool();
    void run();
    void report();

    struct Task {
        std::uint64_t seq;
        Job job;
    };
    std::mutex mutex;
    std::condition_variable work;
    std::condition_variable progress;
    std::deque<Task> tasks;
    std::deque<std::pair<std::uint64_t, std::vector<Range>>> history;
    std::atomic<std::uint64_t> submitted{0};
    std::atomic<std::uint64_t> turn{0};
    std::vector<std::thread> threads;
    std::size_t maxInFlight = 0;
    std::atomic<std::uint64_t> redos{0};
    std::atomic<std::uint64_t> frontReadConflicts{0};
    std::atomic<std::uint64_t> skips{0};
    std::uint64_t jobs = 0;
    std::uint64_t waitAlls = 0;
    std::uint64_t waitAllNs = 0;
    std::uint64_t submitWaitNs = 0;
    std::uint64_t turnWaitNs = 0;
    std::uint64_t inFlightSum = 0;
    std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();
};

}

#endif
