#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPrepPool.hpp"
#include "prx/libc/include/CpuTopology.hpp"
#include <cstdio>
#include <cstdlib>
#include <exception>

namespace AgcDriver::DriverDetail {

namespace {

thread_local bool poolThread = false;
thread_local bool exemptThread = false;

std::uint64_t elapsedNs(std::chrono::steady_clock::time_point start) {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
}

}

std::size_t DrawPrepPool::Threads() {
    static const std::size_t threads = [] {
        const char* text = std::getenv("APS5_DRAW_PREP_THREADS");
        return text != nullptr ? static_cast<std::size_t>(std::strtoull(text, nullptr, 10)) : std::size_t{0};
    }();
    return threads;
}

bool DrawPrepPool::InPoolThread() {
    return poolThread;
}

void DrawPrepPool::ExemptThisThread() {
    exemptThread = true;
}

DrawPrepPool& DrawPrepPool::Get() {
    static DrawPrepPool* pool = new DrawPrepPool();
    return *pool;
}

DrawPrepPool::DrawPrepPool() {
    const auto count = Threads();
    maxInFlight = count * 3;
    for (std::size_t i = 0; i < count; ++i) threads.emplace_back([this] { run(); });
    for (auto& thread : threads) thread.detach();
}

void DrawPrepPool::run() {
    poolThread = true;
    static const std::uint64_t mask = [] {
        const auto configured = CpuTopology::MaskFromEnvironment("APS5_DRAW_PREP_AFFINITY_MASK");
        if (configured != 0) return configured;
        const auto& layout = CpuTopology::Get();
        return layout.hybrid ? layout.efficient : std::uint64_t{0};
    }();
    if (mask != 0) CpuTopology::PinTraced("draw prep", nullptr, mask);
    for (;;) {
        Task task;
        {
            std::unique_lock lock(mutex);
            work.wait(lock, [&] { return !tasks.empty(); });
            task = std::move(tasks.front());
            tasks.pop_front();
        }
        try {
            task.job(task.seq);
        } catch (const std::exception& error) {
            std::fprintf(stderr, "[prep-pool] job %llu failed: %s\n", static_cast<unsigned long long>(task.seq), error.what());
        } catch (...) {
            std::fprintf(stderr, "[prep-pool] job %llu failed\n", static_cast<unsigned long long>(task.seq));
        }
        if (turn.load(std::memory_order_acquire) <= task.seq) {
            WaitTurn(task.seq);
            FinishTurn(task.seq, {});
        }
    }
}

void DrawPrepPool::Submit(Job job) {
    std::unique_lock lock(mutex);
    if (submitted.load(std::memory_order_relaxed) - turn.load(std::memory_order_relaxed) >= maxInFlight) {
        const auto start = std::chrono::steady_clock::now();
        progress.wait(lock, [&] { return submitted.load(std::memory_order_relaxed) - turn.load(std::memory_order_relaxed) < maxInFlight; });
        submitWaitNs += elapsedNs(start);
    }
    const auto seq = submitted.load(std::memory_order_relaxed);
    inFlightSum += seq - turn.load(std::memory_order_relaxed);
    ++jobs;
    tasks.push_back({seq, std::move(job)});
    submitted.store(seq + 1, std::memory_order_release);
    lock.unlock();
    work.notify_one();
}

void DrawPrepPool::WaitTurn(std::uint64_t seq) {
    if (turn.load(std::memory_order_acquire) == seq) return;
    const auto start = std::chrono::steady_clock::now();
    std::unique_lock lock(mutex);
    progress.wait(lock, [&] { return turn.load(std::memory_order_relaxed) == seq; });
    turnWaitNs += elapsedNs(start);
}

void DrawPrepPool::FinishTurn(std::uint64_t seq, std::span<const Range> writes) {
    {
        std::lock_guard lock(mutex);
        history.emplace_back(seq, std::vector<Range>(writes.begin(), writes.end()));
        while (history.size() > 4096) history.pop_front();
        turn.store(seq + 1, std::memory_order_release);
        report();
    }
    progress.notify_all();
}

bool DrawPrepPool::WrittenBetween(std::uint64_t from, std::uint64_t to, std::span<const Range> reads) {
    if (from >= to) return false;
    std::lock_guard lock(mutex);
    if (!history.empty() && history.front().first > from) return true;
    for (const auto& [seq, writes] : history) {
        if (seq < from || seq >= to) continue;
        for (const auto& [begin, end] : writes) {
            for (const auto& [readBegin, readEnd] : reads) {
                if (readBegin < end && begin < readEnd) return true;
            }
        }
    }
    return false;
}

void DrawPrepPool::WaitAll() {
    if (poolThread || exemptThread || !Pending()) return;
    const auto start = std::chrono::steady_clock::now();
    std::unique_lock lock(mutex);
    progress.wait(lock, [&] { return submitted.load(std::memory_order_relaxed) == turn.load(std::memory_order_relaxed); });
    ++waitAlls;
    waitAllNs += elapsedNs(start);
}

void DrawPrepPool::report() {
    const auto now = std::chrono::steady_clock::now();
    if (now - lastReport < std::chrono::seconds(10)) return;
    lastReport = now;
    std::fprintf(stderr, "[prep-pool] 10 s: %llu jobs on %zu threads, avg %.1f in flight; front waited for all %llu times (%.0f ms), for room %.0f ms; turn waits %.0f ms; redos %llu, front-read conflicts %llu, skipped %llu\n", static_cast<unsigned long long>(jobs), threads.size(), jobs != 0 ? static_cast<double>(inFlightSum) / static_cast<double>(jobs) : 0.0, static_cast<unsigned long long>(waitAlls), waitAllNs / 1e6, submitWaitNs / 1e6, turnWaitNs / 1e6, static_cast<unsigned long long>(redos.exchange(0)), static_cast<unsigned long long>(frontReadConflicts.exchange(0)), static_cast<unsigned long long>(skips.exchange(0)));
    jobs = 0;
    waitAlls = 0;
    waitAllNs = 0;
    submitWaitNs = 0;
    turnWaitNs = 0;
    inFlightSum = 0;
}

}
