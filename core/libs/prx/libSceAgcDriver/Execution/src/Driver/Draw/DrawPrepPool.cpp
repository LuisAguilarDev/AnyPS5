#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPrepPool.hpp"
#include "prx/libc/include/CpuTopology.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <exception>

namespace AgcDriver::DriverDetail {

namespace {

thread_local bool poolThread = false;
thread_local bool exemptThread = false;
thread_local std::uint64_t threadWaitedNs = 0;

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

bool DrawPrepPool::WriteBounds() {
    static const bool bounded = std::getenv("APS5_NO_PREP_POOL_WRITE_BOUNDS") == nullptr;
    return bounded;
}

bool DrawPrepPool::VerifyWriteBounds() {
    static const bool verify = std::getenv("APS5_VERIFY_PREP_POOL_WRITE_BOUNDS") != nullptr;
    return verify;
}

bool DrawPrepPool::RefinedBounds() {
    static const bool refined = WriteBounds() && std::getenv("APS5_NO_PREP_POOL_REFINED_BOUNDS") == nullptr;
    return refined;
}

double DrawPrepPool::ThreadWaitedMs() {
    return static_cast<double>(threadWaitedNs) / 1e6;
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

void DrawPrepPool::Submit(Job job, std::optional<std::vector<Range>> bound) {
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
    bounds.emplace_back(seq, std::move(bound));
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
        const auto bound = std::find_if(bounds.begin(), bounds.end(), [&](const auto& pending) { return pending.first == seq; });
        if (bound != bounds.end()) {
            if (VerifyWriteBounds() && bound->second) {
                for (const auto& [begin, end] : writes) {
                    if (std::any_of(bound->second->begin(), bound->second->end(), [&](const Range& range) { return range.first <= begin && end <= range.second; })) continue;
                    std::fprintf(stderr, "APS5_VERIFY_PREP_POOL_WRITE_BOUNDS: prep-pool job %llu wrote 0x%llx-0x%llx outside the bound it was submitted with\n", static_cast<unsigned long long>(seq), static_cast<unsigned long long>(begin), static_cast<unsigned long long>(end));
                    std::abort();
                }
            }
            bounds.erase(bound);
        }
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

bool DrawPrepPool::mayWriteLocked(std::uint64_t address, std::size_t bytes) const {
    const auto end = address + bytes;
    for (const auto& [seq, bound] : bounds) {
        if (!bound) return true;
        for (const auto& [begin, limit] : *bound) {
            if (address < limit && begin < end) return true;
        }
    }
    return false;
}

bool DrawPrepPool::MayWrite(std::uint64_t address, std::size_t bytes) {
    boundedChecks.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard lock(mutex);
    if (mayWriteLocked(address, bytes)) return true;
    if (!bounds.empty()) waitsAvoided.fetch_add(1, std::memory_order_relaxed);
    return false;
}

bool DrawPrepPool::Refine(std::uint64_t seq, std::uint64_t from, std::span<const Range> reads, std::span<const Range> writes) {
    const auto overlapsReads = [&](std::span<const Range> ranges) {
        return std::any_of(ranges.begin(), ranges.end(), [&](const Range& range) { return std::any_of(reads.begin(), reads.end(), [&](const Range& read) { return read.first < range.second && range.first < read.second; }); });
    };
    {
        std::lock_guard lock(mutex);
        const auto bound = std::find_if(bounds.begin(), bounds.end(), [&](const auto& pending) { return pending.first == seq; });
        if (bound == bounds.end() || bound->second) return false;
        if (from < seq) {
            if (!history.empty() && history.front().first > from) return false;
            for (const auto& [done, written] : history) {
                if (done >= from && done < seq && overlapsReads(written)) return false;
            }
            for (const auto& [earlier, range] : bounds) {
                if (earlier >= seq) continue;
                if (!range || overlapsReads(*range)) return false;
            }
        }
        bound->second.emplace(writes.begin(), writes.end());
        ++(from < seq ? refinedEarly : refinedAtTurn);
    }
    progress.notify_all();
    return true;
}

bool DrawPrepPool::mayWriteAnyLocked(std::span<const Range> ranges) const {
    return std::any_of(ranges.begin(), ranges.end(), [&](const Range& range) { return range.second > range.first && mayWriteLocked(range.first, static_cast<std::size_t>(range.second - range.first)); });
}

void DrawPrepPool::WaitForWriters(std::uint64_t address, std::size_t bytes) {
    const Range range{address, address + bytes};
    WaitForWriters(std::span(&range, 1));
}

void DrawPrepPool::WaitForWriters(std::span<const Range> ranges) {
    if (poolThread || exemptThread || !Pending()) return;
    boundedChecks.fetch_add(1, std::memory_order_relaxed);
    std::unique_lock lock(mutex);
    if (!mayWriteAnyLocked(ranges)) {
        if (!bounds.empty()) waitsAvoided.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    const auto start = std::chrono::steady_clock::now();
    progress.wait(lock, [&] { return !mayWriteAnyLocked(ranges); });
    const auto waited = elapsedNs(start);
    ++writerWaits;
    writerWaitNs += waited;
    threadWaitedNs += waited;
}

void DrawPrepPool::WaitAll() {
    if (poolThread || exemptThread || !Pending()) return;
    const auto start = std::chrono::steady_clock::now();
    std::unique_lock lock(mutex);
    progress.wait(lock, [&] { return submitted.load(std::memory_order_relaxed) == turn.load(std::memory_order_relaxed); });
    const auto waited = elapsedNs(start);
    ++waitAlls;
    waitAllNs += waited;
    threadWaitedNs += waited;
}

void DrawPrepPool::report() {
    const auto now = std::chrono::steady_clock::now();
    if (now - lastReport < std::chrono::seconds(10)) return;
    lastReport = now;
    std::fprintf(stderr, "[prep-pool] 10 s: %llu jobs on %zu threads, avg %.1f in flight; front waited for all %llu times (%.0f ms), for room %.0f ms; turn waits %.0f ms; redos %llu, front-read conflicts %llu, skipped %llu; write-bound checks %llu, waits avoided %llu; bounds refined %llu once compiled / %llu at their turn, waits for writers %llu (%.0f ms)\n", static_cast<unsigned long long>(jobs), threads.size(), jobs != 0 ? static_cast<double>(inFlightSum) / static_cast<double>(jobs) : 0.0, static_cast<unsigned long long>(waitAlls), waitAllNs / 1e6, submitWaitNs / 1e6, turnWaitNs / 1e6, static_cast<unsigned long long>(redos.exchange(0)), static_cast<unsigned long long>(frontReadConflicts.exchange(0)), static_cast<unsigned long long>(skips.exchange(0)), static_cast<unsigned long long>(boundedChecks.exchange(0)), static_cast<unsigned long long>(waitsAvoided.exchange(0)), static_cast<unsigned long long>(refinedEarly), static_cast<unsigned long long>(refinedAtTurn), static_cast<unsigned long long>(writerWaits), writerWaitNs / 1e6);
    jobs = 0;
    refinedEarly = 0;
    refinedAtTurn = 0;
    writerWaits = 0;
    writerWaitNs = 0;
    waitAlls = 0;
    waitAllNs = 0;
    submitWaitNs = 0;
    turnWaitNs = 0;
    inFlightSum = 0;
}

}
