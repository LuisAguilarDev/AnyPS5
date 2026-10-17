#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPrepPool.hpp"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

using AgcDriver::DriverDetail::DrawPrepPool;
using Range = DrawPrepPool::Range;

void Check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

class Gate {
public:
    void Open() {
        {
            std::lock_guard lock(mutex);
            open = true;
        }
        opened.notify_all();
    }
    void Wait() {
        std::unique_lock lock(mutex);
        opened.wait(lock, [&] { return open; });
    }

private:
    std::mutex mutex;
    std::condition_variable opened;
    bool open = false;
};

void Finish(DrawPrepPool& pool, std::uint64_t seq, std::vector<Range> writes = {}) {
    pool.WaitTurn(seq);
    pool.FinishTurn(seq, writes);
}

void RefinedJobReleasesDisjointReaders() {
    auto& pool = DrawPrepPool::Get();
    Gate refined, release;
    std::atomic<bool> accepted{false};
    const std::vector<Range> writes{{0x10000, 0x20000}};
    pool.Submit([&](std::uint64_t seq) {
        Check(pool.MayWrite(0x30000, 0x100), "a job without a bound may write anywhere");
        accepted = pool.Refine(seq, seq, {}, writes);
        refined.Open();
        release.Wait();
        Finish(pool, seq, writes);
    });
    refined.Wait();
    Check(accepted, "the only pending job could not narrow its bound");
    Check(!pool.MayWrite(0x30000, 0x100), "a refined bound still covers memory outside it");
    Check(pool.MayWrite(0x18000, 4), "a refined bound no longer covers the memory it writes");
    pool.WaitForWriters(0x30000, 0x100);
    std::atomic<bool> done{false};
    std::thread waiter([&] {
        pool.WaitForWriters(0x18000, 4);
        done = true;
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const bool early = done;
    release.Open();
    waiter.join();
    Check(!early, "a reader of the refined bound did not wait for the writer");
    Check(done, "the reader was not released when the writer finished");
    pool.WaitAll();
}

void RefineRefusedBehindPossibleWriters() {
    auto& pool = DrawPrepPool::Get();
    Gate firstHeld, release;
    std::atomic<int> refusedUnbounded{0}, refusedOverlap{0}, acceptedDisjoint{0}, acceptedAtTurn{0}, refusedWritten{0};
    pool.Submit([&](std::uint64_t seq) {
        firstHeld.Open();
        release.Wait();
        Finish(pool, seq);
    });
    firstHeld.Wait();
    Gate secondTried;
    pool.Submit([&](std::uint64_t seq) {
        const auto from = pool.Enqueued();
        const std::vector<Range> reads{{0x50000, 0x50100}};
        const std::vector<Range> writes{{0x60000, 0x60100}};
        if (!pool.Refine(seq, from, reads, writes)) ++refusedUnbounded;
        secondTried.Open();
        Finish(pool, seq);
    });
    secondTried.Wait();
    release.Open();
    pool.WaitAll();
    Check(refusedUnbounded == 1, "a bound was narrowed while an earlier job had none");

    Gate boundedHeld, releaseBounded;
    pool.Submit([&](std::uint64_t seq) {
        boundedHeld.Open();
        releaseBounded.Wait();
        Finish(pool, seq, {{0x70000, 0x70100}});
    }, std::vector<Range>{{0x70000, 0x70100}});
    boundedHeld.Wait();
    Gate thirdTried, fourthStarted;
    pool.Submit([&](std::uint64_t seq) {
        const auto from = pool.Enqueued();
        const std::vector<Range> writes{{0x60000, 0x60100}};
        const std::vector<Range> overlapping{{0x70080, 0x70090}};
        const std::vector<Range> disjoint{{0x80000, 0x80100}};
        if (!pool.Refine(seq, from, overlapping, writes)) ++refusedOverlap;
        if (pool.Refine(seq, from, disjoint, writes)) ++acceptedDisjoint;
        thirdTried.Open();
        Finish(pool, seq);
    });
    thirdTried.Wait();
    pool.Submit([&](std::uint64_t seq) {
        const auto from = pool.Enqueued();
        fourthStarted.Open();
        pool.WaitTurn(seq);
        const std::vector<Range> written{{0x70000, 0x70004}};
        if (from < seq && !pool.Refine(seq, from, written, {})) ++refusedWritten;
        if (pool.Refine(seq, seq, {}, {})) ++acceptedAtTurn;
        pool.FinishTurn(seq, {});
    });
    fourthStarted.Wait();
    releaseBounded.Open();
    pool.WaitAll();
    Check(refusedOverlap == 1, "a bound was narrowed while an earlier job's bound covered its reads");
    Check(acceptedDisjoint == 1, "a bound was not narrowed behind an earlier job writing elsewhere");
    Check(refusedWritten == 1, "a bound was narrowed after an earlier job wrote its reads");
    Check(acceptedAtTurn == 1, "a job could not narrow its bound at its turn");
}

void ReadersOfSeveralRangesWaitForAnyWriter() {
    auto& pool = DrawPrepPool::Get();
    Gate held, release;
    const std::vector<Range> writes{{0x90000, 0x90100}};
    pool.Submit([&](std::uint64_t seq) {
        held.Open();
        release.Wait();
        Finish(pool, seq, writes);
    }, writes);
    held.Wait();
    const std::vector<Range> clear{{0xa0000, 0xa0100}, {0xb0000, 0xb0100}};
    pool.WaitForWriters(clear);
    const std::vector<Range> written{{0xa0000, 0xa0100}, {0x900f0, 0x90200}};
    std::atomic<bool> done{false};
    std::thread waiter([&] {
        pool.WaitForWriters(written);
        done = true;
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const bool early = done;
    release.Open();
    waiter.join();
    Check(!early, "a reader of several ranges did not wait for the job writing its second range");
    Check(done, "the reader of several ranges was not released when the writer finished");
    pool.WaitAll();
}

}

int main() {
    try {
        Check(DrawPrepPool::Threads() >= 3, "the test needs APS5_DRAW_PREP_THREADS=3 or more");
        if (!DrawPrepPool::RefinedBounds()) {
            std::cout << "refined prep-pool bounds are off; skipped\n";
            return 77;
        }
        RefinedJobReleasesDisjointReaders();
        RefineRefusedBehindPossibleWriters();
        ReadersOfSeveralRangesWaitForAnyWriter();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
