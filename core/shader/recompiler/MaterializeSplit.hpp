#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>

namespace ShaderRecompiler {

// APS5_PROFILE_FRAME: where Recompile(request, capture) spends its time per call, timed on one
// call in SampleEvery ([frame-steps-memo] recompile_split_us/call). The parts of one sampled call
// sum to its time, so a part's total over the sampled calls divided by their count is its share of
// a call, hits and misses together. Counting only, lock-free: a call that races a Take may land in
// either window.
class MaterializeSplit {
public:
    // hash..release: the result memo path (snapshot hash, memo lookup, donor patch, then
    // materializeResult's artifact copy, binding plan and module, Populate and tail, the memo
    // insert and the release of an evicted result).
    enum Part : std::size_t { Hash, Lookup, Patch, Artifact, Program, Populate, Tail, Insert, Release, Parts };
    static constexpr std::array<const char*, Parts> Names{"hash", "lookup", "patch", "artifact", "program", "populate", "tail", "insert", "release"};
    static constexpr std::uint64_t SampleEvery = 16;

    struct Counts {
        std::uint64_t calls = 0;
        std::uint64_t sampled = 0;
        std::array<std::uint64_t, Parts> ns{};
    };

    // One sampled call's clock: Mark(part) books the time since the previous mark to `part`.
    class Sample {
    public:
        Sample() : last(std::chrono::steady_clock::now()) {}
        void Mark(Part part) {
            const auto now = std::chrono::steady_clock::now();
            ns[part] += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now - last).count());
            last = now;
        }
        std::array<std::uint64_t, Parts> ns{};

    private:
        std::chrono::steady_clock::time_point last;
    };

    // Counts the call; true for the calls to time.
    bool Sampled() { return calls.fetch_add(1, std::memory_order_relaxed) % SampleEvery == 0; }

    void Add(const Sample& sample) {
        sampled.fetch_add(1, std::memory_order_relaxed);
        for (std::size_t part = 0; part < Parts; ++part) {
            if (sample.ns[part] != 0) ns[part].fetch_add(sample.ns[part], std::memory_order_relaxed);
        }
    }

    Counts Take() {
        Counts counts{calls.exchange(0, std::memory_order_relaxed), sampled.exchange(0, std::memory_order_relaxed), {}};
        for (std::size_t part = 0; part < Parts; ++part) counts.ns[part] = ns[part].exchange(0, std::memory_order_relaxed);
        return counts;
    }

private:
    std::atomic<std::uint64_t> calls{0}, sampled{0};
    std::array<std::atomic<std::uint64_t>, Parts> ns{};
};

}
