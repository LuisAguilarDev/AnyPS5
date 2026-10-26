#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <span>

namespace ShaderRecompiler {

// APS5_PROFILE_FRAME: the program keys (variant, binding-plan key, module key) materialized in a
// window, how many of them are distinct, and how many a 1,024-entry direct-mapped table of
// program keys would have answered. Counting only, lock-free: a key that races a Take may land
// in either window.
class ProgramKeyCensus {
public:
    static constexpr std::size_t TableSlots = 1024;
    static constexpr std::size_t SetSlots = 16384;
    static constexpr std::size_t Probes = 32;

    struct Counts {
        std::uint64_t keys = 0;
        std::uint64_t distinct = 0;
        std::uint64_t tableHits = 0;
        std::uint64_t overflow = 0;
    };

    void Note(std::uint64_t key) {
        // 0 marks an empty slot.
        if (key == 0) key = 1;
        keys.fetch_add(1, std::memory_order_relaxed);
        if (table[key % TableSlots].exchange(key, std::memory_order_relaxed) == key) tableHits.fetch_add(1, std::memory_order_relaxed);
        const auto start = Mix(key);
        for (std::size_t probe = 0; probe < Probes; ++probe) {
            auto& slot = seen[(start + probe) % SetSlots];
            std::uint64_t held = slot.load(std::memory_order_relaxed);
            if (held == 0 && slot.compare_exchange_strong(held, key, std::memory_order_relaxed)) {
                distinct.fetch_add(1, std::memory_order_relaxed);
                return;
            }
            if (held == key) return;
        }
        overflow.fetch_add(1, std::memory_order_relaxed);
    }

    // The window's counts; the distinct set starts empty again, the table keeps its keys.
    Counts Take() {
        Counts counts{keys.exchange(0, std::memory_order_relaxed), distinct.exchange(0, std::memory_order_relaxed), tableHits.exchange(0, std::memory_order_relaxed), overflow.exchange(0, std::memory_order_relaxed)};
        for (auto& slot : seen) slot.store(0, std::memory_order_relaxed);
        return counts;
    }

    static std::uint64_t Mix(std::uint64_t value) {
        value ^= value >> 30u;
        value *= 0xbf58476d1ce4e5b9ull;
        value ^= value >> 27u;
        value *= 0x94d049bb133111ebull;
        return value ^ (value >> 31u);
    }

    static std::uint64_t Hash(std::span<const std::uint64_t> words) {
        std::uint64_t hash = 0x9e3779b97f4a7c15ull ^ words.size();
        for (const auto word : words) hash = Mix(hash ^ word);
        return hash;
    }

    static std::uint64_t Hash(std::span<const std::uint32_t> words) {
        std::uint64_t hash = 0xc2b2ae3d27d4eb4full ^ words.size();
        for (const auto word : words) hash = Mix(hash ^ word);
        return hash;
    }

    static std::uint64_t Combine(std::uint64_t planKey, std::uint64_t moduleKey) {
        return Mix(planKey ^ (moduleKey * 0x9e3779b97f4a7c15ull));
    }

private:
    std::atomic<std::uint64_t> keys{0}, distinct{0}, tableHits{0}, overflow{0};
    std::array<std::atomic<std::uint64_t>, TableSlots> table{};
    std::array<std::atomic<std::uint64_t>, SetSlots> seen{};
};

}
