#include "prx/libSceAgcDriver/Graphics/include/EpochProof.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/GuestWriteWatch.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#endif
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

extern "C" void AgcDriverStoreGuestLabel_nid_postfix(std::uint64_t* pointer, std::uint64_t value);

namespace {

namespace Graphics = AgcDriver::Graphics;
namespace GuestMemory = AgcDriver::GuestMemory;
using Graphics::EpochProofUse;

constexpr std::size_t Block = 65536;
constexpr std::uint64_t Pending = 7;

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::byte* AllocateGuest(std::size_t bytes) {
#ifdef _WIN32
    void* block = GuestArena::GuestArenaAllocate_nid_postfix(bytes, Block);
    GuestArena::GuestArenaCommit_nid_postfix(block, bytes, PAGE_READWRITE, bytes);
#else
    void* block = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (block == MAP_FAILED) throw std::runtime_error("cannot map the guest block");
    GuestWriteWatch::GuestWriteWatchRegister_nid_postfix(block, bytes);
#endif
    std::memset(block, 0x11, bytes);
    return static_cast<std::byte*>(block);
}

EpochProofUse Use(const Graphics::EpochProofToken& proved, bool verify = false) {
    std::uint64_t reuses = 0;
    return Graphics::ChooseEpochProofUse(proved, Graphics::CurrentEpochProof(Pending), verify, reuses);
}

void CheckReuseInsideWindow() {
    GuestMemory::BumpCollectEpoch();
    const auto proved = Graphics::CurrentEpochProof(Pending);
    std::uint64_t reuses = 0;
    for (int use = 0; use < 10; ++use) {
        Require(Graphics::ChooseEpochProofUse(proved, Graphics::CurrentEpochProof(Pending), false, reuses) == EpochProofUse::Reuse, "a repeated revalidate inside the proof window was not skipped");
    }
    Require(Graphics::ChooseEpochProofUse(Graphics::EpochProofToken{}, Graphics::CurrentEpochProof(Pending), false, reuses) == EpochProofUse::Full, "an object never proved reused a proof");
}

void CheckWritersForceFullRevalidate() {
    auto* memory = AllocateGuest(2 * Block);
    const auto base = reinterpret_cast<std::uint64_t>(memory);
    std::array<std::byte, 16> bytes{};
    bytes.fill(std::byte{0x5a});
    std::array<std::byte, 16> original{};
    std::memcpy(original.data(), memory + 64, original.size());
    const std::vector<std::pair<std::string, std::function<void()>>> writers{
        {"a driver store (StoreOwnBytes)", [&] { GuestMemory::StoreOwnBytes(base, bytes); }},
        {"a driver write (Write)", [&] { GuestMemory::Write(base + 32, bytes, 4); }},
        {"a written mark (MarkWritten)", [&] { GuestMemory::MarkWritten(base + 128, 16); }},
        {"a possibly-written mark (MarkPossiblyWritten)", [&] { GuestMemory::MarkPossiblyWritten(base + 256, 16); }},
        {"a write-back (WriteChanged)", [&] { GuestMemory::WriteChanged(base + 64, bytes, original); }},
        {"a completion or flip label store", [&] { AgcDriverStoreGuestLabel_nid_postfix(reinterpret_cast<std::uint64_t*>(memory + Block), 1); }},
        {"a non-draw commit", [] { GuestMemory::BreakProofs(GuestMemory::ProofBreaker::NonDrawCommit); }},
        {"a recorded completion label", [] { GuestMemory::BreakProofs(GuestMemory::ProofBreaker::CompletionLabel); }},
        {"a GPU lock taken by a thread other than the committer", [] {
            std::thread([] { std::lock_guard lock(GuestMemory::GpuMutex()); }).join();
        }},
        {"an ordering point (a CPU store may be visible)", [] { GuestMemory::BumpCollectEpoch(); }},
    };
    for (const auto& [name, write] : writers) {
        GuestMemory::BumpCollectEpoch();
        const auto proved = Graphics::CurrentEpochProof(Pending);
        Require(Use(proved) == EpochProofUse::Reuse, "the proof did not hold before " + name);
        write();
        Require(Use(proved) == EpochProofUse::Full, name + " did not force a full revalidate");
    }
    GuestMemory::BumpCollectEpoch();
    const auto proved = Graphics::CurrentEpochProof(Pending);
    std::uint64_t reuses = 0;
    Require(Graphics::ChooseEpochProofUse(proved, Graphics::CurrentEpochProof(Pending + 1), false, reuses) == EpochProofUse::Full, "a new pending image result did not force a full revalidate");
}

void CheckCommitterLockKeepsProof() {
    std::thread([] {
        GuestMemory::MarkProofThread();
        GuestMemory::BumpCollectEpoch();
        const auto proved = Graphics::CurrentEpochProof(Pending);
        { std::lock_guard lock(GuestMemory::GpuMutex()); }
        Require(Use(proved) == EpochProofUse::Reuse, "the committer's own GPU lock broke its proofs");
    }).join();
}

void CheckThreadWithoutEpochProvesNothing() {
    std::thread([] {
        const auto proved = Graphics::CurrentEpochProof(Pending);
        Require(Use(proved) == EpochProofUse::Full, "a thread without ordering points reused a proof");
    }).join();
}

// A writer that changed a proved input without breaking the proofs (the injected stale proof):
// the token still holds, and only the verified reuse finds the change.
void CheckVerifyCatchesStaleProof() {
    auto* memory = AllocateGuest(Block);
    std::vector<std::byte> snapshot(memory, memory + 64);
    const auto current = [&] { return std::memcmp(snapshot.data(), memory, snapshot.size()) == 0; };
    GuestMemory::BumpCollectEpoch();
    const auto proved = Graphics::CurrentEpochProof(Pending);
    memory[0] = std::byte{0x77};
    const auto run = [&](bool verify) {
        std::uint64_t reuses = 0;
        std::uint64_t verified = 0;
        std::uint64_t mismatches = 0;
        for (std::uint64_t use = 1; use <= 2 * Graphics::EpochProofVerifyEvery; ++use) {
            const auto chosen = Graphics::ChooseEpochProofUse(proved, Graphics::CurrentEpochProof(Pending), verify, reuses);
            Require(chosen != EpochProofUse::Full, "the injected stale proof did not hold its window");
            if (chosen != EpochProofUse::Verify) continue;
            Require(use % Graphics::EpochProofVerifyEvery == 0, "a reuse other than 1 in 64 was verified");
            ++verified;
            const bool mismatch = !current();
            if (mismatch) ++mismatches;
            Graphics::NoteEpochProofUse(Graphics::EpochProofKind::Resources, chosen, mismatch);
        }
        return std::pair{verified, mismatches};
    };
    Require(run(false) == std::pair<std::uint64_t, std::uint64_t>{0, 0}, "a reuse was verified without verify mode");
    auto& counters = AgcDriver::FrameSteps::Global();
    const auto checksBefore = counters.proofVerifyChecks.load();
    const auto mismatchesBefore = counters.proofVerifyMismatches.load();
    Require(run(true) == std::pair<std::uint64_t, std::uint64_t>{2, 2}, "verify mode did not check 1 reuse in 64 and find the stale proof");
    if (AgcDriver::FrameSteps::Enabled()) {
        Require(counters.proofVerifyChecks.load() - checksBefore == 2 && counters.proofVerifyMismatches.load() - mismatchesBefore == 2, "the verified reuses and their mismatches are not counted for [frame-steps-memo]");
    }
}

}

int main(int argc, char** argv) {
    try {
        // Epoch proofs are on by default, without APS5_DRAW_PIPELINE; the _off test runs with
        // APS5_NO_EPOCH_PROOFS=1 and expects them off.
        const bool expectOff = argc == 2 && std::string(argv[1]) == "--expect-off";
        Require(Graphics::EpochProofsEnabled() == !expectOff, expectOff ? "APS5_NO_EPOCH_PROOFS=1 left epoch proofs on" : "epoch proofs are not on by default");
        if (expectOff) {
            std::cout << "epoch proof switch test passed\n";
            return 0;
        }
        CheckReuseInsideWindow();
        CheckWritersForceFullRevalidate();
        CheckCommitterLockKeepsProof();
        CheckThreadWithoutEpochProvesNothing();
        CheckVerifyCatchesStaleProof();
    } catch (const std::exception& error) {
        std::cerr << "epoch proof test failed: " << error.what() << "\n";
        return 1;
    }
    std::cout << "epoch proof tests passed\n";
    return 0;
}
