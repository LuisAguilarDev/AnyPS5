#pragma once

#include "prx/libSceAgcDriver/Execution/include/FrameSteps.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include <cstdint>
#include <cstdlib>

namespace AgcDriver::Graphics {

// The window an epoch proof holds for (ShaderResources::EpochProofs): a resource template or a
// storage image's refresh proved current keeps its proof while the calling thread's collect epoch
// (no CPU store became visible), GuestMemory::ProofSerial (no driver, GPU or foreign-lock writer
// broke the proofs) and the pending registry's serial (no image result became pending) are all
// still the ones read before the full check that proved it. Epoch 0 (a thread without ordering
// points, APS5_NO_COLLECT_MEMO=1) proves nothing.
struct EpochProofToken {
    std::uint64_t epoch = 0;
    std::uint64_t proof = 0;
    std::uint64_t pending = 0;
    bool operator==(const EpochProofToken&) const = default;
};

// On by default, with or without APS5_DRAW_PIPELINE (they never call the draw finisher);
// APS5_NO_EPOCH_PROOFS=1 runs every Revalidate and storage refresh in full, as before.
inline bool EpochProofsEnabled() {
    static const bool enabled = std::getenv("APS5_NO_EPOCH_PROOFS") == nullptr;
    return enabled;
}

inline EpochProofToken CurrentEpochProof(std::uint64_t pendingSerial) {
    return {GuestMemory::CollectEpoch(), GuestMemory::ProofSerial(), pendingSerial};
}

// How a Revalidate or a storage refresh uses the token it last proved: the full check (no proof,
// or the window moved), the proof reused, or the full check run next to a reuse to verify it.
enum class EpochProofUse { Full, Reuse, Verify };

// APS5_VERIFY_DRAW_PIPELINE=1: 1 reuse in this many runs the full check instead, and a check that
// finds the proved object not current is a mismatch (a proof reused over a real change).
inline constexpr std::uint64_t EpochProofVerifyEvery = 64;

inline bool VerifyEpochProofs() {
    static const bool verify = std::getenv("APS5_VERIFY_DRAW_PIPELINE") != nullptr;
    return verify;
}

// `reuses` counts the caller's reuses, so every EpochProofVerifyEvery-th one is verified.
inline EpochProofUse ChooseEpochProofUse(const EpochProofToken& proved, const EpochProofToken& now, bool verify, std::uint64_t& reuses) {
    if (now.epoch == 0 || !(proved == now)) return EpochProofUse::Full;
    if (verify && ++reuses % EpochProofVerifyEvery == 0) return EpochProofUse::Verify;
    return EpochProofUse::Reuse;
}

// [frame-steps-memo] counts (APS5_PROFILE_FRAME): the proofs reused and taken by a full check, by
// the template proofs and the storage refresh proofs, and the verified reuses and their mismatches.
enum class EpochProofKind { Resources, Refresh };

inline void NoteEpochProofUse(EpochProofKind kind, EpochProofUse use, bool mismatch) {
    if (!FrameSteps::Enabled()) return;
    auto& counters = FrameSteps::Global();
    const bool resources = kind == EpochProofKind::Resources;
    if (use == EpochProofUse::Reuse) FrameSteps::Counters::Add(resources ? counters.epochProofReuses : counters.refreshProofReuses, 1);
    else FrameSteps::Counters::Add(resources ? counters.epochProofFulls : counters.refreshProofFulls, 1);
    if (use != EpochProofUse::Verify) return;
    FrameSteps::Counters::Add(counters.proofVerifyChecks, 1);
    if (mismatch) FrameSteps::Counters::Add(counters.proofVerifyMismatches, 1);
}

}
