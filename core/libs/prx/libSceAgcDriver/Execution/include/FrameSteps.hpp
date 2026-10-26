#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_FRAMESTEPS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_FRAMESTEPS_HPP

#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawTiming.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <utility>

namespace AgcDriver::FrameSteps {

inline constexpr std::size_t LaneThreads = 16;
inline constexpr std::size_t LockRoles = 5;
inline constexpr std::size_t SampleCapacity = 4096;
inline constexpr std::size_t FlipQueueCapacity = 64;
inline constexpr std::int64_t ReportWindowNs = 10'000'000'000;
inline constexpr std::size_t WaitEqueue = 0, WaitSleep = 1, WaitVblank = 2, WaitEventFlag = 3, WaitSemaphore = 4, WaitCond = 5, WaitAddress = 6, KernelWaitKinds = 7;
inline constexpr std::size_t PacketSlots = 258, FlipSlot = 256, RenderWaitSlot = 257, TopPacketSlots = 12;
inline constexpr std::size_t PhaseGate = 0, PhaseDrain = 1, PhaseLock = 2, PhaseFlush = 3, PhasePrepare = 4, PhaseBody = 5, PacketPhases = 6;
inline constexpr std::size_t DrawSteps = DriverDetail::DrawDriverPhaseCount;
// DrawPipeline::CommitKind, then the enqueues made from a draw prep pool thread.
inline constexpr std::size_t EnqueueKinds = 10, PoolEnqueueKind = 9;
// DrawPipeline::DrainReason; drain opcodes 0x00-0xff and 0x100 for a drain without a packet.
inline constexpr std::size_t DrainReasons = 6, DrainOpcodes = 257, NoDrainOpcode = 256, TopDrainOpcodes = 6;
// The Graphics draw phases (Draw.cpp DrawPhase) and the rest of a sampled commit.
inline constexpr std::size_t SpineDrawPhases = 13, SpineOtherPhase = SpineDrawPhases, SpinePhases = SpineDrawPhases + 1;
// GuestMemory::ProofBreaker.
inline constexpr std::size_t ProofBreakers = 9;
inline constexpr std::size_t RoleOther = 0, RolePacket = 1, RoleSpine = 2, RolePool = 3, RoleRecorder = 4, RolePresenter = 5, RoleRelease = 6, ThreadRoles = 7;
inline constexpr std::size_t TriggerOther = 0, TriggerCap = 1, TriggerDeadline = 2, TriggerLabel = 3, TriggerCompletion = 4, TriggerFlip = 5, TriggerPresent = 6, TriggerEnd = 7, TriggerIdle = 8, TriggerWait = 9, TriggerSuspend = 10, SubmitTriggers = 11;
inline constexpr std::size_t CpuPacket = 0, CpuSpine = 1, CpuPresenter = 2, CpuRelease = 3, CpuRecorders = 4, CpuPool = CpuRecorders + LaneThreads, CpuSlots = CpuPool + LaneThreads;
inline constexpr std::uint32_t DefaultSampleEvery = 16, MaxSampleEvery = 1024;
inline constexpr std::uint64_t DigestOffset = 0xcbf29ce484222325ull, DigestPrime = 0x100000001b3ull;

inline bool Enabled() {
    static const bool enabled = std::getenv("APS5_PROFILE_FRAME") != nullptr;
    return enabled;
}

// APS5_PROFILE_FRAME_SAMPLE: one in how many draw commits the spine phase split times (a power
// of two up to 1024; anything else keeps 1 in 16).
inline std::uint32_t SampleEvery(const char* text) {
    if (text == nullptr || *text == '\0') return DefaultSampleEvery;
    char* end = nullptr;
    const auto value = std::strtoul(text, &end, 10);
    if (*end != '\0' || value == 0 || value > MaxSampleEvery || (value & (value - 1)) != 0) return DefaultSampleEvery;
    return static_cast<std::uint32_t>(value);
}

inline std::uint32_t SpineSampleEvery() {
    static const std::uint32_t every = SampleEvery(std::getenv("APS5_PROFILE_FRAME_SAMPLE"));
    return every;
}

// Picks the first of every `every` calls (a power of two); one owner thread.
class Sampler {
public:
    explicit Sampler(std::uint32_t every) : mask(every - 1) {}
    bool Next() { return (count++ & mask) == 0; }

private:
    std::uint32_t mask;
    std::uint32_t count = 0;
};

inline std::uint64_t DigestWords(std::span<const std::uint32_t> words, std::uint64_t digest = DigestOffset) {
    for (const auto word : words) {
        digest ^= word;
        digest *= DigestPrime;
    }
    return digest;
}

inline std::int64_t NowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

inline double PerFrame(double total, std::uint64_t frames) {
    return frames != 0 ? total / static_cast<double>(frames) : 0.0;
}

inline double PerItem(double total, std::uint64_t items) {
    return items != 0 ? total / static_cast<double>(items) : 0.0;
}

struct Spread {
    double p50 = 0;
    double p95 = 0;
    double mean = 0;
    std::size_t count = 0;
};

class SampleWindow {
public:
    void Add(double value) {
        sum += value;
        ++seen;
        if (count < values.size()) values[count++] = value;
    }

    Spread Take() {
        Spread spread;
        spread.count = static_cast<std::size_t>(seen);
        spread.mean = seen != 0 ? sum / static_cast<double>(seen) : 0.0;
        if (count != 0) {
            std::sort(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(count));
            const auto at = [&](double fraction) { return values[std::min(count - 1, static_cast<std::size_t>(fraction * static_cast<double>(count)))]; };
            spread.p50 = at(0.5);
            spread.p95 = at(0.95);
        }
        count = 0;
        seen = 0;
        sum = 0;
        return spread;
    }

private:
    std::array<double, SampleCapacity> values{};
    std::size_t count = 0;
    std::uint64_t seen = 0;
    double sum = 0;
};

struct ClosedFrame {
    double gapMs = -1;
    double spanMs = 0;
    double insideMs = 0;
    std::uint32_t submits = 0;
};

class GuestFrames {
public:
    bool Submit(std::int64_t entryNs, std::int64_t exitNs, bool flip, ClosedFrame& closed) {
        if (!open) {
            open = true;
            firstEntryNs = entryNs;
            insideNs = 0;
            submits = 0;
        }
        insideNs += std::max<std::int64_t>(0, exitNs - entryNs);
        ++submits;
        if (!flip) return false;
        closed.spanMs = static_cast<double>(std::max<std::int64_t>(0, exitNs - firstEntryNs)) / 1e6;
        closed.gapMs = lastFlipExitNs != 0 ? static_cast<double>(std::max<std::int64_t>(0, firstEntryNs - lastFlipExitNs)) / 1e6 : -1.0;
        closed.insideMs = static_cast<double>(insideNs) / 1e6;
        closed.submits = submits;
        lastFlipExitNs = exitNs;
        open = false;
        return true;
    }

private:
    std::int64_t firstEntryNs = 0;
    std::int64_t lastFlipExitNs = 0;
    std::int64_t insideNs = 0;
    std::uint32_t submits = 0;
    bool open = false;
};

struct Totals {
    std::uint64_t submits = 0, submitNs = 0, flipSubmits = 0, flipSubmitNs = 0, eopReserves = 0, eopReserveNs = 0;
    std::uint64_t flipRoomNs = 0, flipHoldNs = 0, registrationNs = 0, renderWaitNs = 0;
    std::uint64_t dwords = 0, packets = 0, draws = 0;
    std::uint64_t queueLatencyNs = 0, queueLatencies = 0;
    std::uint64_t packetBusyNs = 0, packetIdleNs = 0, packetPackets = 0, packetSubmissions = 0, packetPoolWaitNs = 0;
    std::array<std::uint64_t, LockRoles> lockWaitNs{}, lockHoldNs{};
    std::array<std::uint64_t, LaneThreads> prepBusyNs{};
    std::uint64_t prepTurnWaitNs = 0, prepJobs = 0;
    std::uint64_t fifoFullNs = 0, fifoDrainNs = 0, fifoEmptyNs = 0, fifoDepthSum = 0, fifoEnqueued = 0;
    std::uint64_t spineBusyNs = 0, spineCommits = 0, drawCommits = 0, drawCommitNs = 0;
    std::array<std::uint64_t, LaneThreads> recorderBusyNs{};
    std::uint64_t recorderChunks = 0, recorderDraws = 0, keepSealedChunks = 0, chunkEpochJoins = 0, staleRecordInputs = 0, staleRecordResources = 0;
    std::uint64_t vkSubmits = 0, vkSubmitNs = 0;
    std::uint64_t executeNs = 0, drawStepDraws = 0;
    std::array<std::uint64_t, PacketSlots> slotPackets{}, slotNs{};
    std::array<std::uint64_t, PacketPhases> drawPhaseNs{}, otherPhaseNs{};
    std::array<std::uint64_t, 2> phasePackets{};
    std::array<std::uint64_t, DrawSteps> drawStepNs{};
    std::array<std::uint64_t, EnqueueKinds> fifoFullKindNs{};
    std::array<std::uint64_t, DrainReasons> fifoDrainReasonNs{}, fifoDrainReasonCount{};
    std::array<std::uint64_t, DrainOpcodes> fifoDrainOpcodeNs{}, fifoDrainOpcodeCount{};
    std::uint64_t mpassInline = 0, mpassPipelined = 0;
    std::uint64_t spineSamples = 0, spineSampleNs = 0;
    std::array<std::uint64_t, SpinePhases> spinePhaseNs{};
    std::uint64_t submitValidateNs = 0;
    std::array<std::uint64_t, ProofBreakers> proofBreaks{};
    std::uint64_t targetMemoLookups = 0, targetMemoHits = 0, drawInputMemoLookups = 0, drawInputMemoHits = 0;
    std::uint64_t drawStages = 0, stagedDraws = 0, halfChecks = 0, halfStale = 0;
    std::array<std::uint64_t, ThreadRoles * SubmitTriggers> vkSubmitsBy{};
};

struct ChainTotals {
    std::uint64_t firstSubmitNs = 0, firstSubmits = 0, tailNs = 0, tails = 0;
};

// The recompiler's counts over the window (ShaderRecompiler::ResultMemoCounts deltas and the
// program-key census).
struct ShaderTotals {
    std::uint64_t resultMemoHits = 0, resultMemoMisses = 0, resultMemoEvictions = 0;
    std::uint64_t programKeys = 0, programKeysDistinct = 0, programKeyTableHits = 0, programKeyOverflow = 0;
};

struct Window {
    double seconds = 0;
    std::uint64_t frames = 0;
    std::uint64_t presents = 0;
    double gpuBusyMs = 0;
    double gpuGapMs = 0;
    std::array<std::uint64_t, KernelWaitKinds> kernelWaitNs{};
    Spread gapMs, spanMs, insideMs, intervalMs, flipToPresentMs;
    ChainTotals chain;
    std::array<std::uint64_t, CpuSlots> threadCpuNs{};
    ShaderTotals shaders;
};

// The lanes line's derived values: the packet thread's own work, and what the lanes' overlap
// loses (coupling) and the frame's end costs (tail) beyond them.
struct Derived {
    double frameMs = 0, packetOwnMs = 0, couplingMs = 0, tailMs = 0;
};

class Counters {
public:
    using Counter = std::atomic<std::uint64_t>;

    static void Add(Counter& counter, std::uint64_t value) { counter.fetch_add(value, std::memory_order_relaxed); }
    static void AddNs(Counter& counter, std::int64_t ns) { counter.fetch_add(static_cast<std::uint64_t>(std::max<std::int64_t>(0, ns)), std::memory_order_relaxed); }
    static std::size_t Slot(std::size_t index) { return std::min(index, LaneThreads - 1); }

    Totals Take() {
        Totals totals;
        const auto take = [](Counter& counter) { return counter.exchange(0, std::memory_order_relaxed); };
        totals.submits = take(submits);
        totals.submitNs = take(submitNs);
        totals.flipSubmits = take(flipSubmits);
        totals.flipSubmitNs = take(flipSubmitNs);
        totals.flipRoomNs = take(flipRoomNs);
        totals.flipHoldNs = take(flipHoldNs);
        totals.registrationNs = take(registrationNs);
        totals.renderWaitNs = take(renderWaitNs);
        totals.eopReserves = take(eopReserves);
        totals.eopReserveNs = take(eopReserveNs);
        totals.dwords = take(dwords);
        totals.packets = take(packets);
        totals.draws = take(draws);
        totals.queueLatencyNs = take(queueLatencyNs);
        totals.queueLatencies = take(queueLatencies);
        totals.packetBusyNs = take(packetBusyNs);
        totals.packetIdleNs = take(packetIdleNs);
        totals.packetPackets = take(packetPackets);
        totals.packetSubmissions = take(packetSubmissions);
        totals.packetPoolWaitNs = take(packetPoolWaitNs);
        for (std::size_t i = 0; i < LockRoles; ++i) {
            totals.lockWaitNs[i] = take(lockWaitNs[i]);
            totals.lockHoldNs[i] = take(lockHoldNs[i]);
        }
        for (std::size_t i = 0; i < LaneThreads; ++i) {
            totals.prepBusyNs[i] = take(prepBusyNs[i]);
            totals.recorderBusyNs[i] = take(recorderBusyNs[i]);
        }
        totals.prepTurnWaitNs = take(prepTurnWaitNs);
        totals.prepJobs = take(prepJobs);
        totals.fifoFullNs = take(fifoFullNs);
        totals.fifoDrainNs = take(fifoDrainNs);
        totals.fifoEmptyNs = take(fifoEmptyNs);
        totals.fifoDepthSum = take(fifoDepthSum);
        totals.fifoEnqueued = take(fifoEnqueued);
        totals.spineBusyNs = take(spineBusyNs);
        totals.spineCommits = take(spineCommits);
        totals.drawCommits = take(drawCommits);
        totals.drawCommitNs = take(drawCommitNs);
        totals.recorderChunks = take(recorderChunks);
        totals.recorderDraws = take(recorderDraws);
        totals.keepSealedChunks = take(keepSealedChunks);
        totals.chunkEpochJoins = take(chunkEpochJoins);
        totals.staleRecordInputs = take(staleRecordInputs);
        totals.staleRecordResources = take(staleRecordResources);
        totals.vkSubmits = take(vkSubmits);
        totals.vkSubmitNs = take(vkSubmitNs);
        totals.executeNs = take(executeNs);
        totals.drawStepDraws = take(drawStepDraws);
        for (std::size_t i = 0; i < PacketSlots; ++i) {
            totals.slotPackets[i] = take(slotPackets[i]);
            totals.slotNs[i] = take(slotNs[i]);
        }
        for (std::size_t i = 0; i < PacketPhases; ++i) {
            totals.drawPhaseNs[i] = take(drawPhaseNs[i]);
            totals.otherPhaseNs[i] = take(otherPhaseNs[i]);
        }
        for (std::size_t i = 0; i < phasePackets.size(); ++i) totals.phasePackets[i] = take(phasePackets[i]);
        for (std::size_t i = 0; i < DrawSteps; ++i) totals.drawStepNs[i] = take(drawStepNs[i]);
        takeAll(fifoFullKindNs, totals.fifoFullKindNs);
        takeAll(fifoDrainReasonNs, totals.fifoDrainReasonNs);
        takeAll(fifoDrainReasonCount, totals.fifoDrainReasonCount);
        takeAll(fifoDrainOpcodeNs, totals.fifoDrainOpcodeNs);
        takeAll(fifoDrainOpcodeCount, totals.fifoDrainOpcodeCount);
        totals.mpassInline = take(mpassInline);
        totals.mpassPipelined = take(mpassPipelined);
        totals.spineSamples = take(spineSamples);
        totals.spineSampleNs = take(spineSampleNs);
        takeAll(spinePhaseNs, totals.spinePhaseNs);
        totals.submitValidateNs = take(submitValidateNs);
        takeAll(proofBreaks, totals.proofBreaks);
        totals.targetMemoLookups = take(targetMemoLookups);
        totals.targetMemoHits = take(targetMemoHits);
        totals.drawInputMemoLookups = take(drawInputMemoLookups);
        totals.drawInputMemoHits = take(drawInputMemoHits);
        totals.drawStages = take(drawStages);
        totals.stagedDraws = take(stagedDraws);
        totals.halfChecks = take(halfChecks);
        totals.halfStale = take(halfStale);
        takeAll(vkSubmitsBy, totals.vkSubmitsBy);
        return totals;
    }

    Counter submits{0}, submitNs{0}, flipSubmits{0}, flipSubmitNs{0}, eopReserves{0}, eopReserveNs{0};
    Counter flipRoomNs{0}, flipHoldNs{0}, registrationNs{0}, renderWaitNs{0};
    Counter dwords{0}, packets{0}, draws{0};
    Counter queueLatencyNs{0}, queueLatencies{0};
    Counter packetBusyNs{0}, packetIdleNs{0}, packetPackets{0}, packetSubmissions{0}, packetPoolWaitNs{0};
    std::array<Counter, LockRoles> lockWaitNs{}, lockHoldNs{};
    std::array<Counter, LaneThreads> prepBusyNs{};
    Counter prepTurnWaitNs{0}, prepJobs{0};
    Counter fifoFullNs{0}, fifoDrainNs{0}, fifoEmptyNs{0}, fifoDepthSum{0}, fifoEnqueued{0};
    Counter spineBusyNs{0}, spineCommits{0}, drawCommits{0}, drawCommitNs{0};
    std::array<Counter, LaneThreads> recorderBusyNs{};
    Counter recorderChunks{0}, recorderDraws{0}, keepSealedChunks{0}, chunkEpochJoins{0}, staleRecordInputs{0}, staleRecordResources{0};
    Counter vkSubmits{0}, vkSubmitNs{0};
    Counter executeNs{0}, drawStepDraws{0};
    std::array<Counter, PacketSlots> slotPackets{}, slotNs{};
    std::array<Counter, PacketPhases> drawPhaseNs{}, otherPhaseNs{};
    std::array<Counter, 2> phasePackets{};
    std::array<Counter, DrawSteps> drawStepNs{};
    std::array<Counter, EnqueueKinds> fifoFullKindNs{};
    std::array<Counter, DrainReasons> fifoDrainReasonNs{}, fifoDrainReasonCount{};
    std::array<Counter, DrainOpcodes> fifoDrainOpcodeNs{}, fifoDrainOpcodeCount{};
    Counter mpassInline{0}, mpassPipelined{0};
    Counter spineSamples{0}, spineSampleNs{0};
    std::array<Counter, SpinePhases> spinePhaseNs{};
    Counter submitValidateNs{0};
    std::array<Counter, ProofBreakers> proofBreaks{};
    Counter targetMemoLookups{0}, targetMemoHits{0}, drawInputMemoLookups{0}, drawInputMemoHits{0};
    Counter drawStages{0}, stagedDraws{0}, halfChecks{0}, halfStale{0};
    std::array<Counter, ThreadRoles * SubmitTriggers> vkSubmitsBy{};

private:
    template <std::size_t Size>
    static void takeAll(std::array<Counter, Size>& counters, std::array<std::uint64_t, Size>& totals) {
        for (std::size_t i = 0; i < Size; ++i) totals[i] = counters[i].exchange(0, std::memory_order_relaxed);
    }
};

// Every split adds to its total in the same call, so the parts sum to the total.
inline void NoteFifoFull(Counters& counters, std::size_t kind, std::int64_t ns) {
    Counters::AddNs(counters.fifoFullNs, ns);
    // An unknown kind is the commit kinds' 'other', never the pool.
    Counters::AddNs(counters.fifoFullKindNs[kind < EnqueueKinds ? kind : PoolEnqueueKind - 1], ns);
}

inline void NoteFifoDrain(Counters& counters, std::size_t reason, std::uint32_t opcode, std::int64_t ns) {
    const auto reasonSlot = std::min(reason, DrainReasons - 1);
    const auto opcodeSlot = std::min<std::size_t>(opcode, NoDrainOpcode);
    Counters::AddNs(counters.fifoDrainNs, ns);
    Counters::AddNs(counters.fifoDrainReasonNs[reasonSlot], ns);
    Counters::Add(counters.fifoDrainReasonCount[reasonSlot], 1);
    Counters::AddNs(counters.fifoDrainOpcodeNs[opcodeSlot], ns);
    Counters::Add(counters.fifoDrainOpcodeCount[opcodeSlot], 1);
}

// The committer's sampled draw commit (1 in SpineSampleEvery): the Graphics DrawTimer adds each
// phase of the commit's draw here (active only on the committer, for that one commit).
struct SpineSample {
    bool active = false;
    std::array<std::int64_t, SpineDrawPhases> phaseNs{};
};

inline thread_local SpineSample threadSpineSample;

inline SpineSample& ThreadSpineSample() {
    return threadSpineSample;
}

// The commit time the phases did not cover is 'other', so a sample's parts sum to its commit.
inline void NoteSpineSample(Counters& counters, const SpineSample& sample, std::uint64_t commitNs) {
    std::uint64_t phased = 0;
    for (std::size_t i = 0; i < SpineDrawPhases; ++i) {
        if (sample.phaseNs[i] <= 0) continue;
        Counters::AddNs(counters.spinePhaseNs[i], sample.phaseNs[i]);
        phased += static_cast<std::uint64_t>(sample.phaseNs[i]);
    }
    Counters::Add(counters.spinePhaseNs[SpineOtherPhase], commitNs > phased ? commitNs - phased : 0);
    Counters::Add(counters.spineSamples, 1);
    Counters::Add(counters.spineSampleNs, commitNs);
}

// The thread's role (BindThread) and the reason its current submit is made (SubmitCause), for
// vk_submits by thread and trigger.
inline thread_local std::uint8_t threadRole = RoleOther;
inline thread_local std::uint8_t threadTrigger = TriggerOther;

inline std::size_t ThreadRole() {
    return threadRole;
}

inline std::size_t ThreadTrigger() {
    return threadTrigger;
}

// A fallback cause names the submits only where no outer cause did (a flip's label flush stays a
// flip submit).
class SubmitCause {
public:
    explicit SubmitCause(std::size_t trigger, bool fallback = false) : active(Enabled()), previous(active ? threadTrigger : std::uint8_t{0}) {
        if (active && !(fallback && previous != TriggerOther)) threadTrigger = static_cast<std::uint8_t>(std::min(trigger, SubmitTriggers - 1));
    }
    SubmitCause(const SubmitCause&) = delete;
    SubmitCause& operator=(const SubmitCause&) = delete;
    ~SubmitCause() {
        if (active) threadTrigger = previous;
    }

private:
    bool active;
    std::uint8_t previous;
};

inline void NoteVkSubmit(Counters& counters, std::size_t role, std::size_t trigger) {
    Counters::Add(counters.vkSubmitsBy[std::min(role, ThreadRoles - 1) * SubmitTriggers + std::min(trigger, SubmitTriggers - 1)], 1);
}

// The CPU-time slot of a thread role (CpuSlots for a role without one).
inline std::size_t CpuSlot(std::size_t role, std::size_t index) {
    const auto lane = std::min(index, LaneThreads - 1);
    switch (role) {
    case RolePacket: return CpuPacket;
    case RoleSpine: return CpuSpine;
    case RolePresenter: return CpuPresenter;
    case RoleRelease: return CpuRelease;
    case RoleRecorder: return CpuRecorders + lane;
    case RolePool: return CpuPool + lane;
    default: return CpuSlots;
    }
}

// Thread cycle counts to time at the rate the window measured (TSC ticks over steady-clock ns).
inline std::uint64_t CyclesToNs(std::uint64_t cycles, std::uint64_t ticks, std::uint64_t ns) {
    if (ticks == 0) return 0;
    return static_cast<std::uint64_t>(static_cast<long double>(cycles) * static_cast<long double>(ns) / static_cast<long double>(ticks));
}

// Names the calling thread's role and registers it for thread_cpu_ms/f (APS5_PROFILE_FRAME only).
void BindThread(std::size_t role, std::size_t index);
// Each registered thread's CPU time since the last call.
std::array<std::uint64_t, CpuSlots> TakeThreadCpuNs();

// The frame chain on queue 0: a frame opens at the packet thread's first dequeue after a flip;
// its first vkQueueSubmit is timed from there, and every present from the newest submit.
class FrameChain {
public:
    void Dequeued(std::int64_t nowNs) {
        if (open.load(std::memory_order_relaxed)) return;
        startNs.store(nowNs, std::memory_order_relaxed);
        open.store(true, std::memory_order_relaxed);
        firstPending.store(true, std::memory_order_release);
    }

    void Flipped() { open.store(false, std::memory_order_relaxed); }

    void Submitted(std::int64_t nowNs) {
        lastSubmitNs.store(nowNs, std::memory_order_relaxed);
        if (!firstPending.load(std::memory_order_relaxed) || !firstPending.exchange(false, std::memory_order_acq_rel)) return;
        Counters::AddNs(firstSubmitNs, nowNs - startNs.load(std::memory_order_relaxed));
        Counters::Add(firstSubmits, 1);
    }

    void Presented(std::int64_t nowNs) {
        const auto last = lastSubmitNs.load(std::memory_order_relaxed);
        if (last == 0) return;
        Counters::AddNs(tailNs, nowNs - last);
        Counters::Add(tails, 1);
    }

    ChainTotals Take() {
        const auto take = [](Counters::Counter& counter) { return counter.exchange(0, std::memory_order_relaxed); };
        return {take(firstSubmitNs), take(firstSubmits), take(tailNs), take(tails)};
    }

private:
    std::atomic<bool> open{false}, firstPending{false};
    std::atomic<std::int64_t> startNs{0}, lastSubmitNs{0};
    Counters::Counter firstSubmitNs{0}, firstSubmits{0}, tailNs{0}, tails{0};
};

inline FrameChain& GlobalChain() {
    static FrameChain* chain = new FrameChain();
    return *chain;
}

using StepClock = std::int64_t (*)();

class PacketStep {
public:
    PacketStep(Counters& counters, std::size_t slot, bool draw, StepClock clock = NowNs)
        : counters(counters), clock(clock), slot(std::min(slot, PacketSlots - 1)), draw(draw), startNs(clock != nullptr ? clock() : 0), lastNs(startNs) {}
    PacketStep(const PacketStep&) = delete;
    PacketStep& operator=(const PacketStep&) = delete;

    ~PacketStep() {
        if (clock == nullptr) return;
        const auto now = clock();
        phaseNs[phase] += now - lastNs;
        Counters::Add(counters.slotPackets[slot], 1);
        Counters::AddNs(counters.slotNs[slot], now - startNs);
        Counters::Add(counters.phasePackets[draw ? 1 : 0], 1);
        auto& phases = draw ? counters.drawPhaseNs : counters.otherPhaseNs;
        for (std::size_t i = 0; i < PacketPhases; ++i) {
            if (phaseNs[i] != 0) Counters::AddNs(phases[i], phaseNs[i]);
        }
    }

    void Mark(std::size_t next) {
        if (clock == nullptr) return;
        const auto now = clock();
        phaseNs[phase] += now - lastNs;
        lastNs = now;
        phase = std::min(next, PacketPhases - 1);
    }

private:
    Counters& counters;
    StepClock clock;
    std::size_t slot;
    bool draw;
    std::int64_t startNs;
    std::int64_t lastNs;
    std::size_t phase = PhaseGate;
    std::array<std::int64_t, PacketPhases> phaseNs{};
};

class ScopedNs {
public:
    explicit ScopedNs(Counters::Counter* counter) : counter(counter), startNs(counter != nullptr ? NowNs() : 0) {}
    ScopedNs(const ScopedNs&) = delete;
    ScopedNs& operator=(const ScopedNs&) = delete;
    ~ScopedNs() {
        if (counter != nullptr) Counters::AddNs(*counter, NowNs() - startNs);
    }

private:
    Counters::Counter* counter;
    std::int64_t startNs;
};

inline void NoteDrawSteps(Counters& counters, const std::array<double, DrawSteps>& stepMs) {
    for (std::size_t i = 0; i < DrawSteps; ++i) {
        if (stepMs[i] > 0) Counters::Add(counters.drawStepNs[i], static_cast<std::uint64_t>(stepMs[i] * 1e6));
    }
    Counters::Add(counters.drawStepDraws, 1);
}

class Frames {
public:
    void Submit(std::int64_t entryNs, std::int64_t exitNs, bool flip) {
        std::lock_guard lock(mutex);
        ClosedFrame closed;
        if (!guest.Submit(entryNs, exitNs, flip, closed)) return;
        if (closed.gapMs >= 0) gaps.Add(closed.gapMs);
        spans.Add(closed.spanMs);
        inside.Add(closed.insideMs);
        flipQueue[flipTail % FlipQueueCapacity] = exitNs;
        ++flipTail;
        if (flipTail - flipHead > FlipQueueCapacity) flipHead = flipTail - FlipQueueCapacity;
    }

    bool Present(std::int64_t nowNs) {
        std::lock_guard lock(mutex);
        if (windowStartNs == 0) windowStartNs = nowNs;
        if (lastPresentNs != 0) intervals.Add(static_cast<double>(nowNs - lastPresentNs) / 1e6);
        lastPresentNs = nowNs;
        ++presents;
        if (flipHead != flipTail) {
            flipToPresent.Add(static_cast<double>(std::max<std::int64_t>(0, nowNs - flipQueue[flipHead % FlipQueueCapacity])) / 1e6);
            ++flipHead;
        }
        return nowNs - windowStartNs >= ReportWindowNs;
    }

    Window Take(std::int64_t nowNs) {
        std::lock_guard lock(mutex);
        Window window;
        window.seconds = windowStartNs != 0 ? static_cast<double>(nowNs - windowStartNs) / 1e9 : 0.0;
        window.presents = presents;
        window.gapMs = gaps.Take();
        window.spanMs = spans.Take();
        window.insideMs = inside.Take();
        window.intervalMs = intervals.Take();
        window.flipToPresentMs = flipToPresent.Take();
        window.frames = window.spanMs.count;
        presents = 0;
        windowStartNs = nowNs;
        return window;
    }

private:
    std::mutex mutex;
    GuestFrames guest;
    SampleWindow gaps, spans, inside, intervals, flipToPresent;
    std::array<std::int64_t, FlipQueueCapacity> flipQueue{};
    std::uint64_t flipHead = 0, flipTail = 0;
    std::int64_t windowStartNs = 0, lastPresentNs = 0;
    std::uint64_t presents = 0;
};

Derived Derive(const Totals& totals, const Window& window);
std::string FormatGuest(const Totals& totals, const Window& window);
std::string FormatLanes(const Totals& totals, const Window& window);
using SlotNamer = std::string (*)(std::size_t slot);
std::string FormatPackets(const Totals& totals, const Window& window, SlotNamer name);
std::string FormatFifo(const Totals& totals, const Window& window, SlotNamer name);
std::string FormatSpine(const Totals& totals, const Window& window, std::uint32_t sampleEvery);
std::string FormatThreadCpu(const Window& window);
std::string FormatChain(const Totals& totals, const Window& window);
std::string FormatMemo(const Totals& totals, const Window& window);
// The [run-id] line: the libs' git describe, their build time and the set APS5_ variables.
std::string FormatRunId(std::string_view describe, std::string_view built, std::span<const std::string> environment);
// Prints the [run-id] line to stderr (RunId.cpp, which embeds the build's revision).
void PrintRunId();

Counters& Global();

inline Frames& GlobalFrames() {
    static Frames* frames = new Frames();
    return *frames;
}

inline void NoteExecuted(std::span<const std::uint32_t> commands, bool queue0) {
    std::uint64_t packets = 0, draws = 0;
    for (std::size_t cursor = 0; cursor < commands.size();) {
        const auto header = commands[cursor];
        const bool filler = (header >> 30u) == 2u || header == 0xffff1000u;
        cursor += filler ? 1u : static_cast<std::size_t>((header >> 16u) & 0x3fffu) + 2u;
        if (filler) continue;
        ++packets;
        const auto opcode = (header >> 8u) & 0xffu;
        if ((header >> 30u) == 3u && (opcode == 0x27 || opcode == 0x2d || opcode == 0x35 || opcode == 0x24 || opcode == 0x25 || opcode == 0x2c || opcode == 0x38)) ++draws;
    }
    auto& counters = Global();
    Counters::Add(counters.dwords, commands.size());
    Counters::Add(counters.packets, packets);
    Counters::Add(counters.draws, draws);
    if (queue0) Counters::Add(counters.packetPackets, packets);
}

}

#endif
