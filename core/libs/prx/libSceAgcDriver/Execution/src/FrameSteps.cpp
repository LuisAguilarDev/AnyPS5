#include "prx/libSceAgcDriver/Execution/include/FrameSteps.hpp"
#include <cstdio>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <intrin.h>
#else
#include <pthread.h>
#include <time.h>
#endif

namespace AgcDriver::FrameSteps {

namespace {

constexpr std::size_t PacketRole = 1;
constexpr std::size_t SpineRole = 3;
constexpr double BytesPerKb = 1024.0;
constexpr double Percent = 100.0;
constexpr std::array<const char*, EnqueueKinds> EnqueueKindNames{"draw", "label", "dma", "dispatch", "fill", "pass", "completion", "open_work", "other", "pool"};
constexpr std::array<const char*, DrainReasons> DrainReasonNames{"packet", "flush", "labels", "capture", "indirect", "submission"};
constexpr std::array<const char*, SpinePhases> SpinePhaseNames{"validate", "vertex", "setup", "readTarget", "prepare", "lookup", "resources", "pipeline", "record", "keep", "sync", "writeBack", "describe", "other"};
constexpr std::array<const char*, ProofBreakers> ProofBreakerNames{"other", "store", "written", "write_changed", "gpu_hold", "non_draw_commit", "completion_label", "copied_writer", "resource_write_back"};
constexpr std::array<const char*, ThreadRoles> RoleNames{"other", "packet", "spine", "pool", "recorder", "presenter", "release"};
constexpr std::array<const char*, SubmitTriggers> TriggerNames{"other", "cap", "deadline", "label", "completion", "flip", "present", "end", "idle", "wait", "suspend"};
constexpr std::string_view EnvironmentPrefix = "APS5_";

double Ms(std::uint64_t ns) {
    return static_cast<double>(ns) / 1e6;
}

double Us(std::uint64_t ns) {
    return static_cast<double>(ns) / 1e3;
}

std::string ThreadList(const std::array<std::uint64_t, LaneThreads>& busyNs, std::uint64_t frames) {
    std::size_t used = 0;
    for (std::size_t i = 0; i < LaneThreads; ++i) {
        if (busyNs[i] != 0) used = i + 1;
    }
    if (used == 0) return "0";
    std::string list;
    char text[32];
    for (std::size_t i = 0; i < used; ++i) {
        std::snprintf(text, sizeof(text), "%s%.2f", i == 0 ? "" : "/", PerFrame(Ms(busyNs[i]), frames));
        list += text;
    }
    return list;
}

void SumAndMax(const std::array<std::uint64_t, LaneThreads>& busyNs, std::uint64_t frames, double& sum, double& max) {
    sum = 0;
    max = 0;
    for (const auto ns : busyNs) {
        const auto perFrame = PerFrame(Ms(ns), frames);
        sum += perFrame;
        max = std::max(max, perFrame);
    }
}

std::string Finish(const char* text, int length) {
    if (length < 0) return {};
    return std::string(text) + "\n";
}

std::string PhaseList(const std::array<std::uint64_t, PacketPhases>& phaseNs, std::uint64_t packets) {
    std::string list;
    char text[32];
    for (std::size_t i = 0; i < PacketPhases; ++i) {
        std::snprintf(text, sizeof(text), "%s%.2f", i == 0 ? "" : "/", PerItem(Us(phaseNs[i]), packets));
        list += text;
    }
    return list;
}

std::string DrawStepList(const Totals& totals) {
    if (totals.drawStepDraws == 0) return "-";
    std::string list;
    char text[64];
    for (std::size_t i = 0; i < DrawSteps; ++i) {
        if (totals.drawStepNs[i] == 0) continue;
        std::snprintf(text, sizeof(text), "%s%s=%.2f", list.empty() ? "" : "|", DriverDetail::DrawDriverPhaseNames[i], PerItem(Us(totals.drawStepNs[i]), totals.drawStepDraws));
        list += text;
    }
    return list.empty() ? "-" : list;
}

std::string SlotLabel(std::size_t slot, SlotNamer name) {
    if (slot == FlipSlot) return "FLIP";
    if (slot == RenderWaitSlot) return "RENDER_WAIT";
    if (name != nullptr) {
        if (auto named = name(slot); !named.empty()) return named;
    }
    char text[16];
    std::snprintf(text, sizeof(text), "0x%02zx", slot);
    return text;
}

std::string TopSlots(const Totals& totals, std::uint64_t frames, SlotNamer name) {
    std::array<std::size_t, PacketSlots> order{};
    std::size_t used = 0;
    for (std::size_t slot = 0; slot < PacketSlots; ++slot) {
        if (totals.slotPackets[slot] != 0) order[used++] = slot;
    }
    if (used == 0) return "-";
    const auto shown = std::min(used, TopPacketSlots);
    std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(shown), order.begin() + static_cast<std::ptrdiff_t>(used), [&](std::size_t left, std::size_t right) { return totals.slotNs[left] > totals.slotNs[right]; });
    std::string list;
    char text[96];
    for (std::size_t i = 0; i < shown; ++i) {
        const auto slot = order[i];
        std::snprintf(text, sizeof(text), ":%.0f/f:%.2fus:%.2fms/f", PerFrame(static_cast<double>(totals.slotPackets[slot]), frames), PerItem(Us(totals.slotNs[slot]), totals.slotPackets[slot]), PerFrame(Ms(totals.slotNs[slot]), frames));
        if (i != 0) list += ",";
        list += SlotLabel(slot, name) + text;
    }
    return list;
}

template <std::size_t Size>
std::string NameList(const std::array<const char*, Size>& names) {
    std::string list;
    for (std::size_t i = 0; i < Size; ++i) list += std::string(i == 0 ? "" : "/") + names[i];
    return list;
}

// The sampled parts of Recompile(request, capture), in us per call: each part's sampled total over
// the sampled calls, so the parts add up to recompile_us/call.
std::string FormatRecompileSplit(const ShaderRecompiler::MaterializeSplit::Counts& split, std::uint64_t frames) {
    using Split = ShaderRecompiler::MaterializeSplit;
    std::uint64_t total = 0;
    for (const auto ns : split.ns) total += ns;
    char text[160];
    std::snprintf(text, sizeof(text), " recompile_calls/f=%.1f recompile_sampled=%llu recompile_us/call=%.2f recompile_split_us/call(", PerFrame(static_cast<double>(split.calls), frames), static_cast<unsigned long long>(split.sampled), PerItem(Us(total), split.sampled));
    auto line = std::string(text) + NameList(Split::Names) + ")=";
    for (std::size_t part = 0; part < Split::Parts; ++part) {
        std::snprintf(text, sizeof(text), "%s%.2f", part == 0 ? "" : "/", PerItem(Us(split.ns[part]), split.sampled));
        line += text;
    }
    return line;
}

// Each value per frame, '/'-separated: milliseconds of nanosecond totals, or counts.
template <std::size_t Size>
std::string PerFrameList(const std::array<std::uint64_t, Size>& values, std::uint64_t frames, bool nanoseconds) {
    std::string list;
    char text[32];
    for (std::size_t i = 0; i < Size; ++i) {
        const auto value = nanoseconds ? PerFrame(Ms(values[i]), frames) : PerFrame(static_cast<double>(values[i]), frames);
        std::snprintf(text, sizeof(text), nanoseconds ? "%s%.2f" : "%s%.1f", i == 0 ? "" : "/", value);
        list += text;
    }
    return list;
}

std::string DrainOpcodeLabel(std::size_t opcode, SlotNamer name) {
    if (opcode == NoDrainOpcode) return "none";
    if (name != nullptr) {
        if (auto named = name(opcode); !named.empty()) return named;
    }
    char text[16];
    std::snprintf(text, sizeof(text), "0x%02zx", opcode);
    return text;
}

std::string TopDrains(const Totals& totals, std::uint64_t frames, SlotNamer name) {
    std::array<std::size_t, DrainOpcodes> order{};
    std::size_t used = 0;
    for (std::size_t opcode = 0; opcode < DrainOpcodes; ++opcode) {
        if (totals.fifoDrainOpcodeCount[opcode] != 0) order[used++] = opcode;
    }
    if (used == 0) return "-";
    const auto shown = std::min(used, TopDrainOpcodes);
    std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(shown), order.begin() + static_cast<std::ptrdiff_t>(used), [&](std::size_t left, std::size_t right) { return totals.fifoDrainOpcodeNs[left] > totals.fifoDrainOpcodeNs[right]; });
    std::string list;
    char text[64];
    for (std::size_t i = 0; i < shown; ++i) {
        const auto opcode = order[i];
        std::snprintf(text, sizeof(text), ":%.1f/f:%.2fms/f", PerFrame(static_cast<double>(totals.fifoDrainOpcodeCount[opcode]), frames), PerFrame(Ms(totals.fifoDrainOpcodeNs[opcode]), frames));
        if (i != 0) list += ",";
        list += DrainOpcodeLabel(opcode, name) + text;
    }
    return list;
}

std::array<std::uint64_t, LaneThreads> CpuLanes(const Window& window, std::size_t first) {
    std::array<std::uint64_t, LaneThreads> lanes{};
    for (std::size_t i = 0; i < LaneThreads; ++i) lanes[i] = window.threadCpuNs[first + i];
    return lanes;
}

double Rate(std::uint64_t hits, std::uint64_t lookups) {
    return lookups != 0 ? Percent * static_cast<double>(hits) / static_cast<double>(lookups) : 0.0;
}

// The registered threads' CPU time: QueryThreadCycleTime on Windows (cycles, converted at the TSC
// rate measured over the same interval), the thread's CPU-time clock elsewhere.
struct CpuThreads {
    std::mutex mutex;
#ifdef _WIN32
    std::array<HANDLE, CpuSlots> handles{};
    std::uint64_t lastTicks = 0;
    std::int64_t lastNs = 0;
#else
    std::array<clockid_t, CpuSlots> clocks{};
    std::array<bool, CpuSlots> bound{};
#endif
    std::array<std::uint64_t, CpuSlots> last{};
};

CpuThreads& Threads() {
    static CpuThreads* threads = new CpuThreads();
    return *threads;
}

#ifdef _WIN32
std::uint64_t ThreadCycles(HANDLE handle) {
    ULONG64 cycles = 0;
    return QueryThreadCycleTime(handle, &cycles) ? cycles : 0;
}
#else
std::uint64_t ThreadClockNs(clockid_t clock) {
    timespec now{};
    if (clock_gettime(clock, &now) != 0) return 0;
    return static_cast<std::uint64_t>(now.tv_sec) * 1'000'000'000ull + static_cast<std::uint64_t>(now.tv_nsec);
}
#endif

}

void BindThread(std::size_t role, std::size_t index) {
    if (!Enabled()) return;
    threadRole = static_cast<std::uint8_t>(std::min(role, ThreadRoles - 1));
    const auto slot = CpuSlot(role, index);
    if (slot >= CpuSlots) return;
    auto& threads = Threads();
    std::lock_guard lock(threads.mutex);
#ifdef _WIN32
    HANDLE handle = nullptr;
    if (!DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &handle, THREAD_QUERY_LIMITED_INFORMATION, FALSE, 0)) return;
    if (threads.handles[slot] != nullptr) CloseHandle(threads.handles[slot]);
    threads.handles[slot] = handle;
    threads.last[slot] = ThreadCycles(handle);
    if (threads.lastTicks == 0) {
        threads.lastTicks = __rdtsc();
        threads.lastNs = NowNs();
    }
#else
    clockid_t clock{};
    if (pthread_getcpuclockid(pthread_self(), &clock) != 0) return;
    threads.clocks[slot] = clock;
    threads.bound[slot] = true;
    threads.last[slot] = ThreadClockNs(clock);
#endif
}

std::array<std::uint64_t, CpuSlots> TakeThreadCpuNs() {
    std::array<std::uint64_t, CpuSlots> spent{};
    auto& threads = Threads();
    std::lock_guard lock(threads.mutex);
#ifdef _WIN32
    const auto ticks = __rdtsc();
    const auto nowNs = NowNs();
    const auto elapsedTicks = threads.lastTicks != 0 && ticks > threads.lastTicks ? ticks - threads.lastTicks : 0;
    const auto elapsedNs = static_cast<std::uint64_t>(std::max<std::int64_t>(0, nowNs - threads.lastNs));
    threads.lastTicks = ticks;
    threads.lastNs = nowNs;
    for (std::size_t slot = 0; slot < CpuSlots; ++slot) {
        if (threads.handles[slot] == nullptr) continue;
        const auto cycles = ThreadCycles(threads.handles[slot]);
        spent[slot] = CyclesToNs(cycles > threads.last[slot] ? cycles - threads.last[slot] : 0, elapsedTicks, elapsedNs);
        threads.last[slot] = cycles;
    }
#else
    for (std::size_t slot = 0; slot < CpuSlots; ++slot) {
        if (!threads.bound[slot]) continue;
        const auto ns = ThreadClockNs(threads.clocks[slot]);
        spent[slot] = ns > threads.last[slot] ? ns - threads.last[slot] : 0;
        threads.last[slot] = ns;
    }
#endif
    return spent;
}

Derived Derive(const Totals& totals, const Window& window) {
    const auto frames = window.frames;
    const auto perFrameMs = [&](std::uint64_t ns) { return PerFrame(Ms(ns), frames); };
    Derived derived;
    derived.frameMs = window.gapMs.mean + window.spanMs.mean;
    derived.packetOwnMs = std::max(0.0, perFrameMs(totals.packetBusyNs) - perFrameMs(totals.fifoFullNs) - perFrameMs(totals.fifoDrainNs) - perFrameMs(totals.packetPoolWaitNs));
    if (frames == 0) return derived;
    derived.couplingMs = derived.frameMs - perFrameMs(totals.packetIdleNs) - std::max(derived.packetOwnMs, perFrameMs(totals.spineBusyNs));
    derived.tailMs = window.flipToPresentMs.mean - PerItem(Ms(totals.queueLatencyNs), totals.queueLatencies) - perFrameMs(totals.packetBusyNs);
    return derived;
}

std::string FormatFifo(const Totals& totals, const Window& window, SlotNamer name) {
    const auto frames = window.frames;
    char text[192];
    const auto length = std::snprintf(text, sizeof(text), "[frame-steps-fifo] window_s=%.1f frames=%llu fifo_full_ms/f=%.2f ", window.seconds, static_cast<unsigned long long>(frames), PerFrame(Ms(totals.fifoFullNs), frames));
    if (length < 0) return {};
    std::string line(text);
    line += "fifo_full_ms/f(" + NameList(EnqueueKindNames) + ")=" + PerFrameList(totals.fifoFullKindNs, frames, true);
    std::snprintf(text, sizeof(text), " fifo_drain_ms/f=%.2f", PerFrame(Ms(totals.fifoDrainNs), frames));
    line += text;
    line += " fifo_drains/f(" + NameList(DrainReasonNames) + ")=" + PerFrameList(totals.fifoDrainReasonCount, frames, false);
    line += " fifo_drain_ms/f(" + NameList(DrainReasonNames) + ")=" + PerFrameList(totals.fifoDrainReasonNs, frames, true);
    line += " drain_top=" + TopDrains(totals, frames, name);
    std::snprintf(text, sizeof(text), " mpass_inline/f=%.1f mpass_pipelined/f=%.1f\n", PerFrame(static_cast<double>(totals.mpassInline), frames), PerFrame(static_cast<double>(totals.mpassPipelined), frames));
    return line + text;
}

std::string FormatSpine(const Totals& totals, const Window& window, std::uint32_t sampleEvery) {
    char text[256];
    const auto length = std::snprintf(text, sizeof(text), "[frame-steps-spine] window_s=%.1f frames=%llu sampled_commits=%llu sample_every=%u sampled_commit_us=%.2f draw_commit_us=%.2f spine_phases_us/commit=",
        window.seconds, static_cast<unsigned long long>(window.frames), static_cast<unsigned long long>(totals.spineSamples), sampleEvery, PerItem(Us(totals.spineSampleNs), totals.spineSamples), PerItem(Us(totals.drawCommitNs), totals.drawCommits));
    if (length < 0) return {};
    std::string line(text);
    std::string list;
    for (std::size_t i = 0; totals.spineSamples != 0 && i < SpinePhases; ++i) {
        if (totals.spinePhaseNs[i] == 0) continue;
        std::snprintf(text, sizeof(text), "%s%s=%.2f", list.empty() ? "" : "|", SpinePhaseNames[i], PerItem(Us(totals.spinePhaseNs[i]), totals.spineSamples));
        list += text;
    }
    return line + (list.empty() ? "-" : list) + "\n";
}

std::string FormatThreadCpu(const Window& window) {
    const auto frames = window.frames;
    const auto perFrameMs = [&](std::size_t slot) { return PerFrame(Ms(window.threadCpuNs[slot]), frames); };
    const auto recorders = ThreadList(CpuLanes(window, CpuRecorders), frames);
    const auto pool = ThreadList(CpuLanes(window, CpuPool), frames);
    char text[512];
    const auto length = std::snprintf(text, sizeof(text), "[frame-steps-cpu] window_s=%.1f frames=%llu thread_cpu_ms/f packet=%.2f spine=%.2f presenter=%.2f release=%.2f rec=%s pool=%s",
        window.seconds, static_cast<unsigned long long>(frames), perFrameMs(CpuPacket), perFrameMs(CpuSpine), perFrameMs(CpuPresenter), perFrameMs(CpuRelease), recorders.c_str(), pool.c_str());
    return Finish(text, length);
}

std::string FormatChain(const Totals& totals, const Window& window) {
    const auto frames = window.frames;
    char text[384];
    const auto length = std::snprintf(text, sizeof(text), "[frame-steps-chain] window_s=%.1f frames=%llu gpu_gap_ms/f=%.2f first_submit_ms=%.2f last_submit_to_present_ms=%.2f submit_validate_us=%.1f vk_submits/f=%.1f vk_submits/f(thread:trigger)=",
        window.seconds, static_cast<unsigned long long>(frames), PerFrame(window.gpuGapMs, frames), PerItem(Ms(window.chain.firstSubmitNs), window.chain.firstSubmits), PerItem(Ms(window.chain.tailNs), window.chain.tails),
        PerItem(Us(totals.submitValidateNs), totals.submits), PerFrame(static_cast<double>(totals.vkSubmits), frames));
    if (length < 0) return {};
    std::string line(text);
    std::string list;
    for (std::size_t role = 0; role < ThreadRoles; ++role) {
        for (std::size_t trigger = 0; trigger < SubmitTriggers; ++trigger) {
            const auto count = totals.vkSubmitsBy[role * SubmitTriggers + trigger];
            if (count == 0) continue;
            std::snprintf(text, sizeof(text), "%s%s:%s=%.1f", list.empty() ? "" : ",", RoleNames[role], TriggerNames[trigger], PerFrame(static_cast<double>(count), frames));
            list += text;
        }
    }
    return line + (list.empty() ? "-" : list) + "\n";
}

std::string FormatMemo(const Totals& totals, const Window& window) {
    const auto frames = window.frames;
    const auto perFrame = [&](std::uint64_t count) { return PerFrame(static_cast<double>(count), frames); };
    std::uint64_t breaks = 0;
    for (const auto count : totals.proofBreaks) breaks += count;
    const auto& shaders = window.shaders;
    const auto resultLookups = shaders.resultMemoHits + shaders.resultMemoMisses;
    const auto epochProofs = totals.epochProofReuses + totals.epochProofFulls;
    const auto refreshProofs = totals.refreshProofReuses + totals.refreshProofFulls;
    char text[1024];
    const auto length = std::snprintf(text, sizeof(text), "[frame-steps-memo] window_s=%.1f frames=%llu proof_breaks/f=%.1f proof_breaks/f(%s)=%s ",
        window.seconds, static_cast<unsigned long long>(frames), perFrame(breaks), NameList(ProofBreakerNames).c_str(), PerFrameList(totals.proofBreaks, frames, false).c_str());
    if (length < 0) return {};
    std::string line(text);
    const auto tail = std::snprintf(text, sizeof(text),
        "target_memo_hit%%=%.1f target_memo_lookups/f=%.1f draw_input_memo_hit%%=%.1f draw_input_memo_lookups/f=%.1f result_memo_hit%%=%.1f result_memo_lookups/f=%.1f result_memo_evictions/f=%.1f "
        "stages/draw=%.2f program_keys/10s=%llu program_key_materializes/f=%.1f program_key_table_hit%%=%.1f program_key_overflow=%llu half_checks/f=%.1f half_stale/f=%.1f half_copy_us/draw=%.3f hold_waits/f=%.1f hold_wait_ms/f=%.2f "
        "epoch_proof_reuse%%=%.1f epoch_proofs/f=%.1f refresh_proof_reuse%%=%.1f refresh_proofs/f=%.1f proof_verify_checks=%llu proof_verify_mismatches=%llu",
        Rate(totals.targetMemoHits, totals.targetMemoLookups), perFrame(totals.targetMemoLookups), Rate(totals.drawInputMemoHits, totals.drawInputMemoLookups), perFrame(totals.drawInputMemoLookups),
        Rate(shaders.resultMemoHits, resultLookups), perFrame(resultLookups), perFrame(shaders.resultMemoEvictions),
        PerItem(static_cast<double>(totals.drawStages), totals.stagedDraws), static_cast<unsigned long long>(shaders.programKeysDistinct), perFrame(shaders.programKeys), Rate(shaders.programKeyTableHits, shaders.programKeys), static_cast<unsigned long long>(shaders.programKeyOverflow),
        perFrame(totals.halfChecks), perFrame(totals.halfStale), PerItem(Us(totals.halfCopyNs), totals.halfCopies), perFrame(totals.holdWaits), PerFrame(Ms(totals.holdWaitNs), frames),
        Rate(totals.epochProofReuses, epochProofs), perFrame(epochProofs), Rate(totals.refreshProofReuses, refreshProofs), perFrame(refreshProofs),
        static_cast<unsigned long long>(totals.proofVerifyChecks), static_cast<unsigned long long>(totals.proofVerifyMismatches));
    if (tail < 0) return {};
    line += text;
    const auto programLookups = shaders.programMemoHits + shaders.programMemoMisses;
    std::snprintf(text, sizeof(text), " program_memo_hit%%=%.1f program_memo_lookups/f=%.1f program_memo_resets=%llu program_split_verify_checks=%llu program_split_verify_mismatches=%llu",
        Rate(shaders.programMemoHits, programLookups), perFrame(programLookups), static_cast<unsigned long long>(shaders.programMemoResets),
        static_cast<unsigned long long>(shaders.programSplitVerifyChecks), static_cast<unsigned long long>(shaders.programSplitVerifyMismatches));
    return line + text + FormatRecompileSplit(shaders.split, frames) + "\n";
}

std::size_t FormatRunId(std::span<char> out, std::string_view describe, std::string_view built, std::span<std::string_view> environment) {
    // The newline and the NUL always fit.
    constexpr std::size_t Ending = 2;
    if (out.size() < Ending) {
        if (!out.empty()) out[0] = '\0';
        return 0;
    }
    const auto room = out.size() - Ending;
    std::size_t length = 0;
    const auto append = [&](std::string_view text) {
        const auto taken = std::min(text.size(), room - length);
        std::copy_n(text.data(), taken, out.data() + length);
        length += taken;
    };
    const auto set = std::partition(environment.begin(), environment.end(), [](std::string_view entry) { return entry.starts_with(EnvironmentPrefix); });
    std::sort(environment.begin(), set);
    append("[run-id] libs=");
    append(describe.empty() ? "unknown" : describe);
    append(" built=");
    append(built.empty() ? "unknown" : built);
    append(" env=");
    if (set == environment.begin()) append("-");
    for (auto entry = environment.begin(); entry != set; ++entry) {
        if (entry != environment.begin()) append(",");
        append(*entry);
    }
    out[length++] = '\n';
    out[length] = '\0';
    return length;
}

Counters& Global() {
    static Counters* counters = new Counters();
    return *counters;
}

std::string FormatGuest(const Totals& totals, const Window& window) {
    const auto frames = window.frames;
    const auto kbPerFrame = PerFrame(static_cast<double>(totals.dwords) * sizeof(std::uint32_t) / BytesPerKb, frames);
    const auto frameMs = window.gapMs.mean + window.spanMs.mean;
    const auto outsideMs = std::max(0.0, frameMs - window.insideMs.mean);
    const auto waitMs = [&](std::uint64_t ns) { return PerFrame(Ms(ns), frames); };
    const auto kernelMs = [&](std::size_t kind) { return waitMs(window.kernelWaitNs[kind]); };
    const auto backpressureMs = waitMs(totals.flipRoomNs) + waitMs(totals.flipHoldNs) + waitMs(totals.registrationNs) + kernelMs(WaitEqueue) + kernelMs(WaitSleep) + kernelMs(WaitVblank);
    const auto guestSyncMs = kernelMs(WaitEventFlag) + kernelMs(WaitSemaphore) + kernelMs(WaitCond) + kernelMs(WaitAddress);
    char text[4096];
    const auto length = std::snprintf(text, sizeof(text),
        "[frame-steps-guest] window_s=%.1f frames=%llu submits/f=%.1f submit_us/call=%.1f submit_ms/f=%.2f flip_submits/f=%.2f flip_submit_us/call=%.1f eop_reserves/f=%.2f eop_reserve_us/call=%.1f "
        "dwords/f=%.0f kb/f=%.1f packets/f=%.0f draws/f=%.0f "
        "gap_ms(p50/p95/mean)=%.2f/%.2f/%.2f span_ms(p50/p95/mean)=%.2f/%.2f/%.2f inside_calls_ms(p50/p95/mean)=%.2f/%.2f/%.2f outside_calls_ms/f=%.2f "
        "pm4_kb_per_ms_span=%.2f pm4_kb_per_ms_frame=%.2f q0_enqueue_to_packet_us=%.1f frame_ms=%.2f "
        "wait_flip_room_ms/f=%.2f wait_flip_hold_ms/f=%.2f wait_registration_ms/f=%.2f wait_equeue_ms/f=%.2f wait_sleep_ms/f=%.2f wait_vblank_ms/f=%.2f "
        "wait_evf_ms/f=%.2f wait_sema_ms/f=%.2f wait_cond_ms/f=%.2f wait_addr_ms/f=%.2f q0_render_wait_ms/f=%.2f "
        "backpressure_wait_ms/f=%.2f build_ms/f(proxy)=%.2f build_min_ms/f(proxy)=%.2f",
        window.seconds, static_cast<unsigned long long>(frames), PerFrame(static_cast<double>(totals.submits), frames), PerItem(Us(totals.submitNs), totals.submits), PerFrame(Ms(totals.submitNs), frames),
        PerFrame(static_cast<double>(totals.flipSubmits), frames), PerItem(Us(totals.flipSubmitNs), totals.flipSubmits),
        PerFrame(static_cast<double>(totals.eopReserves), frames), PerItem(Us(totals.eopReserveNs), totals.eopReserves),
        PerFrame(static_cast<double>(totals.dwords), frames), kbPerFrame, PerFrame(static_cast<double>(totals.packets), frames), PerFrame(static_cast<double>(totals.draws), frames),
        window.gapMs.p50, window.gapMs.p95, window.gapMs.mean, window.spanMs.p50, window.spanMs.p95, window.spanMs.mean,
        window.insideMs.p50, window.insideMs.p95, window.insideMs.mean, outsideMs,
        window.spanMs.mean > 0 ? kbPerFrame / window.spanMs.mean : 0.0, frameMs > 0 ? kbPerFrame / frameMs : 0.0, PerItem(Us(totals.queueLatencyNs), totals.queueLatencies), frameMs,
        waitMs(totals.flipRoomNs), waitMs(totals.flipHoldNs), waitMs(totals.registrationNs), kernelMs(WaitEqueue), kernelMs(WaitSleep), kernelMs(WaitVblank),
        kernelMs(WaitEventFlag), kernelMs(WaitSemaphore), kernelMs(WaitCond), kernelMs(WaitAddress), waitMs(totals.renderWaitNs),
        backpressureMs, std::max(0.0, frameMs - backpressureMs), std::max(0.0, frameMs - backpressureMs - guestSyncMs));
    return Finish(text, length);
}

std::string FormatLanes(const Totals& totals, const Window& window) {
    const auto frames = window.frames;
    const auto perFrameMs = [&](std::uint64_t ns) { return PerFrame(Ms(ns), frames); };
    const auto perFrameCount = [&](std::uint64_t count) { return PerFrame(static_cast<double>(count), frames); };
    double prepSum = 0, prepMax = 0, recSum = 0, recMax = 0;
    SumAndMax(totals.prepBusyNs, frames, prepSum, prepMax);
    SumAndMax(totals.recorderBusyNs, frames, recSum, recMax);
    const auto prepThreads = ThreadList(totals.prepBusyNs, frames);
    const auto recThreads = ThreadList(totals.recorderBusyNs, frames);
    char text[3072];
    const auto length = std::snprintf(text, sizeof(text),
        "[frame-steps] window_s=%.1f frames=%llu presents=%llu "
        "packet_busy_ms/f=%.2f packet_idle_ms/f=%.2f us/packet=%.2f packet_lock_wait_ms/f=%.2f packet_lock_hold_ms/f=%.2f packet_pool_wait_ms/f=%.2f "
        "prep_busy_ms/f(sum/max)=%.2f/%.2f prep_threads=%s prep_turn_wait_ms/f=%.2f prep_jobs/f=%.1f "
        "fifo_full_ms/f=%.2f fifo_drain_ms/f=%.2f fifo_empty_ms/f=%.2f fifo_depth=%.2f "
        "spine_busy_ms/f=%.2f spine_commits/f=%.0f draw_commit_us=%.2f spine_lock_hold_ms/f=%.2f spine_lock_wait_ms/f=%.2f chunks/f=%.1f draws/chunk=%.1f keep_sealed/f=%.1f epoch_joins/f=%.1f stale_inputs/f=%.1f stale_resources/f=%.1f "
        "rec_busy_ms/f(sum/max)=%.2f/%.2f rec_threads=%s "
        "vk_submits/f=%.1f vk_submit_ms/f=%.2f gpu_busy_ms/f=%.2f "
        "present_interval_ms(p50/p95)=%.2f/%.2f flip_to_present_ms(p50/p95/mean)=%.2f/%.2f/%.2f",
        window.seconds, static_cast<unsigned long long>(frames), static_cast<unsigned long long>(window.presents),
        perFrameMs(totals.packetBusyNs), perFrameMs(totals.packetIdleNs), PerItem(Us(totals.packetBusyNs), totals.packetPackets), perFrameMs(totals.lockWaitNs[PacketRole]), perFrameMs(totals.lockHoldNs[PacketRole]), perFrameMs(totals.packetPoolWaitNs),
        prepSum, prepMax, prepThreads.c_str(), perFrameMs(totals.prepTurnWaitNs), perFrameCount(totals.prepJobs),
        perFrameMs(totals.fifoFullNs), perFrameMs(totals.fifoDrainNs), perFrameMs(totals.fifoEmptyNs), PerItem(static_cast<double>(totals.fifoDepthSum), totals.fifoEnqueued),
        perFrameMs(totals.spineBusyNs), perFrameCount(totals.spineCommits), PerItem(Us(totals.drawCommitNs), totals.drawCommits), perFrameMs(totals.lockHoldNs[SpineRole]), perFrameMs(totals.lockWaitNs[SpineRole]), perFrameCount(totals.recorderChunks), PerItem(static_cast<double>(totals.recorderDraws), totals.recorderChunks), perFrameCount(totals.keepSealedChunks), perFrameCount(totals.chunkEpochJoins), perFrameCount(totals.staleRecordInputs), perFrameCount(totals.staleRecordResources),
        recSum, recMax, recThreads.c_str(),
        perFrameCount(totals.vkSubmits), perFrameMs(totals.vkSubmitNs), PerFrame(window.gpuBusyMs, frames),
        window.intervalMs.p50, window.intervalMs.p95, window.flipToPresentMs.p50, window.flipToPresentMs.p95, window.flipToPresentMs.mean);
    if (length < 0) return {};
    std::string line(text);
    const auto derived = Derive(totals, window);
    std::snprintf(text, sizeof(text), " packet_own_ms/f=%.2f coupling_ms=%.2f tail_ms=%.2f\n", derived.packetOwnMs, derived.couplingMs, derived.tailMs);
    return line + text;
}

std::string FormatPackets(const Totals& totals, const Window& window, SlotNamer name) {
    const auto frames = window.frames;
    std::uint64_t loopPackets = 0, loopNs = 0;
    for (std::size_t slot = 0; slot < PacketSlots; ++slot) {
        loopPackets += totals.slotPackets[slot];
        loopNs += totals.slotNs[slot];
    }
    const auto outside = [&](std::uint64_t whole, std::uint64_t part) { return PerFrame(Ms(whole > part ? whole - part : 0), frames); };
    char text[512];
    const auto length = std::snprintf(text, sizeof(text),
        "[frame-steps-packets] window_s=%.1f frames=%llu loop_packets/f=%.0f loop_ms/f=%.2f us/loop_packet=%.2f execute_outside_loop_ms/f=%.2f busy_outside_execute_ms/f=%.2f ",
        window.seconds, static_cast<unsigned long long>(frames), PerFrame(static_cast<double>(loopPackets), frames), PerFrame(Ms(loopNs), frames), PerItem(Us(loopNs), loopPackets),
        outside(totals.executeNs, loopNs), outside(totals.packetBusyNs, totals.executeNs));
    if (length < 0) return {};
    std::string line(text);
    line += "other_us/packet(gate/drain/lock/flush/prepare/body)=" + PhaseList(totals.otherPhaseNs, totals.phasePackets[0]);
    line += " draw_us/packet(gate/drain/lock/flush/prepare/body)=" + PhaseList(totals.drawPhaseNs, totals.phasePackets[1]);
    line += " draw_steps_us/drawn=" + DrawStepList(totals);
    line += " top=" + TopSlots(totals, frames, name) + "\n";
    return line;
}

}
