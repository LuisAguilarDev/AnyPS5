#include "prx/libSceAgcDriver/Execution/include/CaptureTrace.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/CaptureTrace.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Diagnostics.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/FrameSteps.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Memory/DrawWriteRanges.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawProgramTrace.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPrepPool.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Memory/DrawWriteRanges.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Synchronization/DeferredLabels.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/LockProfile.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceTimer.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "Optimization/ResourceProgram.hpp"
#include "prx/libc/include/HostThreadLocal.hpp"
#include <atomic>
#include <array>
#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdio>
#include <unordered_set>
#include <cstdlib>
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <unordered_map>
#include <atomic>
#include <exception>
#include <mutex>
#include <set>
#include <string>
#include <string_view>
#include <typeinfo>

namespace AgcDriver::DriverDetail {

namespace {

MutexProfile& drawCacheProfile() {
    static MutexProfile profile("draw cache");
    return profile;
}

// Takes PA_CL_VS_OUT_CNTL as a value, so a prep-pool job that runs the lookup decodes with the register
// the packet thread read for its packet, not the one a later packet set.
void decodeDrawVertexInfo(const DrawProgram& program, ShaderRecompiler::ProgramRole role, std::uint32_t vsOutControl, const Graphics::State& graphics, std::optional<ShaderRecompiler::ShaderVertexStageInfo>& info, std::vector<Graphics::DecodeRead>& reads) {
    if (program.binary.stage == ShaderRecompiler::ShaderStage::Fragment || role == ShaderRecompiler::ProgramRole::GeometryBack) return;
    reads.clear();
    info = DecodeOrderedVertexStageInfo(program.binary.header, program.binary.headerAddress, program.userData, reads, 8u - program.firstUserSgpr);
    if (info) {
        info->paClVsOutCntl = PositionExportLayout(program.snapshot.get(), program.codeOffset, program.binary.code, vsOutControl, program.tailContinuation.get()) | graphics.layerExports;
        info->depthRemapMode = graphics.depthRemap.mode;
        Graphics::ApplyPointSize(graphics, *info);
    }
}


bool TraceShaderCorrelation() {
    static const bool enabled = std::getenv("APS5_TRACE_SHADER_CORRELATION") != nullptr;
    return enabled && AgcDriver::CaptureTrace::Enabled();
}

std::atomic<std::uint64_t>& NextDrawTraceToken() {
    static std::atomic<std::uint64_t> next{0};
    return next;
}

std::string ExceptionReason(const std::exception_ptr& failure) {
    try {
        if (failure != nullptr) std::rethrow_exception(failure);
    } catch (const std::exception& error) {
        return error.what();
    } catch (...) {
    }
    return "unknown exception";
}

const char* ExceptionType(const std::exception_ptr& failure) {
    try {
        if (failure != nullptr) std::rethrow_exception(failure);
    } catch (const std::exception& error) {
        return typeid(error).name();
    } catch (...) {
    }
    return "unknown";
}

void TraceMaterializationFailure(std::uint64_t token, const Submission& submission, std::size_t packetOffset, const char* phase, std::size_t completedSubdraws, const std::exception_ptr& failure) {
    if (token == 0) return;
    try {
        const auto reason = ExceptionReason(failure);
        AgcDriver::CaptureTrace::Log("draw-materialization-failed token=%llu submission=%llu queue=%x offset=%zu phase=%s exception=%s reason=%.256s completedSubdraws=%zu",
                static_cast<unsigned long long>(token), static_cast<unsigned long long>(submission.serial), submission.queue, packetOffset, phase, ExceptionType(failure), reason.c_str(), completedSubdraws);
    } catch (...) {

    }
}

void TraceSubmissionFailure(std::uint64_t token, const Submission& submission, std::size_t packetOffset, std::size_t completedSubdraws, const std::exception_ptr& failure) {
    if (token == 0) return;
    try {
        const auto reason = ExceptionReason(failure);
        AgcDriver::CaptureTrace::Log("draw-submission-failed token=%llu submission=%llu queue=%x offset=%zu exception=%s reason=%.256s completedSubdraws=%zu",
                static_cast<unsigned long long>(token), static_cast<unsigned long long>(submission.serial), submission.queue, packetOffset, ExceptionType(failure), reason.c_str(), completedSubdraws);
    } catch (...) {

    }
}

void TraceDrawAttemptFailure(std::uint64_t token, const char* phase, const char* outcome, std::uint64_t submission, std::uint32_t queue, std::size_t packetOffset, std::string_view reason, std::string_view exceptionType = "none") {
    if (token == 0) return;
        AgcDriver::CaptureTrace::LogBestEffort("draw-attempt-failed token=%llu phase=%s outcome=%s submission=%llu queue=%x offset=%zu exception=%s reason=%.256s",
            static_cast<unsigned long long>(token), phase, outcome, static_cast<unsigned long long>(submission), queue, packetOffset, exceptionType.data(), reason.data());
}

void TraceDrawAttemptException(std::uint64_t token, const char* phase, std::uint64_t submission, std::uint32_t queue, std::size_t packetOffset, const std::exception_ptr& failure) noexcept {
    try {
        TraceDrawAttemptFailure(token, phase, "exception", submission, queue, packetOffset, ExceptionReason(failure), ExceptionType(failure));
    } catch (...) {

    }
}

const char* DrawVerdictName(DrawVerdict verdict) {
    switch (verdict) {
        case DrawVerdict::Drawn: return "Drawn";
        case DrawVerdict::Nothing: return "Nothing";
        case DrawVerdict::Rejected: return "Rejected";
    }
    return "Unknown";
}

std::string_view TrimmedReason(std::string_view reason) {
    if (reason.size() > 200) reason = reason.substr(0, 200);
    return reason;
    }

}


struct DrawTail {
    using GpuLock = std::unique_lock<std::remove_reference_t<decltype(GuestMemory::GpuMutex())>>;
    bool profile = false;
    bool frameSteps = false;
    std::array<double, DrawDriverPhaseCount> phaseMs{};
    std::uint64_t captures = 0;
    std::chrono::steady_clock::time_point phaseLap{};
    DrawPhaseTiming phaseTiming{false, phaseMs, phaseLap};
    Pm4::DrawParameters drawParameters{};
    bool pipelined = false;
    bool traceIndirect = false;
    std::uint64_t dumpTarget = 0;
    std::uint64_t dumpSlot1 = 0;
    std::shared_ptr<VulkanDevice> localDevice;
    bool registerKey = false;
    std::uint64_t drawKey = 0;
    std::shared_ptr<DrawEntry> entry;
    std::shared_ptr<const DrawDecode> decode;
    std::shared_ptr<const DrawDecode> relocatedDecode;
    std::shared_ptr<DrawEntry> dataEntry;
    std::uint64_t shapeKey = 0;
    ShaderRecompiler::ShaderPixelStageInfo pixel{};
    std::vector<DrawProgram> writablePrograms;
    const std::vector<DrawProgram>* programs = nullptr;
    std::vector<ShaderRecompiler::MemoryRegion> memory;
    std::vector<ShaderRecompiler::LinkedProgram> linked;
    std::vector<std::optional<ShaderRecompiler::ShaderVertexStageInfo>> vertexInfos;
    std::vector<std::vector<Graphics::DecodeRead>> decodeReads;
    std::unique_ptr<ShaderMemory> shaderMemory;
    std::vector<ShaderRecompiler::RecompileResult> results;
    std::vector<std::shared_ptr<const ShaderRecompiler::RecompileResult>> compiledStages;
    std::vector<Graphics::CompiledShader> stages;
    std::uint32_t pushCursorBytes = 0;
    std::vector<const ShaderRecompiler::RecompileResult*> programResults;
    std::vector<StageCapture> stageCaptures;
    std::vector<std::shared_ptr<DispatchVariant>> matched;
    std::vector<std::vector<ShaderRecompiler::MemoryRegion>> matchedRegions;
    std::vector<std::shared_ptr<DispatchVariant>> fresh;
    std::vector<bool> recompiled;
    DrawDataCandidates dataCandidates;
    std::vector<bool> reused;
    std::vector<StageDataHit> stageData;
    bool drawHit = false;
    bool verifyHit = false;
    bool stageEntries = false;
    bool patchedStages = false;
    const Submission* submission = nullptr;
    std::uint32_t submissionQueue = 0;
    std::uint64_t submissionSerial = 0;
    std::size_t packetOffset = 0;
    std::uint32_t packetOpcode = 0;
    std::uint32_t reg207 = 0;
    std::uint64_t epoch = 0;
    QueueState* queue = nullptr;
    std::span<const std::uint32_t> packet;
    std::string* rejected = nullptr;
    std::uint64_t traceToken = 0;
    VulkanDevice::DrawTraceContext trace{};
    std::string ownRejected;
    GpuLock* gpuLock = nullptr;
    GpuLock ownLock{GuestMemory::GpuMutex(), std::defer_lock};
};

namespace {

void noteFinisherHandOff(bool handed) {
    static std::atomic<std::uint64_t> handedOff{0}, kept{0};
    static std::atomic<std::int64_t> reportedAt{0};
    (handed ? handedOff : kept).fetch_add(1, std::memory_order_relaxed);
    static const bool stats = std::getenv("APS5_PIPELINE_STATS") != nullptr || std::getenv("APS5_PROFILE_DRAW") != nullptr;
    if (!stats) return;
    const auto tenSeconds = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count() / 10;
    if (reportedAt.exchange(tenSeconds, std::memory_order_relaxed) == tenSeconds) return;
    AgcDriver::ProfilePrint_nid_no_patch("[finisher] draws (10 s): %llu handed to the draw finisher, %llu finished on the worker%s\n", static_cast<unsigned long long>(handedOff.exchange(0)), static_cast<unsigned long long>(kept.exchange(0)), DrawFinisher::DrainText().c_str());
}

}

namespace {

enum class HeldReason : std::size_t { Known, NoDecode, Size, NullVertex, NullFragment, NullOther, Bda, Buffer, Storage, Count };
std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(HeldReason::Count)> heldReasons{};

void noteHeld(HeldReason reason) {
    heldReasons[static_cast<std::size_t>(reason)].fetch_add(1, std::memory_order_relaxed);
    static const bool profile = std::getenv("APS5_PIPELINE_STATS") != nullptr || std::getenv("APS5_PROFILE_DRAW") != nullptr;
    if (!profile) return;
    static std::atomic<std::uint64_t> total{0};
    const auto count = ++total;
    if (count % 262144u != 0u && (count > 262144u || (count & (count - 1u)) != 0u)) return;
    constexpr std::array<const char*, static_cast<std::size_t>(HeldReason::Count)> names{"targets only", "no decode", "stage count", "no variant (vertex)", "no variant (fragment)", "no variant (other)", "BDA stores", "written buffer", "storage image"};
    std::string text;
    for (std::size_t i = 0; i < names.size(); ++i) {
        char item[64];
        std::snprintf(item, sizeof(item), "%s %s %llu", i == 0 ? "" : ",", names[i], static_cast<unsigned long long>(count >= 262144u ? heldReasons[i].exchange(0) : heldReasons[i].load()));
        text += item;
    }
    std::fprintf(stderr, "[finisher-held] held draws' writes:%s\n", text.c_str());
}

std::optional<HeldReason> writesBeyondTargets(const ShaderRecompiler::RecompileResult& compiled) {
    using Role = ShaderRecompiler::DescriptorRole;
    for (const auto& binding : compiled.bindings) {
        if (binding.role == Role::BdaPagetable && Graphics::ShaderStoresThroughAddresses(compiled.spirv)) return HeldReason::Bda;
        if (binding.role == Role::GuestBuffers) {
            for (std::uint32_t element = 0; element < binding.count; ++element) {
                if (element >= binding.bufferWritten.size() || binding.bufferWritten[element]) return HeldReason::Buffer;
            }
        }
        if (binding.role == Role::GuestImages && binding.kind == ShaderRecompiler::DescriptorKind::StorageImage) {
            for (std::uint32_t element = 0; element < binding.count; ++element) {
                if (element >= binding.imageWritten.size() || binding.imageWritten[element]) return HeldReason::Storage;
            }
        }
    }
    return std::nullopt;
}

DrawFinisher::Pending heldWrites(const DrawTail& tail) {
    DrawFinisher::Pending held;
    held.key = tail.drawKey;
    held.shape = tail.shapeKey;
    if (tail.decode == nullptr) {
        noteHeld(HeldReason::NoDecode);
        return held;
    }
    if (tail.matched.size() != tail.programs->size()) {
        noteHeld(HeldReason::Size);
        return held;
    }
    const auto& roles = tail.decode->roles;
    for (std::size_t i = 0; i < tail.matched.size(); ++i) {
        if (i < roles.size() && roles[i] == ShaderRecompiler::ProgramRole::GeometryBack) continue;
        const auto& variant = tail.matched[i];
        if (variant == nullptr || variant->compiled == nullptr) {
            const bool candidates = !tail.drawHit && i < tail.dataCandidates.size() && !tail.dataCandidates[i].empty();
            const auto& program = (*tail.programs)[i];
            if (!candidates && program.snapshot != nullptr && !ProgramWritesMemory(*program.snapshot, program.codeOffset) && (program.tailContinuation == nullptr || !ProgramWritesMemory(*program.tailContinuation, 0))) continue;
            if (!candidates) {
                const auto stage = (*tail.programs)[i].binary.stage;
                noteHeld(stage == ShaderRecompiler::ShaderStage::Fragment ? HeldReason::NullFragment : stage == ShaderRecompiler::ShaderStage::Vertex ? HeldReason::NullVertex : HeldReason::NullOther);
                return held;
            }
            for (const auto& [candidate, regions] : tail.dataCandidates[i]) {
                if (candidate == nullptr || candidate->compiled == nullptr) {
                    noteHeld(HeldReason::NullOther);
                    return held;
                }
                if (const auto reason = writesBeyondTargets(*candidate->compiled)) {
                    noteHeld(*reason);
                    return held;
                }
            }
            continue;
        }
        if (const auto reason = writesBeyondTargets(*variant->compiled)) {
            noteHeld(*reason);
            return held;
        }
        if (i < tail.stageData.size() && tail.stageData[i].compiled != nullptr) {
            if (const auto reason = writesBeyondTargets(*tail.stageData[i].compiled)) {
                noteHeld(*reason);
                return held;
            }
        }
    }
    noteHeld(HeldReason::Known);
    held.writes = DrawWriteRanges(tail.decode->state, {}, ExactDrawWrites());
    held.unknown = false;
    return held;
}

bool EmptyIndirectSkip() {
    static const bool enabled = std::getenv("APS5_NO_EMPTY_INDIRECT_SKIP") == nullptr;
    return enabled;
}

struct CpuIndirectProgramsStorage {};

std::unordered_map<std::uint64_t, bool>& CpuIndirectPrograms() {
    return HostThreadLocal<std::unordered_map<std::uint64_t, bool>, CpuIndirectProgramsStorage>();
}

}

DrawVerdict Driver::draw(QueueState& queue, std::span<const std::uint32_t> packet, const Submission& submission, std::size_t packetOffset, std::string& rejected) {
    PerformanceTimer timing("Driver.Draw");
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    auto tailHolder = std::make_shared<DrawTail>();
    auto& tail = *tailHolder;
    auto& phaseMs = tail.phaseMs;
    auto& captures = tail.captures;
    auto& phaseLap = tail.phaseLap;
    tail.frameSteps = FrameSteps::Enabled() && submission.queue == 0;
    phaseLap = profile || tail.frameSteps ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    tail.profile = profile;
    tail.phaseTiming.profile = profile || tail.frameSteps;
    auto& phaseTiming = tail.phaseTiming;
    if (profile && packetStartedAt() != std::chrono::steady_clock::time_point{}) phaseMs[DrawRowPrologue] = std::chrono::duration<double, std::milli>(phaseLap - packetStartedAt()).count();

    const auto drawn = [&] {
        phaseTiming.Phase(DrawRowVectors);
        if (tail.frameSteps) FrameSteps::NoteDrawSteps(FrameSteps::Global(), phaseMs);
        if (!profile) return DrawVerdict::Drawn;
        auto& pending = pendingDrawPhases();
        pending.phases = true;
        pending.captures = captures;
        pending.ms = phaseMs;
        pending.tailAt = phaseLap;
        return DrawVerdict::Drawn;
    };
    const std::uint64_t traceToken = TraceShaderCorrelation() ? NextDrawTraceToken().fetch_add(1, std::memory_order_relaxed) + 1 : 0;
    if (traceToken != 0) {
        AgcDriver::CaptureTrace::LogBestEffort("draw-attempt token=%llu submission=%llu queue=%x offset=%zu",
                static_cast<unsigned long long>(traceToken), static_cast<unsigned long long>(submission.serial), submission.queue, packetOffset);
    }
    auto& drawParameters = tail.drawParameters;
    try {
        drawParameters = Pm4::ResolveDraw(packet, queue);
    } catch (...) {
        TraceDrawAttemptException(traceToken, "resolve-draw", submission.serial, submission.queue, packetOffset, std::current_exception());
        throw;
    }
    tail.pipelined = DrawPipeline::Active();
    const bool pipelined = tail.pipelined;
    static const bool pipelinedIndirect = std::getenv("APS5_NO_PIPELINED_INDIRECT") == nullptr;
    if (pipelined && drawParameters.indirect) {
        const auto& indirect = *drawParameters.indirect;
        auto& pipeline = DrawPipeline::Queue0();
        const bool overlaps = pipeline.Busy() && (pipeline.Overlaps(indirect.arguments, static_cast<std::size_t>(std::max<std::uint64_t>(indirect.RangeBytes(), indirect.recordBytes))) || (indirect.countIndirect && pipeline.Overlaps(indirect.countAddress, 4)));
        if (!pipelinedIndirect || overlaps) pipeline.Drain(DrawPipeline::DrainReason::Indirect);
    }
    auto& traceIndirect = tail.traceIndirect;
    if (const auto verdict = precheckDraw(queue, submission, packetOffset, packet, drawParameters, rejected, traceIndirect)) {
        TraceDrawAttemptFailure(traceToken, "precheck", "rejected", submission.serial, submission.queue, packetOffset, rejected);
        return *verdict;
    }
    phaseTiming.Phase(DrawRowPrecheck);
    using Stage = ShaderRecompiler::ShaderStage;
    using Role = ShaderRecompiler::ProgramRole;

    static const std::uint64_t dumpTarget = [] { const char* text = std::getenv("APS5_DUMP_DRAW_SHADERS"); return text ? std::strtoull(text, nullptr, 16) : 0ull; }();

    static const std::uint64_t dumpSlot1 = [] { const char* text = std::getenv("APS5_DUMP_DRAW_SLOT1"); return text ? std::strtoull(text, nullptr, 16) : 0ull; }();

    static const bool lockedPrepare = std::getenv("APS5_LOCKED_DRAW_PREPARE") != nullptr;
    std::unique_lock gpuLock(GuestMemory::GpuMutex(), std::defer_lock);
    auto& localDevice = tail.localDevice;
    if (lockedPrepare) {

        GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Draw);
        gpuLock.lock();
        timing.Mark("gpu_mutex_wait");
        phaseTiming.Phase(DrawRowLockWait);
        if (device == nullptr) device = std::make_shared<VulkanDevice>();
        localDevice = device;

        recordLabelsForPacket(localDevice.get(), submission.queue);
        phaseTiming.Phase(DrawRowLabels);
    } else if ((localDevice = device.Load()) == nullptr) {
        GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Draw);
        std::lock_guard createLock(GuestMemory::GpuMutex());
        if (device == nullptr) device = std::make_shared<VulkanDevice>();
        localDevice = device;
    }
    timing.Mark("device_setup");
    phaseTiming.Phase(DrawRowVectors);

    const bool useDrawEntries = drawEntries() && !ShaderRecompiler::DebugProbeActive() && dumpTarget == 0 && dumpSlot1 == 0;
    tail.registerKey = useDrawEntries && registerKeyEnabled();
    const bool registerKey = tail.registerKey;
    auto& drawKey = tail.drawKey;
    std::array<std::uint64_t, 5> drawKeyComponents{};
    auto& entry = tail.entry;
    auto& decode = tail.decode;
    auto& dataEntry = tail.dataEntry;
    auto& shapeKey = tail.shapeKey;
    std::uint64_t drawIdentity = 0, drawAlias = 0;
    std::vector<DrawUserPatch> userPatches;
    if (registerKey) {
        const auto keyStart = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        std::shared_ptr<const DrawRelocation> relocation;
        bool aliasFirst = false;
        if (drawRelocationEnabled()) {
            if (DrawFinisher::Busy() && drawRelocationPending.key != 0) DrawFinisher::Get().Drain();
            drawIdentity = drawIdentityKey(queue, *submission.shaders);
            ProfiledLock cacheLock(drawCacheMutex, drawCacheProfile());
            learnPendingRelocation();
            const auto known = drawRelocations.find(drawIdentity);
            if (known != drawRelocations.end()) {
                auto& trial = drawRelocationTrials[drawIdentity];
                if (trial.idle != 0) --trial.idle;
                else relocation = known->second;
                aliasFirst = relocation != nullptr && trial.productive;
            }
        }
        drawKey = drawRegisterKey(queue, *submission.shaders, localDevice->Serial(), &shapeKey, traceDrawCache() ? &drawKeyComponents : nullptr, aliasFirst ? relocation.get() : nullptr, aliasFirst ? &drawAlias : nullptr);
        if (DrawFinisher::Busy()) {
            bool shapeMatters = false;
            if (DrawFinisher::Overlap() && drawDataHits()) {
                std::lock_guard peekLock(drawCacheMutex);
                const auto shape = drawShapes.find(shapeKey);
                shapeMatters = !drawCache.contains(drawKey) && (shape == drawShapes.end() || !drawCache.contains(shape->second));
            }
            const bool drain = !DrawFinisher::Overlap() || DrawFinisher::Get().PendingKey(drawKey, shapeMatters ? shapeKey : 0);
            if (drain) DrawFinisher::Get().Drain();
            if (DrawFinisher::Overlap()) DrawFinisher::NoteDrain(DrawFinisher::DrainSite::Lookup, drain);
        }
        ProfiledLock cacheLock(drawCacheMutex, drawCacheProfile());
        ++drawEntryCounters.lookups;
        ++drawEntryCounters.registerKeyLookups;
        if (profile) drawEntryCounters.keyUs += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - keyStart).count();
        const auto found = drawCache.find(drawKey);
        if (found != drawCache.end()) {
            entry = found->second;
            decode = entry->decode;
        } else {
            ++drawEntryCounters.absent;
            if (relocation != nullptr && !aliasFirst) {
                cacheLock.unlock();
                drawRegisterKey(queue, *submission.shaders, localDevice->Serial(), nullptr, nullptr, relocation.get(), &drawAlias);
                cacheLock.lock();
            }
            if (drawIdentity != 0) drawRelocationPending = {drawKey, drawAlias, drawIdentity};
            if (drawAlias != 0) relocateDrawEntry(queue, drawAlias, entry, decode, userPatches);
            if (relocation != nullptr) {
                auto& trial = drawRelocationTrials[drawIdentity];
                ++trial.lookups;
                if (trial.lookups >= DrawRelocationTrial::Window) {
                    trial.productive = trial.hits * DrawRelocationTrial::MinimumHitShare >= trial.lookups;
                    if (!trial.productive) trial.idle = DrawRelocationTrial::IdleDraws;
                    trial.lookups = trial.hits = 0;
                }
            }
            if (traceDrawCache()) {
                auto& counters = drawEntryCounters;
                if (traceKeysEvicted.contains(drawKey)) ++counters.absentEvicted;
                else if (traceKeysSeen.contains(drawKey)) ++counters.absentNeverInserted;
                else if (traceShapes.contains(shapeKey)) ++counters.absentUserWords;
                else ++counters.absentNewRegisters;
            }
            if (decode == nullptr && drawDataHits() && !drawParameters.indirect) {
                if (const auto shape = drawShapes.find(shapeKey); shape != drawShapes.end()) {
                    const auto kept = drawCache.find(shape->second);
                    if (kept != drawCache.end() && kept->second->decode != nullptr && kept->second->shape == shapeKey) dataEntry = kept->second;
                }
                if (dataEntry != nullptr) ++drawEntryCounters.dataHits;
            }
        }
        if (traceDrawCache()) traceDrawKeyLookup(drawKeyComponents, found == drawCache.end());
        if (traceDrawCache()) {
            traceInsert(traceKeysSeen, drawKey);
            traceInsert(traceShapes, shapeKey);
        }
        maybeReportDrawCache(profile);
    }
    if (decode != nullptr && !DecodeProgramsCurrent(*decode, *submission.shaders)) {
        userPatches.clear();
        ProfiledLock cacheLock(drawCacheMutex, drawCacheProfile());
        eraseDrawEntry(drawKey, entry);
        ++drawEntryCounters.absent;
        entry = nullptr;
        decode = nullptr;
    }
    if (dataEntry != nullptr && !DecodeProgramsCurrent(*dataEntry->decode, *submission.shaders)) dataEntry = nullptr;
    phaseTiming.Phase(DrawRowKeyLookupValidate);

    if (dataEntry != nullptr) {
        auto refreshed = std::make_shared<DrawDecode>(*dataEntry->decode);
        for (auto& program : refreshed->programs) {
            ReadUserWords(queue, program, false);
            if (program.merged) InitializeMerged(queue, program, program.mergedPointer, program.mergedPointerRequired, false);
        }
        if (verifyDrawDataHits()) {
            const auto full = decodeDraw(queue, submission, localDevice->graphicsContext());
            if (!sameDecode(*refreshed, *full)) {
                std::size_t differing = 0;
                while (differing < full->programs.size() && differing < refreshed->programs.size() && full->programs[differing].userData == refreshed->programs[differing].userData) ++differing;
                char message[320];
                std::snprintf(message, sizeof(message), "APS5_VERIFY_DRAW_DATA_HITS: a data-only hit's decode differs from the full decode (key 0x%llx, shape 0x%llx, %zu / %zu programs, first with differing user words %zu, path %u / %u, target 0x%llx / 0x%llx)", static_cast<unsigned long long>(drawKey), static_cast<unsigned long long>(shapeKey), refreshed->programs.size(), full->programs.size(), differing, static_cast<unsigned>(refreshed->state.stages.path), static_cast<unsigned>(full->state.stages.path), static_cast<unsigned long long>(refreshed->state.color.address), static_cast<unsigned long long>(full->state.color.address));
                throw std::runtime_error(message);
            }
        }
        decode = std::move(refreshed);
    }
    try {
        resolveDrawDecode(queue, submission, localDevice->graphicsContext(), decode, registerKey, drawKey, profile);
    } catch (...) {
        TraceDrawAttemptFailure(traceToken, "decode", "exception", submission.serial, submission.queue, packetOffset, ExceptionReason(std::current_exception()));
        throw;
    }
    const auto& graphics = decode->state;
    auto& pixel = tail.pixel;
    pixel = decode->pixel;
    const bool writesPrograms = graphics.stages.mesh || drawParameters.indirect || !userPatches.empty();
    auto& writablePrograms = tail.writablePrograms;
    if (writesPrograms) writablePrograms = decode->programs;
    for (const auto& patch : userPatches) writablePrograms[patch.program].userData[patch.index] = patch.value;
    tail.programs = writesPrograms ? &writablePrograms : &decode->programs;
    const auto& programs = *tail.programs;
    if (const auto skipped = SkippedShader(); skipped != 0 && std::any_of(programs.begin(), programs.end(), [skipped](const DrawProgram& program) { return program.binary.codeAddress == skipped; })) return drawn();
    const bool emptyIndirectSkip = EmptyIndirectSkip();
    auto& cpuIndirectPrograms = CpuIndirectPrograms();
    const auto recordDraws = [&](const Pm4::DrawArguments& arguments) { return arguments.count != 0 && arguments.instances != 0 && (!drawParameters.indexed || !drawParameters.indirect->indexLimitKnown || arguments.firstVertexOrIndex < drawParameters.indexCount); };
    if (emptyIndirectSkip && drawParameters.indirect && !programs.empty()) {
        const auto known = cpuIndirectPrograms.find(programs.front().binary.codeAddress);
        if (known != cpuIndirectPrograms.end() && known->second) {
            const auto& indirect = *drawParameters.indirect;
            recordQueuedLabelsBeforeRead(submission.queue);
            const auto readStart = std::chrono::steady_clock::now();
            const auto count = std::min(indirect.countIndirect ? Pm4::ReadDrawCount(indirect) : indirect.count, indirect.count);
            bool any = false;
            for (std::uint32_t record = 0; record < count && !any; ++record) any = recordDraws(Pm4::ReadDrawArguments(indirect, record));
            if (!any) {
                Graphics::CountEmptyIndirectDraw(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - readStart).count());
                timing.Mark("empty_indirect_skipped");
                return drawn();
            }
        }
    }
    const auto setMeshIndexBuffer = [&](const Pm4::DrawParameters& parameters) {
        if (!graphics.stages.mesh) return;
        auto& words = writablePrograms.front().userData;
        require(programs.front().firstUserSgpr == 0 && words.size() >= ShaderRecompiler::MeshIndexBufferUserWord + 4, "mesh program lacks the hidden user words");
        const auto descriptor = Graphics::MeshIndexBufferDescriptor(parameters);
        std::copy(descriptor.begin(), descriptor.end(), words.begin() + ShaderRecompiler::MeshIndexBufferUserWord);
    };
    if (!drawParameters.indirect) setMeshIndexBuffer(drawParameters);
    else if (graphics.stages.mesh) setMeshIndexBuffer(Pm4::DrawParameters{drawParameters.indexAddress, std::max(drawParameters.indexCount, 1u), drawParameters.indexSize, 1, 0, drawParameters.indexed});
    const std::vector<Role>& roles = decode->roles;
    static const bool tracePrograms = std::getenv("APS5_TRACE_DRAW_PROGRAMS") != nullptr;
    if (tracePrograms) {
        std::vector<DrawProgramTraceEntry> identities;
        identities.reserve(programs.size());
        for (std::size_t i = 0; i < programs.size(); ++i) {
            identities.push_back({programs[i].binary.stage, roles[i], programs[i].snapshot->codeAddress, programs[i].binary.codeAddress});
        }
        const auto line = FormatDrawProgramTrace(submission.serial, submission.queue, identities);
        std::fprintf(stderr, "%s\n", line.c_str());
    }
    phaseTiming.Phase(DrawRowDecode);

    const auto locate = [&](std::uint32_t location) -> std::optional<std::pair<std::size_t, std::size_t>> {
        const auto words = UserDataWords(programs, roles, location);
        if (words.empty()) return std::nullopt;
        return words.front();
    };
    if (drawParameters.indirect) {
        auto& indirect = *drawParameters.indirect;
        const auto sgprOf = [&](std::uint32_t location) -> std::int32_t {
            const auto word = locate(location);
            if (!word || word->first != 0) return -1;
            return static_cast<std::int32_t>(programs.front().firstUserSgpr + word->second);
        };
        indirect.baseVertexSgpr = sgprOf(indirect.baseVertexLocation);
        indirect.startInstanceSgpr = sgprOf(indirect.startInstanceLocation);
        indirect.drawIndexSgpr = sgprOf(indirect.drawIndexLocation);
    }
    if (DrawFinisher::Busy()) {
        const bool drain = !DrawFinisher::Overlap();
        if (drain) DrawFinisher::Get().Drain();
        else DrawFinisher::NoteDrain(DrawFinisher::DrainSite::Reads, false);
    }
    auto& memory = tail.memory;
    auto& linked = tail.linked;
    memory.reserve(2 * programs.size() + 32);
    AppendProgramMemory(memory, programs);
    linked.reserve(programs.size());
    for (std::size_t i = 0; i < programs.size(); ++i) {
        const auto& program = programs[i];
        linked.push_back({roles[i], program.binary, program.userDataBase, program.firstUserSgpr, program.userData});
    }
    timing.Mark("prepare");
    phaseTiming.Phase(DrawRowProgramPrepare);

    const std::uint32_t vsOutControl = readRegister(queue.context, 0x207);
    auto& vertexInfos = tail.vertexInfos;
    vertexInfos.resize(programs.size());
    auto& decodeReads = tail.decodeReads;
    decodeReads.resize(programs.size());
    const auto decodeVertexInfo = [&](std::size_t i) {
        const auto& program = programs[i];
        if (program.binary.stage == Stage::Fragment || roles[i] == Role::GeometryBack) return;
        vertexInfos[i] = DecodeOrderedVertexStageInfo(program.binary.header, program.binary.headerAddress, program.userData, decodeReads[i], 8u - program.firstUserSgpr);
        if (vertexInfos[i]) {
            vertexInfos[i]->paClVsOutCntl = PositionExportLayout(program.snapshot.get(), program.codeOffset, program.binary.code, readRegister(queue.context, 0x207), program.tailContinuation.get()) | graphics.layerExports;
            vertexInfos[i]->depthRemapMode = graphics.depthRemap.mode;
            Graphics::ApplyPointSize(graphics, *vertexInfos[i]);
        }
    };
    if (!registerKey) {
        for (std::size_t i = 0; i < programs.size(); ++i) decodeVertexInfo(i);
        phaseTiming.Phase(DrawRowDecode);
    }
    auto& shaderMemory = tail.shaderMemory;
    shaderMemory = std::make_unique<ShaderMemory>(memory, &queryPendingWrite, &observePendingWrite, hookWaitCounter(), &QueryPendingSpan);
    auto& results = tail.results;
    auto& compiledStages = tail.compiledStages;
    compiledStages.reserve(programs.size());
    auto& stages = tail.stages;
    stages.reserve(programs.size());
    auto& pushCursorBytes = tail.pushCursorBytes;

    auto& programResults = tail.programResults;
    programResults.assign(programs.size(), nullptr);

    auto& stageCaptures = tail.stageCaptures;
    stageCaptures.resize(programs.size());
    auto& matched = tail.matched;
    matched.resize(programs.size());
    auto& matchedRegions = tail.matchedRegions;
    matchedRegions.resize(programs.size());

    auto& fresh = tail.fresh;
    fresh.resize(programs.size());

    auto& recompiled = tail.recompiled;
    recompiled.assign(programs.size(), false);
    auto& dataCandidates = tail.dataCandidates;
    dataCandidates.resize(programs.size());
    auto& reused = tail.reused;
    reused.assign(programs.size(), false);
    auto& stageData = tail.stageData;
    stageData.resize(programs.size());
    auto& drawHit = tail.drawHit;
    auto& verifyHit = tail.verifyHit;
    tail.stageEntries = useDrawEntries && !alwaysCaptureStages() && !(graphics.stages.tessellation && !drawParameters.indirect && drawParameters.instanceCount > 1u);
    const bool stageEntries = tail.stageEntries;
    static const bool prepPool = DrawPrepPool::Threads() != 0;
    static const bool poolDataHits = std::getenv("APS5_POOL_DATA_HITS") != nullptr;
    static const bool poolDataHitsV2 = std::getenv("APS5_NO_POOL_DATA_HITS_V2") == nullptr;
    const bool poolEligible = prepPool && pipelined && !drawParameters.indirect && deferredLabels().labels.empty() && dumpTarget == 0 && dumpSlot1 == 0 && !lockedPrepare && !verifyDrawDataHits() && !verifyDrawRecipe() && !traceCapSync() && !graphics.stages.mesh && traceToken == 0;
    // A draw the pool will take whatever the lookup finds leaves the lookup, its hit validation and
    // what follows from them to the job (APS5_NO_POOL_LOOKUP=1 keeps them here).
    const bool poolLookup = poolEligible && (dataEntry == nullptr || (poolDataHitsV2 && !verifyDataCandidates())) && DrawPrepPool::Lookup();
    if (!poolLookup) {
        try {
            lookupDraw(submission, localDevice, graphics, pixel, programs, roles, vertexInfos, stageEntries, registerKey, profile, drawKey, entry, matched, matchedRegions, drawHit, verifyHit, phaseTiming, phaseMs, dataEntry, dataCandidates, stageData);
        } catch (...) {
            TraceDrawAttemptException(traceToken, "lookup", submission.serial, submission.queue, packetOffset, std::current_exception());
            throw;
        }
    }
    for (std::size_t i = 0; i < matched.size(); ++i) {
        if (matched[i] == nullptr || matched[i]->shaderDataPatches.empty()) continue;
        applyShaderDataPatches(*matched[i]);
        if (stageData[i].compiled != nullptr) stageData[i].compiled = patchDataWords(*matched[i], stageData[i].words);
    }
    const auto relocatedDecode = decode;
    if (!poolLookup && !userPatches.empty()) {
        ProfiledLock cacheLock(drawCacheMutex, drawCacheProfile());
        if (drawHit || verifyHit) {
            ++drawRelocationCounters.hits;
            ++drawRelocationTrials[drawIdentity].hits;
            if (const auto alias = drawAliases.find(drawAlias); alias != drawAliases.end()) {
                const auto donor = drawCache.find(alias->second);
                if (donor != drawCache.end() && drawCacheHits - donor->second->touched > drawCacheEntries() / 8) {
                    drawOrder.splice(drawOrder.begin(), drawOrder, donor->second->order);
                    donor->second->touched = drawCacheHits;
                    ++drawEntryCounters.touches;
                }
            }
        }
        if (profile && drawRelocationCounters.lookups >= 100000) reportDrawRelocation();
    }
    if (!poolLookup && !userPatches.empty() && !drawHit) {
        auto current = std::make_shared<DrawDecode>(*decode);
        for (const auto& patch : userPatches) current->programs[patch.program].userData[patch.index] = patch.value;
        decode = std::move(current);
    }
    tail.patchedStages = std::any_of(stageData.begin(), stageData.end(), [](const StageDataHit& hit) { return hit.compiled != nullptr; });
    tail.relocatedDecode = relocatedDecode;
    std::uint64_t stageDataVerified = 0;
    const auto target = graphics.hasColorTarget ? graphics.color.address : 0;
    const VulkanDevice::DrawTraceContext trace{traceToken, submission.serial, submission.queue, packetOffset, target};
    const auto* tracePtr = trace.token != 0 ? &trace : nullptr;
    tail.traceToken = traceToken;
    tail.trace = trace;
    if (trace.token != 0) {
        for (std::size_t i = 0; i < programs.size(); ++i) {
            const auto& program = programs[i];
            AgcDriver::CaptureTrace::LogBestEffort("draw-selected token=%llu submission=%llu queue=%x offset=%zu snapshot=0x%llx code=%llx entryOffset=%zu entry=%llx stage=%u role=%u drawCacheHit=%u variantHit=%u target=%llx",
                    static_cast<unsigned long long>(trace.token), static_cast<unsigned long long>(submission.serial), submission.queue, packetOffset,
                    static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(program.snapshot.get())), static_cast<unsigned long long>(program.snapshot->codeAddress), program.codeOffset,
                    static_cast<unsigned long long>(program.snapshot->codeAddress + program.codeOffset), static_cast<unsigned>(program.binary.stage), static_cast<unsigned>(roles[i]),
                    drawHit ? 1u : 0u, matched[i] != nullptr ? 1u : 0u, static_cast<unsigned long long>(target));
        }
    }

    if (registerKey && !poolLookup) {

        for (std::size_t i = 0; i < programs.size(); ++i) {
            if (matched[i] != nullptr && drawHit && !verifyDrawRecipe()) {
                if (matched[i]->vertexInfo != nullptr) vertexInfos[i] = *matched[i]->vertexInfo;
                continue;
            }
            if (!dataCandidates[i].empty() && !verifyDataCandidates()) {
                const auto& info = dataCandidates[i].front().first->vertexInfo;
                if (info != nullptr) vertexInfos[i] = *info;
                continue;
            }
            decodeVertexInfo(i);
            if (matched[i] != nullptr && verifyDrawRecipe() && (vertexInfos[i].has_value() != (matched[i]->vertexInfo != nullptr) || (vertexInfos[i] && !sameVertexInfo(*vertexInfos[i], *matched[i]->vertexInfo)))) {
                static std::atomic<std::uint64_t> reports{0};
                if (reports.fetch_add(1) < 20) std::fprintf(stderr, "[draw-cache] verify: stage %zu (program 0x%llx) of a hit has a vertex stage info unlike its variant's\n", i, static_cast<unsigned long long>(programs[i].binary.codeAddress));
                ProfiledLock cacheLock(drawCacheMutex, drawCacheProfile());
                ++drawEntryCounters.verifyDecodeMismatches;
            }
        }
        phaseTiming.Phase(DrawRowDecode);
    }

    if (prepPool && pipelined) {
        static std::array<std::atomic<std::uint64_t>, 8> poolMiss{};
        static std::atomic<std::uint64_t> poolSeen{0};
        const int reason = drawParameters.indirect ? 1 : !deferredLabels().labels.empty() ? 2 : (dumpTarget != 0 || dumpSlot1 != 0) ? 3 : lockedPrepare ? 4 : dataEntry != nullptr && !poolLookup ? 5 : !std::all_of(dataCandidates.begin(), dataCandidates.end(), [](const auto& candidates) { return candidates.empty(); }) ? 6 : (verifyDrawDataHits() || verifyDrawRecipe() || traceCapSync()) ? 7 : 0;
        poolMiss[reason].fetch_add(1, std::memory_order_relaxed);
        if (poolSeen.fetch_add(1, std::memory_order_relaxed) % 100000 == 99999) std::fprintf(stderr, "[prep-pool-gate] eligible %llu, indirect %llu, labels %llu, dump %llu, locked %llu, data entry %llu, candidates %llu, verify %llu\n", static_cast<unsigned long long>(poolMiss[0].load()), static_cast<unsigned long long>(poolMiss[1].load()), static_cast<unsigned long long>(poolMiss[2].load()), static_cast<unsigned long long>(poolMiss[3].load()), static_cast<unsigned long long>(poolMiss[4].load()), static_cast<unsigned long long>(poolMiss[5].load()), static_cast<unsigned long long>(poolMiss[6].load()), static_cast<unsigned long long>(poolMiss[7].load()));
    }
    if (poolEligible && (dataEntry == nullptr || poolDataHits || poolLookup) && std::all_of(dataCandidates.begin(), dataCandidates.end(), [](const auto& candidates) { return candidates.empty(); })) {
        struct Pooled {
            std::shared_ptr<VulkanDevice> localDevice;
            std::shared_ptr<const DrawDecode> decode;
            ShaderRecompiler::ShaderPixelStageInfo pixel;
            std::vector<DrawProgram> programs;
            Pm4::DrawParameters drawParameters;
            std::vector<std::optional<ShaderRecompiler::ShaderVertexStageInfo>> vertexInfos;
            std::vector<std::vector<Graphics::DecodeRead>> decodeReads;
            std::vector<ShaderRecompiler::MemoryRegion> memory;
            std::vector<ShaderRecompiler::LinkedProgram> linked;
            std::vector<std::shared_ptr<DispatchVariant>> matched;
            std::vector<std::vector<ShaderRecompiler::MemoryRegion>> matchedRegions;
            std::shared_ptr<DrawEntry> entry;
            bool drawHit;
            bool verifyHit;
            bool useDrawEntries;
            bool registerKey;
            std::uint64_t drawKey;
            std::uint64_t shapeKey;
            std::uint64_t epoch;
            std::uint32_t queueId;
            std::uint32_t commitPacket;
            std::vector<StageDataHit> stageData;
            bool patchedStages;
            std::shared_ptr<DrawEntry> dataEntry;
            // The lookup moved into the job (APS5_NO_POOL_LOOKUP unset) and what it starts from: the
            // entry and decode the register key found, the relocation patches, and PA_CL_VS_OUT_CNTL.
            bool poolLookup;
            bool lookupEntries;
            std::shared_ptr<DrawEntry> lookupEntry;
            std::shared_ptr<const DrawDecode> relocatedDecode;
            std::vector<DrawUserPatch> userPatches;
            std::uint64_t drawIdentity;
            std::uint32_t vsOutControl;
            DrawDataCandidates candidates;
        };
        std::optional<std::vector<DrawPrepPool::Range>> writeBound;
        if (drawHit && DrawPrepPool::WriteBounds()) {
            std::vector<Graphics::CompiledShader> boundStages;
            for (std::size_t i = 0; i < programs.size(); ++i) {
                if (roles[i] == Role::GeometryBack) continue;
                boundStages.push_back({programs[i].binary.stage, stageData[i].compiled != nullptr ? stageData[i].compiled.get() : matched[i]->compiled.get(), 0});
            }
            const auto ranges = drawWriteRanges(graphics, boundStages);
            writeBound.emplace(ranges.begin(), ranges.end());
        }
        auto pooledDraw = std::make_shared<Pooled>(Pooled{std::move(localDevice), std::move(decode), pixel, std::move(programs), drawParameters, std::move(vertexInfos), std::move(decodeReads), std::move(memory), std::move(linked), std::move(matched), std::move(matchedRegions), entry, drawHit, verifyHit, useDrawEntries, registerKey, drawKey, shapeKey, DrawPipeline::EpochToken().load(std::memory_order_relaxed), submission.queue, (packet[0] >> 8u) & 0xffu, std::move(stageData), tail.patchedStages, std::move(dataEntry), poolLookup, stageEntries, std::move(entry), relocatedDecode, poolLookup ? std::move(userPatches) : std::vector<DrawUserPatch>{}, drawIdentity, vsOutControl, DrawDataCandidates{}});
        const Submission* pooledSubmission = &submission;
        if (DrawFinisher::Busy()) DrawFinisher::Get().Drain();
        DrawPrepPool::Get().Submit([this, pooledDraw, pooledSubmission](std::uint64_t seq) {
            auto& pool = DrawPrepPool::Get();
            auto& P = *pooledDraw;
            DrawPipeline::FollowEpoch(P.epoch);
            GuestMemory::SetCurrentPacket(P.commitPacket, P.queueId);
            const auto& graphics = P.relocatedDecode->state;
            const auto& roles = P.relocatedDecode->roles;
            static const QueueState noQueue{};
            std::array<double, DrawDriverPhaseCount> quietMs{};
            auto quietLap = std::chrono::steady_clock::time_point{};
            DrawPhaseTiming quiet{false, quietMs, quietLap};
            struct Work {
                ShaderRecompiler::ShaderPixelStageInfo pixel;
                Pm4::DrawParameters drawParameters;
                std::vector<ShaderRecompiler::MemoryRegion> memory;
                std::vector<std::shared_ptr<DispatchVariant>> matched;
                std::vector<std::vector<ShaderRecompiler::MemoryRegion>> matchedRegions;
                std::vector<ShaderRecompiler::RecompileResult> results;
                std::vector<std::shared_ptr<const ShaderRecompiler::RecompileResult>> compiledStages;
                std::vector<Graphics::CompiledShader> stages;
                std::vector<const ShaderRecompiler::RecompileResult*> programResults;
                std::vector<StageCapture> stageCaptures;
                std::vector<bool> recompiled;
                std::vector<bool> reused;
                std::vector<std::shared_ptr<DispatchVariant>> fresh;
                std::optional<ShaderMemory> shaderMemory;
                std::string rejected;
                std::uint64_t captures = 0;
            };
            auto work = std::make_unique<Work>();
            // The packet thread's lookup, here. The guest cannot reuse what the validation reads before
            // the label after this draw signals, and that label is a FIFO item enqueued only after this
            // job (Enqueue waits for the pool). What an earlier job writes is caught by the redo at the
            // turn, which repeats the lookup: its reads are the matched regions and the front reads.
            const auto lookup = [&] {
                const auto start = std::chrono::steady_clock::now();
                const auto count = P.programs.size();
                P.matched.assign(count, nullptr);
                P.matchedRegions.assign(count, {});
                P.stageData.assign(count, StageDataHit{});
                P.drawHit = false;
                P.verifyHit = false;
                P.entry = P.lookupEntry;
                P.decode = P.relocatedDecode;
                if (P.registerKey) {
                    P.vertexInfos.assign(count, std::nullopt);
                    P.decodeReads.assign(count, {});
                }
                P.candidates.assign(count, {});
                std::array<double, DrawDriverPhaseCount> lookupMs{};
                auto lookupLap = start;
                DrawPhaseTiming lookupTiming{profile, lookupMs, lookupLap};
                lookupDraw(*pooledSubmission, P.localDevice, graphics, P.pixel, P.programs, roles, P.vertexInfos, P.lookupEntries, P.registerKey, profile, P.drawKey, P.entry, P.matched, P.matchedRegions, P.drawHit, P.verifyHit, lookupTiming, lookupMs, P.dataEntry, P.candidates, P.stageData);
                for (std::size_t i = 0; i < P.matched.size(); ++i) {
                    if (P.matched[i] == nullptr || P.matched[i]->shaderDataPatches.empty()) continue;
                    applyShaderDataPatches(*P.matched[i]);
                    if (P.stageData[i].compiled != nullptr) P.stageData[i].compiled = patchDataWords(*P.matched[i], P.stageData[i].words);
                }
                if (!P.userPatches.empty() && !P.drawHit) {
                    auto current = std::make_shared<DrawDecode>(*P.decode);
                    for (const auto& patch : P.userPatches) current->programs[patch.program].userData[patch.index] = patch.value;
                    P.decode = std::move(current);
                }
                P.patchedStages = std::any_of(P.stageData.begin(), P.stageData.end(), [](const StageDataHit& hit) { return hit.compiled != nullptr; });
                if (P.registerKey) {
                    for (std::size_t i = 0; i < count; ++i) {
                        if (P.matched[i] != nullptr && P.drawHit) {
                            if (P.matched[i]->vertexInfo != nullptr) P.vertexInfos[i] = *P.matched[i]->vertexInfo;
                            continue;
                        }
                        if (!P.candidates[i].empty()) {
                            if (const auto& info = P.candidates[i].front().first->vertexInfo; info != nullptr) P.vertexInfos[i] = *info;
                            continue;
                        }
                        decodeDrawVertexInfo(P.programs[i], roles[i], P.vsOutControl, graphics, P.vertexInfos[i], P.decodeReads[i]);
                    }
                }
                pool.NoteLookup(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count()));
            };
            const auto reusableCandidate = [&](std::size_t i, std::uint32_t pushOffset) -> const std::pair<std::shared_ptr<DispatchVariant>, std::vector<ShaderRecompiler::MemoryRegion>>* {
                if (P.drawHit || i >= P.candidates.size()) return nullptr;
                const auto found = std::find_if(P.candidates[i].begin(), P.candidates[i].end(), [&](const auto& kept) { return kept.first->pushOffset == pushOffset; });
                return found == P.candidates[i].end() ? nullptr : &*found;
            };
            const auto attempt = [&](Work& w) {
                if (P.poolLookup) lookup();
                const auto count = P.programs.size();
                w.pixel = P.pixel;
                w.drawParameters = P.drawParameters;
                w.memory = P.memory;
                w.matched = P.matched;
                w.matchedRegions = P.matchedRegions;
                w.results.clear();
                w.results.reserve(count + 2u);
                w.compiledStages.clear();
                w.stages.clear();
                w.stages.reserve(count + 2u);
                w.programResults.assign(count, nullptr);
                w.stageCaptures.assign(count, StageCapture{});
                w.recompiled.assign(count, false);
                w.reused.assign(count, false);
                w.fresh.assign(count, nullptr);
                w.rejected.clear();
                w.shaderMemory.reset();
                w.shaderMemory.emplace(w.memory, &queryPendingWrite, &observePendingWrite, hookWaitCounter(), &QueryPendingSpan);
                std::uint32_t pushCursorBytes = 0;
                for (std::size_t i = 0; i < count; ++i) {
                    if (roles[i] == Role::GeometryBack) continue;
                    const auto& program = P.programs[i];
                    pushCursorBytes = Graphics::StagePushOffset(pushCursorBytes, program.binary.stage, P.localDevice->GraphicsPipelineLibraries());
                    const auto* candidate = reusableCandidate(i, pushCursorBytes);
                    if (P.drawHit) {
                        w.programResults[i] = P.stageData[i].compiled != nullptr ? P.stageData[i].compiled.get() : w.matched[i]->compiled.get();
                        w.memory.insert(w.memory.end(), w.matchedRegions[i].begin(), w.matchedRegions[i].end());
                    } else if (candidate != nullptr) {
                        w.matched[i] = candidate->first;
                        w.matchedRegions[i] = candidate->second;
                        w.reused[i] = true;
                        w.programResults[i] = P.stageData[i].compiled != nullptr ? P.stageData[i].compiled.get() : w.matched[i]->compiled.get();
                        w.memory.insert(w.memory.end(), w.matchedRegions[i].begin(), w.matchedRegions[i].end());
                    } else {
                        if (i < P.candidates.size() && !P.candidates[i].empty()) decodeDrawVertexInfo(P.programs[i], roles[i], P.vsOutControl, graphics, P.vertexInfos[i], P.decodeReads[i]);
                        if (program.binary.stage == Stage::Fragment) {
                            for (auto previous = i; previous-- > 0;) {
                                if (w.programResults[previous] == nullptr) continue;
                                w.pixel = Graphics::DefaultUnexportedInputs(P.decode->pixel, w.programResults[previous]->parameterExports);
                                break;
                            }
                        }
                        w.compiledStages.push_back(materializeDrawStage(i, pushCursorBytes, noQueue, *pooledSubmission, 0, 0, P.programs, graphics, w.pixel, P.vertexInfos, w.memory, P.linked, w.drawParameters, P.localDevice, *w.shaderMemory, w.stageCaptures, w.recompiled, P.drawHit, w.matched, w.matchedRegions, false, 0, 0, w.captures, quiet, quietMs, w.rejected, std::nullopt));
                        if (!w.rejected.empty()) return false;
                        w.programResults[i] = w.compiledStages.back().get();
                    }
                    const auto& result = *w.programResults[i];
                    if (i == 0) FoldDrawOffsets(result, P.programs.front(), w.drawParameters);
                    const auto slotEnd = (pushCursorBytes / Graphics::PipelinePushSlotBytes + 1u) * Graphics::PipelinePushSlotBytes;
                    require(result.pushConstants.size() <= slotEnd - pushCursorBytes, "stage push constants exceed the pipeline push constant block");
                    w.stages.push_back({program.binary.stage, &result, result.pushConstants.empty() ? 0u : pushCursorBytes});
                    pushCursorBytes += static_cast<std::uint32_t>(result.pushConstants.size());
                }
                return true;
            };
            const auto tryAttempt = [&] {
                try {
                    return attempt(*work);
                } catch (const std::exception& error) {
                    work->rejected = error.what();
                    return false;
                }
            };
            const auto startMark = pool.Enqueued();
            bool ok = tryAttempt();
            // A hit whose lookup ran here was submitted without a write bound, so it refines one like a miss.
            const bool refine = (P.poolLookup || !P.drawHit) && DrawPrepPool::RefinedBounds();
            const auto refineBound = [&](std::uint64_t from, std::span<const DrawPrepPool::Range> reads) {
                const auto ranges = drawWriteRanges(graphics, work->stages);
                const std::vector<DrawPrepPool::Range> bound(ranges.begin(), ranges.end());
                pool.Refine(seq, from, reads, bound);
            };
            const auto attemptReads = [&] {
                std::vector<DrawPrepPool::Range> reads;
                for (const auto& region : work->memory) reads.emplace_back(region.guestAddress, region.guestAddress + region.bytes.size());
                if (!P.poolLookup) return reads;
                for (const auto& stageReads : P.decodeReads) {
                    for (const auto& read : stageReads) reads.emplace_back(read.address, read.address + read.size);
                }
                return reads;
            };
            if (refine && ok && startMark < seq) refineBound(startMark, attemptReads());
            pool.WaitTurn(seq);
            if (startMark < seq) {
                if (!ok || pool.WrittenBetween(startMark, seq, attemptReads())) {
                    pool.NoteRedo();
                    ok = tryAttempt();
                }
                std::vector<DrawPrepPool::Range> frontReads;
                for (const auto& stageReads : P.decodeReads) {
                    for (const auto& read : stageReads) frontReads.emplace_back(read.address, read.address + read.size);
                }
                if (!frontReads.empty() && pool.WrittenBetween(startMark, seq, frontReads)) pool.NoteFrontReadConflict();
            }
            // The relocation trial reads these hits on the packet thread; counting them in turn keeps
            // them in packet order, a few draws later than the packet thread used to.
            if (P.poolLookup && !P.userPatches.empty()) {
                ProfiledLock cacheLock(drawCacheMutex, drawCacheProfile());
                if (P.drawHit || P.verifyHit) {
                    ++drawRelocationCounters.hits;
                    ++drawRelocationTrials[P.drawIdentity].hits;
                }
                if (profile && drawRelocationCounters.lookups >= 100000) reportDrawRelocation();
            }
            if (refine && ok) refineBound(seq, {});
            const auto skip = [&](const std::string& reason) {
                pool.NoteSkip();
                static std::atomic<std::uint64_t> reported{0};
                if (reported.fetch_add(1) < 20) std::fprintf(stderr, "[gpu] skipped draw (prepared on the pool): %s\n", reason.c_str());
                pool.FinishTurn(seq, {});
            };
            if (!ok) {
                skip(work->rejected);
                return;
            }
            auto& w = *work;
            if (P.poolLookup && P.dataEntry != nullptr && !P.drawHit) {
                std::uint64_t stagesReused = 0, stagesCompiled = 0;
                for (std::size_t i = 0; i < P.programs.size(); ++i) {
                    if (roles[i] != Role::GeometryBack) ++(w.reused[i] ? stagesReused : stagesCompiled);
                }
                std::lock_guard cacheLock(drawCacheMutex);
                drawEntryCounters.dataStagesReused += stagesReused;
                drawEntryCounters.dataStagesCompiled += stagesCompiled;
                if (stagesReused != 0 && stagesCompiled == 0) ++drawEntryCounters.dataDrawsReused;
            }
            std::optional<Graphics::IndirectDrawPath> noIndirect;
            cacheDrawStages(P.useDrawEntries, P.drawHit, w.drawParameters, noIndirect, P.programs, w.stageCaptures, P.vertexInfos, P.decodeReads, P.verifyHit, w.matched, P.stageData, w.fresh, P.drawKey, P.registerKey, P.decode, quiet, w.reused, P.entry, P.shapeKey);
            for (const auto& stageReads : P.decodeReads) {
                for (const auto& read : stageReads) w.memory.push_back({read.address, read.Bytes()});
            }
            if (graphics.rectList) {
                require(P.programs.size() == 2 && w.programResults[0] != nullptr && w.programResults[1] != nullptr && w.stages.size() == 2, "rect-list requires vertex and fragment programs");
                auto rectangle = DrawRectangle(*P.programs[0].snapshot, P.programs[1].snapshot, w.programResults[0]->variantId, w.programResults[1]->variantId, P.localDevice->Target());
                const auto rectIndex = w.results.size();
                w.results.push_back(std::move(rectangle.control));
                w.results.push_back(std::move(rectangle.evaluation));
                w.stages.insert(w.stages.begin() + 1, {{Stage::TessellationControl, &w.results[rectIndex], 0}, {Stage::TessellationEvaluation, &w.results[rectIndex + 1], 0}});
            }
            if (!P.programs.empty() && w.programResults.back() != nullptr && w.programResults.back()->barycentricEmulation.active) {
                require(!graphics.rectList && graphics.stages.path == Graphics::ShaderPath::Vertex && P.programs.size() == 2 && w.programResults[0] != nullptr && w.stages.size() == 2, "a pixel shader that reads barycentrics without VK_KHR_fragment_shader_barycentric needs a vertex shader before it");
                w.results.push_back(ShaderRecompiler::BuildBarycentricGeometryShader(*w.programResults[0], *w.programResults[1], P.localDevice->Target(), P.localDevice->GeometryLimits()));
                w.stages.insert(w.stages.begin() + 1, Graphics::CompiledShader{Stage::Geometry, &w.results.back(), 0});
            }
            std::vector<Graphics::GuestMemorySnapshot> snapshots;
            for (const auto& region : w.memory) snapshots.push_back({region.guestAddress, region.bytes});
            std::vector<std::shared_ptr<DispatchVariant>> recipeStages;
            if (P.registerKey && !P.patchedStages && Graphics::DrawRecipes()) {
                for (std::size_t i = 0; i < P.programs.size(); ++i) recipeStages.push_back(P.drawHit ? w.matched[i] : w.fresh[i]);
                if (std::all_of(recipeStages.begin(), recipeStages.end(), [](const std::shared_ptr<DispatchVariant>& variant) { return variant == nullptr; })) recipeStages.clear();
            }
            std::shared_ptr<const DrawRecipe> recipe;
            if (P.drawHit && !recipeStages.empty()) {
                recipe = findDrawRecipe(P.drawKey, recipeStages);
                if (recipe == nullptr) VulkanDevice::NoteDrawRecipeMiss(VulkanDevice::DrawRecipePrecheck::NoRecipe);
            }
            if (recipe == nullptr) {
                if (auto known = P.localDevice->KnownDrawRejection(graphics, w.stages)) {
                    skip(*known);
                    return;
                }
            }
            auto writes = drawWriteRanges(graphics, w.stages);
            const std::vector<DrawPrepPool::Range> written(writes.begin(), writes.end());
            auto reads = QueuedDrawReadRanges(work->drawParameters, work->stages);
            auto commit = [this, pooledDraw, work = std::move(work), snapshots = std::move(snapshots), recipe = std::move(recipe), recipeStages = std::move(recipeStages)] {
                DrawPipeline::FollowEpoch(pooledDraw->epoch);
                GuestMemory::SetCurrentPacket(pooledDraw->commitPacket, pooledDraw->queueId);
                commitDraw(pooledDraw->localDevice, pooledDraw->queueId, pooledDraw->decode->state, work->drawParameters, work->stages, snapshots, recipe, recipeStages, pooledDraw->drawKey);
            };
            auto held = std::make_shared<decltype(commit)>(std::move(commit));
            DrawPipeline::Queue0().Enqueue(DrawPipeline::CommitKind::Draw, [held] { (*held)(); }, std::move(writes), 0, {}, std::move(reads), true);
            pool.FinishTurn(seq, written);
        }, std::move(writeBound));
        phaseTiming.Phase(DrawRowLockWait);
        timing.Mark("draw_prepared_on_pool");
        return drawn();
    }

    tail.submission = &submission;
    tail.submissionQueue = submission.queue;
    tail.packetOpcode = (packet[0] >> 8u) & 0xffu;
    tail.reg207 = readRegister(queue.context, 0x207);
    tail.dumpTarget = dumpTarget;
    tail.dumpSlot1 = dumpSlot1;
    tail.packetOffset = packetOffset;
    const bool handOff = pipelined && !drawParameters.indirect && !profile && !lockedPrepare && dumpTarget == 0 && dumpSlot1 == 0 && !tracePrograms && !verifyDrawDataHits() && deferredLabels().labels.empty() && DrawFinisher::Enabled();
    if (handOff) {
        tail.epoch = DrawPipeline::EpochToken().load(std::memory_order_relaxed);
        tail.submissionSerial = submission.serial;
        tail.packetOffset = static_cast<std::size_t>(packet.data() - submission.commands.data());
        noteFinisherHandOff(true);
        DrawFinisher::Pending held;
        if (DrawFinisher::Overlap()) held = heldWrites(tail);
        DrawFinisher::Get().Enqueue([this, tailHolder] {
            auto& job = *tailHolder;
            DrawPipeline::FollowEpoch(job.epoch);
            GuestMemory::SetCurrentPacket(job.packetOpcode, job.submissionQueue);
            PerformanceTimer finishTiming("Driver.DrawFinish");
            try {
                if (finishDraw(job, finishTiming) == DrawVerdict::Rejected) throw std::runtime_error(job.ownRejected);
            } catch (const std::exception& error) {
                CaptureTrace::Log("draw-error submission=%llu offset=%zu finisher=1 reason=%.256s", static_cast<unsigned long long>(job.submissionSerial), job.packetOffset, error.what());
                throw std::runtime_error("draw at submission " + std::to_string(job.submissionSerial) + " offset " + std::to_string(job.packetOffset) + " (finished on the draw finisher): " + error.what());
            }
        }, std::move(held));
        timing.Mark("draw_handed_off");
        return drawn();
    }
    noteFinisherHandOff(false);
    if (DrawFinisher::Busy()) {
        DrawFinisher::Get().Drain();
        if (DrawFinisher::Overlap()) DrawFinisher::NoteDrain(DrawFinisher::DrainSite::Kept, true);
    }
    tail.queue = &queue;
    tail.packet = packet;
    tail.rejected = &rejected;
    tail.gpuLock = &gpuLock;
    return finishDraw(tail, timing);
}

DrawVerdict Driver::finishDraw(DrawTail& tail, PerformanceTimer& timing) {
    using Stage = ShaderRecompiler::ShaderStage;
    using Role = ShaderRecompiler::ProgramRole;
    const bool profile = tail.profile;
    const auto traceToken = tail.traceToken;
    const auto& trace = tail.trace;
    const auto* tracePtr = trace.token != 0 ? &trace : nullptr;
    const auto packetOffset = tail.packetOffset;
    const auto packet = tail.packet;
    auto& phaseMs = tail.phaseMs;
    auto& phaseLap = tail.phaseLap;
    auto& captures = tail.captures;
    auto& phaseTiming = tail.phaseTiming;
    auto& drawParameters = tail.drawParameters;
    const bool pipelined = tail.pipelined;
    static const bool pipelinedIndirect = std::getenv("APS5_NO_PIPELINED_INDIRECT") == nullptr;
    const auto drainBeforeInline = [&] {
        if (pipelined && DrawPipeline::Queue0().Busy()) DrawPipeline::Queue0().Drain(DrawPipeline::DrainReason::Indirect);
    };
    auto& traceIndirect = tail.traceIndirect;
    const auto dumpTarget = tail.dumpTarget;
    const auto dumpSlot1 = tail.dumpSlot1;
    auto& gpuLock = tail.gpuLock != nullptr ? *tail.gpuLock : tail.ownLock;
    auto& localDevice = tail.localDevice;
    const bool registerKey = tail.registerKey;
    auto& drawKey = tail.drawKey;
    auto& entry = tail.entry;
    auto& decode = tail.decode;
    auto& dataEntry = tail.dataEntry;
    auto& shapeKey = tail.shapeKey;
    const auto& graphics = decode->state;
    auto& pixel = tail.pixel;
    auto& writablePrograms = tail.writablePrograms;
    const auto& programs = *tail.programs;
    const std::vector<Role>& roles = decode->roles;
    auto& memory = tail.memory;
    auto& linked = tail.linked;
    auto& vertexInfos = tail.vertexInfos;
    auto& decodeReads = tail.decodeReads;
    auto& shaderMemory = tail.shaderMemory;
    auto& results = tail.results;
    auto& compiledStages = tail.compiledStages;
    auto& stages = tail.stages;
    auto& pushCursorBytes = tail.pushCursorBytes;
    auto& programResults = tail.programResults;
    auto& stageCaptures = tail.stageCaptures;
    auto& matched = tail.matched;
    auto& matchedRegions = tail.matchedRegions;
    auto& fresh = tail.fresh;
    auto& recompiled = tail.recompiled;
    auto& dataCandidates = tail.dataCandidates;
    auto& reused = tail.reused;
    auto& stageData = tail.stageData;
    std::uint64_t dataStagesReused = 0, dataStagesCompiled = 0, dataVerified = 0;
    auto& drawHit = tail.drawHit;
    auto& verifyHit = tail.verifyHit;
    const bool stageEntries = tail.stageEntries;
    const bool patchedStages = tail.patchedStages;
    std::uint64_t stageDataVerified = 0;
    const Submission& submission = *tail.submission;
    std::string& rejected = tail.rejected != nullptr ? *tail.rejected : tail.ownRejected;
    static const QueueState noQueue{};
    const QueueState& queue = tail.queue != nullptr ? *tail.queue : noQueue;
    const auto drawn = [&] {
        phaseTiming.Phase(DrawRowVectors);
        if (tail.frameSteps) FrameSteps::NoteDrawSteps(FrameSteps::Global(), phaseMs);
        if (!profile) return DrawVerdict::Drawn;
        auto& pending = pendingDrawPhases();
        pending.phases = true;
        pending.captures = captures;
        pending.ms = phaseMs;
        pending.tailAt = phaseLap;
        return DrawVerdict::Drawn;
    };
    const auto setMeshIndexBuffer = [&](const Pm4::DrawParameters& parameters) {
        if (!graphics.stages.mesh) return;
        auto& words = writablePrograms.front().userData;
        require(programs.front().firstUserSgpr == 0 && words.size() >= ShaderRecompiler::MeshIndexBufferUserWord + 4, "mesh program lacks the hidden user words");
        const auto descriptor = Graphics::MeshIndexBufferDescriptor(parameters);
        std::copy(descriptor.begin(), descriptor.end(), words.begin() + ShaderRecompiler::MeshIndexBufferUserWord);
    };
    const auto locate = [&](std::uint32_t location) -> std::optional<std::pair<std::size_t, std::size_t>> {
        if (location == 0x280u) return std::nullopt;
        for (std::size_t i = 0; i < programs.size(); ++i) {
            if (roles[i] == Role::Fragment || roles[i] == Role::GeometryBack || location < programs[i].userDataBase) continue;
            const auto word = location - programs[i].userDataBase + (8u - programs[i].firstUserSgpr);
            if (word < programs[i].userData.size()) return std::make_pair(i, static_cast<std::size_t>(word));
        }
        return std::nullopt;
    };
    const auto decodeVertexInfo = [&](std::size_t i) {
        const auto& program = programs[i];
        if (program.binary.stage == Stage::Fragment || roles[i] == Role::GeometryBack) return;
        decodeReads[i].clear();
        vertexInfos[i] = Graphics::DecodeVertexStageInfo(program.binary.header, program.binary.headerAddress, program.userData, &decodeReads[i], false, 8u - program.firstUserSgpr);
        if (vertexInfos[i]) vertexInfos[i]->paClVsOutCntl = PositionExportLayout(program.snapshot.get(), program.codeOffset, program.binary.code, tail.reg207, program.tailContinuation.get()) | graphics.layerExports;
    };
    const auto fold = [&](const ShaderRecompiler::RecompileResult& main, Pm4::DrawParameters& parameters) { FoldDrawOffsets(main, programs.front(), parameters); };

    const auto freshRuns = [&](std::size_t i, std::vector<std::pair<std::uint64_t, std::uint64_t>>& runs, std::vector<std::uint32_t>& words) {
        std::vector<ShaderRecompiler::MemoryRegion> regions(stageCaptures[i].regions.begin(), stageCaptures[i].regions.end());
        for (const auto& read : decodeReads[i]) regions.push_back({read.address, read.Bytes()});
        std::stable_sort(regions.begin(), regions.end(), [](const ShaderRecompiler::MemoryRegion& a, const ShaderRecompiler::MemoryRegion& b) { return a.guestAddress < b.guestAddress; });
        for (const auto& region : regions) {
            runs.emplace_back(region.guestAddress, region.guestAddress + region.bytes.size());
            const auto count = region.bytes.size() / sizeof(std::uint32_t);
            const auto offset = words.size();
            words.resize(offset + count);
            std::memcpy(words.data() + offset, region.bytes.data(), count * sizeof(std::uint32_t));
        }
    };
    const auto sameBindings = [](const ShaderRecompiler::RecompileResult& a, const ShaderRecompiler::RecompileResult& b) {
        return a.bindings.size() == b.bindings.size() && std::equal(a.bindings.begin(), a.bindings.end(), b.bindings.begin(), [](const auto& x, const auto& y) { return x.kind == y.kind && x.role == y.role && x.binding == y.binding && x.count == y.count && x.guestDescriptor == y.guestDescriptor; });
    };
    const auto flatten = [](const std::vector<std::pair<std::uint64_t, std::uint64_t>>& spans, const std::vector<std::uint32_t>& values) {
        std::vector<std::pair<std::uint64_t, std::uint32_t>> flat;
        std::size_t word = 0;
        for (const auto& [begin, end] : spans) {
            for (auto address = begin; address < end && word < values.size(); address += sizeof(std::uint32_t)) flat.emplace_back(address, values[word++]);
        }
        std::sort(flat.begin(), flat.end());
        return flat;
    };
    const auto verifyDataStage = [&](std::size_t i, const DispatchVariant& kept) {
        std::vector<std::pair<std::uint64_t, std::uint64_t>> runs;
        std::vector<std::uint32_t> words;
        freshRuns(i, runs, words);
        const auto& compiled = *stageCaptures[i].compiled;
        const char* what = flatten(runs, words) != flatten(kept.runs, kept.words) ? "runs or words" : vertexInfos[i].has_value() != (kept.vertexInfo != nullptr) || (vertexInfos[i] && !sameVertexInfo(*vertexInfos[i], *kept.vertexInfo)) ? "vertex stage info" : compiled.variantId != kept.compiled->variantId ? "variant" : compiled.pushConstants != kept.compiled->pushConstants ? "push constants" : !sameBindings(compiled, *kept.compiled) ? "bindings" : nullptr;
        ++dataVerified;
        if (what == nullptr) return;
        char message[352];
        std::snprintf(message, sizeof(message), "APS5_VERIFY_DRAW_DATA_HITS/APS5_VERIFY_DRAW_REBASE: stage %zu (program 0x%llx) of a data-only hit differs from its full build in its %s (%zu runs / %zu words kept, %zu / %zu fresh; variant 0x%llx / 0x%llx; key 0x%llx, shape 0x%llx)", i, static_cast<unsigned long long>(programs[i].binary.codeAddress), what, kept.runs.size(), kept.words.size(), runs.size(), words.size(), static_cast<unsigned long long>(kept.compiled->variantId), static_cast<unsigned long long>(compiled.variantId), static_cast<unsigned long long>(drawKey), static_cast<unsigned long long>(shapeKey));
        throw std::runtime_error(message);
    };
    const auto verifyStageData = [&](std::size_t i) {
        const auto& hit = stageData[i];
        const auto& kept = *matched[i];
        std::vector<std::pair<std::uint64_t, std::uint64_t>> runs;
        std::vector<std::uint32_t> words;
        freshRuns(i, runs, words);
        const auto& fresh = *stageCaptures[i].compiled;
        const auto& patched = *hit.compiled;
        const char* what = flatten(runs, words) != flatten(kept.runs, hit.words) ? "runs or words" : fresh.variantId != patched.variantId ? "variant" : fresh.pushConstants != patched.pushConstants ? "push constants" : !sameBindings(fresh, patched) ? "bindings" : nullptr;
        ++stageDataVerified;
        if (what == nullptr) return;
        char message[256];
        std::snprintf(message, sizeof(message), "APS5_VERIFY_DRAW_DATA_HITS: stage %zu (program 0x%llx) of a stage data hit differs from its full build in its %s (%zu data positions refreshed)", i, static_cast<unsigned long long>(programs[i].binary.codeAddress), what, kept.dataPositions.size());
        throw std::runtime_error(message);
    };
    std::optional<Graphics::IndirectDrawPath> indirectCpu;
    std::size_t completedSubdraws = 0;
    std::vector<std::uint32_t> pushOffsets(programs.size(), 0);
    std::optional<ShaderRecompiler::MeshBarycentricLayout> meshBarycentrics;
    std::vector<std::size_t> resultIndex(programs.size(), 0);
    for (std::size_t i = 0; i < programs.size(); ++i) {
        if (roles[i] == Role::GeometryBack) continue;
        const auto& program = programs[i];
        pushCursorBytes = Graphics::StagePushOffset(pushCursorBytes, program.binary.stage, localDevice->GraphicsPipelineLibraries());
        pushOffsets[i] = pushCursorBytes;
        if (drawHit) {

            programResults[i] = stageData[i].compiled != nullptr ? stageData[i].compiled.get() : matched[i]->compiled.get();
            memory.insert(memory.end(), matchedRegions[i].begin(), matchedRegions[i].end());
        } else {
            const auto candidate = std::find_if(dataCandidates[i].begin(), dataCandidates[i].end(), [&](const auto& kept) { return kept.first->pushOffset == pushCursorBytes; });
            if (candidate != dataCandidates[i].end() && !verifyDataCandidates()) {
                matched[i] = candidate->first;
                matchedRegions[i] = std::move(candidate->second);
                reused[i] = true;
                programResults[i] = stageData[i].compiled != nullptr ? stageData[i].compiled.get() : matched[i]->compiled.get();
                memory.insert(memory.end(), matchedRegions[i].begin(), matchedRegions[i].end());
            } else {
                if (!dataCandidates[i].empty() && !verifyDataCandidates()) decodeVertexInfo(i);
                if (program.binary.stage == ShaderRecompiler::ShaderStage::Fragment) {
                    for (auto previous = i; previous-- > 0;) {
                        if (programResults[previous] == nullptr) continue;
                        pixel = Graphics::DefaultUnexportedInputs(decode->pixel, programResults[previous]->parameterExports);
                        break;
                    }
                }
                compiledStages.push_back(materializeDrawStage(i, pushCursorBytes, queue, submission, packetOffset, trace.token, programs, graphics, pixel, vertexInfos, memory, linked, drawParameters, localDevice, *shaderMemory, stageCaptures, recompiled, drawHit, matched, matchedRegions, profile, dumpTarget, dumpSlot1, captures, phaseTiming, phaseMs, rejected, meshBarycentrics));
                if (!rejected.empty()) return DrawVerdict::Rejected;
                programResults[i] = compiledStages.back().get();
                if (candidate != dataCandidates[i].end()) verifyDataStage(i, *candidate->first);
                if (stageData[i].compiled != nullptr && matched[i] != nullptr && verifyDrawDataHits()) verifyStageData(i);
            }
            if (dataEntry != nullptr && reused[i]) ++dataStagesReused;
            else if (dataEntry != nullptr) ++dataStagesCompiled;
        }
        const auto& result = *programResults[i];
        if (i == 0 && drawParameters.indirect) {
            indirectCpu = ClassifyIndirectDraw(result, graphics, programs.front(), localDevice, drawParameters, traceIndirect);
            if (EmptyIndirectSkip()) CpuIndirectPrograms()[programs.front().binary.codeAddress] = indirectCpu.has_value();
        } else if (i == 0) {
            fold(result, drawParameters);
        }
        try {
            const auto slotEnd = (pushCursorBytes / Graphics::PipelinePushSlotBytes + 1u) * Graphics::PipelinePushSlotBytes;
            require(result.pushConstants.size() <= slotEnd - pushCursorBytes, "stage push constants exceed the pipeline push constant block");
            stages.push_back({program.binary.stage, &result, result.pushConstants.empty() ? 0u : pushCursorBytes});
            pushCursorBytes += static_cast<std::uint32_t>(result.pushConstants.size());
        } catch (...) {
            TraceMaterializationFailure(trace.token, submission, packetOffset, "push-constant-validation", completedSubdraws, std::current_exception());
            throw;
        }
    }

    if (stageDataVerified != 0) {
        ProfiledLock cacheLock(drawCacheMutex, drawCacheProfile());
        drawEntryCounters.stageDataVerified += stageDataVerified;
    }
    if (dataEntry != nullptr) {
        ProfiledLock cacheLock(drawCacheMutex, drawCacheProfile());
        drawEntryCounters.dataStagesReused += dataStagesReused;
        drawEntryCounters.dataStagesCompiled += dataStagesCompiled;
        if (dataStagesReused != 0 && dataStagesCompiled == 0) ++drawEntryCounters.dataDrawsReused;
        drawEntryCounters.dataVerified += dataVerified;
    }
    const bool shapeRefresh = Graphics::DrawPipelineEnabled();
    const auto refreshSeenTwice = [&] {
        std::uint64_t data = shapeKey;
        bool stored = false;
        const auto mix = [&data](std::uint64_t value) {
            data ^= value + 0x9e3779b97f4a7c15ull + (data << 6u) + (data >> 2u);
        };
        for (std::size_t i = 0; i < programs.size(); ++i) {
            const auto& capture = stageCaptures[i];
            if (reused[i] || capture.compiled == nullptr || !CacheableResult(*capture.compiled)) continue;
            stored = true;
            const auto& words = programs[i].userData;
            for (std::size_t w = 0; w < words.size(); ++w) {
                if (w >= 64u || (capture.userDataMask >> w & 1u) != 0) mix((static_cast<std::uint64_t>(i) << 40u) ^ (static_cast<std::uint64_t>(w) << 32u) ^ words[w]);
            }
        }
        if (!stored) return false;
        static std::array<std::atomic<std::uint64_t>, 4096> lastData{};
        return lastData[shapeKey % lastData.size()].exchange(data, std::memory_order_relaxed) == data;
    };
    const bool shapeNeedsEntry = shapeRefresh && registerKey && shapeKey != 0 && drawDataHits() && !drawParameters.indirect && dataEntry != nullptr && dataStagesCompiled != 0 && refreshSeenTwice();
    cacheDrawStages(stageEntries, drawHit, drawParameters, indirectCpu, programs, stageCaptures, vertexInfos, decodeReads, verifyHit, matched, stageData, fresh, drawKey, registerKey, decode, phaseTiming, reused, entry, shapeKey, shapeNeedsEntry);
    timing.Mark("shader_compile_and_link");

    for (const auto& reads : decodeReads) {
        for (const auto& read : reads) memory.push_back({read.address, read.Bytes()});
    }

    if (tail.queue != nullptr && pipelined && !deferredLabels().labels.empty()) DrawPipeline::Queue0().Drain(DrawPipeline::DrainReason::Labels);
    if (recordQueuedLabelsAfterCapture(submission.queue, memory)) return draw(*tail.queue, packet, submission, packetOffset, rejected);

    bool rectListBuilt = false;

    std::size_t rectIndex = 0;
    const auto buildRectList = [&] {
        try {
            phaseTiming.Phase(DrawRowVectors);
            require(programs.size() == 2 && programResults[0] != nullptr && programResults[1] != nullptr, "rect-list requires vertex and fragment programs");
            auto rectangle = DrawRectangle(*programs[0].snapshot, programs[1].snapshot, programResults[0]->variantId, programResults[1]->variantId, localDevice->Target());
            auto& control = graphics.provokingVertexMode == VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT ? rectangle.lastVertexControl : rectangle.control;
            if (rectListBuilt) {
                results[rectIndex] = std::move(control);
                results[rectIndex + 1] = std::move(rectangle.evaluation);
                phaseTiming.Phase(DrawRowRectList);
                return;
            }
            require(stages.size() == 2, "rect-list requires vertex and fragment programs");
            results.reserve(programs.size() + 2u);
            rectIndex = results.size();
            results.push_back(std::move(control));
            results.push_back(std::move(rectangle.evaluation));
            stages.insert(stages.begin() + 1, {{Stage::TessellationControl, &results[rectIndex], 0}, {Stage::TessellationEvaluation, &results[rectIndex + 1], 0}});
            rectListBuilt = true;
            phaseTiming.Phase(DrawRowRectList);
        } catch (...) {
            TraceMaterializationFailure(trace.token, submission, packetOffset, "rect-list", completedSubdraws, std::current_exception());
            throw;
        }
    };
    if (graphics.rectList) buildRectList();
    if (!programs.empty() && programResults.back() != nullptr && programResults.back()->barycentricEmulation.active && graphics.stages.path == Graphics::ShaderPath::Geometry) {
        require(localDevice->Target().mesh.has_value(), "a pixel shader on the mesh path reads barycentrics, and this device has neither VK_KHR_fragment_shader_barycentric nor mesh shaders (VK_EXT_mesh_shader), so they cannot be provided or emulated");
        require(!graphics.rectList && graphics.stages.mesh && programs.size() == 2 && programResults[0] != nullptr && programResults[1] != nullptr && stages.size() == 2, "a pixel shader that reads barycentrics needs exactly one mesh program before it to emulate them");
        meshBarycentrics = ShaderRecompiler::MeshBarycentricLayoutOf(*programResults[1]);
        if (programResults[0]->meshBarycentrics != meshBarycentrics) {
            const auto pushBytes = programResults[0]->pushConstants.size();
            const auto exports = programResults[0]->parameterExports;
            compiledStages.push_back(materializeDrawStage(0, pushOffsets[0], queue, submission, packetOffset, traceToken, programs, graphics, pixel, vertexInfos, memory, linked, drawParameters, localDevice, *shaderMemory, stageCaptures, recompiled, drawHit, matched, matchedRegions, profile, dumpTarget, dumpSlot1, captures, phaseTiming, phaseMs, rejected, meshBarycentrics));
            if (!rejected.empty()) return DrawVerdict::Rejected;
            programResults[0] = compiledStages.back().get();
            stages[0].program = programResults[0];
            require(programResults[0]->pushConstants.size() == pushBytes && programResults[0]->parameterExports == exports, "pairing the mesh program with its pixel shader changed the push constants or the parameter exports the pixel shader was built for");
        }
        require(programResults[0]->meshBarycentrics == meshBarycentrics, "the mesh program was compiled for a different barycentric layout than its pixel shader");
    } else if (!graphics.rectList && !programs.empty() && programResults.back() != nullptr && programResults.back()->barycentricEmulation.active) {
        require(graphics.stages.path == Graphics::ShaderPath::Vertex && programs.size() == 2 && programResults[0] != nullptr && stages.size() == 2, "a pixel shader that reads barycentrics without VK_KHR_fragment_shader_barycentric needs a vertex shader before it");
        const auto geometryLimits = localDevice->GeometryLimits();
        results.reserve(programs.size() + 2u);
        if (geometryLimits || !localDevice->Target().tessellation) {
            results.push_back(ShaderRecompiler::BuildBarycentricGeometryShader(*programResults[0], *programResults[1], localDevice->Target(), geometryLimits));
            stages.insert(stages.begin() + 1, Graphics::CompiledShader{Stage::Geometry, &results.back(), 0});
        } else {
            require((graphics.topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST || graphics.topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP) && !graphics.primitiveRestart, "a pixel shader that reads barycentrics without VK_KHR_fragment_shader_barycentric or geometry shaders needs a triangle list or strip without primitive restart");
            auto patches = ShaderRecompiler::BuildBarycentricTessellationShaders(*programResults[0], *programResults[1], localDevice->Target(), localDevice->MaxFragmentInputComponents(), graphics.provokingVertexMode == VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT ? 2u : 0u);
            const auto first = results.size();
            results.push_back(std::move(patches.control));
            results.push_back(std::move(patches.evaluation));
            stages.insert(stages.begin() + 1, {{Stage::TessellationControl, &results[first], 0}, {Stage::TessellationEvaluation, &results[first + 1], 0}});
        }
    }
    std::vector<Graphics::GuestMemorySnapshot> snapshots;
    const auto snapshot = [&] {
        snapshots.clear();
        snapshots.reserve(memory.size());
        for (const auto& region : memory) snapshots.push_back({region.guestAddress, region.bytes});
    };
    snapshot();
    timing.Mark("post_compile_prepare");
    const auto lockForDraw = [&] {
        require(tail.queue != nullptr, "the draw finisher reached a draw it cannot commit through the pipeline");
        if (gpuLock.owns_lock()) return;
        phaseTiming.Phase(DrawRowVectors);
        GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Draw);
        gpuLock.lock();
        timing.Mark("gpu_mutex_wait");
        phaseTiming.Phase(DrawRowLockWait);

        if (auto current = device.Load(); current != nullptr && current != localDevice) {
            static std::atomic<std::uint64_t> replaced{0};
            std::fprintf(stderr, "[draw] device replaced during unlocked preparation (%llu)\n", static_cast<unsigned long long>(++replaced));
            localDevice = std::move(current);
        }

        recordLabelsForPacket(localDevice.get(), submission.queue);
        phaseTiming.Phase(DrawRowLabels);
    };
    if (drawParameters.indirect && indirectCpu) {

        const auto indirect = *drawParameters.indirect;
        results.reserve(programs.size() + 2u);
        for (std::size_t i = 0; i < programs.size(); ++i) {
            if (programResults[i] == nullptr) continue;
            resultIndex[i] = results.size();
            results.push_back(ShaderRecompiler::RecompileResult(*programResults[i]));
            for (auto& stage : stages) {
                if (stage.program == programResults[i]) stage.program = &results[resultIndex[i]];
            }
            programResults[i] = &results[resultIndex[i]];
        }
        recordQueuedLabelsBeforeRead(submission.queue);
        const auto readStart = std::chrono::steady_clock::now();
        const auto count = std::min(indirect.countIndirect ? Pm4::ReadDrawCount(indirect) : indirect.count, indirect.count);
        std::vector<Pm4::DrawArguments> records;
        for (std::uint32_t record = 0; record < count; ++record) records.push_back(Pm4::ReadDrawArguments(indirect, record));
        Graphics::CountIndirectDraw(*indirectCpu, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - readStart).count());
        const auto baseVertexWords = UserDataWords(programs, roles, indirect.baseVertexLocation);
        const auto startInstanceWords = UserDataWords(programs, roles, indirect.startInstanceLocation);
        const auto drawIndexWords = indirect.drawIndexEnabled ? UserDataWords(programs, roles, indirect.drawIndexLocation) : std::vector<std::pair<std::size_t, std::size_t>>{};
        const auto drawable = [&](const Pm4::DrawArguments& arguments) { return arguments.count != 0 && arguments.instances != 0 && (!drawParameters.indexed || !indirect.indexLimitKnown || arguments.firstVertexOrIndex < drawParameters.indexCount); };
        std::size_t lastDrawable = records.size();
        for (std::size_t record = 0; record < records.size(); ++record) {
            if (drawable(records[record])) lastDrawable = record;
        }
        for (std::uint32_t record = 0; record < records.size(); ++record) {
            const auto& arguments = records[record];
            if (traceIndirect) std::fprintf(stderr, "[draw]   record %u: count %u instances %u first %u vertexOffset %u startInstance %u\n", record, arguments.count, arguments.instances, arguments.firstVertexOrIndex, arguments.vertexOffset, arguments.firstInstance);
            if (arguments.count == 0 || arguments.instances == 0) continue;
            std::set<std::size_t> patched;
            const auto patch = [&](const std::vector<std::pair<std::size_t, std::size_t>>& words, std::uint32_t value) {
                for (const auto& [program, word] : words) {
                    writablePrograms[program].userData[word] = value;
                    patched.insert(program);
                }
            };
            patch(baseVertexWords, indirect.recordBytes == 20 ? arguments.vertexOffset : arguments.firstVertexOrIndex);
            patch(startInstanceWords, arguments.firstInstance);
            patch(drawIndexWords, record);
            if (graphics.stages.tessellation) {
                for (std::size_t i = 0; i < programs.size(); ++i) {
                    if (programs[i].binary.stage == Stage::TessellationControl || programs[i].binary.stage == Stage::TessellationEvaluation) patched.insert(i);
                }
            }
            Pm4::DrawParameters direct{0, arguments.count, 0, arguments.instances, drawParameters.flags, drawParameters.indexed, 0, 0};
            if (drawParameters.indexed) {
                if (indirect.indexLimitKnown) {
                    if (arguments.firstVertexOrIndex >= drawParameters.indexCount) continue;
                    direct.indexCount = std::min(arguments.count, drawParameters.indexCount - arguments.firstVertexOrIndex);
                } else {
                    const auto end = static_cast<std::uint64_t>(arguments.firstVertexOrIndex) + arguments.count;
                    require(end <= std::numeric_limits<std::uint32_t>::max(), "indirect draw index range overflow");
                    require(end * drawParameters.indexSize <= std::numeric_limits<std::uint64_t>::max() - drawParameters.indexAddress, "indirect draw index range address overflow");
                    direct.indexCount = arguments.count;
                }
                direct.indexAddress = drawParameters.indexAddress + static_cast<std::uint64_t>(arguments.firstVertexOrIndex) * drawParameters.indexSize;
                direct.indexSize = drawParameters.indexSize;
            } else {
                direct.firstVertex = indirect.indxOffset;
            }
            if (graphics.stages.mesh) {
                setMeshIndexBuffer(direct);
                patched.insert(0);
            }
            for (const auto programIndex : patched) {
                auto& result = results[resultIndex[programIndex]];
                const auto pushBytes = result.pushConstants.size();
                decodeVertexInfo(programIndex);
                result = *materializeDrawStage(programIndex, pushOffsets[programIndex], queue, submission, packetOffset, trace.token, programs, graphics, pixel, vertexInfos, memory, linked, programs[programIndex].binary.stage == Stage::TessellationControl || programs[programIndex].binary.stage == Stage::TessellationEvaluation ? direct : drawParameters, localDevice, *shaderMemory, stageCaptures, recompiled, drawHit, matched, matchedRegions, profile, dumpTarget, dumpSlot1, captures, phaseTiming, phaseMs, rejected, meshBarycentrics);
                if (!rejected.empty()) return DrawVerdict::Rejected;
                require(result.pushConstants.size() == pushBytes, "patched program changed its push constant layout");
            }
            fold(*programResults[0], direct);
            if (graphics.rectList && patched.contains(0)) buildRectList();
            snapshot();
            if (auto known = localDevice->KnownDrawRejection(graphics, stages)) {
                rejected = std::move(*known);
                return DrawVerdict::Rejected;
            }
            if (pipelined && pipelinedIndirect && record == lastDrawable && !gpuLock.owns_lock() && deferredLabels().labels.empty()) {
                phaseTiming.Phase(DrawRowVectors);
                auto writes = drawWriteRanges(graphics, stages);
                const auto commitQueue = submission.queue;
                const auto commitPacket = tail.packetOpcode;
                const auto epoch = DrawPipeline::EpochToken().load(std::memory_order_relaxed);
                auto commit = [this, localDevice = std::move(localDevice), decode = std::move(decode), direct, stages = std::move(stages), snapshots = std::move(snapshots), commitQueue, commitPacket, epoch, writablePrograms = std::move(writablePrograms), results = std::move(results), memory = std::move(memory), linked = std::move(linked), stageCaptures = std::move(stageCaptures), matched = std::move(matched), matchedRegions = std::move(matchedRegions), fresh = std::move(fresh), decodeReads = std::move(decodeReads), shaderMemory = std::move(shaderMemory), entry = std::move(entry), dataEntry = std::move(dataEntry), compiledStages = std::move(compiledStages), stageData = std::move(stageData)] {
                    DrawPipeline::FollowEpoch(epoch);
                    GuestMemory::SetCurrentPacket(commitPacket, commitQueue);
                    commitDraw(localDevice, commitQueue, decode->state, direct, stages, snapshots, nullptr, {}, 0);
                };
                auto held = std::make_shared<decltype(commit)>(std::move(commit));
                DrawPipeline::Queue0().Enqueue([held] { (*held)(); }, std::move(writes));
                phaseTiming.Phase(DrawRowLockWait);
                timing.Mark("draw_enqueued");
                return drawn();
            }
            drainBeforeInline();
            lockForDraw();
            noteDrawWriters(stages, submission.queue);
            phaseTiming.Phase(DrawRowVectors);
            localDevice->Draw(graphics, direct, stages, snapshots, nullptr, tracePtr);
            phaseTiming.Phase(DrawRowGraphics);
        }
        timing.Mark("draw_and_resource_release");
        return drawn();
    }

    std::vector<std::shared_ptr<DispatchVariant>> recipeStages;
    if (registerKey && !drawParameters.indirect && !patchedStages && Graphics::DrawRecipes()) {
        recipeStages.reserve(programs.size());
        for (std::size_t i = 0; i < programs.size(); ++i) recipeStages.push_back(drawHit ? matched[i] : fresh[i]);
        if (std::all_of(recipeStages.begin(), recipeStages.end(), [](const std::shared_ptr<DispatchVariant>& variant) { return variant == nullptr; })) recipeStages.clear();
    }
    std::shared_ptr<const DrawRecipe> recipe;
    if (drawHit && !recipeStages.empty()) {
        recipe = findDrawRecipe(drawKey, recipeStages);
        if (recipe == nullptr) VulkanDevice::NoteDrawRecipeMiss(VulkanDevice::DrawRecipePrecheck::NoRecipe);
    }
    if (recipe == nullptr) {
        if (auto known = localDevice->KnownDrawRejection(graphics, stages)) {
            rejected = std::move(*known);
            return DrawVerdict::Rejected;
        }
    }
    if (pipelined && !drawParameters.indirect && (tail.queue == nullptr || deferredLabels().labels.empty())) {
        phaseTiming.Phase(DrawRowVectors);
        auto writes = drawWriteRanges(graphics, stages);
        const auto commitQueue = submission.queue;
        const auto commitPacket = tail.packetOpcode;
        const auto epoch = DrawFinisher::OnFinisherThread() ? tail.epoch : DrawPipeline::EpochToken().load(std::memory_order_relaxed);
        auto reads = QueuedDrawReadRanges(drawParameters, stages);
        auto commit = [this, localDevice = std::move(localDevice), decode = std::move(decode), drawParameters, stages = std::move(stages), snapshots = std::move(snapshots), recipe = std::move(recipe), recipeStages = std::move(recipeStages), drawKey, commitQueue, commitPacket, epoch, trace = trace, writablePrograms = std::move(writablePrograms), results = std::move(results), memory = std::move(memory), linked = std::move(linked), stageCaptures = std::move(stageCaptures), matched = std::move(matched), matchedRegions = std::move(matchedRegions), fresh = std::move(fresh), decodeReads = std::move(decodeReads), shaderMemory = std::move(shaderMemory), entry = std::move(entry), dataEntry = std::move(dataEntry), compiledStages = std::move(compiledStages), stageData = std::move(stageData)]() mutable {
            DrawPipeline::FollowEpoch(epoch);
            GuestMemory::SetCurrentPacket(commitPacket, commitQueue);
            commitDraw(localDevice, commitQueue, decode->state, drawParameters, stages, snapshots, recipe, recipeStages, drawKey, trace);
        };
        auto held = std::make_shared<decltype(commit)>(std::move(commit));
        DrawPipeline::Queue0().Enqueue(DrawPipeline::CommitKind::Draw, [held] { (*held)(); }, std::move(writes), 0, {}, std::move(reads), true);
        phaseTiming.Phase(DrawRowLockWait);
        timing.Mark("draw_enqueued");
        return drawn();
    }
    if (drawParameters.indirect) drainBeforeInline();
    lockForDraw();
    noteDrawWriters(stages, submission.queue);
    phaseTiming.Phase(DrawRowVectors);
    if (recipe != nullptr) {
        if (localDevice->DrawFromRecipe(graphics, drawParameters, stages, snapshots, recipe, tracePtr) == RecipeOutcome::Recorded) {
            phaseTiming.Phase(DrawRowGraphics);
            timing.Mark("draw_and_resource_release");
            return drawn();
        }

        VulkanDevice::NoteRecipe(VulkanDevice::RecipeEvent::Restart, VulkanDevice::RecipeKind::Draw);
    }
    std::shared_ptr<const DrawRecipe> built;
    localDevice->Draw(graphics, drawParameters, stages, snapshots, recipeStages.empty() ? nullptr : &built, tracePtr);
    phaseTiming.Phase(DrawRowGraphics);
    if (built != nullptr) attachDrawRecipe(drawKey, recipeStages, std::move(built));
    timing.Mark("draw_and_resource_release");
    return drawn();
}

void Driver::commitDraw(std::shared_ptr<VulkanDevice> localDevice, std::uint32_t queue, const Graphics::State& graphics, const Pm4::DrawParameters& drawParameters, std::span<const Graphics::CompiledShader> stages, std::span<const Graphics::GuestMemorySnapshot> snapshots, const std::shared_ptr<const DrawRecipe>& recipe, const std::vector<std::shared_ptr<DispatchVariant>>& recipeStages, std::uint64_t drawKey, VulkanDevice::DrawTraceContext trace) {
    const auto* tracePtr = trace.token != 0 ? &trace : nullptr;
    GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Draw);
    std::lock_guard gpuLock(GuestMemory::GpuMutex());
    if (auto current = device.Load(); current != nullptr && current != localDevice) localDevice = std::move(current);
    struct SubmitDue {
        Driver& driver;
        const std::shared_ptr<VulkanDevice>& device;
        ~SubmitDue() noexcept(false) {
            if (std::uncaught_exceptions() == 0) driver.submitDueAfterCommit(device.get());
        }
    } submitDue{*this, localDevice};
    recordLabelsForPacket(localDevice.get(), queue);
    noteDrawWriters(stages, queue);
    if (recipe != nullptr) {
        if (localDevice->DrawFromRecipe(graphics, drawParameters, stages, snapshots, recipe, tracePtr) == RecipeOutcome::Recorded) return;
        VulkanDevice::NoteRecipe(VulkanDevice::RecipeEvent::Restart, VulkanDevice::RecipeKind::Draw);
    }
    std::shared_ptr<const DrawRecipe> built;
    localDevice->Draw(graphics, drawParameters, stages, snapshots, recipeStages.empty() ? nullptr : &built, tracePtr);
    if (built != nullptr) attachDrawRecipe(drawKey, recipeStages, std::move(built));
}

}
