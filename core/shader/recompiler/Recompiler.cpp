#include "Recompiler.hpp"
#include "StencilReplay.hpp"
#include <algorithm>
#include <array>
#include <cstring>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include "CacheKey.hpp"
#include "CompiledVariant.hpp"
#include "VertexInputSpecialization.hpp"
#include "FragmentOutputSpecialization.hpp"
#include "SpirvBackend/SpirvSpecialization.hpp"
#include "RdnaDecoder/RdnaDescriptorFormat.hpp"
#include "ShaderDiskCache.hpp"
#include <list>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <mutex>
#include <new>
#include <shared_mutex>
#include <unordered_map>
#include "ControlFlow/include/ControlFlow/GraphBuilder.hpp"
#include "ControlFlow/include/ControlFlow/Structurizer.hpp"
#include "RdnaDecoder/include/RdnaDecoder/RdnaInstructionDecoder.hpp"
#include "IntermediateRepresentation/include/IntermediateRepresentation/IrProgram.hpp"
#include "Optimization/include/Optimization/BindingAllocator.hpp"
#include "Optimization/include/Optimization/ConstantFolder.hpp"
#include "Optimization/include/Optimization/DeadCodeEliminator.hpp"
#include "Optimization/include/Optimization/DenormalFlushEliminator.hpp"
#include "Optimization/include/Optimization/DescriptorBindingBuilder.hpp"
#include "Optimization/include/Optimization/HostInterpolationChecker.hpp"
#include "Optimization/include/Optimization/MaskedSelectEliminator.hpp"
#include "Optimization/include/Optimization/ReadLaneEliminator.hpp"
#include "Optimization/include/Optimization/RedundantValueEliminator.hpp"
#include "Optimization/include/Optimization/RequestMemoryView.hpp"
#include "Optimization/include/Optimization/ResourceMaterializer.hpp"
#include "Optimization/ResourceProgram.hpp"
#include "Optimization/include/Optimization/ResourceTracker.hpp"
#include "Optimization/include/Optimization/VectorWindowStoreEliminator.hpp"
#include "Optimization/include/Optimization/ShaderInfoCollector.hpp"
#include "Optimization/include/Optimization/SrtWalker.hpp"
#include "Optimization/include/Optimization/SsaBuilder.hpp"
#include "Optimization/include/Optimization/TessellationLowering.hpp"
#include "SpirvBackend/include/SpirvBackend/SpirvEmitter.hpp"
#if ANYPS5_ENABLE_SPIRV_TOOLS
#include "SpirvBackend/SpirvOptimizer.hpp"
#endif
#include "SpirvBackend/SpirvMemory/SpirvInputOutput.hpp"
#include "Translation/include/Translation/InstructionTranslator.hpp"
#include "Translation/include/Translation/ShaderInputInfoBuilder.hpp"
#include <exception>
#include <stdexcept>
#include <string>
#include <ControlFlow/RequestSerializer.hpp>

namespace ShaderRecompiler {

namespace {

ShaderStageKind toShaderStageKind(ShaderStage stage) {
    switch (stage) {
    case ShaderStage::Compute:
        return ShaderStageKind::Compute;
    case ShaderStage::Vertex:
        return ShaderStageKind::Vertex;
    case ShaderStage::TessellationControl:
        return ShaderStageKind::TessellationControl;
    case ShaderStage::TessellationEvaluation:
        return ShaderStageKind::TessellationEvaluation;
    case ShaderStage::Fragment:
        return ShaderStageKind::Pixel;
    case ShaderStage::Local:
        return ShaderStageKind::Local;
    case ShaderStage::Mesh:
        return ShaderStageKind::Mesh;
    case ShaderStage::Geometry:
        break;
    }
    throw std::runtime_error("ShaderRecompiler::Recompile: unsupported shader stage");
}

}

namespace {

bool MergedHullStage(const RecompileRequest& request) {
    const auto stage = request.shader.stage;
    return (stage == ShaderStage::Local || stage == ShaderStage::TessellationControl) && request.graphics && request.graphics->tessellation && request.graphics->tessellation->mergedHull;
}

// The host subgroup width wave64 programs are laid out for. Debug aid: APS5_SINGLE_LANE=<hex code
// addresses, comma separated, or "all"> keeps the listed programs at one guest lane per invocation.
std::uint32_t HostSubgroupSize(const RecompileRequest& request) {
    const bool vertexStage = request.shader.stage == ShaderStage::Vertex || request.shader.stage == ShaderStage::Local;
    if (vertexStage && !request.target.vertexSubgroups) return 1u;
    if (request.shader.stage == ShaderStage::TessellationControl && MergedHullStage(request)) return 1u;
    static const std::string list = [] { const char* text = std::getenv("APS5_SINGLE_LANE"); return text ? std::string(text) : std::string(); }();
    if (!list.empty()) {
        if (list == "all") return 64u;
        char address[32];
        std::snprintf(address, sizeof(address), "%llx", static_cast<unsigned long long>(request.shader.codeAddress));
        if (list.find(address) != std::string::npos) return 64u;
    }
    return request.target.subgroupSize;
}

RdnaProgram DecodeStageProgram(ShaderStageKind stage, std::span<const std::uint32_t> code, bool mergedHull) {
    static constexpr std::array<std::uint32_t, 1> EndProgram{0xbf810000u};
    if (stage == ShaderStageKind::Local && mergedHull) {
        RdnaProgram program;
        program.code = EndProgram;
        program.instructions.push_back(DecodeRdnaInstruction(0u, EndProgram, 0u));
        return program;
    }
    if (stage != ShaderStageKind::Local) return RdnaInstructionDecoder{}.Decode(code);
    auto program = DecodeRdnaFrontProgram(code);
    program.instructions.back() = DecodeRdnaInstruction(program.instructions.back().programCounter, EndProgram, 0u);
    return program;
}

ShaderStageInputInfo RequestInputInfo(const RecompileRequest& request) {
    const auto* mesh = request.graphics && request.graphics->mesh ? &*request.graphics->mesh : nullptr;
    std::optional<TessellationConfiguration> tessellationCopy;
    if (request.graphics && request.graphics->tessellation) {
        tessellationCopy = *request.graphics->tessellation;
        tessellationCopy->instancePatches = TessellationInstancePatches(request);
        const auto& capabilities = request.target.supportedCapabilities;
        tessellationCopy->drawParameters = std::find(capabilities.begin(), capabilities.end(), static_cast<std::uint32_t>(spv::CapabilityDrawParameters)) != capabilities.end();
    }
    const auto* tessellation = tessellationCopy ? &*tessellationCopy : nullptr;
    return BuildShaderStageInputInfo(toShaderStageKind(request.shader.stage), request.context, HostSubgroupSize(request), mesh, tessellation);
}

void ValidateRequestInputs(const RecompileRequest& request) {
    const auto stage = toShaderStageKind(request.shader.stage);
    const bool vertexFamily = stage == ShaderStageKind::Vertex || stage == ShaderStageKind::Local || stage == ShaderStageKind::TessellationControl || stage == ShaderStageKind::TessellationEvaluation || stage == ShaderStageKind::Mesh;
    if (!vertexFamily) {
        static_cast<void>(RequestInputInfo(request));
        return;
    }
    const auto* mesh = request.graphics && request.graphics->mesh ? &*request.graphics->mesh : nullptr;
    const auto* tessellation = request.graphics && request.graphics->tessellation ? &*request.graphics->tessellation : nullptr;
    ValidateVertexStageInputs(stage, request.context, mesh, tessellation);
}

std::uint32_t DeviceMemoryLdsBytes(const RecompileRequest& request) {
    if (!request.context.compute.has_value()) return 0u;
    const auto bytes = static_cast<std::uint64_t>(request.context.compute->ldsSizeDwords) * 4u;
    return bytes + 4u > request.target.maxWorkgroupSharedMemoryBytes ? static_cast<std::uint32_t>(bytes) : 0u;
}

bool HardwareFloat32Flush(const RecompileRequest& request) {
    return request.target.float32FlushToZero && (!request.context.floatMode.has_value() || ((request.context.floatMode->floatMode >> 4u) & 3u) == 0u);
}

std::atomic<std::uint64_t> frontEndBuilds{0};

}

struct PreparedControlFlow {
    PreparedControlFlow(const RecompileRequest& request, const SwappcInfo& swappcInfo)
        : stage(request.shader.stage), mergedHull(MergedHullStage(request)), fallbacks(swappcInfo.linkedCallFallbacks.begin(), swappcInfo.linkedCallFallbacks.end()), swappc(WithFallbacks(swappcInfo, fallbacks)), code(request.shader.code.begin(), request.shader.code.end()),
          decoded(DecodeStageProgram(toShaderStageKind(request.shader.stage), code, mergedHull)), cfg(GraphBuilder{}.Build(decoded, &swappc)) {
        Structurizer{}.Structurize(cfg);
    }

    PreparedControlFlow(const PreparedControlFlow&) = delete;
    PreparedControlFlow& operator=(const PreparedControlFlow&) = delete;

    bool Matches(const RecompileRequest& request, const SwappcInfo& swappcInfo) const {
        return stage == request.shader.stage && mergedHull == MergedHullStage(request) && swappc.fetchCallAllowed == swappcInfo.fetchCallAllowed &&
               swappc.userDataBaseRegister == swappcInfo.userDataBaseRegister && swappc.userDataCount == swappcInfo.userDataCount && swappc.waveSize == swappcInfo.waveSize &&
               std::ranges::equal(fallbacks, swappcInfo.linkedCallFallbacks) && std::ranges::equal(code, request.shader.code);
    }

    static SwappcInfo WithFallbacks(SwappcInfo info, const std::vector<std::uint32_t>& fallbacks) {
        info.linkedCallFallbacks = fallbacks;
        return info;
    }

    ShaderStage stage;
    bool mergedHull;
    std::vector<std::uint32_t> fallbacks;
    SwappcInfo swappc;
    std::vector<std::uint32_t> code;
    RdnaProgram decoded;
    ControlFlowGraph cfg;
};

std::shared_ptr<const PreparedControlFlow> ShaderPreparationContext::AcquireFrontend(const RecompileRequest& request) {
    const auto inputInfo = RequestInputInfo(request);
    const SwappcInfo swappc{inputInfo.vertex != nullptr, request.context.userDataBaseRegister, static_cast<std::uint32_t>(request.context.userData.size()), request.context.waveSize, request.context.linkedCallFallbacks};
    static const bool reuse = std::getenv("APS5_NO_PERF_FRONTEND_PAIR") == nullptr;
    if (reuse && frontend != nullptr && frontend->Matches(request, swappc)) {
        return frontend;
    }
    auto prepared = std::make_shared<const PreparedControlFlow>(request, swappc);
    if (reuse) {
        frontend = prepared;
    }
    return prepared;
}

std::uint64_t FrontEndBuilds() {
    return frontEndBuilds.load(std::memory_order_relaxed);
}

std::uint32_t ShaderPreparationContext::ResourcePrograms() const {
    return resourcePrograms;
}

IrProgram PrepareResourceProgram(const RecompileRequest& request) {
    return PrepareResourceProgram(request, nullptr);
}

IrProgram PrepareResourceProgram(const RecompileRequest& request, ShaderPreparationContext* preparation) {
    frontEndBuilds.fetch_add(1, std::memory_order_relaxed);
    ShaderPreparationContext local;
    auto& context = preparation != nullptr ? *preparation : local;
    ++context.resourcePrograms;
    context.interfaceSource.reset();
    context.interfaceProgram.reset();
    const auto frontend = context.AcquireFrontend(request);
    const auto stageKind = toShaderStageKind(request.shader.stage);
    const auto inputInfo = RequestInputInfo(request);
    const auto& decoded = frontend->decoded;
    const auto& cfg = frontend->cfg;

    TranslateOptions translateOptions {};
    translateOptions.stage = stageKind;
    translateOptions.shaderHash = request.shader.codeAddress;
    translateOptions.waveSize = request.context.waveSize;
    translateOptions.userDataBaseRegister = request.context.userDataBaseRegister;
    translateOptions.gcnLegacyVertexInputs = GcnGuestShaders() && request.context.userDataBaseRegister == 0u;
    translateOptions.userDataCount = static_cast<std::uint32_t>(request.context.userData.size());
    translateOptions.scratchDwords = request.context.compute.has_value() ? request.context.compute->scratchDwords : 0u;
    translateOptions.sharedMemoryBytes = DeviceMemoryLdsBytes(request);
    translateOptions.fragmentShaderBarycentricEnabled = request.target.fragmentShaderBarycentricEnabled;
    const auto supports = [&](spv::Capability capability) { return std::ranges::find(request.target.supportedCapabilities, static_cast<std::uint32_t>(capability)) != request.target.supportedCapabilities.end(); };
    translateOptions.nativeFloat16 = supports(spv::CapabilityFloat16) && supports(spv::CapabilityRoundingModeRTE) && supports(spv::CapabilityDenormPreserve);
    translateOptions.interpolationFunctions = supports(spv::CapabilityInterpolationFunction);
    translateOptions.floatMode = request.context.floatMode;
    translateOptions.float32FlushToZero = HardwareFloat32Flush(request);
    translateOptions.inputInfo = inputInfo;
    translateOptions.float32DenormPreserve = request.target.float32DenormPreserve;

    constexpr InstructionTranslator translator;

    if (frontend->mergedHull && cfg.hasFetchCall) throw std::runtime_error("a merged hull program that calls a fetch subroutine is unsupported");
    EmbeddedFetchPlan embeddedFetch;
    if ((stageKind == ShaderStageKind::Vertex || stageKind == ShaderStageKind::Local) && !frontend->mergedHull && inputInfo.vertex != nullptr && inputInfo.vertex->fetchEmbedded) {
        embeddedFetch = EmbeddedVertexFetchAnalyzer{}.Analyze(decoded, inputInfo.vertex->fetchAttribReg, inputInfo.vertex->fetchBufferReg, request.context.userDataBaseRegister, static_cast<std::uint32_t>(request.context.userData.size()), request.context.waveSize);
        translateOptions.embeddedFetch = &embeddedFetch;
    }
    auto program = translator.Translate(decoded, cfg, translateOptions);
    // Debug aid: APS5_DUMP_IR=<hex code address> (or "all") prints the program after each front-end pass.
    const auto dumpIr = [&](const char* pass) {
        static const std::string list = [] { const char* text = std::getenv("APS5_DUMP_IR"); return text ? std::string(text) : std::string(); }();
        if (list.empty()) return;
        char address[32];
        std::snprintf(address, sizeof(address), "%llx", static_cast<unsigned long long>(request.shader.codeAddress));
        if (list != "all" && list.find(address) == std::string::npos) return;
        std::fprintf(stderr, "==== IR 0x%s after %s\n%s\n", address, pass, ProgramToString(program).c_str());
    };

    constexpr SsaBuilder ssaBuilder;
    constexpr ConstantFolder constantFolder;
    constexpr DeadCodeEliminator deadCodeEliminator;
    constexpr ReadLaneEliminator readLaneEliminator;
    constexpr MaskedSelectEliminator maskedSelectEliminator;
    constexpr VectorWindowStoreEliminator vectorWindowStoreEliminator;
    const auto simplify = [&] {
        dumpIr("translate");
        ssaBuilder.Rewrite(program);
        dumpIr("ssa");

        constantFolder.Fold(program);
        ResolveControlFlowIdentities(program);
        deadCodeEliminator.RemoveIdentities(program);
        deadCodeEliminator.Eliminate(program);
        dumpIr("fold");

        const auto readLaneStats = readLaneEliminator.Eliminate(program, translateOptions.waveSize);
        if (readLaneStats.rewrittenReads != 0u) {
            constantFolder.Fold(program);
            ResolveControlFlowIdentities(program);
            deadCodeEliminator.RemoveIdentities(program);
            deadCodeEliminator.Eliminate(program);
        }

        if (vectorWindowStoreEliminator.Eliminate(program).removedStores != 0u) {
            deadCodeEliminator.Eliminate(program);
        }

        if (maskedSelectEliminator.Eliminate(program).removedSelects != 0u) {
            const bool liveAncillary = std::ranges::any_of(program.Blocks(), [](const auto& block) {
                return std::ranges::any_of(block->Instructions(), [](const IrValue* value) {
                    return value->Opcode() == IrOpcode::GetBuiltin && value->HasUses() && value->Argument(0)->Resolve()->ImmediateU32() == static_cast<std::uint32_t>(StageInputKind::PackedAncillary);
                });
            });
            if (liveAncillary) {
                constantFolder.Fold(program);
                ResolveControlFlowIdentities(program);
                deadCodeEliminator.RemoveIdentities(program);
            }
            deadCodeEliminator.Eliminate(program);
        }
        if (constantFolder.ComposePackedAncillary(program)) {
            deadCodeEliminator.Eliminate(program);
        }
    };
    simplify();
    if (stageKind == ShaderStageKind::Pixel && !translateOptions.fragmentShaderBarycentricEnabled && !HostInterpolationChecker{}.Lower(program, *inputInfo.pixel)) {
        translateOptions.fragmentShaderBarycentricEnabled = true;
        program = translator.Translate(decoded, cfg, translateOptions);
        program.Metadata().barycentricEmulation = true;
        simplify();
    }

    constexpr DenormalFlushEliminator denormalFlushEliminator;
    if (denormalFlushEliminator.Eliminate(program).removedFlushes != 0u) {
        deadCodeEliminator.Eliminate(program);
    }

    if (inputInfo.vertex != nullptr) {
        constexpr TessellationLowering tessellationLowering;
        tessellationLowering.Lower(program, inputInfo.vertex->tess, request.target.tessellation ? &*request.target.tessellation : nullptr);
    }

    deadCodeEliminator.EliminateUnobservedPhis(program);
    deadCodeEliminator.Eliminate(program);
    constexpr SrtWalker srtWalker;
    srtWalker.BuildPlan(program);
    deadCodeEliminator.Eliminate(program);
    dumpIr("srt");

    constexpr ResourceTracker resourceTracker;
    resourceTracker.Track(program);
    deadCodeEliminator.Eliminate(program);
    dumpIr("resources");
    program.Resources().srgbDecodeFormats = request.target.srgbDecodeFormats;
    if (request.context.convertedSamples) {
        for (auto& image : program.Resources().info.images) image.convertedSamples = true;
    }

    return program;
}

// A materialized result of one variant over one snapshot (Recompile(request, capture)): the
// shared immutable object every later capture that reproduces the snapshot receives, so Populate
// and the per-request copy run once per distinct snapshot.
struct ResultMemoEntry {
    std::uint64_t variantId = 0;
    std::uint64_t hash = 0;
    std::shared_ptr<const RecompileResult> result;
    std::uint64_t used = 0;
    std::uint64_t shape = 0;
};

struct EmissionFailure {
    std::uint64_t codeAddress;
    BindingLayout layout;
    std::exception_ptr failure;
};

constexpr std::size_t ResultMemoSets = 64;
constexpr std::size_t ResultMemoWays = 4;
using ResultMemoTable = std::array<ResultMemoEntry, ResultMemoSets * ResultMemoWays>;
using SnapshotParts = std::array<std::uint64_t, static_cast<std::size_t>(ResultMemoMissComponent::Count)>;

struct SourceEntry : std::enable_shared_from_this<SourceEntry> {
    std::mutex mutex;
    // The code the entry was built for: the key carries only a hash of it, so a candidate entry is
    // accepted only when its code matches word for word. Owned here because the request's span
    // points into a registration the driver may replace while the entry lives on.
    std::vector<std::uint32_t> code;
    std::vector<std::uint32_t> linkedCallFallbacks;
    std::shared_ptr<const IrResourcePlan> plan;
    bool runtimeCaptured = false;
    // A plan build that threw (an unsupported resource chain or control flow) is remembered and
    // rethrown: the front end ran every pass before failing, ~13 ms per dispatch of a shader the
    // title issues every frame (0x1048947300 at the intro video). APS5_NO_FAILURE_MEMO=1 rebuilds.
    std::exception_ptr planFailure;
    std::unique_ptr<IrProgram> program;
    std::vector<std::shared_ptr<const CompiledVariant>> variants;
    std::vector<EmissionFailure> emissionFailures;
    std::unique_ptr<ResultMemoTable> memo;
    std::uint64_t memoClock = 0;
    // The newest memo entry of each forwarded-user-word shape, a donor for patching (under mutex).
    std::unordered_map<std::uint64_t, ResultMemoEntry*> memoShapes;
    std::unordered_map<std::uint64_t, SnapshotParts> memoNewestParts;
};

namespace {

class PrimedSharedMutex : public std::shared_mutex {
public:
    PrimedSharedMutex() {
        lock();
        unlock();
    }
};

struct ResourceProgram {
    explicit ResourceProgram(const RecompileRequest& request, ShaderPreparationContext* preparation = nullptr, bool reclaim = true) : program(std::make_unique<IrProgram>(PrepareResourceProgram(request, preparation))) {
        const auto clearRegisterState = [](const auto& storage) {
            for (const auto& block : storage) {
                block->ssaScalarValues.fill(nullptr);
                block->ssaVectorValues.fill(nullptr);
                block->ssaThreadBitScalarValues.fill(nullptr);
                block->ssaScalarMaskTags.fill(nullptr);
            }
        };
        clearRegisterState(program->Blocks());
        clearRegisterState(program->Resources().blockStorage);
        if (reclaim) program->ReclaimUnusedValues();
        plan = std::make_shared<const IrResourcePlan>(ResourceMaterializer{}.ExtractPlan(*program));
    }

    std::unique_ptr<IrProgram> program;
    std::shared_ptr<const IrResourcePlan> plan;
};

std::shared_ptr<const IrResourcePlan> makeResourcePlan(const RecompileRequest& request) {
    return ResourceProgram(request, nullptr, false).plan;
}

struct SourceKeyHash {
    std::size_t operator()(const std::vector<std::uint64_t>& key) const {
        std::size_t hash = 0;
        for (const auto value : key) {
            hash ^= static_cast<std::size_t>(value) + static_cast<std::size_t>(0x9e3779b97f4a7c15ull) + (hash << 6u) + (hash >> 2u);
            if constexpr (sizeof(std::size_t) < sizeof(value)) hash ^= static_cast<std::size_t>(value >> 32u);
        }
        return hash;
    }
};

bool FailureMemo() {
    static const bool memo = std::getenv("APS5_NO_FAILURE_MEMO") == nullptr;
    return memo;
}

bool diskPlans() {
    static const bool enabled = [] {
        const char* dumpIr = std::getenv("APS5_DUMP_IR");
        return std::getenv("APS5_NO_PERF_DISK_PLANS") == nullptr && (dumpIr == nullptr || *dumpIr == '\0');
    }();
    return enabled && ShaderDiskCache::Enabled() && !DebugProbeActive();
}

bool sameLayout(const BindingLayout& left, const BindingLayout& right);

std::shared_ptr<SourceEntry> getSource(const RecompileRequest& request, ShaderPreparationContext* preparation = nullptr, bool reclaim = true, const SourceHandle* preparedHandle = nullptr) {
    static PrimedSharedMutex mutex;
    // Entries whose code hashes alike share a bucket; the code comparison picks the right one.
    static std::unordered_map<std::vector<std::uint64_t>, std::vector<std::shared_ptr<SourceEntry>>, SourceKeyHash> sources;
    struct SourceKeyStorage {};
    auto& key = HostThreadLocal<std::vector<std::uint64_t>, SourceKeyStorage>();
    RecompileCacheKey::Build(request, key);
    key.push_back(HostSubgroupSize(request));
    const auto find = [&]() -> std::shared_ptr<SourceEntry> {
        const auto found = sources.find(key);
        if (found == sources.end()) return nullptr;
        for (const auto& entry : found->second) {
            if (std::equal(entry->code.begin(), entry->code.end(), request.shader.code.begin(), request.shader.code.end())) return entry;
        }
        return nullptr;
    };
    std::shared_ptr<SourceEntry> source;
    {
        std::shared_lock lock(mutex);
        source = find();
    }
    if (source == nullptr) {
        std::unique_lock lock(mutex);
        source = find();
        if (source == nullptr) {
            source = std::make_shared<SourceEntry>();
            source->code.assign(request.shader.code.begin(), request.shader.code.end());
            source->linkedCallFallbacks.assign(request.context.linkedCallFallbacks.begin(), request.context.linkedCallFallbacks.end());
            auto& bucket = sources[key];
            if (!bucket.empty()) {
                // A second entry under one key is a code hash collision (or the unhashed key with
                // identical code, which cannot happen); each is reported under the profile switch.
                static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
                static std::uint64_t collisions = 0;
                ++collisions;
                if (profile) std::fprintf(stderr, "[recompile] source key collision %llu: %zu entries share a key (%zu code words)\n", static_cast<unsigned long long>(collisions), bucket.size() + 1, request.shader.code.size());
            }
            bucket.push_back(source);
        }
    }
    {
        std::lock_guard lock(source->mutex);
        if (source->plan == nullptr) {
            const bool hasPreparedArtifact = preparedHandle != nullptr && preparedHandle->artifact != nullptr;
            if (hasPreparedArtifact && (preparedHandle->source != source || !MatchesPreparedShader(request, *preparedHandle))) throw std::runtime_error("ShaderRecompiler: prepared artifact does not match the resource plan source");
            if (FailureMemo() && source->planFailure) std::rethrow_exception(source->planFailure);
            const bool disk = diskPlans();
            std::vector<std::byte> planKey;
            if (disk) {
                ShaderDiskCache::BuildPlanKey(request, HostSubgroupSize(request), planKey);
                source->plan = ShaderDiskCache::LoadSharedPlan(planKey, request.shader.codeAddress);
            }
            if (source->plan == nullptr) {
                try {
                    ResourceProgram resource(request, preparation, reclaim);
                    source->plan = std::move(resource.plan);
                    if (!hasPreparedArtifact && std::ranges::none_of(source->variants, [&](const auto& variant) { return sameLayout(variant->layout, request.layout); })) {
                        source->program = std::move(resource.program);
                    }
                } catch (...) {
                    if (FailureMemo()) source->planFailure = std::current_exception();
                    throw;
                }
                if (disk) ShaderDiskCache::StorePlan(std::move(planKey), source->plan);
            }
        }
    }
    return source;
}

std::shared_ptr<const IrResourcePlan> acquirePlan(const RecompileRequest& request, const std::shared_ptr<SourceEntry>& source, bool runtimeCapture, const SourceHandle* preparedHandle = nullptr) {
    for (;;) {
        {
            std::lock_guard lock(source->mutex);
            if (source->plan != nullptr) {
                if (runtimeCapture) source->runtimeCaptured = true;
                return source->plan;
            }
        }
        if (getSource(request, nullptr, true, preparedHandle) != source) throw std::runtime_error("ShaderRecompiler: resource plan source changed while reloading");
    }
}

void releaseUndrawnPlan(const SourceHandle& handle) {
    if (handle.source == nullptr || !diskPlans()) return;
    std::lock_guard lock(handle.source->mutex);
    if (!handle.source->runtimeCaptured) handle.source->plan.reset();
}

std::array<std::uint32_t, 3> partialThreads(const RecompileRequest& request) {
    return request.context.compute ? request.context.compute->partialThreads : std::array<std::uint32_t, 3>{};
}

std::uint64_t nextVariantId() {
    static std::atomic<std::uint64_t> variants{0};
    return variants.fetch_add(1, std::memory_order_relaxed) + 1;
}

template<typename TStep>
auto recordEmissionFailure(std::exception_ptr* emissionFailure, TStep&& step) -> decltype(step()) {
    try {
        return step();
    } catch (const std::bad_alloc&) {
        throw;
    } catch (...) {
        if (emissionFailure != nullptr) *emissionFailure = std::current_exception();
        throw;
    }
}

void prepareInterface(const RecompileRequest& request, IrProgram& program) {
    const auto inputInfo = RequestInputInfo(request);
    constexpr DeadCodeEliminator deadCodeEliminator;
    constexpr ResourceMaterializer resourceMaterializer;
    const bool nativeSampleOffsets = request.target.nonConstantImageOffsets && std::find(request.target.supportedCapabilities.begin(), request.target.supportedCapabilities.end(), spv::CapabilityImageGatherExtended) != request.target.supportedCapabilities.end();
    resourceMaterializer.ApplyStaticInterface(program, nativeSampleOffsets);

    constexpr ConstantFolder constantFolder;
    if (request.shader.stage == ShaderStage::Vertex && constantFolder.FoldVertexLaneBounds(program, HostSubgroupSize(request), request.target.spirvVersion)) {
        constantFolder.Fold(program);
        ResolveControlFlowIdentities(program);
    }

    deadCodeEliminator.RemoveIdentities(program);
    constexpr RedundantValueEliminator redundantValueEliminator;
    static_cast<void>(redundantValueEliminator.Eliminate(program));
    deadCodeEliminator.Eliminate(program);

    constexpr ShaderInfoCollector shaderInfoCollector;
    shaderInfoCollector.Collect(program, inputInfo);
}

CompiledVariant emitVariant(const RecompileRequest& request, IrProgram& program) {
    const auto inputInfo = RequestInputInfo(request);
    constexpr BindingAllocator bindingAllocator;
    bool registration = request.target.registrationModules && (request.shader.stage == ShaderStage::Vertex || request.shader.stage == ShaderStage::Fragment);
    program.Metadata().bindingLayoutComplete = false;
    const bool stencilReplay = request.shader.stage == ShaderStage::Fragment && std::ranges::any_of(program.Info().outputs, [](const StageOutput& output) { return output.kind == StageOutputKind::StencilRef; }) && std::ranges::find(request.target.supportedCapabilities, static_cast<std::uint32_t>(spv::CapabilityStencilExportEXT)) == request.target.supportedCapabilities.end();
    auto bindings = bindingAllocator.Allocate(program, request.layout, registration, stencilReplay);

    SpirvTargetOptions targetOptions {};
    targetOptions.vulkanVersion = request.target.vulkanVersion;
    targetOptions.spirvVersion = request.target.spirvVersion;
    targetOptions.subgroupSize = HostSubgroupSize(request) == 1u ? 1u : request.target.subgroupSize;
    targetOptions.bdaAbiVersion = request.target.bdaAbiVersion;
    targetOptions.supportedCapabilities = request.target.supportedCapabilities;
    targetOptions.supportedExtensions = request.target.supportedExtensions;
    targetOptions.nonConstantImageOffsets = request.target.nonConstantImageOffsets;
    targetOptions.narrowSubgroupClock = request.target.narrowSubgroupClock;
    targetOptions.float64DenormPreserve = request.target.float64DenormPreserve;
    targetOptions.depthRangeRemap = request.target.depthRangeRemap;
    targetOptions.registrationModules = registration;
    targetOptions.fragmentDescriptorSet = request.target.registrationModules;
    targetOptions.float16DenormPreserve = request.target.float16DenormPreserve;
    targetOptions.spannedWaves = request.target.spannedWaves;
    targetOptions.robustBufferAccess2 = request.target.robustBufferAccess2;
    targetOptions.float32FlushToZero = HardwareFloat32Flush(request);
    targetOptions.float32DenormPreserve = request.target.float32DenormPreserve;
    targetOptions.fusedExactArithmetic = request.target.fusedExactArithmetic;
    targetOptions.largePixelFunctionLdsBuffer = request.target.largePixelFunctionLdsBuffer;
    targetOptions.pixelFunctionStorageDwords = request.target.pixelFunctionStorageDwords;
    targetOptions.meshBarycentricPixel = request.shader.stage == ShaderStage::Fragment && request.graphics && request.graphics->mesh;
    if (request.shader.stage == ShaderStage::Mesh && request.graphics && request.graphics->meshBarycentrics) {
        const auto& paired = *request.graphics->meshBarycentrics;
        const auto emulated = DescribeEmulatedInterface(paired.parameters, paired.emulation, true);
        const auto& layout = emulated.layout;
        targetOptions.meshBarycentricVertex = true;
        targetOptions.barycentricSmoothLocation = layout.smoothLocation;
        targetOptions.barycentricLinearLocation = layout.linearLocation;
        targetOptions.barycentricPositionsLocation = layout.positionsLocation;
        targetOptions.meshExportedLocations = emulated.passed;
        for (const auto& parameter : emulated.perVertex) targetOptions.meshPerVertexLocations.emplace_back(parameter.sourceLocation, parameter.location);
        if (!request.graphics->mesh || !request.target.mesh) throw std::runtime_error("barycentric emulation on the mesh path needs a mesh configuration and a device with mesh shaders");
        const auto& config = *request.graphics->mesh;
        const auto& limits = *request.target.mesh;
        if (config.provokingVertex != 0u) throw std::runtime_error("barycentric emulation on the mesh path is only measured with the first provoking vertex");
        const std::uint64_t perVertexParameters = emulated.perVertex.size();
        const std::uint64_t sharedBytes = 16ull * config.maxVertices * (1u + perVertexParameters) + DeviceMemoryLdsBytes(request);
        if (sharedBytes > limits.maxSharedMemoryBytes) throw std::runtime_error("barycentric emulation on the mesh path needs " + std::to_string(sharedBytes) + " bytes of mesh shared memory for " + std::to_string(config.maxVertices) + " vertices, the device allows " + std::to_string(limits.maxSharedMemoryBytes));
        MeshOutputUse use;
        std::set<std::uint32_t> flatLocations;
        for (const auto& output : program.Info().outputs) {
            if (output.kind == StageOutputKind::Parameter && std::ranges::find(targetOptions.meshExportedLocations, output.location) != targetOptions.meshExportedLocations.end()) flatLocations.insert(output.location);
            if (output.kind == StageOutputKind::ClipDistance) use.clipDistances = std::max(use.clipDistances, output.index + 1u);
            if (output.kind == StageOutputKind::CullDistance) use.cullDistances = std::max(use.cullDistances, output.index + 1u);
            if (output.kind == StageOutputKind::Layer) use.layer = true;
        }
        use.flatParameters = static_cast<std::uint32_t>(flatLocations.size());
        use.perVertexParameters = static_cast<std::uint32_t>(perVertexParameters);
        use.clipVaryings = emulated.barycentrics;
        const auto budget = MeshBarycentricOutputBudget(config, limits, use);
        if (budget.components > limits.maxOutputComponents) throw std::runtime_error("barycentric emulation on the mesh path needs " + std::to_string(budget.components) + " mesh output components (" + std::to_string(budget.vertexLocations) + " per-vertex and " + std::to_string(budget.primitiveLocations) + " per-primitive locations), the device allows " + std::to_string(limits.maxOutputComponents));
        if (budget.bytes > limits.maxOutputMemoryBytes) throw std::runtime_error("barycentric emulation on the mesh path needs " + std::to_string(budget.bytes) + " bytes of mesh output memory, the device allows " + std::to_string(limits.maxOutputMemoryBytes));
    }
    targetOptions.waveLdsMemoryBarriers = request.target.waveLdsMemoryBarriers;
    targetOptions.deviceScopeLdsMemoryBarriers = request.target.deviceScopeLdsMemoryBarriers;

    constexpr SpirvEmitter spirvEmitter;
    CompiledShaderArtifact result;
    result.variantId = nextVariantId();
    SpirvEmitInfo emitted;
    try {
        result.spirv = spirvEmitter.Emit(program, inputInfo, bindings, targetOptions, &emitted);
    } catch (const SpannedWaveUnsupported& error) {
        std::fprintf(stderr, "[spanned-wave] 0x%llx falls back to two lanes per invocation: %s\n", static_cast<unsigned long long>(request.shader.codeAddress), error.what());
        targetOptions.spannedWaves = false;
        result.spirv = spirvEmitter.Emit(program, inputInfo, bindings, targetOptions, &emitted);
    }
    if (registration && result.spirv.size() > request.target.registrationWordLimit) {
        registration = false;
        program.Metadata().bindingLayoutComplete = false;
        bindings = bindingAllocator.Allocate(program, request.layout, false, stencilReplay);
        targetOptions.registrationModules = false;
        result.spirv = spirvEmitter.Emit(program, inputInfo, bindings, targetOptions, &emitted);
    }
    result.registrationModule = registration;
    if (request.shader.stage == ShaderStage::Mesh && request.graphics) result.meshBarycentrics = request.graphics->meshBarycentrics;
    result.stencilReplaySafe = (request.shader.stage == ShaderStage::Vertex || stencilReplay) && StencilReplaySafe(program);
    result.bdaWrites = program.Info().bdaWrites;
    if (stencilReplay) {
        result.stencilReferences = StencilReferenceValues(program, request.context.pixel.has_value() && request.context.pixel->stencilOperationValueExport ? 8u : 0u);
        result.stencilReferenceDword = bindings.layout.stencilReferenceDword;
    }

    result.bdaAbiVersion = program.Info().usesDma || program.Info().usesFaultBuffer ? request.target.bdaAbiVersion : 0u;
    result.windowLaneGuards = 0;
    for (const auto& block : program.Blocks()) result.windowLaneGuards += static_cast<std::uint32_t>(std::ranges::count_if(block->Instructions(), [](const IrValue* value) { return value->Opcode() == IrOpcode::WindowLaneGuard; }));
    result.memoryOffsetDword = bindings.layout.memoryOffsetDword;
    result.shaderDataDwords = bindings.layout.ShaderDataDwords();
    result.imageMetadataDword = bindings.layout.ImageMetadataDword();
    result.runtimeImageCount = bindings.layout.runtimeImageCount;
    result.hostSubgroupSize = HostSubgroupSize(request);
    result.fullSubgroups = emitted.fullSubgroups;
    result.functionLdsBuffer = emitted.functionLdsBuffer;
    result.vertexOffsetSgpr = program.Info().vertexOffsetSgpr;
    result.instanceOffsetSgpr = program.Info().instanceOffsetSgpr;
    result.vertexOffsetShared = program.Info().vertexOffsetShared;
    result.instanceOffsetShared = program.Info().instanceOffsetShared;
    result.vertexOffsetConflict = program.Info().vertexOffsetConflict;
    result.instanceOffsetConflict = program.Info().instanceOffsetConflict;
    for (const auto& output : program.Info().outputs) {
        if (output.kind == StageOutputKind::Parameter) result.parameterExports.push_back(output.location);
    }
    if (request.shader.stage == ShaderStage::Fragment) {
        result.fragmentParameters = DescribeFragmentParameters(program, inputInfo);
        const auto& inputs = program.Info().inputs;
        const auto reads = [&](StageInputKind kind) { return std::any_of(inputs.begin(), inputs.end(), [&](const auto& input) { return input.kind == kind; }); };
        if (program.Metadata().barycentricEmulation) result.barycentricEmulation = {true, reads(StageInputKind::BaryCoordSmooth), reads(StageInputKind::BaryCoordNoPerspective), program.Info().smoothOffsetBarycentrics};
    }
    if (request.shader.stage == ShaderStage::Vertex || request.shader.stage == ShaderStage::Local) {
        if (inputInfo.vertex == nullptr) throw std::runtime_error("vertex input metadata is missing");
        for (const auto& input : program.Info().inputs) {
            if (input.kind != StageInputKind::Parameter) continue;
            if (input.location >= static_cast<std::uint32_t>(inputInfo.vertex->resourcesNum)) throw std::runtime_error("vertex attribute location exceeds resource count");
            result.vertexInputs.push_back({input.location, input.componentCount, inputInfo.vertex->resourcesDst[input.location].fetchIndex});
            if (inputInfo.vertex->fetchEmbedded) {
                for (const auto& block : program.Blocks()) {
                    for (const auto* instruction : block->Instructions()) {
                        if (instruction->Opcode() == IrOpcode::GetAttribute && instruction->Argument(0)->Resolve()->ImmediateU32() == input.location) result.vertexInputs.back().outputMask |= 1u << instruction->Argument(1)->Resolve()->ImmediateU32();
                    }
                }
            }
        }
    }

    const bool embedded = !result.vertexInputs.empty() && inputInfo.vertex != nullptr && inputInfo.vertex->fetchEmbedded;
    if (embedded) PrepareVertexInputSpecialization(result);
#if ANYPS5_ENABLE_SPIRV_TOOLS
    result.spirv = ValidateAndOptimizeSpirv(result.spirv, request.target.vulkanVersion, request.target.spirvVersion, request.target.nonConstantImageOffsets, !embedded);
    if (embedded) {
        std::array<std::uint32_t, ShaderVertexStageInfo::MaxResources> classes{};
        for (std::uint32_t kind = 1u; kind < 3u; ++kind) {
            classes.fill(kind);
            const auto specialized = SpecializeVertexInputTypes(result, classes);
            static_cast<void>(ValidateAndOptimizeSpirv(specialized, request.target.vulkanVersion, request.target.spirvVersion, request.target.nonConstantImageOffsets, false));
        }
    }
#endif

    return {request.layout, {}, std::move(static_cast<CompiledBindingLayout&>(bindings)), std::move(result)};
}

CompiledVariant compileVariant(const RecompileRequest& request, IrProgram program, std::exception_ptr* emissionFailure = nullptr) {
    return recordEmissionFailure(emissionFailure, [&] {
        prepareInterface(request, program);
        auto variant = emitVariant(request, program);
        variant.info = std::move(program).TakeCompiledInfo();
        return variant;
    });
}

struct SpecializedModule {
    std::uint64_t specializationId = 0;
    SharedSpirv spirv;
    std::vector<std::uint32_t> bindings;
    bool pushData = false;
};

std::shared_ptr<const SpecializedModule> buildSpecializedModule(const CompiledShaderArtifact& artifact, std::span<const std::uint32_t> classes, std::span<const PipelineSpecializationConstant> constants, const SpirvTarget& target) {
    auto source = SpecializeVertexInputTypes(artifact, classes);
    std::map<std::uint32_t, std::uint32_t> supplied;
    for (const auto& constant : constants) {
        if (!supplied.emplace(constant.id, constant.value).second) throw std::runtime_error("duplicate prepared specialization ID");
    }
    std::map<std::uint32_t, std::uint32_t> values;
    std::set<std::uint32_t> retained;
    std::set<std::uint32_t> specializedIds;
    std::set<std::uint32_t> pushTypes;
    std::optional<std::uint32_t> replaySlotBytes;
    const auto& words = source.Words();
    if (words.size() < 5u || words[0] != spv::MagicNumber) throw std::runtime_error("invalid prepared specialization module");
    for (std::size_t cursor = 5; cursor < words.size();) {
        const auto count = words[cursor] >> 16u;
        const auto op = static_cast<spv::Op>(words[cursor] & 0xffffu);
        if (count == 0u || count > words.size() - cursor) throw std::runtime_error("truncated prepared specialization instruction");
        if (!artifact.stencilReferences.empty() && op == spv::OpTypePointer && count == 4u && words[cursor + 2u] == spv::StorageClassPushConstant) pushTypes.insert(words[cursor + 3u]);
        if (op == spv::OpDecorate && count == 4u && words[cursor + 2u] == spv::DecorationSpecId) {
            if (PipelineSpecialization::IsWorkgroupOffset(words[cursor + 3u])) {
                if (supplied.contains(words[cursor + 3u])) throw std::runtime_error("a workgroup offset is supplied at dispatch, not prepared");
                if (!retained.insert(words[cursor + 1u]).second) throw std::runtime_error("duplicate workgroup offset specialization");
                cursor += count;
                continue;
            }
            const auto found = supplied.find(words[cursor + 3u]);
            if (found == supplied.end()) throw std::runtime_error("missing prepared specialization value");
            auto value = found->second;
            if (!artifact.stencilReferences.empty()) {
                if (found->first == PipelineSpecialization::StencilReferenceOffset) replaySlotBytes = value / PushData::DwordCount * NativePushSlotSize;
                if (found->first == PipelineSpecialization::StencilReferenceOffset || found->first == PipelineSpecialization::PushDataOffset) value %= PushData::DwordCount;
            }
            if (!values.emplace(words[cursor + 1u], value).second) throw std::runtime_error("duplicate prepared specialization value");
            specializedIds.insert(found->first);
        }
        cursor += count;
    }
    std::vector<std::uint32_t> materialized(words.begin(), words.begin() + 5);
    for (std::size_t cursor = 5; cursor < words.size();) {
        const auto count = words[cursor] >> 16u;
        const auto op = static_cast<spv::Op>(words[cursor] & 0xffffu);
        if (op == spv::OpSpecConstant && count == 4u && retained.contains(words[cursor + 2u])) {
            materialized.insert(materialized.end(), words.begin() + cursor, words.begin() + cursor + count);
        } else if (op == spv::OpDecorate && count == 4u && words[cursor + 2u] == spv::DecorationSpecId && retained.contains(words[cursor + 1u])) {
            materialized.insert(materialized.end(), words.begin() + cursor, words.begin() + cursor + count);
        } else if (op == spv::OpSpecConstant) {
            if (count != 4u || !values.contains(words[cursor + 2u])) throw std::runtime_error("invalid prepared specialization constant");
            materialized.insert(materialized.end(), {(4u << 16u) | spv::OpConstant, words[cursor + 1u], words[cursor + 2u], values.at(words[cursor + 2u])});
        } else if (op == spv::OpMemberDecorate && count == 5u && words[cursor + 3u] == spv::DecorationOffset && pushTypes.contains(words[cursor + 1u])) {
            if (!replaySlotBytes || words[cursor + 2u] != 0u || *replaySlotBytes >= NativePushConstantSize) throw std::runtime_error("invalid stencil replay push constant slot");
            materialized.insert(materialized.end(), words.begin() + cursor, words.begin() + cursor + count - 1u);
            materialized.push_back(*replaySlotBytes);
        } else if (!(op == spv::OpDecorate && count == 4u && words[cursor + 2u] == spv::DecorationSpecId)) {
            materialized.insert(materialized.end(), words.begin() + cursor, words.begin() + cursor + count);
        }
        cursor += count;
    }
    materialized = SpecializeFragmentOutputs(SpecializeSpirv(materialized), constants, specializedIds);
#if ANYPS5_ENABLE_SPIRV_TOOLS
    materialized = ValidateAndOptimizeSpirv(materialized, target.vulkanVersion, target.spirvVersion, target.nonConstantImageOffsets, true, true);
#endif
    std::map<std::uint32_t, std::uint32_t> descriptorVariables;
    std::set<std::uint32_t> usedDescriptors;
    for (std::size_t cursor = 5; cursor < materialized.size();) {
        const auto count = materialized[cursor] >> 16u;
        const auto op = static_cast<spv::Op>(materialized[cursor] & 0xffffu);
        if (op == spv::OpDecorate && count == 4u && materialized[cursor + 2u] == spv::DecorationBinding) descriptorVariables.emplace(materialized[cursor + 1u], materialized[cursor + 3u]);
        if ((op == spv::OpAccessChain || op == spv::OpInBoundsAccessChain || op == spv::OpPtrAccessChain || op == spv::OpInBoundsPtrAccessChain || op == spv::OpLoad || op == spv::OpCopyObject) && count >= 4u) usedDescriptors.insert(materialized[cursor + 3u]);
        cursor += count;
    }
    const auto unused = [&](std::uint32_t id) { return descriptorVariables.contains(id) && !usedDescriptors.contains(id); };
    std::vector<std::uint32_t> compact(materialized.begin(), materialized.begin() + 5);
    for (std::size_t cursor = 5; cursor < materialized.size();) {
        const auto count = materialized[cursor] >> 16u;
        const auto op = static_cast<spv::Op>(materialized[cursor] & 0xffffu);
        if (op == spv::OpEntryPoint) {
            const auto start = compact.size();
            std::size_t interfaceIndex = 3u;
            for (; interfaceIndex < count; ++interfaceIndex) {
                const auto word = materialized[cursor + interfaceIndex];
                if ((word & 0xffu) == 0u || (word & 0xff00u) == 0u || (word & 0xff0000u) == 0u || (word & 0xff000000u) == 0u) { ++interfaceIndex; break; }
            }
            compact.insert(compact.end(), materialized.begin() + cursor, materialized.begin() + cursor + interfaceIndex);
            for (auto index = interfaceIndex; index < count; ++index) if (!unused(materialized[cursor + index])) compact.push_back(materialized[cursor + index]);
            compact[start] = (static_cast<std::uint32_t>(compact.size() - start) << 16u) | spv::OpEntryPoint;
        } else if (!((op == spv::OpVariable && unused(materialized[cursor + 2u])) || ((op == spv::OpDecorate || op == spv::OpName) && unused(materialized[cursor + 1u])))) {
            compact.insert(compact.end(), materialized.begin() + cursor, materialized.begin() + cursor + count);
        }
        cursor += count;
    }
    materialized = std::move(compact);
#if ANYPS5_ENABLE_SPIRV_TOOLS
    materialized = ValidateAndOptimizeSpirv(materialized, target.vulkanVersion, target.spirvVersion, target.nonConstantImageOffsets, false);
#endif
    auto module = std::make_shared<SpecializedModule>();
    std::map<std::uint32_t, std::uint32_t> bindingNumbers;
    std::unordered_set<std::uint32_t> functionLdsBuffer;
    for (std::size_t cursor = 5; cursor < materialized.size();) {
        const auto count = materialized[cursor] >> 16u;
        const auto op = static_cast<spv::Op>(materialized[cursor] & 0xffffu);
        if (op == spv::OpDecorate && count == 4u && materialized[cursor + 2u] == spv::DecorationBinding) bindingNumbers.emplace(materialized[cursor + 1u], materialized[cursor + 3u]);
        if (op == spv::OpDecorate && count == 4u && materialized[cursor + 2u] == spv::DecorationDescriptorSet && materialized[cursor + 3u] == PixelFunctionLdsDescriptorSet) functionLdsBuffer.insert(materialized[cursor + 1u]);
        if (op == spv::OpVariable && count >= 4u && !functionLdsBuffer.contains(materialized[cursor + 2u])) {
            if (const auto found = bindingNumbers.find(materialized[cursor + 2u]); found != bindingNumbers.end()) module->bindings.push_back(found->second);
            module->pushData |= materialized[cursor + 3u] == spv::StorageClassPushConstant;
        }
        cursor += count;
    }
    module->spirv = InternSpirv(std::move(materialized));
    module->specializationId = constants.empty() ? 0u : nextVariantId();
    return module;
}

struct SpecializedModuleEntry {
    std::once_flag ready;
    std::exception_ptr failure;
    std::shared_ptr<const SpecializedModule> module;
};

struct PreparedModuleEntry {
    std::once_flag ready;
    std::exception_ptr failure;
    std::shared_ptr<const SpecializedModule> module;
    DescriptorBindingPlan bindings;
};

struct PreparedBindingPlan {
    std::once_flag ready;
    std::exception_ptr failure;
    DescriptorBindingPlan bindings;
    PrimedSharedMutex mutex;
    std::map<std::vector<std::uint32_t>, std::shared_ptr<PreparedModuleEntry>> modules;
};

struct RegistrationModuleEntry {
    std::once_flag ready;
    std::exception_ptr failure;
    std::shared_ptr<const SpecializedModule> module;
};

std::shared_ptr<const SpecializedModule> registrationModule(std::uint64_t variantId, std::span<const std::uint32_t> moduleKey, const std::function<std::shared_ptr<const SpecializedModule>()>& specialize) {
    std::vector<std::uint64_t> key{variantId};
    key.insert(key.end(), moduleKey.begin(), moduleKey.end());
    static PrimedSharedMutex mutex;
    static std::map<std::vector<std::uint64_t>, std::shared_ptr<RegistrationModuleEntry>> modules;
    std::shared_ptr<RegistrationModuleEntry> entry;
    {
        std::shared_lock lock(mutex);
        if (const auto found = modules.find(key); found != modules.end()) entry = found->second;
    }
    if (entry == nullptr) {
        std::unique_lock lock(mutex);
        const auto found = modules.find(key);
        entry = found != modules.end() ? found->second : modules.emplace(key, std::make_shared<RegistrationModuleEntry>()).first->second;
    }
    std::call_once(entry->ready, [&] {
        try {
            entry->module = specialize();
        } catch (...) {
            entry->failure = std::current_exception();
        }
    });
    if (entry->failure) std::rethrow_exception(entry->failure);
    return entry->module;
}

ProgramKeyCensus& programKeyCensus() {
    static ProgramKeyCensus* census = new ProgramKeyCensus();
    return *census;
}

bool programKeyCensusEnabled() {
    static const bool enabled = std::getenv("APS5_PROFILE_FRAME") != nullptr;
    return enabled;
}

// The binding-plan key of this thread's last preparedBindingPlan, hashed (program-key census).
thread_local std::uint64_t lastPlanKeyHash = 0;

MaterializeSplit& materializeSplit() {
    static MaterializeSplit* split = new MaterializeSplit();
    return *split;
}

// The clock of this thread's sampled Recompile(request, capture) call, null while the call is not
// timed: materializeResult marks its parts on it.
thread_local MaterializeSplit::Sample* splitSample = nullptr;

void markSplit(MaterializeSplit::Part part) {
    if (splitSample != nullptr) splitSample->Mark(part);
}

// APS5_PROFILE_FRAME: times one call in MaterializeSplit::SampleEvery and books it when the call
// ends, whichever way it ends.
class SplitScope {
public:
    SplitScope() {
        if (!programKeyCensusEnabled() || !materializeSplit().Sampled()) return;
        sample.emplace();
        splitSample = &*sample;
    }
    SplitScope(const SplitScope&) = delete;
    SplitScope& operator=(const SplitScope&) = delete;
    ~SplitScope() {
        if (!sample) return;
        splitSample = nullptr;
        materializeSplit().Add(*sample);
    }

private:
    std::optional<MaterializeSplit::Sample> sample;
};

void bindingPlanKey(const CompiledVariant& variant, const ResourceSnapshot& snapshot, std::span<const std::uint8_t> exports, std::vector<std::uint64_t>& key) {
    key.clear();
    key.push_back(variant.artifact.variantId);
    for (std::size_t index = 0; index < variant.info.info.buffers.size(); ++index) {
        const auto& descriptor = snapshot.buffers.at(index);
        key.push_back(descriptor.dwordCount);
        key.push_back(descriptor.dwords[1] & 0xffff0000u);
        key.push_back(descriptor.dwords[3]);
        const bool present = descriptor.dwords[2] != 0u && (descriptor.dwords[0] != 0u || (descriptor.dwords[1] & 0xffffu) != 0u);
        key.push_back(present ? 4u | (descriptor.dwords[0] & 3u) : 0u);
    }
    for (std::size_t index = 0; index < variant.info.info.images.size(); ++index) {
        const auto& descriptor = snapshot.images.at(index);
        key.push_back(descriptor.dwordCount);
        key.push_back(descriptor.dwords[0] != 0u || (descriptor.dwords[1] & 0xffu) != 0u);
        key.push_back(descriptor.dwords[1] & ~0xffu);
        for (std::size_t word = 2; word < 8; ++word) key.push_back(descriptor.dwords[word]);
    }
    for (std::size_t index = 0; index < variant.info.info.samplers.size(); ++index) {
        const auto& descriptor = snapshot.samplers.at(index);
        key.push_back(descriptor.dwordCount);
        for (std::size_t word = 0; word < 4; ++word) key.push_back(descriptor.dwords[word]);
    }
    key.push_back(exports.size());
    for (const auto mapping : exports) key.push_back(mapping);
    if (programKeyCensusEnabled()) lastPlanKeyHash = ProgramKeyCensus::Hash(key);
}

std::shared_ptr<PreparedBindingPlan> preparedBindingPlan(const CompiledVariant& variant, const std::vector<std::uint64_t>& key, const ResourceSnapshot& snapshot, std::span<const std::uint8_t> exports, bool fragmentDescriptorSet) {
    static PrimedSharedMutex mutex;
    static std::map<std::vector<std::uint64_t>, std::shared_ptr<PreparedBindingPlan>> plans;
    std::shared_ptr<PreparedBindingPlan> plan;
    {
        std::shared_lock lock(mutex);
        if (const auto found = plans.find(key); found != plans.end()) plan = found->second;
    }
    if (plan == nullptr) {
        std::unique_lock lock(mutex);
        const auto found = plans.find(key);
        plan = found != plans.end() ? found->second : plans.emplace(key, std::make_shared<PreparedBindingPlan>()).first->second;
    }
    std::call_once(plan->ready, [&] {
        try {
            plan->bindings = DescriptorBindingBuilder{}.Prepare(variant.bindings.layout, variant.info.info, variant.info.stage, snapshot, exports, variant.artifact.registrationModule, fragmentDescriptorSet);
        } catch (...) {
            plan->failure = std::current_exception();
        }
    });
    if (plan->failure) std::rethrow_exception(plan->failure);
    return plan;
}

std::shared_ptr<const SpecializedModule> specializeModule(const CompiledShaderArtifact& artifact, std::span<const std::uint32_t> classes, std::span<const PipelineSpecializationConstant> constants, const SpirvTarget& target) {
    struct ModuleKeyStorage {};
    auto& key = HostThreadLocal<std::vector<std::uint64_t>, ModuleKeyStorage>();
    key.clear();
    key.push_back(artifact.variantId);
    for (const auto& constant : constants) key.push_back((static_cast<std::uint64_t>(constant.id) << 32u) | constant.value);
    static PrimedSharedMutex mutex;
    static std::map<std::vector<std::uint64_t>, std::shared_ptr<SpecializedModuleEntry>> modules;
    std::shared_ptr<SpecializedModuleEntry> entry;
    {
        std::shared_lock lock(mutex);
        if (const auto found = modules.find(key); found != modules.end()) entry = found->second;
    }
    if (entry == nullptr) {
        std::unique_lock lock(mutex);
        const auto found = modules.find(key);
        entry = found != modules.end() ? found->second : modules.emplace(key, std::make_shared<SpecializedModuleEntry>()).first->second;
    }
    std::call_once(entry->ready, [&] {
        try {
            entry->module = buildSpecializedModule(artifact, classes, constants, target);
        } catch (...) {
            entry->failure = std::current_exception();
        }
    });
    if (entry->failure) std::rethrow_exception(entry->failure);
    return entry->module;
}

const ResourceSnapshot& mergeTargetImage(const CompiledVariant& variant, const RecompileRequest& request, const ResourceSnapshot& snapshot, bool registrationDefaults, ResourceSnapshot& merged) {
    const auto& images = variant.info.info.images;
    const auto image = std::ranges::find_if(images, [](const ImageResource& candidate) { return candidate.targetMerge; });
    if (image == images.end() || registrationDefaults) return snapshot;
    if (!request.context.pixel || request.context.pixel->targetMerge.mask == 0u) throw std::runtime_error("ShaderRecompiler: a color target merge variant was materialized without the merged target");
    if (image + 1 != images.end() || snapshot.images.size() + 1u != images.size()) throw std::runtime_error("ShaderRecompiler: the captured images do not precede the color target merge image");
    merged = snapshot;
    DescriptorValue descriptor;
    std::ranges::copy(request.context.pixel->targetMerge.descriptor, descriptor.dwords.begin());
    descriptor.dwordCount = 8u;
    descriptor.availableDwordMask = 0xffu;
    merged.images.push_back(descriptor);
    return merged;
}

struct ProgramMemoCounters {
    std::atomic<std::uint64_t> hits{0}, misses{0}, resets{0}, verifyCalls{0}, verifyChecks{0}, verifyMismatches{0};
};

ProgramMemoCounters& programMemoCounters() {
    static ProgramMemoCounters counters;
    return counters;
}

// APS5_NO_PROGRAM_SPLIT=1: a memoized materialization builds its program fields again for every
// result instead of filling the words into the program memo's program.
bool ProgramSplit() {
    static const bool split = std::getenv("APS5_NO_PROGRAM_SPLIT") == nullptr;
    return split;
}

// What one materialization reads from the request and the snapshot to select its program, and the
// per-draw vertex values found on the way.
struct ProgramInputs {
    std::vector<std::uint64_t> planKey;
    std::vector<std::uint32_t> moduleKey;
    // The program key: the plan key, the module key and the request words the program reads.
    std::vector<std::uint64_t> key;
    std::array<std::uint32_t, ShaderVertexStageInfo::MaxResources> vertexClasses{};
    std::array<std::uint32_t, ShaderVertexStageInfo::MaxResources / RuntimeAbi::VertexSelectorsPerDword> runtimeSelectors{};
    std::vector<VertexAttribute> vertexAttributes;
    bool depthRemap = false;
};

// The program fields of a result (no bindings, push constants or per-draw values) and the plan and
// module they came from.
struct ProgramState : ProgramPart {
    RecompileResult program;
    std::shared_ptr<PreparedBindingPlan> plan;
    std::shared_ptr<PreparedModuleEntry> entry;
    const DescriptorBindingPlan* bindings = nullptr;
};

std::span<const std::uint8_t> exportMapping(const RecompileRequest& request) {
    return request.context.pixel ? std::span<const std::uint8_t>(request.context.pixel->targetExportMapping) : std::span<const std::uint8_t>{};
}

void readProgramInputs(const CompiledVariant& variant, const RecompileRequest& request, const ResourceSnapshot& snapshot, bool registrationDefaults, ProgramInputs& inputs) {
    const auto& artifact = variant.artifact;
    bindingPlanKey(variant, snapshot, exportMapping(request), inputs.planKey);
    auto& moduleKey = inputs.moduleKey;
    moduleKey.clear();
    if (variant.bindings.layout.UsesPushData()) moduleKey.push_back(request.layout.pushConstantOffsetBytes / 4u);
    if (!artifact.stencilReferences.empty()) moduleKey.push_back(request.layout.pushConstantOffsetBytes / 4u + artifact.stencilReferenceDword);
    if (request.context.pixel) {
        for (const auto packing : request.context.pixel->targetExportPacking) moduleKey.push_back(static_cast<std::uint32_t>(packing));
        moduleKey.push_back(request.context.pixel->dualSourceBlend ? 1u : 0u);
        moduleKey.push_back(request.context.pixel->reversedBlend);
        if (variant.info.info.rasterPrimitive) moduleKey.push_back(static_cast<std::uint32_t>(request.context.pixel->rasterPrimitive));
        if (variant.info.info.layerExported) moduleKey.push_back(request.context.pixel->layerExported ? 1u : 0u);
    }
    inputs.depthRemap = request.target.depthRangeRemap && request.shader.stage == ShaderStage::Vertex && request.context.vertex.has_value();
    if (inputs.depthRemap) {
        if (request.context.vertex->depthRemapMode != 0u && request.layout.pushConstantOffsetBytes % (RuntimeAbi::PushConstantDwords * 4u) + request.layout.pushConstantSizeBytes > DepthRemapPushOffsetBytes) {
            throw std::runtime_error("a remapped vertex shader's push data must end before the depth remap words");
        }
        moduleKey.push_back(request.context.vertex->depthRemapMode);
    }
    inputs.vertexAttributes.clear();
    inputs.vertexAttributes.reserve(artifact.vertexInputs.size());
    inputs.vertexClasses = {};
    inputs.runtimeSelectors = {};
    for (const auto& input : artifact.vertexInputs) {
        if (!request.context.vertex || input.location >= request.context.vertex->resourcesNum || input.location >= request.context.vertex->resources.size()) throw std::runtime_error("Shader cache: invalid vertex attribute metadata");
        const auto& vertex = *request.context.vertex;
        const auto& resource = vertex.resources[input.location];
        const auto fetchIndex = vertex.fetchEmbedded ? vertex.resourcesDst[input.location].fetchIndex : input.fetchIndex;
        inputs.vertexAttributes.push_back({input.location, input.components, resource, fetchIndex});
        if (!artifact.vertexInputPatches.empty()) {
            const auto format = static_cast<IrBufferFormat>((resource.fields[3] >> 12u) & 0x7fu);
            const auto numeric = registrationDefaults && format == IrBufferFormat::Invalid ? IrTextureNumericClass::Float : VertexInputNumericClass(format);
            if (numeric == IrTextureNumericClass::Unsupported) throw std::runtime_error("unsupported prepared vertex format");
            const auto kind = numeric == IrTextureNumericClass::Float ? 0u : numeric == IrTextureNumericClass::Sint ? 1u : 2u;
            inputs.vertexClasses.at(input.location) = kind;
            std::uint32_t selectors = kind << 12u;
            auto& formatComponents = inputs.vertexAttributes.back().formatComponents;
            formatComponents = 1u;
            for (std::uint32_t component = 0; component < 4u; ++component) {
                const auto selector = (input.outputMask & (1u << component)) != 0u ? (resource.fields[3] >> (component * 3u)) & 7u : 0u;
                if (selector == 2u || selector == 3u) {
                    const auto& fields = resource.fields;
                    char message[192];
                    std::snprintf(message, sizeof(message), "reserved prepared vertex component selector %u at location %u component %u (V#=[%08x %08x %08x %08x])",
                        selector, input.location, component, fields[0], fields[1], fields[2], fields[3]);
                    throw std::runtime_error(message);
                }
                if (selector >= 4u) formatComponents = std::max(formatComponents, selector - 3u);
                selectors |= selector << (component * 3u);
            }
            moduleKey.push_back(artifact.registrationModule ? selectors & ~0xfffu : selectors);
            if (artifact.registrationModule) inputs.runtimeSelectors.at(input.location / RuntimeAbi::VertexSelectorsPerDword) |= (selectors & 0xfffu) << (input.location % RuntimeAbi::VertexSelectorsPerDword * 16u);
        }
    }
    if (programKeyCensusEnabled()) programKeyCensus().Note(ProgramKeyCensus::Combine(lastPlanKeyHash, ProgramKeyCensus::Hash(std::span<const std::uint32_t>(moduleKey))));
    auto& key = inputs.key;
    key.assign(inputs.planKey.begin(), inputs.planKey.end());
    key.push_back(moduleKey.size());
    key.insert(key.end(), moduleKey.begin(), moduleKey.end());
    key.push_back(request.context.userDataBaseRegister);
    key.push_back(static_cast<std::uint64_t>(request.shader.stage) | (registrationDefaults ? 1ull << 32u : 0u) | (inputs.depthRemap ? 1ull << 33u : 0u) | (request.target.registrationModules ? 1ull << 34u : 0u));
    key.push_back(request.layout.pushConstantOffsetBytes | (static_cast<std::uint64_t>(request.layout.pushConstantSizeBytes) << 32u));
}

// The program fields of every result with `inputs.key`: the artifact, the user data registers read,
// the binding plan and the specialized module.
std::shared_ptr<ProgramState> buildProgram(const CompiledVariant& variant, const RecompileRequest& request, const ResourceSnapshot& snapshot, const ProgramInputs& inputs) {
    auto state = std::make_shared<ProgramState>();
    auto& result = state->program;
    const auto& artifact = variant.artifact;
    static_cast<CompiledShaderArtifact&>(result) = {
        .memoryOffsetDword = artifact.memoryOffsetDword,
        .shaderDataDwords = artifact.shaderDataDwords,
        .imageMetadataDword = artifact.imageMetadataDword,
        .runtimeImageCount = artifact.runtimeImageCount,
        .runtimeImageResources = artifact.runtimeImageResources,
        .registrationModule = artifact.registrationModule,
        .bdaAbiVersion = artifact.bdaAbiVersion,
        .runtimeAbiVersion = artifact.runtimeAbiVersion,
        .vertexInputs = artifact.vertexInputs,
        .vertexOffsetSgpr = artifact.vertexOffsetSgpr,
        .instanceOffsetSgpr = artifact.instanceOffsetSgpr,
        .vertexOffsetShared = artifact.vertexOffsetShared,
        .instanceOffsetShared = artifact.instanceOffsetShared,
        .vertexOffsetConflict = artifact.vertexOffsetConflict,
        .instanceOffsetConflict = artifact.instanceOffsetConflict,
        .hostSubgroupSize = artifact.hostSubgroupSize,
        .fullSubgroups = artifact.fullSubgroups,
        .functionLdsBuffer = artifact.functionLdsBuffer,
        .parameterExports = artifact.parameterExports,
        .fragmentParameters = artifact.fragmentParameters,
        .barycentricEmulation = artifact.barycentricEmulation,
        .meshBarycentrics = artifact.meshBarycentrics,
        .stencilReferences = artifact.stencilReferences,
        .stencilReplaySafe = artifact.stencilReplaySafe,
        .bdaWrites = artifact.bdaWrites,
        .stencilReferenceDword = artifact.stencilReferenceDword,
        .variantId = artifact.variantId
    };
    for (const auto reg : variant.bindings.layout.userDataRegisters) {
        const auto index = reg - request.context.userDataBaseRegister;
        result.userDataRegisterReads |= reg >= request.context.userDataBaseRegister && index < 64u ? std::uint64_t{1} << index : ~std::uint64_t{0};
    }
    markSplit(MaterializeSplit::Artifact);
    const auto& plan = state->plan = preparedBindingPlan(variant, inputs.planKey, snapshot, exportMapping(request), request.target.registrationModules);
    state->bindings = &plan->bindings;
    const auto& moduleKey = inputs.moduleKey;
    if (!plan->bindings.specialization.empty() || variant.bindings.layout.UsesPushData() || !artifact.vertexInputPatches.empty() || request.context.pixel || inputs.depthRemap) {
        auto& entry = state->entry;
        {
            std::shared_lock lock(plan->mutex);
            if (const auto found = plan->modules.find(moduleKey); found != plan->modules.end()) entry = found->second;
        }
        if (entry == nullptr) {
            std::unique_lock lock(plan->mutex);
            const auto found = plan->modules.find(moduleKey);
            entry = found != plan->modules.end() ? found->second : plan->modules.emplace(moduleKey, std::make_shared<PreparedModuleEntry>()).first->second;
        }
        const auto prepare = [&] {
            auto constants = plan->bindings.specialization;
            std::size_t index = 0;
            if (variant.bindings.layout.UsesPushData()) constants.push_back({PipelineSpecialization::PushDataOffset, moduleKey[index++]});
            if (!artifact.stencilReferences.empty()) constants.push_back({PipelineSpecialization::StencilReferenceOffset, moduleKey[index++]});
            if (request.context.pixel) {
                for (std::uint32_t target = 0; target < request.context.pixel->targetExportPacking.size(); ++target) constants.push_back({PipelineSpecialization::ExportPackingBase + target, moduleKey[index++]});
                constants.push_back({PipelineSpecialization::DualSourceBlend, moduleKey[index++]});
                constants.push_back({PipelineSpecialization::ReversedBlend, moduleKey[index++]});
                if (variant.info.info.rasterPrimitive) constants.push_back({PipelineSpecialization::RasterPrimitive, moduleKey[index++]});
                if (variant.info.info.layerExported) constants.push_back({PipelineSpecialization::LayerExported, moduleKey[index++]});
            }
            if (inputs.depthRemap) {
                for (std::uint32_t word = 0; word < PipelineSpecialization::DepthRemapWords; ++word) constants.push_back({PipelineSpecialization::DepthRemapBase + word, moduleKey[index++]});
            }
            if (!artifact.vertexInputPatches.empty()) {
                for (const auto& input : artifact.vertexInputs) {
                    const auto selectors = moduleKey[index++];
                    const auto first = PipelineSpecialization::VertexBase + input.location * PipelineSpecialization::VertexWords;
                    for (std::uint32_t component = 0; component < 4u; ++component) constants.push_back({first + component, (selectors >> (component * 3u)) & 7u});
                    const auto kind = selectors >> 12u;
                    constants.push_back({first + 4u, kind == 0u ? 0x3f800000u : 1u});
                    constants.push_back({first + 5u, kind});
                }
            }
            const auto module = artifact.registrationModule && plan->bindings.specialization.empty() ? registrationModule(artifact.variantId, moduleKey, [&] { return specializeModule(variant.artifact, inputs.vertexClasses, constants, request.target); }) : specializeModule(variant.artifact, inputs.vertexClasses, constants, request.target);
            if (variant.bindings.layout.UsesPushData() && !module->pushData) throw std::runtime_error("specialization removed the prepared push constant interface");
            auto selected = DescriptorBindingBuilder{}.Select(plan->bindings, module->bindings);
            entry->bindings = std::move(selected);
            entry->module = module;
        };
        std::call_once(entry->ready, [&] {
            try {
                prepare();
            } catch (...) {
                entry->failure = std::current_exception();
            }
        });
        if (entry->failure) std::rethrow_exception(entry->failure);
        const auto& module = entry->module;
        result.specializationId = module->specializationId;
        result.spirv = module->spirv;
        state->bindings = &entry->bindings;
    } else {
        result.spirv = artifact.spirv;
    }
    result.workgroupMemoryDwords = WorkgroupMemoryStrideDwords(variant.info.info);
    result.readsDepthRemapPush = inputs.depthRemap;
    if (request.shader.stage == ShaderStage::Compute) {
        for (std::uint32_t component = 0; component < PipelineSpecialization::WorkgroupOffsetWords; ++component) result.specialization.push_back({PipelineSpecialization::WorkgroupOffsetBase + component, 0u});
    }
    return state;
}

// The program memo (Route step 6): one program per program key, for every source.
struct ProgramKeyHash {
    std::size_t operator()(const std::vector<std::uint64_t>& key) const {
        std::uint64_t hash = 14695981039346656037ull;
        for (const auto word : key) hash = (hash ^ word ^ (word >> 32u)) * 1099511628211ull;
        return static_cast<std::size_t>(hash ^ (hash >> 29u));
    }
};

struct ProgramMemo {
    PrimedSharedMutex mutex;
    std::unordered_map<std::vector<std::uint64_t>, std::shared_ptr<const ProgramState>, ProgramKeyHash> programs;
};

// The programs kept before the memo starts again: far more than a game binds, so a reset only
// bounds a run that keeps making new keys.
constexpr std::size_t ProgramMemoCapacity = 1u << 16u;

std::shared_ptr<const ProgramState> memoizedProgram(const CompiledVariant& variant, const RecompileRequest& request, const ResourceSnapshot& snapshot, const ProgramInputs& inputs) {
    static auto* memo = new ProgramMemo();
    auto& counters = programMemoCounters();
    {
        std::shared_lock lock(memo->mutex);
        if (const auto found = memo->programs.find(inputs.key); found != memo->programs.end()) {
            counters.hits.fetch_add(1, std::memory_order_relaxed);
            return found->second;
        }
    }
    auto built = buildProgram(variant, request, snapshot, inputs);
    built->key = inputs.key;
    counters.misses.fetch_add(1, std::memory_order_relaxed);
    std::shared_ptr<const ProgramState> program = std::move(built);
    std::unordered_map<std::vector<std::uint64_t>, std::shared_ptr<const ProgramState>, ProgramKeyHash> dropped;
    std::unique_lock lock(memo->mutex);
    if (memo->programs.size() >= ProgramMemoCapacity) {
        dropped.swap(memo->programs);
        counters.resets.fetch_add(1, std::memory_order_relaxed);
    }
    const auto [found, inserted] = memo->programs.emplace(program->key, program);
    if (!inserted) program = found->second;
    lock.unlock();
    return program;
}

// One draw's result of `program`: its own bindings, push constants and per-draw values over the
// program's fields.
RecompileResult fillData(const ProgramState& program, const CompiledVariant& variant, const RecompileRequest& request, const ResourceSnapshot& snapshot, ProgramInputs& inputs) {
    RecompileResult result = program.program;
    const auto& artifact = variant.artifact;
    result.vertexAttributes = std::move(inputs.vertexAttributes);
    markSplit(MaterializeSplit::Artifact);
    const auto resolved = ResourceMaterializer::ResolveImageTablesShared(variant.info.info, snapshot, artifact.variantId);
    const auto& tables = *resolved;
    BindingAllocationResult bindings;
    DescriptorBindingBuilder{}.Populate(bindings, variant.bindings, *program.bindings, variant.info.userDataBase, snapshot, partialThreads(request), &tables);
    markSplit(MaterializeSplit::Populate);
    result.bindings = std::move(bindings.bindings);
    if (std::ranges::none_of(result.bindings, [](const DescriptorBinding& binding) { return binding.role == DescriptorRole::BdaPagetable || binding.role == DescriptorRole::FaultBuffer; })) result.bdaAbiVersion = 0u;
    result.pushConstants = std::move(bindings.pushConstants);
    if (!artifact.stencilReferences.empty()) {
        if (artifact.stencilReferenceDword * sizeof(std::uint32_t) != result.pushConstants.size() || result.pushConstants.size() + sizeof(std::uint32_t) > request.layout.pushConstantSizeBytes) throw std::runtime_error("stencil replay push constant lies outside the stage range");
        result.pushConstants.resize(result.pushConstants.size() + sizeof(std::uint32_t));
    }
    const auto& layout = variant.bindings.layout;
    if (artifact.registrationModule && request.shader.stage != ShaderStage::Fragment && layout.runtimeStateDwords != 0u) {
        const auto first = layout.RuntimeStateDword() + RuntimeAbi::StateVertexSelectors;
        if (layout.runtimeStateDwords > inputs.runtimeSelectors.size()) throw std::runtime_error("vertex selector state exceeds its capacity");
        if (layout.UsesPushData()) {
            if ((first + layout.runtimeStateDwords) * sizeof(std::uint32_t) > result.pushConstants.size()) throw std::runtime_error("vertex selector state lies outside the push data");
            std::memcpy(result.pushConstants.data() + first * sizeof(std::uint32_t), inputs.runtimeSelectors.data(), layout.runtimeStateDwords * sizeof(std::uint32_t));
        } else {
            const auto data = std::ranges::find_if(result.bindings, [](const DescriptorBinding& binding) { return binding.role == DescriptorRole::ShaderData; });
            if (data == result.bindings.end() || first + layout.runtimeStateDwords > data->guestDescriptor.size()) throw std::runtime_error("vertex selector state lies outside the shader data");
            std::copy_n(inputs.runtimeSelectors.begin(), layout.runtimeStateDwords, data->guestDescriptor.begin() + first);
        }
    }
    result.shaderDataUserWords = std::move(bindings.shaderDataUserWords);
    result.bufferBaseSlots = std::move(bindings.bufferBaseSlots);
    result.poisonedSrtReads = static_cast<std::uint32_t>(snapshot.srtPoison.size()) + snapshot.nullRootReads;
    if (!tables.map.empty()) {
        result.imageTablePoison.reserve(tables.poison.size());
        for (const auto& poison : tables.poison) result.imageTablePoison.push_back({poison.words, poison.dwordCount, poison.resource, static_cast<std::uint32_t>(poison.reason)});
        result.imageTableRanges = snapshot.tables.ranges;
        result.imageTableReadRanges.reserve(snapshot.tables.tables.size());
        for (const auto& table : snapshot.tables.tables) result.imageTableReadRanges.push_back(table.readRanges.Copy());
        result.imageTableShader = snapshot.tables.shader;
        result.imageTableFaults = tables.faults;
    }
    markSplit(MaterializeSplit::Tail);
    return result;
}

// `programs` takes the program from the program memo (APS5_VERIFY_PROGRAM_SPLIT compares a sample
// with a full materialization and keeps the full one on a difference); without it every call
// builds its own program, as before the split.
RecompileResult materializeResult(const CompiledVariant& variant, const RecompileRequest& request, const ResourceSnapshot& captured, bool registrationDefaults = false, bool programs = false) {
    ResourceSnapshot merged;
    const auto& snapshot = mergeTargetImage(variant, request, captured, registrationDefaults, merged);
    struct ProgramInputsStorage {};
    auto& inputs = HostThreadLocal<ProgramInputs, ProgramInputsStorage>();
    readProgramInputs(variant, request, snapshot, registrationDefaults, inputs);
    if (!programs) {
        const auto program = buildProgram(variant, request, snapshot, inputs);
        markSplit(MaterializeSplit::Program);
        return fillData(*program, variant, request, snapshot, inputs);
    }
    const auto program = memoizedProgram(variant, request, snapshot, inputs);
    markSplit(MaterializeSplit::Program);
    auto result = fillData(*program, variant, request, snapshot, inputs);
    result.programPart = program;
    if (ProgramSplitVerifySampled()) {
        auto full = materializeResult(variant, request, captured, registrationDefaults);
        const auto mismatch = ResultMismatch(result, full);
        NoteProgramSplitVerify(!mismatch.empty(), mismatch);
        if (!mismatch.empty()) return full;
    }
    return result;
}

bool sameLayout(const BindingLayout& left, const BindingLayout& right) {
    return left.descriptorSet == right.descriptorSet && left.firstBinding == right.firstBinding && left.pushConstantOffsetBytes == right.pushConstantOffsetBytes && left.pushConstantSizeBytes == right.pushConstantSizeBytes;
}

}

SharedSpirv InternSpirv(std::vector<std::uint32_t> words) {
    SharedSpirv spirv(std::move(words));
    const auto hash = spirv.ContentHash();
    static std::mutex mutex;
    static std::unordered_map<std::uint64_t, std::vector<SharedSpirv>> interned;
    std::lock_guard lock(mutex);
    auto& candidates = interned[hash.low ^ (hash.high * 0x9e3779b97f4a7c15ull)];
    for (const auto& candidate : candidates) {
        if (candidate.ContentHash() == hash && candidate == spirv) return candidate;
    }
    candidates.push_back(spirv);
    return spirv;
}

MeshOutputBudget MeshBarycentricOutputBudget(const MeshConfiguration& mesh, const MeshTargetLimits& limits, const MeshOutputUse& use) {
    const auto alignUp = [](std::uint64_t value, std::uint64_t granularity) { return granularity <= 1u ? value : (value + granularity - 1u) / granularity * granularity; };
    MeshOutputBudget budget;
    budget.vertexLocations = 1u + use.flatParameters + use.clipVaryings + (use.clipDistances + 3u) / 4u + (use.cullDistances + 3u) / 4u;
    budget.primitiveLocations = 3u + 3u * use.perVertexParameters + 1u + (use.layer ? 1u : 0u);
    budget.components = 4ull * (budget.vertexLocations + budget.primitiveLocations);
    budget.bytes = 16ull * (budget.vertexLocations * alignUp(mesh.maxVertices, limits.outputPerVertexGranularity) + budget.primitiveLocations * alignUp(mesh.maxPrimitives, limits.outputPerPrimitiveGranularity));
    return budget;
}

std::optional<MeshBarycentricLayout> MeshBarycentricLayoutOf(const RecompileResult& fragment) {
    if (!fragment.barycentricEmulation.active) return std::nullopt;
    MeshBarycentricLayout layout;
    layout.emulation = fragment.barycentricEmulation;
    for (const auto& parameter : fragment.fragmentParameters) {
        layout.parameters.push_back({parameter.location, parameter.sourceLocation, false, parameter.perVertex});
    }
    return layout;
}

struct PreparedInterface {
    static CompiledVariant Compile(SourceEntry& source, const RecompileRequest& request, ShaderPreparationContext& preparation, std::exception_ptr* emissionFailure) {
        if (preparation.interfaceSource.get() != &source || preparation.interfaceProgram->Resources().shaderHash != request.shader.codeAddress) {
            preparation.interfaceSource.reset();
            preparation.interfaceProgram.reset();
            auto program = source.program != nullptr ? std::move(source.program) : std::make_unique<IrProgram>(PrepareResourceProgram(request, &preparation));
            recordEmissionFailure(emissionFailure, [&] { prepareInterface(request, *program); });
            preparation.interfaceSource = source.shared_from_this();
            preparation.interfaceProgram = std::move(program);
        }
        auto& program = *preparation.interfaceProgram;
        return recordEmissionFailure(emissionFailure, [&] {
            auto variant = emitVariant(request, program);
            variant.info = program.CompiledInfo();
            return variant;
        });
    }
};

namespace {

std::shared_ptr<const CompiledVariant> findOrCompileVariant(SourceEntry& source, const RecompileRequest& request, bool& cacheHit, ShaderPreparationContext* preparation = nullptr) {
    for (const auto& candidate : source.variants) {
        if (sameLayout(candidate->layout, request.layout)) {
            cacheHit = true;
            return candidate;
        }
    }
    for (const auto& failure : source.emissionFailures) {
        if (failure.codeAddress == request.shader.codeAddress && sameLayout(failure.layout, request.layout)) std::rethrow_exception(failure.failure);
    }
    cacheHit = false;
    const bool disk = ShaderDiskCache::Enabled() && !DebugProbeActive();
    std::vector<std::byte> diskKey;
    std::shared_ptr<const CompiledVariant> variant;
    if (disk) {
        ShaderDiskCache::BuildKey(request, HostSubgroupSize(request), diskKey);
        CompiledVariant loaded;
        if (ShaderDiskCache::Load(diskKey, loaded)) {
            loaded.layout = request.layout;
            loaded.artifact.variantId = nextVariantId();
            variant = std::make_shared<const CompiledVariant>(std::move(loaded));
        }
    }
    if (variant == nullptr) {
        static const bool layoutPair = std::getenv("APS5_NO_PERF_LAYOUT_PAIR") == nullptr;
        std::exception_ptr emissionFailure;
        try {
            if (layoutPair && preparation != nullptr) {
                variant = std::make_shared<const CompiledVariant>(PreparedInterface::Compile(source, request, *preparation, &emissionFailure));
            } else {
                auto program = source.program != nullptr ? std::move(*source.program) : PrepareResourceProgram(request, preparation);
                source.program.reset();
                variant = std::make_shared<const CompiledVariant>(compileVariant(request, std::move(program), &emissionFailure));
            }
        } catch (...) {
            if (emissionFailure != nullptr && FailureMemo()) source.emissionFailures.push_back({request.shader.codeAddress, request.layout, emissionFailure});
            throw;
        }
        if (disk) {
            if (auto persisted = ShaderDiskCache::Persist(diskKey, *variant)) variant = std::move(persisted);
            else ShaderDiskCache::Store(std::move(diskKey), variant);
        }
    }
    source.program.reset();
    source.variants.push_back(variant);
    return variant;
}

RecompileResult materializeVariant(SourceEntry& source, const RecompileRequest& request, const ResourceSnapshot& snapshot) {
    std::shared_ptr<const CompiledVariant> variant;
    bool cacheHit = false;
    {
        std::lock_guard lock(source.mutex);
        variant = findOrCompileVariant(source, request, cacheHit);
    }
    auto result = materializeResult(*variant, request, snapshot);
    result.cacheHit = cacheHit;
    return result;
}

RecompileResult recompileOnce(const RecompileRequest& request) {
    static_cast<void>(RequestInputInfo(request));
    RequestMemoryView memory(request.context.memory);
    const auto runtime = memory.MakeRuntime(request.context.userData, request.shader.codeAddress);
    ResourceSnapshot snapshot;
    constexpr ResourceMaterializer materializer;
    if (!request.useCache) {
        auto program = PrepareResourceProgram(request);
        const auto plan = materializer.ExtractPlan(program);
        materializer.Materialize(plan, runtime, snapshot);
        const auto variant = compileVariant(request, std::move(program));
        return materializeResult(variant, request, snapshot);
    }
    const auto source = getSource(request);
    materializer.Materialize(*acquirePlan(request, source, true), runtime, snapshot);
    return materializeVariant(*source, request, snapshot);
}

RecompileResult RecompileImpl(const RecompileRequest& request) {
    try {
        return recompileOnce(request);
    } catch (const ConvertedSampleRequired&) {
        if (request.context.convertedSamples) throw;
    }
    auto converted = request;
    converted.context.convertedSamples = true;
    return recompileOnce(converted);
}

// APS5_NO_RESULT_MEMO=1: every Recompile(request, capture) materializes its own result as before.
bool ResultMemo() {
    static const bool resultMemo = std::getenv("APS5_NO_RESULT_MEMO") == nullptr;
    return resultMemo;
}

ResultMemoEntry* memoSet(SourceEntry& source, std::uint64_t index) {
    if (source.memo == nullptr) return nullptr;
    return source.memo->data() + static_cast<std::size_t>((index ^ (index >> 32u)) % ResultMemoSets) * ResultMemoWays;
}

bool MemoUserWordPatch() {
    static const bool patch = std::getenv("APS5_NO_MEMO_USER_WORD_PATCH") == nullptr;
    return patch;
}

bool VerifyMemoUserWordPatch() {
    static const bool verify = std::getenv("APS5_VERIFY_MEMO_USER_WORD_PATCH") != nullptr;
    return verify;
}

std::atomic<std::uint64_t>& memoUserWordPatches() {
    static std::atomic<std::uint64_t> patches{0};
    return patches;
}

struct ResultMemoCounters {
    std::atomic<std::uint64_t> hits{0}, misses{0}, evictions{0}, populateNanoseconds{0}, patched{0};
    std::atomic<std::uint64_t> hashNanoseconds{0}, lookupWaitNanoseconds{0}, lookupHeldNanoseconds{0}, insertWaitNanoseconds{0}, insertHeldNanoseconds{0}, releaseNanoseconds{0};
    std::atomic<std::int64_t> lastReport{0};
};

ResultMemoCounters& resultMemoCounters() {
    static ResultMemoCounters counters;
    return counters;
}

// APS5_VERIFY_PROGRAM_SPLIT: 0 when unset, else the program-split results per check (a value
// that is not a positive number checks one in 64).
std::uint64_t programSplitVerifyEvery() {
    static const std::uint64_t every = [] {
        const char* text = std::getenv("APS5_VERIFY_PROGRAM_SPLIT");
        if (text == nullptr) return std::uint64_t{0};
        const auto value = std::strtoull(text, nullptr, 10);
        return value != 0 ? static_cast<std::uint64_t>(value) : std::uint64_t{64};
    }();
    return every;
}

// The bindings' words and metadata of two results, in consumer terms.
std::string_view bindingMismatch(const DescriptorBinding& left, const DescriptorBinding& right) {
    if (left.kind != right.kind || left.role != right.role || left.descriptorSet != right.descriptorSet || left.binding != right.binding || left.count != right.count || left.readOnly != right.readOnly || left.imageShape != right.imageShape) return "binding layout";
    if (left.guestDescriptor != right.guestDescriptor) return "binding guest words";
    const bool sameMetadata = left.samplerDepthCompare == right.samplerDepthCompare && left.imageWritten == right.imageWritten && left.imageDepthCompare == right.imageDepthCompare && left.imageAtomic == right.imageAtomic && left.imageAtomic64 == right.imageAtomic64 && left.imageShapes == right.imageShapes &&
        left.bufferAtomic == right.bufferAtomic && left.bufferWritten == right.bufferWritten && left.samplerUnnormalized == right.samplerUnnormalized && left.samplerIntegerBorder == right.samplerIntegerBorder && left.samplerCoordinatesIgnored == right.samplerCoordinatesIgnored &&
        left.imageUnnormalized == right.imageUnnormalized && left.imageUnnormalizedBaseLevel == right.imageUnnormalizedBaseLevel && left.imageSamplers == right.imageSamplers && left.bufferRead == right.bufferRead && left.bufferExtent == right.bufferExtent;
    return sameMetadata ? std::string_view{} : std::string_view{"binding metadata"};
}

struct ResultMemoMissCounters {
    std::atomic<std::uint64_t> firstSeen{0}, unchanged{0}, userDataOnly{0}, basesOnly{0}, basesAndSrtOnly{0}, vertexBasesOnly{0}, evictedReleases{0}, evictedLastReferences{0};
    std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(ResultMemoMissComponent::Count)> byComponent{};
};

ResultMemoMissCounters& resultMemoMissCounters() {
    static ResultMemoMissCounters counters;
    return counters;
}

std::uint64_t elapsedNanoseconds(std::chrono::steady_clock::time_point from, std::chrono::steady_clock::time_point to) {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(to - from).count());
}

class MemoLockSpan {
public:
    MemoLockSpan(bool profile, std::atomic<std::uint64_t>& wait, std::atomic<std::uint64_t>& held) : profile(profile), wait(wait), held(held), started(profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{}) {}
    MemoLockSpan(const MemoLockSpan&) = delete;
    MemoLockSpan& operator=(const MemoLockSpan&) = delete;
    ~MemoLockSpan() {
        if (profile) held.fetch_add(elapsedNanoseconds(locked, std::chrono::steady_clock::now()), std::memory_order_relaxed);
    }
    void Locked() {
        if (!profile) return;
        locked = std::chrono::steady_clock::now();
        wait.fetch_add(elapsedNanoseconds(started, locked), std::memory_order_relaxed);
    }

private:
    bool profile;
    std::atomic<std::uint64_t>& wait;
    std::atomic<std::uint64_t>& held;
    std::chrono::steady_clock::time_point started;
    std::chrono::steady_clock::time_point locked{};
};

constexpr std::uint32_t componentBit(ResultMemoMissComponent component) {
    return 1u << static_cast<std::uint32_t>(component);
}

void classifyMemoMiss(SourceEntry& source, std::uint64_t variantId, const SnapshotParts& parts) {
    auto& counters = resultMemoMissCounters();
    const auto [newest, inserted] = source.memoNewestParts.try_emplace(variantId, parts);
    if (inserted) {
        counters.firstSeen.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    std::uint32_t changed = 0;
    for (std::size_t part = 0; part < parts.size(); ++part) {
        if (parts[part] == newest->second[part]) continue;
        changed |= 1u << part;
        counters.byComponent[part].fetch_add(1, std::memory_order_relaxed);
    }
    newest->second = parts;
    constexpr auto bases = componentBit(ResultMemoMissComponent::BufferBase) | componentBit(ResultMemoMissComponent::ImageBase);
    constexpr auto userData = componentBit(ResultMemoMissComponent::UserData);
    constexpr auto srt = componentBit(ResultMemoMissComponent::FlattenedSrt);
    constexpr auto vertexBases = componentBit(ResultMemoMissComponent::VertexBase);
    if (changed == 0) counters.unchanged.fetch_add(1, std::memory_order_relaxed);
    else if (changed == userData) counters.userDataOnly.fetch_add(1, std::memory_order_relaxed);
    else if ((changed & bases) != 0 && (changed & ~(bases | userData)) == 0) counters.basesOnly.fetch_add(1, std::memory_order_relaxed);
    else if ((changed & bases) != 0 && (changed & srt) != 0 && (changed & ~(bases | userData | srt)) == 0) counters.basesAndSrtOnly.fetch_add(1, std::memory_order_relaxed);
    else if ((changed & vertexBases) != 0 && (changed & ~(bases | vertexBases | userData | srt)) == 0) counters.vertexBasesOnly.fetch_add(1, std::memory_order_relaxed);
}

void releaseEvicted(std::shared_ptr<const RecompileResult> evicted) {
    auto& misses = resultMemoMissCounters();
    misses.evictedReleases.fetch_add(1, std::memory_order_relaxed);
    if (evicted.use_count() == 1) misses.evictedLastReferences.fetch_add(1, std::memory_order_relaxed);
    const auto started = std::chrono::steady_clock::now();
    evicted.reset();
    resultMemoCounters().releaseNanoseconds.fetch_add(elapsedNanoseconds(started, std::chrono::steady_clock::now()), std::memory_order_relaxed);
}

void reportResultMemoSteps(std::uint64_t calls, std::uint64_t misses) {
    auto& counters = resultMemoCounters();
    static ResultMemoMissCauses reported;
    const auto now = ResultMemoMisses();
    const auto perCall = [&](std::atomic<std::uint64_t>& total, std::uint64_t count) { return count != 0 ? static_cast<double>(total.exchange(0, std::memory_order_relaxed)) / 1000.0 / static_cast<double>(count) : 0.0; };
    const auto hash = perCall(counters.hashNanoseconds, calls);
    const auto lookupWait = perCall(counters.lookupWaitNanoseconds, calls);
    const auto lookupHeld = perCall(counters.lookupHeldNanoseconds, calls);
    const auto insertWait = perCall(counters.insertWaitNanoseconds, misses);
    const auto insertHeld = perCall(counters.insertHeldNanoseconds, misses);
    const auto releases = now.evictedReleases - reported.evictedReleases;
    const auto release = perCall(counters.releaseNanoseconds, releases);
    const auto delta = [&](std::size_t component) { return static_cast<unsigned long long>(now.byComponent[component] - reported.byComponent[component]); };
    using Component = ResultMemoMissComponent;
    std::fprintf(stderr, "[recompile] result memo steps (10 s): us per call: hash %.2f, lookup lock wait %.2f / held %.2f; us per miss: insert lock wait %.2f / held %.2f; evicted results released after the lock %llu (%llu last references), %.2f us each; misses against the newest snapshot of the variant: first seen %llu, unchanged (evicted) %llu, user data only %llu, descriptor bases only %llu, bases and flattened SRT only %llu, vertex fetch bases with descriptor bases and flattened SRT only %llu; parts changed: buffer bases %llu, other buffer words %llu, image bases %llu, other image words %llu, samplers %llu, flattened SRT %llu, tables %llu, vertex fetch bases %llu, other vertex fetch words %llu, pixel and layout state %llu, other %llu, user data %llu\n",
        hash, lookupWait, lookupHeld, insertWait, insertHeld, static_cast<unsigned long long>(releases), static_cast<unsigned long long>(now.evictedLastReferences - reported.evictedLastReferences), release,
        static_cast<unsigned long long>(now.firstSeen - reported.firstSeen), static_cast<unsigned long long>(now.unchanged - reported.unchanged), static_cast<unsigned long long>(now.userDataOnly - reported.userDataOnly), static_cast<unsigned long long>(now.basesOnly - reported.basesOnly), static_cast<unsigned long long>(now.basesAndSrtOnly - reported.basesAndSrtOnly), static_cast<unsigned long long>(now.vertexBasesOnly - reported.vertexBasesOnly),
        delta(static_cast<std::size_t>(Component::BufferBase)), delta(static_cast<std::size_t>(Component::BufferOther)), delta(static_cast<std::size_t>(Component::ImageBase)), delta(static_cast<std::size_t>(Component::ImageOther)), delta(static_cast<std::size_t>(Component::Sampler)), delta(static_cast<std::size_t>(Component::FlattenedSrt)), delta(static_cast<std::size_t>(Component::Tables)), delta(static_cast<std::size_t>(Component::VertexBase)), delta(static_cast<std::size_t>(Component::VertexOther)), delta(static_cast<std::size_t>(Component::PixelAndLayout)), delta(static_cast<std::size_t>(Component::Other)), delta(static_cast<std::size_t>(Component::UserData)));
    reported = now;
}

void reportResultMemo() {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    if (!profile) return;
    auto& counters = resultMemoCounters();
    const auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    auto last = counters.lastReport.load(std::memory_order_relaxed);
    if (last == 0) {
        counters.lastReport.compare_exchange_strong(last, now, std::memory_order_relaxed);
        return;
    }
    if (now - last < 10'000'000'000ll || !counters.lastReport.compare_exchange_strong(last, now, std::memory_order_relaxed)) return;
    // Deltas: the totals stay cumulative for ResultMemoCounts. Only the thread that won the
    // lastReport exchange gets here.
    static ResultMemoTotals reported;
    const auto totals = ResultMemoCounts();
    const auto hits = totals.hits - reported.hits;
    const auto misses = totals.misses - reported.misses;
    const auto evictions = totals.evictions - reported.evictions;
    reported = totals;
    const auto populate = counters.populateNanoseconds.exchange(0, std::memory_order_relaxed);
    const auto patched = counters.patched.exchange(0, std::memory_order_relaxed);
    std::fprintf(stderr, "[recompile] result memo (10 s): %llu hits, %llu misses (%.1f%% hits), Populate %.1f us per miss / %.1f ms in total, %llu evictions; %llu misses patched from a result that differs only in forwarded user words\n", static_cast<unsigned long long>(hits), static_cast<unsigned long long>(misses), hits + misses != 0 ? 100.0 * static_cast<double>(hits) / static_cast<double>(hits + misses) : 0.0, misses != 0 ? static_cast<double>(populate) / 1000.0 / static_cast<double>(misses) : 0.0, static_cast<double>(populate) / 1e6, static_cast<unsigned long long>(evictions), static_cast<unsigned long long>(patched));
    reportResultMemoSteps(hits + misses, misses);
}

// Everything materializeResult reads besides the variant: the snapshot (the descriptor words, the
// flattened SRT, the user data, the uniform fill) and, for the vertex family, the V# table the
// attributes are resolved from.
constexpr std::uint64_t SnapshotHashBasis = 0xcbf29ce484222325ull;
constexpr std::uint64_t SnapshotHashPrime = 0x100000001b3ull;
constexpr std::uint32_t BufferBaseHighMask = 0xffffu;
constexpr std::uint32_t ImageBaseHighMask = 0xffu;

std::uint64_t snapshotHash(const RecompileRequest& request, const ResourceSnapshot& snapshot, std::uint64_t* shape, SnapshotParts* parts = nullptr) {
    using Component = ResultMemoMissComponent;
    std::uint64_t hash = SnapshotHashBasis;
    auto current = Component::Other;
    const auto mixPart = [&](Component component, std::uint64_t value) {
        if (parts == nullptr) return;
        auto& part = (*parts)[static_cast<std::size_t>(component)];
        part ^= value;
        part *= SnapshotHashPrime;
    };
    const auto mixHash = [&](std::uint64_t value) {
        hash ^= value;
        hash *= SnapshotHashPrime;
    };
    const auto mix = [&](std::uint64_t value) {
        mixHash(value);
        mixPart(current, value);
    };
    const auto mixWords = [&](std::span<const std::uint32_t> words) {
        mix(words.size());
        for (const auto word : words) mix(word);
    };
    const auto mixDescriptors = [&](const std::vector<DescriptorValue>& values, Component base, Component other, std::uint32_t baseHighMask) {
        current = other;
        mix(values.size());
        for (const auto& value : values) {
            mix(value.dwordCount);
            mix(value.availableDwordMask);
            for (std::uint32_t i = 0; i < value.dwordCount && i < value.dwords.size(); ++i) {
                const auto word = value.dwords[i];
                mixHash(word);
                if (i == 0) {
                    mixPart(base, word);
                } else if (i == 1) {
                    mixPart(base, word & baseHighMask);
                    mixPart(other, word & ~baseHighMask);
                } else {
                    mixPart(other, word);
                }
            }
        }
    };
    if (parts != nullptr) parts->fill(SnapshotHashBasis);
    mixDescriptors(snapshot.buffers, Component::BufferBase, Component::BufferOther, BufferBaseHighMask);
    mixDescriptors(snapshot.images, Component::ImageBase, Component::ImageOther, ImageBaseHighMask);
    mixDescriptors(snapshot.samplers, Component::Sampler, Component::Sampler, ~0u);
    current = Component::FlattenedSrt;
    mixWords(snapshot.flattenedSrt);
    current = Component::Other;
    mix(static_cast<std::uint64_t>(snapshot.uniformFill.kind));
    mix(snapshot.uniformFill.resource);
    for (const auto stride : snapshot.uniformFill.groupStride) mix(stride);
    mix(snapshot.uniformFill.words);
    mix(snapshot.uniformFill.value);
    current = Component::Tables;
    const auto& tables = snapshot.tables;
    mix(tables.shader);
    mix(tables.tables.size());
    for (const auto& table : tables.tables) {
        mix(table.base);
        mix(table.size);
        mix(table.records);
        mix(table.keys);
        mix(table.fault);
        mix(table.outside ? 1u : 0u);
        mix(table.codes.Hash());
        mix(table.words.Hash());
        mix(table.readRanges.Hash());
    }
    mix(tables.ranges.size());
    for (const auto& [base, size] : tables.ranges) {
        mix(base);
        mix(size);
    }
    current = Component::PixelAndLayout;
    for (const auto threads : partialThreads(request)) mix(threads);
    mix(request.layout.pushConstantOffsetBytes);
    if (request.context.pixel) {
        for (const auto mapping : request.context.pixel->targetExportMapping) mix(mapping);
        for (const auto packing : request.context.pixel->targetExportPacking) mix(static_cast<std::uint64_t>(packing));
        mix(request.context.pixel->dualSourceBlend);
        mix(request.context.pixel->reversedBlend);
        mix(static_cast<std::uint64_t>(request.context.pixel->rasterPrimitive));
        mix(request.context.pixel->layerExported);
        const auto& merge = request.context.pixel->targetMerge;
        mix(merge.target);
        mix(merge.mask);
        mix(merge.samples);
        for (const auto word : merge.descriptor) mix(word);
    }
    current = Component::VertexOther;
    if (request.context.vertex) {
        const auto& vertex = *request.context.vertex;
        const auto count = std::min<std::uint32_t>(vertex.resourcesNum, ShaderVertexStageInfo::MaxResources);
        mix(count);
        for (std::uint32_t i = 0; i < count; ++i) {
            const auto& fields = vertex.resources[i].fields;
            for (std::size_t f = 0; f < fields.size(); ++f) {
                mixHash(fields[f]);
                if (f == 0) {
                    mixPart(Component::VertexBase, fields[f]);
                } else if (f == 1) {
                    mixPart(Component::VertexBase, fields[f] & BufferBaseHighMask);
                    mixPart(Component::VertexOther, fields[f] & ~BufferBaseHighMask);
                } else {
                    mixPart(Component::VertexOther, fields[f]);
                }
            }
            mix(vertex.resourcesDst[i].fetchIndex);
        }
    } else {
        mix(1ull << 32u);
    }
    current = Component::Other;
    mix(snapshot.srtPoison.size());
    mix(snapshot.nullRootReads);
    if (shape != nullptr) *shape = hash;
    current = Component::UserData;
    mixWords(snapshot.userData);
    return hash;
}

std::shared_ptr<RecompileResult> patchForwardedUserWords(const RecompileResult& donor, const CompiledVariant& variant, const ResourceSnapshot& snapshot) {
    const auto& registers = variant.bindings.layout.userDataRegisters;
    const auto base = variant.info.userDataBase;
    if (donor.shaderDataUserWords.size() != registers.size()) return nullptr;
    auto result = std::make_shared<RecompileResult>(donor);
    const auto shaderData = std::ranges::find_if(result->bindings, [](const DescriptorBinding& binding) { return binding.role == DescriptorRole::ShaderData; });
    auto* data = shaderData != result->bindings.end() ? &shaderData->guestDescriptor : nullptr;
    const bool pushData = variant.bindings.layout.UsesPushData();
    for (std::size_t i = 0; i < registers.size(); ++i) {
        if (registers[i] < base || registers[i] - base >= snapshot.userData.size()) return nullptr;
        const auto word = snapshot.userData[registers[i] - base];
        if (pushData && (i + 1) * sizeof(std::uint32_t) <= result->pushConstants.size()) std::memcpy(result->pushConstants.data() + i * sizeof(std::uint32_t), &word, sizeof(word));
        if (data != nullptr && i < data->size()) (*data)[i] = word;
    }
    return result;
}

void verifyPatchedResult(const RecompileResult& patched, const RecompileResult& full) {
    bool same = patched.pushConstants == full.pushConstants && patched.bindings.size() == full.bindings.size() && patched.variantId == full.variantId && patched.specializationId == full.specializationId;
    for (std::size_t b = 0; same && b < full.bindings.size(); ++b) same = patched.bindings[b].role == full.bindings[b].role && patched.bindings[b].guestDescriptor == full.bindings[b].guestDescriptor;
    if (!same) throw std::runtime_error("APS5_VERIFY_MEMO_USER_WORD_PATCH: a result patched with the forwarded user words differs from its full materialization");
}

// The memo'd result of `source`'s variant for the snapshot (design13 R5): a hit returns the shared
// object, a miss materializes outside the source mutex and inserts (a concurrent miss's object is
// as good). `memoHit` reports the hit.
std::shared_ptr<const RecompileResult> materializePreparedMemoized(SourceEntry& source, const std::shared_ptr<const CompiledVariant>& variant, const RecompileRequest& request, const ResourceSnapshot& snapshot, bool cacheHit, bool* memoHit) {
    const SplitScope split;
    if (ProgramSplit()) {
        auto result = std::make_shared<RecompileResult>(materializeResult(*variant, request, snapshot, false, true));
        result->cacheHit = cacheHit;
        return result;
    }
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    auto& counters = resultMemoCounters();
    std::uint64_t shape = 0;
    SnapshotParts parts{};
    const auto hashStarted = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const auto hash = snapshotHash(request, snapshot, &shape, profile ? &parts : nullptr);
    if (profile) counters.hashNanoseconds.fetch_add(elapsedNanoseconds(hashStarted, std::chrono::steady_clock::now()), std::memory_order_relaxed);
    markSplit(MaterializeSplit::Hash);
    const bool patching = MemoUserWordPatch() && !variant->bindings.layout.userDataRegisters.empty();
    std::uint64_t index = 0;
    std::uint64_t shapeIndex = 0;
    std::shared_ptr<const RecompileResult> donor;
    {
        MemoLockSpan span(profile, counters.lookupWaitNanoseconds, counters.lookupHeldNanoseconds);
        std::lock_guard lock(source.mutex);
        span.Locked();
        index = (variant->artifact.variantId * 0x9e3779b97f4a7c15ull) ^ hash;
        if (auto* set = memoSet(source, index)) {
            for (std::size_t way = 0; way < ResultMemoWays; ++way) {
                auto& entry = set[way];
                if (entry.result == nullptr || entry.variantId != variant->artifact.variantId || entry.hash != hash) continue;
                entry.used = ++source.memoClock;
                counters.hits.fetch_add(1, std::memory_order_relaxed);
                if (memoHit != nullptr) *memoHit = true;
                reportResultMemo();
                markSplit(MaterializeSplit::Lookup);
                return entry.result;
            }
        }
        if (patching) {
            shapeIndex = (variant->artifact.variantId * 0xc2b2ae3d27d4eb4full) ^ shape;
            const auto similar = source.memoShapes.find(shapeIndex);
            if (similar != source.memoShapes.end() && similar->second->variantId == variant->artifact.variantId && similar->second->shape == shape) donor = similar->second->result;
        }
    }
    markSplit(MaterializeSplit::Lookup);
    const auto started = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    std::shared_ptr<RecompileResult> result = donor != nullptr ? patchForwardedUserWords(*donor, *variant, snapshot) : nullptr;
    markSplit(MaterializeSplit::Patch);
    if (result != nullptr) {
        memoUserWordPatches().fetch_add(1, std::memory_order_relaxed);
        counters.patched.fetch_add(1, std::memory_order_relaxed);
        if (VerifyMemoUserWordPatch()) verifyPatchedResult(*result, materializeResult(*variant, request, snapshot));
    } else {
        result = std::make_shared<RecompileResult>(materializeResult(*variant, request, snapshot));
        markSplit(MaterializeSplit::Tail);
    }
    result->cacheHit = cacheHit;
    if (profile) counters.populateNanoseconds.fetch_add(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count()), std::memory_order_relaxed);
    counters.misses.fetch_add(1, std::memory_order_relaxed);
    std::shared_ptr<const RecompileResult> shared = std::move(result);
    std::shared_ptr<const RecompileResult> evicted;
    {
        MemoLockSpan span(profile, counters.insertWaitNanoseconds, counters.insertHeldNanoseconds);
        std::lock_guard lock(source.mutex);
        span.Locked();
        if (profile) classifyMemoMiss(source, variant->artifact.variantId, parts);
        if (source.memo == nullptr) source.memo = std::make_unique<ResultMemoTable>();
        auto* set = memoSet(source, index);
        auto* victim = set;
        bool present = false;
        for (std::size_t way = 0; way < ResultMemoWays && !present; ++way) {
            auto& entry = set[way];
            if (entry.result != nullptr && entry.variantId == variant->artifact.variantId && entry.hash == hash) {
                entry.used = ++source.memoClock;
                shared = entry.result;
                present = true;
            } else if (entry.used < victim->used) {
                victim = &entry;
            }
        }
        if (!present) {
            if (victim->result != nullptr) {
                counters.evictions.fetch_add(1, std::memory_order_relaxed);
                evicted = std::move(victim->result);
                const auto staleShape = source.memoShapes.find((victim->variantId * 0xc2b2ae3d27d4eb4full) ^ victim->shape);
                if (staleShape != source.memoShapes.end() && staleShape->second == victim) source.memoShapes.erase(staleShape);
            }
            *victim = {variant->artifact.variantId, hash, shared, ++source.memoClock, shape};
            if (patching) source.memoShapes[shapeIndex] = victim;
        }
    }
    markSplit(MaterializeSplit::Insert);
    if (profile && evicted != nullptr) releaseEvicted(std::move(evicted));
    evicted.reset();
    markSplit(MaterializeSplit::Release);
    reportResultMemo();
    return shared;
}

std::shared_ptr<const RecompileResult> materializeMemoized(SourceEntry& source, const RecompileRequest& request, const ResourceSnapshot& snapshot, bool* memoHit) {
    std::shared_ptr<const CompiledVariant> variant;
    bool cacheHit = false;
    {
        std::lock_guard lock(source.mutex);
        variant = findOrCompileVariant(source, request, cacheHit);
    }
    return materializePreparedMemoized(source, variant, request, snapshot, cacheHit, memoHit);
}

// The capture already resolved the source entry (stage input validation included) and materialized
// the request over exactly the words the driver captured, so neither is repeated here.
std::shared_ptr<const RecompileResult> RecompileImpl(const RecompileRequest& request, const ResourceCapture& capture, bool* memoHit) {
    if (!request.useCache || capture.source == nullptr) {
        auto program = PrepareResourceProgram(request);
        const auto variant = compileVariant(request, std::move(program));
        return std::make_shared<const RecompileResult>(materializeResult(variant, request, capture.snapshot));
    }
    if (!ResultMemo()) return std::make_shared<const RecompileResult>(materializeVariant(*capture.source, request, capture.snapshot));
    return materializeMemoized(*capture.source, request, capture.snapshot, memoHit);
}

template <typename Impl>
auto recompileReporting(const RecompileRequest& request, Impl&& impl) -> decltype(impl()) {
    try {
        return impl();
    } catch (const std::exception& e) {
        constexpr auto requestSerializer = RequestSerializer{};
        const auto inputInfo = "\nRecompileRequest:\n" + requestSerializer.Serialize(request);
        throw std::runtime_error(std::string("ShaderRecompiler::Recompile: ") + e.what() + inputInfo);
    } catch (...) {
        throw std::runtime_error("ShaderRecompiler::Recompile: unknown exception");
    }
}

}

std::shared_ptr<const IrResourcePlan> GetResourcePlan(const RecompileRequest& request) {
    static_cast<void>(RequestInputInfo(request));
    if (request.useCache) {
        const auto source = getSource(request);
        return acquirePlan(request, source, false);
    }
    return makeResourcePlan(request);
}

void ReleaseUndrawnResourcePlan(const SourceHandle& handle) {
    releaseUndrawnPlan(handle);
}

namespace {

// The user words the walk read only as dword 0 of buffer descriptors (inline V# bases) become
// SrtUserDataUse::asBufferBase, paired with their buffers. Every word read that way joins asValue
// as before, so a consumer that does not patch the bases keeps the word exact.
void noteBufferBaseWords(ResourceCapture& capture) {
    auto& use = capture.userDataUse;
    if (use.descriptorBases.empty()) return;
    const auto& plan = *capture.plan;
    std::uint64_t read = 0, buffers = 0, others = 0;
    for (const auto& [source, word] : use.descriptorBases) {
        const auto bit = std::uint64_t {1} << word;
        read |= bit;
        const bool shared = std::any_of(plan.info.images.begin(), plan.info.images.end(), [&](const ImageResource& image) { return image.source == source; }) || std::any_of(plan.info.samplers.begin(), plan.info.samplers.end(), [&](const SamplerResource& sampler) { return sampler.source == source; });
        bool found = false;
        for (std::uint32_t buffer = 0; buffer < plan.info.buffers.size() && buffer < plan.materializationSources.size(); ++buffer) {
            if (plan.materializationSources[buffer] != source) continue;
            found = true;
            const std::pair<std::uint32_t, std::uint32_t> base{word, buffer};
            if (std::find(use.bufferBases.begin(), use.bufferBases.end(), base) == use.bufferBases.end()) use.bufferBases.push_back(base);
        }
        (found && !shared ? buffers : others) |= bit;
    }
    use.asBufferBase = buffers & ~others & ~use.asValue & ~use.asBase;
    use.asValue |= read;
    std::erase_if(use.bufferBases, [&](const std::pair<std::uint32_t, std::uint32_t>& base) { return (use.asBufferBase >> base.first & 1u) == 0; });
}

// The capture's materialization; with pure flat slots in the plan the walk's read addresses are
// traced into the capture (ResourceCapture::readTrace).
void materializeCapture(ResourceCapture& capture, const SrtRuntime& runtime) {
    const auto& plan = *capture.plan;
    SrtRuntime traced = runtime;
    traced.userDataUse = &capture.userDataUse;
    if (std::none_of(plan.pureFlatSlots.begin(), plan.pureFlatSlots.end(), [](std::uint8_t pure) { return pure != 0u; })) {
        ResourceMaterializer{}.Materialize(plan, traced, capture.snapshot);
        noteBufferBaseWords(capture);
        return;
    }
    traced.readTrace = &capture.readTrace;
    ResourceMaterializer{}.Materialize(plan, traced, capture.snapshot);
    noteBufferBaseWords(capture);
    auto& other = capture.readTrace.otherReads;
    std::sort(other.begin(), other.end());
    other.erase(std::unique(other.begin(), other.end()), other.end());
}

}

std::shared_ptr<const ResourceCapture> CaptureResources(const RecompileRequest& request, const SrtRuntime& runtime) {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    const auto started = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    // Validates the stage inputs once per request, as GetResourcePlan and Recompile(request) do.
    static_cast<void>(RequestInputInfo(request));
    auto capture = std::make_shared<ResourceCapture>();
    if (request.useCache) {
        capture->source = getSource(request);
        capture->plan = acquirePlan(request, capture->source, true);
    } else {
        capture->plan = makeResourcePlan(request);
    }
    if (profile) capture->sourceNanoseconds = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count());
    materializeCapture(*capture, runtime);
    return capture;
}

std::shared_ptr<const SourceHandle> ResolveSource(const RecompileRequest& request) {
    if (!request.useCache) return nullptr;
    static_cast<void>(RequestInputInfo(request));
    return std::make_shared<const SourceHandle>(SourceHandle{getSource(request)});
}

std::shared_ptr<const SourceHandle> PrepareShader(const RecompileRequest& request) {
    return PrepareShader(request, nullptr);
}

std::shared_ptr<const SourceHandle> PrepareShader(const RecompileRequest& request, ShaderPreparationContext* preparation) {
    return recompileReporting(request, [&]() -> std::shared_ptr<const SourceHandle> {
        static_cast<void>(RequestInputInfo(request));
        auto handle = std::make_shared<SourceHandle>();
        handle->source = getSource(request, preparation, false);
        RecompileCacheKey::BuildInterface(request, handle->staticKey);
        handle->staticKey.push_back(HostSubgroupSize(request));
        bool cacheHit = false;
        std::lock_guard lock(handle->source->mutex);
        handle->artifact = findOrCompileVariant(*handle->source, request, cacheHit, preparation);
        return handle;
    });
}

bool MatchesPreparedShader(const RecompileRequest& request, const SourceHandle& handle) {
    struct PreparedKeyStorage {};
    auto& key = HostThreadLocal<std::vector<std::uint64_t>, PreparedKeyStorage>();
    BuildPreparedShaderKey(request, key);
    return MatchesPreparedShader(request, handle, key);
}

void BuildPreparedShaderKey(const RecompileRequest& request, std::vector<std::uint64_t>& key) {
    RecompileCacheKey::BuildInterface(request, key);
    key.push_back(HostSubgroupSize(request));
}

bool MatchesPreparedShader(const RecompileRequest& request, const SourceHandle& handle, std::span<const std::uint64_t> key) {
    if (handle.source == nullptr || handle.artifact == nullptr) return false;
    const auto& layout = request.layout;
    const auto& prepared = handle.artifact->bindings.layout;
    if (layout.descriptorSet != handle.artifact->layout.descriptorSet || layout.firstBinding != handle.artifact->layout.firstBinding) return false;
    if (layout.pushConstantOffsetBytes % 4u != 0u || layout.pushConstantSizeBytes % 4u != 0u || layout.pushConstantOffsetBytes > NativePushConstantSize || layout.pushConstantSizeBytes > NativePushConstantSize - layout.pushConstantOffsetBytes) return false;
    const auto words = prepared.ShaderDataDwords();
    const bool usesPush = prepared.runtimeImageCount == 0u && words != 0u && words <= layout.pushConstantSizeBytes / 4u;
    if (prepared.UsesPushData() != usesPush) return false;
    if (!handle.artifact->artifact.stencilReferences.empty() && (prepared.stencilReferenceDword >= layout.pushConstantSizeBytes / 4u || prepared.stencilReferenceDword != (usesPush ? words : 0u))) return false;
    if (!std::ranges::equal(key, handle.staticKey) || handle.source->code.size() != request.shader.code.size()) return false;
    return handle.source->code.data() == request.shader.code.data() || std::ranges::equal(handle.source->code, request.shader.code);
}

std::span<const std::uint32_t> GetPreparedCode(const SourceHandle& handle) {
    if (handle.source == nullptr || handle.artifact == nullptr) throw std::runtime_error("ShaderRecompiler: prepared artifact is missing");
    return handle.source->code;
}

PreparedShaderInvocation::PreparedShaderInvocation(const RecompileRequest& request, const std::shared_ptr<const SourceHandle>& handle) : request(request), handle(handle) {
    this->request.context.linkedCallFallbacks = handle->source->linkedCallFallbacks;
}

std::optional<PreparedShaderInvocation> PreparedShaderInvocation::TryCreate(const RecompileRequest& request, const std::shared_ptr<const SourceHandle>& handle) {
    struct PreparedKeyStorage {};
    auto& key = HostThreadLocal<std::vector<std::uint64_t>, PreparedKeyStorage>();
    BuildPreparedShaderKey(request, key);
    return TryCreate(request, handle, key);
}

std::optional<PreparedShaderInvocation> PreparedShaderInvocation::TryCreate(const RecompileRequest& request, const std::shared_ptr<const SourceHandle>& handle, std::span<const std::uint64_t> key) {
    if (handle == nullptr || !MatchesPreparedShader(request, *handle, key)) return std::nullopt;
    if (request.shader.stage != ShaderStage::Compute && request.shader.stage != ShaderStage::Fragment) ValidateRequestInputs(request);
    return PreparedShaderInvocation(request, handle);
}

std::shared_ptr<const ResourceCapture> PreparedShaderInvocation::Capture(const SrtRuntime& runtime) const {
    auto capture = std::make_shared<ResourceCapture>();
    capture->source = handle->source;
    capture->plan = acquirePlan(request, handle->source, true, handle.get());
    materializeCapture(*capture, runtime);
    return capture;
}

std::shared_ptr<const RecompileResult> PreparedShaderInvocation::Materialize(const ResourceCapture& capture) const {
    if (capture.source != handle->source) throw std::runtime_error("ShaderRecompiler: resource capture belongs to another prepared shader");
    if (request.useCache && ResultMemo()) return materializePreparedMemoized(*handle->source, handle->artifact, request, capture.snapshot, true, nullptr);
    auto result = std::make_shared<RecompileResult>(materializeResult(*handle->artifact, request, capture.snapshot));
    result->cacheHit = true;
    return result;
}

const CompiledShaderArtifact& GetPreparedArtifact(const SourceHandle& handle) {
    if (handle.artifact == nullptr) throw std::runtime_error("ShaderRecompiler: prepared artifact is missing");
    return handle.artifact->artifact;
}

std::shared_ptr<const RecompileResult> RegistrationResult(const RecompileRequest& request, const SourceHandle& handle) {
    if (handle.artifact == nullptr || !MatchesPreparedShader(request, handle)) throw std::runtime_error("ShaderRecompiler: prepared artifact does not match the static ABI");
    if (!request.target.registrationModules || !handle.artifact->artifact.registrationModule) return nullptr;
    const auto& info = handle.artifact->info.info;
    if (std::ranges::any_of(info.buffers, [](const BufferResource& buffer) { return buffer.descriptorFormatted; }) || std::ranges::any_of(info.samplers, [](const SamplerResource& sampler) { return FoldsTexelOffsets(sampler); })) return nullptr;
    ResourceSnapshot snapshot;
    snapshot.buffers.assign(info.buffers.size(), {{}, 4u});
    snapshot.images.assign(info.images.size(), {{}, 8u});
    snapshot.samplers.assign(info.samplers.size(), {{}, 4u});
    snapshot.userData.assign(RuntimeAbi::UserDataCapacity, 0u);
    snapshot.flattenedSrt.assign(1u, 0u);
    return std::make_shared<RecompileResult>(materializeResult(*handle.artifact, request, snapshot, true));
}

std::shared_ptr<const RecompileResult> MaterializeShader(const RecompileRequest& request, const ResourceCapture& capture, const SourceHandle& handle) {
    if (!MatchesPreparedShader(request, handle)) throw std::runtime_error("ShaderRecompiler: prepared artifact does not match the static ABI");
    if (capture.source != handle.source) throw std::runtime_error("ShaderRecompiler: resource capture belongs to another prepared shader");
    if (request.useCache && ResultMemo()) return materializePreparedMemoized(*handle.source, handle.artifact, request, capture.snapshot, true, nullptr);
    auto result = std::make_shared<RecompileResult>(materializeResult(*handle.artifact, request, capture.snapshot));
    result->cacheHit = true;
    return result;
}

std::shared_ptr<const ResourceCapture> CaptureResources(const RecompileRequest& request, const SrtRuntime& runtime, const SourceHandle& handle) {
    if (handle.source == nullptr) throw std::runtime_error("ShaderRecompiler: source handle is missing");
    if (handle.artifact != nullptr && !MatchesPreparedShader(request, handle)) throw std::runtime_error("ShaderRecompiler: prepared artifact does not match the static ABI");
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    const auto started = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    // The whole vertex family (Vertex, Local, TC, TE, Mesh) validates V# fields the memo key does not cover.
    if (request.shader.stage != ShaderStage::Compute && request.shader.stage != ShaderStage::Fragment) ValidateRequestInputs(request);
    auto capture = std::make_shared<ResourceCapture>();
    capture->source = handle.source;
    capture->plan = acquirePlan(request, handle.source, true, &handle);
    if (profile) capture->sourceNanoseconds = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count());
    materializeCapture(*capture, runtime);
    return capture;
}

RecompileResult Recompile(const RecompileRequest& request) {
    return recompileReporting(request, [&] { return RecompileImpl(request); });
}

std::shared_ptr<const RecompileResult> Recompile(const RecompileRequest& request, const ResourceCapture& capture, bool* memoHit) {
    if (memoHit != nullptr) *memoHit = false;
    return recompileReporting(request, [&] { return RecompileImpl(request, capture, memoHit); });
}

std::uint64_t ResultMemoUserWordPatches() {
    return memoUserWordPatches().load(std::memory_order_relaxed);
}

ResultMemoTotals ResultMemoCounts() {
    const auto& counters = resultMemoCounters();
    return {counters.hits.load(std::memory_order_relaxed), counters.misses.load(std::memory_order_relaxed), counters.evictions.load(std::memory_order_relaxed)};
}

ProgramKeyCensus::Counts TakeProgramKeyCensus() {
    return programKeyCensus().Take();
}

MaterializeSplit::Counts TakeMaterializeSplit() {
    return materializeSplit().Take();
}

std::string_view ResultMismatch(const RecompileResult& left, const RecompileResult& right) {
    if (left.variantId != right.variantId || left.memoryOffsetDword != right.memoryOffsetDword || left.shaderDataDwords != right.shaderDataDwords || left.imageMetadataDword != right.imageMetadataDword || left.runtimeImageCount != right.runtimeImageCount || left.runtimeImageResources != right.runtimeImageResources ||
        left.registrationModule != right.registrationModule || left.bdaAbiVersion != right.bdaAbiVersion || left.windowLaneGuards != right.windowLaneGuards || left.runtimeAbiVersion != right.runtimeAbiVersion || left.vertexInputs != right.vertexInputs || left.vertexInputPatches != right.vertexInputPatches ||
        left.vertexOffsetSgpr != right.vertexOffsetSgpr || left.instanceOffsetSgpr != right.instanceOffsetSgpr || left.vertexOffsetShared != right.vertexOffsetShared || left.instanceOffsetShared != right.instanceOffsetShared || left.vertexOffsetConflict != right.vertexOffsetConflict || left.instanceOffsetConflict != right.instanceOffsetConflict ||
        left.hostSubgroupSize != right.hostSubgroupSize || left.fullSubgroups != right.fullSubgroups || left.functionLdsBuffer != right.functionLdsBuffer || left.parameterExports != right.parameterExports || left.fragmentParameters != right.fragmentParameters || left.barycentricEmulation != right.barycentricEmulation ||
        left.meshBarycentrics != right.meshBarycentrics || left.stencilReferences != right.stencilReferences || left.stencilReplaySafe != right.stencilReplaySafe || left.bdaWrites != right.bdaWrites || left.stencilReferenceDword != right.stencilReferenceDword || left.workgroupMemoryDwords != right.workgroupMemoryDwords) {
        return "artifact";
    }
    if (left.specializationId != right.specializationId || left.specialization != right.specialization || (left.spirv.data() != right.spirv.data() && left.spirv != right.spirv)) return "module";
    if (left.userDataRegisterReads != right.userDataRegisterReads) return "user data registers";
    if (left.bindings.size() != right.bindings.size()) return "binding layout";
    for (std::size_t b = 0; b < left.bindings.size(); ++b) {
        if (const auto field = bindingMismatch(left.bindings[b], right.bindings[b]); !field.empty()) return field;
    }
    if (left.pushConstants != right.pushConstants || left.readsDepthRemapPush != right.readsDepthRemapPush) return "push constants";
    if (left.shaderDataUserWords != right.shaderDataUserWords || left.bufferBaseSlots != right.bufferBaseSlots) return "shader data slots";
    if (left.vertexAttributes.size() != right.vertexAttributes.size()) return "vertex attributes";
    for (std::size_t a = 0; a < left.vertexAttributes.size(); ++a) {
        const auto& l = left.vertexAttributes[a];
        const auto& r = right.vertexAttributes[a];
        if (l.location != r.location || l.components != r.components || l.resource.fields != r.resource.fields || l.fetchIndex != r.fetchIndex || l.formatComponents != r.formatComponents) return "vertex attributes";
    }
    if (left.poisonedSrtReads != right.poisonedSrtReads || left.imageTableRanges != right.imageTableRanges || left.imageTableReadRanges != right.imageTableReadRanges || left.imageTableShader != right.imageTableShader || left.imageTableFaults != right.imageTableFaults || left.imageTablePoison.size() != right.imageTablePoison.size()) return "image tables";
    for (std::size_t p = 0; p < left.imageTablePoison.size(); ++p) {
        const auto& l = left.imageTablePoison[p];
        const auto& r = right.imageTablePoison[p];
        if (l.words != r.words || l.dwordCount != r.dwordCount || l.resource != r.resource || l.reason != r.reason) return "image tables";
    }
    return {};
}

bool ProgramSplitVerifySampled() {
    const auto every = programSplitVerifyEvery();
    return every != 0 && programMemoCounters().verifyCalls.fetch_add(1, std::memory_order_relaxed) % every == 0;
}

void NoteProgramSplitVerify(bool mismatch, std::string_view what) {
    auto& counters = programMemoCounters();
    counters.verifyChecks.fetch_add(1, std::memory_order_relaxed);
    if (!mismatch) return;
    constexpr std::uint64_t PrintedMismatches = 16;
    if (counters.verifyMismatches.fetch_add(1, std::memory_order_relaxed) < PrintedMismatches) std::fprintf(stderr, "[program-split] APS5_VERIFY_PROGRAM_SPLIT: a split result differs from its full materialization in %.*s\n", static_cast<int>(what.size()), what.data());
}

ProgramMemoTotals ProgramMemoCounts() {
    const auto& counters = programMemoCounters();
    return {counters.hits.load(std::memory_order_relaxed), counters.misses.load(std::memory_order_relaxed), counters.resets.load(std::memory_order_relaxed), counters.verifyChecks.load(std::memory_order_relaxed), counters.verifyMismatches.load(std::memory_order_relaxed)};
}

ResultMemoMissCauses ResultMemoMisses() {
    const auto& counters = resultMemoMissCounters();
    ResultMemoMissCauses causes;
    causes.firstSeen = counters.firstSeen.load(std::memory_order_relaxed);
    causes.unchanged = counters.unchanged.load(std::memory_order_relaxed);
    causes.userDataOnly = counters.userDataOnly.load(std::memory_order_relaxed);
    causes.basesOnly = counters.basesOnly.load(std::memory_order_relaxed);
    causes.basesAndSrtOnly = counters.basesAndSrtOnly.load(std::memory_order_relaxed);
    causes.vertexBasesOnly = counters.vertexBasesOnly.load(std::memory_order_relaxed);
    causes.evictedReleases = counters.evictedReleases.load(std::memory_order_relaxed);
    causes.evictedLastReferences = counters.evictedLastReferences.load(std::memory_order_relaxed);
    for (std::size_t part = 0; part < causes.byComponent.size(); ++part) causes.byComponent[part] = counters.byComponent[part].load(std::memory_order_relaxed);
    return causes;
}

}
