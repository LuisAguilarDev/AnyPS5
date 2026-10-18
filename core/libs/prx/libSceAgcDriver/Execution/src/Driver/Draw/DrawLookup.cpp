#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Diagnostics.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPrepPool.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include <algorithm>
#include <atomic>
#include <bit>
#include <shared_mutex>

namespace AgcDriver::DriverDetail {

std::shared_ptr<DispatchVariant> Driver::dataCandidate(const std::shared_ptr<DispatchVariant>& variant, const DrawProgram& now, const DrawProgram& was, std::uint32_t queue, std::uint64_t& imagesFlushed, std::uint64_t& runsSynced, std::optional<SampledReadScope>& sampling, DrawCandidateCounters& counters) {
    std::uint64_t moved = 0;
    bool movedBeyond = false;
    for (std::size_t w = 0; w < now.userData.size(); ++w) {
        if (now.userData[w] == was.userData[w]) continue;
        if (w >= 64u) movedBeyond = true;
        else if (((variant->userDataMask >> w) & 1u) != 0) moved |= std::uint64_t {1} << w;
    }
    ++counters.tried;
    auto candidate = variant;
    const bool rebased = moved != 0 || movedBeyond;
    const auto based = drawBufferBaseWords() ? moved & variant->bufferBaseWords & ~variant->rebaseWords : 0;
    if (rebased) {
        if (!drawRebaseEnabled()) {
            ++counters.refusedOff;
            return nullptr;
        }
        const auto fetchWords = variant->vertexInfo == nullptr ? std::uint64_t{0} : fetchTableWords(*variant->vertexInfo);
        const auto forwarded = drawForwardedWords() && now.hiddenUserWords == 0 ? forwardedOnlyWords(*variant, fetchWords) : 0;
        const auto untagged = moved & ~variant->rebaseWords & ~forwarded & ~based;
        if (movedBeyond || untagged != 0) {
            ++counters.refusedUntagged;
            const auto index = movedBeyond ? 64u : static_cast<std::uint32_t>(std::countr_zero(untagged));
            ++counters.refusedBy[static_cast<std::size_t>(refusedWordRole(*variant, index, fetchWords, now.hiddenUserWords))];
            static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
            if (profile) counters.refusedWords.push_back((now.binary.codeAddress << 8u) | std::min(index, 255u));
            return nullptr;
        }
        if (now.tailContinuation != was.tailContinuation) {
            ++counters.refusedTail;
            return nullptr;
        }
        auto copy = copyVariant(*variant);
        for (auto bits = moved; bits != 0; bits &= bits - 1) {
            const auto index = static_cast<std::uint32_t>(std::countr_zero(bits));
            const auto from = was.userData[index], to = now.userData[index];
            if (((variant->rebaseWords >> index) & 1u) == 0) {
                if (((based >> index) & 1u) != 0 && (to == 0 || ((from ^ to) & 3u) != 0)) {
                    ++counters.refusedPointer;
                    return nullptr;
                }
                forwardVariantWord(*copy, index, to);
                continue;
            }
            if (from == 0 || to == 0 || ((from ^ to) & 3u) != 0) {
                ++counters.refusedPointer;
                return nullptr;
            }
            rebaseVariantWord(*copy, index, from, to);
        }
        if (based != 0) copy->compiled = patchBufferBaseWords(*copy, based, now.userData);
        applyShaderDataPatches(*copy);
        candidate = std::move(copy);
    }
    std::vector<std::pair<std::uint32_t, std::uint32_t>> liveData;
    auto* live = candidateDataWords() ? &liveData : nullptr;
    std::vector<ShaderRecompiler::MemoryRegion> regions;
    appendEntryRegions(*candidate, regions);
    auto result = validateVariant(now.binary.codeAddress, queue, *candidate, regions, imagesFlushed, runsSynced, sampling, live);
    if (gateRetry() && (result == EntryOutcome::PublishMoved || result == EntryOutcome::PendingMoved)) result = validateVariant(now.binary.codeAddress, queue, *candidate, regions, imagesFlushed, runsSynced, sampling, live);
    if (result != EntryOutcome::Equal && result != EntryOutcome::EqualData) {
        ++(rebased ? counters.differingRebased : counters.differingEqual);
        return nullptr;
    }
    const bool data = result == EntryOutcome::EqualData;
    if (data) {
        if (!rebased) candidate = copyVariant(*variant);
        for (const auto& [position, value] : liveData) candidate->words[position] = value;
        candidate->compiled = patchDataWords(*candidate, candidate->words);
    }
    if (!rebased) ++(data ? counters.equalData : counters.equal);
    else if (based != 0) ++counters.bufferBases;
    else if ((moved & ~variant->rebaseWords) != 0) ++counters.forwarded;
    else ++(data ? counters.rebasedData : counters.rebased);
    const auto movedWord = [&](std::uint32_t index) { return index < 64u && ((moved >> index) & 1u) != 0; };
    const auto& info = variant->vertexInfo;
    if (rebased && info != nullptr && info->fetchEmbedded && (movedWord(info->fetchAttribReg) || movedWord(info->fetchBufferReg))) ++counters.rebasedFetch;
    return candidate;
}

void Driver::lookupDraw(const Submission& submission, const std::shared_ptr<VulkanDevice>& localDevice, const Graphics::State& graphics, const ShaderRecompiler::ShaderPixelStageInfo& pixel, const std::vector<DrawProgram>& programs, const std::vector<ShaderRecompiler::ProgramRole>& roles, const std::vector<std::optional<ShaderRecompiler::ShaderVertexStageInfo>>& vertexInfos, bool useDrawEntries, bool registerKey, bool profile, std::uint64_t& drawKey, std::shared_ptr<DrawEntry>& entry, std::vector<std::shared_ptr<DispatchVariant>>& matched, std::vector<std::vector<ShaderRecompiler::MemoryRegion>>& matchedRegions, bool& drawHit, bool& verifyHit, DrawPhaseTiming& phaseTiming, std::array<double, DrawDriverPhaseCount>& phaseMs, const std::shared_ptr<DrawEntry>& dataEntry, DrawDataCandidates& dataCandidates, std::vector<StageDataHit>& stageData) {
    using Role = ShaderRecompiler::ProgramRole;
    if (useDrawEntries) {
        if (!registerKey) {
            drawKey = 0xcbf29ce484222325ull;
            const auto mix = [&](std::uint64_t value) {
                drawKey ^= value;
                drawKey *= 0x100000001b3ull;
            };
            mix(localDevice->Serial());
            mix(static_cast<std::uint64_t>(graphics.stages.path));
            mix(graphics.stages.registerValue);
            mix(graphics.stages.vertexWaveSize);
            mix(graphics.stages.fragmentWaveSize);
            mix(graphics.stages.mesh.has_value());
            if (graphics.stages.mesh) {
                const auto& mesh = *graphics.stages.mesh;
                for (const auto value : {mesh.inputPrimitive, mesh.primitivesPerGroup, mesh.verticesPerGroup, mesh.maxVertices, mesh.maxPrimitives, mesh.threadsPerGroup, mesh.ldsSizeDwords, mesh.provokingVertex, mesh.esgsItemSize, mesh.outputPrimitiveSize}) mix(value);
            }
            mix(graphics.stages.tessellation.has_value());
            if (graphics.stages.tessellation) {
                const auto& tess = *graphics.stages.tessellation;
                for (const auto value : {tess.inputControlPoints, tess.outputControlPoints, tess.domain, tess.partitioning, tess.outputTopology, tess.patchesPerGroup, tess.primitiveGroupSize, static_cast<std::uint32_t>(tess.breakAtInstance)}) mix(value);
            }
            mix(graphics.rectList);
            mix(programs.size());
            for (std::size_t i = 0; i < programs.size(); ++i) {
                const auto& program = programs[i];
                mix(reinterpret_cast<std::uintptr_t>(program.snapshot.get()));
                mix(program.codeOffset);
                mix(static_cast<std::uint64_t>(roles[i]));
                mix(static_cast<std::uint64_t>(program.binary.stage));
                mix(program.userDataBase);
                mix(program.firstUserSgpr);
                mix(program.userData.size());
                for (const auto word : program.userData) mix(word);
                mix(vertexInfos[i].has_value());
                if (!vertexInfos[i]) continue;
                const auto& vertex = *vertexInfos[i];
                require(vertex.resourcesNum <= vertex.resources.size(), "vertex stage info resource count exceeds its table");
                mix(vertex.resourcesNum);
                mix(vertex.fetchAttribReg);
                mix(vertex.fetchBufferReg);
                mix(vertex.fetchEmbedded);
                mix(vertex.paClVsOutCntl);
                for (std::uint32_t r = 0; r < vertex.resourcesNum; ++r) {
                    for (const auto field : vertex.resources[r].fields) mix(field);
                    const auto& destination = vertex.resourcesDst[r];
                    mix(static_cast<std::uint32_t>(destination.registerStart));
                    mix(static_cast<std::uint32_t>(destination.registersNum));
                    mix(static_cast<std::uint32_t>(destination.attrId));
                    mix(destination.fetchIndex);
                }
            }
            require(pixel.interpolatorCount <= pixel.interpolatorSettings.size(), "pixel stage info interpolator count exceeds its table");
            mix(pixel.interpolatorCount);
            for (std::uint32_t i = 0; i < pixel.interpolatorCount; ++i) mix(pixel.interpolatorSettings[i]);
            mix(pixel.inputAddr);
            for (const bool flag : {pixel.wave32, pixel.hasPerspectiveCenterVgpr, pixel.perspectiveCentroid, pixel.posX, pixel.posY, pixel.posZ, pixel.posW, pixel.frontFace, pixel.frontFaceAllBits, pixel.ancillary, pixel.sampleShading, pixel.noPerspective, pixel.linearCentroid, pixel.pixelKillEnable, pixel.depthExportEnable, pixel.sampleMaskExportEnable, pixel.earlyZ, pixel.executeOnNoop}) mix(flag);
            mix(static_cast<std::uint64_t>(pixel.conservativeZExport));
            mix(pixel.orderedPixelShader);
            mix(pixel.stencilRefExportEnable);
            for (const auto value : pixel.targetOutputMode) mix(value);
            for (const auto value : pixel.targetExportMapping) mix(value);
            for (const auto value : pixel.targetExportPacking) mix(static_cast<std::uint64_t>(value));
            mix(pixel.dualSourceBlend);
            mix(pixel.reversedBlend);
            mix(pixel.alphaToCoverage);
            mix(static_cast<std::uint64_t>(pixel.rasterPrimitive));
            mix(pixel.ditheredAlphaToCoverage);
            mix(pixel.quadPixelMask);
            for (const auto mask : pixel.quadSampleMask) mix(mask);
            std::lock_guard cacheLock(drawCacheMutex);
            ++drawEntryCounters.lookups;
            const auto found = drawCache.find(drawKey);
            if (found != drawCache.end()) entry = found->second;
            else ++drawEntryCounters.absent;
            maybeReportDrawCache(profile);
        }
        if (entry != nullptr) {
            const auto waitedBeforeValidate = profile ? Graphics::Recorder::ThreadWaitedMs() : 0.0;
            const auto poolWaitedBeforeValidate = profile ? DrawPrepPool::ThreadWaitedMs() : 0.0;
            const auto stepsBefore = validateSteps();
            std::optional<DrawMiss> miss;
            std::vector<std::size_t> ranks(programs.size(), 0);
            std::uint64_t stageValidations = 0, stageEqual = 0, compared = 0, imagesFlushed = 0, runsSynced = 0, dataStages = 0, dataWords = 0;
            std::vector<std::pair<std::uint32_t, std::uint32_t>> liveData;
            auto* live = drawStageDataHits() ? &liveData : nullptr;
            if (entry->stages.size() != programs.size()) miss = DrawMiss::Stages;

            std::uint32_t cursor = 0;
            {
                const GuestMemory::ReadSiteScope site(GuestMemory::ReadSite::DrawCache);

                std::optional<SampledReadScope> sampling;
                for (std::size_t i = 0; !miss && i < programs.size(); ++i) {
                    if (roles[i] == Role::GeometryBack) continue;
                    ++stageValidations;
                    cursor = Graphics::StagePushOffset(cursor, programs[i].binary.stage, localDevice->GraphicsPipelineLibraries());
                    const auto& variants = entry->stages[i];
                    auto outcome = EntryOutcome::Differing;
                    bool anyLayout = false;
                    for (std::size_t rank = 0; rank < variants.size(); ++rank) {
                        const auto& variant = variants[rank];
                        if (variant->pushOffset != cursor) continue;
                        if (programs[i].binary.stage == ShaderRecompiler::ShaderStage::Mesh && !std::equal(variant->meshIndexDescriptor.begin(), variant->meshIndexDescriptor.end(), programs[i].userData.begin() + ShaderRecompiler::MeshIndexBufferUserWord)) {
                            if (!anyLayout) outcome = EntryOutcome::Differing;
                            anyLayout = true;
                            continue;
                        }
                        auto& regions = matchedRegions[i];
                        regions.clear();
                        appendEntryRegions(*variant, regions);
                        ++compared;
                        auto result = validateVariant(programs[i].binary.codeAddress, submission.queue, *variant, regions, imagesFlushed, runsSynced, sampling, live);
                        if (gateRetry() && (result == EntryOutcome::PublishMoved || result == EntryOutcome::PendingMoved)) result = validateVariant(programs[i].binary.codeAddress, submission.queue, *variant, regions, imagesFlushed, runsSynced, sampling, live);
                        if (!anyLayout) outcome = result;
                        anyLayout = true;
                        if (result != EntryOutcome::Equal && result != EntryOutcome::EqualData) continue;
                        if (result == EntryOutcome::EqualData) {
                            auto& hit = stageData[i];
                            hit.words = variant->words;
                            for (const auto& [position, value] : liveData) hit.words[position] = value;
                            regions.clear();
                            appendEntryRegions(*variant, regions, &hit.words);
                            hit.compiled = patchDataWords(*variant, hit.words);
                            ++dataStages;
                            dataWords += liveData.size();
                        }
                        matched[i] = variant;
                        ranks[i] = rank;
                        ++stageEqual;
                        cursor += static_cast<std::uint32_t>(variant->compiled->pushConstants.size());
                        break;
                    }
                    if (matched[i] != nullptr) continue;
                    if (!anyLayout) miss = DrawMiss::Layout;
                    else if (outcome != EntryOutcome::Differing) miss = DrawMiss::Gate;
                    else miss = roles[i] == Role::Fragment ? DrawMiss::FragmentDiffering : i == 0 ? DrawMiss::FrontDiffering : DrawMiss::OtherDiffering;
                }
            }
            drawHit = !miss;
            if (profile) {

                phaseTiming.Phase(DrawRowKeyLookupValidate);
                const auto waited = std::min(Graphics::Recorder::ThreadWaitedMs() - waitedBeforeValidate, phaseMs[DrawRowKeyLookupValidate]);
                phaseMs[DrawRowKeyLookupValidate] -= waited;
                phaseMs[DrawRowValidateWait] += waited;
                const auto poolWaited = std::min(DrawPrepPool::ThreadWaitedMs() - poolWaitedBeforeValidate, phaseMs[DrawRowKeyLookupValidate]);
                phaseMs[DrawRowKeyLookupValidate] -= poolWaited;
                phaseMs[DrawRowValidatePoolWait] += poolWaited;
            }
            // Prep-pool jobs validate concurrently (APS5_NO_POOL_LOOKUP unset), so the tally takes the
            // cache lock shared and counts through atomic_ref; the exclusive holders (inserts, reports,
            // the LRU touch below) never overlap it. Only a rank rotation or a due touch upgrades.
            const auto add = [](auto& counter, auto value) { std::atomic_ref(counter).fetch_add(value, std::memory_order_relaxed); };
            bool rotate = false, touch = false;
            {
                std::shared_lock cacheLock(drawCacheMutex);
                auto& counters = drawEntryCounters;
                add(counters.stageValidations, stageValidations);
                add(counters.stageEqual, stageEqual);
                add(counters.variantsCompared, compared);
                const auto steps = validateSteps().Since(stepsBefore);
                for (const auto field : {&ValidateSteps::calls, &ValidateSteps::runs, &ValidateSteps::bytes, &ValidateSteps::pipelineNs, &ValidateSteps::syncNs, &ValidateSteps::gatesNs, &ValidateSteps::compareNs}) add(counters.validateSteps.*field, steps.*field);
                if (drawHit) {
                    add(counters.hits, std::uint64_t{1});
                    add(counters.stageDataHits, dataStages);
                    add(counters.stageDataWords, dataWords);
                    if (registerKey) add(counters.registerKeyHits, std::uint64_t{1});
                    const auto hits = std::atomic_ref(drawCacheHits).fetch_add(1, std::memory_order_relaxed) + 1;
                    for (std::size_t i = 0; i < programs.size(); ++i) {
                        if (matched[i] == nullptr) continue;
                        add(counters.variantHitsByRank[ranks[i]], std::uint64_t{1});
                        if (ranks[i] != 0) rotate = true;
                    }
                    const auto again = drawCache.find(drawKey);
                    touch = again != drawCache.end() && again->second == entry && hits - again->second->touched > drawCacheEntries() / 8;
                    if (verifyDrawEntries()) {
                        add(counters.verifyHits, std::uint64_t{1});
                        verifyHit = true;
                        drawHit = false;
                    }
                    if (dataStages != 0 && verifyDrawDataHits()) drawHit = false;
                } else {
                    add(counters.misses[static_cast<std::size_t>(*miss)], std::uint64_t{1});
                    if (registerKey && entry->decode != nullptr) add(counters.decodePartial, std::uint64_t{1});
                }
                if (profile) add(counters.validateUs, phaseMs[DrawRowKeyLookupValidate] * 1000);
            }
            if (rotate || touch) {
                std::lock_guard cacheLock(drawCacheMutex);
                const auto again = drawCache.find(drawKey);
                if (again != drawCache.end() && again->second == entry) {
                    if (rotate) {
                        auto rotated = std::make_shared<DrawEntry>();
                        rotated->decode = entry->decode;
                        rotated->stages = entry->stages;
                        rotated->shape = entry->shape;
                        rotated->recipes.store(entry->recipes.load());
                        for (std::size_t i = 0; i < programs.size(); ++i) {
                            if (ranks[i] == 0) continue;
                            auto& variants = rotated->stages[i];
                            variants.erase(variants.begin() + static_cast<std::ptrdiff_t>(ranks[i]));
                            variants.insert(variants.begin(), matched[i]);
                        }
                        rotated->touched = entry->touched;
                        rotated->order = entry->order;
                        again->second = std::move(rotated);
                    }
                    if (drawCacheHits - again->second->touched > drawCacheEntries() / 8) {
                        drawOrder.splice(drawOrder.begin(), drawOrder, again->second->order);
                        again->second->touched = drawCacheHits;
                        ++drawEntryCounters.touches;
                    }
                }
            }
        } else if (dataEntry != nullptr && graphics.stages.path == Graphics::ShaderPath::Vertex && dataEntry->stages.size() == programs.size() && dataEntry->decode->programs.size() == programs.size()) {
            std::uint64_t imagesFlushed = 0, runsSynced = 0;
            DrawCandidateCounters tally;
            const auto stepsBefore = validateSteps();
            {
                const GuestMemory::ReadSiteScope site(GuestMemory::ReadSite::DrawCache);
                std::optional<SampledReadScope> sampling;
                for (std::size_t i = 0; i < programs.size(); ++i) {
                    const auto& was = dataEntry->decode->programs[i];
                    if (roles[i] == Role::GeometryBack || programs[i].userData.size() != was.userData.size()) continue;
                    auto& candidates = dataCandidates[i];
                    for (const auto& variant : dataEntry->stages[i]) {
                        if ((i == 0 && variant->pushOffset != 0) || std::any_of(candidates.begin(), candidates.end(), [&](const auto& candidate) { return candidate.first->pushOffset == variant->pushOffset; })) continue;
                        auto candidate = dataCandidate(variant, programs[i], was, submission.queue, imagesFlushed, runsSynced, sampling, tally);
                        if (candidate == nullptr) continue;
                        std::vector<ShaderRecompiler::MemoryRegion> regions;
                        appendEntryRegions(*candidate, regions);
                        candidates.emplace_back(std::move(candidate), std::move(regions));
                    }
                }
            }
            std::lock_guard cacheLock(drawCacheMutex);
            drawEntryCounters.candidates += tally;
            for (const auto word : tally.refusedWords) ++drawEntryCounters.refusedByProgramWord[word];
            drawEntryCounters.validateSteps += validateSteps().Since(stepsBefore);
        }
        phaseTiming.Phase(DrawRowKeyLookupValidate);
    }
}

}
