#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Diagnostics.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace AgcDriver::DriverDetail {

namespace {

constexpr std::size_t MaxAliases = 16384;
constexpr std::size_t MaxContentWords = 64;

std::uint64_t AlignedWord(std::uint32_t word) {
    return word & ~std::uint32_t {3};
}

}

bool Driver::drawRelocationEnabled() {
    static const bool enabled = std::getenv("APS5_NO_DRAW_RELOCATION") == nullptr;
    return enabled;
}

std::uint64_t Driver::relocatedContentKey(const RelocatableUserWord& word, std::uint32_t value, bool& readable) {
    std::uint64_t hash = 0x9e3779b97f4a7c15ull;
    const auto mix = [&](std::uint64_t mixed) {
        hash ^= mixed;
        hash *= 0x100000001b3ull;
    };
    mix(value & 3u);
    if (value == 0) {
        readable = false;
        return hash;
    }
    std::array<std::uint32_t, MaxContentWords> content;
    for (const auto& [offset, words] : word.reads) {
        mix(static_cast<std::uint64_t>(offset));
        mix(words);
        for (std::size_t done = 0; done < words; done += MaxContentWords) {
            const auto count = std::min<std::size_t>(words - done, MaxContentWords);
            const auto address = AlignedWord(value) + static_cast<std::uint64_t>(offset) + done * sizeof(std::uint32_t);
            if (GuestMemory::CopyMapped(address, std::as_writable_bytes(std::span(content).first(count))) != GuestMemory::Compare::Equal) {
                readable = false;
                return hash;
            }
            for (std::size_t i = 0; i < count; ++i) mix(content[i]);
        }
    }
    return hash;
}

bool Driver::drawRebaseEnabled() {
    static const bool enabled = std::getenv("APS5_NO_DRAW_REBASE") == nullptr;
    return enabled;
}

bool Driver::drawRebaseFetch() {
    static const bool enabled = drawRebaseEnabled() && std::getenv("APS5_NO_DRAW_REBASE_FETCH") == nullptr;
    return enabled;
}

bool Driver::drawForwardedWords() {
    static const bool enabled = drawRebaseEnabled() && std::getenv("APS5_NO_DRAW_FORWARDED_WORDS") == nullptr;
    return enabled;
}

bool Driver::drawBufferBaseWords() {
    static const bool enabled = drawRebaseEnabled() && std::getenv("APS5_NO_DRAW_BUFFER_BASE_WORDS") == nullptr;
    return enabled;
}

// A user word the walk read only as the base of inline buffer descriptors (V# dword 0) can move
// without a capture when every GuestBuffers slot of those buffers holds its value and the value
// appears nowhere else in the result: the copy then writes the new base into those slots.
void Driver::tagBufferBaseWords(DispatchVariant& variant, const ShaderRecompiler::SrtUserDataUse& use, std::span<const std::uint32_t> userData, std::uint32_t hiddenWords) {
    variant.bufferBaseCandidates = use.asBufferBase;
    variant.bufferBaseWords = 0;
    variant.bufferBaseSlots.clear();
    if (!drawBufferBaseWords() || hiddenWords != 0 || use.asBufferBase == 0 || variant.compiled == nullptr) return;
    const auto& compiled = *variant.compiled;
    const auto fetchWords = variant.vertexInfo == nullptr ? std::uint64_t{0} : fetchTableWords(*variant.vertexInfo);
    std::vector<std::array<std::uint32_t, 3>> slots;
    std::uint64_t kept = 0;
    for (auto bits = use.asBufferBase & ~fetchWords; bits != 0; bits &= bits - 1) {
        const auto index = static_cast<std::uint32_t>(std::countr_zero(bits));
        if (index >= userData.size() || userData[index] == 0) continue;
        bool exact = true;
        const auto first = slots.size();
        for (const auto& [word, buffer] : use.bufferBases) {
            if (word != index) continue;
            std::size_t found = 0;
            for (const auto& slot : compiled.bufferBaseSlots) {
                if (slot[0] != buffer) continue;
                ++found;
                if (slot[1] >= compiled.bindings.size() || compiled.bindings[slot[1]].role != ShaderRecompiler::DescriptorRole::GuestBuffers || slot[2] >= compiled.bindings[slot[1]].guestDescriptor.size() || compiled.bindings[slot[1]].guestDescriptor[slot[2]] != userData[index]) exact = false;
                else slots.push_back({index, slot[1], slot[2]});
            }
            if (found == 0) exact = false;
        }
        if (!exact || slots.size() == first) {
            slots.resize(first);
            continue;
        }
        kept |= std::uint64_t {1} << index;
    }
    const auto held = [&](std::uint32_t binding, std::uint32_t at) { return std::any_of(slots.begin(), slots.end(), [&](const std::array<std::uint32_t, 3>& slot) { return slot[1] == binding && slot[2] == at; }); };
    const auto forwardedAs = [&](std::size_t at, std::uint32_t index) { return at < compiled.shaderDataUserWords.size() && compiled.shaderDataUserWords[at] == index; };
    for (auto bits = kept; bits != 0; bits &= bits - 1) {
        const auto index = static_cast<std::uint32_t>(std::countr_zero(bits));
        const auto value = userData[index];
        bool unique = std::none_of(compiled.specialization.begin(), compiled.specialization.end(), [&](const ShaderRecompiler::PipelineSpecializationConstant& constant) { return constant.value == value; });
        for (std::uint32_t binding = 0; unique && binding < compiled.bindings.size(); ++binding) {
            const auto& words = compiled.bindings[binding].guestDescriptor;
            const bool shaderData = compiled.bindings[binding].role == ShaderRecompiler::DescriptorRole::ShaderData;
            for (std::uint32_t at = 0; unique && at < words.size(); ++at) unique = words[at] != value || held(binding, at) || (shaderData && forwardedAs(at, index));
        }
        for (std::size_t at = 0; unique && (at + 1) * sizeof(std::uint32_t) <= compiled.pushConstants.size(); ++at) {
            std::uint32_t word = 0;
            std::memcpy(&word, compiled.pushConstants.data() + at * sizeof(std::uint32_t), sizeof(word));
            unique = word != value || forwardedAs(at, index);
        }
        if (!unique) kept &= ~(std::uint64_t {1} << index);
    }
    variant.bufferBaseWords = kept;
    for (const auto& slot : slots) {
        if ((kept >> slot[0] & 1u) != 0) variant.bufferBaseSlots.push_back(slot);
    }
}

std::shared_ptr<const ShaderRecompiler::RecompileResult> Driver::patchBufferBaseWords(const DispatchVariant& variant, std::uint64_t words, std::span<const std::uint32_t> userData) {
    auto patched = std::make_shared<ShaderRecompiler::RecompileResult>(*variant.compiled);
    for (const auto& [word, binding, at] : variant.bufferBaseSlots) {
        if ((words >> word & 1u) != 0 && word < userData.size()) patched->bindings[binding].guestDescriptor[at] = userData[word];
    }
    return patched;
}

std::uint64_t Driver::fetchTableWords(const ShaderRecompiler::ShaderVertexStageInfo& vertexInfo) {
    std::uint64_t words = 0;
    for (const auto reg : {static_cast<std::uint32_t>(vertexInfo.fetchAttribReg), static_cast<std::uint32_t>(vertexInfo.fetchBufferReg)}) words |= reg + 1u < 64u ? std::uint64_t{3} << reg : ~std::uint64_t{0};
    return words;
}

std::uint64_t Driver::forwardedOnlyWords(const DispatchVariant& variant, std::uint64_t fetchWords) {
    const auto forwarded = variant.compiled->userDataRegisterReads;
    if (forwarded == ~std::uint64_t{0} || variant.valueWords == ~std::uint64_t{0}) return 0;
    return forwarded & ~(variant.baseWords | variant.valueWords) & ~fetchWords;
}

DrawRefusedWord Driver::refusedWordRole(const DispatchVariant& variant, std::uint32_t index, std::uint64_t fetchWords, std::uint32_t hiddenWords) {
    if (index >= 64u) return DrawRefusedWord::Beyond;
    const auto bit = std::uint64_t{1} << index;
    if ((fetchWords & bit) != 0) return DrawRefusedWord::Fetch;
    const bool base = (variant.baseWords & bit) != 0, value = variant.valueWords != ~std::uint64_t{0} && (variant.valueWords & bit) != 0;
    if (base && value) return DrawRefusedWord::BaseAndValue;
    if (value && (variant.bufferBaseCandidates & bit) != 0) return DrawRefusedWord::BufferBase;
    if (value) return DrawRefusedWord::Value;
    if (base) return DrawRefusedWord::BaseDropped;
    if ((variant.compiled->userDataRegisterReads & bit) != 0 || hiddenWords != 0) return DrawRefusedWord::Forwarded;
    return DrawRefusedWord::Unwalked;
}

bool Driver::candidateDataWords() {
    static const bool enabled = drawStageDataHits() && std::getenv("APS5_NO_DRAW_CANDIDATE_DATA_WORDS") == nullptr;
    return enabled;
}

bool Driver::verifyDrawRebase() {
    static const bool verify = std::getenv("APS5_VERIFY_DRAW_REBASE") != nullptr;
    return verify;
}

bool Driver::verifyDataCandidates() {
    return verifyDrawDataHits() || verifyDrawRebase();
}

std::vector<std::pair<std::uint64_t, std::int32_t>> Driver::vertexFetchReads(const std::optional<ShaderRecompiler::ShaderVertexStageInfo>& vertexInfo, std::span<const Graphics::DecodeRead> decodeReads, std::span<const std::uint32_t> userData, std::uint32_t hiddenWords) {
    constexpr std::uint64_t MaxTableBytes = ShaderRecompiler::ShaderVertexStageInfo::MaxResources * 16u;
    std::vector<std::pair<std::uint64_t, std::int32_t>> reads;
    if (!drawRebaseFetch() || hiddenWords != 0 || !vertexInfo || !vertexInfo->fetchEmbedded || vertexInfo->resourcesNum == 0 || decodeReads.size() != 2u * static_cast<std::size_t>(vertexInfo->resourcesNum)) return reads;
    const std::array<std::uint32_t, 2> words{vertexInfo->fetchAttribReg, vertexInfo->fetchBufferReg};
    for (std::size_t i = 0; i < decodeReads.size(); ++i) {
        const auto word = words[i % 2];
        const auto& read = decodeReads[i];
        if (word >= 63 || word + 1u >= userData.size() || userData[word] == 0) return {};
        const auto table = static_cast<std::uint64_t>(userData[word]) | (static_cast<std::uint64_t>(userData[word + 1u]) << 32u);
        if (read.address < table || read.address - table >= MaxTableBytes || (read.address & 3u) != 0 || read.bytes.empty() || read.bytes.size() % sizeof(std::uint32_t) != 0) return {};
        for (std::size_t at = 0; at < read.bytes.size(); at += sizeof(std::uint32_t)) reads.emplace_back(read.address + at, static_cast<std::int32_t>(word));
    }
    return reads;
}

void Driver::tagVariantRuns(DispatchVariant& variant, std::span<const ShaderRecompiler::MemoryRegion> regions, const ShaderRecompiler::SrtUserDataUse& use, bool eligible, std::uint32_t hiddenWords, std::span<const std::pair<std::uint64_t, std::int32_t>> fetchReads) {
    std::uint64_t relocatable = eligible && drawRelocationEnabled() ? use.asBase & ~use.asValue : 0;
    if (hiddenWords != 0) relocatable &= hiddenWords >= 64 ? 0 : ~((std::uint64_t {1} << hiddenWords) - 1);
    std::uint64_t rebase = drawRebaseEnabled() && hiddenWords == 0 ? use.asBase & ~use.asValue : 0;
    std::uint64_t fetch = 0;
    for (const auto& read : fetchReads) fetch |= std::uint64_t {1} << read.second;
    if ((fetch & use.asValue) != 0 || ((fetch << 1u) & (rebase | fetch)) != 0) fetch = 0;
    rebase |= fetch;
    std::vector<std::pair<std::uint64_t, std::int32_t>> reads;
    const auto covered = [&](std::uint64_t address) {
        return std::any_of(regions.begin(), regions.end(), [&](const ShaderRecompiler::MemoryRegion& region) { return address >= region.guestAddress && address - region.guestAddress + sizeof(std::uint32_t) <= region.bytes.size(); });
    };
    if ((relocatable | rebase) != 0) {
        reads = use.reads;
        if (fetch != 0) reads.insert(reads.end(), fetchReads.begin(), fetchReads.end());
        std::sort(reads.begin(), reads.end());
        reads.erase(std::unique(reads.begin(), reads.end()), reads.end());
        const auto drop = [&](std::int32_t origin) {
            if (origin < 0) return;
            relocatable &= ~(std::uint64_t {1} << origin);
            rebase &= ~(std::uint64_t {1} << origin);
        };
        for (std::size_t i = 0; i < reads.size(); ++i) {
            if (i + 1 < reads.size() && reads[i].first == reads[i + 1].first) {
                drop(reads[i].second);
                drop(reads[i + 1].second);
            }
            if (reads[i].second >= 0 && ((reads[i].first & 3u) != 0 || !covered(reads[i].first))) drop(reads[i].second);
        }
    }
    const auto origin = [&](std::uint64_t address) -> std::int8_t {
        const auto found = std::lower_bound(reads.begin(), reads.end(), std::make_pair(address, std::numeric_limits<std::int32_t>::min()));
        if (found == reads.end() || found->first != address || found->second < 0) return -1;
        return ((relocatable | rebase) >> found->second & 1u) != 0 ? static_cast<std::int8_t>(found->second) : std::int8_t {-1};
    };
    variant.relocatableWords = relocatable;
    variant.rebaseWords = rebase;
    for (const auto& region : regions) {
        const auto count = region.bytes.size() / sizeof(std::uint32_t);
        const auto offset = variant.words.size();
        variant.words.resize(offset + count);
        std::memcpy(variant.words.data() + offset, region.bytes.data(), count * sizeof(std::uint32_t));
        if ((relocatable | rebase) == 0) {
            variant.runs.emplace_back(region.guestAddress, region.guestAddress + region.bytes.size());
            continue;
        }
        std::size_t start = 0;
        auto current = count != 0 ? origin(region.guestAddress) : std::int8_t {-1};
        for (std::size_t word = 1; word <= count; ++word) {
            const auto next = word < count ? origin(region.guestAddress + word * sizeof(std::uint32_t)) : std::int8_t {-2};
            if (next == current) continue;
            variant.runs.emplace_back(region.guestAddress + start * sizeof(std::uint32_t), region.guestAddress + word * sizeof(std::uint32_t));
            variant.runBases.push_back(current);
            start = word;
            current = next;
        }
    }
}

namespace {

template <typename TMap>
std::size_t dropEveryOther(TMap& map) {
    std::size_t dropped = 0;
    bool drop = false;
    for (auto it = map.begin(); it != map.end();) {
        drop = !drop;
        if (!drop) {
            ++it;
            continue;
        }
        it = map.erase(it);
        ++dropped;
    }
    return dropped;
}

}

void Driver::learnPendingRelocation() {
    const auto pending = drawRelocationPending;
    drawRelocationPending = {};
    if (pending.key == 0) return;
    const auto inserted = drawCache.find(pending.key);
    if (inserted == drawCache.end() || inserted->second->decode == nullptr) return;
    const auto& entry = *inserted->second;
    const auto& programs = entry.decode->programs;
    if (entry.stages.size() != programs.size()) return;
    const auto known = drawRelocations.find(pending.identity);
    auto shape = std::make_shared<DrawRelocation>();
    if (known != drawRelocations.end()) *shape = *known->second;
    bool changed = known == drawRelocations.end();
    for (std::size_t i = 0; i < programs.size(); ++i) {
        if (entry.stages[i].empty()) continue;
        const auto& variant = *entry.stages[i].front();
        const auto& program = programs[i];
        for (std::uint32_t index = program.hiddenUserWords; index < program.userData.size() && index < 64; ++index) {
            const auto reg = program.userDataBase + index - program.hiddenUserWords;
            const auto held = std::find_if(shape->words.begin(), shape->words.end(), [&](const RelocatableUserWord& word) { return word.reg == reg; });
            if ((variant.relocatableWords >> index & 1u) == 0) {
                if (held == shape->words.end()) continue;
                shape->words.erase(held);
                shape->banned.push_back(reg);
                changed = true;
                continue;
            }
            if (held != shape->words.end() || std::find(shape->banned.begin(), shape->banned.end(), reg) != shape->banned.end()) continue;
            RelocatableUserWord word{reg, {}};
            const auto base = AlignedWord(program.userData[index]);
            for (std::size_t run = 0; run < variant.runs.size() && run < variant.runBases.size(); ++run) {
                if (variant.runBases[run] != static_cast<std::int8_t>(index)) continue;
                word.reads.emplace_back(static_cast<std::int64_t>(variant.runs[run].first - base), static_cast<std::uint32_t>((variant.runs[run].second - variant.runs[run].first) / sizeof(std::uint32_t)));
            }
            shape->words.push_back(std::move(word));
            changed = true;
        }
    }
    if (changed) {
        std::sort(shape->words.begin(), shape->words.end(), [](const RelocatableUserWord& a, const RelocatableUserWord& b) { return a.reg < b.reg; });
        drawRelocations[pending.identity] = std::move(shape);
    }
    if (pending.alias == 0) return;
    if (drawAliases.size() >= MaxAliases) drawRelocationCounters.evicted += dropEveryOther(drawAliases);
    drawAliases[pending.alias] = inserted->second;
}

void Driver::relocateDrawEntry(const QueueState& queue, std::uint64_t alias, std::shared_ptr<DrawEntry>& entry, std::shared_ptr<const DrawDecode>& decode, std::vector<DrawUserPatch>& patches) {
    auto& counters = drawRelocationCounters;
    ++counters.lookups;
    const auto found = drawAliases.find(alias);
    if (found == drawAliases.end() || found->second->decode == nullptr) return;
    const auto donor = found->second;
    const auto& programs = donor->decode->programs;
    if (donor->stages.size() != programs.size()) return;
    struct Move {
        std::size_t program;
        std::uint32_t index, from, to;
    };
    std::vector<Move> moves;
    for (std::size_t i = 0; i < programs.size(); ++i) {
        const auto& program = programs[i];
        for (std::uint32_t index = program.hiddenUserWords; index < program.userData.size(); ++index) {
            const auto current = readUserData(queue.shader, program.userDataBase + index - program.hiddenUserWords);
            if (current != program.userData[index]) moves.push_back({i, index, program.userData[index], current});
        }
    }
    if (moves.empty()) return;
    ++counters.donors;
    auto relocated = std::make_shared<DrawEntry>();
    relocated->decode = donor->decode;
    relocated->stages.resize(donor->stages.size());
    for (std::size_t i = 0; i < donor->stages.size(); ++i) {
        const bool moved = std::any_of(moves.begin(), moves.end(), [&](const Move& move) { return move.program == i; });
        if (!moved) {
            relocated->stages[i] = donor->stages[i];
            continue;
        }
        for (const auto& variant : donor->stages[i]) {
            if (variant == nullptr || variant->compiled == nullptr) continue;
            const bool allowed = std::all_of(moves.begin(), moves.end(), [&](const Move& move) { return move.program != i || (move.index < 64 && (variant->relocatableWords >> move.index & 1u) != 0 && move.from != 0 && move.to != 0 && ((move.from ^ move.to) & 3u) == 0); });
            if (!allowed) {
                ++counters.refused;
                continue;
            }
            auto copy = std::make_shared<DispatchVariant>();
            copy->runs = variant->runs;
            copy->runBases = variant->runBases;
            copy->relocatableWords = variant->relocatableWords;
            copy->words = variant->words;
            copy->forgetSerial = variant->forgetSerial;
            copy->compiled = variant->compiled;
            copy->shader = variant->shader;
            copy->dataPositions = variant->dataPositions;
            copy->dataSlots = variant->dataSlots;
            copy->flatBinding = variant->flatBinding;
            copy->pushOffset = variant->pushOffset;
            copy->vertexInfo = variant->vertexInfo;
            for (const auto& move : moves) {
                if (move.program == i) rebaseVariantWord(*copy, move.index, move.from, move.to);
            }
            relocated->stages[i].push_back(std::move(copy));
        }
    }
    patches.clear();
    for (const auto& move : moves) patches.push_back({move.program, move.index, move.to});
    entry = std::move(relocated);
    decode = donor->decode;
}

std::shared_ptr<DispatchVariant> Driver::copyVariant(const DispatchVariant& variant) {
    auto copy = std::make_shared<DispatchVariant>();
    copy->runs = variant.runs;
    copy->words = variant.words;
    copy->forgetSerial = variant.forgetSerial;
    copy->compiled = variant.compiled;
    copy->meshIndexDescriptor = variant.meshIndexDescriptor;
    copy->shader = variant.shader;
    copy->dataPositions = variant.dataPositions;
    copy->dataSlots = variant.dataSlots;
    copy->flatBinding = variant.flatBinding;
    copy->pushOffset = variant.pushOffset;
    copy->runBases = variant.runBases;
    copy->relocatableWords = variant.relocatableWords;
    copy->rebaseWords = variant.rebaseWords;
    copy->shaderDataPatches = variant.shaderDataPatches;
    copy->userDataMask = variant.userDataMask;
    copy->baseWords = variant.baseWords;
    copy->valueWords = variant.valueWords;
    copy->bufferBaseCandidates = variant.bufferBaseCandidates;
    copy->bufferBaseWords = variant.bufferBaseWords;
    copy->bufferBaseSlots = variant.bufferBaseSlots;
    copy->vertexInfo = variant.vertexInfo;
    return copy;
}

void Driver::rebaseVariantWord(DispatchVariant& copy, std::uint32_t index, std::uint32_t from, std::uint32_t to) {
    const auto delta = AlignedWord(to) - AlignedWord(from);
    for (std::size_t run = 0; run < copy.runs.size() && run < copy.runBases.size(); ++run) {
        if (copy.runBases[run] != static_cast<std::int8_t>(index)) continue;
        copy.runs[run].first += delta;
        copy.runs[run].second += delta;
    }
    forwardVariantWord(copy, index, to);
}

void Driver::forwardVariantWord(DispatchVariant& copy, std::uint32_t index, std::uint32_t to) {
    const auto& forwarded = copy.compiled->shaderDataUserWords;
    for (std::uint32_t slot = 0; slot < forwarded.size(); ++slot) {
        if (forwarded[slot] == index) copy.shaderDataPatches.emplace_back(slot, to);
    }
}

void Driver::applyShaderDataPatches(DispatchVariant& variant) {
    if (variant.shaderDataPatches.empty()) return;
    auto rebased = std::make_shared<ShaderRecompiler::RecompileResult>(*variant.compiled);
    for (const auto& [slot, value] : variant.shaderDataPatches) {
        const auto at = static_cast<std::size_t>(slot) * sizeof(std::uint32_t);
        if (at + sizeof(std::uint32_t) <= rebased->pushConstants.size()) std::memcpy(rebased->pushConstants.data() + at, &value, sizeof(value));
        for (auto& binding : rebased->bindings) {
            if (binding.role == ShaderRecompiler::DescriptorRole::ShaderData && slot < binding.guestDescriptor.size()) binding.guestDescriptor[slot] = value;
        }
    }
    variant.compiled = std::move(rebased);
    variant.shaderDataPatches.clear();
}

void Driver::reportDrawRelocation() {
    auto& counters = drawRelocationCounters;
    AgcDriver::ProfilePrint_nid_no_patch("[draw-relocate] %llu alias lookups: %llu donors with moved user words, %llu refusals, %llu hits through a relocated entry; %zu shader sets learned, %zu aliases held (%llu dropped at the limit)\n", static_cast<unsigned long long>(counters.lookups), static_cast<unsigned long long>(counters.donors), static_cast<unsigned long long>(counters.refused), static_cast<unsigned long long>(counters.hits), drawRelocations.size(), drawAliases.size(), static_cast<unsigned long long>(counters.evicted));
    counters = {};
}

}
