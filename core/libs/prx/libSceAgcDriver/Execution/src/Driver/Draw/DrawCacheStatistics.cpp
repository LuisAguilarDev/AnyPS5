#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include <algorithm>
#include <cstdlib>
#include <string>
#include <vector>

namespace AgcDriver::DriverDetail {

bool Driver::traceDrawCache() {
    static const bool trace = std::getenv("APS5_TRACE_DRAW_CACHE") != nullptr;
    return trace;
}

void Driver::traceInsert(std::unordered_set<std::uint64_t>& set, std::uint64_t key) {
    if (set.size() >= (1u << 16u)) set.clear();
    set.insert(key);
}

void Driver::maybeReportDrawCache(bool profile) {
    auto& counters = drawEntryCounters;
    if (!profile && !traceDrawCache()) return;
    const auto now = std::chrono::steady_clock::now();
    if (now - counters.lastReport < std::chrono::seconds(10)) return;
    reportDrawCache(counters, profile);
    counters.lastReport = now;
}

void Driver::reportDrawCache(DrawEntryCounters& counters, bool profile) {
    const auto count = [](std::uint64_t value) { return static_cast<unsigned long long>(value); };
    const auto miss = [&](DrawMiss reason) { return count(counters.misses[static_cast<std::size_t>(reason)]); };
    std::string text;
    if (traceDrawCache()) {
        char line[1536];
        std::uint64_t misses = 0;
        for (const auto value : counters.misses) misses += value;
        std::snprintf(line, sizeof(line), "[draw-cache] 10 s: %llu lookups, %llu hits (%.1f%%), %llu stage misses, %llu without an entry, %llu data-only hits (stages: %llu reused, %llu compiled; %llu verified); no entry: new registers %llu, user words/pointers only %llu, evicted %llu, seen but never inserted %llu; stage misses: front differing %llu, fragment differing %llu, other differing %llu, layout %llu, gate %llu, stage count %llu; differing captures: same runs %llu (%.1f words changed each), other runs %llu; %zu entries, %llu evicted\n", count(counters.lookups), count(counters.hits), counters.lookups != 0 ? 100.0 * static_cast<double>(counters.hits) / static_cast<double>(counters.lookups) : 0.0, count(misses), count(counters.absent), count(counters.dataHits), count(counters.dataStagesReused), count(counters.dataStagesCompiled), count(counters.dataVerified), count(counters.absentNewRegisters), count(counters.absentUserWords), count(counters.absentEvicted), count(counters.absentNeverInserted), miss(DrawMiss::FrontDiffering), miss(DrawMiss::FragmentDiffering), miss(DrawMiss::OtherDiffering), miss(DrawMiss::Layout), miss(DrawMiss::Gate), miss(DrawMiss::Stages), count(counters.differingSameRuns), counters.differingSameRuns != 0 ? static_cast<double>(counters.differingWords) / static_cast<double>(counters.differingSameRuns) : 0.0, count(counters.differingRunsChanged), drawCache.size(), count(counters.evictions));
        text += line;
        std::snprintf(line, sizeof(line), "[draw-cache] register-key miss component changes (10 s): stable registers %llu, user data %llu, auxiliary registers %llu, device %llu, first shader set %llu, unchanged components %llu\n", count(counters.registerKeyMissChanges[0]), count(counters.registerKeyMissChanges[1]), count(counters.registerKeyMissChanges[2]), count(counters.registerKeyMissChanges[3]), count(counters.registerKeyMissChanges[4]), count(counters.registerKeyMissChanges[5]));
        text += line;
    }
    if (profile) text += profileDrawCacheLines(counters);
    AgcDriver::ProfilePrint_nid_no_patch("%s", text.c_str());
    counters = DrawEntryCounters{};
}

DrawEntryCounters Driver::DrawCacheCounters() {
    std::lock_guard cacheLock(drawCacheMutex);
    return drawEntryCounters;
}

std::string Driver::profileDrawCacheLines(const DrawEntryCounters& counters) {
    std::string text;
    const auto append = [&](const char* format, auto... values) {
        char line[2048];
        std::snprintf(line, sizeof(line), format, values...);
        text += line;
    };
    const auto count = [](std::uint64_t value) { return static_cast<unsigned long long>(value); };
    const auto validated = counters.lookups > counters.absent ? counters.lookups - counters.absent : 0;
    const auto miss = [&](DrawMiss reason) { return count(counters.misses[static_cast<std::size_t>(reason)]); };
    std::string ranks;
    for (std::size_t rank = 0; rank < dispatchVariants(); ++rank) {
        char item[32];
        std::snprintf(item, sizeof(item), "%s%llu", rank == 0 ? "" : " / ", count(counters.variantHitsByRank[rank]));
        ranks += item;
    }
    append("[draw-cache] %llu lookups (10 s): %llu no entry, %llu validated in %.1f us each (key, lookup and every stage): %llu hits (every stage equal; stage hits by variant rank 1..k %s), misses by reason: front stage differing %llu, fragment differing %llu, other stage differing %llu, layout (push offset) %llu, gate %llu, stage count %llu; %llu stage validations (%llu equal, %.2f variants compared each); %llu inserts (%llu variants inserted, %llu evicted beyond k, %llu already present, %llu unstable), %zu entries (%llu variants, ~%.1f MiB), %llu evictions in total, %llu LRU moves; verify: %llu hits captured again, %llu stages differed\n", count(counters.lookups), count(counters.absent), count(validated), validated != 0 ? counters.validateUs / static_cast<double>(validated) : 0.0, count(counters.hits), ranks.c_str(), miss(DrawMiss::FrontDiffering), miss(DrawMiss::FragmentDiffering), miss(DrawMiss::OtherDiffering), miss(DrawMiss::Layout), miss(DrawMiss::Gate), miss(DrawMiss::Stages), count(counters.stageValidations), count(counters.stageEqual), counters.stageValidations != 0 ? static_cast<double>(counters.variantsCompared) / static_cast<double>(counters.stageValidations) : 0.0, count(counters.inserts), count(counters.variantsInserted), count(counters.variantsEvicted), count(counters.present), count(counters.unstable), drawCache.size(), count(drawCacheVariants), static_cast<double>(drawCacheVariantBytes) / (1024.0 * 1024.0), count(drawCacheEvictions), count(counters.touches), count(counters.verifyHits), count(counters.verifyMismatches));
    append("[draw-cache] register key (10 s): %llu lookups, %llu hits, key build %.1f us each; decode skipped %llu, partial (state/pixel/programs from the entry) %llu; facade log != table %llu; verify: %llu decodes compared, %llu differed\n", count(counters.registerKeyLookups), count(counters.registerKeyHits), counters.registerKeyLookups != 0 ? counters.keyUs / static_cast<double>(counters.registerKeyLookups) : 0.0, count(counters.decodeSkipped), count(counters.decodePartial), count(counters.facadeMismatches), count(counters.verifyDecodes), count(counters.verifyDecodeMismatches));
    const auto& candidates = counters.candidates;
    const auto refused = [&](DrawRefusedWord role) { return count(candidates.refusedBy[static_cast<std::size_t>(role)]); };
    std::vector<std::pair<std::uint64_t, std::uint64_t>> refusedWords(counters.refusedByProgramWord.begin(), counters.refusedByProgramWord.end());
    std::sort(refusedWords.begin(), refusedWords.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    std::string topRefused;
    for (std::size_t i = 0; i < refusedWords.size() && i < 4; ++i) {
        char item[64];
        std::snprintf(item, sizeof(item), " 0x%llx word %llu x%llu", static_cast<unsigned long long>(refusedWords[i].first >> 8u), static_cast<unsigned long long>(refusedWords[i].first & 0xffu), count(refusedWords[i].second));
        topRefused += item;
    }
    append("[draw-cache] data-only refusals (10 s): first moved word not a read base, by role: used as a value %llu, base and value %llu, base not taggable %llu, vertex fetch table %llu, forwarded but not patchable %llu, unread by a stage that walks no user word %llu, beyond word 63 %llu; forwarded words patched %llu; top program words:%s\n", refused(DrawRefusedWord::Value), refused(DrawRefusedWord::BaseAndValue), refused(DrawRefusedWord::BaseDropped), refused(DrawRefusedWord::Fetch), refused(DrawRefusedWord::Forwarded), refused(DrawRefusedWord::Unwalked), refused(DrawRefusedWord::Beyond), count(candidates.forwarded), topRefused.c_str());
    append("[draw-cache] data-only hits (10 s): %llu draws, %llu with every stage reused; stages %llu reused / %llu compiled; %llu candidates tried: equal %llu (+%llu with buffer bases refreshed), rebased %llu (+%llu with buffer bases refreshed; %llu moved vertex fetch tables); refused: %llu moved word not a read base, %llu moved pointer zero or misaligned, %llu tail continuation changed, %llu rebasing off; differing after validation: %llu equal, %llu rebased\n", count(counters.dataHits), count(counters.dataDrawsReused), count(counters.dataStagesReused), count(counters.dataStagesCompiled), count(candidates.tried), count(candidates.equal), count(candidates.equalData), count(candidates.rebased), count(candidates.rebasedData), count(candidates.rebasedFetch), count(candidates.refusedUntagged), count(candidates.refusedPointer), count(candidates.refusedTail), count(candidates.refusedOff), count(candidates.differingEqual), count(candidates.differingRebased));
    const auto& steps = counters.validateSteps;
    const auto perCall = [&](std::uint64_t ns) { return steps.calls != 0 ? static_cast<double>(ns) / 1000.0 / static_cast<double>(steps.calls) : 0.0; };
    append("[draw-cache] variant validations (10 s): %llu, %.1f runs (%.0f bytes) each; us each: pipeline overlap checks %.2f (prep-pool waits included), pending-write syncs %.2f, image/shadow/label gates %.2f, compare %.2f\n", count(steps.calls), steps.calls != 0 ? static_cast<double>(steps.runs) / static_cast<double>(steps.calls) : 0.0, steps.calls != 0 ? static_cast<double>(steps.bytes) / static_cast<double>(steps.calls) : 0.0, perCall(steps.pipelineNs), perCall(steps.syncNs), perCall(steps.gatesNs), perCall(steps.compareNs));
    append("[draw-cache] stage data hits (10 s): %llu stages (%llu words refreshed), verified %llu; stage inserts with data positions %llu (%.1f positions each)\n", count(counters.stageDataHits), count(counters.stageDataWords), count(counters.stageDataVerified), count(counters.stageDataInserts), counters.stageDataInserts != 0 ? static_cast<double>(counters.stageDataPositions) / static_cast<double>(counters.stageDataInserts) : 0.0);
    return text;
}

}
