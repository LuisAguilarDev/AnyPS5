#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_LRUSTAMP_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_LRUSTAMP_HPP

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <iterator>

namespace AgcDriver::Graphics {

// Use order of the entries of an LRU list (ResourceCache, the sampled and storage texture caches).
// A hit stores a fresh stamp from one process-wide counter instead of splicing its entry to the
// front, so it writes one word and leaves the list alone; the list itself stays in the order the
// entries were placed in (front = newest). Before an eviction walks the list from the back,
// OrderLru moves the entries stamped since they were placed back into stamp order, which is the
// order the splices made: the same entries go, in the same order. The stamp is atomic so that a
// lookup that holds its cache's lock only shared can still store it. Debug aid:
// APS5_NO_LRU_STAMPS=1 splices on every hit as before.
class LruStamp {
public:
    LruStamp() : value(Next()), placed(value.load(std::memory_order_relaxed)) {}
    LruStamp(const LruStamp& other) : value(other.value.load(std::memory_order_relaxed)), placed(other.placed) {}
    LruStamp& operator=(const LruStamp& other) {
        value.store(other.value.load(std::memory_order_relaxed), std::memory_order_relaxed);
        placed = other.placed;
        return *this;
    }
    void Touch() { value.store(Next(), std::memory_order_relaxed); }
    std::uint64_t Value() const { return value.load(std::memory_order_relaxed); }
    // Touched since the entry took its place in the list.
    bool Moved() const { return Value() != placed; }
    void Place() { placed = Value(); }
    static bool Enabled() {
        static const bool disabled = std::getenv("APS5_NO_LRU_STAMPS") != nullptr;
        return !disabled;
    }

private:
    static std::uint64_t Next() {
        static std::atomic<std::uint64_t> counter{0};
        return counter.fetch_add(1, std::memory_order_relaxed) + 1;
    }
    std::atomic<std::uint64_t> value;
    // Written only with the list's lock held exclusively (placing, ordering).
    std::uint64_t placed;
};

// A hit on `it` (the entry type has an LruStamp named `recency`).
template<typename TList>
void TouchLru(TList& list, typename TList::iterator it) {
    it->recency.Touch();
    if (LruStamp::Enabled()) return;
    list.splice(list.begin(), list, it);
    it->recency.Place();
}

// Moves `it` to the front as the most recently used entry, with stamps or without.
template<typename TList>
void PromoteLru(TList& list, typename TList::iterator it) {
    it->recency.Touch();
    list.splice(list.begin(), list, it);
    it->recency.Place();
}

// Stamps the entry just put at the front: it is the most recently used one now, whenever it was made.
template<typename TList>
void PlacedFrontLru(TList& list) {
    list.front().recency.Touch();
    list.front().recency.Place();
}

// Restores use order (front = most recent) before an eviction takes entries from the back: the
// entries touched since they were placed are taken out, sorted by stamp and merged back. The rest
// are in stamp order already, so this costs one walk of the list plus a sort of the touched ones,
// and it keeps every iterator valid. A no-op with stamps off, where hits keep the order.
template<typename TList>
void OrderLru(TList& list) {
    if (!LruStamp::Enabled()) return;
    TList moved;
    for (auto it = list.begin(); it != list.end();) {
        const auto next = std::next(it);
        if (it->recency.Moved()) moved.splice(moved.end(), list, it);
        it = next;
    }
    if (moved.empty()) return;
    const auto newer = [](const auto& a, const auto& b) { return a.recency.Value() > b.recency.Value(); };
    moved.sort(newer);
    for (auto& entry : moved) entry.recency.Place();
    list.merge(moved, newer);
}

// Makes the back the least recently used entry, for an eviction that takes only the back. An
// untouched back already is: every entry in front of it was placed, and so last used, after it.
// Only a touched one costs the ordering, after which the list holds no touched entry.
template<typename TList>
void OrderLruBack(TList& list) {
    if (!list.empty() && list.back().recency.Moved()) OrderLru(list);
}

}

#endif
