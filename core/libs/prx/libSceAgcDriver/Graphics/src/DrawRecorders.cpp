#include "prx/libSceAgcDriver/Graphics/include/DrawRecorders.hpp"
#include "prx/libSceAgcDriver/Execution/include/LockProfile.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/HandoffSpin.hpp"
#include "prx/libc/include/CpuTopology.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace AgcDriver::Graphics {

namespace {

std::atomic<std::uint64_t> failAtDraw{0};
std::atomic<std::uint64_t> recordedDraws{0};

std::int64_t nowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

}

void DrawRecorders::FailAtDraw(std::uint64_t n) {
    recordedDraws.store(0, std::memory_order_relaxed);
    failAtDraw.store(n, std::memory_order_relaxed);
}

DrawRecorders::DrawRecorders(const Context& context, Recorder& owner, std::size_t threads) : context(context), owner(owner), lastReport(nowNs()) {
    Require(threads != 0, "draw recorders need a thread");
    for (std::size_t index = 0; index < threads; ++index) {
        auto worker = std::make_unique<Worker>();
        VkCommandPoolCreateInfo info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        info.queueFamilyIndex = context.queueFamily;
        Check(context.Function<PFN_vkCreateCommandPool>("vkCreateCommandPool")(context.device, &info, nullptr, &worker->pool), "vkCreateCommandPool draw recorder");
        // The recorder thread's draws reach no recorder through the context (a sampled draw's
        // NoteSampledDraw, say): everything they need of the batch went through the spine.
        auto own = context;
        own.recorder = nullptr;
        worker->recording = std::make_unique<RecordingContext>(own, owner, true);
        workers.push_back(std::move(worker));
    }
    for (std::size_t index = 0; index < workers.size(); ++index) {
        auto& worker = *workers[index];
        worker.thread = std::thread([this, &worker] { run(worker); });
    }
}

DrawRecorders::~DrawRecorders() {
    for (auto& worker : workers) {
        {
            std::lock_guard lock(worker->mutex);
            worker->stopping = true;
        }
        worker->wake.notify_all();
    }
    for (auto& worker : workers) {
        if (worker->thread.joinable()) worker->thread.join();
        // Every batch completed before the recorder destroys this (its destructor syncs), so no
        // command buffer of the pool is pending; destroying the pool frees them all.
        context.Function<PFN_vkDestroyCommandPool>("vkDestroyCommandPool")(context.device, worker->pool, nullptr);
    }
}

std::size_t DrawRecorders::Assign() {
    const auto thread = next;
    next = (next + 1) % workers.size();
    return thread;
}

void DrawRecorders::push(Worker& worker, Item item) {
    std::unique_lock lock(worker.mutex);
    // Bounded: the spine waits for room (the recorder thread never takes the GPU mutex, so it
    // always makes progress while the spine holds it).
    worker.room.wait(lock, [&] { return worker.items.size() < QueueDepth; });
    worker.items.push_back(std::move(item));
    const auto queued = worker.items.size();
    lock.unlock();
    worker.wake.notify_one();
    auto seen = maxQueued.load(std::memory_order_relaxed);
    while (queued > seen && !maxQueued.compare_exchange_weak(seen, queued, std::memory_order_relaxed)) {
    }
}

void DrawRecorders::Record(std::size_t thread, const std::shared_ptr<RecordedChunk>& chunk, Work work) {
    push(*workers.at(thread), Item{chunk, std::move(work), false});
}

void DrawRecorders::Seal(std::size_t thread, const std::shared_ptr<RecordedChunk>& chunk) {
    push(*workers.at(thread), Item{chunk, {}, true});
}

void DrawRecorders::Wait(const RecordedChunk& chunk) {
    if (SpinUntil(HandoffSpinWindow(), [&] { return chunk.done.load(std::memory_order_acquire); })) return;
    while (!chunk.done.load(std::memory_order_acquire)) chunk.done.wait(false, std::memory_order_acquire);
}

void DrawRecorders::begin(Worker& worker, RecordedChunk& chunk) {
    {
        std::lock_guard lock(worker.returnedMutex);
        worker.free.insert(worker.free.end(), worker.returned.begin(), worker.returned.end());
        worker.returned.clear();
    }
    VkCommandBuffer commands = VK_NULL_HANDLE;
    if (!worker.free.empty()) {
        commands = worker.free.back();
        worker.free.pop_back();
    } else {
        VkCommandBufferAllocateInfo allocation{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocation.commandPool = worker.pool;
        allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocation.commandBufferCount = 1;
        Check(context.Function<PFN_vkAllocateCommandBuffers>("vkAllocateCommandBuffers")(context.device, &allocation, &commands), "vkAllocateCommandBuffers draw recorder");
    }
    chunk.recycle = [&worker](VkCommandBuffer buffer) {
        std::lock_guard lock(worker.returnedMutex);
        worker.returned.push_back(buffer);
    };
    // The pool resets the buffer on begin.
    VkCommandBufferBeginInfo info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (const auto begun = context.Function<PFN_vkBeginCommandBuffer>("vkBeginCommandBuffer")(commands, &info); begun != VK_SUCCESS) {
        worker.free.push_back(commands);
        Check(begun, "vkBeginCommandBuffer draw recorder");
    }
    // Set once begun: finish ends (and Submit submits) only a begun buffer.
    chunk.commands = commands;
    // A chunk's snapshots see guest memory as of its own collects: the thread's collect epoch
    // (GuestMemory::CollectEpoch, per thread) starts anew with every chunk.
    GuestMemory::BumpCollectEpoch();
    worker.recording->BeginChunk(chunk, commands);
}

void DrawRecorders::finish(Worker& worker, RecordedChunk& chunk) noexcept {
    if (chunk.commands != VK_NULL_HANDLE) {
        try {
            if (chunk.failure == nullptr) worker.recording->EndChunk();
        } catch (...) {
            chunk.failure = std::current_exception();
        }
        worker.recording->LeaveChunk();
        const auto ended = context.Function<PFN_vkEndCommandBuffer>("vkEndCommandBuffer")(chunk.commands);
        if (ended != VK_SUCCESS && chunk.failure == nullptr) chunk.failure = std::make_exception_ptr(std::runtime_error("vkEndCommandBuffer draw recorder failed: " + std::to_string(static_cast<int>(ended))));
    }
    chunk.done.store(true, std::memory_order_release);
    chunk.done.notify_all();
}

bool DrawRecorders::Profiled() {
    static const bool profiled = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    return profiled;
}

void DrawRecorders::report() {
    constexpr std::int64_t window = 10'000'000'000;
    const auto now = nowNs();
    auto last = lastReport.load(std::memory_order_relaxed);
    if (now - last < window || !lastReport.compare_exchange_strong(last, now, std::memory_order_relaxed)) return;
    // [record] thread N and [record] balance over the window: the busy split into the bindings
    // (PrepareDrawBindings) and the rest of the record half, and every chunk's first pass begun
    // because of the chunk boundary.
    const auto ms = [](std::uint64_t ns) { return static_cast<double>(ns) / 1e6; };
    double maxBusy = 0;
    double sumBusy = 0;
    for (std::size_t index = 0; index < workers.size(); ++index) {
        auto& worker = *workers[index];
        const auto chunks = worker.chunks.exchange(0, std::memory_order_relaxed);
        const auto draws = worker.draws.exchange(0, std::memory_order_relaxed);
        const auto busy = ms(worker.busyNs.exchange(0, std::memory_order_relaxed));
        const auto idle = ms(worker.idleNs.exchange(0, std::memory_order_relaxed));
        const auto bindings = ms(worker.bindingsNs.exchange(0, std::memory_order_relaxed));
        const auto commands = ms(worker.commandsNs.exchange(0, std::memory_order_relaxed));
        const auto fresh = worker.freshSets.exchange(0, std::memory_order_relaxed);
        const auto passes = worker.passesBegun.exchange(0, std::memory_order_relaxed);
        maxBusy = std::max(maxBusy, busy);
        sumBusy += busy;
        std::fprintf(stderr, "[record] thread %zu over 10 s: %llu chunks, %llu draws (%.1f per chunk), busy %.1f ms (bindings %.1f, commands %.1f), idle %.1f ms, %llu fresh descriptor sets, %llu passes begun (%llu at a chunk boundary)\n", index, static_cast<unsigned long long>(chunks), static_cast<unsigned long long>(draws), chunks != 0 ? static_cast<double>(draws) / static_cast<double>(chunks) : 0.0, busy, bindings, commands, idle, static_cast<unsigned long long>(fresh), static_cast<unsigned long long>(passes), static_cast<unsigned long long>(chunks));
    }
    const auto meanBusy = sumBusy / static_cast<double>(workers.size());
    std::fprintf(stderr, "[record] balance over 10 s: %zu threads, max busy %.1f ms / mean busy %.1f ms = %.2f\n", workers.size(), maxBusy, meanBusy, meanBusy > 0 ? maxBusy / meanBusy : 0.0);
}

void DrawRecorders::run(Worker& worker) {
    // The recorder threads take the performance cores of a hybrid CPU unless the mask names others.
    static const std::uint64_t mask = [] {
        const auto named = CpuTopology::MaskFromEnvironment("APS5_DRAW_RECORD_AFFINITY_MASK");
        if (named != 0) return named;
        const auto& layout = CpuTopology::Get();
        return layout.hybrid ? layout.performant : std::uint64_t{0};
    }();
    if (mask != 0) CpuTopology::PinTraced("draw recorder", nullptr, mask);
    Recorder::RecordingOffLock() = true;
    TagLockRole(LockRole::Recorder);
    const bool profiled = Profiled();
    for (;;) {
        Item item;
        {
            std::unique_lock lock(worker.mutex);
            const auto idleStart = profiled ? nowNs() : 0;
            worker.wake.wait(lock, [&] { return worker.stopping || !worker.items.empty(); });
            if (profiled) worker.idleNs.fetch_add(static_cast<std::uint64_t>(nowNs() - idleStart), std::memory_order_relaxed);
            if (worker.items.empty()) return;
            item = std::move(worker.items.front());
            worker.items.pop_front();
        }
        worker.room.notify_one();
        const auto busyStart = profiled ? nowNs() : 0;
        auto& chunk = *item.chunk;
        if (item.seal) {
            finish(worker, chunk);
            if (profiled) worker.chunks.fetch_add(1, std::memory_order_relaxed);
        } else if (chunk.failure == nullptr) {
            DrawStats stats;
            try {
                if (chunk.commands == VK_NULL_HANDLE) begin(worker, chunk);
                if (const auto failAt = failAtDraw.load(std::memory_order_relaxed); failAt != 0 && recordedDraws.fetch_add(1, std::memory_order_relaxed) + 1 == failAt) throw std::runtime_error("draw recorder test failure");
                item.work(*worker.recording, stats);
            } catch (...) {
                chunk.failure = std::current_exception();
            }
            if (profiled) {
                worker.draws.fetch_add(1, std::memory_order_relaxed);
                worker.bindingsNs.fetch_add(static_cast<std::uint64_t>(stats.bindingsUs * 1000.0), std::memory_order_relaxed);
                worker.commandsNs.fetch_add(static_cast<std::uint64_t>(stats.commandsUs * 1000.0), std::memory_order_relaxed);
                worker.freshSets.fetch_add(stats.freshSets, std::memory_order_relaxed);
                if (stats.passBegun) worker.passesBegun.fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (profiled) {
            worker.busyNs.fetch_add(static_cast<std::uint64_t>(nowNs() - busyStart), std::memory_order_relaxed);
            report();
        }
    }
}

}
