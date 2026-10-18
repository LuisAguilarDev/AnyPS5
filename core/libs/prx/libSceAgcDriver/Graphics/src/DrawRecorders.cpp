#include "prx/libSceAgcDriver/Graphics/include/DrawRecorders.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/HandoffSpin.hpp"
#include "prx/libc/include/CpuTopology.hpp"
#include <cstdio>
#include <stdexcept>
#include <string>

namespace AgcDriver::Graphics {

namespace {

std::atomic<std::uint64_t> failAtDraw{0};
std::atomic<std::uint64_t> recordedDraws{0};

}

void DrawRecorders::FailAtDraw(std::uint64_t n) {
    recordedDraws.store(0, std::memory_order_relaxed);
    failAtDraw.store(n, std::memory_order_relaxed);
}

DrawRecorders::DrawRecorders(const Context& context, Recorder& owner, std::size_t threads) : context(context), owner(owner) {
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

void DrawRecorders::run(Worker& worker) {
    CpuTopology::PinHelperThread("draw recorder");
    Recorder::RecordingOffLock() = true;
    for (;;) {
        Item item;
        {
            std::unique_lock lock(worker.mutex);
            worker.wake.wait(lock, [&] { return worker.stopping || !worker.items.empty(); });
            if (worker.items.empty()) return;
            item = std::move(worker.items.front());
            worker.items.pop_front();
        }
        worker.room.notify_one();
        auto& chunk = *item.chunk;
        if (item.seal) {
            finish(worker, chunk);
            continue;
        }
        if (chunk.failure != nullptr) continue;
        try {
            if (chunk.commands == VK_NULL_HANDLE) begin(worker, chunk);
            if (const auto failAt = failAtDraw.load(std::memory_order_relaxed); failAt != 0 && recordedDraws.fetch_add(1, std::memory_order_relaxed) + 1 == failAt) throw std::runtime_error("draw recorder test failure");
            item.work(*worker.recording);
        } catch (...) {
            chunk.failure = std::current_exception();
        }
    }
}

}
