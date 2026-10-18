#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_DRAWRECORDERS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_DRAWRECORDERS_HPP

#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace AgcDriver::Graphics {

// The commands of consecutive lean draws, recorded by a recorder thread into a command buffer of
// its own while the spine (the thread holding GuestMemory::GpuMutex) resolves the next draws. The
// spine appends the chunk to the open batch in program order (Recorder::JoinChunk) and submits its
// command buffer between the spine's own ones once the recorder thread finished it
// (Recorder::Submit). A chunk is sealed (no draw joins it any more) as soon as the spine records
// anything else into the batch.
struct RecordedChunk {
    RecordedChunk() = default;
    RecordedChunk(const RecordedChunk&) = delete;
    RecordedChunk& operator=(const RecordedChunk&) = delete;
    // Written by the recorder thread, read by the spine once `done` is set.
    VkCommandBuffer commands = VK_NULL_HANDLE;
    std::vector<std::shared_ptr<void>> kept;
    std::vector<std::pair<const void*, std::size_t>> keptBytes;
    std::size_t bytes = 0;
    std::exception_ptr failure;
    // Returns the command buffer to the recorder thread that recorded it (any thread, once the
    // batch completed or was discarded).
    std::function<void(VkCommandBuffer)> recycle;
    std::atomic<bool> done{false};
    // Spine only: the draws that joined it, and whether its seal was dispatched.
    std::uint32_t draws = 0;
    bool sealed = false;
};

// The recorder threads (APS5_DRAW_RECORD_THREADS) of one recorder: each records the chunks given
// to it, in order, through a recording context of its own (its draw snapshot arena and caches)
// into command buffers from a command pool of its own, and never takes GuestMemory::GpuMutex, so
// the spine may wait for it while holding the mutex. Each thread runs on the cores of
// APS5_DRAW_RECORD_AFFINITY_MASK (a hex mask; unset: the performance cores of a hybrid CPU, else
// anywhere). Work is queued per thread with a bounded depth (the spine waits for room); a chunk's
// draws and its seal go to the same thread.
class DrawRecorders {
public:
    // What a record half reports for the [record] lines (APS5_PROFILE_DRAW).
    struct DrawStats {
        double bindingsUs = 0;
        double commandsUs = 0;
        std::uint64_t freshSets = 0;
        bool passBegun = false;
    };
    using Work = std::function<void(RecordingContext&, DrawStats&)>;
    // APS5_PROFILE_DRAW: a recorder thread fills DrawStats and the [record] lines are printed.
    static bool Profiled();
    DrawRecorders(const Context& context, Recorder& owner, std::size_t threads);
    ~DrawRecorders();
    DrawRecorders(const DrawRecorders&) = delete;
    DrawRecorders& operator=(const DrawRecorders&) = delete;
    // The recorder thread a new chunk goes to (round robin).
    std::size_t Assign();
    void Record(std::size_t thread, const std::shared_ptr<RecordedChunk>& chunk, Work work);
    void Seal(std::size_t thread, const std::shared_ptr<RecordedChunk>& chunk);
    static void Wait(const RecordedChunk& chunk);
    std::size_t Threads() const { return workers.size(); }
    // The most work items any thread had queued at once since start-up (the flood test's cap check).
    std::size_t MaxQueued() const { return maxQueued.load(std::memory_order_relaxed); }
    static constexpr std::size_t QueueDepth = 256;
    // Test aid: throws from the recorder thread at its `n`th draw (1-based; 0 never).
    static void FailAtDraw(std::uint64_t n);

private:
    struct Item {
        std::shared_ptr<RecordedChunk> chunk;
        Work work;
        bool seal = false;
    };
    struct Worker {
        std::mutex mutex;
        std::condition_variable wake;
        std::condition_variable room;
        std::deque<Item> items;
        bool stopping = false;
        VkCommandPool pool = VK_NULL_HANDLE;
        std::unique_ptr<RecordingContext> recording;
        // Command buffers of completed batches, returned by RecordedChunk::recycle (any thread).
        std::mutex returnedMutex;
        std::vector<VkCommandBuffer> returned;
        std::vector<VkCommandBuffer> free;
        std::thread thread;
        // The [record] counters since the last report (nanoseconds for the times).
        std::atomic<std::uint64_t> chunks{0};
        std::atomic<std::uint64_t> draws{0};
        std::atomic<std::uint64_t> busyNs{0};
        std::atomic<std::uint64_t> idleNs{0};
        std::atomic<std::uint64_t> bindingsNs{0};
        std::atomic<std::uint64_t> commandsNs{0};
        std::atomic<std::uint64_t> freshSets{0};
        std::atomic<std::uint64_t> passesBegun{0};
    };
    void run(Worker& worker);
    void begin(Worker& worker, RecordedChunk& chunk);
    void finish(Worker& worker, RecordedChunk& chunk) noexcept;
    void push(Worker& worker, Item item);
    void report();

    Context context;
    Recorder& owner;
    std::vector<std::unique_ptr<Worker>> workers;
    std::size_t next = 0;
    std::atomic<std::size_t> maxQueued{0};
    std::atomic<std::int64_t> lastReport;
};

}

#endif
