// The [gpu-time] breakdown under APS5_PROFILE_GPU=1 (set by the test's registration): render pass
// bodies are timed by target with their draws, and a device without timestamp queries times
// nothing instead of failing the batch.
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

using namespace AgcDriver::Graphics;

namespace {

int failures = 0;

void Expect(bool condition, const std::string& what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
}

// Each timestamp write takes the next tick of a clock (1 ns per tick), so a range's length is the
// number of stamps written from its begin to its end.
constexpr std::uint64_t TickNs = 1000;

struct MockDevice {
    std::uintptr_t next = 1;
    std::map<VkFence, bool> signaled;
    std::map<VkQueryPool, std::map<std::uint32_t, std::uint64_t>> stamps;
    std::uint64_t clock = 0;
    std::uint64_t createdPools = 0;
    bool timestamps = true;
};

MockDevice mock;

VKAPI_ATTR VkResult VKAPI_CALL mockAllocateCommandBuffers(VkDevice, const VkCommandBufferAllocateInfo*, VkCommandBuffer* commands) {
    *commands = reinterpret_cast<VkCommandBuffer>(mock.next++);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL mockFreeCommandBuffers(VkDevice, VkCommandPool, std::uint32_t, const VkCommandBuffer*) {}
VKAPI_ATTR VkResult VKAPI_CALL mockCreateFence(VkDevice, const VkFenceCreateInfo*, const VkAllocationCallbacks*, VkFence* fence) {
    *fence = reinterpret_cast<VkFence>(mock.next++);
    mock.signaled[*fence] = false;
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL mockDestroyFence(VkDevice, VkFence fence, const VkAllocationCallbacks*) { mock.signaled.erase(fence); }
VKAPI_ATTR VkResult VKAPI_CALL mockGetFenceStatus(VkDevice, VkFence fence) { return mock.signaled.at(fence) ? VK_SUCCESS : VK_NOT_READY; }
VKAPI_ATTR VkResult VKAPI_CALL mockWaitForFences(VkDevice, std::uint32_t count, const VkFence* fences, VkBool32, std::uint64_t) {
    for (std::uint32_t i = 0; i < count; ++i) mock.signaled.at(fences[i]) = true;
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL mockResetFences(VkDevice, std::uint32_t count, const VkFence* fences) {
    for (std::uint32_t i = 0; i < count; ++i) mock.signaled.at(fences[i]) = false;
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL mockBeginCommandBuffer(VkCommandBuffer, const VkCommandBufferBeginInfo*) { return VK_SUCCESS; }
VKAPI_ATTR VkResult VKAPI_CALL mockEndCommandBuffer(VkCommandBuffer) { return VK_SUCCESS; }
VKAPI_ATTR VkResult VKAPI_CALL mockQueueSubmit(VkQueue, std::uint32_t, const VkSubmitInfo*, VkFence) { return VK_SUCCESS; }
VKAPI_ATTR VkResult VKAPI_CALL mockDeviceWaitIdle(VkDevice) { return VK_SUCCESS; }
VKAPI_ATTR void VKAPI_CALL mockCmdPipelineBarrier(VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags, std::uint32_t, const VkMemoryBarrier*, std::uint32_t, const VkBufferMemoryBarrier*, std::uint32_t, const VkImageMemoryBarrier*) {}
VKAPI_ATTR void VKAPI_CALL mockCmdEndRenderPass(VkCommandBuffer) {}
VKAPI_ATTR void VKAPI_CALL mockCmdBindPipeline(VkCommandBuffer, VkPipelineBindPoint, VkPipeline) {}
VKAPI_ATTR void VKAPI_CALL mockCmdPushConstants(VkCommandBuffer, VkPipelineLayout, VkShaderStageFlags, std::uint32_t, std::uint32_t, const void*) {}
VKAPI_ATTR void VKAPI_CALL mockCmdDispatch(VkCommandBuffer, std::uint32_t, std::uint32_t, std::uint32_t) {}
VKAPI_ATTR void VKAPI_CALL mockCmdUpdateBuffer(VkCommandBuffer, VkBuffer, VkDeviceSize, VkDeviceSize, const void*) {}
VKAPI_ATTR void VKAPI_CALL mockCmdBeginQuery(VkCommandBuffer, VkQueryPool, std::uint32_t, VkQueryControlFlags) {}
VKAPI_ATTR void VKAPI_CALL mockCmdEndQuery(VkCommandBuffer, VkQueryPool, std::uint32_t) {}
VKAPI_ATTR void VKAPI_CALL mockCmdCopyQueryPoolResults(VkCommandBuffer, VkQueryPool, std::uint32_t, std::uint32_t, VkBuffer, VkDeviceSize, VkDeviceSize, VkQueryResultFlags) {}
VKAPI_ATTR VkResult VKAPI_CALL mockCreateQueryPool(VkDevice, const VkQueryPoolCreateInfo*, const VkAllocationCallbacks*, VkQueryPool* pool) {
    *pool = reinterpret_cast<VkQueryPool>(mock.next++);
    ++mock.createdPools;
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL mockDestroyQueryPool(VkDevice, VkQueryPool pool, const VkAllocationCallbacks*) { mock.stamps.erase(pool); }
VKAPI_ATTR void VKAPI_CALL mockCmdResetQueryPool(VkCommandBuffer, VkQueryPool pool, std::uint32_t, std::uint32_t) { mock.stamps[pool].clear(); }
VKAPI_ATTR void VKAPI_CALL mockCmdWriteTimestamp(VkCommandBuffer, VkPipelineStageFlagBits, VkQueryPool pool, std::uint32_t query) { mock.stamps[pool][query] = ++mock.clock * TickNs; }
VKAPI_ATTR VkResult VKAPI_CALL mockGetQueryPoolResults(VkDevice, VkQueryPool pool, std::uint32_t first, std::uint32_t count, std::size_t, void* data, VkDeviceSize stride, VkQueryResultFlags) {
    for (std::uint32_t i = 0; i < count; ++i) {
        const auto& written = mock.stamps[pool];
        const auto found = written.find(first + i);
        Expect(found != written.end(), "query " + std::to_string(first + i) + " was read without a timestamp written");
        *reinterpret_cast<std::uint64_t*>(static_cast<std::byte*>(data) + i * stride) = found != written.end() ? found->second : 0;
    }
    return VK_SUCCESS;
}

PFN_vkVoidFunction VKAPI_CALL mockProc(VkDevice, const char* name) {
    static const std::map<std::string_view, PFN_vkVoidFunction> table{
        {"vkAllocateCommandBuffers", reinterpret_cast<PFN_vkVoidFunction>(mockAllocateCommandBuffers)},
        {"vkFreeCommandBuffers", reinterpret_cast<PFN_vkVoidFunction>(mockFreeCommandBuffers)},
        {"vkCreateFence", reinterpret_cast<PFN_vkVoidFunction>(mockCreateFence)},
        {"vkDestroyFence", reinterpret_cast<PFN_vkVoidFunction>(mockDestroyFence)},
        {"vkGetFenceStatus", reinterpret_cast<PFN_vkVoidFunction>(mockGetFenceStatus)},
        {"vkWaitForFences", reinterpret_cast<PFN_vkVoidFunction>(mockWaitForFences)},
        {"vkResetFences", reinterpret_cast<PFN_vkVoidFunction>(mockResetFences)},
        {"vkBeginCommandBuffer", reinterpret_cast<PFN_vkVoidFunction>(mockBeginCommandBuffer)},
        {"vkEndCommandBuffer", reinterpret_cast<PFN_vkVoidFunction>(mockEndCommandBuffer)},
        {"vkQueueSubmit", reinterpret_cast<PFN_vkVoidFunction>(mockQueueSubmit)},
        {"vkDeviceWaitIdle", reinterpret_cast<PFN_vkVoidFunction>(mockDeviceWaitIdle)},
        {"vkCmdPipelineBarrier", reinterpret_cast<PFN_vkVoidFunction>(mockCmdPipelineBarrier)},
        {"vkCmdEndRenderPass", reinterpret_cast<PFN_vkVoidFunction>(mockCmdEndRenderPass)},
        {"vkCmdBindPipeline", reinterpret_cast<PFN_vkVoidFunction>(mockCmdBindPipeline)},
        {"vkCmdPushConstants", reinterpret_cast<PFN_vkVoidFunction>(mockCmdPushConstants)},
        {"vkCmdDispatch", reinterpret_cast<PFN_vkVoidFunction>(mockCmdDispatch)},
        {"vkCmdUpdateBuffer", reinterpret_cast<PFN_vkVoidFunction>(mockCmdUpdateBuffer)},
        {"vkCmdBeginQuery", reinterpret_cast<PFN_vkVoidFunction>(mockCmdBeginQuery)},
        {"vkCmdEndQuery", reinterpret_cast<PFN_vkVoidFunction>(mockCmdEndQuery)},
        {"vkCmdCopyQueryPoolResults", reinterpret_cast<PFN_vkVoidFunction>(mockCmdCopyQueryPoolResults)},
    };
    static const std::map<std::string_view, PFN_vkVoidFunction> queries{
        {"vkCreateQueryPool", reinterpret_cast<PFN_vkVoidFunction>(mockCreateQueryPool)},
        {"vkDestroyQueryPool", reinterpret_cast<PFN_vkVoidFunction>(mockDestroyQueryPool)},
        {"vkCmdResetQueryPool", reinterpret_cast<PFN_vkVoidFunction>(mockCmdResetQueryPool)},
        {"vkCmdWriteTimestamp", reinterpret_cast<PFN_vkVoidFunction>(mockCmdWriteTimestamp)},
        {"vkGetQueryPoolResults", reinterpret_cast<PFN_vkVoidFunction>(mockGetQueryPoolResults)},
    };
    if (const auto it = table.find(name); it != table.end()) return it->second;
    // Like the kept-bytes mock: query commands but no query pools.
    if (!mock.timestamps && std::string_view(name) == "vkCreateQueryPool") return nullptr;
    const auto it = queries.find(name);
    return it == queries.end() ? nullptr : it->second;
}

Context mockContext() {
    Context context{};
    context.device = reinterpret_cast<VkDevice>(mock.next++);
    context.queue = reinterpret_cast<VkQueue>(mock.next++);
    context.deviceProc = mockProc;
    context.limits.timestampPeriod = 1.0f;
    return context;
}

constexpr std::uint64_t FullKey = 0x1234;
constexpr std::uint64_t HalfKey = 0x5678;

Recorder::PassTarget FourK() {
    Recorder::PassTarget target;
    target.width = 3840;
    target.height = 2160;
    target.colors = 2;
    target.color = VK_FORMAT_R16G16B16A16_SFLOAT;
    target.depth = VK_FORMAT_D32_SFLOAT_S8_UINT;
    return target;
}

// One draw beginning a pass and `continued` more continuing it, as Draw.cpp records them.
void RecordPass(Recorder& recorder, std::uint64_t key, const Recorder::PassTarget& target, std::uint32_t continued) {
    const auto start = recorder.StartDrawPass(key, false, false, PassAccess{});
    Expect(!start.continued, "the first draw continued a pass");
    const auto timing = recorder.BeginGpuTiming(Recorder::CommandClass::Draw);
    recorder.BeginPassTiming(target);
    recorder.LeaveRenderPassOpen(key, timing, false, PassAccess{});
    recorder.CountPassDraw();
    for (std::uint32_t draw = 0; draw < continued; ++draw) {
        Expect(recorder.StartDrawPass(key, false, false, PassAccess{}).continued, "a draw of the same pass did not continue it");
        recorder.LeaveRenderPassOpen(key, Recorder::NoTiming, false, PassAccess{});
        recorder.CountPassDraw();
    }
}

const Recorder::PassTotals* TotalsFor(const std::vector<std::pair<Recorder::PassTarget, Recorder::PassTotals>>& totals, const Recorder::PassTarget& target) {
    for (const auto& [key, value] : totals) {
        if (key == target) return &value;
    }
    return nullptr;
}

void NoTimestampQueries() {
    mock = MockDevice{};
    mock.timestamps = false;
    std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
    Recorder recorder(mockContext());
    Expect(recorder.BeginGpuTiming(Recorder::CommandClass::Copy) == Recorder::NoTiming, "a device without vkCreateQueryPool returned a timed range");
    auto target = FourK();
    target.width = 1;
    RecordPass(recorder, FullKey, target, 1);
    recorder.Sync();
    Expect(recorder.Idle(), "the batch without timestamp queries did not finish");
    Expect(TotalsFor(Recorder::PassTimingTotals(), target) == nullptr, "a pass was timed without timestamp queries");
}

void PassBodiesByTarget() {
    mock = MockDevice{};
    std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
    Recorder recorder(mockContext());
    const auto target = FourK();
    RecordPass(recorder, FullKey, target, 2);
    auto half = target;
    half.width /= 2;
    half.height /= 2;
    // A different target ends the first pass: its body ends at the pass's end, before the
    // trailing barrier and the draw class range's end.
    RecordPass(recorder, HalfKey, half, 0);
    recorder.Sync();
    const auto totals = Recorder::PassTimingTotals();
    const auto* full = TotalsFor(totals, target);
    const auto* small = TotalsFor(totals, half);
    Expect(full != nullptr && full->passes == 1 && full->draws == 3, "the 4K pass was not counted once with its three draws");
    Expect(small != nullptr && small->passes == 1 && small->draws == 1, "the half-size pass was not counted once with its draw");
    // Body begin and end are consecutive stamps: one tick.
    if (full != nullptr) Expect(std::abs(full->ms - TickNs / 1e6) < 1e-9, "the 4K pass body took " + std::to_string(full->ms) + " ms, not one tick");
    Expect(mock.createdPools == 1, "one batch created " + std::to_string(mock.createdPools) + " query pools");
    // A second batch reuses the released pool.
    RecordPass(recorder, FullKey, target, 0);
    recorder.Sync();
    Expect(mock.createdPools == 1, "the second batch created a query pool instead of reusing the released one");
    const auto* again = TotalsFor(Recorder::PassTimingTotals(), target);
    Expect(again != nullptr && again->passes == 2 && again->draws == 4, "the second batch's 4K pass was not added to the totals");
}

}

int main() {
    if (!Recorder::GpuTimingEnabled()) {
        std::fprintf(stderr, "APS5_PROFILE_GPU is not set: the recorder GPU timing tests need it\n");
        return 77;
    }
    try {
        NoTimestampQueries();
        PassBodiesByTarget();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    if (failures != 0) {
        std::fprintf(stderr, "%d recorder GPU timing checks failed\n", failures);
        return 1;
    }
    std::printf("recorder GPU timing tests passed\n");
    return 0;
}
