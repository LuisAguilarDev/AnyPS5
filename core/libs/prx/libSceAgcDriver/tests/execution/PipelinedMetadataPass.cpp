#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "VulkanTestDevice.hpp"
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <initializer_list>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

using AgcDriver::Graphics::Require;
using AgcDriver::DriverDetail::DrawPipeline;

constexpr std::uint32_t DrawIndexAuto = 0x2d;

void Registers(std::vector<std::uint32_t>& words, std::uint32_t opcode, std::uint32_t first, std::initializer_list<std::uint32_t> values) {
    words.push_back(0xc0000000u | (static_cast<std::uint32_t>(values.size()) << 16u) | (opcode << 8u));
    words.push_back(first);
    words.insert(words.end(), values.begin(), values.end());
}

std::vector<std::uint32_t> EliminateFastClearWithoutTargets() {
    std::vector<std::uint32_t> words;
    constexpr std::uint32_t context = 0x69, userConfig = 0x79;
    Registers(words, context, 0x202, {0x00cc0020u});
    Registers(words, context, 0x010, {0, 0});
    Registers(words, context, 0x2f8, {0});
    Registers(words, context, 0x080, {0});
    Registers(words, context, 0x206, {0x43fu});
    Registers(words, context, 0x10f, {std::bit_cast<std::uint32_t>(64.0f), std::bit_cast<std::uint32_t>(64.0f), std::bit_cast<std::uint32_t>(64.0f), std::bit_cast<std::uint32_t>(64.0f)});
    Registers(words, context, 0x00c, {0, 0x40004000u});
    Registers(words, context, 0x081, {0, 0x40004000u});
    Registers(words, context, 0x090, {0, 0x40004000u});
    Registers(words, context, 0x292, {0});
    Registers(words, context, 0x08e, {0});
    Registers(words, userConfig, 0x24a, {0});
    words.insert(words.end(), {0xc0002f00u, 1});
    words.insert(words.end(), {0xc0012d00u, 3, 2});
    return words;
}

struct Blocker {
    std::atomic<bool> release{false};
    void Enqueue() {
        DrawPipeline::Queue0().Enqueue([this] {
            while (!release.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }, {});
    }
};

void Finish() {
    auto idle = std::async(std::launch::async, [] { AgcDriverWaitIdle_nid_postfix(); });
    if (idle.wait_for(std::chrono::seconds(20)) != std::future_status::ready) {
        std::fputs("pipelined metadata pass: the queue worker and the draw pipeline deadlocked\n", stderr);
        std::fflush(stderr);
        std::_Exit(1);
    }
    idle.get();
}

void MetadataPassBehindQueuedWork() {
    const auto drainsBefore = DrawPipeline::DrainRequestsByOpcode(DrawIndexAuto);
    Blocker blocker;
    blocker.Enqueue();
    auto words = EliminateFastClearWithoutTargets();
    Packet packet{words.data(), static_cast<std::uint32_t>(words.size()), 0, {}};
    AgcDriver::Submit(&packet, 0);
    const auto start = std::chrono::steady_clock::now();
    bool queued = false;
    while (std::chrono::steady_clock::now() - start < std::chrono::seconds(5)) {
        if (DrawPipeline::DrainRequestsByOpcode(DrawIndexAuto) != drainsBefore) break;
        if (DrawPipeline::Queue0().Outstanding() >= 2) {
            queued = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    blocker.release = true;
    Finish();
    Require(DrawPipeline::DrainRequestsByOpcode(DrawIndexAuto) == drainsBefore, "a CB metadata pass drained the draw pipeline");
    Require(queued, "the CB metadata pass was not queued behind the pending item");
}

}

int main() {
#ifdef _WIN32
    _putenv_s("APS5_PIPELINED_DRAWS", "1");
#else
    setenv("APS5_PIPELINED_DRAWS", "1", 1);
#endif
    try {
        {
            const auto device = OpenVulkanTestDevice();
            if (!device) return VulkanTestSkipped;
        }
        MetadataPassBehindQueuedWork();
        AgcDriverShutdown_nid_postfix();
        std::puts("pipelined metadata pass tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        AgcDriverShutdown_nid_postfix();
        return 1;
    }
}
