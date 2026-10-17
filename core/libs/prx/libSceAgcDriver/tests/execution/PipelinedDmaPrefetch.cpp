#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Submit/include/Dcb.hpp"
#include "prx/libc/include/Shutdown.hpp"
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

void check(bool condition, const std::string& what) {
    if (!condition) throw std::runtime_error("pipelined DMA prefetch: " + what);
}

constexpr std::uint32_t DmaPrefetchDestination = 2u << 20u;

std::array<std::uint32_t, 7> DmaData(std::uint32_t control, std::uint64_t source, std::uint64_t destination, std::uint32_t bytes) {
    return {0xc0055000u, control, static_cast<std::uint32_t>(source), static_cast<std::uint32_t>(source >> 32u), static_cast<std::uint32_t>(destination), static_cast<std::uint32_t>(destination >> 32u), bytes};
}

void Submit(std::vector<std::uint32_t>& words) {
    Packet packet{words.data(), static_cast<std::uint32_t>(words.size()), 0, {}};
    check(sceAgcDriverSubmitDcb(&packet) == 0, "submission failed");
}

}

int main() {
#ifdef _WIN32
    _putenv_s("APS5_PIPELINED_DRAWS", "1");
    _putenv_s("APS5_PIPELINE_DMA", "1");
#else
    setenv("APS5_PIPELINED_DRAWS", "1", 1);
    setenv("APS5_PIPELINE_DMA", "1", 1);
#endif
    try {
        alignas(256) static std::array<std::uint32_t, 1024> source{};
        alignas(256) static std::array<std::uint32_t, 1024> destination{};
        for (std::size_t i = 0; i < source.size(); ++i) source[i] = 0xa5000000u | static_cast<std::uint32_t>(i);
        destination.fill(0);
        using AgcDriver::DriverDetail::DrawPipeline;
        const auto drainsBefore = DrawPipeline::DrainRequestsByOpcode(0x50);
        std::vector<std::uint32_t> words;
        const auto copy = DmaData(0, reinterpret_cast<std::uintptr_t>(source.data()), reinterpret_cast<std::uintptr_t>(destination.data()), static_cast<std::uint32_t>(sizeof(source)));
        const auto prefetch = DmaData(DmaPrefetchDestination, reinterpret_cast<std::uintptr_t>(source.data()), 0, static_cast<std::uint32_t>(sizeof(source)));
        words.insert(words.end(), copy.begin(), copy.end());
        words.insert(words.end(), prefetch.begin(), prefetch.end());
        {
            std::unique_lock gpu(AgcDriver::GuestMemory::GpuMutex());
            Submit(words);
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
            while (DrawPipeline::DrainRequestsByOpcode(0x50) == drainsBefore && std::chrono::steady_clock::now() < deadline) std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        AgcDriverWaitIdle_nid_postfix();
        for (std::size_t i = 0; i < destination.size(); ++i) check(destination[i] == source[i], "the copy queued before the prefetch did not land");
        check(DrawPipeline::DrainRequestsByOpcode(0x50) == drainsBefore, "a DMA_DATA prefetch drained the draw pipeline behind the queued copy");
        LibcRunShutdown_nid_postfix();
        std::puts("pipelined DMA prefetch tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        try { LibcRunShutdown_nid_postfix(); } catch (...) {}
        return 1;
    }
}
