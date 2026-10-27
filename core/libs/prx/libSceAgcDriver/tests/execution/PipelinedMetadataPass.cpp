#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ColorTargetLayout.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Submit/include/Dcb.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/Shutdown.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <thread>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace {

constexpr std::uint32_t Width = 64;
constexpr std::uint32_t Height = 4;
constexpr std::uint32_t Stale = 0x5a5a5a5au;
constexpr std::uint32_t Clear = 0x11223344u;

constexpr std::size_t MemoryBytes = 65536;

std::byte* allocateMemory() {
#ifdef _WIN32
    auto* memory = static_cast<std::byte*>(VirtualAlloc(nullptr, MemoryBytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#else
    auto* memory = static_cast<std::byte*>(std::aligned_alloc(MemoryBytes, MemoryBytes));
#endif
    if (memory == nullptr) throw std::runtime_error("pipelined metadata pass: cannot allocate the surface memory");
    std::memset(memory, 0, MemoryBytes);
    return memory;
}

std::byte* const Memory = allocateMemory();
const std::span<std::uint32_t, Width * Height> Surface{reinterpret_cast<std::uint32_t*>(Memory), Width * Height};
const std::span<std::uint8_t, 4096> Cmask{reinterpret_cast<std::uint8_t*>(Memory + 4096), 4096};

// A plain draw into a target of its own (the pixel stage adds user word 2 to a constant it loads).
alignas(256) constexpr std::array<std::uint32_t, 16> TriangleVertexCode{
    0x7e000d05u, 0x080200f4u, 0x10020300u, 0x100202f4u, 0x060202f3u, 0x060400f3u, 0x10040500u, 0x060404f3u,
    0x7e060280u, 0x7e0802f2u, 0xf80008cfu, 0x04030201u, 0xbf810000u,
};
alignas(256) constexpr std::array<std::uint32_t, 10> ArithmeticPixelCode{0xf4000100u, 0xfa000000u, 0xbf8cc07fu, 0x7e000204u, 0x06000002u, 0x7e020280u, 0xf800180fu, 0x00010100u, 0xbf810000u, 0xbf810000u};
alignas(256) std::array<float, 4> PixelConstant{};
alignas(256) std::array<std::byte, 1024> ColorMemory{};
alignas(256) std::array<std::uint32_t, 4> Label{};

struct Header {
    Shader shader{};
    ShaderUserData userData{};
};

alignas(8) Header VertexHeader{};
alignas(8) Header PixelHeader{};

void registered(Header& header, const std::uint32_t* code, std::size_t bytes, std::uint8_t type) {
    header.shader.file_header = 0x34333231u;
    header.shader.version = 0x18u;
    header.shader.header_size = sizeof(Header);
    header.shader.shader_size = static_cast<std::uint32_t>(bytes);
    header.shader.code = code;
    header.shader.type = type;
    header.shader.user_data = &header.userData;
    AgcDriverRegisterShader_nid_postfix(&header.shader);
}

void check(bool condition, const std::string& what) {
    if (!condition) throw std::runtime_error("pipelined metadata pass: " + what);
}

void setContext(std::vector<std::uint32_t>& words, std::uint32_t offset, std::uint32_t value) {
    words.insert(words.end(), {0xc0016900u, offset, value});
}

void setShader(std::vector<std::uint32_t>& words, std::uint32_t offset, std::uint32_t value) {
    words.insert(words.end(), {0xc0017600u, offset, value});
}

void colorDraw(std::vector<std::uint32_t>& words) {
    const auto color = reinterpret_cast<std::uintptr_t>(ColorMemory.data());
    const auto vertex = reinterpret_cast<std::uintptr_t>(TriangleVertexCode.data());
    const auto pixel = reinterpret_cast<std::uintptr_t>(ArithmeticPixelCode.data());
    const auto table = reinterpret_cast<std::uintptr_t>(PixelConstant.data());
    const std::vector<std::pair<std::uint32_t, std::uint32_t>> context{
        {0x2d5, 0x2000},
        {0x1b6, 0}, {0x207, 0}, {0x200, 0}, {0x203, 0x800},
        {0x2dc, 0xaa00}, {0x2f8, 0}, {0x292, 2}, {0x293, 0},
        {0x80, 0}, {0x8d, 0}, {0x83, 0xffff}, {0x8c, 0xa},
        {0x2f9, 0x2d}, {0x313, 0x6000}, {0x30e, 0xffffffff}, {0x30f, 0xffffffff},
        {0x206, 0x43f}, {0x204, 0x80000}, {0x205, 0x240},
        {0x8e, 0xf}, {0x8f, 0xf}, {0x202, 0xcc0010},
        {0x1c4, 0}, {0x1c5, 9}, {0x1c3, 4}, {0x31c, 0x28028},
        {0x31b, 0}, {0x31d, 0}, {0x3b0, (63u << 14u) | 3u},
        {0x3b8, 0x9000000}, {0x1e0, 0},
        {0xc, 0}, {0xd, 0x40040},
        {0x81, 0x80000000}, {0x82, 0x40040},
        {0x90, 0x80000000}, {0x91, 0x40040},
        {0x94, 0x80000000}, {0x95, 0x40040},
        {0x318, static_cast<std::uint32_t>(color >> 8u)}, {0x390, static_cast<std::uint32_t>(color >> 40u)},
        {0x10f, std::bit_cast<std::uint32_t>(32.0f)}, {0x110, std::bit_cast<std::uint32_t>(32.0f)},
        {0x111, std::bit_cast<std::uint32_t>(-2.0f)}, {0x112, std::bit_cast<std::uint32_t>(2.0f)},
        {0x113, std::bit_cast<std::uint32_t>(1.0f)}, {0x114, 0},
        {0xb4, 0}, {0xb5, std::bit_cast<std::uint32_t>(1.0f)},
        {0x2fe, 0}, {0x302, 0}, {0x306, 0}, {0x30a, 0},
    };
    words.insert(words.end(), {0xc0017900u, 0x242u, 4u});
    for (const auto& [offset, value] : context) setContext(words, offset, value);
    setShader(words, 0xc8, static_cast<std::uint32_t>(vertex >> 8u));
    setShader(words, 0xc9, static_cast<std::uint32_t>(vertex >> 40u));
    setShader(words, 0x8b, 0);
    setShader(words, 0x8c, 0);
    setShader(words, 0x008, static_cast<std::uint32_t>(pixel >> 8u));
    setShader(words, 0x009, static_cast<std::uint32_t>(pixel >> 40u));
    setShader(words, 0x00b, 3u << 1u);
    setShader(words, 0x00c, static_cast<std::uint32_t>(table));
    setShader(words, 0x00d, static_cast<std::uint32_t>(table >> 32u));
    setShader(words, 0x00e, std::bit_cast<std::uint32_t>(0.25f));
    words.insert(words.end(), {0xc0012d00u, 3u, 2u});
}

void writeData(std::vector<std::uint32_t>& words, const void* destination, std::uint32_t value) {
    const auto pointer = reinterpret_cast<std::uintptr_t>(destination);
    words.insert(words.end(), {0xc0033700u, 0x00100200u, static_cast<std::uint32_t>(pointer), static_cast<std::uint32_t>(pointer >> 32u), value});
}

unsigned brightestColorByte() {
    AgcDriver::GuestMemory::FlushGpuWrites(reinterpret_cast<std::uintptr_t>(ColorMemory.data()), ColorMemory.size());
    unsigned brightest = 0;
    for (const auto value : ColorMemory) brightest = std::max(brightest, static_cast<unsigned>(std::to_integer<std::uint8_t>(value)));
    return brightest;
}

void dmaFill(std::vector<std::uint32_t>& words, std::uint32_t data, const void* destination, std::uint32_t bytes) {
    const auto address = reinterpret_cast<std::uintptr_t>(destination);
    words.insert(words.end(), {0xc0055000u, 2u << 29u, data, 0u, static_cast<std::uint32_t>(address), static_cast<std::uint32_t>(address >> 32u), bytes});
}

void eliminate(std::vector<std::uint32_t>& words) {
    const auto surface = reinterpret_cast<std::uintptr_t>(Surface.data());
    const auto cmask = reinterpret_cast<std::uintptr_t>(Cmask.data());
    words.insert(words.end(), {0xc0017900u, 0x242u, 4u});
    words.insert(words.end(), {0xc0017600u, 0x008u, 0x100u, 0xc0017600u, 0x009u, 0u});
    const std::vector<std::pair<std::uint32_t, std::uint32_t>> context{
        {0x2d5, 0x2000}, {0x1b6, 0}, {0x1b8, 0}, {0x207, 0}, {0x200, 0}, {0x203, 0x800},
        {0x2dc, 0xaa00}, {0x2f8, 0}, {0x292, 2}, {0x293, 0},
        {0x80, 0}, {0x8d, 0}, {0x83, 0xffff}, {0x8c, 0xa},
        {0x2f9, 0x2d}, {0x313, 0x6000}, {0x30e, 0xffffffff}, {0x30f, 0xffffffff},
        {0x206, 0x43f}, {0x204, 0x80000}, {0x205, 0x240},
        {0x8e, 0xf}, {0x8f, 0xf}, {0x202, 0xcc0020},
        {0x1c4, 0}, {0x1c5, 9}, {0x1c3, 4}, {0x31c, 0x28028u | 0x2000u},
        {0x31b, 0}, {0x31d, 0}, {0x3b0, (63u << 14u) | 3u},
        {0x3b8, 0x9000000u | (0x18u << 19u) | 0x4000000u}, {0x1e0, 0},
        {0xc, 0}, {0xd, 0x40040},
        {0x81, 0x80000000}, {0x82, 0x40040},
        {0x90, 0x80000000}, {0x91, 0x40040},
        {0x94, 0x80000000}, {0x95, 0x40040},
        {0x318, static_cast<std::uint32_t>(surface >> 8u)}, {0x390, static_cast<std::uint32_t>(surface >> 40u)},
        {0x31f, static_cast<std::uint32_t>(cmask >> 8u)}, {0x398, static_cast<std::uint32_t>(cmask >> 40u)},
        {0x323, Clear}, {0x324, 0},
        {0x0, 0}, {0xb4, 0},
        {0x10f, std::bit_cast<std::uint32_t>(32.0f)}, {0x110, std::bit_cast<std::uint32_t>(32.0f)},
        {0x111, std::bit_cast<std::uint32_t>(-2.0f)}, {0x112, std::bit_cast<std::uint32_t>(2.0f)},
        {0x113, std::bit_cast<std::uint32_t>(1.0f)}, {0x114, 0}, {0xb5, std::bit_cast<std::uint32_t>(1.0f)},
    };
    for (const auto& [offset, value] : context) setContext(words, offset, value);
    words.insert(words.end(), {0xc0002f00u, 1u});
    words.insert(words.end(), {0xc0012d00u, 3u, 2u});
}

void submit(std::vector<std::uint32_t>& words) {
    Packet packet{words.data(), static_cast<std::uint32_t>(words.size()), 0, {}};
    check(sceAgcDriverSubmitDcb(&packet) == 0, "submission failed");
    AgcDriverWaitIdle_nid_postfix();
}

void expectEliminated(const std::string& what) {
    {
        std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
        AgcDriver::Graphics::StorageTexture::FlushPending(reinterpret_cast<std::uintptr_t>(Surface.data()), Surface.size_bytes(), nullptr, "test");
    }
    for (std::size_t i = 0; i < Surface.size(); ++i) check(Surface[i] == Clear, what + ": texel " + std::to_string(i) + " holds " + std::to_string(Surface[i]));
    const auto cmaskBytes = AgcDriver::Graphics::CmaskLayout(Width, Height).Bytes();
    for (std::size_t i = 0; i < cmaskBytes; ++i) check(Cmask[i] == 0xffu, what + ": the CMASK was not left expanded");
}

// Holds the committer behind a commit of its own until `release` is set by a thread that, after
// 300 ms, notes whether the label had been written by then.
struct HeldCommitter {
    explicit HeldCommitter(std::atomic<bool>& early) : early(early) {
        AgcDriver::DriverDetail::DrawPipeline::Queue0().Enqueue([this] {
            while (!release.load(std::memory_order_acquire)) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }, {});
        releaser = std::thread([this] {
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
            this->early.store(std::atomic_ref<std::uint32_t>(Label[0]).load(std::memory_order_acquire) != 0, std::memory_order_release);
            release.store(true, std::memory_order_release);
        });
    }
    ~HeldCommitter() {
        if (releaser.joinable()) releaser.join();
    }
    std::atomic<bool>& early;
    std::atomic<bool> release{false};
    std::thread releaser;
};

// A draw, a CB metadata pass and a label behind them keep their program order whether the pass is
// a commit of the pipeline or runs inline after a drain: while the committer is held behind an
// earlier commit nothing behind it lands (the label is not written early), and once the label is
// written the draw and the pass have landed too.
void drawPassLabelOrder(bool pipelined) {
    using AgcDriver::DriverDetail::DrawPipeline;
    registered(VertexHeader, TriangleVertexCode.data(), sizeof(TriangleVertexCode), 2);
    registered(PixelHeader, ArithmeticPixelCode.data(), sizeof(ArithmeticPixelCode), 1);
    PixelConstant[0] = 0.25f;
    std::atomic_ref<std::uint32_t>(Label[0]).store(0, std::memory_order_release);
    const auto before = DrawPipeline::PipelinedMetadataPasses();
    std::vector<std::uint32_t> words;
    dmaFill(words, Stale, Surface.data(), static_cast<std::uint32_t>(Surface.size_bytes()));
    dmaFill(words, 0, Cmask.data(), static_cast<std::uint32_t>(Cmask.size()));
    colorDraw(words);
    eliminate(words);
    writeData(words, Label.data(), 1);
    std::atomic<bool> early{false};
    {
        HeldCommitter held(early);
        submit(words);
    }
    check(!early.load(std::memory_order_acquire), "the label behind the draw and the pass was written while the committer was held ahead of them");
    check(std::atomic_ref<std::uint32_t>(Label[0]).load(std::memory_order_acquire) == 1, "the label behind the draw and the pass was not written");
    const auto passes = DrawPipeline::PipelinedMetadataPasses() - before;
    check(passes == (pipelined ? 1u : 0u), "the pass between a draw and a label was " + std::string(passes != 0 ? "" : "not ") + "pipelined");
    expectEliminated("the eliminate between a draw and a label");
    const auto brightest = brightestColorByte();
    check(brightest + 2 >= 128 && brightest <= 130, "the draw ahead of the pass wrote " + std::to_string(brightest) + ", expected 128");
}

}

// The passes are pipelined by default, with or without APS5_DRAW_PIPELINE=1;
// APS5_NO_PIPELINED_METADATA_PASSES=1 runs them inline, in the same order.
int main() {
#ifdef _WIN32
    _putenv_s("APS5_PIPELINED_DRAWS", "1");
#else
    setenv("APS5_PIPELINED_DRAWS", "1", 1);
#endif
    const bool pipelined = std::getenv("APS5_NO_PIPELINED_METADATA_PASSES") == nullptr;
    try {
        {
            GuestAllocations::Mutation mutation;
            mutation.Add(Surface.data(), Surface.size_bytes(), true, true, true);
            mutation.Add(Cmask.data(), Cmask.size(), true, true, true);
        }
        using AgcDriver::DriverDetail::DrawPipeline;
        std::fill(Surface.begin(), Surface.end(), Stale);
        std::fill(Cmask.begin(), Cmask.end(), std::uint8_t{0});
        std::vector<std::uint32_t> first;
        eliminate(first);
        submit(first);
        expectEliminated("the first eliminate");
        const auto before = DrawPipeline::PipelinedMetadataPasses();
        std::vector<std::uint32_t> second;
        std::atomic<bool> release{false};
        DrawPipeline::Queue0().Enqueue([&release] {
            while (!release.load(std::memory_order_acquire)) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }, {});
        std::thread releaser([&release] {
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
            release.store(true, std::memory_order_release);
        });
        dmaFill(second, Stale, Surface.data(), static_cast<std::uint32_t>(Surface.size_bytes()));
        dmaFill(second, 0, Cmask.data(), static_cast<std::uint32_t>(Cmask.size()));
        eliminate(second);
        submit(second);
        releaser.join();
        check(DrawPipeline::PipelinedMetadataPasses() == before + (pipelined ? 1u : 0u), "the second eliminate was not ordered with the commits (" + std::to_string(DrawPipeline::PipelinedMetadataPasses() - before) + " pipelined)");
        expectEliminated("the eliminate after the queued stores");
        drawPassLabelOrder(pipelined);
        {
            GuestAllocations::Mutation mutation;
            mutation.Remove(Surface.data());
            mutation.Remove(Cmask.data());
        }
        LibcRunShutdown_nid_postfix();
        std::puts(pipelined ? "pipelined metadata pass tests passed" : "inline metadata pass tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        try { LibcRunShutdown_nid_postfix(); } catch (...) {}
        return 1;
    }
}
