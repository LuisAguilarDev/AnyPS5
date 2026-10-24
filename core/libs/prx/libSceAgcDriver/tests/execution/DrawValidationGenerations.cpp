#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Submit/include/Dcb.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/GuestWriteWatch.hpp"
#include "VulkanTestDevice.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#endif
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <initializer_list>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

using AgcDriver::Graphics::Require;

alignas(256) constexpr std::array<std::uint32_t, 16> TriangleVertexCode{
    0x7e000d05u, 0x080200f4u, 0x10020300u, 0x100202f4u, 0x060202f3u, 0x060400f3u, 0x10040500u, 0x060404f3u,
    0x7e060280u, 0x7e0802f2u, 0xf80008cfu, 0x04030201u, 0xbf810000u,
};
alignas(256) constexpr std::array<std::uint32_t, 10> ArithmeticPixelCode{0xf4000100u, 0xfa000000u, 0xbf8cc07fu, 0x7e000204u, 0x06000002u, 0x7e020280u, 0xf800180fu, 0x00010100u, 0xbf810000u, 0xbf810000u};
alignas(256) std::array<std::byte, 1024> ColorMemory{};

constexpr std::size_t BlockBytes = 65536;
constexpr std::size_t TargetBytes = 1024;

struct Header {
    Shader shader{};
    ShaderUserData userData{};
};

alignas(8) Header VertexHeader{};
alignas(8) Header PixelHeader{};

void Registered(Header& header, const std::uint32_t* code, std::size_t bytes, std::uint8_t type) {
    header.shader.file_header = 0x34333231u;
    header.shader.version = 0x18u;
    header.shader.header_size = sizeof(Header);
    header.shader.shader_size = static_cast<std::uint32_t>(bytes);
    header.shader.code = code;
    header.shader.type = type;
    header.shader.user_data = &header.userData;
    AgcDriverRegisterShader_nid_postfix(&header.shader);
}

struct WatchedBlocks {
    explicit WatchedBlocks(std::size_t bytes) : bytes(bytes) {
#ifdef _WIN32
        data = static_cast<std::uint8_t*>(GuestArena::GuestArenaAllocate_nid_postfix(bytes, BlockBytes));
        Require(data != nullptr, "cannot allocate watched guest blocks");
        GuestArena::GuestArenaCommit_nid_postfix(data, bytes, PAGE_READWRITE, bytes);
#else
        void* raw = mmap(nullptr, bytes + BlockBytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        Require(raw != MAP_FAILED, "cannot map watched guest blocks");
        const auto begin = reinterpret_cast<std::uintptr_t>(raw);
        const auto aligned = (begin + BlockBytes - 1) & ~(static_cast<std::uintptr_t>(BlockBytes) - 1);
        if (aligned != begin) munmap(raw, aligned - begin);
        if (aligned + bytes != begin + bytes + BlockBytes) munmap(reinterpret_cast<void*>(aligned + bytes), begin + BlockBytes - aligned);
        data = reinterpret_cast<std::uint8_t*>(aligned);
        GuestWriteWatch::GuestWriteWatchRegister_nid_postfix(data, bytes);
#endif
        std::memset(data, 0, bytes);
        Require(AgcDriver::GuestMemory::Watched(Address(), bytes), "the guest blocks are not write-watched");
        GuestAllocations::Mutation mutation;
        mutation.Add(data, bytes, true, true, true);
    }
    ~WatchedBlocks() {
        AgcDriver::Graphics::StorageTexture::FlushPending(Address(), bytes, nullptr, "test release");
        {
            GuestAllocations::Mutation mutation;
            mutation.Remove(data);
        }
#ifdef _WIN32
        GuestArena::GuestArenaReset_nid_postfix(data, bytes);
        GuestArena::GuestArenaRelease_nid_postfix(data, bytes);
#else
        munmap(data, bytes);
        GuestWriteWatch::GuestWriteWatchUnregister_nid_postfix(data, bytes);
#endif
    }
    WatchedBlocks(const WatchedBlocks&) = delete;
    WatchedBlocks& operator=(const WatchedBlocks&) = delete;
    std::uint64_t Address() const { return reinterpret_cast<std::uintptr_t>(data); }
    std::uint32_t* Words(std::size_t block) const { return reinterpret_cast<std::uint32_t*>(data + block * BlockBytes); }
    std::size_t bytes;
    std::uint8_t* data = nullptr;
};

void Append(std::vector<std::uint32_t>& commands, std::uint32_t opcode, std::initializer_list<std::uint32_t> payload) {
    commands.push_back(0xc0000000u | (static_cast<std::uint32_t>(payload.size() - 1) << 16u) | (opcode << 8u));
    commands.insert(commands.end(), payload);
}

std::vector<std::uint32_t> Setup(std::uintptr_t color) {
    const auto vertex = reinterpret_cast<std::uintptr_t>(TriangleVertexCode.data());
    const auto pixel = reinterpret_cast<std::uintptr_t>(ArithmeticPixelCode.data());
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
    const std::vector<std::pair<std::uint32_t, std::uint32_t>> shader{
        {0xc8, static_cast<std::uint32_t>(vertex >> 8u)}, {0xc9, static_cast<std::uint32_t>(vertex >> 40u)}, {0x8b, 0}, {0x8c, 0},
        {0x008, static_cast<std::uint32_t>(pixel >> 8u)}, {0x009, static_cast<std::uint32_t>(pixel >> 40u)}, {0x00b, 3u << 1u},
    };
    std::vector<std::uint32_t> commands;
    Append(commands, 0x79, {0x242, 4});
    for (const auto& [offset, value] : context) Append(commands, 0x69, {offset, value});
    for (const auto& [offset, value] : shader) Append(commands, 0x76, {offset, value});
    return commands;
}

AgcDriver::DriverDetail::DrawEntryCounters Counters() {
    return AgcDriver::DriverDetail::Driver::Get().DrawCacheCounters();
}

std::string Counts(const AgcDriver::DriverDetail::DrawEntryCounters& counters) {
    return std::to_string(counters.lookups) + " lookups, " + std::to_string(counters.hits) + " hits, " + std::to_string(counters.registerKeyHits) + " register-key hits; generation gate: " + std::to_string(counters.generationSkips) + " skips, " + std::to_string(counters.generationCompares) + " compares, " + std::to_string(counters.generationVerified) + " verified";
}

void Draw(std::uintptr_t color, const std::uint32_t* table, float word) {
    auto commands = Setup(color);
    const auto address = reinterpret_cast<std::uintptr_t>(table);
    Append(commands, 0x76, {0x00c, static_cast<std::uint32_t>(address)});
    Append(commands, 0x76, {0x00d, static_cast<std::uint32_t>(address >> 32u)});
    Append(commands, 0x76, {0x00e, std::bit_cast<std::uint32_t>(word)});
    Append(commands, 0x2d, {3, 2});
    Packet packet{commands.data(), static_cast<std::uint32_t>(commands.size()), 0, {}};
    Require(sceAgcDriverSubmitDcb(&packet) == 0, "the draw submission failed");
    AgcDriverWaitIdle_nid_postfix();
}

unsigned BrightestColorByte() {
    AgcDriver::GuestMemory::FlushGpuWrites(reinterpret_cast<std::uintptr_t>(ColorMemory.data()), ColorMemory.size());
    unsigned brightest = 0;
    for (const auto value : ColorMemory) brightest = std::max(brightest, static_cast<unsigned>(std::to_integer<std::uint8_t>(value)));
    return brightest;
}

unsigned Expected(float loaded, float word) {
    return static_cast<unsigned>(std::lround(std::clamp(loaded + word, 0.0f, 1.0f) * 255.0f));
}

bool CloseTo(unsigned value, unsigned expected) {
    return value + 2 >= expected && value <= expected + 2;
}

struct Mode {
    bool off = false;
    bool verify = false;
};

unsigned DrawRead(const std::uint32_t* table, float word) {
    ColorMemory.fill(std::byte{0});
    Draw(reinterpret_cast<std::uintptr_t>(ColorMemory.data()), table, word);
    return BrightestColorByte();
}

void RequireGateEngaged(const Mode& mode, const AgcDriver::DriverDetail::DrawEntryCounters& before, const AgcDriver::DriverDetail::DrawEntryCounters& after, const std::string& what) {
    const auto skips = after.generationSkips - before.generationSkips;
    const auto verified = after.generationVerified - before.generationVerified;
    if (mode.off) Require(skips == 0 && verified == 0 && after.generationCompares == before.generationCompares, "with APS5_NO_DRAW_VALIDATION_GENERATIONS=1, " + what + " used the generation gate: " + Counts(after));
    else if (mode.verify) Require(skips == 0 && verified != 0, "verify mode: " + what + " did not run the gate beside the compare: " + Counts(after));
    else Require(skips != 0 && verified == 0, what + " compared the captured runs again over unchanged watched memory: " + Counts(after));
}

void RequireCompared(const Mode& mode, const AgcDriver::DriverDetail::DrawEntryCounters& before, const AgcDriver::DriverDetail::DrawEntryCounters& after, const std::string& what) {
    if (mode.off) return;
    Require(after.generationSkips == before.generationSkips && after.generationVerified == before.generationVerified && after.generationCompares > before.generationCompares, what + " skipped the compare: " + Counts(after) + " (before: " + Counts(before) + ")");
}

void UnchangedMemorySkipsAndCpuStoresCompare(const Mode& mode, WatchedBlocks& blocks) {
    auto* table = blocks.Words(0);
    table[0] = std::bit_cast<std::uint32_t>(0.25f);
    for (int warm = 0; warm < 4; ++warm) {
        const auto value = DrawRead(table, 0.25f);
        Require(CloseTo(value, Expected(0.25f, 0.25f)), "a warm-up draw loading 0.25 wrote " + std::to_string(value) + ": " + Counts(Counters()));
    }
    for (int repeat = 0; repeat < 3; ++repeat) {
        const auto before = Counters();
        const auto value = DrawRead(table, 0.25f);
        const auto after = Counters();
        Require(after.registerKeyHits == before.registerKeyHits + 1, "a repeated draw over unchanged memory was not a register-key hit: " + Counts(after));
        Require(CloseTo(value, Expected(0.25f, 0.25f)), "a repeated draw loading 0.25 wrote " + std::to_string(value));
        RequireGateEngaged(mode, before, after, "a repeated draw over unchanged memory");
    }
    table[0] = std::bit_cast<std::uint32_t>(0.5f);
    const auto before = Counters();
    const auto value = DrawRead(table, 0.25f);
    const auto after = Counters();
    Require(CloseTo(value, Expected(0.5f, 0.25f)), "after a CPU store of 0.5 into the loaded constant the draw wrote " + std::to_string(value) + ", expected " + std::to_string(Expected(0.5f, 0.25f)) + ": " + Counts(after));
    RequireCompared(mode, before, after, "the draw after a CPU store into its block");
    const auto again = DrawRead(table, 0.25f);
    Require(CloseTo(again, Expected(0.5f, 0.25f)), "the draw after the stored 0.5 wrote " + std::to_string(again));
}

void GpuWritesCompare(const Mode& mode, WatchedBlocks& blocks) {
    auto* source = blocks.Words(1);
    source[0] = std::bit_cast<std::uint32_t>(0.0f);
    auto* target = blocks.Words(2);
    target[0] = std::bit_cast<std::uint32_t>(0.125f);
    for (int warm = 0; warm < 5; ++warm) {
        const auto value = DrawRead(target, 0.25f);
        Require(CloseTo(value, Expected(0.125f, 0.25f)), "a warm-up draw loading 0.125 wrote " + std::to_string(value) + ": " + Counts(Counters()));
    }
    {
        const auto before = Counters();
        DrawRead(target, 0.25f);
        RequireGateEngaged(mode, before, Counters(), "a repeated draw reading a future render target");
    }
    Draw(reinterpret_cast<std::uintptr_t>(target), source, 63.0f / 255.0f);
    AgcDriver::Graphics::StorageTexture::FlushPending(reinterpret_cast<std::uintptr_t>(target), TargetBytes, nullptr, "test read");
    AgcDriver::GuestMemory::FlushGpuWrites(reinterpret_cast<std::uintptr_t>(target), TargetBytes);
    const auto written = std::bit_cast<float>(target[0]);
    Require(std::isfinite(written) && written != 0.125f, "the draw rendering into the block left its first dword at " + std::to_string(written));
    const auto before = Counters();
    const auto value = DrawRead(target, 0.25f);
    const auto after = Counters();
    Require(CloseTo(value, Expected(written, 0.25f)), "after a draw rendered " + std::to_string(written) + " into the loaded dword the cached draw wrote " + std::to_string(value) + ", expected " + std::to_string(Expected(written, 0.25f)) + ": " + Counts(after));
    RequireCompared(mode, before, after, "the draw after a GPU write into its block");
}

}

int main() {
    try {
        bool libraries = false;
        {
            const auto device = OpenVulkanTestDevice();
            if (!device) return VulkanTestSkipped;
            libraries = device->GraphicsPipelineLibraries();
        }
        if (!libraries && std::getenv("APS5_NO_GPL") == nullptr) return VulkanTestSkipped;
        if (!AgcDriver::GuestMemory::WriteWatched()) {
            std::cout << "skipped, the guest arena is not write-watched\n";
            return VulkanTestSkipped;
        }
        const Mode mode{std::getenv("APS5_NO_DRAW_VALIDATION_GENERATIONS") != nullptr, std::getenv("APS5_VERIFY_DRAW_VALIDATION_GENERATIONS") != nullptr};
        Registered(VertexHeader, TriangleVertexCode.data(), sizeof(TriangleVertexCode), 2);
        Registered(PixelHeader, ArithmeticPixelCode.data(), sizeof(ArithmeticPixelCode), 1);
        {
            WatchedBlocks blocks(3 * BlockBytes);
            UnchangedMemorySkipsAndCpuStoresCompare(mode, blocks);
            GpuWritesCompare(mode, blocks);
        }
        AgcDriverShutdown_nid_postfix();
        std::cout << "Draw hit validation " << (mode.off ? "compared every hit" : mode.verify ? "verified the generation gate against the compare" : "skipped compares over unchanged write generations") << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        AgcDriverShutdown_nid_postfix();
        return 1;
    }
}
