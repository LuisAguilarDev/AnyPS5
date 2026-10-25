#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPrepPool.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Submit/include/Dcb.hpp"
#include "VulkanTestDevice.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <initializer_list>
#include <iostream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using AgcDriver::Graphics::Require;
using AgcDriver::DriverDetail::DrawPrepPool;

alignas(256) constexpr std::array<std::uint32_t, 16> TriangleVertexCode{
    0x7e000d05u, 0x080200f4u, 0x10020300u, 0x100202f4u, 0x060202f3u, 0x060400f3u, 0x10040500u, 0x060404f3u,
    0x7e060280u, 0x7e0802f2u, 0xf80008cfu, 0x04030201u, 0xbf810000u,
};
alignas(256) constexpr std::array<std::uint32_t, 8> ConstantPixelCode{0xf4000080u, 0xfa000000u, 0xbf8cc07fu, 0x7e000202u, 0x7e020280u, 0xf800180fu, 0x00010100u, 0xbf810000u};

constexpr std::size_t TargetBytes = 1024;
constexpr std::size_t Slots = 8;
constexpr std::size_t Targets = 1;
constexpr std::size_t StaleRounds = 4;

using Target = std::array<std::byte, TargetBytes>;
alignas(256) std::array<Target, Targets> Colors{};
alignas(256) std::array<Target, Slots + StaleRounds> RingSlots{};
alignas(256) volatile std::uint32_t Label = 0;

struct Header {
    Shader shader{};
    ShaderUserData userData{};
};

alignas(8) Header VertexHeader{};
alignas(8) Header PixelHeader{};

void Register(Header& header, const std::uint32_t* code, std::size_t bytes, std::uint8_t type) {
    header.shader.file_header = 0x34333231u;
    header.shader.version = 0x18u;
    header.shader.header_size = sizeof(Header);
    header.shader.shader_size = static_cast<std::uint32_t>(bytes);
    header.shader.code = code;
    header.shader.type = type;
    header.shader.user_data = &header.userData;
    AgcDriverRegisterShader_nid_postfix(&header.shader);
}

void Append(std::vector<std::uint32_t>& commands, std::uint32_t opcode, std::initializer_list<std::uint32_t> payload) {
    commands.push_back(0xc0000000u | (static_cast<std::uint32_t>(payload.size() - 1) << 16u) | (opcode << 8u));
    commands.insert(commands.end(), payload);
}

void Draw(std::vector<std::uint32_t>& commands, const void* target, const void* slot) {
    const auto color = reinterpret_cast<std::uintptr_t>(target);
    const auto vertex = reinterpret_cast<std::uintptr_t>(TriangleVertexCode.data());
    const auto pixel = reinterpret_cast<std::uintptr_t>(ConstantPixelCode.data());
    const auto table = reinterpret_cast<std::uintptr_t>(slot);
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
        {0x008, static_cast<std::uint32_t>(pixel >> 8u)}, {0x009, static_cast<std::uint32_t>(pixel >> 40u)}, {0x00b, 2u << 1u},
        {0x00c, static_cast<std::uint32_t>(table)}, {0x00d, static_cast<std::uint32_t>(table >> 32u)},
    };
    Append(commands, 0x79, {0x242, 4});
    for (const auto& [offset, value] : context) Append(commands, 0x69, {offset, value});
    for (const auto& [offset, value] : shader) Append(commands, 0x76, {offset, value});
    Append(commands, 0x2d, {3, 2});
}

std::uint32_t Low(const volatile void* pointer) {
    return static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(pointer));
}

std::uint32_t High(const volatile void* pointer) {
    return static_cast<std::uint32_t>(static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(pointer)) >> 32u);
}

void SubmitAndWait(std::vector<std::uint32_t> commands) {
    static std::uint32_t label = 0;
    Append(commands, 0x49, {0x514u, (1u << 29u) | (2u << 24u), Low(&Label), High(&Label), ++label, 0u, 0u});
    Packet packet{commands.data(), static_cast<std::uint32_t>(commands.size()), 0, {}};
    Require(sceAgcDriverSubmitDcb(&packet) == 0, "the draw submission failed");
    const auto start = std::chrono::steady_clock::now();
    while (Label != label) {
        Require(std::chrono::steady_clock::now() - start < std::chrono::seconds(10), "the label after the draws did not signal");
        std::this_thread::yield();
    }
    AgcDriverWaitIdle_nid_postfix();
}

float& Word(Target& slot) {
    return *reinterpret_cast<float*>(slot.data());
}

unsigned Brightest(Target& target) {
    AgcDriver::GuestMemory::FlushGpuWrites(reinterpret_cast<std::uintptr_t>(target.data()), target.size());
    unsigned brightest = 0;
    for (const auto value : target) brightest = std::max(brightest, static_cast<unsigned>(value));
    return brightest;
}

float Unorm(unsigned value) {
    return static_cast<float>(value) / 255.0f;
}

struct Mode {
    bool pool = false;
    bool dataHitsOnPool = false;
    bool verify = false;
};

AgcDriver::DriverDetail::DrawEntryCounters Counters() {
    return AgcDriver::DriverDetail::Driver::Get().DrawCacheCounters();
}

std::string Counts(const AgcDriver::DriverDetail::DrawEntryCounters& counters) {
    return std::to_string(counters.lookups) + " lookups, " + std::to_string(counters.hits) + " hits, " + std::to_string(counters.dataHits) + " data hits (" + std::to_string(counters.dataDrawsReused) + " draws reused every stage, " + std::to_string(counters.dataStagesReused) + " stages reused, " + std::to_string(counters.dataStagesCompiled) + " compiled), " + std::to_string(DrawPrepPool::Get().Lookups()) + " lookups on the prep pool";
}

void Warm() {
    Word(RingSlots[0]) = Unorm(32);
    for (int round = 0; round < 2; ++round) {
        std::vector<std::uint32_t> commands;
        Draw(commands, Colors[0].data(), RingSlots[0].data());
        SubmitAndWait(std::move(commands));
    }
    Require(Brightest(Colors[0]) == 32, "the warm-up draw wrote " + std::to_string(Brightest(Colors[0])) + ", expected 32");
}

void DataHitsReuseTheirStagesOnThePool(const Mode& mode) {
    for (std::size_t slot = 1; slot < Slots; ++slot) Word(RingSlots[slot]) = Unorm(32);
    Colors[0].fill(std::byte{0});
    const auto before = Counters();
    const auto poolLookups = DrawPrepPool::Get().Lookups();
    std::vector<std::uint32_t> commands;
    constexpr std::size_t draws = Slots - 1;
    for (std::size_t draw = 0; draw < draws; ++draw) Draw(commands, Colors[0].data(), RingSlots[draw + 1].data());
    SubmitAndWait(std::move(commands));
    const auto after = Counters();
    const auto seen = Brightest(Colors[0]);
    Require(seen == 32, "the data-only hits wrote " + std::to_string(seen) + ", expected 32: " + Counts(after));
    Require(after.dataHits - before.dataHits == draws, "the draws through new ring slots were not all data-only hits: " + Counts(after));
    if (!mode.verify) Require(after.dataDrawsReused - before.dataDrawsReused == draws, "a data-only hit with unchanged slot contents compiled a stage instead of reusing it: " + Counts(after));
    const auto onPool = DrawPrepPool::Get().Lookups() - poolLookups;
    if (mode.dataHitsOnPool) Require(onPool == draws, std::to_string(onPool) + " of " + std::to_string(draws) + " data-only hits were looked up in prep-pool jobs: " + Counts(after));
    else Require(onPool == 0, std::to_string(onPool) + " data-only hits were looked up in prep-pool jobs with APS5_NO_POOL_DATA_HITS_V2=1 or the pool off: " + Counts(after));
}

void StaleDataHitAcrossAPoolJob() {
    for (std::size_t round = 0; round < StaleRounds; ++round) {
        auto& read = RingSlots[Slots + round];
        read.fill(std::byte{0});
        Word(read) = Unorm(32);
        Word(RingSlots[1]) = Unorm(63);
        Colors[0].fill(std::byte{0});
        std::vector<std::uint32_t> commands;
        Draw(commands, read.data(), RingSlots[1].data());
        Draw(commands, Colors[0].data(), read.data());
        SubmitAndWait(std::move(commands));
        AgcDriver::GuestMemory::FlushGpuWrites(reinterpret_cast<std::uintptr_t>(read.data()), read.size());
        const auto written = Word(read);
        Require(written > 0.4f && written < 0.6f, "round " + std::to_string(round) + ": the draw rendering into the ring slot left its first dword at " + std::to_string(written));
        const auto expected = static_cast<unsigned>(written * 255.0f + 0.5f);
        const auto seen = Brightest(Colors[0]);
        Require(seen + 1 >= expected && seen <= expected + 1, "round " + std::to_string(round) + ": a data-only hit reading a ring slot an earlier draw of its submission rendered into wrote " + std::to_string(seen) + ", expected " + std::to_string(expected) + " (the slot's old contents give 32): " + Counts(Counters()));
    }
}

}

int main() {
    try {
        {
            const auto device = OpenVulkanTestDevice();
            if (!device) return VulkanTestSkipped;
        }
        Mode mode;
        mode.pool = DrawPrepPool::Threads() != 0;
        mode.verify = std::getenv("APS5_VERIFY_DRAW_DATA_HITS") != nullptr || std::getenv("APS5_VERIFY_DRAW_REBASE") != nullptr;
        mode.dataHitsOnPool = mode.pool && DrawPrepPool::Lookup() && std::getenv("APS5_NO_POOL_DATA_HITS_V2") == nullptr && std::getenv("APS5_VERIFY_DRAW_DATA_HITS") == nullptr && std::getenv("APS5_VERIFY_DRAW_REBASE") == nullptr;
        Register(VertexHeader, TriangleVertexCode.data(), sizeof(TriangleVertexCode), 2);
        Register(PixelHeader, ConstantPixelCode.data(), sizeof(ConstantPixelCode), 1);
        Warm();
        DataHitsReuseTheirStagesOnThePool(mode);
        StaleDataHitAcrossAPoolJob();
        AgcDriverShutdown_nid_postfix();
        std::cout << "Data-only hits " << (mode.dataHitsOnPool ? "were looked up and reused in prep-pool jobs" : "stayed on the packet thread") << " and read what earlier draws rendered\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        AgcDriverShutdown_nid_postfix();
        return 1;
    }
}
