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

// The T7 assumption of the multi-threaded draw design: with the draw-cache lookup and its hit
// validation running in a prep-pool job instead of on the packet thread, a title that rewrites a
// buffer once the label after a draw has signaled must still see the draw use the old bytes, and the
// next draw the new ones. The label is a FIFO item behind the draw, so it cannot signal before the
// job that validated the draw has finished.

namespace {

using AgcDriver::Graphics::Require;
using AgcDriver::DriverDetail::DrawPrepPool;

alignas(256) constexpr std::array<std::uint32_t, 16> TriangleVertexCode{
    0x7e000d05u, 0x080200f4u, 0x10020300u, 0x100202f4u, 0x060202f3u, 0x060400f3u, 0x10040500u, 0x060404f3u,
    0x7e060280u, 0x7e0802f2u, 0xf80008cfu, 0x04030201u, 0xbf810000u,
};
alignas(256) constexpr std::array<std::uint32_t, 8> ConstantPixelCode{0xf4000080u, 0xfa000000u, 0xbf8cc07fu, 0x7e000202u, 0x7e020280u, 0xf800180fu, 0x00010100u, 0xbf810000u};
alignas(256) std::array<std::byte, 1024> EarlierColor{};
alignas(256) std::array<std::byte, 1024> LaterColor{};
alignas(256) std::array<float, 4> RingSlot{};
alignas(256) volatile std::uint32_t Label = 0;

constexpr std::size_t DrawsPerSubmission = 4;
constexpr std::uint32_t Rounds = 6;

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

std::uint32_t Low(const volatile void* pointer) {
    return static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(pointer));
}

std::uint32_t High(const volatile void* pointer) {
    return static_cast<std::uint32_t>(static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(pointer)) >> 32u);
}

// Draws that cover the target with the value the pixel shader loads from the ring slot through its
// user words, followed by a RELEASE_MEM label: the label the guest waits for before it reuses the slot.
std::vector<std::uint32_t> DrawsThenLabel(std::array<std::byte, 1024>& target, std::uint32_t label, std::size_t draws = DrawsPerSubmission) {
    const auto color = reinterpret_cast<std::uintptr_t>(target.data());
    const auto vertex = reinterpret_cast<std::uintptr_t>(TriangleVertexCode.data());
    const auto pixel = reinterpret_cast<std::uintptr_t>(ConstantPixelCode.data());
    const auto slot = reinterpret_cast<std::uintptr_t>(RingSlot.data());
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
    };
    std::vector<std::uint32_t> commands;
    Append(commands, 0x79, {0x242, 4});
    for (const auto& [offset, value] : context) Append(commands, 0x69, {offset, value});
    for (const auto& [offset, value] : shader) Append(commands, 0x76, {offset, value});
    for (std::size_t draw = 0; draw < draws; ++draw) {
        Append(commands, 0x76, {0x00c, static_cast<std::uint32_t>(slot)});
        Append(commands, 0x76, {0x00d, static_cast<std::uint32_t>(slot >> 32u)});
        Append(commands, 0x2d, {3, 2});
    }
    Append(commands, 0x49, {0x514u, (1u << 29u) | (2u << 24u), Low(&Label), High(&Label), label, 0u, 0u});
    return commands;
}

void Submit(std::vector<std::uint32_t>& commands) {
    Packet packet{commands.data(), static_cast<std::uint32_t>(commands.size()), 0, {}};
    Require(sceAgcDriverSubmitDcb(&packet) == 0, "the draw submission failed");
}

void WaitForLabel(std::uint32_t value) {
    const auto start = std::chrono::steady_clock::now();
    while (Label != value) {
        Require(std::chrono::steady_clock::now() - start < std::chrono::seconds(10), "the label after the draws did not signal");
        std::this_thread::yield();
    }
}

unsigned Brightest(std::array<std::byte, 1024>& target) {
    AgcDriver::GuestMemory::FlushGpuWrites(reinterpret_cast<std::uintptr_t>(target.data()), target.size());
    unsigned brightest = 0;
    for (const auto value : target) brightest = std::max(brightest, static_cast<unsigned>(value));
    return brightest;
}

float Unorm(unsigned value) {
    return static_cast<float>(value) / 255.0f;
}

void RingSlotRewrittenAfterTheLabel() {
    Register(VertexHeader, TriangleVertexCode.data(), sizeof(TriangleVertexCode), 2);
    Register(PixelHeader, ConstantPixelCode.data(), sizeof(ConstantPixelCode), 1);
    // Warm the draw cache so the timed rounds hit it: the hit validation, the code under test, then
    // decides what the earlier draws record.
    RingSlot[0] = Unorm(8);
    std::uint32_t label = 1;
    auto warm = DrawsThenLabel(EarlierColor, label);
    Submit(warm);
    WaitForLabel(label);
    AgcDriverWaitIdle_nid_postfix();
    const auto before = AgcDriver::DriverDetail::Driver::Get().DrawCacheCounters();
    for (std::uint32_t round = 0; round < Rounds; ++round) {
        // Values grow from round to round, so a stale target from an earlier round cannot look right.
        const unsigned earlier = 32 + round * 32, later = earlier + 16;
        EarlierColor.fill(std::byte{0});
        LaterColor.fill(std::byte{0});
        RingSlot[0] = Unorm(earlier);
        auto first = DrawsThenLabel(EarlierColor, ++label);
        Submit(first);
        WaitForLabel(label);
        RingSlot[0] = Unorm(later);
        auto second = DrawsThenLabel(LaterColor, ++label);
        Submit(second);
        WaitForLabel(label);
        AgcDriverWaitIdle_nid_postfix();
        const auto seenEarlier = Brightest(EarlierColor), seenLater = Brightest(LaterColor);
        Require(seenEarlier == earlier, "round " + std::to_string(round) + ": the draws before the label wrote " + std::to_string(seenEarlier) + ", expected the ring slot's old value " + std::to_string(earlier));
        Require(seenLater == later, "round " + std::to_string(round) + ": the draws after the rewrite wrote " + std::to_string(seenLater) + ", expected the ring slot's new value " + std::to_string(later));
    }
    const auto after = AgcDriver::DriverDetail::Driver::Get().DrawCacheCounters();
    Require(after.hits - before.hits >= Rounds, std::to_string(after.hits - before.hits) + " of " + std::to_string(Rounds * 2 * DrawsPerSubmission) + " timed draws hit the draw cache, expected at least one a round: the hit validation was not exercised");
}

// A synthetic stream for the packet-thread budget (run by hand with APS5_PROFILE_DRAW=1 and the
// prep-pool switches; not a ctest): submissions of draws that hit the cache with a refreshed constant,
// each closed by a label the CPU waits for, until the given seconds pass. The 10 s [draw] driver
// phases and [prep-pool] lines carry the per-packet split.
void Bench(double seconds) {
    Register(VertexHeader, TriangleVertexCode.data(), sizeof(TriangleVertexCode), 2);
    Register(PixelHeader, ConstantPixelCode.data(), sizeof(ConstantPixelCode), 1);
    constexpr std::size_t BenchDraws = 256;
    std::uint32_t label = 0;
    std::uint64_t submissions = 0;
    const auto start = std::chrono::steady_clock::now();
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() < seconds) {
        RingSlot[0] = Unorm(16 + static_cast<unsigned>(submissions % 200));
        auto commands = DrawsThenLabel(EarlierColor, ++label, BenchDraws);
        Submit(commands);
        WaitForLabel(label);
        ++submissions;
    }
    AgcDriverWaitIdle_nid_postfix();
    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << "bench: " << submissions * BenchDraws << " draws in " << elapsed << " s, " << elapsed * 1e6 / static_cast<double>(submissions * BenchDraws) << " us per draw end to end, " << DrawPrepPool::Get().Lookups() << " lookups on the prep pool\n";
}

}

int main(int argc, char** argv) {
    try {
        {
            const auto device = OpenVulkanTestDevice();
            if (!device) return VulkanTestSkipped;
        }
        if (argc == 3 && std::string(argv[1]) == "--bench") {
            Bench(std::strtod(argv[2], nullptr));
            AgcDriverShutdown_nid_postfix();
            return 0;
        }
        RingSlotRewrittenAfterTheLabel();
        const bool pool = DrawPrepPool::Threads() != 0;
        const bool poolLookup = pool && std::getenv("APS5_NO_POOL_LOOKUP") == nullptr;
        const auto lookups = DrawPrepPool::Get().Lookups();
        if (pool) Require(DrawPrepPool::Get().Enqueued() != 0, "no draw was prepared on the prep pool");
        Require(poolLookup ? lookups != 0 : lookups == 0, std::to_string(lookups) + " draw-cache lookups ran in prep-pool jobs, expected " + (poolLookup ? "some" : "none"));
        AgcDriverShutdown_nid_postfix();
        std::cout << "A ring slot rewritten after a label kept its old bytes for the draws before it (" << lookups << " lookups on the prep pool)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        AgcDriverShutdown_nid_postfix();
        return 1;
    }
}
