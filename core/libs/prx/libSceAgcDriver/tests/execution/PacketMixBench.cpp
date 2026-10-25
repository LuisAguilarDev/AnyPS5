#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/FrameSteps.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Submit/include/Dcb.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "VulkanTestDevice.hpp"
#ifdef _WIN32
#include <windows.h>
#endif
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
#include <string_view>
#include <utility>
#include <vector>

namespace {

using AgcDriver::Graphics::Require;
namespace FrameSteps = AgcDriver::FrameSteps;

alignas(256) constexpr std::array<std::uint32_t, 8> VertexCode{0x7e000208u, 0x7e020280u, 0x7e040280u, 0x7e0602f2u, 0xf80008cfu, 0x03020100u, 0xbf810000u, 0xbf810000u};
alignas(256) constexpr std::array<std::uint32_t, 8> PixelCode{0x7e000200u, 0x7e020280u, 0xf800180fu, 0x00010100u, 0xbf810000u, 0xbf810000u, 0xbf810000u, 0xbf810000u};
alignas(256) std::array<std::byte, 1024> ColorMemory{};
constexpr std::size_t TargetBytes = 0x100000;
constexpr std::uint32_t TargetSide = 512;
constexpr std::uint32_t RenderTargetSwizzle = 0x1bu;
std::byte* ColorTarget = ColorMemory.data();

struct TargetBlock {
    TargetBlock() {
#ifdef _WIN32
        if (!AgcDriver::GuestMemory::WriteWatched()) return;
        data = static_cast<std::byte*>(GuestArena::GuestArenaAllocate_nid_postfix(TargetBytes, TargetBytes));
        Require(data != nullptr, "cannot allocate the color target");
        GuestArena::GuestArenaCommit_nid_postfix(data, TargetBytes, PAGE_READWRITE, TargetBytes);
        GuestAllocations::Mutation mutation;
        mutation.Add(data, TargetBytes, true, true, true);
        ColorTarget = data;
#endif
    }
    ~TargetBlock() {
#ifdef _WIN32
        if (data == nullptr) return;
        {
            GuestAllocations::Mutation mutation;
            mutation.Remove(data);
        }
        GuestArena::GuestArenaReset_nid_postfix(data, TargetBytes);
        GuestArena::GuestArenaRelease_nid_postfix(data, TargetBytes);
        ColorTarget = ColorMemory.data();
#endif
    }
    TargetBlock(const TargetBlock&) = delete;
    TargetBlock& operator=(const TargetBlock&) = delete;
    std::byte* data = nullptr;
};
alignas(256) volatile std::uint32_t Label = 0;

constexpr std::size_t FrameDraws = 600;
constexpr std::size_t LabelEvery = 8;
constexpr std::uint32_t PartialFlushEvent = 0x40fu;
constexpr std::array<std::uint32_t, 7> AcquireAll{0x82007fc0u, 1u, 0u, 0xffffffffu, 0u, 0xffffu, 0x200u};

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

template <std::size_t Words>
void Append(std::vector<std::uint32_t>& commands, std::uint32_t opcode, const std::array<std::uint32_t, Words>& payload) {
    commands.push_back(0xc0000000u | (static_cast<std::uint32_t>(Words - 1) << 16u) | (opcode << 8u));
    commands.insert(commands.end(), payload.begin(), payload.end());
}

std::uint32_t Low(const volatile void* address) {
    return static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(address));
}

std::uint32_t High(const volatile void* address) {
    return static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(address) >> 32u);
}

enum class Mix { Regs, Labels };

std::vector<std::uint32_t> Frame(Mix mix, std::uint32_t round) {
    const auto color = reinterpret_cast<std::uintptr_t>(ColorTarget);
    const bool resident = ColorTarget != ColorMemory.data();
    const auto vertex = reinterpret_cast<std::uintptr_t>(VertexCode.data());
    const auto pixel = reinterpret_cast<std::uintptr_t>(PixelCode.data());
    constexpr float scale = 32.0f;
    const std::vector<std::pair<std::uint32_t, std::uint32_t>> context{
        {0x2d5, 0x2000},
        {0x1b6, 0}, {0x207, 0}, {0x200, 0}, {0x203, 0x800},
        {0x2dc, 0xaa00}, {0x2f8, 0}, {0x292, 2}, {0x293, 0},
        {0x80, 0}, {0x8d, 0}, {0x83, 0xffff}, {0x8c, 0xa},
        {0x2f9, 0x2d}, {0x313, 0x6000}, {0x30e, 0xffffffff}, {0x30f, 0xffffffff},
        {0x206, 0x43f}, {0x204, 0x80000}, {0x205, 0x240},
        {0x8e, 0xf}, {0x8f, 0xf}, {0x202, 0xcc0010},
        {0x1c4, 0}, {0x1c5, 9}, {0x1c3, 4}, {0x31c, 0x28028},
        {0x31b, 0}, {0x31d, 0}, {0x3b0, resident ? ((TargetSide - 1u) << 14u) | (TargetSide - 1u) : (63u << 14u) | 3u},
        {0x3b8, resident ? (0x9000000u & ~(0x1fu << 14u)) | (RenderTargetSwizzle << 14u) : 0x9000000u}, {0x1e0, 0},
        {0xc, 0}, {0xd, 0x40040},
        {0x81, 0x80000000}, {0x82, 0x40040},
        {0x90, 0x80000000}, {0x91, 0x40040},
        {0x94, 0x80000000}, {0x95, 0x40040},
        {0x318, static_cast<std::uint32_t>(color >> 8u)}, {0x390, static_cast<std::uint32_t>(color >> 40u)},
        {0x10f, std::bit_cast<std::uint32_t>(scale)}, {0x110, std::bit_cast<std::uint32_t>(scale)},
        {0x111, std::bit_cast<std::uint32_t>(-2.0f)}, {0x112, std::bit_cast<std::uint32_t>(2.0f)},
        {0x113, std::bit_cast<std::uint32_t>(1.0f)}, {0x114, 0},
        {0xb4, 0}, {0xb5, std::bit_cast<std::uint32_t>(1.0f)},
        {0x2fe, 0}, {0x302, 0}, {0x306, 0}, {0x30a, 0},
    };
    const std::vector<std::pair<std::uint32_t, std::uint32_t>> shader{
        {0xc8, static_cast<std::uint32_t>(vertex >> 8u)}, {0xc9, static_cast<std::uint32_t>(vertex >> 40u)}, {0x8b, 1u << 1u}, {0x8c, 0},
        {0x008, static_cast<std::uint32_t>(pixel >> 8u)}, {0x009, static_cast<std::uint32_t>(pixel >> 40u)}, {0x00b, 1u << 1u},
    };
    std::vector<std::uint32_t> commands;
    commands.reserve(FrameDraws * 32u + 512u);
    Append(commands, 0x79, {0x242, 4});
    for (const auto& [offset, value] : context) Append(commands, 0x69, {offset, value});
    for (const auto& [offset, value] : shader) Append(commands, 0x76, {offset, value});
    const auto one = std::bit_cast<std::uint32_t>(1.0f);
    for (std::size_t draw = 0; draw < FrameDraws; ++draw) {
        Append(commands, 0x10, {static_cast<std::uint32_t>(draw)});
        Append(commands, 0x76, {0x00c, one});
        Append(commands, 0x76, {0x00d, static_cast<std::uint32_t>(draw)});
        Append(commands, 0x76, {0x00e, round});
        Append(commands, 0x69, {0x8e, 0xf});
        Append(commands, 0x69, {0x202, 0xcc0010});
        Append(commands, 0x79, {0x242, 4});
        Append(commands, 0x2a, {0});
        Append(commands, 0x2f, {1});
        Append(commands, 0x2d, {3, 2});
        if (mix == Mix::Labels && draw % LabelEvery == LabelEvery - 1) {
            Append(commands, 0x46, {PartialFlushEvent});
            Append(commands, 0x58, AcquireAll);
            Append(commands, 0x37, {0x00100500u, Low(&Label), High(&Label), static_cast<std::uint32_t>(round * FrameDraws + draw)});
        }
    }
    return commands;
}

std::size_t Packets(const std::vector<std::uint32_t>& commands) {
    std::size_t packets = 0;
    for (std::size_t cursor = 0; cursor < commands.size(); cursor += AgcDriver::Pm4::PacketWords(commands[cursor])) ++packets;
    return packets;
}

void Submit(const std::vector<std::uint32_t>& commands) {
    Packet packet{const_cast<std::uint32_t*>(commands.data()), static_cast<std::uint32_t>(commands.size()), 0, {}};
    Require(sceAgcDriverSubmitDcb(&packet) == 0, "the packet mix failed to submit");
    AgcDriverWaitIdle_nid_postfix();
}

std::string SlotName(std::size_t slot) {
    return AgcDriver::Pm4::Name(0xc0000000u | (static_cast<std::uint32_t>(slot) << 8u));
}

void CheckMixes() {
    for (const auto mix : {Mix::Regs, Mix::Labels}) {
        const auto frame = Frame(mix, 1);
        Submit(frame);
        const auto first = AgcDriver::DriverDetail::Driver::Get().DrawCacheCounters();
        Submit(Frame(mix, 2));
        const auto second = AgcDriver::DriverDetail::Driver::Get().DrawCacheCounters();
        const auto drawn = second.hits + second.dataHits - first.hits - first.dataHits;
        Require(drawn + 1 >= FrameDraws, "a frame of " + std::to_string(FrameDraws) + " repeated draws made " + std::to_string(drawn) + " draw-cache hits");
        if (mix == Mix::Labels) Require(Label == 2u * FrameDraws + FrameDraws - 1u, "the last WRITE_DATA label of the frame did not land");
        Require(Packets(frame) > FrameDraws * 10u, "a frame carries fewer than ten packets per draw");
    }
}

void Benchmark(Mix mix, double seconds) {
    Submit(Frame(mix, 0));
    static_cast<void>(FrameSteps::Global().Take());
    const auto packetsPerFrame = Packets(Frame(mix, 0));
    const auto start = std::chrono::steady_clock::now();
    std::uint64_t frames = 0;
    for (std::uint32_t round = 1; std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() < seconds; ++round) {
        Submit(Frame(mix, round));
        ++frames;
    }
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    const auto packets = static_cast<double>(frames * packetsPerFrame);
    std::cout << "[packet-mix-bench] " << (mix == Mix::Regs ? "regs" : "labels") << ": " << frames << " frames of " << packetsPerFrame << " packets (" << FrameDraws << " draws) in " << elapsed << " s, "
              << elapsed * 1e6 / packets << " us/packet wall, " << elapsed * 1e3 / static_cast<double>(frames) << " ms/frame wall\n";
    if (!FrameSteps::Enabled()) return;
    const auto totals = FrameSteps::Global().Take();
    FrameSteps::Window window;
    window.seconds = elapsed;
    window.frames = frames;
    std::cout << FrameSteps::FormatLanes(totals, window) << FrameSteps::FormatPackets(totals, window, SlotName);
}

}

int main(int argc, char** argv) {
    try {
        {
            const auto device = OpenVulkanTestDevice();
            if (!device) return VulkanTestSkipped;
        }
        Register(VertexHeader, VertexCode.data(), sizeof(VertexCode), 2);
        Register(PixelHeader, PixelCode.data(), sizeof(PixelCode), 1);
        const TargetBlock target;
        if (argc >= 3 && std::string_view(argv[1]) == "--benchmark") {
            Benchmark(std::string_view(argv[2]) == "labels" ? Mix::Labels : Mix::Regs, argc >= 4 ? std::strtod(argv[3], nullptr) : 5.0);
        } else {
            CheckMixes();
            std::cout << "packet mixes of cache-hit draws ran\n";
        }
        AgcDriverShutdown_nid_postfix();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        AgcDriverShutdown_nid_postfix();
        return 1;
    }
}
