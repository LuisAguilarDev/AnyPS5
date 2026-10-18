#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
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
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Draws into a render target with compute dispatches between them in one DCB; each dispatch reads
// the first row of the target as a buffer and stores every dword plus one. A dispatch must see the
// draws recorded before it (a draw chunk submitted after the dispatch's spine work fails) and none
// of the draws recorded after it.

namespace {

using AgcDriver::Graphics::Require;

constexpr std::uint32_t Width = 128;
constexpr std::uint32_t Height = 128;
// CB_COLOR0_ATTRIB3 of a 64 KiB R_X swizzled target (SW_MODE 27): only swizzled targets stay
// resident, so only their draws are recorded (and chunked) instead of drawn synchronously.
constexpr std::uint32_t TiledAttrib3 = 0x0906c000u;
constexpr std::uint32_t SetContextReg = 0x69u;
constexpr std::uint32_t SetShReg = 0x76u;
constexpr std::uint32_t SetUconfigReg = 0x79u;
constexpr std::uint32_t DrawIndexAuto = 0x2du;
constexpr std::uint32_t Lanes = 32;
constexpr std::uint32_t LaneBytes = Lanes * 4u;
constexpr std::uint32_t OutputStride = 64;
constexpr std::uint32_t MaxDispatches = 8;
constexpr std::size_t Block = 65536;
constexpr std::size_t ColorBytes = Width * Height * 4;
std::byte* ColorMemory = nullptr;
std::uint32_t* Output = nullptr;
alignas(64) volatile std::uint32_t EndOfPipe = 0;

std::byte* AllocateWatched(std::size_t bytes) {
#ifdef _WIN32
    void* block = GuestArena::GuestArenaAllocate_nid_postfix(bytes, Block);
    GuestArena::GuestArenaCommit_nid_postfix(block, bytes, PAGE_READWRITE, bytes);
#else
    void* raw = mmap(nullptr, bytes + Block, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (raw == MAP_FAILED) throw std::runtime_error("cannot map the watched block");
    const auto begin = reinterpret_cast<std::uintptr_t>(raw);
    const auto aligned = (begin + Block - 1) & ~static_cast<std::uintptr_t>(Block - 1);
    if (aligned != begin) munmap(raw, aligned - begin);
    if (aligned + bytes != begin + bytes + Block) munmap(reinterpret_cast<void*>(aligned + bytes), begin + Block - aligned);
    void* block = reinterpret_cast<void*>(aligned);
    GuestWriteWatch::GuestWriteWatchRegister_nid_postfix(block, bytes);
#endif
    return static_cast<std::byte*>(block);
}

// The output block of PipelinedDispatch: guest memory the CPU and the GPU both reach.
std::uint32_t* NewBlock() {
#ifdef _WIN32
    auto* block = static_cast<std::uint32_t*>(VirtualAlloc(nullptr, Block, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#else
    auto* block = static_cast<std::uint32_t*>(std::aligned_alloc(Block, Block));
#endif
    if (block == nullptr) throw std::runtime_error("chunked dispatch reads: cannot allocate a guest block");
    std::memset(block, 0, Block);
    GuestAllocations::Mutation().Add(block, Block, true, true, false);
    return block;
}

// The vertex shader of SampleMaskDraw: one full-target triangle fetched from a vertex buffer.
alignas(256) constexpr std::array<std::uint32_t, 6> VertexCode{
    0xe0382000, 0x80020005, 0xbf8c3f70, 0xf80008cf, 0x03020100, 0xbf810000,
};

// v_mov_b32 v0, s0; exp mrt0 v0, v0, v0, v0 done vm.
alignas(256) constexpr std::array<std::uint32_t, 4> ValueCode{
    0x7e000200, 0xf800180f, 0x00000000, 0xbf810000,
};

// The compute shader of PipelinedDispatch: lane i loads dword i of s[0:3] and stores it plus one
// to dword i of s[4:7].
alignas(256) constexpr std::array<std::uint32_t, 8> IncrementCode{0x34020082u, 0xe0301000u, 0x80000201u, 0xbf8c3f70u, 0x4a060481u, 0xe0701000u, 0x80010301u, 0xbf810000u};

alignas(256) constexpr std::array<std::array<float, 4>, 3> Vertices{{
    {-1.0f, -1.0f, 0.5f, 1.0f}, {3.0f, -1.0f, 0.5f, 1.0f}, {-1.0f, 3.0f, 0.5f, 1.0f},
}};

struct RegisteredShader {
    Shader shader;
    ShaderUserData userData;
};
RegisteredShader VertexShader{};
RegisteredShader PixelShader{};
Shader ComputeShader{};

const std::vector<std::pair<std::uint32_t, std::uint32_t>> ContextBaseline{
    {0x2d5, 0x2000},
    {0x1b6, 0}, {0x207, 0}, {0x200, 0}, {0x203, 0x800},
    {0x2dc, 0xaa00}, {0x2f8, 0}, {0x292, 2}, {0x293, 0},
    {0x80, 0}, {0x8d, 0}, {0x83, 0xffff}, {0x8c, 0xa},
    {0x2f9, 0x2d}, {0x313, 0x6000}, {0x30e, 0xffffffff}, {0x30f, 0xffffffff},
    {0x206, 0x43f}, {0x204, 0x80000}, {0x205, 0x240},
    {0x8e, 0xf}, {0x8f, 0xf}, {0x202, 0xcc0010},
    {0x1c4, 0}, {0x1c5, 9}, {0x1c3, 4}, {0x31c, 0x28028},
    {0x31b, 0}, {0x31d, 0}, {0x3b8, TiledAttrib3}, {0x1e0, 0},
    {0xc, 0},
    {0x81, 0x80000000}, {0x90, 0x80000000}, {0x94, 0x80000000},
    {0xb4, 0}, {0xb5, std::bit_cast<std::uint32_t>(1.0f)}, {0x114, 0}, {0x113, std::bit_cast<std::uint32_t>(1.0f)},
};

void Register(RegisteredShader& registered, const void* code, std::uint32_t bytes, std::uint8_t type) {
    registered.shader.file_header = 0x34333231u;
    registered.shader.version = 0x18u;
    registered.shader.user_data = &registered.userData;
    registered.shader.code = code;
    registered.shader.header_size = sizeof(RegisteredShader);
    registered.shader.shader_size = bytes;
    registered.shader.type = type;
    AgcDriverRegisterShader_nid_postfix(&registered.shader);
}

std::array<std::uint32_t, 4> VertexDescriptor(const void* data, std::uint32_t stride, std::uint32_t count) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu) | (stride << 16u), count, 0x01016facu};
}

std::array<std::uint32_t, 4> BufferDescriptor(const void* data, std::uint32_t bytes) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), bytes, 0x31016facu};
}

void SetRegister(std::vector<std::uint32_t>& words, std::uint32_t opcode, std::uint32_t offset, std::uint32_t value) {
    words.push_back(0xc0010000u | (opcode << 8u));
    words.push_back(offset);
    words.push_back(value);
}

void State(std::vector<std::uint32_t>& words) {
    const auto colorAddress = reinterpret_cast<std::uintptr_t>(ColorMemory);
    const auto vertexAddress = reinterpret_cast<std::uintptr_t>(VertexCode.data());
    const auto pixelAddress = reinterpret_cast<std::uintptr_t>(ValueCode.data());
    for (const auto& [offset, value] : ContextBaseline) SetRegister(words, SetContextReg, offset, value);
    SetRegister(words, SetContextReg, 0x318, static_cast<std::uint32_t>(colorAddress >> 8u));
    SetRegister(words, SetContextReg, 0x390, static_cast<std::uint32_t>(colorAddress >> 40u));
    SetRegister(words, SetContextReg, 0x3b0, ((Width - 1u) << 14u) | (Height - 1u));
    SetRegister(words, SetContextReg, 0x10f, std::bit_cast<std::uint32_t>(Width / 2.0f));
    SetRegister(words, SetContextReg, 0x110, std::bit_cast<std::uint32_t>(Width / 2.0f));
    SetRegister(words, SetContextReg, 0x111, std::bit_cast<std::uint32_t>(-(Height / 2.0f)));
    SetRegister(words, SetContextReg, 0x112, std::bit_cast<std::uint32_t>(Height / 2.0f));
    const auto extent = (Height << 16u) | Width;
    SetRegister(words, SetContextReg, 0xd, extent);
    SetRegister(words, SetContextReg, 0x82, extent);
    SetRegister(words, SetContextReg, 0x91, extent);
    SetRegister(words, SetContextReg, 0x95, extent);
    SetRegister(words, SetUconfigReg, 0x242, 4u);
    SetRegister(words, SetShReg, 0x008, static_cast<std::uint32_t>(pixelAddress >> 8u));
    SetRegister(words, SetShReg, 0x009, static_cast<std::uint32_t>(pixelAddress >> 40u));
    SetRegister(words, SetShReg, 0x00b, 1u << 1u);
    SetRegister(words, SetShReg, 0x0c8, static_cast<std::uint32_t>(vertexAddress >> 8u));
    SetRegister(words, SetShReg, 0x0c9, static_cast<std::uint32_t>(vertexAddress >> 40u));
    SetRegister(words, SetShReg, 0x08b, 4u << 1u);
    const auto descriptor = VertexDescriptor(Vertices.data(), 16u, static_cast<std::uint32_t>(Vertices.size()));
    for (std::uint32_t i = 0; i < descriptor.size(); ++i) SetRegister(words, SetShReg, 0x08c + i, descriptor[i]);
}

void Draw(std::vector<std::uint32_t>& words, std::uint8_t value) {
    SetRegister(words, SetShReg, 0x00c, std::bit_cast<std::uint32_t>(static_cast<float>(value) / 255.0f));
    words.insert(words.end(), {0xc0000000u | (1u << 16u) | (DrawIndexAuto << 8u), static_cast<std::uint32_t>(Vertices.size()), 2u});
}

// The dispatch packets of PipelinedDispatch: one group of Lanes threads, input in s[0:3], output in
// s[4:7].
void Dispatch(std::vector<std::uint32_t>& words, const void* input, const void* output) {
    const auto codeAddress = reinterpret_cast<std::uintptr_t>(IncrementCode.data());
    const auto in = BufferDescriptor(input, LaneBytes);
    const auto out = BufferDescriptor(output, LaneBytes);
    const std::vector<std::uint32_t> commands{
        0xc0027600u, 0x20cu, static_cast<std::uint32_t>(codeAddress >> 8u), static_cast<std::uint32_t>(codeAddress >> 40u),
        0xc0037600u, 0x207u, Lanes, 1u, 1u,
        0xc0017600u, 0x213u, 8u << 1u,
        0xc0087600u, 0x240u, in[0], in[1], in[2], in[3], out[0], out[1], out[2], out[3],
        0xc0031500u, 1u, 1u, 1u, 0x8041u,
    };
    words.insert(words.end(), commands.begin(), commands.end());
}

void ReleaseLabel(std::vector<std::uint32_t>& words, volatile std::uint32_t* target, std::uint32_t value) {
    const auto address = reinterpret_cast<std::uintptr_t>(target);
    words.insert(words.end(), {0xc0064900u, 0x514u, (1u << 29u) | (2u << 24u), static_cast<std::uint32_t>(address), static_cast<std::uint32_t>(static_cast<std::uint64_t>(address) >> 32u), value, 0u, 0u});
}

std::uint8_t ValueOf(std::uint32_t draw) {
    return static_cast<std::uint8_t>(15u + 20u * draw);
}

// A pixel of value v stores v in all four bytes.
std::uint32_t PixelWord(std::uint8_t value) {
    return static_cast<std::uint32_t>(value) * 0x01010101u;
}

// Records draws 1..draws with a dispatch after each draw in dispatchesAfter (ascending); dispatch k
// stores its output at Output + k * OutputStride.
void Run(const std::string& name, std::uint32_t draws, const std::vector<std::uint32_t>& dispatchesAfter) {
    Require(dispatchesAfter.size() <= MaxDispatches, name + ": too many dispatches");
    std::fill_n(ColorMemory, ColorBytes, std::byte{0});
    std::fill_n(Output, MaxDispatches * OutputStride, 0xdeadbeefu);
    EndOfPipe = 0;
    std::vector<std::uint32_t> words;
    State(words);
    std::size_t dispatches = 0;
    for (std::uint32_t draw = 1; draw <= draws; ++draw) {
        Draw(words, ValueOf(draw));
        if (dispatches < dispatchesAfter.size() && dispatchesAfter[dispatches] == draw) {
            Dispatch(words, ColorMemory, Output + dispatches * OutputStride);
            ++dispatches;
        }
    }
    Require(dispatches == dispatchesAfter.size(), name + ": a dispatch names a draw past the last one");
    ReleaseLabel(words, &EndOfPipe, 1u);
    Packet packet{words.data(), static_cast<std::uint32_t>(words.size()), 0, {}};
    Require(sceAgcDriverSubmitDcb(&packet) == 0, name + ": DCB submission failed");
    AgcDriverWaitIdle_nid_postfix();
    Require(EndOfPipe == 1u, name + ": the end-of-pipe label was not written");
    AgcDriver::GuestMemory::FlushGpuWrites(reinterpret_cast<std::uintptr_t>(Output), MaxDispatches * OutputStride * 4u);
    for (std::size_t dispatch = 0; dispatch < dispatches; ++dispatch) {
        const auto after = dispatchesAfter[dispatch];
        const auto expected = PixelWord(ValueOf(after)) + 1u;
        for (std::uint32_t lane = 0; lane < Lanes; ++lane) {
            const auto stored = Output[dispatch * OutputStride + lane];
            Require(stored == expected, name + ": the dispatch after draw " + std::to_string(after) + " stored " + std::to_string(stored) + " in lane " + std::to_string(lane) + ", expected " + std::to_string(expected) + " (it read " + (stored == 1u ? "the cleared target" : "another draw's pixels") + ")");
        }
    }
    std::vector<std::byte> colors(ColorBytes);
    AgcDriver::GuestMemory::Read(static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(ColorMemory)), colors);
    for (std::size_t index = 0; index < ColorBytes; ++index) {
        const auto stored = std::to_integer<std::uint8_t>(colors[index]);
        Require(stored == ValueOf(draws), name + ": byte " + std::to_string(index) + " of the target stored " + std::to_string(stored) + " after the submission, expected the last draw's " + std::to_string(ValueOf(draws)));
    }
}

}

int main() {
    try {
        {
            const auto device = OpenVulkanTestDevice();
            if (!device) return VulkanTestSkipped;
        }
        ColorMemory = AllocateWatched(Block);
        GuestAllocations::Mutation().Add(ColorMemory, Block, true, true, false);
        Output = NewBlock();
        Register(VertexShader, VertexCode.data(), sizeof(VertexCode), 2);
        Register(PixelShader, ValueCode.data(), sizeof(ValueCode), 1);
        ComputeShader.file_header = 0x34333231;
        ComputeShader.version = 0x18;
        ComputeShader.header_size = sizeof(Shader);
        ComputeShader.shader_size = sizeof(IncrementCode);
        ComputeShader.code = IncrementCode.data();
        AgcDriverRegisterShader_nid_postfix(&ComputeShader);
        for (int round = 0; round < 2; ++round) {
            const auto suffix = " (round " + std::to_string(round + 1) + ")";
            Run("a dispatch after draw 3 of 5" + suffix, 5, {3});
            Run("a dispatch after draw 2 of 5" + suffix, 5, {2});
            Run("dispatches after draws 1, 2, 4 and 7 of 8" + suffix, 8, {1, 2, 4, 7});
            Run("a dispatch after every draw of 6" + suffix, 6, {1, 2, 3, 4, 5, 6});
        }
        AgcDriverShutdown_nid_postfix();
        std::puts("chunked dispatch reads passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        AgcDriverShutdown_nid_postfix();
        return 1;
    }
}
