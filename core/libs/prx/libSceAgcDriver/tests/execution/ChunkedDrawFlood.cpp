#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DrawRecorders.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Submit/include/Dcb.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
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
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Two thousand small draws in one DCB, every hundredth followed by a label (RELEASE_MEM and// WRITE_DATA in turn). With draws recorded in chunks of two this is a thousand chunks and twenty// sealing points: the submission must finish (ctest's TIMEOUT catches a hang), every label must// land, the target must hold the last draw, and the recorder's caps must hold (CheckCaps).

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
constexpr std::uint32_t Draws = 2000;
constexpr std::uint32_t DrawsPerSeal = 100;
constexpr std::uint32_t Seals = Draws / DrawsPerSeal;
constexpr std::size_t Block = 65536;
constexpr std::size_t ColorBytes = Width * Height * 4;
std::byte* ColorMemory = nullptr;
alignas(256) volatile std::uint32_t SealLabels[Seals]{};
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
    // Registered like a guest allocation: a resident target's results are stored back only into
    // registered memory.
    GuestAllocations::Mutation().Add(block, bytes, true, true, false);
    return static_cast<std::byte*>(block);
}

// The vertex shader of SampleMaskDraw: one full-target triangle fetched from a vertex buffer.
alignas(256) constexpr std::array<std::uint32_t, 6> VertexCode{
    0xe0382000, 0x80020005, 0xbf8c3f70, 0xf80008cf, 0x03020100, 0xbf810000,
};

// v_mov_b32 v0, s0; exp mrt0 v0, v0, v0, v0 done vm.
alignas(256) constexpr std::array<std::uint32_t, 4> ValueCode{
    0x7e000200, 0xf800180f, 0x00000000, 0xbf810000,
};

alignas(256) constexpr std::array<std::array<float, 4>, 3> Vertices{{
    {-1.0f, -1.0f, 0.5f, 1.0f}, {3.0f, -1.0f, 0.5f, 1.0f}, {-1.0f, 3.0f, 0.5f, 1.0f},
}};

struct RegisteredShader {
    Shader shader;
    ShaderUserData userData;
};
RegisteredShader VertexShader{};
RegisteredShader PixelShader{};

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

std::array<std::uint32_t, 4> BufferDescriptor(const void* data, std::uint32_t stride, std::uint32_t count) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu) | (stride << 16u), count, 0x01016facu};
}

std::uint32_t Low(const volatile void* pointer) {
    return static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(pointer));
}

std::uint32_t High(const volatile void* pointer) {
    return static_cast<std::uint32_t>(static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(pointer)) >> 32u);
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
    const auto descriptor = BufferDescriptor(Vertices.data(), 16u, static_cast<std::uint32_t>(Vertices.size()));
    for (std::uint32_t i = 0; i < descriptor.size(); ++i) SetRegister(words, SetShReg, 0x08c + i, descriptor[i]);
}

void Draw(std::vector<std::uint32_t>& words, std::uint8_t value) {
    SetRegister(words, SetShReg, 0x00c, std::bit_cast<std::uint32_t>(static_cast<float>(value) / 255.0f));
    words.insert(words.end(), {0xc0000000u | (1u << 16u) | (DrawIndexAuto << 8u), static_cast<std::uint32_t>(Vertices.size()), 2u});
}


void ReleaseLabel(std::vector<std::uint32_t>& words, volatile std::uint32_t* target, std::uint32_t value) {
    words.insert(words.end(), {0xc0064900u, 0x514u, (1u << 29u) | (2u << 24u), Low(target), High(target), value, 0u, 0u});
}

void WriteData(std::vector<std::uint32_t>& words, volatile std::uint32_t* target, std::uint32_t value) {
    words.insert(words.end(), {0xc0033700u, 0x00100200u, Low(target), High(target), value});
}

// Consecutive draws store different values in 1..254.
std::uint8_t ValueOf(std::uint32_t draw) {
    return static_cast<std::uint8_t>(1u + (draw * 37u) % 254u);
}

std::uint32_t PixelWord(std::uint8_t value) {
    return static_cast<std::uint32_t>(value) * 0x01010101u;
}

std::uint32_t LabelValue(std::uint32_t round, std::uint32_t seal) {
    return 0x5ea10000u | (round << 8u) | seal;
}

void Run(std::uint32_t round) {
    const auto name = "round " + std::to_string(round + 1);
    std::fill_n(ColorMemory, ColorBytes, std::byte{0});
    for (auto& label : SealLabels) label = 0;
    EndOfPipe = 0;
    std::vector<std::uint32_t> words;
    words.reserve(Draws * 6u + Seals * 16u + 512u);
    State(words);
    for (std::uint32_t draw = 1; draw <= Draws; ++draw) {
        Draw(words, ValueOf(draw));
        if (draw % DrawsPerSeal != 0) continue;
        const auto seal = draw / DrawsPerSeal - 1u;
        if (seal % 2u == 0) {
            ReleaseLabel(words, &SealLabels[seal], LabelValue(round, seal));
        } else {
            WriteData(words, &SealLabels[seal], LabelValue(round, seal));
        }
    }
    ReleaseLabel(words, &EndOfPipe, 1u);
    const auto start = std::chrono::steady_clock::now();
    Packet packet{words.data(), static_cast<std::uint32_t>(words.size()), 0, {}};
    Require(sceAgcDriverSubmitDcb(&packet) == 0, name + ": DCB submission failed");
    AgcDriverWaitIdle_nid_postfix();
    const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
    Require(EndOfPipe == 1u, name + ": the end-of-pipe label was not written");
    for (std::uint32_t seal = 0; seal < Seals; ++seal) {
        Require(SealLabels[seal] == LabelValue(round, seal), name + ": the label after draw " + std::to_string((seal + 1u) * DrawsPerSeal) + " holds " + std::to_string(SealLabels[seal]) + ", expected " + std::to_string(LabelValue(round, seal)));
    }
    std::vector<std::byte> colors(ColorBytes);
    AgcDriver::Graphics::StorageTexture::FlushPending(reinterpret_cast<std::uintptr_t>(ColorMemory), ColorBytes, nullptr, "test read-back");
    AgcDriverWaitIdle_nid_postfix();
    AgcDriver::GuestMemory::Read(static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(ColorMemory)), colors);
    for (std::size_t index = 0; index < ColorBytes; ++index) {
        const auto stored = std::to_integer<std::uint8_t>(colors[index]);
        Require(stored == ValueOf(Draws), name + ": byte " + std::to_string(index) + " of the target stored " + std::to_string(stored) + ", expected the last draw's " + std::to_string(ValueOf(Draws)));
    }
    std::printf("chunked draw flood: %s, %u draws, %u labels in %lld ms\n", name.c_str(), Draws, Seals + 1u, static_cast<long long>(milliseconds));
}

// With recorder threads, the caps held: no recorder thread queued more than its depth, no batch
// had more than MaxChunksInFlight unfinished chunks, and the draws went through chunks.
void CheckCaps() {
    using AgcDriver::Graphics::DrawRecorders;
    using AgcDriver::Graphics::Recorder;
    if (Recorder::DrawRecordThreads() == 0) return;
    auto* recorder = Recorder::Active();
    Require(recorder != nullptr, "no device recorder after the draws");
    const auto totals = recorder->ChunkTotals();
    Require(totals.chunks != 0 && totals.draws >= totals.chunks, "the draws were not recorded in chunks");
    Require(totals.maxInFlight <= Recorder::MaxChunksInFlight, "a batch had " + std::to_string(totals.maxInFlight) + " chunks in flight, over the cap");
    Require(totals.failedChunks == 0, "a chunk failed");
    auto* recorders = recorder->Recorders();
    Require(recorders != nullptr && recorders->MaxQueued() <= DrawRecorders::QueueDepth, "a recorder thread queued over its depth");
    std::printf("chunked draw flood: %llu chunks, %llu chunked draws, %llu chunks in flight at most, %zu items queued at most\n", static_cast<unsigned long long>(totals.chunks), static_cast<unsigned long long>(totals.draws), static_cast<unsigned long long>(totals.maxInFlight), recorders->MaxQueued());
}

}

int main() {
    try {
        {
            const auto device = OpenVulkanTestDevice();
            if (!device) return VulkanTestSkipped;
        }
        ColorMemory = AllocateWatched(Block);
        Register(VertexShader, VertexCode.data(), sizeof(VertexCode), 2);
        Register(PixelShader, ValueCode.data(), sizeof(ValueCode), 1);
        for (std::uint32_t round = 0; round < 2; ++round) Run(round);
        CheckCaps();
        AgcDriverShutdown_nid_postfix();
        std::puts("chunked draw flood passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        AgcDriverShutdown_nid_postfix();
        return 1;
    }
}
