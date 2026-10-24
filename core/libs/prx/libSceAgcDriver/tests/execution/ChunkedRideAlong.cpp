#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
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
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using AgcDriver::Graphics::Recorder;
using AgcDriver::Graphics::Require;

constexpr std::uint32_t Width = 128;
constexpr std::uint32_t Height = 128;
constexpr std::uint32_t TiledAttrib3 = 0x0906c000u;
constexpr std::uint32_t SetContextReg = 0x69u;
constexpr std::uint32_t SetShReg = 0x76u;
constexpr std::uint32_t SetUconfigReg = 0x79u;
constexpr std::uint32_t DrawIndexAuto = 0x2du;
constexpr std::size_t Block = 65536;
constexpr std::size_t ColorBytes = Width * Height * 4;
constexpr std::uint32_t VertexCount = 3;
constexpr std::size_t VertexBytes = VertexCount * 4 * sizeof(float);
constexpr std::size_t LabelSlots = 4;
std::byte* ColorMemory = nullptr;
float* VertexMemory = nullptr;
volatile std::uint32_t* Labels = nullptr;
volatile std::uint32_t EndOfPipe = 0;

constexpr std::array<std::array<float, 4>, VertexCount> FullTarget{{
    {-1.0f, -1.0f, 0.5f, 1.0f}, {3.0f, -1.0f, 0.5f, 1.0f}, {-1.0f, 3.0f, 0.5f, 1.0f},
}};
constexpr std::size_t ThirdVertexY = 2 * 4 + 1;

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
    GuestAllocations::Mutation().Add(block, bytes, true, true, false);
    return static_cast<std::byte*>(block);
}

alignas(256) constexpr std::array<std::uint32_t, 6> VertexCode{
    0xe0382000, 0x80020005, 0xbf8c3f70, 0xf80008cf, 0x03020100, 0xbf810000,
};

alignas(256) constexpr std::array<std::uint32_t, 4> ValueCode{
    0x7e000200, 0xf800180f, 0x00000000, 0xbf810000,
};

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
    const auto descriptor = BufferDescriptor(VertexMemory, 16u, VertexCount);
    for (std::uint32_t i = 0; i < descriptor.size(); ++i) SetRegister(words, SetShReg, 0x08c + i, descriptor[i]);
}

void Draw(std::vector<std::uint32_t>& words, std::uint8_t value) {
    SetRegister(words, SetShReg, 0x00c, std::bit_cast<std::uint32_t>(static_cast<float>(value) / 255.0f));
    words.insert(words.end(), {0xc0000000u | (1u << 16u) | (DrawIndexAuto << 8u), VertexCount, 2u});
}

void ReleaseLabel(std::vector<std::uint32_t>& words, volatile void* target, std::uint32_t value, bool interrupt = true) {
    words.insert(words.end(), {0xc0064900u, 0x514u, (1u << 29u) | ((interrupt ? 2u : 0u) << 24u), Low(target), High(target), value, 0u, 0u});
}

void WriteData(std::vector<std::uint32_t>& words, volatile void* target, std::uint32_t value) {
    words.insert(words.end(), {0xc0033700u, 0x00100200u, Low(target), High(target), value});
}

std::uint8_t ValueOf(std::uint32_t draw) {
    return static_cast<std::uint8_t>(15u + 20u * draw);
}

std::uint32_t LabelValue(std::uint32_t round, std::uint32_t label) {
    return 0x7a1e0000u | (round << 8u) | label;
}

std::string Mismatch(std::uint8_t value) {
    std::vector<std::byte> colors(ColorBytes);
    AgcDriver::Graphics::StorageTexture::FlushPending(reinterpret_cast<std::uintptr_t>(ColorMemory), ColorBytes, nullptr, "test read-back");
    AgcDriverWaitIdle_nid_postfix();
    AgcDriver::GuestMemory::Read(static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(ColorMemory)), colors);
    for (std::size_t index = 0; index < ColorBytes; ++index) {
        const auto stored = std::to_integer<std::uint8_t>(colors[index]);
        if (stored != value) {
            const auto pixel = index / 4u;
            return "pixel (" + std::to_string(pixel % Width) + ", " + std::to_string(pixel / Width) + ") stored " + std::to_string(stored) + ", expected " + std::to_string(value);
        }
    }
    return {};
}

struct Chunks {
    std::uint64_t chunks = 0;
    std::uint64_t draws = 0;
};

Chunks Totals() {
    auto* recorder = Recorder::Active();
    if (recorder == nullptr) return {};
    const auto totals = recorder->ChunkTotals();
    return {totals.chunks, totals.draws};
}

bool Chunking() {
    auto* recorder = Recorder::Active();
    return Recorder::DrawRecordThreads() != 0 && recorder != nullptr && recorder->Recorders() != nullptr;
}

enum class Between : std::uint8_t { WriteData, ReleaseMem, Interrupt };

struct Label {
    std::uint32_t afterDraw;
    Between kind;
    std::uint32_t slot;
};

void Submit(std::vector<std::uint32_t>& words, const std::string& name) {
    ReleaseLabel(words, &EndOfPipe, 1u);
    Packet packet{words.data(), static_cast<std::uint32_t>(words.size()), 0, {}};
    Require(sceAgcDriverSubmitDcb(&packet) == 0, name + ": DCB submission failed");
    AgcDriverWaitIdle_nid_postfix();
    Require(EndOfPipe == 1u, name + ": the end-of-pipe label was not written");
}

void Reset() {
    std::fill_n(ColorMemory, ColorBytes, std::byte{0});
    std::memcpy(VertexMemory, FullTarget.data(), VertexBytes);
    for (std::size_t slot = 0; slot < LabelSlots; ++slot) Labels[slot] = 0;
    EndOfPipe = 0;
}

// The draws of the stream go to ceil(draws / ChunkDraws) chunks when nothing between them records
// commands, whatever labels the stream stores between them, except that an end-of-pipe interrupt
// submits the batch and so ends the chunk (`cuts`: the draws such a label follows); with
// APS5_NO_CHUNK_RIDE_ALONG every label seals the open chunk.
void CheckChunks(const std::string& name, const Chunks& before, std::uint32_t draws, const std::vector<std::uint32_t>& cuts) {
    if (!Chunking()) return;
    const auto after = Totals();
    const auto chunks = after.chunks - before.chunks;
    const auto chunked = after.draws - before.draws;
    std::printf("chunked ride-along: %s: %llu draws in %llu chunks (cap %zu)\n", name.c_str(), static_cast<unsigned long long>(chunked), static_cast<unsigned long long>(chunks), Recorder::ChunkDraws());
    Require(chunked == draws, name + ": " + std::to_string(chunked) + " of " + std::to_string(draws) + " draws were chunked");
    if (std::getenv("APS5_NO_CHUNK_RIDE_ALONG") != nullptr) return;
    std::uint64_t expected = 0;
    std::uint32_t first = 1;
    for (const auto cut : cuts) {
        expected += (cut - first + Recorder::ChunkDraws()) / Recorder::ChunkDraws();
        first = cut + 1;
    }
    expected += (draws - first + Recorder::ChunkDraws()) / Recorder::ChunkDraws();
    Require(chunks == expected, name + ": " + std::to_string(draws) + " draws took " + std::to_string(chunks) + " chunks, expected " + std::to_string(expected));
}

void RunLabels(const std::string& name, std::uint32_t round, std::uint32_t draws, const std::vector<Label>& labels) {
    Reset();
    std::vector<std::uint32_t> words;
    State(words);
    std::array<std::uint32_t, LabelSlots> expected{};
    std::size_t next = 0;
    for (std::uint32_t draw = 1; draw <= draws; ++draw) {
        Draw(words, ValueOf(draw));
        for (; next < labels.size() && labels[next].afterDraw == draw; ++next) {
            const auto& label = labels[next];
            const auto value = LabelValue(round, static_cast<std::uint32_t>(next));
            if (label.kind == Between::WriteData) WriteData(words, &Labels[label.slot], value);
            else ReleaseLabel(words, &Labels[label.slot], value, label.kind == Between::Interrupt);
            expected[label.slot] = value;
        }
    }
    Require(next == labels.size(), name + ": a label names a draw past the last one");
    const auto before = Totals();
    Submit(words, name);
    for (std::size_t slot = 0; slot < LabelSlots; ++slot) {
        Require(Labels[slot] == expected[slot], name + ": label slot " + std::to_string(slot) + " holds " + std::to_string(Labels[slot]) + ", expected " + std::to_string(expected[slot]));
    }
    const auto last = Mismatch(ValueOf(draws));
    Require(last.empty(), name + ": after the submission, " + last);
    std::vector<std::uint32_t> cuts;
    for (const auto& label : labels) {
        if (label.kind == Between::Interrupt && label.afterDraw < draws && (cuts.empty() || cuts.back() != label.afterDraw)) cuts.push_back(label.afterDraw);
    }
    CheckChunks(name, before, draws, cuts);
}

// Draw 1 covers the target; a WRITE_DATA then moves the third vertex onto the first two, so the
// triangle of draw 2 covers nothing. A draw 2 recorded ahead of the store (one that joined the
// chunk of draw 1 while the store went into the spine's command buffer behind it) covers the
// target again.
void RunLabelReadByNextDraw(const std::string& name) {
    Reset();
    std::vector<std::uint32_t> words;
    State(words);
    Draw(words, ValueOf(1));
    WriteData(words, &VertexMemory[ThirdVertexY], std::bit_cast<std::uint32_t>(-1.0f));
    Draw(words, ValueOf(2));
    Submit(words, name);
    const auto last = Mismatch(ValueOf(1));
    Require(last.empty(), name + ": draw 2 ran before the store that moved its vertex; " + last);
}

}

int main() {
    try {
        {
            const auto device = OpenVulkanTestDevice();
            if (!device) return VulkanTestSkipped;
        }
        ColorMemory = AllocateWatched(Block);
        VertexMemory = reinterpret_cast<float*>(AllocateWatched(Block));
        Labels = reinterpret_cast<volatile std::uint32_t*>(AllocateWatched(Block));
        Register(VertexShader, VertexCode.data(), sizeof(VertexCode), 2);
        Register(PixelShader, ValueCode.data(), sizeof(ValueCode), 1);
        for (std::uint32_t round = 0; round < 2; ++round) {
            const auto suffix = " (round " + std::to_string(round + 1) + ")";
            RunLabels("seven plain draws" + suffix, round, 7, {});
            RunLabels("write-data labels between draws" + suffix, round, 7, {{1, Between::WriteData, 0}, {2, Between::WriteData, 1}, {4, Between::WriteData, 0}, {6, Between::WriteData, 2}});
            RunLabels("release-mem labels between draws" + suffix, round, 7, {{1, Between::ReleaseMem, 0}, {3, Between::ReleaseMem, 1}, {3, Between::ReleaseMem, 0}, {5, Between::ReleaseMem, 3}});
            RunLabels("a label after every draw" + suffix, round, 5, {{1, Between::WriteData, 0}, {2, Between::ReleaseMem, 0}, {3, Between::WriteData, 0}, {4, Between::ReleaseMem, 0}, {5, Between::WriteData, 0}});
            RunLabels("an end-of-pipe interrupt between draws" + suffix, round, 7, {{2, Between::ReleaseMem, 0}, {3, Between::Interrupt, 1}, {5, Between::WriteData, 2}});
            RunLabelReadByNextDraw("a store the next draw reads" + suffix);
        }
        AgcDriverShutdown_nid_postfix();
        std::puts("chunked ride-along passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        AgcDriverShutdown_nid_postfix();
        return 1;
    }
}
