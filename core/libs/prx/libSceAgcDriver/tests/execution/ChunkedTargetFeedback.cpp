#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
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
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// One submission renders into a target, draws elsewhere for a few draws, then samples the target as a
// texture into a second target, writes the first target again and samples it into a third. With
// draws recorded in chunks the writer, the reader and the rewriter land in different chunks (or one
// chunk holds a reader and the writer after it): the second target must hold the first value and
// the third the rewritten one.

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
constexpr std::uint32_t Format8888Unorm = 56;
constexpr std::uint32_t TileR64KBX = 27;
constexpr std::uint32_t Type2D = 9;
constexpr std::size_t Block = 65536;
constexpr std::size_t ColorBytes = Width * Height * 4;
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

// v_mov_b32 v4, 0.5 / 64; v_mov_b32 v5, 0.5 / 16; image_sample_lz v[0:3], v[4:5], s[0:7], s[8:11]
// dmask 0xf dim 2D (the encoding of StagedClearSampledRead with other registers); s_waitcnt vmcnt(0);
// exp mrt0 v0, v1, v2, v3 done vm: every pixel copies texel (0, 0) of the sampled target.
alignas(256) constexpr std::array<std::uint32_t, 10> SampleCode{
    0x7e0802ff, 0x3c000000, 0x7e0a02ff, 0x3d000000, 0xf09c0f08, 0x00400004, 0xbf8c3f70, 0xf800180f, 0x03020100, 0xbf810000,
};

constexpr std::array<std::uint32_t, 4> PointSampler{0x00000092u, 0x00fff000u, 0u, 0u};

alignas(256) constexpr std::array<std::array<float, 4>, 3> Vertices{{
    {-1.0f, -1.0f, 0.5f, 1.0f}, {3.0f, -1.0f, 0.5f, 1.0f}, {-1.0f, 3.0f, 0.5f, 1.0f},
}};

struct RegisteredShader {
    Shader shader;
    ShaderUserData userData;
};
RegisteredShader VertexShader{};
RegisteredShader ValueShader{};
RegisteredShader SampleShader{};

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

// A 128x128 8_8_8_8 UNORM view of a target in its swizzle (TiledAttrib3), the layout the color
// target registers describe.
std::array<std::uint32_t, 8> TextureDescriptor(const std::byte* texels) {
    const auto address = reinterpret_cast<std::uintptr_t>(texels);
    return {
        static_cast<std::uint32_t>(address >> 8u),
        static_cast<std::uint32_t>((address >> 40u) & 0xffu) | (Format8888Unorm << 20u) | (((Width - 1u) & 3u) << 30u),
        ((Width - 1u) >> 2u) | ((Height - 1u) << 14u),
        0xfacu | (TileR64KBX << 20u) | (Type2D << 28u),
        0u,
        0u,
        0u,
        0u,
    };
}

void SetRegister(std::vector<std::uint32_t>& words, std::uint32_t opcode, std::uint32_t offset, std::uint32_t value) {
    words.push_back(0xc0010000u | (opcode << 8u));
    words.push_back(offset);
    words.push_back(value);
}

void PixelProgram(std::vector<std::uint32_t>& words, const void* code, std::uint32_t userSgprs) {
    const auto address = reinterpret_cast<std::uintptr_t>(code);
    SetRegister(words, SetShReg, 0x008, static_cast<std::uint32_t>(address >> 8u));
    SetRegister(words, SetShReg, 0x009, static_cast<std::uint32_t>(address >> 40u));
    SetRegister(words, SetShReg, 0x00b, userSgprs << 1u);
}

void State(std::vector<std::uint32_t>& words) {
    const auto vertexAddress = reinterpret_cast<std::uintptr_t>(VertexCode.data());
    for (const auto& [offset, value] : ContextBaseline) SetRegister(words, SetContextReg, offset, value);
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
    SetRegister(words, SetShReg, 0x0c8, static_cast<std::uint32_t>(vertexAddress >> 8u));
    SetRegister(words, SetShReg, 0x0c9, static_cast<std::uint32_t>(vertexAddress >> 40u));
    SetRegister(words, SetShReg, 0x08b, 4u << 1u);
    const auto descriptor = BufferDescriptor(Vertices.data(), 16u, static_cast<std::uint32_t>(Vertices.size()));
    for (std::uint32_t i = 0; i < descriptor.size(); ++i) SetRegister(words, SetShReg, 0x08c + i, descriptor[i]);
}

void Target(std::vector<std::uint32_t>& words, const std::byte* color) {
    const auto address = reinterpret_cast<std::uintptr_t>(color);
    SetRegister(words, SetContextReg, 0x318, static_cast<std::uint32_t>(address >> 8u));
    SetRegister(words, SetContextReg, 0x390, static_cast<std::uint32_t>(address >> 40u));
}

void DrawCall(std::vector<std::uint32_t>& words) {
    words.insert(words.end(), {0xc0000000u | (1u << 16u) | (DrawIndexAuto << 8u), static_cast<std::uint32_t>(Vertices.size()), 2u});
}

void DrawValue(std::vector<std::uint32_t>& words, const std::byte* color, std::uint8_t value) {
    Target(words, color);
    PixelProgram(words, ValueCode.data(), 1);
    SetRegister(words, SetShReg, 0x00c, std::bit_cast<std::uint32_t>(static_cast<float>(value) / 255.0f));
    DrawCall(words);
}

void DrawSampled(std::vector<std::uint32_t>& words, const std::byte* color, const std::byte* texture) {
    Target(words, color);
    PixelProgram(words, SampleCode.data(), 12);
    const auto descriptor = TextureDescriptor(texture);
    for (std::uint32_t i = 0; i < descriptor.size(); ++i) SetRegister(words, SetShReg, 0x00c + i, descriptor[i]);
    for (std::uint32_t i = 0; i < PointSampler.size(); ++i) SetRegister(words, SetShReg, 0x014 + i, PointSampler[i]);
    DrawCall(words);
}

void ReleaseLabel(std::vector<std::uint32_t>& words, volatile std::uint32_t* target, std::uint32_t value) {
    const auto address = reinterpret_cast<std::uintptr_t>(target);
    words.insert(words.end(), {0xc0064900u, 0x514u, (1u << 29u) | (2u << 24u), static_cast<std::uint32_t>(address), static_cast<std::uint32_t>(static_cast<std::uint64_t>(address) >> 32u), value, 0u, 0u});
}

void Check(const std::string& what, const std::byte* color, std::uint8_t value) {
    std::vector<std::byte> colors(ColorBytes);
    // As DualSourceBlend's read-back: the resident image's results are stored back and landed first.
    AgcDriver::Graphics::StorageTexture::FlushPending(reinterpret_cast<std::uintptr_t>(color), ColorBytes, nullptr, "test read-back");
    AgcDriverWaitIdle_nid_postfix();
    AgcDriver::GuestMemory::Read(static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(color)), colors);
    for (std::size_t index = 0; index < ColorBytes; ++index) {
        const auto stored = std::to_integer<std::uint8_t>(colors[index]);
        const auto pixel = index / 4u;
        Require(stored == value, what + ": pixel (" + std::to_string(pixel % Width) + ", " + std::to_string(pixel / Width) + ") channel " + std::to_string(index % 4u) + " stored " + std::to_string(stored) + ", expected " + std::to_string(value));
    }
}

struct Targets {
    std::byte* written = nullptr;
    std::byte* firstRead = nullptr;
    std::byte* secondRead = nullptr;
    std::byte* filler = nullptr;
};

// written <- first; fillers draws into filler; firstRead <- sample(written); written <- second;
// filler <- one more; secondRead <- sample(written).
void Run(const std::string& name, const Targets& targets, std::uint32_t fillers, std::uint8_t first, std::uint8_t second) {
    for (auto* color : {targets.written, targets.firstRead, targets.secondRead, targets.filler}) std::fill_n(color, ColorBytes, std::byte{0});
    EndOfPipe = 0;
    std::vector<std::uint32_t> words;
    State(words);
    DrawValue(words, targets.written, first);
    std::uint8_t filled = 0;
    for (std::uint32_t filler = 0; filler < fillers; ++filler) {
        filled = static_cast<std::uint8_t>(10u + 10u * filler);
        DrawValue(words, targets.filler, filled);
    }
    DrawSampled(words, targets.firstRead, targets.written);
    DrawValue(words, targets.written, second);
    filled = static_cast<std::uint8_t>(filled + 5u);
    DrawValue(words, targets.filler, filled);
    DrawSampled(words, targets.secondRead, targets.written);
    ReleaseLabel(words, &EndOfPipe, 1u);
    Packet packet{words.data(), static_cast<std::uint32_t>(words.size()), 0, {}};
    Require(sceAgcDriverSubmitDcb(&packet) == 0, name + ": DCB submission failed");
    AgcDriverWaitIdle_nid_postfix();
    Require(EndOfPipe == 1u, name + ": the end-of-pipe label was not written");
    Check(name + ": the target sampled after its first draw", targets.firstRead, first);
    Check(name + ": the target sampled after the target was drawn again", targets.secondRead, second);
    Check(name + ": the sampled target", targets.written, second);
    Check(name + ": the filler target", targets.filler, filled);
}

}

int main() {
    try {
        {
            const auto device = OpenVulkanTestDevice();
            if (!device) return VulkanTestSkipped;
        }
        Targets targets;
        targets.written = AllocateWatched(Block);
        targets.firstRead = AllocateWatched(Block);
        targets.secondRead = AllocateWatched(Block);
        targets.filler = AllocateWatched(Block);
        Register(VertexShader, VertexCode.data(), sizeof(VertexCode), 2);
        Register(ValueShader, ValueCode.data(), sizeof(ValueCode), 1);
        Register(SampleShader, SampleCode.data(), sizeof(SampleCode), 1);
        for (int round = 0; round < 2; ++round) {
            const auto suffix = " (round " + std::to_string(round + 1) + ")";
            // Three fillers put the writer and its reader two chunks of two draws apart, and the
            // reader and the rewrite in one chunk; four shift every boundary by one draw.
            Run("three draws between the writer and its reader" + suffix, targets, 3, 200, 60);
            Run("four draws between the writer and its reader" + suffix, targets, 4, 120, 240);
            Run("no draw between the writer and its reader" + suffix, targets, 0, 90, 180);
        }
        AgcDriverShutdown_nid_postfix();
        std::puts("chunked target feedback passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        AgcDriverShutdown_nid_postfix();
        return 1;
    }
}
