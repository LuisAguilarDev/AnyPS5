#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Submit/include/Dcb.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "VulkanTestDevice.hpp"
#ifdef _WIN32
#include <windows.h>
#endif
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
#include <new>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// Streams of lean draws that hit the caches, the kinds a game's frame is made of after its first
// frames: `hits` repeats one draw (draw-cache hits), `data` changes a pixel user word on every draw
// (data-only hits of one template), `moved` points a pixel stage that loads its color through a
// user-data pointer at another table on every draw (template hits whose data buffer moved, the
// MovedReadOnlyBuffers path), `rings` moves six such pointers per draw, `buffers` four guest
// buffer descriptors and `textures` samples four guest textures through one table pointer (template
// hits whose revalidation proves textures). The default run submits a short stream of each and
// checks that the draws hit; `--benchmark hits|data|moved|rings|buffers|textures [seconds]` submits
// one kind for the given time
// (default 21 s) so the APS5_PROFILE_DRAW=1 lines ([draws], [spine]) cover two whole 10 s windows.

namespace {

using AgcDriver::Graphics::Require;

alignas(256) constexpr std::array<std::uint32_t, 8> VertexCode{0x7e000208u, 0x7e020280u, 0x7e040280u, 0x7e0602f2u, 0xf80008cfu, 0x03020100u, 0xbf810000u, 0xbf810000u};
alignas(256) constexpr std::array<std::uint32_t, 8> PixelCode{0x7e000200u, 0x7e020280u, 0xf800180fu, 0x00010100u, 0xbf810000u, 0xbf810000u, 0xbf810000u, 0xbf810000u};
// s_load_dword s4 from the table at user words 0-1, then exports it (DrawCacheRelocation's stage).
alignas(256) constexpr std::array<std::uint32_t, 8> LoadingPixelCode{0xf4000100u, 0xfa000000u, 0x7e000204u, 0x7e020280u, 0xf800180fu, 0x00010100u, 0xbf810000u, 0xbf810000u};
// s_load_dword s12-s17 from the six tables at user words 0-11, then exports their sum: a stage
// reading several constant-ring slices, each of which moves on every draw as a game's do.
constexpr std::uint32_t RingCount = 6;
alignas(256) constexpr std::array<std::uint32_t, 24> RingPixelCode{
    0xf4000300u, 0xfa000000u, 0xf4000341u, 0xfa000000u, 0xf4000382u, 0xfa000000u,
    0xf40003c3u, 0xfa000000u, 0xf4000404u, 0xfa000000u, 0xf4000445u, 0xfa000000u,
    0xbf8cc07fu, 0x7e00020cu, 0x0600000du, 0x0600000eu, 0x0600000fu, 0x06000010u, 0x06000011u,
    0x7e020280u, 0xf800180fu, 0x00010100u, 0xbf810000u, 0xbf810000u};
// buffer_load_dword v0-v3 through the V#s at user words 0-15, then exports their sum: a stage
// reading four guest buffers (constant-ring slices bound as buffer descriptors), each of which
// moves on every draw.
constexpr std::uint32_t BufferCount = 4;
alignas(256) constexpr std::array<std::uint32_t, 20> BufferPixelCode{
    0xe0300000u, 0x80000000u, 0xe0300000u, 0x80010100u, 0xe0300000u, 0x80020200u, 0xe0300000u, 0x80030300u,
    0xbf8c3f70u, 0x06000101u, 0x06000102u, 0x06000103u,
    0x7e020280u, 0xf800180fu, 0x00010100u, 0xbf810000u, 0xbf810000u, 0xbf810000u, 0xbf810000u, 0xbf810000u};
// s_load_dwordx8 s[16:23], s[24:31], s[32:39], s[40:47] (four T#s) and s_load_dwordx4 s[48:51] (an
// S#) from the table at user words 0-1, then image_sample_lz of texel (0, 0) through each T# into
// v[0:3], v[6:9], v[10:13], v[14:17] and exports v0 + v6 + v10 + v14, v1, v2, v3: a stage sampling
// four textures, as a game's material draws do.
constexpr std::uint32_t TextureCount = 4;
alignas(256) constexpr std::array<std::uint32_t, 34> TexturePixelCode{
    0xf40c0400u, 0xfa000000u, 0xf40c0600u, 0xfa000020u, 0xf40c0800u, 0xfa000040u, 0xf40c0a00u, 0xfa000060u,
    0xf4080c00u, 0xfa000080u, 0xbf8cc07fu,
    0x7e0802ffu, 0x3c000000u, 0x7e0a02ffu, 0x3d000000u,
    0xf09c0f08u, 0x01840004u, 0xf09c0f08u, 0x01860604u, 0xf09c0f08u, 0x01880a04u, 0xf09c0f08u, 0x018a0e04u,
    0xbf8c3f70u, 0x06000106u, 0x0600010au, 0x0600010eu,
    0xf800180fu, 0x03020100u, 0xbf810000u, 0xbf810000u, 0xbf810000u, 0xbf810000u, 0xbf810000u};
// The textures a `textures` stream samples: 128x128 8_8_8_8 UNORM in the R 64 KiB swizzle, one
// 64 KiB block each, and the table of their T#s and a point S# its draws point at.
constexpr std::uint32_t TextureSide = 128;
constexpr std::size_t TextureBytes = 65536;
alignas(256) std::array<std::uint32_t, TextureCount * 8u + 4u> TextureTable{};
std::byte* Texels = nullptr;
// The guest ring a `buffers` stream's descriptors point into, in slices of SliceBytes.
constexpr std::size_t RingBytes = 65536;
constexpr std::size_t SliceBytes = 256;
alignas(4096) std::array<std::uint32_t, RingBytes / 4> Ring{};
// The tables a `moved` or `rings` stream cycles through, each holding another color.
constexpr std::size_t TableCount = 64;
alignas(16) std::array<std::array<float, 4>, TableCount> Tables{};
alignas(256) std::array<std::byte, 1024> ColorMemory{};
// The color target: a write-watched guest arena block where there is one (Windows), so the draws
// are recorded into the open batch against a resident target as a game's are; else ColorMemory.
// 512x512 of 32 bits is 1 MiB, the size from which a host import of it keeps the write watch
// (as a game's render targets do), so its refreshes can be proved unchanged.
constexpr std::size_t TargetBytes = 0x100000;
constexpr std::uint32_t TargetSide = 512;
std::byte* ColorTarget = ColorMemory.data();

struct TargetBlock {
    TargetBlock() {
#ifdef _WIN32
        if (!AgcDriver::GuestMemory::WriteWatched()) return;
        data = static_cast<std::byte*>(GuestArena::GuestArenaAllocate_nid_postfix(TargetBytes, TargetBytes));
        GuestArena::GuestArenaCommit_nid_postfix(data, TargetBytes, PAGE_READWRITE, TargetBytes);
        Require(data != nullptr, "cannot allocate the color target");
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
#endif
    }
    std::byte* data = nullptr;
};

// The ring as guest memory, so its buffer descriptors are guest buffers.
struct RingBlock {
    RingBlock() { GuestAllocations::Mutation().Add(Ring.data(), RingBytes, true, true, true); }
    ~RingBlock() { GuestAllocations::Mutation().Remove(Ring.data()); }
};

// The `textures` stream's texels as guest memory, and the table of their descriptors. The texels
// are a write-watched guest arena block where there is one (Windows), so their revalidation is the
// stamp proof a game's textures get; else host memory, whose proof is the full walk.
struct TextureBlock {
    TextureBlock() {
#ifdef _WIN32
        if (AgcDriver::GuestMemory::WriteWatched()) {
            arena = static_cast<std::byte*>(GuestArena::GuestArenaAllocate_nid_postfix(TextureCount * TextureBytes, TextureBytes));
            Require(arena != nullptr, "cannot allocate the textures");
            GuestArena::GuestArenaCommit_nid_postfix(arena, TextureCount * TextureBytes, PAGE_READWRITE, TextureBytes);
        }
#endif
        data = arena != nullptr ? arena : static_cast<std::byte*>(::operator new(TextureCount * TextureBytes, std::align_val_t{TextureBytes}));
        for (std::uint32_t texture = 0; texture < TextureCount; ++texture) {
            auto* texels = data + texture * TextureBytes;
            std::fill(texels, texels + TextureBytes, static_cast<std::byte>(0x10u * (texture + 1u)));
            const auto address = reinterpret_cast<std::uintptr_t>(texels);
            constexpr std::uint32_t format8888Unorm = 56;
            constexpr std::uint32_t tileR64KBX = 27;
            constexpr std::uint32_t type2D = 9;
            const std::array<std::uint32_t, 8> descriptor{
                static_cast<std::uint32_t>(address >> 8u),
                static_cast<std::uint32_t>((address >> 40u) & 0xffu) | (format8888Unorm << 20u) | (((TextureSide - 1u) & 3u) << 30u),
                ((TextureSide - 1u) >> 2u) | ((TextureSide - 1u) << 14u),
                0xfacu | (tileR64KBX << 20u) | (type2D << 28u),
                0u, 0u, 0u, 0u};
            std::copy(descriptor.begin(), descriptor.end(), TextureTable.begin() + texture * 8u);
        }
        const std::array<std::uint32_t, 4> pointSampler{0x00000092u, 0x00fff000u, 0u, 0u};
        std::copy(pointSampler.begin(), pointSampler.end(), TextureTable.begin() + TextureCount * 8u);
        GuestAllocations::Mutation().Add(data, TextureCount * TextureBytes, true, true, true);
        Texels = data;
        GuestAllocations::Mutation().Add(TextureTable.data(), sizeof(TextureTable), true, true, true);
    }
    ~TextureBlock() {
        GuestAllocations::Mutation().Remove(TextureTable.data());
        GuestAllocations::Mutation().Remove(data);
#ifdef _WIN32
        if (arena != nullptr) {
            GuestArena::GuestArenaReset_nid_postfix(arena, TextureCount * TextureBytes);
            GuestArena::GuestArenaRelease_nid_postfix(arena, TextureCount * TextureBytes);
            return;
        }
#endif
        ::operator delete(data, std::align_val_t{TextureBytes});
    }
    std::byte* arena = nullptr;
    std::byte* data = nullptr;
};

struct Header {
    Shader shader{};
    ShaderUserData userData{};
};

alignas(8) Header VertexHeader{};
alignas(8) Header PixelHeader{};
alignas(8) Header LoadingPixelHeader{};
alignas(8) Header RingPixelHeader{};
alignas(8) Header BufferPixelHeader{};
alignas(8) Header TexturePixelHeader{};

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

// The pixel stage of a stream and the user words each of its draws sets from 0x00c on: one word,
// or one table pointer (two words) per table the stage reads.
struct PixelStage {
    const std::uint32_t* code;
    std::uint32_t userWords;
};

// The state of DrawCacheReuse's draws, then one draw per stage.userWords words of pixelWords.
std::vector<std::uint32_t> Draws(PixelStage stage, const std::vector<std::uint32_t>& pixelWords) {
    const auto color = reinterpret_cast<std::uintptr_t>(ColorTarget);
    const auto vertex = reinterpret_cast<std::uintptr_t>(VertexCode.data());
    const auto pixel = reinterpret_cast<std::uintptr_t>(stage.code);
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
        {0x31b, 0}, {0x31d, 0}, {0x3b0, ColorTarget == ColorMemory.data() ? (63u << 14u) | 3u : ((TargetSide - 1u) << 14u) | (TargetSide - 1u)},
        // CB_COLOR0_ATTRIB3 with the R 64 KiB swizzle (0x1b) on the arena block: a tiled target, kept
        // resident. ColorMemory is too small for one and stays linear.
        {0x3b8, ColorTarget == ColorMemory.data() ? 0x9000000u : (0x9000000u & ~(0x1fu << 14u)) | (0x1bu << 14u)}, {0x1e0, 0},
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
        {0x008, static_cast<std::uint32_t>(pixel >> 8u)}, {0x009, static_cast<std::uint32_t>(pixel >> 40u)}, {0x00b, stage.userWords << 1u},
    };
    std::vector<std::uint32_t> commands;
    Append(commands, 0x79, {0x242, 4});
    for (const auto& [offset, value] : context) Append(commands, 0x69, {offset, value});
    for (const auto& [offset, value] : shader) Append(commands, 0x76, {offset, value});
    for (std::size_t first = 0; first + stage.userWords <= pixelWords.size(); first += stage.userWords) {
        for (std::uint32_t word = 0; word < stage.userWords; ++word) Append(commands, 0x76, {0x00c + word, pixelWords[first + word]});
        Append(commands, 0x2d, {3, 2});
    }
    return commands;
}

enum class Kind { Hits, Data, Moved, Rings, Buffers, Textures };

// Appends the two user words of a pointer to the table `index` (modulo the table count).
void AppendTable(std::vector<std::uint32_t>& words, std::size_t index) {
    const auto address = reinterpret_cast<std::uintptr_t>(Tables[index % TableCount].data());
    words.push_back(static_cast<std::uint32_t>(address));
    words.push_back(static_cast<std::uint32_t>(address >> 32u));
}

// Appends the four words of a 16-byte buffer descriptor over the ring slice `index` (modulo the
// slice count), as DrawDataHitUserWords describes its buffer.
void AppendSlice(std::vector<std::uint32_t>& words, std::size_t index) {
    const auto address = reinterpret_cast<std::uintptr_t>(Ring.data()) + (index % (RingBytes / SliceBytes)) * SliceBytes;
    words.insert(words.end(), {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>(address >> 32u) & 0xffffu, 16u, 0x31016facu});
}

// One stream of `draws` draws of the kind; `round` varies a data, moved, rings or buffers stream
// between streams.
std::vector<std::uint32_t> Stream(Kind kind, std::size_t draws, std::uint32_t round) {
    std::vector<std::uint32_t> words;
    if (kind == Kind::Textures) {
        const auto table = reinterpret_cast<std::uintptr_t>(TextureTable.data());
        for (std::size_t draw = 0; draw < draws; ++draw) words.insert(words.end(), {static_cast<std::uint32_t>(table), static_cast<std::uint32_t>(table >> 32u)});
        return Draws({TexturePixelCode.data(), 2}, words);
    }
    if (kind == Kind::Buffers) {
        words.reserve(draws * BufferCount * 4u);
        for (std::size_t draw = 0; draw < draws; ++draw) {
            for (std::uint32_t buffer = 0; buffer < BufferCount; ++buffer) AppendSlice(words, draw * BufferCount + round + buffer);
        }
        return Draws({BufferPixelCode.data(), BufferCount * 4u}, words);
    }
    if (kind == Kind::Moved || kind == Kind::Rings) {
        const auto tables = kind == Kind::Moved ? 1u : RingCount;
        words.reserve(draws * tables * 2u);
        // Every table pointer moves on every draw, each ring to another table than the others.
        for (std::size_t draw = 0; draw < draws; ++draw) {
            for (std::uint32_t ring = 0; ring < tables; ++ring) AppendTable(words, draw + round + ring * 7u);
        }
        return Draws(kind == Kind::Moved ? PixelStage{LoadingPixelCode.data(), 2} : PixelStage{RingPixelCode.data(), RingCount * 2u}, words);
    }
    words.assign(draws, std::bit_cast<std::uint32_t>(1.0f));
    if (kind == Kind::Data) {
        for (std::size_t draw = 0; draw < draws; ++draw) words[draw] = std::bit_cast<std::uint32_t>(0.25f + static_cast<float>((draw + round) % 64u) / 128.0f);
    }
    return Draws({PixelCode.data(), 1}, words);
}

AgcDriver::DriverDetail::DrawEntryCounters Submit(const std::vector<std::uint32_t>& commands) {
    Packet packet{const_cast<std::uint32_t*>(commands.data()), static_cast<std::uint32_t>(commands.size()), 0, {}};
    Require(sceAgcDriverSubmitDcb(&packet) == 0, "the draw stream failed to submit");
    AgcDriverWaitIdle_nid_postfix();
    return AgcDriver::DriverDetail::Driver::Get().DrawCacheCounters();
}

void CheckStreams() {
    constexpr std::size_t draws = 64;
    const auto first = Submit(Stream(Kind::Hits, draws, 0));
    const auto hits = Submit(Stream(Kind::Hits, draws, 0));
    Require(hits.hits - first.hits >= draws - 1, "a repeated draw stream made " + std::to_string(hits.hits - first.hits) + " draw-cache hits of " + std::to_string(draws));
    const auto data = Submit(Stream(Kind::Data, draws, 1));
    Require(data.dataHits + data.hits - hits.dataHits - hits.hits >= draws / 2, "a stream changing one user word made " + std::to_string(data.dataHits - hits.dataHits) + " data hits and " + std::to_string(data.hits - hits.hits) + " hits of " + std::to_string(draws));
    const auto teach = Submit(Stream(Kind::Moved, draws, 0));
    const auto moved = Submit(Stream(Kind::Moved, draws, 1));
    Require(moved.lookups - teach.lookups == draws, "a stream moving one table pointer made " + std::to_string(moved.lookups - teach.lookups) + " draw-cache lookups of " + std::to_string(draws));
    const auto rings = Submit(Stream(Kind::Rings, draws, 0));
    const auto ringsAgain = Submit(Stream(Kind::Rings, draws, 1));
    Require(ringsAgain.lookups - rings.lookups == draws, "a stream moving six table pointers made " + std::to_string(ringsAgain.lookups - rings.lookups) + " draw-cache lookups of " + std::to_string(draws));
    const auto buffers = Submit(Stream(Kind::Buffers, draws, 0));
    const auto buffersAgain = Submit(Stream(Kind::Buffers, draws, 1));
    Require(buffersAgain.lookups - buffers.lookups == draws, "a stream moving four buffer descriptors made " + std::to_string(buffersAgain.lookups - buffers.lookups) + " draw-cache lookups of " + std::to_string(draws));
    const auto textures = Submit(Stream(Kind::Textures, draws, 0));
    const auto texturesAgain = Submit(Stream(Kind::Textures, draws, 1));
    Require(texturesAgain.hits - textures.hits >= draws - 1, "a repeated stream sampling four textures made " + std::to_string(texturesAgain.hits - textures.hits) + " draw-cache hits of " + std::to_string(draws));
}

// The epoch gate of ShaderResources::Revalidate: the proofs of a stream of template hits in one
// submission are skipped after its first, and a CPU write to a texture between two submissions (an
// ordering point) makes the first draw of the next one prove again. APS5_VERIFY_RESCACHE_EPOCH_GATE=1
// proves the skipped ones too and aborts on a disagreement; APS5_NO_RESCACHE_EPOCH_GATE=1 proves
// every draw.
void CheckEpochGate() {
    constexpr std::size_t draws = 64;
    const bool gate = std::getenv("APS5_NO_RESCACHE_EPOCH_GATE") == nullptr;
    for (const std::uint8_t value : {std::uint8_t{0x10}, std::uint8_t{0x20}}) {
        std::fill(Texels, Texels + TextureBytes, static_cast<std::byte>(value));
        const auto before = AgcDriver::Graphics::ShaderResources::GatedProofs();
        static_cast<void>(Submit(Stream(Kind::Textures, draws, 0)));
        const auto gated = AgcDriver::Graphics::ShaderResources::GatedProofs() - before;
        if (!gate) Require(gated == 0, "a stream of template hits skipped " + std::to_string(gated) + " proofs with the gate off");
        else Require(gated + 4u >= draws && gated < draws, "a stream of " + std::to_string(draws) + " template hits after a CPU write skipped " + std::to_string(gated) + " proofs");
    }
}

void Benchmark(Kind kind, double seconds) {
    constexpr std::size_t draws = 512;
    static_cast<void>(Submit(Stream(kind, draws, 0)));
    const auto start = std::chrono::steady_clock::now();
    std::uint64_t submitted = 0;
    for (std::uint32_t round = 1; std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() < seconds; ++round) {
        static_cast<void>(Submit(Stream(kind, draws, round)));
        submitted += draws;
    }
    // The draw cache's counters restart with each 10 s profile window: the [draw-cache] and
    // [spine] lines give the hits.
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    constexpr std::array<const char*, 6> names{"hits", "data", "moved", "rings", "buffers", "textures"};
    std::cout << "[resolve-bench] " << names[static_cast<std::size_t>(kind)] << ": " << submitted << " draws in " << elapsed << " s (" << elapsed * 1e6 / static_cast<double>(submitted) << " us/draw wall)\n";
}

Kind ParseKind(std::string_view name) {
    if (name == "data") return Kind::Data;
    if (name == "moved") return Kind::Moved;
    if (name == "rings") return Kind::Rings;
    if (name == "buffers") return Kind::Buffers;
    if (name == "textures") return Kind::Textures;
    return Kind::Hits;
}

}

int main(int argc, char** argv) {
    try {
        {
            const auto device = OpenVulkanTestDevice();
            if (!device) return VulkanTestSkipped;
        }
        TargetBlock target;
        const RingBlock ring;
        const TextureBlock textures;
        for (std::size_t slice = 0; slice < RingBytes / SliceBytes; ++slice) Ring[slice * SliceBytes / 4] = std::bit_cast<std::uint32_t>(static_cast<float>(slice % 64u) / 256.0f);
        for (std::size_t i = 0; i < TableCount; ++i) Tables[i] = {0.25f + static_cast<float>(i) / 128.0f, 0.0f, 0.0f, 0.0f};
        Register(VertexHeader, VertexCode.data(), sizeof(VertexCode), 2);
        Register(PixelHeader, PixelCode.data(), sizeof(PixelCode), 1);
        Register(LoadingPixelHeader, LoadingPixelCode.data(), sizeof(LoadingPixelCode), 1);
        Register(RingPixelHeader, RingPixelCode.data(), sizeof(RingPixelCode), 1);
        Register(BufferPixelHeader, BufferPixelCode.data(), sizeof(BufferPixelCode), 1);
        Register(TexturePixelHeader, TexturePixelCode.data(), sizeof(TexturePixelCode), 1);
        if (argc >= 3 && std::string_view(argv[1]) == "--benchmark") {
            Benchmark(ParseKind(argv[2]), argc >= 4 ? std::strtod(argv[3], nullptr) : 21.0);
        } else {
            CheckStreams();
            CheckEpochGate();
            std::cout << "draw streams of cache hits hit\n";
        }
        AgcDriverShutdown_nid_postfix();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        AgcDriverShutdown_nid_postfix();
        return 1;
    }
}
