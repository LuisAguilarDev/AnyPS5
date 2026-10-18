#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Submit/include/Dcb.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "VulkanTestDevice.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <algorithm>
#include <array>
#include <bit>
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

alignas(256) constexpr std::array<std::uint32_t, 8> VertexCode{0x7e000208u, 0x7e020280u, 0x7e040280u, 0x7e0602f2u, 0xf80008cfu, 0x03020100u, 0xbf810000u, 0xbf810000u};
alignas(256) constexpr std::array<std::uint32_t, 11> PixelCode{0xf4080100u, 0xfa000000u, 0xbf8cc07fu, 0xe0300000u, 0x80010000u, 0xbf8c3f70u, 0x7e020280u, 0xf800180fu, 0x00010100u, 0xbf810000u, 0xbf810000u};
alignas(256) constexpr std::array<std::uint32_t, 16> TriangleVertexCode{
    0x7e000d05u, 0x080200f4u, 0x10020300u, 0x100202f4u, 0x060202f3u, 0x060400f3u, 0x10040500u, 0x060404f3u,
    0x7e060280u, 0x7e0802f2u, 0xf80008cfu, 0x04030201u, 0xbf810000u,
};
alignas(256) constexpr std::array<std::uint32_t, 10> ArithmeticPixelCode{0xf4000100u, 0xfa000000u, 0xbf8cc07fu, 0x7e000204u, 0x06000002u, 0x7e020280u, 0xf800180fu, 0x00010100u, 0xbf810000u, 0xbf810000u};
alignas(256) constexpr std::array<std::uint32_t, 8> InlinePixelCode{0xe0300000u, 0x80000000u, 0xbf8c3f70u, 0x7e020280u, 0xf800180fu, 0x00010100u, 0xbf810000u, 0xbf810000u};
alignas(256) constexpr std::array<std::uint32_t, 10> InlineAndScalarPixelCode{0xf4000200u, 0xfa000000u, 0xe0300000u, 0x80000000u, 0xbf8c0070u, 0x7e020208u, 0xf800180fu, 0x00010100u, 0xbf810000u, 0xbf810000u};
alignas(256) std::array<float, 4> PixelConstant{};
alignas(256) std::array<std::byte, 1024> ColorMemory{};

constexpr std::size_t BlockBytes = 65536;

struct Header {
    Shader shader{};
    ShaderUserData userData{};
};

alignas(8) Header VertexHeader{};
alignas(8) Header PixelHeader{};
alignas(8) Header ArithmeticPixelHeader{};
alignas(8) Header InlinePixelHeader{};
alignas(8) Header InlineAndScalarPixelHeader{};

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

std::uint32_t* GuestBlock() {
#ifdef _WIN32
    auto* block = static_cast<std::uint32_t*>(VirtualAlloc(nullptr, BlockBytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#else
    auto* block = static_cast<std::uint32_t*>(std::aligned_alloc(BlockBytes, BlockBytes));
#endif
    Require(block != nullptr, "data-only hit: cannot allocate a guest block");
    std::memset(block, 0, BlockBytes);
    GuestAllocations::Mutation().Add(block, BlockBytes, true, true, true);
    return block;
}

void Append(std::vector<std::uint32_t>& commands, std::uint32_t opcode, std::initializer_list<std::uint32_t> payload) {
    commands.push_back(0xc0000000u | (static_cast<std::uint32_t>(payload.size() - 1) << 16u) | (opcode << 8u));
    commands.insert(commands.end(), payload);
}

void PixelUserData(std::vector<std::uint32_t>& commands, const std::uint32_t* pointer, std::uint32_t unread) {
    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    Append(commands, 0x76, {0x00c, static_cast<std::uint32_t>(address)});
    Append(commands, 0x76, {0x00d, static_cast<std::uint32_t>(address >> 32u)});
    Append(commands, 0x76, {0x00e, unread});
}

struct Modes {
    bool verify = false, rebase = true, forwarded = true, bufferBases = true;
};

std::vector<std::uint32_t> Setup(bool triangle = false, bool loadPixel = false, const std::uint32_t* pixelCode = nullptr, std::uint32_t pixelUserWords = 3) {
    const auto color = reinterpret_cast<std::uintptr_t>(ColorMemory.data());
    const auto vertex = reinterpret_cast<std::uintptr_t>(triangle ? TriangleVertexCode.data() : VertexCode.data());
    const auto pixel = reinterpret_cast<std::uintptr_t>(pixelCode != nullptr ? pixelCode : triangle && !loadPixel ? ArithmeticPixelCode.data() : PixelCode.data());
    std::vector<std::pair<std::uint32_t, std::uint32_t>> context{
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
    };
    const std::vector<std::pair<std::uint32_t, std::uint32_t>> shader{
        {0xc8, static_cast<std::uint32_t>(vertex >> 8u)}, {0xc9, static_cast<std::uint32_t>(vertex >> 40u)}, {0x8b, triangle ? 0u : 1u << 1u}, {0x8c, 0},
        {0x008, static_cast<std::uint32_t>(pixel >> 8u)}, {0x009, static_cast<std::uint32_t>(pixel >> 40u)}, {0x00b, pixelUserWords << 1u},
    };
    if (triangle) context.insert(context.end(), {{0x2fe, 0}, {0x302, 0}, {0x306, 0}, {0x30a, 0}});
    std::vector<std::uint32_t> commands;
    Append(commands, 0x79, {0x242, 4});
    for (const auto& [offset, value] : context) Append(commands, 0x69, {offset, value});
    for (const auto& [offset, value] : shader) Append(commands, 0x76, {offset, value});
    return commands;
}

AgcDriver::DriverDetail::DrawEntryCounters Submit(std::vector<std::uint32_t> commands) {
    Packet packet{commands.data(), static_cast<std::uint32_t>(commands.size()), 0, {}};
    Require(sceAgcDriverSubmitDcb(&packet) == 0, "the draw submission failed");
    AgcDriverWaitIdle_nid_postfix();
    return AgcDriver::DriverDetail::Driver::Get().DrawCacheCounters();
}

std::string Counts(const AgcDriver::DriverDetail::DrawEntryCounters& counters) {
    return std::to_string(counters.lookups) + " lookups, " + std::to_string(counters.dataHits) + " data-only hits, stages " + std::to_string(counters.dataStagesReused) + " reused / " + std::to_string(counters.dataStagesCompiled) + " compiled / " + std::to_string(counters.dataVerified) + " verified";
}

void Describe(std::uint32_t* table, std::uint32_t* data, float value) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    data[0] = std::bit_cast<std::uint32_t>(value);
    table[0] = static_cast<std::uint32_t>(address);
    table[1] = static_cast<std::uint32_t>(address >> 32u) & 0xffffu;
    table[2] = 16;
    table[3] = 0x31016facu;
}

std::string CandidateCounts(const AgcDriver::DriverDetail::DrawCandidateCounters& counters) {
    return std::to_string(counters.tried) + " candidates tried, equal " + std::to_string(counters.equal) + " / " + std::to_string(counters.equalData) + ", rebased " + std::to_string(counters.rebased) + " / " + std::to_string(counters.rebasedData) + ", forwarded " + std::to_string(counters.forwarded) + ", buffer bases " + std::to_string(counters.bufferBases) + ", refused " + std::to_string(counters.refusedUntagged) + " untagged / " + std::to_string(counters.refusedOff) + " off, differing " + std::to_string(counters.differingEqual) + " / " + std::to_string(counters.differingRebased);
}

void UnreadUserWordsKeepTheStage(const Modes& modes) {
    const bool verify = modes.verify;
    Registered(VertexHeader, VertexCode.data(), sizeof(VertexCode), 2);
    Registered(PixelHeader, PixelCode.data(), sizeof(PixelCode), 1);
    auto* first = GuestBlock();
    auto* second = first + 2048;
    Describe(first, first + 1024, 1.0f);
    std::copy_n(first, 4, second);

    auto commands = Setup();
    PixelUserData(commands, first, 0x11);
    Append(commands, 0x2d, {3, 2});
    auto counters = Submit(std::move(commands));
    Require(counters.lookups == 1 && counters.dataHits == 0, "the first draw: " + Counts(counters));
    commands.clear();
    PixelUserData(commands, first, 0x11);
    Append(commands, 0x2d, {3, 2});
    counters = Submit(std::move(commands));
    Require(counters.inserts != 0, "the repeated register key was not admitted: " + Counts(counters));

    commands.clear();
    PixelUserData(commands, first, 0x22);
    Append(commands, 0x2d, {3, 2});
    const auto before = counters;
    counters = Submit(std::move(commands));
    const auto reused = counters.dataStagesReused - before.dataStagesReused;
    const auto compiled = counters.dataStagesCompiled - before.dataStagesCompiled;
    Require(counters.dataHits == before.dataHits + 1, "a draw that changed only a user word no stage reads was not a data-only hit: " + Counts(counters));
    if (verify) Require(compiled == 2 && counters.dataVerified - before.dataVerified == 2, "verify mode: a draw that changed only an unread user word did not build and compare both stages: " + Counts(counters));
    else Require(reused == 2 && compiled == 0, "a draw that changed only a user word no stage reads did not reuse both stages: " + Counts(counters));

    commands.clear();
    PixelUserData(commands, second, 0x22);
    Append(commands, 0x2d, {3, 2});
    const auto moved = counters;
    counters = Submit(std::move(commands));
    Require(counters.dataHits == moved.dataHits + 1, "a draw that pointed the pixel stage at another buffer descriptor was not a data-only hit: " + Counts(counters));
    const auto movedReused = counters.dataStagesReused - moved.dataStagesReused;
    const auto movedCompiled = counters.dataStagesCompiled - moved.dataStagesCompiled;
    if (verify) {
        Require(movedCompiled == 2, "verify mode: a draw that moved the pixel stage's table did not build both stages: " + Counts(counters));
    } else if (modes.rebase) {
        Require(movedReused == 2 && movedCompiled == 0, "a draw that moved only the pixel stage's table pointer did not reuse both stages: " + Counts(counters) + "; " + CandidateCounts(counters.candidates));
        Require(counters.candidates.rebased - moved.candidates.rebased == 1, "the moved table was not reused by a rebased candidate: " + CandidateCounts(counters.candidates));
    } else {
        Require(movedCompiled >= 1, "rebasing off: a draw that moved the pixel stage's table reused the pixel stage: " + Counts(counters));
        Require(movedReused == 1, "a draw that changed only the pixel stage's descriptor pointer did not reuse the vertex stage: " + Counts(counters));
    }
}

unsigned BrightestColorByte() {
    AgcDriver::GuestMemory::FlushGpuWrites(reinterpret_cast<std::uintptr_t>(ColorMemory.data()), ColorMemory.size());
    unsigned brightest = 0;
    for (const auto value : ColorMemory) brightest = std::max(brightest, static_cast<unsigned>(std::to_integer<std::uint8_t>(value)));
    return brightest;
}

void ReadUserWordsChangeTheOutput(const Modes& modes) {
    Registered(VertexHeader, TriangleVertexCode.data(), sizeof(TriangleVertexCode), 2);
    Registered(ArithmeticPixelHeader, ArithmeticPixelCode.data(), sizeof(ArithmeticPixelCode), 1);
    PixelConstant[0] = 0.25f;
    auto* table = reinterpret_cast<std::uint32_t*>(PixelConstant.data());
    const auto draw = [&](float word, bool setup) {
        auto commands = setup ? Setup(true) : std::vector<std::uint32_t>{};
        PixelUserData(commands, table, std::bit_cast<std::uint32_t>(word));
        Append(commands, 0x2d, {3, 2});
        Submit(std::move(commands));
        return BrightestColorByte();
    };
    draw(0.25f, true);
    const auto first = draw(0.25f, false);
    const auto before = AgcDriver::DriverDetail::Driver::Get().DrawCacheCounters();
    const auto second = draw(0.5f, false);
    const auto after = AgcDriver::DriverDetail::Driver::Get().DrawCacheCounters();
    const auto withinTwo = [](unsigned value, unsigned expected) { return value + 2 >= expected && value <= expected + 2; };
    Require(withinTwo(first, 128), "a pixel stage adding user word 2 (0.25) to a loaded 0.25 wrote " + std::to_string(first) + ", expected 128");
    Require(withinTwo(second, 191), "a pixel stage adding user word 2 changed to 0.5 wrote " + std::to_string(second) + ", expected 191 (" + Counts(after) + ", before: " + Counts(before) + ")");
    Require(after.dataHits == before.dataHits + 1, "changing a forwarded user word was not a data-only hit: " + Counts(after));
    const auto forwarded = after.candidates.forwarded - before.candidates.forwarded;
    if (!modes.rebase || !modes.forwarded) {
        Require(forwarded == 0, "forwarded words off: the pixel stage was reused with its forwarded user word patched: " + CandidateCounts(after.candidates));
    } else {
        Require(forwarded == 1, "the pixel stage whose forwarded user word changed was not reused with the word patched: " + CandidateCounts(after.candidates));
        if (modes.verify) Require(after.dataVerified - before.dataVerified == 2, "verify mode: the patched draw was not built and compared on both stages: " + Counts(after));
        else Require(after.dataDrawsReused - before.dataDrawsReused == 1, "the draw whose forwarded user word changed did not reuse both stages: " + Counts(after));
    }
}

void MovedTablesRenderTheirBuffers(const Modes& modes) {
    Registered(VertexHeader, TriangleVertexCode.data(), sizeof(TriangleVertexCode), 2);
    Registered(PixelHeader, PixelCode.data(), sizeof(PixelCode), 1);
    auto* block = GuestBlock();
    auto* tableA = block;
    auto* tableB = block + 256;
    auto* tableC = block + 512;
    auto* dataA = block + 4096;
    auto* dataC = block + 8192;
    Describe(tableA, dataA, 0.25f);
    std::copy_n(tableA, 4, tableB);
    Describe(tableC, dataC, 0.5f);
    const auto draw = [&](const std::uint32_t* table, bool setup) {
        auto commands = setup ? Setup(true, true) : std::vector<std::uint32_t>{};
        PixelUserData(commands, table, 0x33);
        Append(commands, 0x2d, {3, 2});
        Submit(std::move(commands));
        return BrightestColorByte();
    };
    const auto withinTwo = [](unsigned value, unsigned expected) { return value + 2 >= expected && value <= expected + 2; };
    const auto counters = [] { return AgcDriver::DriverDetail::Driver::Get().DrawCacheCounters(); };
    draw(tableA, true);
    const auto a = draw(tableA, false);
    Require(withinTwo(a, 64), "the pixel stage loading 0.25 through table A wrote " + std::to_string(a) + ", expected 64");

    const auto beforeB = counters();
    const auto b = draw(tableB, false);
    const auto afterB = counters();
    Require(withinTwo(b, 64), "the pixel stage loading 0.25 through table B wrote " + std::to_string(b) + ", expected 64 (" + CandidateCounts(afterB.candidates) + ")");
    Require(afterB.dataHits == beforeB.dataHits + 1, "moving the table pointer was not a data-only hit: " + Counts(afterB));
    if (!modes.rebase) {
        Require(afterB.candidates.rebased == beforeB.candidates.rebased && afterB.candidates.refusedOff > beforeB.candidates.refusedOff, "rebasing off: the moved table was rebased: " + CandidateCounts(afterB.candidates));
    } else {
        Require(afterB.candidates.rebased - beforeB.candidates.rebased == 1, "the copied table was not reused by a rebased candidate: " + CandidateCounts(afterB.candidates));
        if (modes.verify) Require(afterB.dataVerified - beforeB.dataVerified == 2, "verify mode: the rebased draw was not built and compared on both stages: " + Counts(afterB));
        else Require(afterB.dataDrawsReused - beforeB.dataDrawsReused == 1, "the rebased draw did not reuse both stages: " + Counts(afterB));
    }

    const auto c = draw(tableC, false);
    Require(withinTwo(c, 128), "the pixel stage loading 0.5 through table C wrote " + std::to_string(c) + ", expected 128 (" + CandidateCounts(counters().candidates) + ")");

    dataA[0] = std::bit_cast<std::uint32_t>(0.75f);
    const auto again = draw(tableB, false);
    Require(withinTwo(again, 191), "the pixel stage loading 0.75 through table B wrote " + std::to_string(again) + ", expected 191 (" + CandidateCounts(counters().candidates) + ")");
}

}

void InlineDescriptor(std::vector<std::uint32_t>& commands, const std::uint32_t* data, std::uint32_t records) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    const std::array<std::uint32_t, 4> descriptor{static_cast<std::uint32_t>(address), static_cast<std::uint32_t>(address >> 32u) & 0xffffu, records, 0x31016facu};
    for (std::uint32_t word = 0; word < descriptor.size(); ++word) Append(commands, 0x76, {0x00c + word, descriptor[word]});
}

void InlineBufferBasesRenderTheirBuffers(const Modes& modes) {
    using AgcDriver::DriverDetail::DrawRefusedWord;
    Registered(VertexHeader, TriangleVertexCode.data(), sizeof(TriangleVertexCode), 2);
    Registered(InlinePixelHeader, InlinePixelCode.data(), sizeof(InlinePixelCode), 1);
    auto* block = GuestBlock();
    auto* dataA = block;
    auto* dataB = block + 1024;
    auto* dataC = block + 2048;
    dataA[0] = std::bit_cast<std::uint32_t>(0.25f);
    dataB[0] = std::bit_cast<std::uint32_t>(0.25f);
    dataC[0] = std::bit_cast<std::uint32_t>(0.5f);
    const auto draw = [&](const std::uint32_t* data, std::uint32_t records, bool setup) {
        auto commands = setup ? Setup(true, true, InlinePixelCode.data(), 4) : std::vector<std::uint32_t>{};
        InlineDescriptor(commands, data, records);
        Append(commands, 0x2d, {3, 2});
        Submit(std::move(commands));
        return BrightestColorByte();
    };
    const auto withinTwo = [](unsigned value, unsigned expected) { return value + 2 >= expected && value <= expected + 2; };
    const auto counters = [] { return AgcDriver::DriverDetail::Driver::Get().DrawCacheCounters(); };
    const auto refusedAsBase = [](const AgcDriver::DriverDetail::DrawEntryCounters& value) { return value.candidates.refusedBy[static_cast<std::size_t>(DrawRefusedWord::BufferBase)]; };
    draw(dataA, 16, true);
    const auto a = draw(dataA, 16, false);
    Require(withinTwo(a, 64), "the pixel stage loading 0.25 through an inline V# at A wrote " + std::to_string(a) + ", expected 64");

    const auto beforeB = counters();
    const auto b = draw(dataB, 16, false);
    const auto afterB = counters();
    Require(withinTwo(b, 64), "the pixel stage loading 0.25 through an inline V# at B wrote " + std::to_string(b) + ", expected 64 (" + CandidateCounts(afterB.candidates) + ")");
    Require(afterB.dataHits == beforeB.dataHits + 1, "moving the inline V# base was not a data-only hit: " + Counts(afterB));
    if (!modes.rebase) {
        Require(afterB.candidates.bufferBases == beforeB.candidates.bufferBases, "rebasing off: the moved inline V# base was patched: " + CandidateCounts(afterB.candidates));
    } else if (!modes.bufferBases) {
        Require(afterB.candidates.bufferBases == beforeB.candidates.bufferBases && refusedAsBase(afterB) > refusedAsBase(beforeB), "buffer base words off: the moved inline V# base was not refused as a buffer base: " + CandidateCounts(afterB.candidates));
    } else {
        Require(afterB.candidates.bufferBases - beforeB.candidates.bufferBases == 1, "the pixel stage whose inline V# base moved was not reused with the base patched: " + CandidateCounts(afterB.candidates));
        if (modes.verify) Require(afterB.dataVerified - beforeB.dataVerified == 2, "verify mode: the patched draw was not built and compared on both stages: " + Counts(afterB));
        else Require(afterB.dataDrawsReused - beforeB.dataDrawsReused == 1, "the draw whose inline V# base moved did not reuse both stages: " + Counts(afterB));
    }

    const auto beforeC = counters();
    const auto c = draw(dataC, 16, false);
    const auto afterC = counters();
    Require(withinTwo(c, 128), "the pixel stage loading 0.5 through an inline V# at C wrote " + std::to_string(c) + ", expected 128 (" + CandidateCounts(afterC.candidates) + ")");
    if (modes.rebase && modes.bufferBases) Require(afterC.candidates.bufferBases - beforeC.candidates.bufferBases == 1, "the inline V# moved to C was not patched: " + CandidateCounts(afterC.candidates));

    const auto beforeRecords = counters();
    const auto d = draw(dataC, 32, false);
    const auto afterRecords = counters();
    Require(withinTwo(d, 128), "the pixel stage loading 0.5 through an inline V# at C with 32 records wrote " + std::to_string(d) + ", expected 128");
    Require(afterRecords.candidates.bufferBases == beforeRecords.candidates.bufferBases, "an inline V# whose record count changed was patched: " + CandidateCounts(afterRecords.candidates));

    dataC[0] = std::bit_cast<std::uint32_t>(0.75f);
    const auto again = draw(dataB, 16, false);
    Require(withinTwo(again, 64), "the pixel stage loading 0.25 through an inline V# back at B wrote " + std::to_string(again) + ", expected 64");
    const auto last = draw(dataC, 16, false);
    Require(withinTwo(last, 191), "the pixel stage loading 0.75 through an inline V# back at C wrote " + std::to_string(last) + ", expected 191 (" + CandidateCounts(counters().candidates) + ")");
}

void InlineBasesTheWalkReadsStayExact(const Modes& modes) {
    Registered(VertexHeader, TriangleVertexCode.data(), sizeof(TriangleVertexCode), 2);
    Registered(InlineAndScalarPixelHeader, InlineAndScalarPixelCode.data(), sizeof(InlineAndScalarPixelCode), 1);
    auto* block = GuestBlock();
    auto* dataA = block;
    auto* dataB = block + 1024;
    dataA[0] = std::bit_cast<std::uint32_t>(0.25f);
    dataB[0] = std::bit_cast<std::uint32_t>(0.5f);
    const auto draw = [&](const std::uint32_t* data, bool setup) {
        auto commands = setup ? Setup(true, true, InlineAndScalarPixelCode.data(), 4) : std::vector<std::uint32_t>{};
        InlineDescriptor(commands, data, 16);
        Append(commands, 0x2d, {3, 2});
        Submit(std::move(commands));
        return BrightestColorByte();
    };
    const auto withinTwo = [](unsigned value, unsigned expected) { return value + 2 >= expected && value <= expected + 2; };
    draw(dataA, true);
    const auto a = draw(dataA, false);
    Require(withinTwo(a, 64), "the pixel stage loading 0.25 through an inline V# it also reads as a pointer wrote " + std::to_string(a) + ", expected 64");
    const auto before = AgcDriver::DriverDetail::Driver::Get().DrawCacheCounters();
    const auto b = draw(dataB, false);
    const auto after = AgcDriver::DriverDetail::Driver::Get().DrawCacheCounters();
    Require(withinTwo(b, 128), "the pixel stage loading 0.5 through a moved inline V# it also reads as a pointer wrote " + std::to_string(b) + ", expected 128 (" + CandidateCounts(after.candidates) + ")");
    Require(after.candidates.bufferBases == before.candidates.bufferBases, "an inline V# base the walk also reads through was patched: " + CandidateCounts(after.candidates));
    const auto both = static_cast<std::size_t>(AgcDriver::DriverDetail::DrawRefusedWord::BaseAndValue);
    if (modes.rebase) Require(after.candidates.refusedBy[both] > before.candidates.refusedBy[both], "an inline V# base the walk also reads through was not refused as a base and a value: " + CandidateCounts(after.candidates));
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
        Modes modes;
        modes.verify = std::getenv("APS5_VERIFY_DRAW_DATA_HITS") != nullptr || std::getenv("APS5_VERIFY_DRAW_REBASE") != nullptr;
        modes.rebase = std::getenv("APS5_NO_DRAW_REBASE") == nullptr;
        modes.forwarded = std::getenv("APS5_NO_DRAW_FORWARDED_WORDS") == nullptr;
        modes.bufferBases = std::getenv("APS5_NO_DRAW_BUFFER_BASE_WORDS") == nullptr;
        UnreadUserWordsKeepTheStage(modes);
        ReadUserWordsChangeTheOutput(modes);
        MovedTablesRenderTheirBuffers(modes);
        InlineBufferBasesRenderTheirBuffers(modes);
        InlineBasesTheWalkReadsStayExact(modes);
        AgcDriverShutdown_nid_postfix();
        std::cout << "Data-only hits " << (modes.verify ? "verified" : "reused") << " the stages whose read user words did not change" << (modes.rebase ? " or whose tables only moved" : "") << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        AgcDriverShutdown_nid_postfix();
        return 1;
    }
}
