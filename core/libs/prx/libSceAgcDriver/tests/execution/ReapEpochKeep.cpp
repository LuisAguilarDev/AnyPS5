#include "VulkanTestDevice.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Submit/include/Dcb.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/Shutdown.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using AgcDriver::DriverDetail::Driver;

alignas(256) std::array<std::uint32_t, 15> SpinCode{0xf4080200u, 0xfa000000u, 0xbe840381u, 0x9304ff04u, 0x0019660du, 0x8004ff04u, 0x3c6ef35fu, 0x80828102u, 0xbf078002u, 0xbf85fff9u, 0xbf8cc07fu, 0x7e000204u, 0xe0700000u, 0x80020000u, 0xbf810000u};

constexpr std::uint32_t SpinIterations = 1u << 20u;
constexpr std::size_t BlockBytes = 65536;
constexpr int Rounds = 8;

struct Slot {
    alignas(256) std::array<std::uint32_t, 4> spinOutput{};
    alignas(256) std::array<std::uint32_t, 4> spinDescriptor{};
    alignas(256) volatile std::uint32_t done = 0;
};

void check(bool condition, const std::string& what) {
    if (!condition) throw std::runtime_error("reap epoch keep: " + what);
}

class GuestBlock {
public:
    GuestBlock() {
#ifdef _WIN32
        block = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, BlockBytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#else
        block = static_cast<std::uint8_t*>(std::aligned_alloc(BlockBytes, BlockBytes));
#endif
        check(block != nullptr, "cannot allocate the guest block");
        std::memset(block, 0, BlockBytes);
        GuestAllocations::Mutation().Add(block, BlockBytes, true, true, true);
    }

    ~GuestBlock() {
        GuestAllocations::Mutation().Remove(block);
#ifdef _WIN32
        VirtualFree(block, 0, MEM_RELEASE);
#else
        std::free(block);
#endif
    }

    GuestBlock(const GuestBlock&) = delete;
    GuestBlock& operator=(const GuestBlock&) = delete;

    Slot& Data() { return *reinterpret_cast<Slot*>(block); }

private:
    std::uint8_t* block = nullptr;
};

std::uint32_t Low(const volatile void* pointer) {
    return static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(pointer));
}

std::uint32_t High(const volatile void* pointer) {
    return static_cast<std::uint32_t>(static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(pointer)) >> 32u);
}

void SetShaderRegister(std::vector<std::uint32_t>& words, std::uint32_t offset, std::uint32_t value) {
    words.insert(words.end(), {0xc0017600u, offset, value});
}

void Dispatch(std::vector<std::uint32_t>& words, const void* code, std::initializer_list<std::uint32_t> userData) {
    const auto address = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(code));
    SetShaderRegister(words, 0x20c, static_cast<std::uint32_t>(address >> 8u));
    SetShaderRegister(words, 0x20d, static_cast<std::uint32_t>(address >> 40u) & 0xffu);
    for (const std::uint32_t offset : {0x207u, 0x208u, 0x209u}) SetShaderRegister(words, offset, 1);
    SetShaderRegister(words, 0x212, 0);
    SetShaderRegister(words, 0x213, static_cast<std::uint32_t>(userData.size()) << 1u);
    std::uint32_t offset = 0x240;
    for (const auto value : userData) SetShaderRegister(words, offset++, value);
    words.insert(words.end(), {0xc0031500u, 1, 1, 1, 0x8041u});
}

void Run(Slot& slot) {
    slot.spinDescriptor = {Low(slot.spinOutput.data()), High(slot.spinOutput.data()) & 0xffffu, sizeof(slot.spinOutput), 0x31016facu};
    slot.done = 0;
    std::vector<std::uint32_t> words;
    Dispatch(words, SpinCode.data(), {Low(slot.spinDescriptor.data()), High(slot.spinDescriptor.data()), SpinIterations});
    words.insert(words.end(), {0xc0064900u, 0x514u, (1u << 29u) | (2u << 24u), Low(&slot.done), High(&slot.done), 1u, 0u, 0u});
    Packet packet{words.data(), static_cast<std::uint32_t>(words.size()), 0, {}};
    check(sceAgcDriverSubmitDcb(&packet) == 0, "submission failed");
    const auto start = std::chrono::steady_clock::now();
    while (slot.done == 0) {
        check(std::chrono::steady_clock::now() - start < std::chrono::seconds(10), "the submission did not finish");
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    AgcDriverWaitIdle_nid_postfix();
}

}

int main() {
#ifdef _WIN32
    _putenv_s("APS5_PIPELINED_DRAWS", "1");
#else
    setenv("APS5_PIPELINED_DRAWS", "1", 1);
#endif
    try {
        auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        device.reset();
        GuestBlock block;
        const auto before = Driver::ReapEpochs();
        for (int round = 0; round < Rounds; ++round) Run(block.Data());
        const auto after = Driver::ReapEpochs();
        const auto reaps = after.reaps - before.reaps;
        const auto advances = after.committerAdvances - before.committerAdvances;
        std::cout << reaps << " reaps over " << Rounds << " submissions, " << advances << " of them advanced the committer's epoch\n";
        check(reaps != 0, "no completion reap ran, so the test proves nothing");
        if (std::getenv("APS5_NO_REAP_EPOCH_KEEP") != nullptr) check(advances != 0, "APS5_NO_REAP_EPOCH_KEEP: no reap advanced the committer's epoch");
        else check(advances == 0, "a completion reap advanced the committer's collect epoch");
        LibcRunShutdown_nid_postfix();
        std::puts("reap epoch keep tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        try { LibcRunShutdown_nid_postfix(); } catch (...) {}
        return 1;
    }
}
