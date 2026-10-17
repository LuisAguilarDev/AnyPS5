#include "prx/libSceVideoOut/include/Xv2Force1080p.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

#include "prx/libkernel/AppMetadata/include/AppMetadata.hpp"

#if defined(_WIN32)
#include <windows.h>
#endif

namespace Xv2Force1080p {
namespace {

struct Signature {
    std::uint64_t address;
    std::array<std::uint8_t, 10> bytes;
    std::size_t size;
    const char* what;
};

// Version 01.004.002 of the executable: the display set-up stores 3840x2160 after sceVideoOutGetOutputStatus
// and replaces it with the 1920x1080 constant only when the flag is nonzero.
constexpr Signature SIGNATURES[] = {
    {0x1eb01cb, {0xe8, 0x00, 0x57, 0x4d, 0x00}, 5, "sceVideoOutGetOutputStatus call"},
    {0x1eb01d4, {0x48, 0xb8, 0x00, 0x0f, 0x00, 0x00, 0x70, 0x08, 0x00, 0x00}, 10, "3840x2160 store"},
    {0x1eb01f0, {0x83, 0x3d, 0x89, 0xb3, 0x91, 0x01, 0x00}, 7, "flag test"},
    {0x1eb032b, {0x83, 0x3d, 0x4e, 0xb2, 0x91, 0x01, 0x00}, 7, "flag test"},
    {0x1eb0b1a, {0x83, 0x3d, 0x5f, 0xaa, 0x91, 0x01, 0x00}, 7, "flag test"},
    {0x2dbfd00, {0x80, 0x07, 0x00, 0x00, 0x38, 0x04, 0x00, 0x00}, 8, "1920x1080 constant"},
    {0x2dbe910, {0x00, 0x0f, 0x00, 0x00, 0x70, 0x08, 0x00, 0x00}, 8, "3840x2160 constant"},
};

constexpr const char* ENVIRONMENT_VARIABLE = "APS5_XV2_FORCE_1080P";
// A converted executable maps guest virtual address 0 one 64 KiB header page past its PE image base.
constexpr std::uintptr_t GUEST_IMAGE_OFFSET = 0x10000;

std::string Hex(std::uint64_t value) {
    char text[24];
    std::snprintf(text, sizeof(text), "0x%llx", static_cast<unsigned long long>(value));
    return text;
}

}

Result Apply(std::byte* image, std::string_view titleId) {
    if (titleId != TITLE_ID) return {false, "title " + std::string(titleId) + " is not " + std::string(TITLE_ID) + "; not patched"};
    for (const auto& signature : SIGNATURES) {
        if (std::memcmp(image + signature.address, signature.bytes.data(), signature.size) != 0) {
            return {false, std::string("executable differs at ") + Hex(signature.address) + " (" + signature.what + "); unknown version, not patched"};
        }
    }
    std::uint32_t flag = 0;
    std::memcpy(&flag, image + FLAG_ADDRESS, sizeof(flag));
    if (flag != 0) return {false, "flag at " + Hex(FLAG_ADDRESS) + " is already " + Hex(flag) + "; not patched"};
    constexpr std::uint32_t forced = 1;
    std::memcpy(image + FLAG_ADDRESS, &forced, sizeof(forced));
    return {true, "set flag at guest " + Hex(FLAG_ADDRESS) + " (host " + Hex(reinterpret_cast<std::uintptr_t>(image + FLAG_ADDRESS)) + ") 0x0 -> 0x1; the title selects a 1920x1080 display"};
}

void ApplyToRunningTitleOnce() {
    static std::once_flag once;
    std::call_once(once, [] {
        const char* value = std::getenv(ENVIRONMENT_VARIABLE);
        if (value == nullptr || std::strcmp(value, "1") != 0) return;
#if defined(_WIN32)
        auto* base = reinterpret_cast<std::byte*>(GetModuleHandleW(nullptr));
        const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
        const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(base + dos->e_lfanew);
        if (nt->OptionalHeader.SizeOfImage < GUEST_IMAGE_OFFSET + IMAGE_SPAN) {
            std::fprintf(stderr, "[videoout] %s: executable image of 0x%lx bytes is too small; not patched\n", ENVIRONMENT_VARIABLE, static_cast<unsigned long>(nt->OptionalHeader.SizeOfImage));
            return;
        }
        const auto result = Apply(base + GUEST_IMAGE_OFFSET, GetAppTitleId_nid_postfix().value);
        std::fprintf(stderr, "[videoout] %s: %s\n", ENVIRONMENT_VARIABLE, result.message.c_str());
#else
        std::fprintf(stderr, "[videoout] %s: only supported on Windows; not patched\n", ENVIRONMENT_VARIABLE);
#endif
    });
}

}
