#include "prx/libSceVideoOut/include/Xv2Force1080p.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <vector>

static void Require(bool value, const char* reason) {
    if (!value) {
        std::fprintf(stderr, "%s\n", reason);
        std::abort();
    }
}

static void Put(std::vector<std::byte>& image, std::uint64_t address, std::initializer_list<std::uint8_t> bytes) {
    std::size_t offset = 0;
    for (const auto byte : bytes) image[address + offset++] = static_cast<std::byte>(byte);
}

static std::vector<std::byte> Xv2Image() {
    std::vector<std::byte> image(Xv2Force1080p::IMAGE_SPAN);
    Put(image, 0x1eb01cb, {0xe8, 0x00, 0x57, 0x4d, 0x00});
    Put(image, 0x1eb01d4, {0x48, 0xb8, 0x00, 0x0f, 0x00, 0x00, 0x70, 0x08, 0x00, 0x00});
    Put(image, 0x1eb01f0, {0x83, 0x3d, 0x89, 0xb3, 0x91, 0x01, 0x00});
    Put(image, 0x1eb032b, {0x83, 0x3d, 0x4e, 0xb2, 0x91, 0x01, 0x00});
    Put(image, 0x1eb0b1a, {0x83, 0x3d, 0x5f, 0xaa, 0x91, 0x01, 0x00});
    Put(image, 0x2dbfd00, {0x80, 0x07, 0x00, 0x00, 0x38, 0x04, 0x00, 0x00});
    Put(image, 0x2dbe910, {0x00, 0x0f, 0x00, 0x00, 0x70, 0x08, 0x00, 0x00});
    return image;
}

static std::uint32_t Flag(const std::vector<std::byte>& image) {
    std::uint32_t flag = 0;
    std::memcpy(&flag, image.data() + Xv2Force1080p::FLAG_ADDRESS, sizeof(flag));
    return flag;
}

int main() {
    auto image = Xv2Image();
    Require(!Xv2Force1080p::Apply(image.data(), "PPSA01234").applied && Flag(image) == 0, "another title was patched");
    Require(Xv2Force1080p::Apply(image.data(), "PPSA08587").applied && Flag(image) == 1, "the known executable was not patched");
    Require(!Xv2Force1080p::Apply(image.data(), "PPSA08587").applied && Flag(image) == 1, "a set flag was patched again");
    for (const std::uint64_t changed : {0x1eb01cbull, 0x1eb01d6ull, 0x1eb0b1cull, 0x2dbfd00ull}) {
        auto other = Xv2Image();
        other[changed] ^= std::byte{0xff};
        Require(!Xv2Force1080p::Apply(other.data(), "PPSA08587").applied && Flag(other) == 0, "an unknown executable version was patched");
    }
    return 0;
}
