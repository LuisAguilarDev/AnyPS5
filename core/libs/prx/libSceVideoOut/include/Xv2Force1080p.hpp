#ifndef CORE_LIBS_PRX_LIBSCEVIDEOOUT_INCLUDE_XV2FORCE1080P_HPP
#define CORE_LIBS_PRX_LIBSCEVIDEOOUT_INCLUDE_XV2FORCE1080P_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

// Experimental, local-only title patch (APS5_XV2_FORCE_1080P=1). Dragon Ball Xenoverse 2 (PPSA08587)
// ignores every display query and sizes its display at 3840x2160 unless a flag in its own data is set;
// nothing in the executable sets it. Setting it makes the title take its own 1920x1080 path.
namespace Xv2Force1080p {

inline constexpr std::string_view TITLE_ID = "PPSA08587";
inline constexpr std::uint64_t FLAG_ADDRESS = 0x37cb580;
// Highest guest address the patch reads or writes, exclusive.
inline constexpr std::uint64_t IMAGE_SPAN = FLAG_ADDRESS + sizeof(std::uint32_t);

struct Result {
    bool applied = false;
    std::string message;
};

// image points at guest virtual address 0 of the title's executable and holds at least IMAGE_SPAN bytes.
Result Apply(std::byte* image, std::string_view titleId);

// Applies the patch to the running title once, when APS5_XV2_FORCE_1080P=1, and logs the outcome.
void ApplyToRunningTitleOnce();

}

#endif
