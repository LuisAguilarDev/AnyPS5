#ifndef CORE_SHADER_RECOMPILER_USERDATAUSE_HPP
#define CORE_SHADER_RECOMPILER_USERDATAUSE_HPP

#include <cstdint>
#include <utility>
#include <vector>

namespace ShaderRecompiler {

struct SrtUserDataUse {
    static constexpr std::uint32_t TrackedWords = 64;
    std::uint64_t asBase = 0;
    std::uint64_t asValue = 0;
    // The words whose only use is dword 0 of buffer descriptors (inline V# bases). Each is in
    // asValue too; bufferBases pairs it with the buffers it is the base of.
    std::uint64_t asBufferBase = 0;
    std::vector<std::pair<std::uint64_t, std::int32_t>> reads;
    // (descriptor source, user word) for each 4-dword source whose dword 0 is a bare user word,
    // read without evaluating the word, so asValue holds only the word's other uses until
    // the capture notes the buffer bases.
    std::vector<std::pair<std::uint32_t, std::uint32_t>> descriptorBases;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> bufferBases;
};

}

#endif
