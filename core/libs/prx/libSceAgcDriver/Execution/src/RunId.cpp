#include "prx/libSceAgcDriver/Execution/include/FrameSteps.hpp"
#include "prx/libSceAgcDriver/TraceDescribe.hpp"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <string_view>
#ifndef _WIN32
extern char** environ;
#endif

namespace AgcDriver::FrameSteps {

namespace {

constexpr std::size_t MaxRunIdVariables = 256;
constexpr std::size_t RunIdBytes = 4096;
constexpr std::string_view RunIdPrefix = "APS5_";

}

// Every run, profiled or not: which libs ran and with which APS5_ switches. The build time is the
// compile time of this file, which the describe header (rewritten on every source change) rebuilds.
// Stack buffers only (see FormatRunId).
void PrintRunId() {
    std::array<std::string_view, MaxRunIdVariables> set;
    std::size_t count = 0;
    for (char** entry = environ; entry != nullptr && *entry != nullptr && count < set.size(); ++entry) {
        if (std::string_view(*entry).starts_with(RunIdPrefix)) set[count++] = *entry;
    }
    std::array<char, RunIdBytes> line;
    FormatRunId(line, APS5_TRACE_DESCRIBE, __DATE__ " " __TIME__, std::span(set.data(), count));
    std::fputs(line.data(), stderr);
    std::fflush(stderr);
}

}
