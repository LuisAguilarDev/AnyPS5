#include "prx/libSceAgcDriver/Execution/include/FrameSteps.hpp"
#include "prx/libSceAgcDriver/TraceDescribe.hpp"
#include <cstdio>
#include <string>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
extern char** environ;
#endif

namespace AgcDriver::FrameSteps {

namespace {

std::vector<std::string> Environment() {
    std::vector<std::string> entries;
#ifdef _WIN32
    char* block = GetEnvironmentStringsA();
    if (block == nullptr) return entries;
    for (const char* entry = block; *entry != '\0'; entry += std::char_traits<char>::length(entry) + 1) entries.emplace_back(entry);
    FreeEnvironmentStringsA(block);
#else
    for (char** entry = environ; entry != nullptr && *entry != nullptr; ++entry) entries.emplace_back(*entry);
#endif
    return entries;
}

}

// Every run, profiled or not: which libs ran and with which APS5_ switches. The build time is the
// compile time of this file, which the describe header (rewritten on every source change) rebuilds.
void PrintRunId() {
    const auto line = FormatRunId(APS5_TRACE_DESCRIBE, __DATE__ " " __TIME__, Environment());
    std::fputs(line.c_str(), stderr);
    std::fflush(stderr);
}

}
