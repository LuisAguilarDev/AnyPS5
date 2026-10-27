#include "prx/libSceAgcDriver/Execution/include/Driver/PresentRelease.hpp"
#include <cstdlib>

namespace AgcDriver::DriverDetail {

bool EarlyFlipRelease() {
    static const bool enabled = std::getenv("APS5_NO_EARLY_FLIP_RELEASE") == nullptr;
    return enabled;
}

double FinishTrailingPresent(TrailingPresent& device, bool early) {
    double waitedMs = 0;
    if (early && device.WaitFrameBatches(waitedMs)) return waitedMs + device.RetirePresents(1);
    return waitedMs + device.RetirePresents(0);
}

}
