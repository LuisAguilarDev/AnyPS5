#include "prx/libSceAgcDriver/Execution/include/Driver/PresentRelease.hpp"
#include <cstddef>
#include <cstdint>
#include <deque>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace DriverDetail = AgcDriver::DriverDetail;

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

// A presenter device with VulkanDevice's two slots (APS5_FLIP_INFLIGHT=1): each blit's fence
// signals only when the device is made to wait for it, so a release that happens while a fence
// is still unsignaled is visible to the test.
class FakeDevice final : public DriverDetail::TrailingPresent {
public:
    static constexpr std::size_t Slots = 2;
    bool canWaitUnlocked = true;
    std::uint64_t batchesDone = 0;
    std::uint64_t batchesSubmitted = 0;
    // Blits in flight, oldest first, by frame; and every fence that signaled, in order.
    std::deque<int> inFlight;
    std::vector<int> signaled;
    std::vector<std::string> log;
    int slotsReusedInFlight = 0;

    // VulkanDevice::present: the frame's batches go first, then the blit into the cursor slot,
    // which waits for that slot's previous blit when it is still in flight.
    void Present(int frame) {
        batchesSubmitted = static_cast<std::uint64_t>(frame);
        if (inFlight.size() == Slots) RetirePresents(Slots - 1);
        if (inFlight.size() == Slots) ++slotsReusedInFlight;
        inFlight.push_back(frame);
        log.push_back("blit " + std::to_string(frame));
    }
    bool WaitFrameBatches(double& waitedMs) override {
        if (!canWaitUnlocked) return false;
        batchesDone = batchesSubmitted;
        waitedMs += 1.0;
        log.push_back("batches " + std::to_string(batchesDone));
        return true;
    }
    double RetirePresents(std::size_t keepInFlight) override {
        double waited = 0;
        while (inFlight.size() > keepInFlight) {
            signaled.push_back(inFlight.front());
            log.push_back("fence " + std::to_string(inFlight.front()));
            inFlight.pop_front();
            waited += 2.0;
        }
        return waited;
    }
    bool Signaled(int frame) const {
        for (const auto value : signaled) {
            if (value == frame) return true;
        }
        return false;
    }
};

// Driver::Present's trailing path for one frame: queue the blit, wait, then report the flip.
double PresentFrame(FakeDevice& device, int frame, bool early, bool& releasedBeforeFence) {
    device.Present(frame);
    const auto waited = DriverDetail::FinishTrailingPresent(device, early);
    releasedBeforeFence = !device.Signaled(frame);
    device.log.push_back("release " + std::to_string(frame));
    return waited;
}

void CheckEarlyRelease() {
    FakeDevice device;
    bool beforeFence = false;
    const auto waited = PresentFrame(device, 1, true, beforeFence);
    Require(beforeFence, "the guest was not released before the frame's blit fence signaled");
    Require(device.batchesDone == 1, "the guest was released before the frame's batches completed");
    Require(waited == 1.0, "the early release did not report the batch wait");
    Require(device.inFlight.size() == 1 && device.inFlight.front() == 1, "the frame's blit did not stay in flight past the release");
}

void CheckPreviousBlitRetired() {
    // The flip of frame 2 clears the label of frame 1's buffer: its blit must be done by then.
    FakeDevice device;
    bool beforeFence = false;
    PresentFrame(device, 1, true, beforeFence);
    PresentFrame(device, 2, true, beforeFence);
    Require(beforeFence, "frame 2 was not released before its own blit fence");
    Require(device.Signaled(1), "frame 2 was released while frame 1's blit was still in flight");
    const std::vector<std::string> expected{"blit 1", "batches 1", "release 1", "blit 2", "batches 2", "fence 1", "release 2"};
    Require(device.log == expected, "the early release did not wait for the batches, then the older blit, then release");
}

void CheckSlotNotReusedBeforeFence() {
    FakeDevice device;
    bool beforeFence = false;
    for (int frame = 1; frame <= 8; ++frame) PresentFrame(device, frame, true, beforeFence);
    Require(device.slotsReusedInFlight == 0, "a presentation slot was reused before its blit fence signaled");
    for (int frame = 1; frame <= 7; ++frame) Require(device.Signaled(frame), "an older blit fence was never awaited");
    Require(device.inFlight.size() == 1 && device.inFlight.front() == 8, "more than the newest blit trails the last release");
    for (std::size_t i = 1; i < device.signaled.size(); ++i) Require(device.signaled[i - 1] < device.signaled[i], "blit fences were awaited out of order");
}

void CheckOldPathWaitsForFence() {
    FakeDevice device;
    bool beforeFence = true;
    const auto waited = PresentFrame(device, 1, false, beforeFence);
    Require(!beforeFence, "APS5_NO_EARLY_FLIP_RELEASE released the guest before the blit fence");
    Require(device.batchesDone == 0, "the old path waited for the batches on their own");
    Require(waited == 2.0 && device.inFlight.empty(), "the old path did not retire every slot");
}

void CheckFallbackWithoutTimeline() {
    FakeDevice device;
    device.canWaitUnlocked = false;
    bool beforeFence = true;
    PresentFrame(device, 1, true, beforeFence);
    Require(!beforeFence, "a device that cannot wait for a batch unlocked released the guest before the blit fence");
    Require(device.inFlight.empty(), "the fallback did not retire every slot");
}

}

int main() {
    try {
        CheckEarlyRelease();
        CheckPreviousBlitRetired();
        CheckSlotNotReusedBeforeFence();
        CheckOldPathWaitsForFence();
        CheckFallbackWithoutTimeline();
    } catch (const std::exception& error) {
        std::cerr << "present release test failed: " << error.what() << "\n";
        return 1;
    }
    std::cout << "present release tests passed\n";
    return 0;
}
