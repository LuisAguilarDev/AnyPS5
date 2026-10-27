#pragma once

#include <cstddef>

namespace AgcDriver::DriverDetail {

// What a trailing presentation (APS5_FLIP_INFLIGHT >= 1) waits for before the presenter reports
// its flip to the guest. The title reads the flip's completion as "the GPU is done with the
// frame" and rewrites the frame's memory, which the frame's batches read in place; the blit reads
// only the frame's resident image or the presenter's own staging copy (taken on the CPU when the
// blit was recorded), never guest memory. So the guest is released once the frame's batches
// completed, and the blit's fence is awaited when its slot is next needed: RetirePresents before
// the next blit, or the slot's own reuse in VulkanDevice::present (route step 3).
class TrailingPresent {
public:
    virtual ~TrailingPresent() = default;
    // Waits WITHOUT GuestMemory::GpuMutex for the batches the newest blit was queued behind and
    // adds the time to `waitedMs`; false (nothing waited) when the device cannot wait for a batch
    // without the mutex (no timeline semaphore).
    virtual bool WaitFrameBatches(double& waitedMs) = 0;
    // VulkanDevice::RetirePresents: waits for the oldest blits until at most `keepInFlight` trail.
    virtual double RetirePresents(std::size_t keepInFlight) = 0;
};

// On by default; APS5_NO_EARLY_FLIP_RELEASE=1 waits for the frame's blit fence before the flip
// completes, as before.
bool EarlyFlipRelease();

// The waits between queueing a trailing presentation and reporting its flip; returns the time
// spent waiting. Early: the frame's batches, then every older blit (the flip clears the label of
// the buffer the previous blit read), leaving only this frame's blit in flight. Otherwise, or when
// the device cannot wait for the batches unlocked: every blit, this frame's included.
double FinishTrailingPresent(TrailingPresent& device, bool early);

}
