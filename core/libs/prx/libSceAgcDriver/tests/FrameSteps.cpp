#include "prx/libSceAgcDriver/Execution/include/FrameSteps.hpp"
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace FrameSteps = AgcDriver::FrameSteps;

constexpr std::int64_t Ms = 1'000'000;

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

bool Near(double value, double expected) {
    return std::fabs(value - expected) < 1e-9;
}

void Contains(const std::string& line, const std::string& field) {
    Require(line.find(field) != std::string::npos, "missing '" + field + "' in: " + line);
}

void CheckGuestFrames() {
    FrameSteps::GuestFrames guest;
    FrameSteps::ClosedFrame closed;
    Require(!guest.Submit(0, 1 * Ms, false, closed), "a submit without a flip closed a frame");
    Require(!guest.Submit(5 * Ms, 6 * Ms, false, closed), "a second submit without a flip closed a frame");
    Require(guest.Submit(10 * Ms, 12 * Ms, true, closed), "the flip submit did not close the frame");
    Require(Near(closed.spanMs, 12.0), "the first frame does not span its first entry to the flip exit");
    Require(closed.gapMs < 0, "the first frame has a gap with no flip before it");
    Require(Near(closed.insideMs, 4.0), "time inside the calls is not the sum of the three calls");
    Require(closed.submits == 3, "the first frame does not count its three submits");
    Require(guest.Submit(20 * Ms, 21 * Ms, true, closed), "a frame of one flip submit did not close");
    Require(Near(closed.gapMs, 8.0), "the gap is not the previous flip exit to this frame's first entry");
    Require(Near(closed.spanMs, 1.0), "a frame of one submit does not span that submit");
    Require(closed.submits == 1, "the second frame kept the first frame's submits");
}

void CheckSampleWindow() {
    FrameSteps::SampleWindow window;
    for (int i = 1; i <= 100; ++i) window.Add(static_cast<double>(i));
    const auto spread = window.Take();
    Require(spread.count == 100, "the window did not count every sample");
    Require(Near(spread.mean, 50.5), "the mean of 1..100 is not 50.5");
    Require(Near(spread.p50, 51.0), "the p50 of 1..100 is not the 51st value");
    Require(Near(spread.p95, 96.0), "the p95 of 1..100 is not the 96th value");
    const auto empty = window.Take();
    Require(empty.count == 0 && empty.mean == 0 && empty.p50 == 0 && empty.p95 == 0, "taking the window did not reset it");
    for (std::size_t i = 0; i < FrameSteps::SampleCapacity + 10; ++i) window.Add(1.0);
    const auto full = window.Take();
    Require(full.count == FrameSteps::SampleCapacity + 10 && Near(full.mean, 1.0) && Near(full.p95, 1.0), "samples past the capacity were not counted in the mean");
}

void CheckCountersReset() {
    FrameSteps::Counters counters;
    FrameSteps::Counters::Add(counters.submits, 30);
    FrameSteps::Counters::AddNs(counters.submitNs, 6 * Ms);
    FrameSteps::Counters::AddNs(counters.packetIdleNs, -5);
    FrameSteps::Counters::AddNs(counters.recorderBusyNs[FrameSteps::Counters::Slot(99)], 2 * Ms);
    const auto first = counters.Take();
    Require(first.submits == 30 && first.submitNs == static_cast<std::uint64_t>(6 * Ms), "the window lost its submits");
    Require(first.packetIdleNs == 0, "a negative duration was added");
    Require(first.recorderBusyNs[FrameSteps::LaneThreads - 1] == static_cast<std::uint64_t>(2 * Ms), "a thread past the lane count was not folded into the last slot");
    const auto second = counters.Take();
    Require(second.submits == 0 && second.submitNs == 0 && second.recorderBusyNs[FrameSteps::LaneThreads - 1] == 0, "taking the counters did not reset them");
}

void CheckExecutedCount() {
    FrameSteps::Counters& counters = FrameSteps::Global();
    counters.Take();
    const std::vector<std::uint32_t> commands{0xc0012700u, 1u, 2u, 0x80000000u, 0xc0002d00u, 3u, 0xc0001000u, 4u};
    FrameSteps::NoteExecuted(commands, true);
    const auto totals = counters.Take();
    Require(totals.dwords == commands.size(), "the executed dwords are not the command size");
    Require(totals.packets == 3, "a filler was counted as a packet, or a packet was missed");
    Require(totals.draws == 2, "DRAW_INDEX_2 and DRAW_INDEX_AUTO were not both counted as draws");
    Require(totals.packetPackets == 3, "queue 0 packets were not counted for the packet thread");
}

void CheckPresentPairing() {
    FrameSteps::Frames frames;
    frames.Submit(0, 1 * Ms, false);
    frames.Submit(2 * Ms, 4 * Ms, true);
    frames.Submit(10 * Ms, 11 * Ms, false);
    frames.Submit(12 * Ms, 14 * Ms, true);
    Require(!frames.Present(20 * Ms), "the window was due after one present");
    Require(!frames.Present(36 * Ms), "the window was due after 16 ms");
    Require(!frames.Present(40 * Ms), "an unpaired present was due");
    const auto window = frames.Take(50 * Ms);
    Require(window.frames == 2, "two flips are not two frames");
    Require(window.presents == 3, "the presents were not counted");
    Require(window.flipToPresentMs.count == 2 && Near(window.flipToPresentMs.mean, (16.0 + 22.0) / 2.0), "presents were not paired with flips in order");
    Require(window.intervalMs.count == 2 && Near(window.intervalMs.mean, 10.0), "present intervals are not the gaps between presents");
    Require(window.gapMs.count == 1 && Near(window.gapMs.mean, 6.0), "the gap is not the first flip exit to the next frame's first entry");
    Require(Near(window.spanMs.mean, 4.0), "the spans are not first entry to flip exit");
    Require(Near(window.seconds, 0.03), "the window does not start at its first present");
    const auto next = frames.Take(60 * Ms);
    Require(next.frames == 0 && next.presents == 0 && next.flipToPresentMs.count == 0, "taking the frames did not reset them");
    Require(Near(next.seconds, 0.01), "the next window does not start where the last one was taken");
    Require(frames.Present(60 * Ms + FrameSteps::ReportWindowNs), "the window was not due after its length");
}

void CheckFormatting() {
    FrameSteps::Totals totals;
    totals.submits = 40;
    totals.submitNs = static_cast<std::uint64_t>(8 * Ms);
    totals.dwords = 400100;
    totals.packets = 20000;
    totals.draws = 3000;
    totals.packetBusyNs = static_cast<std::uint64_t>(30 * Ms);
    totals.packetPackets = 20000;
    totals.prepBusyNs[0] = static_cast<std::uint64_t>(6 * Ms);
    totals.prepBusyNs[1] = static_cast<std::uint64_t>(2 * Ms);
    totals.recorderBusyNs[2] = static_cast<std::uint64_t>(4 * Ms);
    totals.recorderChunks = 10;
    totals.recorderDraws = 250;
    totals.vkSubmits = 8;
    totals.vkSubmitNs = static_cast<std::uint64_t>(1 * Ms);
    FrameSteps::Window window;
    window.seconds = 10.0;
    window.frames = 2;
    window.presents = 2;
    window.gpuBusyMs = 20.0;
    window.spanMs.count = 2;
    window.spanMs.mean = 8.0;
    window.gapMs.count = 1;
    window.gapMs.mean = 2.0;
    const auto guest = FrameSteps::FormatGuest(totals, window);
    Contains(guest, "[frame-steps-guest] ");
    Contains(guest, "frames=2 ");
    Contains(guest, "submits/f=20.0 ");
    Contains(guest, "submit_us/call=200.0 ");
    Contains(guest, "submit_ms/f=4.00 ");
    Contains(guest, "dwords/f=200050 ");
    Contains(guest, "kb/f=781.4 ");
    Contains(guest, "packets/f=10000 ");
    Contains(guest, "draws/f=1500 ");
    Contains(guest, "pm4_kb_per_ms_span=97.68 ");
    Contains(guest, "pm4_kb_per_ms_frame=78.14");
    Require(guest.back() == '\n', "the guest line does not end the line");
    const auto lanes = FrameSteps::FormatLanes(totals, window);
    Contains(lanes, "[frame-steps] ");
    Contains(lanes, "packet_busy_ms/f=15.00 ");
    Contains(lanes, "us/packet=1.50 ");
    Contains(lanes, "prep_busy_ms/f(sum/max)=4.00/3.00 ");
    Contains(lanes, "rec_busy_ms/f(sum/max)=2.00/2.00 ");
    Contains(lanes, "rec_threads=0.00/0.00/2.00 ");
    Contains(lanes, "chunks/f=5.0 draws/chunk=25.0 ");
    Contains(lanes, "vk_submits/f=4.0 vk_submit_ms/f=0.50 ");
    Contains(lanes, "gpu_busy_ms/f=10.00 ");
    Require(lanes.back() == '\n', "the lanes line does not end the line");
    FrameSteps::Window idle;
    const auto quiet = FrameSteps::FormatLanes(FrameSteps::Totals{}, idle);
    Contains(quiet, "frames=0 ");
    Contains(quiet, "packet_busy_ms/f=0.00 ");
}

}

int main() {
    try {
        CheckGuestFrames();
        CheckSampleWindow();
        CheckCountersReset();
        CheckExecutedCount();
        CheckPresentPairing();
        CheckFormatting();
        std::cout << "frame steps tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
