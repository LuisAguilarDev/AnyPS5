#include "prx/libSceAgcDriver/Execution/include/FrameSteps.hpp"
#include "ProgramKeyCensus.hpp"
#include <cmath>
#include <cstdint>
#include <cstdlib>
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

std::int64_t FakeNowNs = 0;

std::int64_t FakeClock() {
    return FakeNowNs;
}

void CheckPacketSteps() {
    FrameSteps::Counters counters;
    FakeNowNs = 1000;
    {
        FrameSteps::PacketStep step(counters, 0x76, false, FakeClock);
        FakeNowNs += 100;
        step.Mark(FrameSteps::PhaseFlush);
        FakeNowNs += 300;
        step.Mark(FrameSteps::PhaseBody);
        FakeNowNs += 600;
    }
    {
        FrameSteps::PacketStep step(counters, 0x2d, true, FakeClock);
        FakeNowNs += 50;
        step.Mark(FrameSteps::PhasePrepare);
        FakeNowNs += 2000;
    }
    {
        FrameSteps::PacketStep idle(counters, 0x10, false, nullptr);
        FakeNowNs += 5000;
        idle.Mark(FrameSteps::PhaseBody);
    }
    const auto totals = counters.Take();
    Require(totals.slotPackets[0x76] == 1 && totals.slotNs[0x76] == 1000, "a SET_SH_REG step did not count its packet and time");
    Require(totals.slotPackets[0x2d] == 1 && totals.slotNs[0x2d] == 2050, "a draw step did not count its packet and time");
    Require(totals.slotPackets[0x10] == 0 && totals.slotNs[0x10] == 0, "a step without a clock counted");
    Require(totals.otherPhaseNs[FrameSteps::PhaseGate] == 100 && totals.otherPhaseNs[FrameSteps::PhaseFlush] == 300 && totals.otherPhaseNs[FrameSteps::PhaseBody] == 600, "the phases of a non-draw packet are not the time between its marks");
    Require(totals.drawPhaseNs[FrameSteps::PhaseGate] == 50 && totals.drawPhaseNs[FrameSteps::PhasePrepare] == 2000, "the phases of a draw are not the time between its marks");
    Require(totals.phasePackets[0] == 1 && totals.phasePackets[1] == 1, "the draw and non-draw steps were not counted apart");
    Require(counters.Take().slotPackets[0x76] == 0, "taking the counters did not reset the packet slots");
}

void CheckPacketFormatting() {
    FrameSteps::Totals totals;
    totals.packetBusyNs = static_cast<std::uint64_t>(30 * Ms);
    totals.executeNs = static_cast<std::uint64_t>(26 * Ms);
    totals.slotPackets[0x76] = 8000;
    totals.slotNs[0x76] = static_cast<std::uint64_t>(8 * Ms);
    totals.slotPackets[0x2d] = 1000;
    totals.slotNs[0x2d] = static_cast<std::uint64_t>(12 * Ms);
    totals.slotPackets[FrameSteps::FlipSlot] = 2;
    totals.slotNs[FrameSteps::FlipSlot] = static_cast<std::uint64_t>(2 * Ms);
    totals.phasePackets[0] = 9002;
    totals.phasePackets[1] = 1000;
    totals.otherPhaseNs[FrameSteps::PhaseFlush] = static_cast<std::uint64_t>(9 * Ms);
    totals.drawPhaseNs[FrameSteps::PhaseBody] = static_cast<std::uint64_t>(10 * Ms);
    FrameSteps::Window window;
    window.seconds = 10.0;
    window.frames = 2;
    const auto line = FrameSteps::FormatPackets(totals, window, [](std::size_t slot) { return slot == 0x76 ? std::string("SET_SH_REG") : std::string(); });
    Contains(line, "[frame-steps-packets] ");
    Contains(line, "loop_packets/f=4501 ");
    Contains(line, "loop_ms/f=11.00 ");
    Contains(line, "us/loop_packet=2.44 ");
    Contains(line, "execute_outside_loop_ms/f=2.00 ");
    Contains(line, "busy_outside_execute_ms/f=2.00 ");
    Contains(line, "other_us/packet(gate/drain/lock/flush/prepare/body)=0.00/0.00/0.00/1.00/0.00/0.00 ");
    Contains(line, "draw_us/packet(gate/drain/lock/flush/prepare/body)=0.00/0.00/0.00/0.00/0.00/10.00 ");
    Contains(line, "top=0x2d:500/f:12.00us:6.00ms/f,SET_SH_REG:4000/f:1.00us:4.00ms/f,FLIP:1/f:1000.00us:1.00ms/f");
    Require(line.back() == '\n', "the packets line does not end the line");
    const auto quiet = FrameSteps::FormatPackets(FrameSteps::Totals{}, FrameSteps::Window{}, nullptr);
    Contains(quiet, "loop_packets/f=0 ");
    Contains(quiet, "top=-");
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
    totals.keepSealedChunks = 3;
    totals.chunkEpochJoins = 4;
    totals.staleRecordInputs = 6;
    totals.staleRecordResources = 2;
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
    totals.flipRoomNs = static_cast<std::uint64_t>(2 * Ms);
    totals.renderWaitNs = static_cast<std::uint64_t>(6 * Ms);
    window.kernelWaitNs[FrameSteps::WaitEqueue] = static_cast<std::uint64_t>(4 * Ms);
    window.kernelWaitNs[FrameSteps::WaitSleep] = static_cast<std::uint64_t>(2 * Ms);
    window.kernelWaitNs[FrameSteps::WaitCond] = static_cast<std::uint64_t>(2 * Ms);
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
    Contains(guest, "frame_ms=10.00 ");
    Contains(guest, "wait_flip_room_ms/f=1.00 wait_flip_hold_ms/f=0.00 wait_registration_ms/f=0.00 ");
    Contains(guest, "wait_equeue_ms/f=2.00 wait_sleep_ms/f=1.00 wait_vblank_ms/f=0.00 ");
    Contains(guest, "wait_evf_ms/f=0.00 wait_sema_ms/f=0.00 wait_cond_ms/f=1.00 wait_addr_ms/f=0.00 ");
    Contains(guest, "q0_render_wait_ms/f=3.00 ");
    Contains(guest, "backpressure_wait_ms/f=4.00 build_ms/f(proxy)=6.00 build_min_ms/f(proxy)=5.00");
    Require(guest.back() == '\n', "the guest line does not end the line");
    const auto lanes = FrameSteps::FormatLanes(totals, window);
    Contains(lanes, "[frame-steps] ");
    Contains(lanes, "packet_busy_ms/f=15.00 ");
    Contains(lanes, "us/packet=1.50 ");
    Contains(lanes, "prep_busy_ms/f(sum/max)=4.00/3.00 ");
    Contains(lanes, "rec_busy_ms/f(sum/max)=2.00/2.00 ");
    Contains(lanes, "rec_threads=0.00/0.00/2.00 ");
    Contains(lanes, "chunks/f=5.0 draws/chunk=25.0 keep_sealed/f=1.5 epoch_joins/f=2.0 stale_inputs/f=3.0 stale_resources/f=1.0 ");
    Contains(lanes, "vk_submits/f=4.0 vk_submit_ms/f=0.50 ");
    Contains(lanes, "gpu_busy_ms/f=10.00 ");
    Require(lanes.back() == '\n', "the lanes line does not end the line");
    FrameSteps::Window idle;
    const auto quiet = FrameSteps::FormatLanes(FrameSteps::Totals{}, idle);
    Contains(quiet, "frames=0 ");
    Contains(quiet, "packet_busy_ms/f=0.00 ");
}

void CheckFifoSplits() {
    FrameSteps::Counters counters;
    FrameSteps::NoteFifoFull(counters, 0, 3 * Ms);
    FrameSteps::NoteFifoFull(counters, FrameSteps::PoolEnqueueKind, 2 * Ms);
    FrameSteps::NoteFifoFull(counters, 99, 1 * Ms);
    FrameSteps::NoteFifoDrain(counters, 3, 0x100, 4 * Ms);
    FrameSteps::NoteFifoDrain(counters, 0, 0x2d, 1 * Ms);
    FrameSteps::NoteFifoDrain(counters, 0, 0x2d, 2 * Ms);
    FrameSteps::NoteFifoDrain(counters, 77, 0x9999, -5);
    const auto totals = counters.Take();
    std::uint64_t kinds = 0;
    for (const auto ns : totals.fifoFullKindNs) kinds += ns;
    Require(totals.fifoFullNs == static_cast<std::uint64_t>(6 * Ms) && kinds == totals.fifoFullNs, "the fifo_full kinds do not sum to fifo_full");
    Require(totals.fifoFullKindNs[FrameSteps::PoolEnqueueKind] == static_cast<std::uint64_t>(2 * Ms), "a pool caller was not its own kind");
    Require(totals.fifoFullKindNs[FrameSteps::PoolEnqueueKind - 1] == static_cast<std::uint64_t>(1 * Ms), "a kind past the list was not folded into other");
    std::uint64_t reasons = 0, opcodes = 0, drains = 0;
    for (const auto ns : totals.fifoDrainReasonNs) reasons += ns;
    for (const auto ns : totals.fifoDrainOpcodeNs) opcodes += ns;
    for (const auto count : totals.fifoDrainReasonCount) drains += count;
    Require(totals.fifoDrainNs == static_cast<std::uint64_t>(7 * Ms) && reasons == totals.fifoDrainNs && opcodes == totals.fifoDrainNs, "the drain reasons or opcodes do not sum to fifo_drain");
    Require(drains == 4 && totals.fifoDrainReasonCount[FrameSteps::DrainReasons - 1] == 1, "a drain was not counted, or a reason past the list was not folded");
    Require(totals.fifoDrainOpcodeNs[FrameSteps::NoDrainOpcode] == static_cast<std::uint64_t>(4 * Ms) && totals.fifoDrainOpcodeCount[FrameSteps::NoDrainOpcode] == 2, "a drain without an opcode (or past 0x100) was not counted at 0x100");
    Require(totals.fifoDrainOpcodeCount[0x2d] == 2 && totals.fifoDrainOpcodeNs[0x2d] == static_cast<std::uint64_t>(3 * Ms), "the drains of one opcode were not added up");
    Require(counters.Take().fifoFullKindNs[0] == 0, "taking the counters did not reset the kinds");
}

void CheckDerived() {
    FrameSteps::Totals totals;
    totals.packetBusyNs = static_cast<std::uint64_t>(94 * Ms);
    totals.packetIdleNs = static_cast<std::uint64_t>(6 * Ms);
    totals.fifoFullNs = static_cast<std::uint64_t>(10 * Ms);
    totals.fifoDrainNs = static_cast<std::uint64_t>(6 * Ms);
    totals.packetPoolWaitNs = static_cast<std::uint64_t>(2 * Ms);
    totals.spineBusyNs = static_cast<std::uint64_t>(84 * Ms);
    totals.queueLatencyNs = static_cast<std::uint64_t>(4 * Ms);
    totals.queueLatencies = 4;
    FrameSteps::Window window;
    window.frames = 2;
    window.gapMs.mean = 48.0;
    window.spanMs.mean = 2.0;
    window.flipToPresentMs.mean = 50.0;
    const auto derived = FrameSteps::Derive(totals, window);
    Require(Near(derived.frameMs, 50.0), "frame_ms is not gap + span");
    Require(Near(derived.packetOwnMs, 38.0), "packet_own is not packet_busy less fifo_full, fifo_drain and pool_wait");
    Require(Near(derived.couplingMs, 50.0 - 3.0 - 42.0), "coupling is not frame less packet_idle and the larger lane");
    Require(Near(derived.tailMs, 50.0 - 1.0 - 47.0), "tail is not flip_to_present less q0_enqueue and packet_busy");
    totals.spineBusyNs = 0;
    Require(Near(FrameSteps::Derive(totals, window).couplingMs, 50.0 - 3.0 - 38.0), "coupling did not take packet_own when it is the larger lane");
    const auto quiet = FrameSteps::Derive(FrameSteps::Totals{}, FrameSteps::Window{});
    Require(Near(quiet.packetOwnMs, 0.0) && Near(quiet.couplingMs, 0.0) && Near(quiet.tailMs, 0.0), "an empty window derived nonzero values");
    totals.fifoFullNs = static_cast<std::uint64_t>(200 * Ms);
    Require(Near(FrameSteps::Derive(totals, window).packetOwnMs, 0.0), "packet_own went below zero");
    const auto lanes = FrameSteps::FormatLanes(totals, window);
    Contains(lanes, "packet_own_ms/f=0.00 ");
    Contains(lanes, "coupling_ms=47.00 ");
    Contains(lanes, "tail_ms=2.00");
}

void CheckSampler() {
    FrameSteps::Sampler every16(16);
    std::size_t picked = 0, first = 99, second = 99;
    for (std::size_t i = 0; i < 64; ++i) {
        if (!every16.Next()) continue;
        if (picked == 0) first = i;
        if (picked == 1) second = i;
        ++picked;
    }
    Require(picked == 4 && first == 0 && second == 16, "the sampler does not pick one commit in 16, starting with the first");
    Require(FrameSteps::SampleEvery(nullptr) == 16 && FrameSteps::SampleEvery("") == 16, "an unset sample switch is not 1 in 16");
    Require(FrameSteps::SampleEvery("64") == 64 && FrameSteps::SampleEvery("1") == 1, "a power of two was not taken");
    Require(FrameSteps::SampleEvery("48") == 16 && FrameSteps::SampleEvery("0") == 16 && FrameSteps::SampleEvery("x") == 16 && FrameSteps::SampleEvery("4096") == 16, "a value that is not a power of two up to 1024 was taken");
}

void CheckSpineSample() {
    FrameSteps::Counters counters;
    FrameSteps::SpineSample sample;
    sample.phaseNs[0] = 10'000;
    sample.phaseNs[FrameSteps::SpineDrawPhases - 1] = 20'000;
    FrameSteps::NoteSpineSample(counters, sample, 40'000);
    sample.phaseNs = {};
    sample.phaseNs[1] = 50'000;
    FrameSteps::NoteSpineSample(counters, sample, 40'000);
    const auto totals = counters.Take();
    Require(totals.spineSamples == 2 && totals.spineSampleNs == 80'000, "the sampled commits were not counted");
    std::uint64_t phases = 0;
    for (const auto ns : totals.spinePhaseNs) phases += ns;
    Require(totals.spinePhaseNs[FrameSteps::SpineOtherPhase] == 10'000, "the commit time outside the draw marks is not 'other'");
    Require(phases == 90'000, "a phase past its commit was not kept as measured");
    FrameSteps::Window window;
    window.frames = 2;
    FrameSteps::Totals split;
    split.spineSamples = 2;
    split.spineSampleNs = 80'000;
    split.drawCommits = 32;
    split.drawCommitNs = 32 * 41'000;
    split.spinePhaseNs[0] = 30'000;
    split.spinePhaseNs[FrameSteps::SpineOtherPhase] = 50'000;
    const auto line = FrameSteps::FormatSpine(split, window, 16);
    Contains(line, "[frame-steps-spine] ");
    Contains(line, "sampled_commits=2 sample_every=16 sampled_commit_us=40.00 draw_commit_us=41.00 ");
    Contains(line, "spine_phases_us/commit=validate=15.00|other=25.00");
    Require(line.back() == '\n', "the spine line does not end the line");
    Contains(FrameSteps::FormatSpine(FrameSteps::Totals{}, window, 16), "spine_phases_us/commit=-");
}

void CheckDigest() {
    const std::vector<std::uint32_t> words{1u, 2u, 3u};
    auto changed = words;
    changed[2] = 4u;
    Require(FrameSteps::DigestWords(words) == FrameSteps::DigestWords(std::vector<std::uint32_t>{1u, 2u, 3u}), "equal words digest apart");
    Require(FrameSteps::DigestWords(words) != FrameSteps::DigestWords(changed), "a changed word kept the digest");
    Require(FrameSteps::DigestWords({}) == FrameSteps::DigestOffset, "no words do not digest to the offset basis");
    const std::vector<std::uint32_t> head{1u, 2u}, tail{3u};
    Require(FrameSteps::DigestWords(tail, FrameSteps::DigestWords(head)) == FrameSteps::DigestWords(words), "chained spans do not digest as their concatenation");
}

void CheckFrameChain() {
    FrameSteps::FrameChain chain;
    chain.Submitted(1 * Ms);
    chain.Dequeued(10 * Ms);
    chain.Dequeued(11 * Ms);
    chain.Submitted(13 * Ms);
    chain.Submitted(15 * Ms);
    chain.Flipped();
    chain.Presented(18 * Ms);
    chain.Dequeued(30 * Ms);
    chain.Submitted(31 * Ms);
    chain.Flipped();
    chain.Presented(40 * Ms);
    const auto totals = chain.Take();
    Require(totals.firstSubmits == 2 && totals.firstSubmitNs == static_cast<std::uint64_t>(4 * Ms), "the first submit is not timed from the frame's first dequeue");
    Require(totals.tails == 2 && totals.tailNs == static_cast<std::uint64_t>(12 * Ms), "the tail is not the last submit to the present");
    const auto next = chain.Take();
    Require(next.firstSubmits == 0 && next.tails == 0, "taking the chain did not reset it");
    FrameSteps::FrameChain quiet;
    quiet.Presented(5 * Ms);
    Require(quiet.Take().tails == 0, "a present with no submit before it was timed");
}

std::string RunIdLine(std::string_view describe, std::string_view built, std::vector<std::string_view> environment, std::size_t room = 512) {
    std::vector<char> out(room, 'x');
    const auto length = FrameSteps::FormatRunId(out, describe, built, environment);
    Require(length < room && out[length] == '\0', "the run id line is not NUL-terminated inside its buffer");
    return std::string(out.data(), length);
}

void CheckRunId() {
    const std::vector<std::string_view> environment{"PATH=C:\\bin", "APS5_PROFILE_FRAME=1", "APS5_DRAW_PIPELINE=1", "aps5_lower=1", "APS5_EMPTY="};
    const auto line = RunIdLine("ba27916dd-dirty+1a2b3c4d", "Oct 26 2026 10:00:00", environment);
    Require(line == "[run-id] libs=ba27916dd-dirty+1a2b3c4d built=Oct 26 2026 10:00:00 env=APS5_DRAW_PIPELINE=1,APS5_EMPTY=,APS5_PROFILE_FRAME=1\n", "the run id line is not the build and the sorted APS5_ variables: " + line);
    Require(RunIdLine("", "", {"PATH=C:\\bin"}) == "[run-id] libs=unknown built=unknown env=-\n", "an unknown build or no APS5_ variable is not named so");
    // A line past its buffer is cut, still ending in a newline.
    const auto cut = RunIdLine("ba27916dd", "Oct 26 2026 10:00:00", environment, 24);
    Require(cut == "[run-id] libs=ba27916d\n", "a run id line past its buffer is not cut to fit: " + cut);
}

void CheckThreadCpu() {
    Require(FrameSteps::CyclesToNs(3'000'000, 3'000'000'000, 1'000'000'000) == 1'000'000, "cycles at the calibrated rate are not converted to time");
    Require(FrameSteps::CyclesToNs(5, 0, 10) == 0, "cycles without a calibration were converted");
    Require(FrameSteps::CpuSlot(FrameSteps::RolePacket, 3) == FrameSteps::CpuPacket && FrameSteps::CpuSlot(FrameSteps::RoleSpine, 0) == FrameSteps::CpuSpine, "the packet thread or the spine has another slot");
    Require(FrameSteps::CpuSlot(FrameSteps::RoleRecorder, 2) == FrameSteps::CpuRecorders + 2 && FrameSteps::CpuSlot(FrameSteps::RolePool, 99) == FrameSteps::CpuPool + FrameSteps::LaneThreads - 1, "a recorder or pool thread has another slot");
    Require(FrameSteps::CpuSlot(FrameSteps::RoleOther, 0) == FrameSteps::CpuSlots, "a thread without a role has a slot");
    FrameSteps::Window window;
    window.frames = 2;
    window.threadCpuNs[FrameSteps::CpuPacket] = static_cast<std::uint64_t>(80 * Ms);
    window.threadCpuNs[FrameSteps::CpuSpine] = static_cast<std::uint64_t>(70 * Ms);
    window.threadCpuNs[FrameSteps::CpuPresenter] = static_cast<std::uint64_t>(2 * Ms);
    window.threadCpuNs[FrameSteps::CpuRecorders + 1] = static_cast<std::uint64_t>(4 * Ms);
    window.threadCpuNs[FrameSteps::CpuPool] = static_cast<std::uint64_t>(1 * Ms);
    const auto line = FrameSteps::FormatThreadCpu(window);
    Contains(line, "[frame-steps-cpu] ");
    Contains(line, "thread_cpu_ms/f packet=40.00 spine=35.00 presenter=1.00 release=0.00 rec=0.00/2.00 pool=0.50");
    Require(line.back() == '\n', "the cpu line does not end the line");
    Contains(FrameSteps::FormatThreadCpu(FrameSteps::Window{}), "rec=0 pool=0");
}

void CheckSubmitSplits() {
    FrameSteps::Counters counters;
    FrameSteps::NoteVkSubmit(counters, FrameSteps::RolePacket, FrameSteps::TriggerCap);
    FrameSteps::NoteVkSubmit(counters, FrameSteps::RolePacket, FrameSteps::TriggerCap);
    FrameSteps::NoteVkSubmit(counters, FrameSteps::RoleSpine, FrameSteps::TriggerLabel);
    FrameSteps::NoteVkSubmit(counters, 99, 99);
    const auto totals = counters.Take();
    Require(totals.vkSubmitsBy[FrameSteps::RolePacket * FrameSteps::SubmitTriggers + FrameSteps::TriggerCap] == 2, "the cap submits of the packet thread were not counted");
    Require(totals.vkSubmitsBy[(FrameSteps::ThreadRoles - 1) * FrameSteps::SubmitTriggers + FrameSteps::SubmitTriggers - 1] == 1, "a role or trigger past the list was not folded");
    FrameSteps::Window window;
    window.frames = 2;
    window.gpuGapMs = 3.0;
    window.chain.firstSubmitNs = static_cast<std::uint64_t>(3 * Ms);
    window.chain.firstSubmits = 2;
    window.chain.tailNs = static_cast<std::uint64_t>(5 * Ms);
    window.chain.tails = 2;
    auto split = totals;
    split.submits = 4;
    split.submitValidateNs = 8'000;
    split.vkSubmits = 4;
    const auto line = FrameSteps::FormatChain(split, window);
    Contains(line, "[frame-steps-chain] ");
    Contains(line, "gpu_gap_ms/f=1.50 first_submit_ms=1.50 last_submit_to_present_ms=2.50 submit_validate_us=2.0 ");
    Contains(line, "vk_submits/f=2.0 vk_submits/f(thread:trigger)=packet:cap=1.0,spine:label=0.5,release:suspend=0.5");
    Require(line.back() == '\n', "the chain line does not end the line");
    Contains(FrameSteps::FormatChain(FrameSteps::Totals{}, FrameSteps::Window{}), "vk_submits/f(thread:trigger)=-");
    {
        const FrameSteps::SubmitCause cause(FrameSteps::TriggerFlip);
        Require(FrameSteps::ThreadTrigger() == (FrameSteps::Enabled() ? FrameSteps::TriggerFlip : FrameSteps::TriggerOther), "a submit cause did not name the trigger while profiling");
        {
            const FrameSteps::SubmitCause inner(FrameSteps::TriggerLabel, true);
            Require(FrameSteps::ThreadTrigger() == (FrameSteps::Enabled() ? FrameSteps::TriggerFlip : FrameSteps::TriggerOther), "a fallback cause replaced the outer one");
        }
        Require(FrameSteps::ThreadTrigger() == (FrameSteps::Enabled() ? FrameSteps::TriggerFlip : FrameSteps::TriggerOther), "a fallback cause did not restore the outer one");
    }
    {
        const FrameSteps::SubmitCause fallback(FrameSteps::TriggerLabel, true);
        Require(FrameSteps::ThreadTrigger() == (FrameSteps::Enabled() ? FrameSteps::TriggerLabel : FrameSteps::TriggerOther), "a fallback cause without an outer one did not name the trigger");
    }
    Require(FrameSteps::ThreadTrigger() == FrameSteps::TriggerOther, "a submit cause outlived its scope");
}

void CheckFifoFormatting() {
    FrameSteps::Totals totals;
    totals.fifoFullNs = static_cast<std::uint64_t>(6 * Ms);
    totals.fifoFullKindNs[0] = static_cast<std::uint64_t>(4 * Ms);
    totals.fifoFullKindNs[FrameSteps::PoolEnqueueKind] = static_cast<std::uint64_t>(2 * Ms);
    totals.fifoDrainNs = static_cast<std::uint64_t>(5 * Ms);
    totals.fifoDrainReasonNs[0] = static_cast<std::uint64_t>(1 * Ms);
    totals.fifoDrainReasonNs[3] = static_cast<std::uint64_t>(4 * Ms);
    totals.fifoDrainReasonCount[0] = 2;
    totals.fifoDrainReasonCount[3] = 6;
    totals.fifoDrainOpcodeNs[0x2d] = static_cast<std::uint64_t>(1 * Ms);
    totals.fifoDrainOpcodeCount[0x2d] = 2;
    totals.fifoDrainOpcodeNs[FrameSteps::NoDrainOpcode] = static_cast<std::uint64_t>(4 * Ms);
    totals.fifoDrainOpcodeCount[FrameSteps::NoDrainOpcode] = 6;
    totals.mpassInline = 4;
    totals.mpassPipelined = 6;
    FrameSteps::Window window;
    window.frames = 2;
    const auto line = FrameSteps::FormatFifo(totals, window, [](std::size_t opcode) { return opcode == 0x2d ? std::string("DRAW_INDEX_AUTO") : std::string(); });
    Contains(line, "[frame-steps-fifo] ");
    Contains(line, "fifo_full_ms/f=3.00 fifo_full_ms/f(draw/label/dma/dispatch/fill/pass/completion/open_work/other/pool)=2.00/0.00/0.00/0.00/0.00/0.00/0.00/0.00/0.00/1.00 ");
    Contains(line, "fifo_drain_ms/f=2.50 fifo_drains/f(packet/flush/labels/capture/indirect/submission)=1.0/0.0/0.0/3.0/0.0/0.0 ");
    Contains(line, "fifo_drain_ms/f(packet/flush/labels/capture/indirect/submission)=0.50/0.00/0.00/2.00/0.00/0.00 ");
    Contains(line, "drain_top=none:3.0/f:2.00ms/f,DRAW_INDEX_AUTO:1.0/f:0.50ms/f ");
    Contains(line, "mpass_inline/f=2.0 mpass_pipelined/f=3.0");
    Require(line.back() == '\n', "the fifo line does not end the line");
    Contains(FrameSteps::FormatFifo(FrameSteps::Totals{}, window, nullptr), "drain_top=-");
}

void CheckMemoFormatting() {
    FrameSteps::Totals totals;
    totals.proofBreaks[0] = 2;
    totals.proofBreaks[4] = 8;
    totals.targetMemoLookups = 10;
    totals.targetMemoHits = 9;
    totals.drawInputMemoLookups = 4;
    totals.drawInputMemoHits = 1;
    totals.drawStages = 30;
    totals.stagedDraws = 12;
    totals.halfChecks = 20;
    totals.halfStale = 2;
    totals.halfCopies = 20;
    totals.halfCopyNs = 2000;
    totals.holdWaits = 3;
    totals.holdWaitNs = 3000000;
    FrameSteps::Window window;
    window.frames = 2;
    window.shaders.resultMemoHits = 3;
    window.shaders.resultMemoMisses = 1;
    window.shaders.resultMemoEvictions = 4;
    window.shaders.programKeys = 40;
    window.shaders.programKeysDistinct = 7;
    window.shaders.programKeyTableHits = 30;
    const auto line = FrameSteps::FormatMemo(totals, window);
    Contains(line, "[frame-steps-memo] ");
    Contains(line, "proof_breaks/f=5.0 proof_breaks/f(other/store/written/write_changed/gpu_hold/non_draw_commit/completion_label/copied_writer/resource_write_back)=1.0/0.0/0.0/0.0/4.0/0.0/0.0/0.0/0.0 ");
    Contains(line, "target_memo_hit%=90.0 target_memo_lookups/f=5.0 draw_input_memo_hit%=25.0 draw_input_memo_lookups/f=2.0 ");
    Contains(line, "result_memo_hit%=75.0 result_memo_lookups/f=2.0 result_memo_evictions/f=2.0 ");
    Contains(line, "stages/draw=2.50 ");
    Contains(line, "program_keys/10s=7 program_key_materializes/f=20.0 program_key_table_hit%=75.0 program_key_overflow=0 ");
    Contains(line, "half_checks/f=10.0 half_stale/f=1.0 half_copy_us/draw=0.100 hold_waits/f=1.5 hold_wait_ms/f=1.50");
    Require(line.back() == '\n', "the memo line does not end the line");
    Contains(FrameSteps::FormatMemo(FrameSteps::Totals{}, FrameSteps::Window{}), "target_memo_hit%=0.0 ");
}

void CheckProgramKeyCensus() {
    ShaderRecompiler::ProgramKeyCensus census;
    for (std::uint64_t key : {5ull, 7ull, 5ull, 0ull, 5ull + ShaderRecompiler::ProgramKeyCensus::TableSlots}) census.Note(key);
    const auto counts = census.Take();
    Require(counts.keys == 5 && counts.distinct == 4 && counts.overflow == 0, "the census does not count the distinct keys");
    Require(counts.tableHits == 1, "a direct-mapped table hit was not counted, or a collision counted as a hit");
    census.Note(7);
    const auto next = census.Take();
    Require(next.keys == 1 && next.distinct == 1 && next.tableHits == 1, "the next window did not start empty, or the table forgot its entries");
    std::vector<std::uint64_t> plan{1, 2, 3};
    std::vector<std::uint32_t> module{4, 5};
    const auto key = ShaderRecompiler::ProgramKeyCensus::Combine(ShaderRecompiler::ProgramKeyCensus::Hash(std::span<const std::uint64_t>(plan)), ShaderRecompiler::ProgramKeyCensus::Hash(std::span<const std::uint32_t>(module)));
    module[1] = 6;
    Require(key != ShaderRecompiler::ProgramKeyCensus::Combine(ShaderRecompiler::ProgramKeyCensus::Hash(std::span<const std::uint64_t>(plan)), ShaderRecompiler::ProgramKeyCensus::Hash(std::span<const std::uint32_t>(module))), "a changed module key kept the program key");
}

}

int main() {
    // Before the first Enabled(): the submit causes are checked as they run under profiling.
#ifdef _WIN32
    _putenv("APS5_PROFILE_FRAME=1");
#else
    setenv("APS5_PROFILE_FRAME", "1", 1);
#endif
    try {
        CheckGuestFrames();
        CheckSampleWindow();
        CheckCountersReset();
        CheckExecutedCount();
        CheckPacketSteps();
        CheckPacketFormatting();
        CheckPresentPairing();
        CheckFormatting();
        CheckFifoSplits();
        CheckDerived();
        CheckSampler();
        CheckSpineSample();
        CheckDigest();
        CheckFrameChain();
        CheckRunId();
        CheckThreadCpu();
        CheckSubmitSplits();
        CheckFifoFormatting();
        CheckMemoFormatting();
        CheckProgramKeyCensus();
        std::cout << "frame steps tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
