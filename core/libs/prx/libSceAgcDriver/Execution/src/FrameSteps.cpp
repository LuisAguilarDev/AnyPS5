#include "prx/libSceAgcDriver/Execution/include/FrameSteps.hpp"
#include <cstdio>

namespace AgcDriver::FrameSteps {

namespace {

constexpr std::size_t PacketRole = 1;
constexpr std::size_t SpineRole = 3;
constexpr double BytesPerKb = 1024.0;

double Ms(std::uint64_t ns) {
    return static_cast<double>(ns) / 1e6;
}

double Us(std::uint64_t ns) {
    return static_cast<double>(ns) / 1e3;
}

std::string ThreadList(const std::array<std::uint64_t, LaneThreads>& busyNs, std::uint64_t frames) {
    std::size_t used = 0;
    for (std::size_t i = 0; i < LaneThreads; ++i) {
        if (busyNs[i] != 0) used = i + 1;
    }
    if (used == 0) return "0";
    std::string list;
    char text[32];
    for (std::size_t i = 0; i < used; ++i) {
        std::snprintf(text, sizeof(text), "%s%.2f", i == 0 ? "" : "/", PerFrame(Ms(busyNs[i]), frames));
        list += text;
    }
    return list;
}

void SumAndMax(const std::array<std::uint64_t, LaneThreads>& busyNs, std::uint64_t frames, double& sum, double& max) {
    sum = 0;
    max = 0;
    for (const auto ns : busyNs) {
        const auto perFrame = PerFrame(Ms(ns), frames);
        sum += perFrame;
        max = std::max(max, perFrame);
    }
}

std::string Finish(const char* text, int length) {
    if (length < 0) return {};
    return std::string(text) + "\n";
}

std::string PhaseList(const std::array<std::uint64_t, PacketPhases>& phaseNs, std::uint64_t packets) {
    std::string list;
    char text[32];
    for (std::size_t i = 0; i < PacketPhases; ++i) {
        std::snprintf(text, sizeof(text), "%s%.2f", i == 0 ? "" : "/", PerItem(Us(phaseNs[i]), packets));
        list += text;
    }
    return list;
}

std::string DrawStepList(const Totals& totals) {
    if (totals.drawStepDraws == 0) return "-";
    std::string list;
    char text[64];
    for (std::size_t i = 0; i < DrawSteps; ++i) {
        if (totals.drawStepNs[i] == 0) continue;
        std::snprintf(text, sizeof(text), "%s%s=%.2f", list.empty() ? "" : "|", DriverDetail::DrawDriverPhaseNames[i], PerItem(Us(totals.drawStepNs[i]), totals.drawStepDraws));
        list += text;
    }
    return list.empty() ? "-" : list;
}

std::string SlotLabel(std::size_t slot, SlotNamer name) {
    if (slot == FlipSlot) return "FLIP";
    if (slot == RenderWaitSlot) return "RENDER_WAIT";
    if (name != nullptr) {
        if (auto named = name(slot); !named.empty()) return named;
    }
    char text[16];
    std::snprintf(text, sizeof(text), "0x%02zx", slot);
    return text;
}

std::string TopSlots(const Totals& totals, std::uint64_t frames, SlotNamer name) {
    std::array<std::size_t, PacketSlots> order{};
    std::size_t used = 0;
    for (std::size_t slot = 0; slot < PacketSlots; ++slot) {
        if (totals.slotPackets[slot] != 0) order[used++] = slot;
    }
    if (used == 0) return "-";
    const auto shown = std::min(used, TopPacketSlots);
    std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(shown), order.begin() + static_cast<std::ptrdiff_t>(used), [&](std::size_t left, std::size_t right) { return totals.slotNs[left] > totals.slotNs[right]; });
    std::string list;
    char text[96];
    for (std::size_t i = 0; i < shown; ++i) {
        const auto slot = order[i];
        std::snprintf(text, sizeof(text), ":%.0f/f:%.2fus:%.2fms/f", PerFrame(static_cast<double>(totals.slotPackets[slot]), frames), PerItem(Us(totals.slotNs[slot]), totals.slotPackets[slot]), PerFrame(Ms(totals.slotNs[slot]), frames));
        if (i != 0) list += ",";
        list += SlotLabel(slot, name) + text;
    }
    return list;
}

}

Counters& Global() {
    static Counters* counters = new Counters();
    return *counters;
}

std::string FormatGuest(const Totals& totals, const Window& window) {
    const auto frames = window.frames;
    const auto kbPerFrame = PerFrame(static_cast<double>(totals.dwords) * sizeof(std::uint32_t) / BytesPerKb, frames);
    const auto frameMs = window.gapMs.mean + window.spanMs.mean;
    const auto outsideMs = std::max(0.0, frameMs - window.insideMs.mean);
    const auto waitMs = [&](std::uint64_t ns) { return PerFrame(Ms(ns), frames); };
    const auto kernelMs = [&](std::size_t kind) { return waitMs(window.kernelWaitNs[kind]); };
    const auto backpressureMs = waitMs(totals.flipRoomNs) + waitMs(totals.flipHoldNs) + waitMs(totals.registrationNs) + kernelMs(WaitEqueue) + kernelMs(WaitSleep) + kernelMs(WaitVblank);
    const auto guestSyncMs = kernelMs(WaitEventFlag) + kernelMs(WaitSemaphore) + kernelMs(WaitCond) + kernelMs(WaitAddress);
    char text[4096];
    const auto length = std::snprintf(text, sizeof(text),
        "[frame-steps-guest] window_s=%.1f frames=%llu submits/f=%.1f submit_us/call=%.1f submit_ms/f=%.2f flip_submits/f=%.2f flip_submit_us/call=%.1f eop_reserves/f=%.2f eop_reserve_us/call=%.1f "
        "dwords/f=%.0f kb/f=%.1f packets/f=%.0f draws/f=%.0f "
        "gap_ms(p50/p95/mean)=%.2f/%.2f/%.2f span_ms(p50/p95/mean)=%.2f/%.2f/%.2f inside_calls_ms(p50/p95/mean)=%.2f/%.2f/%.2f outside_calls_ms/f=%.2f "
        "pm4_kb_per_ms_span=%.2f pm4_kb_per_ms_frame=%.2f q0_enqueue_to_packet_us=%.1f frame_ms=%.2f "
        "wait_flip_room_ms/f=%.2f wait_flip_hold_ms/f=%.2f wait_registration_ms/f=%.2f wait_equeue_ms/f=%.2f wait_sleep_ms/f=%.2f wait_vblank_ms/f=%.2f "
        "wait_evf_ms/f=%.2f wait_sema_ms/f=%.2f wait_cond_ms/f=%.2f wait_addr_ms/f=%.2f q0_render_wait_ms/f=%.2f "
        "backpressure_wait_ms/f=%.2f build_ms/f(proxy)=%.2f build_min_ms/f(proxy)=%.2f",
        window.seconds, static_cast<unsigned long long>(frames), PerFrame(static_cast<double>(totals.submits), frames), PerItem(Us(totals.submitNs), totals.submits), PerFrame(Ms(totals.submitNs), frames),
        PerFrame(static_cast<double>(totals.flipSubmits), frames), PerItem(Us(totals.flipSubmitNs), totals.flipSubmits),
        PerFrame(static_cast<double>(totals.eopReserves), frames), PerItem(Us(totals.eopReserveNs), totals.eopReserves),
        PerFrame(static_cast<double>(totals.dwords), frames), kbPerFrame, PerFrame(static_cast<double>(totals.packets), frames), PerFrame(static_cast<double>(totals.draws), frames),
        window.gapMs.p50, window.gapMs.p95, window.gapMs.mean, window.spanMs.p50, window.spanMs.p95, window.spanMs.mean,
        window.insideMs.p50, window.insideMs.p95, window.insideMs.mean, outsideMs,
        window.spanMs.mean > 0 ? kbPerFrame / window.spanMs.mean : 0.0, frameMs > 0 ? kbPerFrame / frameMs : 0.0, PerItem(Us(totals.queueLatencyNs), totals.queueLatencies), frameMs,
        waitMs(totals.flipRoomNs), waitMs(totals.flipHoldNs), waitMs(totals.registrationNs), kernelMs(WaitEqueue), kernelMs(WaitSleep), kernelMs(WaitVblank),
        kernelMs(WaitEventFlag), kernelMs(WaitSemaphore), kernelMs(WaitCond), kernelMs(WaitAddress), waitMs(totals.renderWaitNs),
        backpressureMs, std::max(0.0, frameMs - backpressureMs), std::max(0.0, frameMs - backpressureMs - guestSyncMs));
    return Finish(text, length);
}

std::string FormatLanes(const Totals& totals, const Window& window) {
    const auto frames = window.frames;
    const auto perFrameMs = [&](std::uint64_t ns) { return PerFrame(Ms(ns), frames); };
    const auto perFrameCount = [&](std::uint64_t count) { return PerFrame(static_cast<double>(count), frames); };
    double prepSum = 0, prepMax = 0, recSum = 0, recMax = 0;
    SumAndMax(totals.prepBusyNs, frames, prepSum, prepMax);
    SumAndMax(totals.recorderBusyNs, frames, recSum, recMax);
    const auto prepThreads = ThreadList(totals.prepBusyNs, frames);
    const auto recThreads = ThreadList(totals.recorderBusyNs, frames);
    char text[3072];
    const auto length = std::snprintf(text, sizeof(text),
        "[frame-steps] window_s=%.1f frames=%llu presents=%llu "
        "packet_busy_ms/f=%.2f packet_idle_ms/f=%.2f us/packet=%.2f packet_lock_wait_ms/f=%.2f packet_lock_hold_ms/f=%.2f packet_pool_wait_ms/f=%.2f "
        "prep_busy_ms/f(sum/max)=%.2f/%.2f prep_threads=%s prep_turn_wait_ms/f=%.2f prep_jobs/f=%.1f "
        "fifo_full_ms/f=%.2f fifo_drain_ms/f=%.2f fifo_empty_ms/f=%.2f fifo_depth=%.2f "
        "spine_busy_ms/f=%.2f spine_commits/f=%.0f draw_commit_us=%.2f spine_lock_hold_ms/f=%.2f spine_lock_wait_ms/f=%.2f chunks/f=%.1f draws/chunk=%.1f keep_sealed/f=%.1f "
        "rec_busy_ms/f(sum/max)=%.2f/%.2f rec_threads=%s "
        "vk_submits/f=%.1f vk_submit_ms/f=%.2f gpu_busy_ms/f=%.2f "
        "present_interval_ms(p50/p95)=%.2f/%.2f flip_to_present_ms(p50/p95/mean)=%.2f/%.2f/%.2f",
        window.seconds, static_cast<unsigned long long>(frames), static_cast<unsigned long long>(window.presents),
        perFrameMs(totals.packetBusyNs), perFrameMs(totals.packetIdleNs), PerItem(Us(totals.packetBusyNs), totals.packetPackets), perFrameMs(totals.lockWaitNs[PacketRole]), perFrameMs(totals.lockHoldNs[PacketRole]), perFrameMs(totals.packetPoolWaitNs),
        prepSum, prepMax, prepThreads.c_str(), perFrameMs(totals.prepTurnWaitNs), perFrameCount(totals.prepJobs),
        perFrameMs(totals.fifoFullNs), perFrameMs(totals.fifoDrainNs), perFrameMs(totals.fifoEmptyNs), PerItem(static_cast<double>(totals.fifoDepthSum), totals.fifoEnqueued),
        perFrameMs(totals.spineBusyNs), perFrameCount(totals.spineCommits), PerItem(Us(totals.drawCommitNs), totals.drawCommits), perFrameMs(totals.lockHoldNs[SpineRole]), perFrameMs(totals.lockWaitNs[SpineRole]), perFrameCount(totals.recorderChunks), PerItem(static_cast<double>(totals.recorderDraws), totals.recorderChunks), perFrameCount(totals.keepSealedChunks),
        recSum, recMax, recThreads.c_str(),
        perFrameCount(totals.vkSubmits), perFrameMs(totals.vkSubmitNs), PerFrame(window.gpuBusyMs, frames),
        window.intervalMs.p50, window.intervalMs.p95, window.flipToPresentMs.p50, window.flipToPresentMs.p95, window.flipToPresentMs.mean);
    return Finish(text, length);
}

std::string FormatPackets(const Totals& totals, const Window& window, SlotNamer name) {
    const auto frames = window.frames;
    std::uint64_t loopPackets = 0, loopNs = 0;
    for (std::size_t slot = 0; slot < PacketSlots; ++slot) {
        loopPackets += totals.slotPackets[slot];
        loopNs += totals.slotNs[slot];
    }
    const auto outside = [&](std::uint64_t whole, std::uint64_t part) { return PerFrame(Ms(whole > part ? whole - part : 0), frames); };
    char text[512];
    const auto length = std::snprintf(text, sizeof(text),
        "[frame-steps-packets] window_s=%.1f frames=%llu loop_packets/f=%.0f loop_ms/f=%.2f us/loop_packet=%.2f execute_outside_loop_ms/f=%.2f busy_outside_execute_ms/f=%.2f ",
        window.seconds, static_cast<unsigned long long>(frames), PerFrame(static_cast<double>(loopPackets), frames), PerFrame(Ms(loopNs), frames), PerItem(Us(loopNs), loopPackets),
        outside(totals.executeNs, loopNs), outside(totals.packetBusyNs, totals.executeNs));
    if (length < 0) return {};
    std::string line(text);
    line += "other_us/packet(gate/drain/lock/flush/prepare/body)=" + PhaseList(totals.otherPhaseNs, totals.phasePackets[0]);
    line += " draw_us/packet(gate/drain/lock/flush/prepare/body)=" + PhaseList(totals.drawPhaseNs, totals.phasePackets[1]);
    line += " draw_steps_us/drawn=" + DrawStepList(totals);
    line += " top=" + TopSlots(totals, frames, name) + "\n";
    return line;
}

}
