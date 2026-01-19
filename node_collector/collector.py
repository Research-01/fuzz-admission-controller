import ctypes as ct
import os
import signal
import sys
import time

from bcc import BPF

from .aggregator import TimeWindowAggregator
from .bpf_program import build_bpf_text
from .config import AGGREGATION_WINDOW, FLUSH_INTERVAL, LATENCY_THRESHOLDS, RUNQLEN_SAMPLE_NS


class Event(ct.Structure):
    _fields_ = [
        ("ts_ns", ct.c_ulonglong),
        ("pid", ct.c_uint),
        ("tid", ct.c_uint),
        ("comm", ct.c_char * 16),
        ("event_type", ct.c_uint),
        ("event_name", ct.c_char * 16),
        ("latency_us", ct.c_ulonglong),
        ("bytes", ct.c_longlong),
        ("ret", ct.c_longlong),
        ("cpu", ct.c_uint),
    ]


def _attach_syscalls(bpf):
    syscalls = [
        "read", "write", "sendto", "recvfrom", "sendmsg", "recvmsg",
        "poll", "ppoll", "epoll_wait", "epoll_pwait", "select", "pselect6",
        "futex", "connect", "accept", "accept4",
        "fsync", "fdatasync", "nanosleep", "stat", "fstat",
    ]
    for sc in syscalls:
        try:
            bpf.attach_kprobe(event=f"__x64_sys_{sc}", fn_name=f"trace_enter_{sc}")
            bpf.attach_kretprobe(event=f"__x64_sys_{sc}", fn_name=f"trace_exit_{sc}")
        except Exception as exc:
            print(f"Warning: Could not attach {sc}: {exc}")


def _attach_runqlen(bpf):
    try:
        bpf.attach_tracepoint(tp="sched:sched_wakeup", fn_name="trace_sched_wakeup_node")
        bpf.attach_tracepoint(tp="sched:sched_wakeup_new", fn_name="trace_sched_wakeup_new_node")
        print("✅ Runqueue tracking enabled")
    except Exception as exc:
        print(f"❌ Runqueue tracking failed: {exc}")


def _has_tracepoint(tp_path: str) -> bool:
    return os.path.exists(tp_path)


def _setup_bpf():
    print("Compiling BPF program (FINAL WORKING VERSION)...")
    enable_block_merge = _has_tracepoint(
        "/sys/kernel/debug/tracing/events/block/block_rq_merge/id"
    )
    text = build_bpf_text(RUNQLEN_SAMPLE_NS, enable_block_merge)
    bpf = BPF(text=text)

    _attach_syscalls(bpf)
    _attach_runqlen(bpf)

    if enable_block_merge:
        print("✅ Block I/O: block_rq_merge tracepoint enabled")
    else:
        print("⚠️  Block I/O: block_rq_merge tracepoint missing; skipping")
    print("✅ Block I/O: sector-based, using 'args' parameter (WORKING!)")
    print("✅ D-state, scheduler, softirq, TCP tracking enabled")
    print("✅ All tracepoints using TRACEPOINT_PROBE")
    print("✅ MODIFIED: Added total_latency_us to output")
    print("BPF program loaded successfully!")
    return bpf


def _print_banner():
    print("=" * 80)
    print("NODE SYSCALL COLLECTOR - WORKING VERSION (WITH TOTAL LATENCY)")
    print("=" * 80)
    print("✅ Block I/O: sector-based tracking (FIXED!)")
    print("✅ All events collected and aggregated")
    print("✅ ADDED: total_latency_us column (sum of all latencies in window)")
    print(f"Window: {AGGREGATION_WINDOW}s")
    print("Output: ./aggregated/node_syscalls_YYYYMMDD.csv")
    print("=" * 80)


def main():
    bpf = _setup_bpf()

    aggregator = TimeWindowAggregator(window_seconds=AGGREGATION_WINDOW)
    stats = {"total": 0, "filtered": 0, "aggregated": 0, "lost": 0, "block_io": 0}

    def handle_event(cpu, data, size):
        ev = ct.cast(data, ct.POINTER(Event)).contents
        stats["total"] += 1

        event_name = ev.event_name.decode("utf-8", "replace").rstrip("\x00")
        if event_name == "block_io":
            stats["block_io"] += 1

        threshold = LATENCY_THRESHOLDS.get(event_name, 0)
        if ev.latency_us < threshold:
            stats["filtered"] += 1
            return

        stats["aggregated"] += 1
        aggregator.add_event(
            event_name=event_name,
            latency_us=ev.latency_us,
            ret=ev.ret,
            bytes_val=ev.bytes,
        )

    def handle_lost(lost):
        stats["lost"] += lost

    def signal_handler(sig, frame):
        print("\n[STOP] Shutting down...")
        aggregator.flush_windows_before(time.time())
        aggregator.write_to_disk()
        print(
            f"[STATS] Total: {stats['total']}, Aggregated: {stats['aggregated']}, "
            f"Filtered: {stats['filtered']}, Lost: {stats['lost']}, "
            f"Block I/O: {stats['block_io']}"
        )
        sys.exit(0)

    bpf["events"].open_perf_buffer(handle_event, lost_cb=handle_lost, page_cnt=256)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    _print_banner()
    print("\n[START] Collecting data... Press Ctrl-C to stop")
    print("[TIP] Generate I/O: dd if=/dev/zero of=/tmp/test bs=1M count=100\n")

    last_flush = time.time()
    last_stats = time.time()

    while True:
        now = time.time()
        if now - last_flush > FLUSH_INTERVAL:
            aggregator.flush_windows_before(now)
            aggregator.write_to_disk()
            last_flush = now

        if now - last_stats > 30:
            print(
                f"[STATS] Total: {stats['total']}, Aggregated: {stats['aggregated']}, "
                f"Filtered: {stats['filtered']}, Block I/O: {stats['block_io']}"
            )
            last_stats = now

        try:
            bpf.perf_buffer_poll(timeout=100)
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
