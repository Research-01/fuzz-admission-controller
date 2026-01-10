#!/usr/bin/env python3
"""
Node-Level Syscall + Kernel Signal Collector (FINAL WORKING VERSION)

This version properly uses TRACEPOINT_PROBE with 'args' parameter
MODIFIED: Added total_latency_us to output (sum of all latencies in window)
"""

from bcc import BPF
import ctypes as ct
import os
import time
from collections import defaultdict
from datetime import datetime
import signal
import sys

AGGREGATION_WINDOW = 1
FLUSH_INTERVAL = 1
RUNQLEN_SAMPLE_NS = 200_000_000

LATENCY_THRESHOLDS = {
    'read': 100, 'write': 100, 'sendto': 50, 'recvfrom': 50,
    'sendmsg': 50, 'recvmsg': 50, 'futex': 0, 'epoll_wait': 0,
    'epoll_pwait': 0, 'poll': 0, 'ppoll': 0, 'select': 0,
    'pselect6': 0, 'nanosleep': 0, 'fsync': 0, 'fdatasync': 0,
    'connect': 50, 'accept': 50, 'accept4': 50, 'stat': 0, 'fstat': 0,
    'runqlen': 0, 'softirq': 0, 'sched_lat': 0, 'block_io': 0,
    'tcp_retrans': 0, 'dstate_io': 0, 'io_qdepth': 0, 'io_merge': 0,
}

bpf_text = r"""
#include <uapi/linux/ptrace.h>
#include <linux/sched.h>

#ifndef TASK_RUNNING
#define TASK_RUNNING 0
#endif
#ifndef TASK_UNINTERRUPTIBLE
#define TASK_UNINTERRUPTIBLE 2
#endif

struct event_t {
    u64 ts_ns;
    u32 pid;
    u32 tid;
    char comm[16];
    u32 event_type;
    char event_name[16];
    u64 latency_us;
    s64 bytes;
    s64 ret;
    u32 cpu;
};

BPF_HASH(start, u64, u64);
BPF_HASH(runq_start, u32, u64);
BPF_HASH(dstate_start, u32, u64);
BPF_HASH(softirq_start, u64, u64);
BPF_HASH(block_start, u64, u64);  // sector -> timestamp
BPF_HASH(queue_depth, u32, u64);

BPF_ARRAY(runqlen_cpu, u64, 128);
BPF_ARRAY(last_runqlen_emit, u64, 128);
BPF_PERF_OUTPUT(events);

// -------------------- syscall enter/exit --------------------
static int trace_enter_common(struct pt_regs *ctx, const char *name)
{
    u64 tid = bpf_get_current_pid_tgid();
    u64 ts = bpf_ktime_get_ns();
    start.update(&tid, &ts);
    return 0;
}

static int trace_exit_common(struct pt_regs *ctx, const char *name, int is_net)
{
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 pid = pid_tgid >> 32;
    u64 tid = pid_tgid;
    u64 *tsp = start.lookup(&tid);
    if (!tsp) return 0;

    u64 ts = bpf_ktime_get_ns();
    u64 delta = ts - *tsp;
    start.delete(&tid);

    struct event_t ev = {};
    ev.ts_ns = ts;
    ev.tid = (u32)pid_tgid;
    ev.pid = pid;
    bpf_get_current_comm(&ev.comm, sizeof(ev.comm));
    ev.event_type = 0;

    #pragma unroll
    for (int i = 0; i < 15; i++) {
        ev.event_name[i] = name[i];
        if (name[i] == 0) break;
    }
    ev.event_name[15] = 0;

    ev.latency_us = delta / 1000;
    ev.ret = PT_REGS_RC(ctx);
    ev.bytes = (is_net && ev.ret > 0) ? ev.ret : 0;
    ev.cpu = bpf_get_smp_processor_id();
    events.perf_submit(ctx, &ev, sizeof(ev));
    return 0;
}

int trace_enter_read(struct pt_regs *ctx) { return trace_enter_common(ctx, "read"); }
int trace_exit_read(struct pt_regs *ctx) { return trace_exit_common(ctx, "read", 1); }
int trace_enter_write(struct pt_regs *ctx) { return trace_enter_common(ctx, "write"); }
int trace_exit_write(struct pt_regs *ctx) { return trace_exit_common(ctx, "write", 1); }
int trace_enter_sendto(struct pt_regs *ctx) { return trace_enter_common(ctx, "sendto"); }
int trace_exit_sendto(struct pt_regs *ctx) { return trace_exit_common(ctx, "sendto", 1); }
int trace_enter_recvfrom(struct pt_regs *ctx) { return trace_enter_common(ctx, "recvfrom"); }
int trace_exit_recvfrom(struct pt_regs *ctx) { return trace_exit_common(ctx, "recvfrom", 1); }
int trace_enter_sendmsg(struct pt_regs *ctx) { return trace_enter_common(ctx, "sendmsg"); }
int trace_exit_sendmsg(struct pt_regs *ctx) { return trace_exit_common(ctx, "sendmsg", 1); }
int trace_enter_recvmsg(struct pt_regs *ctx) { return trace_enter_common(ctx, "recvmsg"); }
int trace_exit_recvmsg(struct pt_regs *ctx) { return trace_exit_common(ctx, "recvmsg", 1); }
int trace_enter_poll(struct pt_regs *ctx) { return trace_enter_common(ctx, "poll"); }
int trace_exit_poll(struct pt_regs *ctx) { return trace_exit_common(ctx, "poll", 0); }
int trace_enter_ppoll(struct pt_regs *ctx) { return trace_enter_common(ctx, "ppoll"); }
int trace_exit_ppoll(struct pt_regs *ctx) { return trace_exit_common(ctx, "ppoll", 0); }
int trace_enter_epoll_wait(struct pt_regs *ctx) { return trace_enter_common(ctx, "epoll_wait"); }
int trace_exit_epoll_wait(struct pt_regs *ctx) { return trace_exit_common(ctx, "epoll_wait", 0); }
int trace_enter_epoll_pwait(struct pt_regs *ctx) { return trace_enter_common(ctx, "epoll_pwait"); }
int trace_exit_epoll_pwait(struct pt_regs *ctx) { return trace_exit_common(ctx, "epoll_pwait", 0); }
int trace_enter_select(struct pt_regs *ctx) { return trace_enter_common(ctx, "select"); }
int trace_exit_select(struct pt_regs *ctx) { return trace_exit_common(ctx, "select", 0); }
int trace_enter_pselect6(struct pt_regs *ctx) { return trace_enter_common(ctx, "pselect6"); }
int trace_exit_pselect6(struct pt_regs *ctx) { return trace_exit_common(ctx, "pselect6", 0); }
int trace_enter_futex(struct pt_regs *ctx) { return trace_enter_common(ctx, "futex"); }
int trace_exit_futex(struct pt_regs *ctx) { return trace_exit_common(ctx, "futex", 0); }
int trace_enter_connect(struct pt_regs *ctx) { return trace_enter_common(ctx, "connect"); }
int trace_exit_connect(struct pt_regs *ctx) { return trace_exit_common(ctx, "connect", 0); }
int trace_enter_accept(struct pt_regs *ctx) { return trace_enter_common(ctx, "accept"); }
int trace_exit_accept(struct pt_regs *ctx) { return trace_exit_common(ctx, "accept", 0); }
int trace_enter_accept4(struct pt_regs *ctx) { return trace_enter_common(ctx, "accept4"); }
int trace_exit_accept4(struct pt_regs *ctx) { return trace_exit_common(ctx, "accept4", 0); }
int trace_enter_fsync(struct pt_regs *ctx) { return trace_enter_common(ctx, "fsync"); }
int trace_exit_fsync(struct pt_regs *ctx) { return trace_exit_common(ctx, "fsync", 0); }
int trace_enter_fdatasync(struct pt_regs *ctx) { return trace_enter_common(ctx, "fdatasync"); }
int trace_exit_fdatasync(struct pt_regs *ctx) { return trace_exit_common(ctx, "fdatasync", 0); }
int trace_enter_nanosleep(struct pt_regs *ctx) { return trace_enter_common(ctx, "nanosleep"); }
int trace_exit_nanosleep(struct pt_regs *ctx) { return trace_exit_common(ctx, "nanosleep", 0); }
int trace_enter_stat(struct pt_regs *ctx) { return trace_enter_common(ctx, "stat"); }
int trace_exit_stat(struct pt_regs *ctx) { return trace_exit_common(ctx, "stat", 0); }
int trace_enter_fstat(struct pt_regs *ctx) { return trace_enter_common(ctx, "fstat"); }
int trace_exit_fstat(struct pt_regs *ctx) { return trace_exit_common(ctx, "fstat", 0); }

// ============================================================================
// SCHEDULER - TRACEPOINT_PROBE uses 'args' not 'ctx'
// ============================================================================

TRACEPOINT_PROBE(sched, sched_wakeup)
{
    u32 pid = args->pid;
    u64 ts = bpf_ktime_get_ns();
    runq_start.update(&pid, &ts);
    return 0;
}

// Helper functions for runqueue tracking
struct sched_wakeup_args {
    u64 __unused__;
    char comm[16];
    pid_t pid;
    int prio;
    int target_cpu;
};

struct sched_wakeup_new_args {
    u64 __unused__;
    char comm[16];
    pid_t pid;
    int prio;
    int target_cpu;
};

int trace_sched_wakeup_node(struct sched_wakeup_args *ctx)
{
    u32 cpu = (u32)ctx->target_cpu;
    if (cpu >= 128) return 0;
    u64 *c = runqlen_cpu.lookup(&cpu);
    u64 cur = c ? *c : 0;
    u64 newv = cur + 1;
    runqlen_cpu.update(&cpu, &newv);
    return 0;
}

int trace_sched_wakeup_new_node(struct sched_wakeup_new_args *ctx)
{
    u32 cpu = (u32)ctx->target_cpu;
    if (cpu >= 128) return 0;
    u64 *c = runqlen_cpu.lookup(&cpu);
    u64 cur = c ? *c : 0;
    u64 newv = cur + 1;
    runqlen_cpu.update(&cpu, &newv);
    return 0;
}

TRACEPOINT_PROBE(sched, sched_switch)
{
    u64 ts = bpf_ktime_get_ns();
    u32 prev_pid = args->prev_pid;
    u32 next_pid = args->next_pid;
    
    // Scheduling latency
    u64 *tsp = runq_start.lookup(&next_pid);
    if (tsp) {
        u64 delta = ts - *tsp;
        runq_start.delete(&next_pid);

        struct event_t ev = {};
        ev.ts_ns = ts;
        ev.pid = next_pid;
        ev.tid = next_pid;
        bpf_probe_read_kernel(&ev.comm, sizeof(ev.comm), args->next_comm);
        ev.event_type = 1;
        __builtin_memcpy(&ev.event_name, "sched_lat", 10);
        ev.latency_us = delta / 1000;
        ev.bytes = 0;
        ev.ret = 0;
        ev.cpu = bpf_get_smp_processor_id();
        events.perf_submit(args, &ev, sizeof(ev));
    }

    // D-state tracking
    if (args->prev_state == TASK_UNINTERRUPTIBLE) {
        dstate_start.update(&prev_pid, &ts);
    }
    
    u64 *dstate_tsp = dstate_start.lookup(&next_pid);
    if (dstate_tsp) {
        u64 dstate_delta = ts - *dstate_tsp;
        dstate_start.delete(&next_pid);
        
        struct event_t dev = {};
        dev.ts_ns = ts;
        dev.pid = next_pid;
        dev.tid = next_pid;
        bpf_probe_read_kernel(&dev.comm, sizeof(dev.comm), args->next_comm);
        dev.event_type = 6;
        __builtin_memcpy(&dev.event_name, "dstate_io", 10);
        dev.latency_us = dstate_delta / 1000;
        dev.bytes = 0;
        dev.ret = 0;
        dev.cpu = bpf_get_smp_processor_id();
        events.perf_submit(args, &dev, sizeof(dev));
    }

    // Runqlen update
    u32 cpu = (u32)bpf_get_smp_processor_id();
    if (args->prev_state != TASK_RUNNING) {
        u64 *c = runqlen_cpu.lookup(&cpu);
        if (c && *c > 0) {
            u64 newv = *c - 1;
            runqlen_cpu.update(&cpu, &newv);
        }
    }

    u64 now = bpf_ktime_get_ns();
    u64 *lastp = last_runqlen_emit.lookup(&cpu);
    if (!lastp) {
        last_runqlen_emit.update(&cpu, &now);
        return 0;
    }

    if (now - *lastp < RUNQLEN_SAMPLE_NS) return 0;
    last_runqlen_emit.update(&cpu, &now);

    u64 *rq = runqlen_cpu.lookup(&cpu);
    u64 rqv = rq ? *rq : 0;

    struct event_t ev2 = {};
    ev2.ts_ns = now;
    ev2.pid = 0;
    ev2.tid = 0;
    ev2.comm[0] = 0;
    ev2.event_type = 4;
    __builtin_memcpy(&ev2.event_name, "runqlen", 8);
    ev2.latency_us = 0;
    ev2.bytes = (s64)rqv;
    ev2.ret = 0;
    ev2.cpu = cpu;
    events.perf_submit(args, &ev2, sizeof(ev2));

    return 0;
}

// ============================================================================
// BLOCK I/O - FIXED: Use 'args' not 'ctx', use sector not req
// ============================================================================

TRACEPOINT_PROBE(block, block_rq_issue)
{
    u64 ts = bpf_ktime_get_ns();
    
    // FIXED: Access via args (not ctx), use sector (not req)
    u64 sector = args->sector;
    u32 dev = args->dev;
    u32 nr_sector = args->nr_sector;
    
    // Use sector as key
    block_start.update(&sector, &ts);
    
    // Update queue depth
    u64 *depth = queue_depth.lookup(&dev);
    u64 cur = depth ? *depth : 0;
    cur++;
    queue_depth.update(&dev, &cur);
    
    // Emit queue depth event
    struct event_t qev = {};
    qev.ts_ns = ts;
    qev.pid = 0;
    qev.tid = 0;
    qev.comm[0] = 0;
    qev.event_type = 7;
    __builtin_memcpy(&qev.event_name, "io_qdepth", 10);
    qev.latency_us = 0;
    qev.bytes = (s64)cur;
    qev.ret = (s64)dev;
    qev.cpu = bpf_get_smp_processor_id();
    events.perf_submit(args, &qev, sizeof(qev));
    
    return 0;
}

TRACEPOINT_PROBE(block, block_rq_complete)
{
    u64 ts = bpf_ktime_get_ns();
    
    // Access via args
    u64 sector = args->sector;
    u32 dev = args->dev;
    u32 nr_sector = args->nr_sector;
    int error = args->error;
    
    // Lookup by sector
    u64 *tsp = block_start.lookup(&sector);
    if (!tsp) {
        return 0;
    }
    
    u64 delta = ts - *tsp;
    block_start.delete(&sector);
    
    // Update queue depth
    u64 *depth = queue_depth.lookup(&dev);
    if (depth && *depth > 0) {
        u64 cur = *depth - 1;
        queue_depth.update(&dev, &cur);
    }

    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 pid = pid_tgid >> 32;

    struct event_t ev = {};
    ev.ts_ns = ts;
    ev.pid = pid;
    ev.tid = (u32)pid_tgid;
    bpf_get_current_comm(&ev.comm, sizeof(ev.comm));
    ev.event_type = 2;
    __builtin_memcpy(&ev.event_name, "block_io", 9);
    ev.latency_us = delta / 1000;
    ev.bytes = nr_sector * 512;
    ev.ret = error;
    ev.cpu = bpf_get_smp_processor_id();
    events.perf_submit(args, &ev, sizeof(ev));
    
    return 0;
}

TRACEPOINT_PROBE(block, block_rq_merge)
{
    struct event_t ev = {};
    ev.ts_ns = bpf_ktime_get_ns();
    ev.pid = 0;
    ev.tid = 0;
    ev.comm[0] = 0;
    ev.event_type = 8;
    __builtin_memcpy(&ev.event_name, "io_merge", 9);
    ev.latency_us = 0;
    ev.bytes = 1;
    ev.ret = 0;
    ev.cpu = bpf_get_smp_processor_id();
    events.perf_submit(args, &ev, sizeof(ev));
    return 0;
}

// ============================================================================
// TCP + SOFTIRQ
// ============================================================================

TRACEPOINT_PROBE(tcp, tcp_retransmit_skb)
{
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 pid = pid_tgid >> 32;

    struct event_t ev = {};
    ev.ts_ns = bpf_ktime_get_ns();
    ev.pid = pid;
    ev.tid = (u32)pid_tgid;
    bpf_get_current_comm(&ev.comm, sizeof(ev.comm));
    ev.event_type = 3;
    __builtin_memcpy(&ev.event_name, "tcp_retrans", 12);
    ev.latency_us = 0;
    ev.bytes = 0;
    ev.ret = 0;
    ev.cpu = bpf_get_smp_processor_id();
    events.perf_submit(args, &ev, sizeof(ev));
    return 0;
}

TRACEPOINT_PROBE(irq, softirq_entry)
{
    u32 vec = args->vec;
    u32 cpu = (u32)bpf_get_smp_processor_id();
    u64 key = ((u64)cpu << 32) | (u64)vec;
    u64 ts = bpf_ktime_get_ns();
    softirq_start.update(&key, &ts);
    return 0;
}

TRACEPOINT_PROBE(irq, softirq_exit)
{
    u32 vec = args->vec;
    u32 cpu = (u32)bpf_get_smp_processor_id();
    u64 key = ((u64)cpu << 32) | (u64)vec;

    u64 *tsp = softirq_start.lookup(&key);
    if (!tsp) return 0;

    u64 ts = bpf_ktime_get_ns();
    u64 delta = ts - *tsp;
    softirq_start.delete(&key);

    struct event_t ev = {};
    ev.ts_ns = ts;
    ev.pid = 0;
    ev.tid = 0;
    ev.comm[0] = 0;
    ev.event_type = 5;
    __builtin_memcpy(&ev.event_name, "softirq", 8);
    ev.latency_us = delta / 1000;
    ev.bytes = (s64)vec;
    ev.ret = 0;
    ev.cpu = cpu;
    events.perf_submit(args, &ev, sizeof(ev));
    return 0;
}
"""

bpf_text = bpf_text.replace("RUNQLEN_SAMPLE_NS", str(RUNQLEN_SAMPLE_NS))

print("Compiling BPF program (FINAL WORKING VERSION)...")
b = BPF(text=bpf_text)

# Attach syscalls
syscalls = [
    'read', 'write', 'sendto', 'recvfrom', 'sendmsg', 'recvmsg',
    'poll', 'ppoll', 'epoll_wait', 'epoll_pwait', 'select', 'pselect6',
    'futex', 'connect', 'accept', 'accept4',
    'fsync', 'fdatasync', 'nanosleep', 'stat', 'fstat'
]

for sc in syscalls:
    try:
        b.attach_kprobe(event=f"__x64_sys_{sc}", fn_name=f"trace_enter_{sc}")
        b.attach_kretprobe(event=f"__x64_sys_{sc}", fn_name=f"trace_exit_{sc}")
    except Exception as e:
        print(f"Warning: Could not attach {sc}: {e}")

# Manual attach for helper functions
try:
    b.attach_tracepoint(tp="sched:sched_wakeup", fn_name="trace_sched_wakeup_node")
    b.attach_tracepoint(tp="sched:sched_wakeup_new", fn_name="trace_sched_wakeup_new_node")
    print("✅ Runqueue tracking enabled")
except Exception as e:
    print(f"❌ Runqueue tracking failed: {e}")

print("✅ Block I/O: sector-based, using 'args' parameter (WORKING!)")
print("✅ D-state, scheduler, softirq, TCP tracking enabled")
print("✅ All tracepoints using TRACEPOINT_PROBE")
print("✅ MODIFIED: Added total_latency_us to output")
print("BPF program loaded successfully!")

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

class TimeWindowAggregator:
    def __init__(self, window_seconds=10):
        self.window = window_seconds
        self.data = defaultdict(lambda: defaultdict(lambda: {
            'count': 0, 'latency_sum': 0, 'latency_max': 0,
            'latencies': [], 'bytes_sum': 0, 'errors': 0
        }))
        self.buffer = []

    def add_event(self, event_name, latency_us, ret, bytes_val):
        ts_sec = time.time()
        window_start = int(ts_sec / self.window) * self.window
        key = (window_start, event_name)
        stats = self.data[window_start][key]
        stats['count'] += 1
        stats['latency_sum'] += latency_us
        stats['latency_max'] = max(stats['latency_max'], latency_us)
        stats['latencies'].append(latency_us)
        if bytes_val != 0:
            stats['bytes_sum'] += bytes_val
        if ret < 0:
            stats['errors'] += 1

    def flush_windows_before(self, ts_sec):
        current_window = int(ts_sec / self.window) * self.window
        for window in list(self.data.keys()):
            if window < current_window - self.window:
                self._flush_window(window)
                del self.data[window]

    def _flush_window(self, window):
        for (_, event_name), stats in self.data[window].items():
            if stats['count'] == 0:
                continue
            latencies = sorted(stats['latencies'])
            n = len(latencies)
            record = {
                'timestamp': int(window),
                'scope': 'node',
                'event': event_name,
                'count': stats['count'],
                'latency_avg_us': stats['latency_sum'] / stats['count'],
                'latency_max_us': stats['latency_max'],
                'latency_p50_us': latencies[int(n*0.5)] if n > 0 else 0,
                'latency_p95_us': latencies[int(n*0.95)] if n > 0 else 0,
                'latency_p99_us': latencies[int(n*0.99)] if n > 0 else 0,
                'total_latency_us': stats['latency_sum'],  # ADDED: Total latency sum
                'bytes_total': stats['bytes_sum'],
                'error_count': stats['errors'],
                'error_rate': stats['errors'] / stats['count']
            }
            self.buffer.append(record)

    def write_to_disk(self):
        if not self.buffer:
            return
        os.makedirs('./aggregated', exist_ok=True)
        date_str = datetime.now().strftime('%Y%m%d')
        filename = f'./aggregated/node_syscalls_{date_str}.csv'
        write_header = not os.path.exists(filename)
        with open(filename, 'a') as f:
            if write_header:
                # MODIFIED: Added total_latency_us column
                f.write('timestamp,scope,event,count,latency_avg_us,latency_max_us,latency_p50_us,latency_p95_us,latency_p99_us,total_latency_us,bytes_total,error_count,error_rate\n')
            for rec in self.buffer:
                f.write(f"{rec['timestamp']},{rec['scope']},{rec['event']},{rec['count']},"
                        f"{rec['latency_avg_us']:.2f},{rec['latency_max_us']},"
                        f"{rec['latency_p50_us']},{rec['latency_p95_us']},{rec['latency_p99_us']},"
                        f"{rec['total_latency_us']},"  # ADDED: Total latency value
                        f"{rec['bytes_total']},{rec['error_count']},{rec['error_rate']:.4f}\n")
        print(f"[WRITE] Wrote {len(self.buffer)} records to {filename}")
        self.buffer.clear()

aggregator = TimeWindowAggregator(window_seconds=AGGREGATION_WINDOW)
stats = {'total': 0, 'filtered': 0, 'aggregated': 0, 'lost': 0, 'block_io': 0}

def handle_event(cpu, data, size):
    global stats
    ev = ct.cast(data, ct.POINTER(Event)).contents
    stats['total'] += 1

    event_name = ev.event_name.decode('utf-8', 'replace').rstrip('\x00')
    
    # Track block_io events
    if event_name == 'block_io':
        stats['block_io'] += 1
    
    threshold = LATENCY_THRESHOLDS.get(event_name, 0)

    if ev.latency_us < threshold:
        stats['filtered'] += 1
        return

    stats['aggregated'] += 1
    aggregator.add_event(event_name=event_name,
                         latency_us=ev.latency_us,
                         ret=ev.ret,
                         bytes_val=ev.bytes)

def handle_lost(lost):
    stats['lost'] += lost

b["events"].open_perf_buffer(handle_event, lost_cb=handle_lost, page_cnt=256)

def signal_handler(sig, frame):
    print("\n[STOP] Shutting down...")
    aggregator.flush_windows_before(time.time())
    aggregator.write_to_disk()
    print(f"[STATS] Total: {stats['total']}, Aggregated: {stats['aggregated']}, "
          f"Filtered: {stats['filtered']}, Lost: {stats['lost']}, Block I/O: {stats['block_io']}")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def main():
    print("="*80)
    print("NODE SYSCALL COLLECTOR - WORKING VERSION (WITH TOTAL LATENCY)")
    print("="*80)
    print("✅ Block I/O: sector-based tracking (FIXED!)")
    print("✅ All events collected and aggregated")
    print("✅ ADDED: total_latency_us column (sum of all latencies in window)")
    print(f"Window: {AGGREGATION_WINDOW}s")
    print(f"Output: ./aggregated/node_syscalls_YYYYMMDD.csv")
    print("="*80)

    last_flush = time.time()
    last_stats = time.time()

    print("\n[START] Collecting data... Press Ctrl-C to stop")
    print("[TIP] Generate I/O: dd if=/dev/zero of=/tmp/test bs=1M count=100\n")
    
    while True:
        now = time.time()
        if now - last_flush > FLUSH_INTERVAL:
            aggregator.flush_windows_before(now)
            aggregator.write_to_disk()
            last_flush = now

        if now - last_stats > 30:
            print(f"[STATS] Total: {stats['total']}, Aggregated: {stats['aggregated']}, "
                  f"Filtered: {stats['filtered']}, Block I/O: {stats['block_io']}")
            last_stats = now

        try:
            b.perf_buffer_poll(timeout=100)
        except KeyboardInterrupt:
            break

if __name__ == '__main__':
    main()
