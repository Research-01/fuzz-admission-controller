BPF_TEMPLATE = r"""
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
        bpf_probe_read(&ev.comm, sizeof(ev.comm), args->next_comm);
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
        bpf_probe_read(&dev.comm, sizeof(dev.comm), args->next_comm);
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

{{BLOCK_RQ_MERGE_PROBE}}

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


def build_bpf_text(runqlen_sample_ns: int, enable_block_rq_merge: bool) -> str:
    if enable_block_rq_merge:
        block_probe = r"""
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
"""
    else:
        block_probe = ""

    text = BPF_TEMPLATE.replace("RUNQLEN_SAMPLE_NS", str(runqlen_sample_ns))
    return text.replace("{{BLOCK_RQ_MERGE_PROBE}}", block_probe)
