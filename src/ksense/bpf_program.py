bpf_text = r"""
#include <linux/sched.h>

#define SUBBITS 4
#define SUBBUCKETS (1 << SUBBITS)

struct val_t {
    u64 sched_lat_us_sum;
    u64 sched_lat_cnt;
    u64 sched_lat_us_max;

    u64 dstate_us_sum;
    u64 dstate_cnt;

    u64 softirq_us_sum;
    u64 softirq_cnt;

    u64 sched_lat_dropped;
};

BPF_HASH(stats, u32, struct val_t);

BPF_HASH(ts_runnable, u32, u64);
BPF_HASH(ts_dstate, u32, u64);
BPF_HASH(ts_softirq, u32, u64);

/*
 * Higher-resolution histogram:
 * key = (b << SUBBITS) | s
 * b = floor(log2(delta_us))  (0..)
 * s = 0..(SUBBUCKETS-1) sub-range within [2^b, 2^(b+1))
 */
BPF_HISTOGRAM(sched_lat_hist);

static __always_inline u64 make_hist_key(u64 delta_us) {
    if (delta_us == 0) {
        return 0;
    }

    u64 b = bpf_log2l(delta_us);

    // Range [lo, hi] = [2^b, 2^(b+1)-1]
    // We want sub-buckets across that range.
    u64 lo = 1ULL << b;

    // For small b, avoid shifting by negative amounts when normalizing.
    // Use shift = max(b - SUBBITS, 0)
    u64 shift = (b > SUBBITS) ? (b - SUBBITS) : 0;

    // Normalize delta within the bucket by shifting down.
    // Sub-bucket is the top SUBBITS bits of (delta - lo) relative to bucket width.
    // This is an efficient approximation without division.
    u64 norm = (delta_us - lo) >> shift;
    u64 s = norm & (SUBBUCKETS - 1);

    return (b << SUBBITS) | s;
}

TRACEPOINT_PROBE(sched, sched_wakeup) {
    u32 pid = args->pid;
    u64 now = bpf_ktime_get_ns();

    u64 *dt = ts_dstate.lookup(&pid);
    if (dt) {
        if (now > *dt) {
            u64 delta_us = (now - *dt) / 1000;
            struct val_t *v, zero = {};
            u32 k = 0;
            v = stats.lookup_or_init(&k, &zero);
            if (v) {
                v->dstate_us_sum += delta_us;
                v->dstate_cnt += 1;
            }
        }
        ts_dstate.delete(&pid);
    }

    ts_runnable.update(&pid, &now);
    return 0;
}

TRACEPOINT_PROBE(sched, sched_wakeup_new) {
    u32 pid = args->pid;
    u64 now = bpf_ktime_get_ns();
    ts_runnable.update(&pid, &now);
    return 0;
}

TRACEPOINT_PROBE(sched, sched_switch) {
    u32 prev_pid = args->prev_pid;
    u32 next_pid = args->next_pid;
    u64 now = bpf_ktime_get_ns();

    if (args->prev_state == 0) {
        ts_runnable.update(&prev_pid, &now);
    }

    u64 *tsp = ts_runnable.lookup(&next_pid);
    if (tsp) {
        struct val_t *v, zero = {};
        u32 k = 0;
        v = stats.lookup_or_init(&k, &zero);

        if (v) {
            if (now > *tsp) {
                u64 delta_us = (now - *tsp) / 1000;

                v->sched_lat_us_sum += delta_us;
                v->sched_lat_cnt += 1;
                if (delta_us > v->sched_lat_us_max) v->sched_lat_us_max = delta_us;

                u64 key = make_hist_key(delta_us);
                sched_lat_hist.increment(key);
            } else {
                v->sched_lat_dropped += 1;
            }
        }
        ts_runnable.delete(&next_pid);
    }

    if (args->prev_state & 2) {
        ts_dstate.update(&prev_pid, &now);
    }

    return 0;
}

TRACEPOINT_PROBE(irq, softirq_entry) {
    if (args->vec != 3) return 0; // NET_RX typically
    u32 cpu = bpf_get_smp_processor_id();
    u64 now = bpf_ktime_get_ns();
    ts_softirq.update(&cpu, &now);
    return 0;
}

TRACEPOINT_PROBE(irq, softirq_exit) {
    if (args->vec != 3) return 0;
    u32 cpu = bpf_get_smp_processor_id();
    u64 *ts = ts_softirq.lookup(&cpu);
    if (ts) {
        u64 now = bpf_ktime_get_ns();
        if (now > *ts) {
            u64 delta_us = (now - *ts) / 1000;
            struct val_t *v, zero = {};
            u32 k = 0;
            v = stats.lookup_or_init(&k, &zero);
            if (v) {
                v->softirq_us_sum += delta_us;
                v->softirq_cnt += 1;
            }
        }
        ts_softirq.delete(&cpu);
    }
    return 0;
}

/* Cleanup to avoid PID map growth */
TRACEPOINT_PROBE(sched, sched_process_exit) {
    u32 pid = args->pid;
    ts_runnable.delete(&pid);
    ts_dstate.delete(&pid);
    return 0;
}
"""
