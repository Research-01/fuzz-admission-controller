#!/usr/bin/env python3
import os
import time
import csv
from datetime import datetime
from collections import deque
from typing import Tuple

import numpy as np
from bcc import BPF
import ctypes as ct

# -------------------------
# Collector cadence
# -------------------------
WINDOW_SEC = 1.0
GRID_STEP_S = 1  # should match WINDOW_SEC

# -------------------------
# Output files
# -------------------------
OUT_CSV  = "kernel_metrics.csv"
LIVE_PNG = "kernel_live.png"  # updated continuously in headless mode

# -------------------------
# Energy Settings (ADAPTIVE WINDOW ON |ΔF|)
# -------------------------
W_MIN = 5
W_MAX = 40

# Volatility estimator (EW-RMS of ΔF)
VOLATILITY_ALPHA = 0.10
VOL_EPS = 1e-6

ENERGY_CALIBRATE_AFTER_FREEZE = True
ENERGY_CALIB_WIN_S = 60          # seconds of friction history to calibrate from
ENERGY_RECALIB_EVERY_S = 300     # re-calibrate every N seconds after freeze (set None/0 to disable)
ENERGY_TARGET_W = (W_MIN + W_MAX) / 2.0

# Optional: provide a smooth "energy level" for display/alerting
ENERGY_SMOOTH_ALPHA = 0.15       # EMA on energy output; set 0 to disable

# -------------------------
# Mahalanobis friction config
# -------------------------
MAHAL_MIN_SAMPLES = 20
MAHAL_REG_REL = 1e-3
MAHAL_REG_ABS = 1e-9

# -------------------------
# Calibration / baseline
# -------------------------
WARMUP_S = 600
BASELINE_WIN_S = 5 * 60
FREEZE_BASELINE_AFTER_WARMUP = True
MIN_SCHED_CNT_FOR_BASELINE = 50
BASELINE_GATE_DIST = 3.08  # not used if baseline frozen

# -------------------------
# Plotting controls
# -------------------------
ENABLE_PLOT = True
PLOT_WINDOW_MIN = 60
SAVE_PNG_EVERY_S = 10

# -------------------------
# Matplotlib backend (GUI vs headless)
# -------------------------
HEADLESS = (os.environ.get("DISPLAY", "") == "")
if ENABLE_PLOT:
    import matplotlib
    if HEADLESS:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt


# ============================================================
# eBPF program
# ============================================================
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

BPF_HISTOGRAM(sched_lat_hist);

static __always_inline u64 make_hist_key(u64 delta_us) {
    if (delta_us == 0) {
        return 0;
    }

    u64 b = bpf_log2l(delta_us);
    u64 lo = 1ULL << b;
    u64 shift = (b > SUBBITS) ? (b - SUBBITS) : 0;

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

TRACEPOINT_PROBE(sched, sched_process_exit) {
    u32 pid = args->pid;
    ts_runnable.delete(&pid);
    ts_dstate.delete(&pid);
    return 0;
}
"""


# ============================================================
# Helpers
# ============================================================
def ensure_csv(path: str, header):
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(header)

def percentiles_from_subbucket_hist(items, ps=(0.95, 0.99), subbits=4, mode="mid"):
    buckets = sorted((int(k.value), int(v.value)) for k, v in items)
    total = sum(c for _, c in buckets)
    if total == 0:
        return {p: 0 for p in ps}

    targets = {p: int(total * p + 0.999999) for p in ps}
    out = {}
    running = 0

    subbuckets = 1 << int(subbits)

    def decode_range(key: int):
        b = key >> subbits
        s = key & (subbuckets - 1)

        lo = 1 << b
        hi = (1 << (b + 1)) - 1
        width = hi - lo + 1

        sub_lo = lo + (width * s) // subbuckets
        sub_hi = lo + (width * (s + 1)) // subbuckets - 1
        if sub_hi < sub_lo:
            sub_hi = sub_lo
        return sub_lo, sub_hi

    def pick_value(key: int) -> int:
        sub_lo, sub_hi = decode_range(key)
        if mode == "lower":
            return sub_lo
        if mode == "upper":
            return sub_hi
        return (sub_lo + sub_hi) // 2

    for key, c in buckets:
        running += c
        for p, t in targets.items():
            if p not in out and running >= t:
                out[p] = pick_value(key)

    last_key = buckets[-1][0]
    for p in ps:
        out.setdefault(p, pick_value(last_key))
    return out

def mahalanobis_distance(x: np.ndarray, X: np.ndarray) -> float:
    try:
        x = np.asarray(x, dtype=float).reshape(-1)
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or x.ndim != 1:
            return float("nan")
        if X.shape[1] != x.shape[0]:
            return float("nan")

        X = X[np.all(np.isfinite(X), axis=1)]
        if not np.all(np.isfinite(x)):
            return float("nan")

        d = x.shape[0]
        if X.shape[0] < max(MAHAL_MIN_SAMPLES, d + 2):
            return float("nan")

        mu = np.mean(X, axis=0)
        Sigma = np.cov(X, rowvar=False, bias=False)
        if Sigma.shape != (d, d) or not np.all(np.isfinite(Sigma)):
            return float("nan")

        diag = np.diag(Sigma)
        diag_mean = float(np.mean(diag)) if np.all(np.isfinite(diag)) else 0.0
        reg = MAHAL_REG_ABS + MAHAL_REG_REL * max(diag_mean, 0.0)
        Sigma_reg = Sigma + reg * np.eye(d)

        L = np.linalg.cholesky(Sigma_reg)
        diff = (x - mu).reshape(-1, 1)
        y = np.linalg.solve(L, diff)
        dist2 = float(np.dot(y[:, 0], y[:, 0]))
        return float(np.sqrt(max(dist2, 0.0)))
    except Exception:
        return float("nan")


# ============================================================
# Adaptive energy: mean(|ΔF|) over volatility-chosen window
# - FIXES:
#   * Calibrate using full ENERGY_CALIB_WIN_S
#   * Recalibrate periodically so k_factor tracks regime changes
#   * Vol uses EW-RMS of ΔF for stability
#   * Optional EMA smoothing on energy output (Energy_Smooth)
# ============================================================
class AdaptiveVolatilityEnergy:
    def __init__(self, w_min=W_MIN, w_max=W_MAX, alpha=VOLATILITY_ALPHA,
                 smooth_alpha=ENERGY_SMOOTH_ALPHA):
        self.w_min = int(w_min)
        self.w_max = int(w_max)
        self.alpha = float(alpha)

        # EW-RMS of ΔF: keep EW mean of ΔF^2
        self.vol2 = 0.0
        self.vol = 0.0

        self.buf = deque(maxlen=self.w_max + 5)
        self.current_w = self.w_max
        self.k_factor = 1.0
        self._calibrated = False

        self.smooth_alpha = float(smooth_alpha)
        self.energy_smooth = float("nan")

    def calibrate_from_friction(self, fric_hist: np.ndarray):
        fric_hist = np.asarray(fric_hist, dtype=float)
        fric_hist = fric_hist[np.isfinite(fric_hist)]
        if fric_hist.size < 3:
            self.k_factor = 1.0
            self._calibrated = True
            return

        abs_d = np.abs(np.diff(fric_hist))
        abs_d = abs_d[np.isfinite(abs_d)]
        if abs_d.size < 10:
            self.k_factor = 1.0
            self._calibrated = True
            return

        med = float(np.median(abs_d))
        # Choose k so that when vol ≈ median(|ΔF|), W ≈ target
        self.k_factor = max(1e-6, med * float(ENERGY_TARGET_W))
        self._calibrated = True

    def update(self, friction: float) -> Tuple[float, float, int]:
        """
        Returns (energy_raw, energy_smooth, current_w)
        energy_raw = mean(|ΔF|) over last W
        energy_smooth = EMA(energy_raw) if enabled else NaN
        """
        if not np.isfinite(friction):
            return float("nan"), self.energy_smooth, self.current_w

        self.buf.append(float(friction))

        # Update vol (EW-RMS of ΔF)
        if len(self.buf) >= 2:
            d = float(self.buf[-1] - self.buf[-2])
            d2 = d * d
            self.vol2 = self.alpha * d2 + (1.0 - self.alpha) * self.vol2
            self.vol = float(np.sqrt(max(self.vol2, 0.0)))

        raw_w = int(self.k_factor / (self.vol + VOL_EPS))
        self.current_w = max(self.w_min, min(self.w_max, raw_w))

        if len(self.buf) < 2:
            energy_raw = 0.0
        else:
            lookback = min(len(self.buf), self.current_w)
            window = np.array(list(self.buf)[-lookback:], dtype=float)
            energy_raw = float(np.mean(np.abs(np.diff(window)))) if window.size >= 2 else 0.0

        # Optional smoothing for interpretability
        if self.smooth_alpha and self.smooth_alpha > 0.0 and np.isfinite(energy_raw):
            if not np.isfinite(self.energy_smooth):
                self.energy_smooth = energy_raw
            else:
                a = self.smooth_alpha
                self.energy_smooth = a * energy_raw + (1.0 - a) * self.energy_smooth

        return energy_raw, self.energy_smooth, self.current_w


# ============================================================
# Live plot
# ============================================================
class LivePlot:
    def __init__(self):
        if not HEADLESS:
            plt.ion()
        self.fig = plt.figure(figsize=(14, 7))
        self.ax1 = self.fig.add_subplot(2, 1, 1)
        self.ax2 = self.fig.add_subplot(2, 1, 2, sharex=self.ax1)

        (self.l_fric,) = self.ax1.plot([], [], label="Friction (Mahalanobis)", color='red')
        self.ax1.set_ylabel("Distance")
        self.ax1.grid(True, alpha=0.3)
        self.ax1.legend(loc="upper right")
        self.ax1.set_title("Mahalanobis Friction (Magnitude)")

        (self.l_eng,) = self.ax2.plot([], [], label="Energy (mean |ΔF|)", color='orange')
        (self.l_eng_s,) = self.ax2.plot([], [], label="Energy_Smooth (EMA)", color='blue')
        self.ax2.set_ylabel("Energy")
        self.ax2.grid(True, alpha=0.3)
        self.ax2.legend(loc="upper right")
        self.ax2.set_title("System Energy (Instability)")

        self.fig.tight_layout()

    def update(self, t, fric, eng, eng_s):
        n = len(t)
        if not (len(fric) == n and len(eng) == n and len(eng_s) == n):
            return
        self.l_fric.set_data(t, fric)
        self.l_eng.set_data(t, eng)
        self.l_eng_s.set_data(t, eng_s)

        self.ax1.relim(); self.ax1.autoscale_view()
        self.ax2.relim(); self.ax2.autoscale_view()

        if not HEADLESS:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.001)

    def save(self, path: str):
        self.fig.savefig(path, dpi=140, bbox_inches="tight")


# ============================================================
# Main
# ============================================================
def main():
    headers = [
        "Time",
        "SchedLat_Total_ms", "SchedLat_Avg_ms", "SchedLat_P95_ms", "SchedLat_P99_ms", "SchedLat_Max_ms",
        "SchedLat_Count", "SchedLat_Dropped",
        "DState_Total_ms", "DState_Count",
        "SoftIRQ_Total_ms", "SoftIRQ_Count",
        "BaselineMode",
        "BaselineSamples",
        "Friction",
        "dF_dt",
        "Energy",
        "Energy_Smooth",
        "Energy_W",
        "Energy_Vol",
        "Energy_kFactor",
    ]
    ensure_csv(OUT_CSV, headers)

    baseline_w = max(10, int(BASELINE_WIN_S / GRID_STEP_S))

    plot_points = int((PLOT_WINDOW_MIN * 60) / GRID_STEP_S)
    keep_points = max(plot_points, baseline_w + 100)

    t_buf = deque(maxlen=keep_points)
    baseline_feat_b = deque(maxlen=baseline_w)

    fric_b = deque(maxlen=keep_points)
    dfr_b  = deque(maxlen=keep_points)
    eng_b  = deque(maxlen=keep_points)
    engs_b = deque(maxlen=keep_points)

    lp = LivePlot() if ENABLE_PLOT else None
    last_png_save = 0.0

    b = BPF(text=bpf_text)

    baseline_X = None
    energy_calc = AdaptiveVolatilityEnergy(w_min=W_MIN, w_max=W_MAX, alpha=VOLATILITY_ALPHA,
                                           smooth_alpha=ENERGY_SMOOTH_ALPHA)

    # keep enough friction for calibration/recalibration
    calib_len = max(10, int(ENERGY_CALIB_WIN_S / GRID_STEP_S) + 5)
    calib_fric_b = deque(maxlen=calib_len)

    # schedule recalibration
    last_recalib_t = None

    print("\n=== K-Sense Kernel Collector (Frozen Baseline + Adaptive Energy Window) ===")
    print(f"Sampling Rate: {GRID_STEP_S}s")
    print(f"Calibration (Warmup) Period: {WARMUP_S}s")
    print(f"Freeze baseline after warmup: {FREEZE_BASELINE_AFTER_WARMUP}")
    print(f"Min sched events for baseline sample: {MIN_SCHED_CNT_FOR_BASELINE}")
    print(f"Energy: mean(|ΔF|) over adaptive window W in [{W_MIN},{W_MAX}], vol EW-RMS alpha={VOLATILITY_ALPHA}")
    print(f"Energy calibration window: {ENERGY_CALIB_WIN_S}s; recalibrate every: {ENERGY_RECALIB_EVERY_S}s")
    print(f"Output: {OUT_CSV}")
    print("Press Ctrl+C to stop.\n")

    t0 = time.time()

    try:
        while True:
            time.sleep(WINDOW_SEC)

            now = datetime.now()
            ts_str = now.strftime("%Y-%m-%d %H:%M:%S")
            t_buf.append(now)

            # --- read BPF stats ---
            sched_total_ms = sched_avg_ms = sched_p95_ms = sched_p99_ms = sched_max_ms = 0.0
            sched_cnt = 0
            sched_dropped = 0
            dstate_total_ms = 0.0
            dstate_cnt = 0
            softirq_total_ms = 0.0
            softirq_cnt = 0

            v = b["stats"].get(ct.c_uint(0))
            if v:
                sched_cnt = int(v.sched_lat_cnt)
                sched_dropped = int(v.sched_lat_dropped)
                sched_total_ms = float(v.sched_lat_us_sum) / 1000.0
                sched_max_ms = float(v.sched_lat_us_max) / 1000.0
                if sched_cnt > 0:
                    sched_avg_ms = sched_total_ms / sched_cnt

                pct = percentiles_from_subbucket_hist(
                    b["sched_lat_hist"].items(),
                    ps=(0.95, 0.99),
                    subbits=4,
                    mode="mid",
                )
                sched_p95_ms = float(pct[0.95]) / 1000.0
                sched_p99_ms = float(pct[0.99]) / 1000.0

                dstate_total_ms = float(v.dstate_us_sum) / 1000.0
                dstate_cnt = int(v.dstate_cnt)

                softirq_total_ms = float(v.softirq_us_sum) / 1000.0
                softirq_cnt = int(v.softirq_cnt)

                b["stats"].clear()
                b["sched_lat_hist"].clear()

            # --- Feature vector ---
            dstate_avg_ms  = (dstate_total_ms / max(dstate_cnt, 1)) if dstate_total_ms > 0 else 0.0
            softirq_avg_ms = (softirq_total_ms / max(softirq_cnt, 1)) if softirq_total_ms > 0 else 0.0

            x_t = np.array([
                sched_p99_ms,
                sched_avg_ms,
                dstate_avg_ms,
                softirq_avg_ms,
            ], dtype=float)

            accept_baseline = np.all(np.isfinite(x_t)) and (sched_cnt >= MIN_SCHED_CNT_FOR_BASELINE)

            elapsed_s = time.time() - t0
            in_warmup = elapsed_s < WARMUP_S

            baseline_mode = "CALIBRATING" if (in_warmup and baseline_X is None) else "FROZEN"

            if baseline_X is None:
                if accept_baseline:
                    baseline_feat_b.append(x_t)

                friction = float("nan")

                if (not in_warmup) and FREEZE_BASELINE_AFTER_WARMUP:
                    if len(baseline_feat_b) >= max(MAHAL_MIN_SAMPLES, x_t.shape[0] + 2):
                        baseline_X = np.array(baseline_feat_b, dtype=float)
                        baseline_mode = "FROZEN"
                        print(f"[BASELINE] Frozen with {baseline_X.shape[0]} samples at t={int(elapsed_s)}s")

                        # reset calibration schedule at freeze
                        last_recalib_t = time.time()
                    else:
                        baseline_mode = "CALIBRATING"
            else:
                friction = mahalanobis_distance(x_t, baseline_X)

            fric_b.append(float(friction))

            # --- Maintain friction history for calibration ---
            if np.isfinite(friction):
                calib_fric_b.append(float(friction))

            # --- dF/dt ---
            if len(fric_b) >= 2 and np.isfinite(fric_b[-1]) and np.isfinite(fric_b[-2]):
                dF_dt = (fric_b[-1] - fric_b[-2]) / WINDOW_SEC
            else:
                dF_dt = float("nan")
            dfr_b.append(float(dF_dt))

            # --- Energy calibration / recalibration ---
            if baseline_X is not None and ENERGY_CALIBRATE_AFTER_FREEZE:
                need = int(ENERGY_CALIB_WIN_S / GRID_STEP_S)
                if (not energy_calc._calibrated) and (len(calib_fric_b) >= need):
                    energy_calc.calibrate_from_friction(np.array(calib_fric_b)[-need:])
                elif ENERGY_RECALIB_EVERY_S and ENERGY_RECALIB_EVERY_S > 0 and (len(calib_fric_b) >= need):
                    now_t = time.time()
                    if last_recalib_t is not None and (now_t - last_recalib_t) >= ENERGY_RECALIB_EVERY_S:
                        energy_calc.calibrate_from_friction(np.array(calib_fric_b)[-need:])
                        last_recalib_t = now_t

            # --- Adaptive Energy update ---
            energy, energy_s, w = energy_calc.update(friction)
            eng_b.append(float(energy) if np.isfinite(energy) else float("nan"))
            engs_b.append(float(energy_s) if np.isfinite(energy_s) else float("nan"))

            # --- CSV Output ---
            with open(OUT_CSV, "a", newline="") as f:
                csv.writer(f).writerow([
                    ts_str,
                    f"{sched_total_ms:.2f}", f"{sched_avg_ms:.4f}", f"{sched_p95_ms:.4f}", f"{sched_p99_ms:.4f}", f"{sched_max_ms:.4f}",
                    sched_cnt, sched_dropped,
                    f"{dstate_total_ms:.2f}", dstate_cnt,
                    f"{softirq_total_ms:.2f}", softirq_cnt,
                    baseline_mode,
                    len(baseline_feat_b) if baseline_X is None else baseline_X.shape[0],
                    f"{friction:.6f}" if np.isfinite(friction) else "",
                    f"{dF_dt:.6f}" if np.isfinite(dF_dt) else "",
                    f"{energy:.6f}" if np.isfinite(energy) else "",
                    f"{energy_s:.6f}" if np.isfinite(energy_s) else "",
                    int(w),
                    f"{energy_calc.vol:.6f}",
                    f"{energy_calc.k_factor:.6f}",
                ])

            # --- Plot Update ---
            if lp is not None:
                t_list = list(t_buf)[-plot_points:]
                fr_list = list(fric_b)[-plot_points:]
                en_list = list(eng_b)[-plot_points:]
                es_list = list(engs_b)[-plot_points:]
                lp.update(t_list, fr_list, en_list, es_list)

                if HEADLESS:
                    now_t = time.time()
                    if (now_t - last_png_save) >= SAVE_PNG_EVERY_S:
                        lp.save(LIVE_PNG)
                        last_png_save = now_t

    except KeyboardInterrupt:
        print("\nStopping. Outputs saved:")
        print(f" - {OUT_CSV}")
        if ENABLE_PLOT and HEADLESS:
            print(f" - {LIVE_PNG}")


if __name__ == "__main__":
    main()
