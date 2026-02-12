import csv
import os
import time
from collections import deque
from datetime import datetime

import ctypes as ct
import numpy as np
from bcc import BPF

from . import bpf_program
from .config import (
    BASELINE_WIN_S,
    ENERGY_CALIBRATE_AFTER_FREEZE,
    ENERGY_CALIB_WIN_S,
    FREEZE_BASELINE_AFTER_WARMUP,
    GRID_STEP_S,
    MAHAL_MIN_SAMPLES,
    MIN_SCHED_CNT_FOR_BASELINE,
    OUT_CSV,
    WARMUP_S,
    WINDOW_SEC,
)
from .energy import AdaptiveVolatilityEnergy
from .friction import mahalanobis_distance_and_direction
from .helpers import ensure_csv, percentiles_from_subbucket_hist


class ResourceSampler:
    """
    Sample node CPU utilization and PSI from /proc.
    Keeps internal state to compute deltas between calls.
    """

    def __init__(self):
        self._prev_total = None
        self._prev_idle = None
        self._psi_prev = {}

    def cpu_util(self):
        try:
            with open("/proc/stat", "r", encoding="utf-8") as f:
                line = f.readline()
        except FileNotFoundError:
            return None

        parts = line.split()
        if len(parts) < 5:
            return None

        try:
            values = [int(v) for v in parts[1:]]
        except ValueError:
            return None

        idle = values[3] + (values[4] if len(values) > 4 else 0)
        total = sum(values)

        if self._prev_total is None:
            self._prev_total = total
            self._prev_idle = idle
            return None

        total_delta = total - self._prev_total
        idle_delta = idle - self._prev_idle
        self._prev_total = total
        self._prev_idle = idle

        if total_delta <= 0:
            return None

        util = 1.0 - (idle_delta / total_delta)
        return max(0.0, min(100.0, util * 100.0))

    def psi(self):
        psi_vals = []
        now = time.monotonic()

        for path in ("/proc/pressure/cpu", "/proc/pressure/memory", "/proc/pressure/io"):
            total_us = None
            try:
                with open(path, "r", encoding="utf-8") as f:
                    lines = f.read().splitlines()
            except FileNotFoundError:
                continue

            for line in lines:
                if line.startswith("some"):
                    for part in line.split():
                        if part.startswith("total="):
                            try:
                                total_us = float(part.split("=")[1])
                            except ValueError:
                                total_us = None

            if total_us is None:
                continue

            prev = self._psi_prev.get(path)
            self._psi_prev[path] = (total_us, now)
            if not prev:
                continue

            prev_total, prev_ts = prev
            dt = now - prev_ts
            if dt <= 0:
                continue
            delta_us = total_us - prev_total
            if delta_us < 0:
                continue

            psi_pct = (delta_us / 1_000_000.0) / dt * 100.0
            psi_vals.append(psi_pct)

        if not psi_vals:
            return None

        psi = max(psi_vals)
        return max(0.0, min(100.0, psi))


def _ensure_or_rotate_csv(path: str, headers: list) -> None:
    """
    Ensure a CSV exists with the expected header. If the file exists but the
    header differs (e.g., after adding new columns), rotate it aside.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    expected = ",".join(headers)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                first = f.readline().strip()
        except OSError:
            first = ""

        if first and first != expected:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            rotated = os.path.join(parent, f"kernel_metrics_{ts}.csv")
            try:
                os.rename(path, rotated)
            except OSError:
                pass

    ensure_csv(path, headers)
def main():
    headers = [
        "Time",
        "SchedLat_Total_ms", "SchedLat_Avg_ms", "SchedLat_P95_ms", "SchedLat_P99_ms", "SchedLat_Max_ms",
        "SchedLat_Count", "SchedLat_Dropped",
        "DState_Total_ms", "DState_Count",
        "SoftIRQ_Total_ms", "SoftIRQ_Count",
        "CPUUtil",
        "PSI",
        "BaselineMode",
        "BaselineSamples",
        "Friction",
        "Direction",
        "dF_dt",
        "Energy",
        "Energy_W",
        "Energy_Vol",
        "Energy_kFactor",
    ]
    _ensure_or_rotate_csv(OUT_CSV, headers)

    baseline_w = max(10, int(BASELINE_WIN_S / GRID_STEP_S))

    keep_points = baseline_w + 100

    baseline_feat_b = deque(maxlen=baseline_w)

    fric_b = deque(maxlen=keep_points)
    dir_b = deque(maxlen=keep_points)
    dfr_b = deque(maxlen=keep_points)
    eng_b = deque(maxlen=keep_points)

    b = BPF(text=bpf_program.bpf_text, cflags=["-Wno-macro-redefined"])

    baseline_X = None
    energy_calc = AdaptiveVolatilityEnergy()
    resource = ResourceSampler()

    calib_fric_b = deque(maxlen=max(10, int(ENERGY_CALIB_WIN_S / GRID_STEP_S)))

    print("\n=== K-Sense Kernel Collector (Frozen Baseline + Adaptive Energy Window) ===")
    print(f"Sampling Rate: {GRID_STEP_S}s")
    print(f"Calibration (Warmup) Period: {WARMUP_S}s")
    print(f"Freeze baseline after warmup: {FREEZE_BASELINE_AFTER_WARMUP}")
    print(f"Min sched events for baseline sample: {MIN_SCHED_CNT_FOR_BASELINE}")
    print("Energy: mean(|ΔF|) over adaptive window W in "
          f"[{energy_calc.w_min},{energy_calc.w_max}], vol EMA alpha={energy_calc.alpha}")
    print(f"Output: {OUT_CSV}")
    print("Press Ctrl+C to stop.\n")

    t0 = time.time()

    try:
        while True:
            time.sleep(WINDOW_SEC)

            now = datetime.now()
            ts_str = now.strftime("%Y-%m-%d %H:%M:%S")

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

                # Updated percentile extraction (sub-bucket histogram, mid-point mapping)
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
            dstate_avg_ms = (dstate_total_ms / max(dstate_cnt, 1)) if dstate_total_ms > 0 else 0.0
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
                direction = float("nan")

                if (not in_warmup) and FREEZE_BASELINE_AFTER_WARMUP:
                    if len(baseline_feat_b) >= max(MAHAL_MIN_SAMPLES, x_t.shape[0] + 2):
                        baseline_X = np.array(baseline_feat_b, dtype=float)
                        baseline_mode = "FROZEN"
                        print(f"[BASELINE] Frozen with {baseline_X.shape[0]} samples at t={int(elapsed_s)}s")
                    else:
                        baseline_mode = "CALIBRATING"
            else:
                friction, direction = mahalanobis_distance_and_direction(x_t, baseline_X)

            fric_b.append(float(friction))
            dir_b.append(float(direction))
            if np.isfinite(friction):
                calib_fric_b.append(float(friction))

            # --- dF/dt ---
            if len(fric_b) >= 2 and np.isfinite(fric_b[-1]) and np.isfinite(fric_b[-2]):
                dF_dt = (fric_b[-1] - fric_b[-2]) / WINDOW_SEC
            else:
                dF_dt = float("nan")
            dfr_b.append(float(dF_dt))

            # --- Adaptive Energy update ---
            if (baseline_X is not None) and ENERGY_CALIBRATE_AFTER_FREEZE and (not energy_calc._calibrated):
                if len(calib_fric_b) >= 20:
                    arr = np.array(calib_fric_b, dtype=float)
                    abs_d = np.abs(np.diff(arr))
                    energy_calc.calibrate(abs_d)

            energy, w = energy_calc.update(friction)
            eng_b.append(float(energy) if np.isfinite(energy) else float("nan"))

            cpu_util = resource.cpu_util()
            psi = resource.psi()

            # --- CSV Output ---
            with open(OUT_CSV, "a", newline="") as f:
                csv.writer(f).writerow([
                    ts_str,
                    f"{sched_total_ms:.2f}", f"{sched_avg_ms:.4f}", f"{sched_p95_ms:.4f}",
                    f"{sched_p99_ms:.4f}", f"{sched_max_ms:.4f}",
                    sched_cnt, sched_dropped,
                    f"{dstate_total_ms:.2f}", dstate_cnt,
                    f"{softirq_total_ms:.2f}", softirq_cnt,
                    f"{cpu_util:.6f}" if cpu_util is not None else "",
                    f"{psi:.6f}" if psi is not None else "",
                    baseline_mode,
                    len(baseline_feat_b) if baseline_X is None else baseline_X.shape[0],
                    f"{friction:.6f}" if np.isfinite(friction) else "",
                    f"{direction:.1f}" if np.isfinite(direction) else "",
                    f"{dF_dt:.6f}" if np.isfinite(dF_dt) else "",
                    f"{energy:.6f}" if np.isfinite(energy) else "",
                    int(w),
                    f"{energy_calc.vol:.6f}",
                    f"{energy_calc.k_factor:.6f}",
                ])

    except KeyboardInterrupt:
        print("\nStopping. Outputs saved:")
        print(f" - {OUT_CSV}")
