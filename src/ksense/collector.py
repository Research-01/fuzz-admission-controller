import csv
import os
import re
import time
from datetime import datetime
from typing import Dict, Tuple

import ctypes as ct
from bcc import BPF

from . import bpf_program
from .config import (
    GRID_STEP_S,
    OUT_CSV,
    WINDOW_SEC,
)
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


_POD_UID_RE = re.compile(r"pod([0-9a-fA-F]{8}[-_][0-9a-fA-F]{4}[-_][0-9a-fA-F]{4}[-_][0-9a-fA-F]{4}[-_][0-9a-fA-F]{12})")


class CgroupIndex:
    def __init__(self, refresh_s: float = 10.0):
        self._refresh_s = refresh_s
        self._last_refresh = 0.0
        self._cache: Dict[int, Tuple[str, str]] = {}
        self._warned = False

    def _candidate_roots(self):
        roots = []
        for path in ("/sys/fs/cgroup/kubepods.slice", "/sys/fs/cgroup/kubepods"):
            if os.path.isdir(path):
                roots.append(path)
        roots.append("/sys/fs/cgroup")
        return roots

    def _scan(self) -> Dict[int, Tuple[str, str]]:
        if not os.path.exists("/sys/fs/cgroup/cgroup.controllers"):
            if not self._warned:
                print("[WARN] cgroup v2 not detected; pod-level metrics disabled.")
                self._warned = True
            return {}

        mapping: Dict[int, Tuple[str, str]] = {}
        roots = self._candidate_roots()
        for root in roots:
            if not os.path.isdir(root):
                continue
            for dirpath, dirnames, _ in os.walk(root):
                base = os.path.basename(dirpath)
                m = _POD_UID_RE.search(base)
                if not m:
                    continue
                uid = m.group(1).replace("_", "-").lower()
                cgid_path = os.path.join(dirpath, "cgroup.id")
                try:
                    with open(cgid_path, "r", encoding="utf-8") as f:
                        cgid = int(f.read().strip())
                except (OSError, ValueError):
                    continue
                mapping[cgid] = (uid, dirpath)
                dirnames[:] = []
        return mapping

    def get(self) -> Dict[int, Tuple[str, str]]:
        now = time.monotonic()
        if not self._cache or (now - self._last_refresh) >= self._refresh_s:
            self._cache = self._scan()
            self._last_refresh = now
        return self._cache


class CgroupResourceSampler:
    def __init__(self):
        self._prev_cpu: Dict[str, Tuple[float, float]] = {}
        self._prev_psi: Dict[str, Tuple[float, float]] = {}
        self._ncpu = os.cpu_count() or 1

    def cpu_util(self, cg_path: str):
        stat_path = os.path.join(cg_path, "cpu.stat")
        usage_usec = None
        try:
            with open(stat_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) == 2 and parts[0] == "usage_usec":
                        usage_usec = float(parts[1])
                        break
        except OSError:
            return None

        if usage_usec is None:
            return None

        now = time.monotonic()
        prev = self._prev_cpu.get(cg_path)
        self._prev_cpu[cg_path] = (usage_usec, now)
        if not prev:
            return None

        prev_usage, prev_ts = prev
        dt = now - prev_ts
        if dt <= 0:
            return None

        delta = usage_usec - prev_usage
        if delta < 0:
            return None

        util = (delta / 1_000_000.0) / (dt * self._ncpu) * 100.0
        return max(0.0, min(100.0, util))

    def psi(self, cg_path: str):
        psi_vals = []
        now = time.monotonic()
        for fname in ("cpu.pressure", "memory.pressure", "io.pressure"):
            path = os.path.join(cg_path, fname)
            total_us = None
            try:
                with open(path, "r", encoding="utf-8") as f:
                    lines = f.read().splitlines()
            except OSError:
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

            prev = self._prev_psi.get(path)
            self._prev_psi[path] = (total_us, now)
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


_REQUIRED_TRACEPOINTS = [
    ("sched", "sched_wakeup"),
    ("sched", "sched_wakeup_new"),
    ("sched", "sched_switch"),
    ("irq", "softirq_entry"),
    ("irq", "softirq_exit"),
    ("sched", "sched_process_exit"),
]


def _has_tracepoint_format(category: str, event: str) -> bool:
    for p in (
        f"/sys/kernel/tracing/events/{category}/{event}/format",
        f"/sys/kernel/debug/tracing/events/{category}/{event}/format",
    ):
        if os.path.exists(p) and os.access(p, os.R_OK):
            return True
    return False


def _preflight_or_die() -> None:
    if hasattr(os, "geteuid") and os.geteuid() != 0:
        raise SystemExit(
            "This collector needs root privileges for eBPF.\n"
            "Run: sudo -E python3 collector_only.py"
        )

    missing = [(c, e) for (c, e) in _REQUIRED_TRACEPOINTS if not _has_tracepoint_format(c, e)]
    if missing:
        missing_s = ", ".join([f"{c}:{e}" for c, e in missing])
        raise SystemExit(
            "Required tracepoint format files are not readable; BCC cannot build TRACEPOINT_PROBE structs.\n"
            f"Missing/unreadable: {missing_s}\n\n"
            "Fix (run as root) by mounting tracefs/debugfs:\n"
            "  sudo mount -t tracefs nodev /sys/kernel/tracing 2>/dev/null || true\n"
            "  sudo mount -t debugfs none /sys/kernel/debug 2>/dev/null || true\n\n"
            "Also ensure the kernel exposes these tracepoints and that /sys is not restricted."
        )


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
    _preflight_or_die()
    headers = [
        "Level",
        "Time",
        "PodUID",
        "CgroupID",
        "SchedLat_Total_ms", "SchedLat_Avg_ms", "SchedLat_P95_ms", "SchedLat_P99_ms", "SchedLat_Max_ms",
        "SchedLat_Count", "SchedLat_Dropped",
        "DState_Total_ms", "DState_Count",
        "SoftIRQ_Total_ms", "SoftIRQ_Count",
        "CPUUtil",
        "PSI",
    ]
    _ensure_or_rotate_csv(OUT_CSV, headers)

    b = BPF(text=bpf_program.bpf_text, cflags=["-Wno-macro-redefined"])

    resource = ResourceSampler()
    cg_index = CgroupIndex()
    cg_resource = CgroupResourceSampler()

    print("\n=== K-Sense Kernel Collector (Node + Pod Metrics) ===")
    print(f"Sampling Rate: {GRID_STEP_S}s")
    print(f"Output: {OUT_CSV}")
    print("Press Ctrl+C to stop.\n")

    next_t = time.monotonic()

    try:
        while True:
            now_m = time.monotonic()
            if now_m < next_t:
                time.sleep(next_t - now_m)
            next_t += WINDOW_SEC

            now = datetime.now()
            ts_str = now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

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

            cpu_util = resource.cpu_util()
            psi = resource.psi()

            stats_cg = {}
            for k, v in b["stats_cg"].items():
                stats_cg[int(k.value)] = {
                    "sched_lat_cnt": int(v.sched_lat_cnt),
                    "sched_lat_dropped": int(v.sched_lat_dropped),
                    "sched_lat_us_sum": int(v.sched_lat_us_sum),
                    "sched_lat_us_max": int(v.sched_lat_us_max),
                    "dstate_us_sum": int(v.dstate_us_sum),
                    "dstate_cnt": int(v.dstate_cnt),
                    "softirq_us_sum": int(v.softirq_us_sum),
                    "softirq_cnt": int(v.softirq_cnt),
                }
            b["stats_cg"].clear()

            pod_map = cg_index.get()

            # --- CSV Output ---
            with open(OUT_CSV, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "node",
                    ts_str,
                    "",
                    "",
                    f"{sched_total_ms:.2f}", f"{sched_avg_ms:.4f}", f"{sched_p95_ms:.4f}",
                    f"{sched_p99_ms:.4f}", f"{sched_max_ms:.4f}",
                    sched_cnt, sched_dropped,
                    f"{dstate_total_ms:.2f}", dstate_cnt,
                    f"{softirq_total_ms:.2f}", softirq_cnt,
                    f"{cpu_util:.6f}" if cpu_util is not None else "",
                    f"{psi:.6f}" if psi is not None else "",
                ])

                for cgid, (pod_uid, cg_path) in sorted(pod_map.items(), key=lambda x: x[1][0]):
                    v = stats_cg.get(cgid)
                    if v:
                        sched_cnt_p = v["sched_lat_cnt"]
                        sched_dropped_p = v["sched_lat_dropped"]
                        sched_total_ms_p = v["sched_lat_us_sum"] / 1000.0
                        sched_max_ms_p = v["sched_lat_us_max"] / 1000.0
                        sched_avg_ms_p = (sched_total_ms_p / sched_cnt_p) if sched_cnt_p else 0.0
                        dstate_total_ms_p = v["dstate_us_sum"] / 1000.0
                        dstate_cnt_p = v["dstate_cnt"]
                        softirq_total_ms_p = v["softirq_us_sum"] / 1000.0
                        softirq_cnt_p = v["softirq_cnt"]
                    else:
                        sched_cnt_p = sched_dropped_p = 0
                        sched_total_ms_p = sched_avg_ms_p = sched_max_ms_p = 0.0
                        dstate_total_ms_p = 0.0
                        dstate_cnt_p = 0
                        softirq_total_ms_p = 0.0
                        softirq_cnt_p = 0

                    cpu_p = cg_resource.cpu_util(cg_path)
                    psi_p = cg_resource.psi(cg_path)

                    writer.writerow([
                        "pod",
                        ts_str,
                        pod_uid,
                        cgid,
                        f"{sched_total_ms_p:.2f}", f"{sched_avg_ms_p:.4f}", "",
                        "", f"{sched_max_ms_p:.4f}",
                        sched_cnt_p, sched_dropped_p,
                        f"{dstate_total_ms_p:.2f}", dstate_cnt_p,
                        f"{softirq_total_ms_p:.2f}", softirq_cnt_p,
                        f"{cpu_p:.6f}" if cpu_p is not None else "",
                        f"{psi_p:.6f}" if psi_p is not None else "",
                    ])

    except KeyboardInterrupt:
        print("\nStopping. Outputs saved:")
        print(f" - {OUT_CSV}")
