#!/usr/bin/env python3
import csv
import glob
import json
import math
import os
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional

import uvicorn
from fastapi import Body, FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse

from .config import MAHAL_MIN_SAMPLES
from .energy import AdaptiveVolatilityEnergy
from .friction import mahalanobis_distance_and_direction
from .fuzzy_controller import FuzzyConfig, FuzzyController


USAGE_API_FIELDS = [
    "timestamp",
    "cpu_usage_pct",
    "cpu_psi_some_pct",
    "ram_usage_pct",
    "ram_usage_mi",
    "disk_used_pct",
    "sched_total_ms",
    "dstate_total_ms",
    "softirq_total_ms",
]


def _to_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_ts(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    s = str(value).strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _get_col(row: Dict[str, str], names: List[str]) -> Optional[str]:
    for key in names:
        if key in row:
            return row[key]
    lowered = {k.lower(): k for k in row.keys()}
    for key in names:
        source = lowered.get(key.lower())
        if source is not None:
            return row[source]
    return None


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _as_resource_int(value: Optional[float]) -> int:
    if value is None:
        return 0
    return int(max(0.0, math.floor(float(value))))


def _percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    vals = sorted(values)
    idx = int(round((p / 100.0) * (len(vals) - 1)))
    idx = max(0, min(idx, len(vals) - 1))
    return vals[idx]


def _std(values: List[float]) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return 0.0
    m = sum(values) / float(len(values))
    var = sum((v - m) * (v - m) for v in values) / float(len(values))
    return max(0.0, var) ** 0.5


def _round_or_none(value: Optional[float], digits: int) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def _csv_path_writable(path: str) -> bool:
    parent = os.path.dirname(path) or "."
    try:
        os.makedirs(parent, exist_ok=True)
    except OSError:
        return False

    # If the target exists, verify append permissions directly.
    if os.path.exists(path):
        try:
            with open(path, "a", encoding="utf-8"):
                pass
            return True
        except OSError:
            return False

    # Otherwise, verify directory writability via a probe file.
    probe = os.path.join(parent, f".ksense_write_probe_{os.getpid()}_{time.time_ns()}")
    try:
        with open(probe, "w", encoding="utf-8"):
            pass
        os.remove(probe)
        return True
    except OSError:
        return False


def _resolve_writable_csv_path(path: str, label: str, fallback_dir: str = "/tmp/ksense") -> str:
    requested = (path or "").strip()
    if not requested:
        requested = os.path.join(fallback_dir, f"{label}.csv")

    if _csv_path_writable(requested):
        return requested

    fallback = os.path.join(fallback_dir, os.path.basename(requested) or f"{label}.csv")
    if _csv_path_writable(fallback):
        print(f"[resource-offer] warning: {label} path not writable: {requested}; using {fallback}")
        return fallback

    raise OSError(f"{label} CSV path is not writable: {requested}; fallback also not writable: {fallback}")


# Lightweight ARIMA(1,1,0)-style forecast on differenced series.
def _forecast_arima_like(values: List[float]) -> Optional[float]:
    if not values:
        return None
    series = values[-60:]
    if len(series) == 1:
        return series[-1]
    if len(series) == 2:
        return series[-1] + (series[-1] - series[-2])

    diffs = [series[i] - series[i - 1] for i in range(1, len(series))]
    if not diffs:
        return series[-1]

    if len(diffs) == 1:
        return series[-1] + diffs[-1]

    num = 0.0
    den = 0.0
    for i in range(1, len(diffs)):
        num += diffs[i] * diffs[i - 1]
        den += diffs[i - 1] * diffs[i - 1]

    phi = (num / den) if den > 1e-9 else 0.0
    phi = _clamp(phi, -0.98, 0.98)

    mu = sum(diffs) / float(len(diffs))
    pred_diff = mu + phi * (diffs[-1] - mu)
    return series[-1] + pred_diff


@dataclass
class NodeCapacity:
    cpu_cores: float = float(os.getenv("MZ_TOTAL_CPU_CORES", "256"))
    ram_gb: float = float(os.getenv("MZ_TOTAL_RAM_GB", "2048"))
    storage_gb: float = float(os.getenv("MZ_TOTAL_STORAGE_GB", "80078"))
    gpu_count: float = float(os.getenv("MZ_TOTAL_GPU", "0"))


@dataclass
class UsageSample:
    ts: datetime
    cpu_pct: Optional[float]
    psi_pct: Optional[float]
    ram_gb: Optional[float]
    disk_pct: Optional[float]
    sched_ms: Optional[float]
    dstate_ms: Optional[float]
    softirq_ms: Optional[float]
    ebpf_total_ms: Optional[float]


@dataclass
class ControllerMetricRow:
    ts: datetime
    friction: Optional[float]
    energy: Optional[float]
    direction: Optional[float]
    cpu: Optional[float]
    psi: Optional[float]


class ControllerMetricsReplay:
    def __init__(self, csv_path: str, loop: bool = True):
        self.csv_path = csv_path
        self.loop = loop
        self._rows = self._load_rows(csv_path)
        self._idx = 0
        self._lock = threading.Lock()
        self._last: Optional[ControllerMetricRow] = None

    def _row_to_metrics(self, row: ControllerMetricRow) -> Dict[str, object]:
        fric = row.friction
        direction = row.direction
        if fric is not None and direction is not None:
            fric = fric * direction
        return {
            "fric": fric,
            "eng": row.energy,
            "cpu": row.cpu,
            "psi": row.psi,
            "last_direction": row.direction,
            "fric_short_p99": None,
            "fric_long_p99": None,
            "eng_short_p99": None,
            "eng_long_p99": None,
        }

    def _load_rows(self, path: str) -> List[ControllerMetricRow]:
        rows: List[ControllerMetricRow] = []
        if not os.path.exists(path):
            return rows
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                ts = _parse_ts(r.get("Time"))
                if ts is None:
                    continue
                fr = _to_float(r.get("Friction"))
                en = _to_float(r.get("Energy"))
                dr = _to_float(r.get("Direction"))
                cpu = _to_float(r.get("CPUUtil"))
                psi = _to_float(r.get("PSI"))
                # Replay should drive fuzzy with fully populated controller rows.
                if fr is None or en is None:
                    continue
                if dr is None:
                    dr = 1.0
                rows.append(
                    ControllerMetricRow(
                        ts=ts,
                        friction=fr,
                        energy=en,
                        direction=dr,
                        cpu=cpu,
                        psi=psi,
                    )
                )
        return rows

    @property
    def size(self) -> int:
        return len(self._rows)

    def step(self) -> Optional[Dict[str, object]]:
        if not self._rows:
            return None
        with self._lock:
            row = self._rows[self._idx]
            self._last = row
            if self._idx < len(self._rows) - 1:
                self._idx += 1
            elif self.loop:
                self._idx = 0
        return self._row_to_metrics(row)

    def current_metrics(self) -> Optional[Dict[str, object]]:
        with self._lock:
            row = self._last
        if row is None:
            return None
        return self._row_to_metrics(row)

    def next_metrics(self) -> Optional[Dict[str, object]]:
        return self.step()

    def latest(self) -> Optional[Dict[str, object]]:
        with self._lock:
            row = self._last
        if row is None:
            return None
        return {
            "timestamp": row.ts.strftime("%Y-%m-%d %H:%M:%S"),
            "friction": row.friction,
            "energy": row.energy,
            "direction": row.direction,
            "cpu": row.cpu,
            "psi": row.psi,
        }


class EmulatedUsageHistory:
    def __init__(self, csv_paths: List[str], window_s: int, max_rows_per_file: int, total_ram_gb: float):
        self.csv_paths = csv_paths
        self.window_s = window_s
        self.max_rows_per_file = max_rows_per_file
        self.total_ram_gb = total_ram_gb

    def _tail_rows(self, path: str) -> List[Dict[str, str]]:
        buf = deque(maxlen=self.max_rows_per_file)
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                buf.append(row)
        return list(buf)

    def _row_to_sample(self, row: Dict[str, str]) -> Optional[UsageSample]:
        ts = _parse_ts(_get_col(row, ["timestamp", "Time"]))
        if ts is None:
            return None

        cpu_pct = _to_float(_get_col(row, ["cpu_usage_pct", "CPUUtil"]))
        psi_pct = _to_float(_get_col(row, ["cpu_psi_some_pct", "PSI"]))

        ram_mi = _to_float(_get_col(row, ["ram_usage_mi"]))
        ram_pct = _to_float(_get_col(row, ["ram_usage_pct"]))
        ram_gb = None
        if ram_mi is not None:
            ram_gb = ram_mi / 1024.0
        elif ram_pct is not None:
            ram_gb = (self.total_ram_gb * ram_pct) / 100.0

        disk_pct = _to_float(_get_col(row, ["disk_used_pct"]))

        sched_ms = _to_float(_get_col(row, ["sched_total_ms", "SchedLat_Total_ms"]))
        dstate_ms = _to_float(_get_col(row, ["dstate_total_ms", "DState_Total_ms"]))
        softirq_ms = _to_float(_get_col(row, ["softirq_total_ms", "SoftIRQ_Total_ms"]))

        ebpf_parts = [v for v in (sched_ms, dstate_ms, softirq_ms) if v is not None]
        ebpf_total_ms = sum(ebpf_parts) if ebpf_parts else None

        if cpu_pct is None and psi_pct is None and ram_gb is None and disk_pct is None and ebpf_total_ms is None:
            return None

        return UsageSample(
            ts=ts,
            cpu_pct=cpu_pct,
            psi_pct=psi_pct,
            ram_gb=ram_gb,
            disk_pct=disk_pct,
            sched_ms=sched_ms,
            dstate_ms=dstate_ms,
            softirq_ms=softirq_ms,
            ebpf_total_ms=ebpf_total_ms,
        )

    def read_recent(self) -> List[UsageSample]:
        samples: List[UsageSample] = []

        for path in self.csv_paths:
            if not os.path.exists(path):
                continue
            try:
                rows = self._tail_rows(path)
            except Exception:
                continue
            for row in rows:
                sample = self._row_to_sample(row)
                if sample is not None:
                    samples.append(sample)

        if not samples:
            return []

        samples.sort(key=lambda x: x.ts)
        newest = samples[-1].ts
        cutoff = newest - timedelta(seconds=self.window_s)
        return [s for s in samples if s.ts >= cutoff]


class UsageApiMirror:
    def __init__(self, url: str, out_csv: str):
        self.url = url
        self.out_csv = out_csv
        self._last_ts: Optional[str] = None
        self._latest_payload: Optional[Dict[str, object]] = None
        self._lock = threading.Lock()
        self._ensure_csv()

    def _ensure_csv(self) -> None:
        out_dir = os.path.dirname(self.out_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        if os.path.exists(self.out_csv):
            return
        with open(self.out_csv, "w", encoding="utf-8", newline="") as f:
            f.write(",".join(USAGE_API_FIELDS) + "\n")

    def _fetch(self) -> Optional[Dict[str, object]]:
        req = urllib.request.Request(self.url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=2.0) as r:
                data = json.loads(r.read().decode("utf-8"))
                if not isinstance(data, dict):
                    return None
                return data
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
            return None

    def poll_once(self) -> bool:
        payload = self._fetch()
        if payload is None:
            return False

        ts = str(payload.get("timestamp") or "").strip()
        if not ts:
            return False
        with self._lock:
            self._latest_payload = dict(payload)
            if ts == self._last_ts:
                return False

        row = [ts]
        for key in USAGE_API_FIELDS[1:]:
            val = payload.get(key)
            if val is None:
                row.append("")
                continue
            row.append(str(val))

        with open(self.out_csv, "a", encoding="utf-8", newline="") as f:
            f.write(",".join(row) + "\n")

        with self._lock:
            self._last_ts = ts
        return True

    def latest(self) -> Optional[Dict[str, object]]:
        with self._lock:
            if self._latest_payload is None:
                return None
            return dict(self._latest_payload)


class ApiControllerMetricsWriter:
    """
    Builds controller metrics (friction/energy/direction/cpu/psi) from usage API samples
    and appends them to kernel_metrics.csv for fuzzy controller consumption.
    """

    _CSV_HEADER = [
        "Time",
        "SchedLat_Total_ms",
        "SchedLat_Avg_ms",
        "SchedLat_P95_ms",
        "SchedLat_P99_ms",
        "SchedLat_Max_ms",
        "SchedLat_Count",
        "SchedLat_Dropped",
        "DState_Total_ms",
        "DState_Count",
        "SoftIRQ_Total_ms",
        "SoftIRQ_Count",
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

    def __init__(self, out_csv: str, baseline_samples: int = 20):
        self.out_csv = out_csv
        self.baseline_samples = max(int(MAHAL_MIN_SAMPLES), int(baseline_samples))
        self._lock = threading.Lock()
        self._baseline_buf: deque = deque(maxlen=self.baseline_samples)
        self._baseline_frozen: Optional[List[List[float]]] = None
        self._energy = AdaptiveVolatilityEnergy()
        self._last_friction: Optional[float] = None
        self._last_ts: Optional[datetime] = None
        self._latest: Optional[Dict[str, float]] = None
        self._ensure_csv()

    def _ensure_csv(self) -> None:
        out_dir = os.path.dirname(self.out_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        if os.path.exists(self.out_csv):
            return
        with open(self.out_csv, "w", encoding="utf-8", newline="") as f:
            f.write(",".join(self._CSV_HEADER) + "\n")

    def baseline_state(self) -> Dict[str, object]:
        with self._lock:
            if self._baseline_frozen is None:
                mode = "CALIBRATING"
                samples = len(self._baseline_buf)
                ready = False
            else:
                mode = "FROZEN"
                samples = len(self._baseline_frozen)
                ready = samples >= self.baseline_samples
        return {
            "mode": mode,
            "samples": int(samples),
            "target_samples": int(self.baseline_samples),
            "ready": bool(ready),
        }

    def observe_sample(self, sample: UsageSample) -> None:
        if sample is None:
            return

        sched_total_ms = float(sample.sched_ms or 0.0)
        dstate_total_ms = float(sample.dstate_ms or 0.0)
        softirq_total_ms = float(sample.softirq_ms or 0.0)
        cpu_util = float(sample.cpu_pct or 0.0)
        psi = float(sample.psi_pct or 0.0)

        x_t = [
            sched_total_ms,
            sched_total_ms,  # sched_avg proxy
            dstate_total_ms,
            softirq_total_ms,
        ]

        with self._lock:
            if self._baseline_frozen is None:
                self._baseline_buf.append(list(x_t))
                if len(self._baseline_buf) >= self.baseline_samples:
                    self._baseline_frozen = list(self._baseline_buf)

            if self._baseline_frozen is None:
                friction = 0.0
                direction = 1.0
                baseline_mode = "CALIBRATING"
                baseline_samples = len(self._baseline_buf)
            else:
                fr, dr = mahalanobis_distance_and_direction(x_t, self._baseline_frozen)
                friction = float(fr if math.isfinite(fr) else 0.0)
                direction = float(dr if math.isfinite(dr) else 1.0)
                baseline_mode = "FROZEN"
                baseline_samples = len(self._baseline_frozen)

            if self._last_friction is not None and self._last_ts is not None:
                dt = max(1e-6, (sample.ts - self._last_ts).total_seconds())
                d_f_dt = (friction - self._last_friction) / dt
            else:
                d_f_dt = 0.0

            energy, w = self._energy.update(friction)
            if not math.isfinite(energy):
                energy = 0.0

            self._last_friction = friction
            self._last_ts = sample.ts
            self._latest = {
                "sched_total_ms": sched_total_ms,
                "dstate_total_ms": dstate_total_ms,
                "softirq_total_ms": softirq_total_ms,
                "cpu_util": cpu_util,
                "psi": psi,
                "baseline_mode": baseline_mode,
                "baseline_samples": float(baseline_samples),
                "friction": friction,
                "direction": direction,
                "d_f_dt": d_f_dt,
                "energy": float(energy),
                "energy_w": float(w),
                "energy_vol": float(self._energy.vol),
                "energy_k": float(self._energy.k_factor),
                "source_ts": sample.ts.timestamp(),
            }

    def write_latest_row(self, now_dt: datetime) -> bool:
        with self._lock:
            latest = dict(self._latest) if self._latest is not None else None
        if latest is None:
            return False

        ts_str = now_dt.strftime("%Y-%m-%d %H:%M:%S")
        sched_total_ms = latest["sched_total_ms"]
        dstate_total_ms = latest["dstate_total_ms"]
        softirq_total_ms = latest["softirq_total_ms"]
        cpu_util = latest["cpu_util"]
        psi = latest["psi"]
        baseline_mode = latest["baseline_mode"]
        baseline_samples = int(latest["baseline_samples"])
        friction = latest["friction"]
        direction = latest["direction"]
        d_f_dt = latest["d_f_dt"]
        energy = latest["energy"]
        energy_w = int(latest["energy_w"])
        energy_vol = latest["energy_vol"]
        energy_k = latest["energy_k"]

        with open(self.out_csv, "a", encoding="utf-8", newline="") as f:
            f.write(
                ",".join(
                    [
                        ts_str,
                        f"{sched_total_ms:.2f}",
                        f"{sched_total_ms:.4f}",
                        f"{sched_total_ms:.4f}",
                        f"{sched_total_ms:.4f}",
                        f"{sched_total_ms:.4f}",
                        "1",
                        "0",
                        f"{dstate_total_ms:.2f}",
                        "1",
                        f"{softirq_total_ms:.2f}",
                        "1",
                        f"{cpu_util:.6f}",
                        f"{psi:.6f}",
                        baseline_mode,
                        str(baseline_samples),
                        f"{friction:.6f}",
                        f"{direction:.1f}",
                        f"{d_f_dt:.6f}",
                        f"{energy:.6f}",
                        str(energy_w),
                        f"{energy_vol:.6f}",
                        f"{energy_k:.6f}",
                    ]
                )
                + "\n"
            )
        return True


class ResourceOfferEngine:
    def __init__(
        self,
        controller: FuzzyController,
        history: EmulatedUsageHistory,
        capacity: NodeCapacity,
        offer_csv_path: str,
        controller_replay: Optional[ControllerMetricsReplay] = None,
        get_current_usage: Optional[Callable[[], Optional[Dict[str, object]]]] = None,
        controller_writer: Optional[ApiControllerMetricsWriter] = None,
    ):
        self.controller = controller
        self.history = history
        self.capacity = capacity
        self.offer_csv_path = offer_csv_path
        self.controller_replay = controller_replay
        self.get_current_usage = get_current_usage
        self.controller_writer = controller_writer
        self._lock = threading.Lock()
        self._latest_snapshot: Optional[Dict[str, object]] = None
        self._reservations: List[Dict[str, object]] = []
        self._reservation_seq = 0
        self._tick_usage_override: Optional[Dict[str, object]] = None
        self._reservations_enabled = os.getenv("MZ_ENABLE_RESERVATIONS", "false").strip().lower() == "true"
        self._guard_mode = os.getenv("MZ_GUARD_BAND_MODE", "percent").strip().lower()
        self._guard_cpu = max(0.0, float(os.getenv("MZ_GUARD_BAND_CPU", "0")))
        self._guard_ram_gb = max(0.0, float(os.getenv("MZ_GUARD_BAND_RAM_GB", "0")))
        self._guard_storage_gb = max(0.0, float(os.getenv("MZ_GUARD_BAND_STORAGE_GB", "0")))
        self._guard_pct_min = _clamp(float(os.getenv("MZ_GUARD_BAND_PCT_MIN", "0.05")), 0.0, 1.0)
        self._guard_pct_max = _clamp(float(os.getenv("MZ_GUARD_BAND_PCT_MAX", "0.10")), 0.0, 1.0)
        if self._guard_pct_max < self._guard_pct_min:
            self._guard_pct_min, self._guard_pct_max = self._guard_pct_max, self._guard_pct_min
        self._default_reservation_ttl_s = float(os.getenv("MZ_RESERVATION_TTL_S", "60"))
        self._min_sellable_cpu = float(os.getenv("MZ_MIN_SELLABLE_CPU", "1"))
        self._min_sellable_ram_gb = float(os.getenv("MZ_MIN_SELLABLE_RAM_GB", "1"))
        self._block_until_baseline = os.getenv("MZ_BLOCK_DURING_BASELINE", "true").strip().lower() == "true"
        self._ensure_offer_csv()

    def _set_tick_usage_override(self, usage: Optional[Dict[str, object]]) -> None:
        with self._lock:
            self._tick_usage_override = dict(usage) if isinstance(usage, dict) else None

    def _get_tick_usage_override(self) -> Optional[Dict[str, object]]:
        with self._lock:
            if self._tick_usage_override is None:
                return None
            return dict(self._tick_usage_override)

    def _usage_payload_to_sample(self, payload: Optional[Dict[str, object]]) -> Optional[UsageSample]:
        if not isinstance(payload, dict):
            return None
        ts = _parse_ts(payload.get("timestamp"))
        if ts is None:
            return None

        cpu_pct = _to_float(payload.get("cpu_usage_pct"))
        psi_pct = _to_float(payload.get("cpu_psi_some_pct"))

        ram_mi = _to_float(payload.get("ram_usage_mi"))
        ram_pct = _to_float(payload.get("ram_usage_pct"))
        ram_gb = None
        if ram_mi is not None:
            ram_gb = ram_mi / 1024.0
        elif ram_pct is not None:
            ram_gb = (self.capacity.ram_gb * ram_pct) / 100.0

        disk_pct = _to_float(payload.get("disk_used_pct"))
        sched_ms = _to_float(payload.get("sched_total_ms"))
        dstate_ms = _to_float(payload.get("dstate_total_ms"))
        softirq_ms = _to_float(payload.get("softirq_total_ms"))
        ebpf_parts = [v for v in (sched_ms, dstate_ms, softirq_ms) if v is not None]
        ebpf_total_ms = sum(ebpf_parts) if ebpf_parts else None

        if cpu_pct is None and psi_pct is None and ram_gb is None and disk_pct is None and ebpf_total_ms is None:
            return None

        return UsageSample(
            ts=ts,
            cpu_pct=cpu_pct,
            psi_pct=psi_pct,
            ram_gb=ram_gb,
            disk_pct=disk_pct,
            sched_ms=sched_ms,
            dstate_ms=dstate_ms,
            softirq_ms=softirq_ms,
            ebpf_total_ms=ebpf_total_ms,
        )

    def _ensure_offer_csv(self) -> None:
        out_dir = os.path.dirname(self.offer_csv_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        if os.path.exists(self.offer_csv_path):
            return
        with open(self.offer_csv_path, "w", encoding="utf-8", newline="") as f:
            f.write(
                "Time,Decision,Score,Level,SellableCPU,SellableRAM_GB,SellableGPU,SellableStorage_GB,"
                "PredCPU_pct,PredPSI_pct,PredRAM_GB,PredDisk_pct,SafetyMultiplier\n"
            )

    def _predict(self, samples: List[UsageSample]) -> Dict[str, Optional[float]]:
        cpu_hist = [x.cpu_pct for x in samples if x.cpu_pct is not None]
        psi_hist = [x.psi_pct for x in samples if x.psi_pct is not None]
        ram_hist = [x.ram_gb for x in samples if x.ram_gb is not None]
        disk_hist = [x.disk_pct for x in samples if x.disk_pct is not None]
        ebpf_hist = [x.ebpf_total_ms for x in samples if x.ebpf_total_ms is not None]

        def _forecast_conservative(hist: List[float]) -> Optional[float]:
            if not hist:
                return None
            pred = _forecast_arima_like(hist)
            if pred is None:
                pred = hist[-1]
            # Never reserve less than current observed pressure.
            return max(float(pred), float(hist[-1]))

        pred_cpu = _forecast_conservative(cpu_hist)
        pred_psi = _forecast_conservative(psi_hist)
        pred_ram = _forecast_conservative(ram_hist)
        pred_disk = _forecast_conservative(disk_hist)
        pred_ebpf = _forecast_conservative(ebpf_hist)

        cpu_std = _std(cpu_hist)
        psi_std = _std(psi_hist)
        ram_std = _std(ram_hist)
        disk_std = _std(disk_hist)

        return {
            "cpu_pct": _clamp(float(pred_cpu), 0.0, 100.0) if pred_cpu is not None else None,
            "psi_pct": _clamp(float(pred_psi), 0.0, 100.0) if pred_psi is not None else None,
            "ram_gb": max(0.0, float(pred_ram)) if pred_ram is not None else None,
            "disk_pct": _clamp(float(pred_disk), 0.0, 100.0) if pred_disk is not None else None,
            "ebpf_ms": max(0.0, float(pred_ebpf)) if pred_ebpf is not None else None,
            "ebpf_p95": _percentile(ebpf_hist, 95.0),
            "cpu_std": cpu_std,
            "psi_std": psi_std,
            "ram_std": ram_std,
            "disk_std": disk_std,
        }

    def _safety_multiplier(self, fuzzy_report: Dict[str, object], pred: Dict[str, Optional[float]]) -> float:
        # Score-driven risk (continuous), no static level buckets.
        score = _to_float(fuzzy_report.get("score")) or 0.0
        fuzzy_risk = _clamp((score - 45.0) / 55.0, 0.0, 1.0)
        fuzzy_margin = 0.20 * fuzzy_risk

        # PSI pressure directly increases reserve.
        psi_pct = float(pred.get("psi_pct") or 0.0)
        psi_margin = 0.18 * _clamp(psi_pct / 100.0, 0.0, 1.0)

        # Volatility margin from recent history dispersion (CV-like).
        cpu_pct = float(pred.get("cpu_pct") or 0.0)
        ram_gb = float(pred.get("ram_gb") or 0.0)
        disk_pct = float(pred.get("disk_pct") or 0.0)
        cpu_cv = float(pred.get("cpu_std") or 0.0) / max(1.0, cpu_pct)
        ram_cv = float(pred.get("ram_std") or 0.0) / max(1.0, ram_gb)
        disk_cv = float(pred.get("disk_std") or 0.0) / max(1.0, disk_pct)
        psi_cv = float(pred.get("psi_std") or 0.0) / max(1.0, psi_pct)
        volatility = (0.40 * cpu_cv) + (0.25 * ram_cv) + (0.20 * psi_cv) + (0.15 * disk_cv)
        volatility_margin = min(0.20, max(0.0, volatility) * 0.20)

        # eBPF surge margin vs recent p95.
        ebpf_margin = 0.0
        ebpf_ms = pred.get("ebpf_ms")
        ebpf_p95 = pred.get("ebpf_p95")
        if ebpf_ms is not None and ebpf_p95 is not None and ebpf_p95 > 0.0:
            ratio = (float(ebpf_ms) / float(ebpf_p95)) - 1.0
            ebpf_margin = min(0.15, max(0.0, ratio) * 0.15)

        multiplier = 1.0 + fuzzy_margin + psi_margin + volatility_margin + ebpf_margin
        return _clamp(multiplier, 1.0, 1.65)

    def _prune_expired_reservations_locked(self, now_dt: datetime) -> None:
        kept: List[Dict[str, object]] = []
        for r in self._reservations:
            exp = r.get("expires_at")
            if isinstance(exp, datetime) and exp > now_dt:
                kept.append(r)
        self._reservations = kept

    def _reserved_totals_locked(self) -> Dict[str, float]:
        cpu = 0.0
        ram = 0.0
        storage = 0.0
        for r in self._reservations:
            cpu += float(r.get("cpu", 0.0) or 0.0)
            ram += float(r.get("ram", 0.0) or 0.0)
            storage += float(r.get("storage", 0.0) or 0.0)
        return {
            "cpu": round(max(0.0, cpu), 2),
            "ram": round(max(0.0, ram), 2),
            "storage": round(max(0.0, storage), 2),
        }

    def _apply_reservations(self, offer: Dict[str, float], reserved: Dict[str, float]) -> Dict[str, float]:
        out = dict(offer)
        out["cpu"] = _as_resource_int(float(out.get("cpu", 0.0) or 0.0) - reserved["cpu"])
        out["ram"] = _as_resource_int(float(out.get("ram", 0.0) or 0.0) - reserved["ram"])
        out["storage"] = _as_resource_int(float(out.get("storage", 0.0) or 0.0) - reserved["storage"])
        out["GPU"] = _as_resource_int(float(out.get("GPU", 0.0) or 0.0))
        return out

    def _effective_guard_band(self, score: float) -> Dict[str, float]:
        # Default mode: guard band is 5%-10% of node capacity, adapting with risk score.
        if self._guard_mode in ("absolute", "fixed"):
            return {
                "cpu": self._guard_cpu,
                "ram": self._guard_ram_gb,
                "storage": self._guard_storage_gb,
                "pct": None,
                "mode": "absolute",
            }

        risk = _clamp(score / 100.0, 0.0, 1.0)
        guard_pct = self._guard_pct_min + ((self._guard_pct_max - self._guard_pct_min) * risk)
        return {
            "cpu": self.capacity.cpu_cores * guard_pct,
            "ram": self.capacity.ram_gb * guard_pct,
            "storage": self.capacity.storage_gb * guard_pct,
            "pct": guard_pct,
            "mode": "percent",
        }

    def _compute_offer_snapshot(self) -> Dict[str, object]:
        # Tick barrier: freeze one usage snapshot for this 40s decision cycle.
        samples = self.history.read_recent()
        current_usage_payload = self.get_current_usage() if self.get_current_usage is not None else None
        current_usage_sample = self._usage_payload_to_sample(current_usage_payload)
        if current_usage_sample is not None:
            if not samples or samples[-1].ts != current_usage_sample.ts:
                samples = list(samples) + [current_usage_sample]
        latest_sample = current_usage_sample if current_usage_sample is not None else (samples[-1] if samples else None)
        oldest_sample = samples[0] if samples else None

        tick_usage_override = {
            "cpu_usage_pct": latest_sample.cpu_pct if latest_sample is not None else None,
            "cpu_psi_some_pct": latest_sample.psi_pct if latest_sample is not None else None,
        }

        if self.controller_writer is not None and latest_sample is not None:
            try:
                self.controller_writer.observe_sample(latest_sample)
                # Persist computed controller metrics once per decision tick.
                self.controller_writer.write_latest_row(datetime.now())
            except Exception:
                pass
        controller_baseline = (
            self.controller_writer.baseline_state() if self.controller_writer is not None else None
        )

        with self._lock:
            if self._reservations_enabled:
                self._prune_expired_reservations_locked(datetime.now())
                reserved = self._reserved_totals_locked()
                active_reservations = len(self._reservations)
            else:
                reserved = {"cpu": 0.0, "ram": 0.0, "storage": 0.0}
                active_reservations = 0

        usage_latest = {
            "timestamp": latest_sample.ts.strftime("%Y-%m-%d %H:%M:%S") if latest_sample is not None else None,
            "cpu_usage_pct": round(float(latest_sample.cpu_pct), 4)
            if latest_sample is not None and latest_sample.cpu_pct is not None
            else None,
            "cpu_psi_some_pct": round(float(latest_sample.psi_pct), 4)
            if latest_sample is not None and latest_sample.psi_pct is not None
            else None,
            "ram_usage_gb": round(float(latest_sample.ram_gb), 4)
            if latest_sample is not None and latest_sample.ram_gb is not None
            else None,
            "disk_used_pct": round(float(latest_sample.disk_pct), 4)
            if latest_sample is not None and latest_sample.disk_pct is not None
            else None,
        }

        ebpf_inputs = {
            "latest_sched_total_ms": _round_or_none(float(latest_sample.sched_ms), 3)
            if latest_sample is not None and latest_sample.sched_ms is not None
            else None,
            "latest_dstate_total_ms": _round_or_none(float(latest_sample.dstate_ms), 3)
            if latest_sample is not None and latest_sample.dstate_ms is not None
            else None,
            "latest_softirq_total_ms": _round_or_none(float(latest_sample.softirq_ms), 3)
            if latest_sample is not None and latest_sample.softirq_ms is not None
            else None,
            "latest_total_ms": _round_or_none(float(latest_sample.ebpf_total_ms), 3)
            if latest_sample is not None and latest_sample.ebpf_total_ms is not None
            else None,
            "predicted_total_ms": None,
            "window_p95_total_ms": None,
        }

        if (
            self.controller_writer is not None
            and self._block_until_baseline
            and isinstance(controller_baseline, dict)
            and not bool(controller_baseline.get("ready"))
        ):
            return {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "decision": "deny",
                "decision_reason": "baseline_warmup",
                "fuzzy_decision": "deny",
                "level": "warmup",
                "score": 0.0,
                "offer": {"cpu": 0, "ram": 0, "GPU": 0, "storage": 0},
                "offer_before_reservations": {"cpu": 0, "ram": 0, "GPU": 0, "storage": 0},
                "reserved_totals": reserved,
                "active_reservations": active_reservations,
                "min_sellable": {"cpu": self._min_sellable_cpu, "ram": self._min_sellable_ram_gb},
                "reservation_mode": "enabled" if self._reservations_enabled else "disabled",
                "guard_band": {
                    "mode": self._guard_mode,
                    "pct": None,
                    "cpu": 0.0,
                    "ram": 0.0,
                    "storage": 0.0,
                },
                "predicted": {
                    "cpu_pct": None,
                    "psi_pct": None,
                    "ram_gb": None,
                    "disk_pct": None,
                    "ebpf_total_ms": None,
                    "ebpf_total_p95_ms": None,
                    "cpu_std": None,
                    "psi_std": None,
                    "ram_std": None,
                    "disk_std": None,
                },
                "usage_window": {
                    "sample_count": len(samples),
                    "window_seconds": self.history.window_s,
                    "oldest_timestamp": oldest_sample.ts.strftime("%Y-%m-%d %H:%M:%S")
                    if oldest_sample is not None
                    else None,
                    "latest_timestamp": latest_sample.ts.strftime("%Y-%m-%d %H:%M:%S")
                    if latest_sample is not None
                    else None,
                },
                "usage_latest": usage_latest,
                "fuzzy_inputs": {
                    "friction_signed": None,
                    "energy_scaled": None,
                    "cpu_util": None,
                    "psi_1s": None,
                    "direction": None,
                },
                "ebpf_inputs": ebpf_inputs,
                "controller_metrics_latest": None,
                "controller_baseline": controller_baseline,
                "capacity_totals": {
                    "cpu": round(self.capacity.cpu_cores, 2),
                    "ram": round(self.capacity.ram_gb, 2),
                    "GPU": round(self.capacity.gpu_count, 2),
                    "storage": round(self.capacity.storage_gb, 2),
                },
                "safety_multiplier": None,
            }

        pred: Dict[str, Optional[float]] = {}
        fuzzy_report: Dict[str, object] = {"level": "medium", "score": 55.0, "decision": "deny"}

        def _run_predict() -> None:
            nonlocal pred
            pred = self._predict(samples)

        def _run_fuzzy() -> None:
            nonlocal fuzzy_report
            try:
                fuzzy_report = self.controller.decide()
            except Exception:
                fuzzy_report = {"level": "medium", "score": 55.0, "decision": "deny"}

        self._set_tick_usage_override(tick_usage_override)
        try:
            # Fuzzy and predictor run in parallel from the same frozen tick context.
            t_pred = threading.Thread(target=_run_predict)
            t_fuzzy = threading.Thread(target=_run_fuzzy)
            t_pred.start()
            t_fuzzy.start()
            t_pred.join()
            t_fuzzy.join()
        finally:
            self._set_tick_usage_override(None)

        fuzzy_decision = str(fuzzy_report.get("decision", "deny")).lower()
        level = str(fuzzy_report.get("level", "medium")).lower()
        score = float(fuzzy_report.get("score", 0.0))
        guard = self._effective_guard_band(score)

        safety_multiplier: Optional[float] = None
        decision = fuzzy_decision
        decision_reason = "fuzzy_reject"
        offer_before_reservations = {
            "cpu": 0,
            "ram": 0,
            "GPU": 0,
            "storage": 0,
        }

        if fuzzy_decision != "allow":
            offer = dict(offer_before_reservations)
        else:
            decision_reason = "fuzzy_allow"
            pred_cpu = pred.get("cpu_pct")
            pred_ram = pred.get("ram_gb")
            pred_disk = pred.get("disk_pct")
            if pred_cpu is None or pred_ram is None or pred_disk is None:
                decision = "deny"
                decision_reason = "insufficient_usage_history"
                offer = {
                    "cpu": 0,
                    "ram": 0,
                    "GPU": 0,
                    "storage": 0,
                }
            else:
                multiplier = self._safety_multiplier(fuzzy_report, pred)
                safety_multiplier = round(multiplier, 4)

                reserve_cpu = self.capacity.cpu_cores * (float(pred_cpu) / 100.0) * multiplier
                reserve_ram = float(pred_ram) * multiplier
                reserve_storage = self.capacity.storage_gb * (float(pred_disk) / 100.0) * multiplier

                sellable_cpu = max(0.0, self.capacity.cpu_cores - reserve_cpu - float(guard["cpu"]))
                sellable_ram = max(0.0, self.capacity.ram_gb - reserve_ram - float(guard["ram"]))
                sellable_storage = max(0.0, self.capacity.storage_gb - reserve_storage - float(guard["storage"]))

                offer_before_reservations = {
                    "cpu": _as_resource_int(sellable_cpu),
                    "ram": _as_resource_int(sellable_ram),
                    "GPU": _as_resource_int(self.capacity.gpu_count),
                    "storage": _as_resource_int(sellable_storage),
                }

                if self._reservations_enabled:
                    offer = self._apply_reservations(offer_before_reservations, reserved)
                else:
                    offer = dict(offer_before_reservations)

                # Final gate: even if fuzzy allows, sellability must stay deployable.
                if offer["cpu"] < self._min_sellable_cpu or offer["ram"] < self._min_sellable_ram_gb:
                    decision = "deny"
                    decision_reason = "capacity_guard"
                    offer = {
                        "cpu": 0,
                        "ram": 0,
                        "GPU": 0,
                        "storage": 0,
                    }
                else:
                    decision = "allow"
        fuzzy_metrics = fuzzy_report.get("metrics", {}) if isinstance(fuzzy_report.get("metrics"), dict) else {}
        fuzzy_inputs = {
            "friction_signed": round(float(fuzzy_metrics.get("friction_signed")), 6)
            if _to_float(fuzzy_metrics.get("friction_signed")) is not None
            else None,
            "energy_scaled": round(float(fuzzy_metrics.get("energy_scaled")), 6)
            if _to_float(fuzzy_metrics.get("energy_scaled")) is not None
            else None,
            "cpu_util": round(float(fuzzy_metrics.get("cpu_util")), 6)
            if _to_float(fuzzy_metrics.get("cpu_util")) is not None
            else None,
            "psi_1s": round(float(fuzzy_metrics.get("psi_1s")), 6)
            if _to_float(fuzzy_metrics.get("psi_1s")) is not None
            else None,
            "direction": fuzzy_metrics.get("direction"),
        }

        ebpf_inputs["predicted_total_ms"] = _round_or_none(pred.get("ebpf_ms"), 3)
        ebpf_inputs["window_p95_total_ms"] = _round_or_none(pred.get("ebpf_p95"), 3)
        controller_metrics_latest = (
            self.controller_replay.latest() if self.controller_replay is not None else None
        )

        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "decision": decision,
            "decision_reason": decision_reason,
            "fuzzy_decision": fuzzy_decision,
            "level": level,
            "score": round(score, 3),
            "offer": offer,
            "offer_before_reservations": offer_before_reservations,
            "reserved_totals": reserved,
            "active_reservations": active_reservations,
            "min_sellable": {
                "cpu": self._min_sellable_cpu,
                "ram": self._min_sellable_ram_gb,
            },
            "reservation_mode": "enabled" if self._reservations_enabled else "disabled",
            "guard_band": {
                "mode": guard["mode"],
                "pct": _round_or_none(guard.get("pct"), 4),
                "cpu": round(float(guard["cpu"]), 2),
                "ram": round(float(guard["ram"]), 2),
                "storage": round(float(guard["storage"]), 2),
            },
            "predicted": {
                "cpu_pct": _round_or_none(pred.get("cpu_pct"), 4),
                "psi_pct": _round_or_none(pred.get("psi_pct"), 4),
                "ram_gb": _round_or_none(pred.get("ram_gb"), 4),
                "disk_pct": _round_or_none(pred.get("disk_pct"), 4),
                "ebpf_total_ms": _round_or_none(pred.get("ebpf_ms"), 3),
                "ebpf_total_p95_ms": _round_or_none(pred.get("ebpf_p95"), 3),
                "cpu_std": _round_or_none(pred.get("cpu_std"), 4),
                "psi_std": _round_or_none(pred.get("psi_std"), 4),
                "ram_std": _round_or_none(pred.get("ram_std"), 4),
                "disk_std": _round_or_none(pred.get("disk_std"), 4),
            },
            "usage_window": {
                "sample_count": len(samples),
                "window_seconds": self.history.window_s,
                "oldest_timestamp": oldest_sample.ts.strftime("%Y-%m-%d %H:%M:%S")
                if oldest_sample is not None
                else None,
                "latest_timestamp": latest_sample.ts.strftime("%Y-%m-%d %H:%M:%S")
                if latest_sample is not None
                else None,
            },
            "usage_latest": usage_latest,
            "fuzzy_inputs": fuzzy_inputs,
            "ebpf_inputs": ebpf_inputs,
            "controller_metrics_latest": controller_metrics_latest,
            "controller_baseline": controller_baseline,
            "capacity_totals": {
                "cpu": round(self.capacity.cpu_cores, 2),
                "ram": round(self.capacity.ram_gb, 2),
                "GPU": round(self.capacity.gpu_count, 2),
                "storage": round(self.capacity.storage_gb, 2),
            },
            "safety_multiplier": safety_multiplier,
        }

    def _append_snapshot(self, snapshot: Dict[str, object]) -> None:
        offer = snapshot["offer"]
        pred = snapshot["predicted"]
        safety = snapshot["safety_multiplier"]
        safety_s = ""
        if safety is not None:
            safety_s = f"{float(safety):.4f}"
        pred_cpu = float(pred.get("cpu_pct") or 0.0)
        pred_psi = float(pred.get("psi_pct") or 0.0)
        pred_ram = float(pred.get("ram_gb") or 0.0)
        pred_disk = float(pred.get("disk_pct") or 0.0)
        with open(self.offer_csv_path, "a", encoding="utf-8", newline="") as f:
            f.write(
                f"{snapshot['timestamp']},{snapshot['decision']},{snapshot['score']:.3f},"
                f"{snapshot['level']},{int(offer['cpu'])},{int(offer['ram'])},{int(offer['GPU'])},"
                f"{int(offer['storage'])},{pred_cpu:.4f},{pred_psi:.4f},"
                f"{pred_ram:.4f},{pred_disk:.4f},{safety_s}\n"
            )

    def refresh_once(self) -> Dict[str, object]:
        snapshot = self._compute_offer_snapshot()
        with self._lock:
            self._latest_snapshot = snapshot
        self._append_snapshot(snapshot)
        return snapshot

    def get_latest_offer(self) -> Dict[str, float]:
        with self._lock:
            snapshot = self._latest_snapshot
        if snapshot is None:
            snapshot = self.refresh_once()
        return dict(snapshot["offer"])

    def get_latest_snapshot(self) -> Dict[str, object]:
        with self._lock:
            snapshot = self._latest_snapshot
        if snapshot is None:
            snapshot = self.refresh_once()
        return dict(snapshot)

    def compute_offer(self) -> Dict[str, float]:
        # Backward-compatible call site.
        return self.get_latest_offer()

    def reserve(
        self,
        cpu: float,
        ram: float,
        storage: float,
        ttl_s: Optional[float] = None,
        owner: str = "",
    ) -> Dict[str, object]:
        if not self._reservations_enabled:
            return {
                "accepted": False,
                "reason": "reservation_disabled",
                "error": "reservations are disabled in current emulation mode",
            }
        self.get_latest_snapshot()

        req_cpu = max(0.0, float(cpu or 0.0))
        req_ram = max(0.0, float(ram or 0.0))
        req_storage = max(0.0, float(storage or 0.0))
        if req_cpu <= 0.0 and req_ram <= 0.0 and req_storage <= 0.0:
            return {"accepted": False, "reason": "invalid_request", "error": "requested resources must be > 0"}

        ttl = float(ttl_s if ttl_s is not None else self._default_reservation_ttl_s)
        ttl = _clamp(ttl, 1.0, 3600.0)

        with self._lock:
            now_dt = datetime.now()
            self._prune_expired_reservations_locked(now_dt)
            snap = self._latest_snapshot if self._latest_snapshot is not None else {}
            decision = str(snap.get("decision", "deny")).lower()
            offer = dict(snap.get("offer", {})) if isinstance(snap.get("offer"), dict) else {}

            avail_cpu = float(offer.get("cpu", 0.0) or 0.0)
            avail_ram = float(offer.get("ram", 0.0) or 0.0)
            avail_storage = float(offer.get("storage", 0.0) or 0.0)

            if decision != "allow":
                return {
                    "accepted": False,
                    "reason": "not_sellable",
                    "decision": decision,
                    "available": {"cpu": avail_cpu, "ram": avail_ram, "storage": avail_storage},
                }

            if req_cpu > avail_cpu or req_ram > avail_ram or req_storage > avail_storage:
                return {
                    "accepted": False,
                    "reason": "insufficient_available",
                    "requested": {"cpu": req_cpu, "ram": req_ram, "storage": req_storage},
                    "available": {"cpu": avail_cpu, "ram": avail_ram, "storage": avail_storage},
                }

            self._reservation_seq += 1
            rid = f"res-{self._reservation_seq}"
            expires_at = now_dt + timedelta(seconds=ttl)
            self._reservations.append(
                {
                    "id": rid,
                    "owner": owner,
                    "cpu": req_cpu,
                    "ram": req_ram,
                    "storage": req_storage,
                    "created_at": now_dt,
                    "expires_at": expires_at,
                }
            )

            # Apply reservation immediately to cached offer to avoid double-selling before refresh.
            if isinstance(self._latest_snapshot, dict):
                latest_offer = dict(self._latest_snapshot.get("offer", {}))
                latest_offer["cpu"] = round(max(0.0, float(latest_offer.get("cpu", 0.0) or 0.0) - req_cpu), 2)
                latest_offer["ram"] = round(max(0.0, float(latest_offer.get("ram", 0.0) or 0.0) - req_ram), 2)
                latest_offer["storage"] = round(
                    max(0.0, float(latest_offer.get("storage", 0.0) or 0.0) - req_storage), 2
                )
                self._latest_snapshot["offer"] = latest_offer
                self._latest_snapshot["active_reservations"] = len(self._reservations)
                self._latest_snapshot["reserved_totals"] = self._reserved_totals_locked()

            return {
                "accepted": True,
                "reservation": {
                    "id": rid,
                    "owner": owner,
                    "cpu": round(req_cpu, 2),
                    "ram": round(req_ram, 2),
                    "storage": round(req_storage, 2),
                    "expires_at": expires_at.strftime("%Y-%m-%d %H:%M:%S"),
                },
                "remaining_offer": self._latest_snapshot.get("offer", {}) if isinstance(self._latest_snapshot, dict) else {},
            }

    def list_reservations(self) -> Dict[str, object]:
        if not self._reservations_enabled:
            return {
                "active_reservations": 0,
                "reserved_totals": {"cpu": 0.0, "ram": 0.0, "storage": 0.0},
                "items": [],
                "reservation_mode": "disabled",
            }
        with self._lock:
            self._prune_expired_reservations_locked(datetime.now())
            items = []
            for r in self._reservations:
                items.append(
                    {
                        "id": str(r.get("id", "")),
                        "owner": str(r.get("owner", "")),
                        "cpu": round(float(r.get("cpu", 0.0) or 0.0), 2),
                        "ram": round(float(r.get("ram", 0.0) or 0.0), 2),
                        "storage": round(float(r.get("storage", 0.0) or 0.0), 2),
                        "expires_at": r.get("expires_at").strftime("%Y-%m-%d %H:%M:%S")
                        if isinstance(r.get("expires_at"), datetime)
                        else "",
                    }
                )
            return {
                "active_reservations": len(items),
                "reserved_totals": self._reserved_totals_locked(),
                "items": items,
            }


def _default_controller_csv() -> str:
    candidates = [
        os.getenv("KSENSE_METRICS_CSV", "").strip(),
        os.path.join(os.getcwd(), "kernel_metrics.csv"),
        "/tmp/ksense/kernel_metrics.csv",
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return "/tmp/ksense/kernel_metrics.csv"


def _default_usage_csvs() -> List[str]:
    hotels = sorted(glob.glob(os.path.join(os.getcwd(), "hotel*.csv")))
    if hotels:
        return hotels

    fallback = _default_controller_csv()
    return [fallback] if fallback else []


def _create_app(engine: ResourceOfferEngine) -> FastAPI:
    app = FastAPI(title="ksense-resource-offer-api")

    @app.get("/healthz", response_class=PlainTextResponse)
    def healthz() -> str:
        return "ok\n"

    @app.get("/resource_offer_debug")
    def resource_offer_debug() -> Dict[str, object]:
        return engine.get_latest_snapshot()

    @app.get("/resource_offer_now")
    def resource_offer_now() -> Dict[str, object]:
        return engine.refresh_once()

    @app.get("/resource_offer")
    def resource_offer() -> Dict[str, float]:
        return engine.compute_offer()

    @app.get("/reservations")
    def reservations() -> Dict[str, object]:
        return engine.list_reservations()

    @app.post("/reserve")
    def reserve(payload: Dict[str, object] = Body(default_factory=dict)):
        cpu = _to_float(payload.get("cpu")) or 0.0
        ram = _to_float(payload.get("ram")) or 0.0
        storage = _to_float(payload.get("storage")) or 0.0
        ttl_s = _to_float(payload.get("ttl_s"))
        owner = str(payload.get("owner") or "").strip()

        result = engine.reserve(cpu=cpu, ram=ram, storage=storage, ttl_s=ttl_s, owner=owner)
        status = 201 if result.get("accepted") else 409
        return JSONResponse(status_code=status, content=result)

    return app


def run_server(host: str = "127.0.0.1", port: int = 8080) -> None:
    usage_api_url = os.getenv("MZ_USAGE_API_URL", "").strip()
    usage_api_poll_s = float(os.getenv("MZ_USAGE_API_POLL_S", "5"))
    usage_api_csv = os.getenv("MZ_USAGE_API_CSV", "/tmp/ksense/usage_api_metrics.csv").strip()
    usage_api_csv = _resolve_writable_csv_path(usage_api_csv, label="usage_api")

    usage_mirror = None
    if usage_api_url:
        usage_mirror = UsageApiMirror(url=usage_api_url, out_csv=usage_api_csv)
        usage_mirror.poll_once()
        usage_csvs = [usage_api_csv]
    else:
        usage_csvs_env = os.getenv("MZ_USAGE_CSVS", "").strip()
        if usage_csvs_env:
            usage_csvs = [x.strip() for x in usage_csvs_env.split(",") if x.strip()]
        else:
            usage_csvs = _default_usage_csvs()

    controller_csv = os.getenv("MZ_CONTROLLER_CSV", "").strip() or _default_controller_csv()
    replay_mode = os.getenv("MZ_CONTROLLER_REPLAY", "auto").strip().lower()
    controller_from_usage = os.getenv("MZ_CONTROLLER_FROM_USAGE_ENABLED", "true").strip().lower() == "true"
    controller_writer_baseline_samples = int(os.getenv("MZ_CONTROLLER_BASELINE_SAMPLES", "20"))
    if usage_api_url and controller_from_usage:
        controller_csv = _resolve_writable_csv_path(controller_csv, label="controller")

    window_s = int(os.getenv("MZ_PREDICT_WINDOW_S", "300"))
    max_rows_per_file = int(os.getenv("MZ_USAGE_MAX_ROWS", "20000"))
    offer_csv_path = os.getenv("MZ_RESOURCE_OFFER_CSV", "/tmp/ksense/resource_offer.csv")
    offer_csv_path = _resolve_writable_csv_path(offer_csv_path, label="resource_offer")
    refresh_s = float(os.getenv("MZ_OFFER_REFRESH_S", "40"))

    capacity = NodeCapacity()
    history = EmulatedUsageHistory(
        csv_paths=usage_csvs,
        window_s=window_s,
        max_rows_per_file=max_rows_per_file,
        total_ram_gb=capacity.ram_gb,
    )

    cfg = FuzzyConfig(csv_path=controller_csv, rules_enabled=False)
    controller = FuzzyController(cfg=cfg)
    controller_replay: Optional[ControllerMetricsReplay] = None

    replay_enabled = False
    if replay_mode in ("1", "true", "yes", "on"):
        replay_enabled = True
    elif replay_mode in ("0", "false", "no", "off"):
        replay_enabled = False
    else:
        # Auto mode:
        # - If usage API is enabled and controller-from-usage writer is enabled, do not replay.
        # - Otherwise preserve old behavior (replay for emulated usage-api flows).
        replay_enabled = bool(usage_api_url) and (not controller_from_usage)

    controller_writer: Optional[ApiControllerMetricsWriter] = None
    if usage_api_url and controller_from_usage:
        controller_writer = ApiControllerMetricsWriter(
            out_csv=controller_csv,
            baseline_samples=controller_writer_baseline_samples,
        )

    if replay_enabled:
        replay_loop = os.getenv("MZ_CONTROLLER_REPLAY_LOOP", "true").strip().lower() == "true"
        controller_replay = ControllerMetricsReplay(csv_path=controller_csv, loop=replay_loop)
        if controller_replay.size > 0:
            controller_replay.step()

    engine = ResourceOfferEngine(
        controller=controller,
        history=history,
        capacity=capacity,
        offer_csv_path=offer_csv_path,
        controller_replay=controller_replay,
        get_current_usage=usage_mirror.latest if usage_mirror is not None else None,
        controller_writer=controller_writer,
    )

    if replay_enabled and controller_replay is not None and controller_replay.size > 0:
        original_collect = controller._collect_metrics

        def _inject_collect_metrics():
            m = controller_replay.current_metrics()
            if m is None:
                return original_collect()
            usage_override = engine._get_tick_usage_override()
            if isinstance(usage_override, dict):
                usage_cpu = _to_float(usage_override.get("cpu_usage_pct"))
                usage_psi = _to_float(usage_override.get("cpu_psi_some_pct"))
                if usage_cpu is not None:
                    m["cpu"] = usage_cpu
                if usage_psi is not None:
                    m["psi"] = usage_psi
            return m

        controller._collect_metrics = _inject_collect_metrics  # type: ignore[attr-defined]

    def _usage_api_loop() -> None:
        if usage_mirror is None:
            return
        next_t = time.monotonic() + usage_api_poll_s
        while True:
            sleep_s = next_t - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            try:
                changed = usage_mirror.poll_once()
                if changed and controller_writer is not None:
                    payload = usage_mirror.latest()
                    sample = engine._usage_payload_to_sample(payload)
                    if sample is not None:
                        controller_writer.observe_sample(sample)
                if changed and controller_replay is not None:
                    controller_replay.step()
            except Exception:
                pass
            next_t += usage_api_poll_s

    def _offer_loop() -> None:
        next_t = time.monotonic() + refresh_s
        while True:
            sleep_s = next_t - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            try:
                if usage_mirror is None and controller_replay is not None:
                    controller_replay.step()
                engine.refresh_once()
            except Exception:
                pass
            next_t += refresh_s

    # Seed first snapshot immediately, then refresh periodically in background.
    engine.refresh_once()
    if usage_mirror is not None:
        threading.Thread(target=_usage_api_loop, daemon=True).start()
    threading.Thread(target=_offer_loop, daemon=True).start()

    print(f"[resource-offer] listening on http://{host}:{port}")
    print(f"[resource-offer] usage CSVs: {usage_csvs}")
    if usage_mirror is not None:
        print(f"[resource-offer] usage API source: {usage_api_url}")
        print(f"[resource-offer] usage API poll: {usage_api_poll_s}s")
        print(f"[resource-offer] mirrored usage CSV: {usage_api_csv}")
    print(f"[resource-offer] controller CSV: {controller_csv}")
    print(f"[resource-offer] controller replay mode: {replay_mode}")
    print(f"[resource-offer] controller-from-usage writer: {'enabled' if controller_writer is not None else 'disabled'}")
    if controller_replay is not None:
        print(f"[resource-offer] controller replay rows: {controller_replay.size}")
    print(f"[resource-offer] output CSV: {offer_csv_path}")
    print(f"[resource-offer] refresh interval: {refresh_s}s")

    app = _create_app(engine=engine)
    uvicorn.run(app, host=host, port=port, log_level="info")
