import math
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional


@dataclass
class FuzzyConfig:
    csv_path: str = os.getenv("KSENSE_METRICS_CSV", "/tmp/ksense/kernel_metrics.csv")
    short_win_s: int = int(os.getenv("FUZZY_SHORT_WIN_S", "10"))
    long_win_s: int = int(os.getenv("FUZZY_LONG_WIN_S", "60"))
    max_csv_lines: int = int(os.getenv("FUZZY_MAX_CSV_LINES", "2000"))

    bad_threshold: int = int(os.getenv("FUZZY_BAD_THRESHOLD", "3"))
    good_threshold: int = int(os.getenv("FUZZY_GOOD_THRESHOLD", "2"))

    allow_on_missing: bool = os.getenv("FUZZY_ALLOW_ON_MISSING", "false").lower() == "true"

    # Resource thresholds (fractions for CPU/Mem, PSI avg10)
    cpu_low: float = float(os.getenv("FUZZY_CPU_LOW", "0.50"))
    cpu_mid: float = float(os.getenv("FUZZY_CPU_MID", "0.70"))
    cpu_high: float = float(os.getenv("FUZZY_CPU_HIGH", "0.85"))

    mem_low: float = float(os.getenv("FUZZY_MEM_LOW", "0.60"))
    mem_mid: float = float(os.getenv("FUZZY_MEM_MID", "0.75"))
    mem_high: float = float(os.getenv("FUZZY_MEM_HIGH", "0.90"))

    psi_low: float = float(os.getenv("FUZZY_PSI_LOW", "0.03"))
    psi_mid: float = float(os.getenv("FUZZY_PSI_MID", "0.10"))
    psi_high: float = float(os.getenv("FUZZY_PSI_HIGH", "0.20"))

    # Minimum thresholds to avoid overly permissive dynamic scaling.
    fric_min_low: float = float(os.getenv("FUZZY_FRIC_MIN_LOW", "1.0"))
    fric_min_mid: float = float(os.getenv("FUZZY_FRIC_MIN_MID", "2.0"))
    fric_min_high: float = float(os.getenv("FUZZY_FRIC_MIN_HIGH", "3.5"))

    eng_min_low: float = float(os.getenv("FUZZY_ENG_MIN_LOW", "0.05"))
    eng_min_mid: float = float(os.getenv("FUZZY_ENG_MIN_MID", "0.15"))
    eng_min_high: float = float(os.getenv("FUZZY_ENG_MIN_HIGH", "0.30"))


def _percentile(values, p):
    if not values:
        return None
    vals = sorted(values)
    k = int(round((p / 100.0) * (len(vals) - 1)))
    return vals[max(0, min(k, len(vals) - 1))]


def _dynamic_thresholds(values, min_low, min_mid, min_high):
    if not values:
        return min_low, min_mid, min_high
    p50 = _percentile(values, 50) or 0.0
    p90 = _percentile(values, 90) or 0.0
    p99 = _percentile(values, 99) or 0.0
    low = max(p50, min_low)
    mid = max(p90, min_mid, low + 1e-6)
    high = max(p99, min_high, mid + 1e-6)
    return low, mid, high


def _low_mf(x, low, mid):
    if x <= low:
        return 1.0
    if x >= mid:
        return 0.0
    return (mid - x) / (mid - low)


def _med_mf(x, low, mid, high):
    if x <= low or x >= high:
        return 0.0
    if x == mid:
        return 1.0
    if x < mid:
        return (x - low) / (mid - low)
    return (high - x) / (high - mid)


def _high_mf(x, mid, high):
    if x <= mid:
        return 0.0
    if x >= high:
        return 1.0
    return (x - mid) / (high - mid)


def _read_last_lines(path, max_lines, max_bytes=1024 * 1024):
    try:
        size = os.stat(path).st_size
    except FileNotFoundError:
        return []
    with open(path, "r", encoding="utf-8") as f:
        if size > max_bytes:
            f.seek(max(0, size - max_bytes))
        data = f.read().splitlines()
    return data[-max_lines:]


def _parse_recent_metrics(cfg: FuzzyConfig):
    lines = _read_last_lines(cfg.csv_path, cfg.max_csv_lines)
    if not lines:
        return []
    headers = lines[0].split(",")
    try:
        time_idx = headers.index("Time")
        fric_idx = headers.index("Friction")
        eng_idx = headers.index("Energy")
    except ValueError:
        return []

    now = datetime.now()
    long_cutoff = now - timedelta(seconds=cfg.long_win_s)
    entries = []
    for line in lines[1:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) <= max(time_idx, fric_idx, eng_idx):
            continue
        ts = parts[time_idx]
        try:
            ts_dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
        if ts_dt < long_cutoff:
            continue
        fric = parts[fric_idx]
        eng = parts[eng_idx]
        try:
            fric_val = float(fric) if fric else None
        except ValueError:
            fric_val = None
        try:
            eng_val = float(eng) if eng else None
        except ValueError:
            eng_val = None
        entries.append((ts_dt, fric_val, eng_val))
    return entries


class ResourceSampler:
    def __init__(self):
        self._prev_total = None
        self._prev_idle = None

    def cpu_util(self):
        try:
            with open("/proc/stat", "r", encoding="utf-8") as f:
                line = f.readline()
        except FileNotFoundError:
            return None
        parts = line.split()
        if len(parts) < 5:
            return None
        values = [int(v) for v in parts[1:]]
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
        return max(0.0, min(1.0, 1.0 - (idle_delta / total_delta)))

    def mem_util(self):
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                data = f.read().splitlines()
        except FileNotFoundError:
            return None
        mem_total = None
        mem_avail = None
        for line in data:
            if line.startswith("MemTotal:"):
                mem_total = float(line.split()[1])
            elif line.startswith("MemAvailable:"):
                mem_avail = float(line.split()[1])
        if not mem_total or mem_avail is None:
            return None
        used = max(0.0, mem_total - mem_avail)
        return max(0.0, min(1.0, used / mem_total))

    def psi(self):
        psi_vals = []
        for path in ("/proc/pressure/cpu", "/proc/pressure/memory", "/proc/pressure/io"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    lines = f.read().splitlines()
            except FileNotFoundError:
                continue
            for line in lines:
                if line.startswith("some"):
                    for part in line.split():
                        if part.startswith("avg10="):
                            try:
                                psi_vals.append(float(part.split("=")[1]))
                            except ValueError:
                                pass
        if not psi_vals:
            return None
        psi = max(psi_vals)
        # Kernel PSI avg is expressed as a percentage. Normalize to 0-1 if needed.
        if psi > 1.0:
            psi = psi / 100.0
        return psi


class FuzzyController:
    def __init__(self, cfg: Optional[FuzzyConfig] = None):
        self.cfg = cfg or FuzzyConfig()
        self._lock = threading.Lock()
        self._bad = 0
        self._good = 0
        self._last_decision = "allow"
        self._sampler = ResourceSampler()

    def _fuzzy_risk(self, fric, eng, cpu, mem, psi, fric_thr, eng_thr):
        fric_low, fric_mid, fric_high = fric_thr
        eng_low, eng_mid, eng_high = eng_thr

        fric_l = _low_mf(fric, fric_low, fric_mid)
        fric_m = _med_mf(fric, fric_low, fric_mid, fric_high)
        fric_h = _high_mf(fric, fric_mid, fric_high)

        eng_l = _low_mf(eng, eng_low, eng_mid)
        eng_m = _med_mf(eng, eng_low, eng_mid, eng_high)
        eng_h = _high_mf(eng, eng_mid, eng_high)

        cpu_l = _low_mf(cpu, self.cfg.cpu_low, self.cfg.cpu_mid)
        cpu_m = _med_mf(cpu, self.cfg.cpu_low, self.cfg.cpu_mid, self.cfg.cpu_high)
        cpu_h = _high_mf(cpu, self.cfg.cpu_mid, self.cfg.cpu_high)

        mem_l = _low_mf(mem, self.cfg.mem_low, self.cfg.mem_mid)
        mem_m = _med_mf(mem, self.cfg.mem_low, self.cfg.mem_mid, self.cfg.mem_high)
        mem_h = _high_mf(mem, self.cfg.mem_mid, self.cfg.mem_high)

        psi_l = _low_mf(psi, self.cfg.psi_low, self.cfg.psi_mid)
        psi_m = _med_mf(psi, self.cfg.psi_low, self.cfg.psi_mid, self.cfg.psi_high)
        psi_h = _high_mf(psi, self.cfg.psi_mid, self.cfg.psi_high)

        risk_high = max(
            min(fric_h, eng_h),
            min(psi_h, max(cpu_h, mem_h)),
        )

        risk_med = max(
            min(fric_m, eng_m),
            min(psi_m, max(cpu_m, mem_m)),
            max(fric_m, eng_m),
        )

        risk_low = min(fric_l, eng_l, psi_l, cpu_l, mem_l)

        denom = risk_low + risk_med + risk_high
        if denom <= 0:
            return 0.0, "low"
        score = (0.2 * risk_low + 0.6 * risk_med + 1.0 * risk_high) / denom
        if score >= 0.70:
            level = "high"
        elif score >= 0.45:
            level = "medium"
        else:
            level = "low"
        return score, level

    def evaluate(self):
        cfg = self.cfg
        entries = _parse_recent_metrics(cfg)
        now = datetime.now()
        short_cutoff = now - timedelta(seconds=cfg.short_win_s)

        fric_vals_short = []
        eng_vals_short = []
        fric_vals_long = []
        eng_vals_long = []

        for ts, fric, eng in entries:
            if fric is not None:
                fric_vals_long.append(fric)
                if ts >= short_cutoff:
                    fric_vals_short.append(fric)
            if eng is not None:
                eng_vals_long.append(eng)
                if ts >= short_cutoff:
                    eng_vals_short.append(eng)

        fric_long_p99 = _percentile(fric_vals_long, 99) if fric_vals_long else None
        eng_long_p99 = _percentile(eng_vals_long, 99) if eng_vals_long else None
        fric_short_p99 = _percentile(fric_vals_short, 99) if fric_vals_short else None
        eng_short_p99 = _percentile(eng_vals_short, 99) if eng_vals_short else None

        fric_candidates = [v for v in [fric_short_p99, fric_long_p99] if v is not None]
        eng_candidates = [v for v in [eng_short_p99, eng_long_p99] if v is not None]
        fric = max(fric_candidates) if fric_candidates else None
        eng = max(eng_candidates) if eng_candidates else None

        cpu = self._sampler.cpu_util()
        mem = self._sampler.mem_util()
        psi = self._sampler.psi()

        missing = [name for name, val in [("friction", fric), ("energy", eng), ("cpu", cpu), ("mem", mem), ("psi", psi)] if val is None]
        critical_missing = [name for name in missing if name in ("friction", "energy")]
        if critical_missing and not cfg.allow_on_missing:
            return {
                "decision": "deny",
                "reason": f"missing metrics: {', '.join(missing)}",
                "score": 1.0,
                "level": "high",
                "metrics": {},
                "missing": missing,
            }

        fric_val = fric or 0.0
        eng_val = eng or 0.0
        cpu_val = cpu or 0.0
        mem_val = mem or 0.0
        psi_val = psi or 0.0

        fric_thr_raw = _dynamic_thresholds(fric_vals_long, cfg.fric_min_low, cfg.fric_min_mid, cfg.fric_min_high)
        eng_thr_raw = _dynamic_thresholds(eng_vals_long, cfg.eng_min_low, cfg.eng_min_mid, cfg.eng_min_high)

        fric_val_log = math.log1p(fric_val)
        eng_val_log = math.log1p(eng_val)
        fric_thr = tuple(math.log1p(v) for v in fric_thr_raw)
        eng_thr = tuple(math.log1p(v) for v in eng_thr_raw)

        score, level = self._fuzzy_risk(fric_val_log, eng_val_log, cpu_val, mem_val, psi_val, fric_thr, eng_thr)

        return {
            "decision": "allow",
            "score": score,
            "level": level,
            "metrics": {
                "friction_p99_short": fric_short_p99,
                "friction_p99_long": fric_long_p99,
                "energy_p99_short": eng_short_p99,
                "energy_p99_long": eng_long_p99,
                "cpu_util": cpu,
                "mem_util": mem,
                "psi_avg10": psi,
                "friction_thresholds": fric_thr_raw,
                "energy_thresholds": eng_thr_raw,
                "friction_log": fric_val_log,
                "energy_log": eng_val_log,
            },
            "missing": missing,
        }

    def decide(self):
        with self._lock:
            report = self.evaluate()
            if report["decision"] == "deny":
                self._bad += 1
                self._good = 0
                decision = "deny" if self._bad >= self.cfg.bad_threshold else self._last_decision
            else:
                if report["level"] == "high":
                    self._bad += 1
                    self._good = 0
                else:
                    self._good += 1
                    self._bad = 0
                if report["level"] == "high" and self._bad >= self.cfg.bad_threshold:
                    decision = "deny"
                elif report["level"] != "high" and self._good >= self.cfg.good_threshold:
                    decision = "allow"
                else:
                    decision = self._last_decision

            self._last_decision = decision
            report["decision"] = decision
            report["bad_count"] = self._bad
            report["good_count"] = self._good
            return report
